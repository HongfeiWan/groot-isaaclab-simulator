#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-effector IK teleop for xMate3 + right L10 hand in Isaac Sim 5.1.

Run with IsaacLab's Python:

    ~/Project/IsaacLab/isaaclab.sh -p examples/IsaacLab/teleop_xmate3_l10hand_eef_ik.py

This script reuses:
- ``examples/IsaacLab/load_xmate3_with_right_l10hand_urdf.py`` for the combined
  xMate3 + right L10 hand URDF and L10 mimic-joint handling.
- ``external_dependencies/rci_client/python/xmate_ik_ctypes.py`` for xMate3 IK
  when RCI offline kinematics parameters are available.

The RCI shared library is expected at:

    external_dependencies/rci_client/build/bindings/libxmate_ik_c.so

If it is missing, build it as described in ``external_dependencies/rci_client/python/README.md``.
When RCI cannot initialize because only offline IK is enabled, this script falls
back to a small URDF-based numerical IK solver so keyboard teleop still works.

    cmake -S external_dependencies/rci_client -B external_dependencies/rci_client/build
    cmake --build external_dependencies/rci_client/build --target xmate_ik_c -j2

Keyboard controls in the Isaac Sim viewport:
- W/S: move target +X/-X in the xMate3 base frame
- A/D: move target +Y/-Y in the xMate3 base frame
- E/Q: move target +Z/-Z in the xMate3 base frame
- Z/X: open/close the L10 hand

You can also send a fixed target:

    ~/Project/IsaacLab/isaaclab.sh -p examples/IsaacLab/teleop_xmate3_l10hand_eef_ik.py \
        --target-xyz 0.35,0.0,0.75
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from isaacsim.core.prims import SingleArticulation  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
LOADER_SCRIPT_PATH = REPO_ROOT / "examples" / "IsaacLab" / "load_xmate3_with_right_l10hand_urdf.py"
RCI_ROOT = REPO_ROOT / "external_dependencies" / "rci_client"
RCI_LIB_PATH = RCI_ROOT / "build" / "bindings" / "libxmate_ik_c.so"

ARM_JOINT_NAMES = [f"xMate3_joint_{i}" for i in range(1, 7)]
HAND_JOINT_NAMES = [
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
    "index_mcp_roll",
    "ring_mcp_roll",
    "pinky_mcp_roll",
    "thumb_cmc_roll",
]
HAND_DERIVED_JOINT_NAMES = [
    "thumb_mcp",
    "thumb_ip",
    "index_pip",
    "index_dip",
    "middle_pip",
    "middle_dip",
    "ring_pip",
    "ring_dip",
    "pinky_pip",
    "pinky_dip",
]
HAND_MIMIC_JOINT_NAMES = {
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_roll",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_roll",
    "ring_mcp_pitch",
    "pinky_mcp_roll",
    "pinky_mcp_pitch",
    "index_pip",
    "index_dip",
    "middle_pip",
    "middle_dip",
    "ring_pip",
    "ring_dip",
    "pinky_pip",
    "pinky_dip",
    "thumb_mcp",
    "thumb_ip",
}

DEFAULT_ARM_Q = np.array([0.0, -0.45, 0.8, 0.0, 0.55, 0.0], dtype=np.float64)
HAND_OPEN_Q = np.zeros(10, dtype=np.float64)
HAND_CLOSED_Q = np.array([0.75, 0.25, 0.95, 0.95, 0.9, 0.85, 0.15, -0.1, -0.15, 0.25])

_WORLD_REF = None
_LOADER_REF = None
_HAND_PRIMS_REF = None
_KEYBOARD_SUB_REF = None


def _parse_args() -> tuple[argparse.Namespace, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-prim-path", type=str, default="/World/xMate3_L10Hand")
    parser.add_argument(
        "--initial-arm-q",
        type=str,
        default=",".join(str(v) for v in DEFAULT_ARM_Q),
    )
    parser.add_argument(
        "--target-xyz",
        type=str,
        default=None,
        help="Optional fixed x,y,z target in the xMate3 base frame. Defaults to current FK pose.",
    )
    parser.add_argument(
        "--target-rpy",
        type=str,
        default=None,
        help="Optional fixed roll,pitch,yaw in radians. Defaults to current IK FK rotation.",
    )
    parser.add_argument(
        "--eef-speed",
        type=float,
        default=0.08,
        help="Keyboard target speed in m/s.",
    )
    parser.add_argument("--command-hz", type=float, default=30.0, help="IK/action command rate.")
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=0.045,
        help="Max rad per command tick.",
    )
    parser.add_argument("--psi", type=float, default=0.0, help="RCI IK redundant parameter.")
    parser.add_argument("--show-target-marker", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--ik-backend",
        choices=("auto", "rci", "urdf"),
        default="auto",
        help="IK backend. auto tries RCI first, then falls back to URDF numerical IK.",
    )
    parser.add_argument("--rci-lib-path", type=str, default=str(RCI_LIB_PATH))
    parser.add_argument("--rci-ip", type=str, default="127.0.0.1")
    parser.add_argument("--rci-port", type=int, default=1337)
    parser.add_argument(
        "--kinematics-json",
        type=str,
        default=None,
        help="Optional JSON with dh_params[28] and robot_dims[21] for offline RCI IK.",
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    return args, app


def _parse_vector(text: str, length: int, *, name: str) -> np.ndarray:
    values = np.asarray([float(v.strip()) for v in text.split(",") if v.strip()], dtype=np.float64)
    if values.shape != (length,):
        raise ValueError(
            f"{name} must contain {length} comma-separated floats, got {values.shape[0]}"
        )
    return values


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64).reshape(3)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _import_module_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Module file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path.as_posix())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_rci_ik():
    """Import the Python wrapper exactly as documented by rci_client/python/README.md."""
    if str(RCI_ROOT) not in sys.path:
        sys.path.insert(0, str(RCI_ROOT))
    from python.xmate_ik_ctypes import XMATE3, XmateIk, pose_from_xyz_rotation  # type: ignore

    return XmateIk, XMATE3, pose_from_xyz_rotation


def _read_kinematics_json(path: str | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if path is None:
        return None, None
    p = Path(path).expanduser()
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    dh = np.asarray(data["dh_params"], dtype=np.float64).reshape(28)
    dims = np.asarray(data["robot_dims"], dtype=np.float64).reshape(21)
    return dh, dims


def _create_ik(args: argparse.Namespace):
    if str(args.ik_backend) == "urdf":
        print("[ik] using URDF numerical IK backend", flush=True)
        return UrdfXmate3Ik(), _pose_from_xyz_rotation

    try:
        XmateIk, model_type, pose_from_xyz_rotation = _import_rci_ik()
        lib_path = Path(args.rci_lib_path).expanduser()
        if not lib_path.exists():
            raise FileNotFoundError(
                f"RCI IK shared library not found: {lib_path}. Build target xmate_ik_c first."
            )

        dh, dims = _read_kinematics_json(args.kinematics_json)
        if dh is not None and dims is not None:
            ik = XmateIk(lib_path=lib_path, model_type=model_type, dh_params=dh, robot_dims=dims)
            print(f"[rci_ik] offline kinematics mode, lib={lib_path}", flush=True)
        else:
            ik = XmateIk(
                lib_path=lib_path,
                ip=str(args.rci_ip),
                port=int(args.rci_port),
                model_type=model_type,
            )
            print(
                f"[rci_ik] connected mode, lib={lib_path}, "
                f"endpoint={args.rci_ip}:{args.rci_port}",
                flush=True,
            )
        return ik, pose_from_xyz_rotation
    except Exception as e:
        if str(args.ik_backend) != "auto":
            raise
        print(f"[warn] RCI IK unavailable, falling back to URDF numerical IK: {e}", flush=True)
        return UrdfXmate3Ik(), _pose_from_xyz_rotation


def _pose_from_xyz_rotation(xyz: np.ndarray, rotation: np.ndarray | None = None) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(xyz, dtype=np.float64).reshape(3)
    if rotation is not None:
        pose[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return pose


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    v = 1.0 - c
    return np.array(
        [
            [x * x * v + c, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, y * y * v + c, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, z * z * v + c],
        ],
        dtype=np.float64,
    )


def _homogeneous(
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if translation is not None:
        transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


class UrdfXmate3Ik:
    """Numerical IK for the xMate3 URDF chain used by this example.

    The chain is read from ``demo_data/l10_hand/xMate3_with_right_L10hand.urdf``:
    axes are z, y, y, z, y, z and joint origins are the link offsets in meters.
    This fallback intentionally controls ``xMate3_link6``; the L10 hand remains
    attached by the fixed joint from the loader script.
    """

    _origins = (
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.404], dtype=np.float64),
        np.array([0.0, 0.0, 0.4375], dtype=np.float64),
        np.array([0.0, 0.0, 0.4125], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.2755], dtype=np.float64),
    )
    _axes = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    _lower = np.array([-2.9671, -2.0944, -2.0944, -2.9671, -2.0944, -6.2832])
    _upper = np.array([2.9671, 2.0944, 2.0944, 2.9671, 2.0944, 6.2832])

    def close(self) -> None:
        return

    def fk6(self, q: np.ndarray) -> np.ndarray:
        return self._fk_with_jacobian(q)[0]

    def solve6(self, cart_pose: np.ndarray, q_init: np.ndarray, psi: float = 0.0) -> np.ndarray:
        del psi
        target = np.asarray(cart_pose, dtype=np.float64).reshape(4, 4)
        q = np.clip(np.asarray(q_init, dtype=np.float64).reshape(6), self._lower, self._upper)
        target_pos = target[:3, 3]
        target_rot = target[:3, :3]

        for _ in range(80):
            current, jacobian = self._fk_with_jacobian(q)
            pos_error = target_pos - current[:3, 3]
            rot_error = 0.5 * (
                np.cross(current[:3, 0], target_rot[:3, 0])
                + np.cross(current[:3, 1], target_rot[:3, 1])
                + np.cross(current[:3, 2], target_rot[:3, 2])
            )
            error = np.concatenate([pos_error, 0.35 * rot_error])
            if np.linalg.norm(pos_error) < 1e-4 and np.linalg.norm(rot_error) < 1e-3:
                break

            weighted_jacobian = jacobian.copy()
            weighted_jacobian[3:6, :] *= 0.35
            damping = 2.5e-3
            lhs = weighted_jacobian @ weighted_jacobian.T + damping * np.eye(6)
            dq = weighted_jacobian.T @ np.linalg.solve(lhs, error)
            q = np.clip(q + np.clip(dq, -0.08, 0.08), self._lower, self._upper)

        return q

    def _fk_with_jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transform = np.eye(4, dtype=np.float64)
        joint_positions: list[np.ndarray] = []
        joint_axes: list[np.ndarray] = []

        for origin, axis, angle in zip(self._origins, self._axes, np.asarray(q).reshape(6)):
            transform = transform @ _homogeneous(translation=origin)
            joint_positions.append(transform[:3, 3].copy())
            joint_axes.append(transform[:3, :3] @ axis)
            transform = transform @ _homogeneous(rotation=_rot_axis(axis, float(angle)))

        end_pos = transform[:3, 3]
        jacobian = np.zeros((6, 6), dtype=np.float64)
        for i, (joint_pos, joint_axis) in enumerate(zip(joint_positions, joint_axes)):
            jacobian[:3, i] = np.cross(joint_axis, end_pos - joint_pos)
            jacobian[3:6, i] = joint_axis
        return transform, jacobian


def _init_robot_scene(*, prim_path: str) -> "SingleArticulation":
    global _WORLD_REF, _LOADER_REF, _HAND_PRIMS_REF

    loader = _import_module_from_path("load_xmate3_with_right_l10hand_urdf", LOADER_SCRIPT_PATH)
    _LOADER_REF = loader

    if _WORLD_REF is None:
        try:
            from isaacsim.core.api.world import World  # type: ignore
        except Exception:
            from isaacsim.core.world import World  # type: ignore

        _WORLD_REF = World(stage_units_in_meters=1.0)
        _WORLD_REF.reset()

    loader._add_ground_plane()  # noqa: SLF001
    loader._add_scene_light()  # noqa: SLF001
    loader._import_urdf(  # noqa: SLF001
        urdf_path=str(loader.COMBINED_URDF_PATH),
        prim_path=prim_path,
        fix_base=True,
        make_instanceable=False,
    )
    loader._apply_hand_drive_gains(root_prim_path=prim_path)  # noqa: SLF001

    try:
        _WORLD_REF.reset()
    except Exception:
        pass

    _HAND_PRIMS_REF = loader._collect_named_prims_under(  # noqa: SLF001
        root_prim_path=prim_path,
        names=HAND_MIMIC_JOINT_NAMES,
    )

    from isaacsim.core.prims import SingleArticulation  # type: ignore

    robot = SingleArticulation(prim_path=prim_path, name="xMate3_L10Hand_EEF_IK")
    try:
        robot.initialize(physics_sim_view=getattr(_WORLD_REF, "physics_sim_view", None))
    except TypeError:
        robot.initialize()
    return robot


def _name_to_dof_index(robot: "SingleArticulation") -> dict[str, int]:
    if hasattr(robot, "dof_names"):
        names = list(getattr(robot, "dof_names"))
        if hasattr(robot, "get_dof_index") and callable(getattr(robot, "get_dof_index")):
            return {
                name: int(robot.get_dof_index(name)) for name in names  # type: ignore[attr-defined]
            }
        return {name: i for i, name in enumerate(names)}
    if hasattr(robot, "get_dof_names") and callable(getattr(robot, "get_dof_names")):
        names = list(robot.get_dof_names())  # type: ignore[attr-defined]
        return {name: i for i, name in enumerate(names)}
    raise RuntimeError("Cannot read DOF names from SingleArticulation.")


def _current_arm_q(
    robot: "SingleArticulation",
    arm_indices: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    try:
        q = np.asarray(robot.get_joint_positions(), dtype=np.float64).reshape(-1)
        if q.size > int(np.max(arm_indices)):
            return q[arm_indices]
    except Exception:
        pass
    return fallback.copy()


def _apply_joint_positions(
    robot: "SingleArticulation",
    joint_indices: np.ndarray,
    q: np.ndarray,
) -> None:
    if joint_indices.size == 0:
        return
    from isaacsim.core.utils.types import ArticulationAction  # type: ignore

    robot.apply_action(
        ArticulationAction(
            joint_positions=np.asarray(q, dtype=np.float32)[: joint_indices.size],
            joint_indices=joint_indices,
        )
    )


def _derive_hand_mimic_positions(master_q: np.ndarray) -> dict[str, float]:
    master = {name: float(value) for name, value in zip(HAND_JOINT_NAMES, master_q)}
    return {
        "thumb_mcp": 1.3898 * master["thumb_cmc_pitch"],
        "thumb_ip": 1.5080 * master["thumb_cmc_pitch"],
        "index_pip": 1.3462 * master["index_mcp_pitch"],
        "index_dip": 0.4616 * master["index_mcp_pitch"],
        "middle_pip": 1.3462 * master["middle_mcp_pitch"],
        "middle_dip": 0.4616 * master["middle_mcp_pitch"],
        "ring_pip": 1.3462 * master["ring_mcp_pitch"],
        "ring_dip": 0.4616 * master["ring_mcp_pitch"],
        "pinky_pip": 1.3462 * master["pinky_mcp_pitch"],
        "pinky_dip": 0.4616 * master["pinky_mcp_pitch"],
    }


def _set_hand_drive_targets_for_loader(master_q: np.ndarray) -> None:
    """Mirror master targets into USD drive attrs used by the loader mimic helper."""
    if not _HAND_PRIMS_REF:
        return
    try:
        from pxr import UsdPhysics  # type: ignore
    except Exception:
        return

    # _update_hand_mimic_targets() reads angular drive targetPosition attributes, then
    # writes thumb_mcp/thumb_ip and pip/dip targets from those masters. PhysX angular
    # drive targetPosition is stored in degrees, while ArticulationAction uses radians.
    for name, value in zip(HAND_JOINT_NAMES, master_q):
        prim = _HAND_PRIMS_REF.get(name)
        if prim is None:
            continue
        try:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTargetPositionAttr(0.0)
            drive.GetTargetPositionAttr().Set(float(math.degrees(float(value))))
        except Exception:
            continue


def _apply_l10_hand_positions(
    robot: "SingleArticulation",
    master_indices: np.ndarray,
    derived_indices: dict[str, int],
    master_q: np.ndarray,
) -> None:
    del derived_indices
    _apply_joint_positions(robot, master_indices, master_q)
    _set_hand_drive_targets_for_loader(master_q)


def _create_target_marker(path: str) -> str | None:
    try:
        import omni.usd  # type: ignore
        from pxr import Gf, UsdGeom  # type: ignore
    except Exception:
        return None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(0.025)
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    if xform.GetOrderedXformOps():
        xform.ClearXformOpOrder()
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    return path


def _set_target_marker(path: str | None, xyz: np.ndarray) -> None:
    if path is None:
        return
    try:
        import omni.usd  # type: ignore
        from pxr import Gf, UsdGeom  # type: ignore
    except Exception:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    if ops:
        ops[0].Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))


class KeyboardTeleop:
    """Small Isaac Sim keyboard helper that degrades gracefully in headless mode."""

    def __init__(self) -> None:
        self._pressed: set[str] = set()
        self._enabled = False

    @staticmethod
    def _normalize_key(value: object) -> str:
        key_name = str(getattr(value, "name", value)).split(".")[-1].upper()
        if key_name.startswith("KEY_"):
            key_name = key_name[4:]
        return key_name

    def start(self) -> None:
        global _KEYBOARD_SUB_REF
        try:
            import carb.input  # type: ignore
            import omni.appwindow  # type: ignore
        except Exception as e:
            print(f"[teleop] keyboard input unavailable: {e}", flush=True)
            return

        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            print("[teleop] no default app window; keyboard disabled.", flush=True)
            return
        keyboard = app_window.get_keyboard()
        input_iface = carb.input.acquire_input_interface()

        press_types = {
            getattr(carb.input.KeyboardEventType, "KEY_PRESS", None),
            getattr(carb.input.KeyboardEventType, "KEY_REPEAT", None),
        }
        release_type = getattr(carb.input.KeyboardEventType, "KEY_RELEASE", None)

        def _on_keyboard_event(event, *_) -> bool:
            key_name = self._normalize_key(event.input)
            if event.type in press_types:
                self._pressed.add(key_name)
            elif event.type == release_type:
                self._pressed.discard(key_name)
            return True

        _KEYBOARD_SUB_REF = input_iface.subscribe_to_keyboard_events(keyboard, _on_keyboard_event)
        self._enabled = True

    def target_delta(self, *, speed: float, dt: float) -> np.ndarray:
        if not self._enabled:
            return np.zeros(3, dtype=np.float64)
        delta = np.zeros(3, dtype=np.float64)
        step = float(speed) * float(dt)
        if "W" in self._pressed or "UP" in self._pressed:
            delta[0] += step
        if "S" in self._pressed or "DOWN" in self._pressed:
            delta[0] -= step
        if "A" in self._pressed or "LEFT" in self._pressed:
            delta[1] += step
        if "D" in self._pressed or "RIGHT" in self._pressed:
            delta[1] -= step
        if "E" in self._pressed or "PAGE_UP" in self._pressed:
            delta[2] += step
        if "Q" in self._pressed or "PAGE_DOWN" in self._pressed:
            delta[2] -= step
        return delta

    def hand_delta(self, *, speed: float, dt: float) -> float:
        if not self._enabled:
            return 0.0
        delta = 0.0
        if "X" in self._pressed:
            delta += float(speed) * float(dt)
        if "Z" in self._pressed:
            delta -= float(speed) * float(dt)
        return delta


def _print_controls() -> None:
    print(
        "[teleop] controls: W/S X, A/D Y, E/Q Z, Z/X hand open/close. "
        "Click the Isaac Sim viewport first so it receives keyboard focus.",
        flush=True,
    )


def main() -> None:
    args, simulation_app = _parse_args()
    try:
        ik, pose_from_xyz_rotation = _create_ik(args)
        robot = _init_robot_scene(prim_path=str(args.robot_prim_path))
        dof_index = _name_to_dof_index(robot)

        missing = [n for n in ARM_JOINT_NAMES + HAND_JOINT_NAMES if n not in dof_index]
        if missing:
            raise RuntimeError(f"Missing expected DOFs in imported robot: {missing}")

        arm_indices = np.asarray([dof_index[n] for n in ARM_JOINT_NAMES], dtype=np.int32)
        hand_indices = np.asarray([dof_index[n] for n in HAND_JOINT_NAMES], dtype=np.int32)
        derived_hand_indices = {
            name: int(dof_index[name]) for name in HAND_DERIVED_JOINT_NAMES if name in dof_index
        }
        missing_derived = [name for name in HAND_DERIVED_JOINT_NAMES if name not in dof_index]
        if missing_derived:
            print(
                f"[warn] Missing L10 mimic DOFs; USD drive targets only: {missing_derived}",
                flush=True,
            )

        q_cmd = _parse_vector(str(args.initial_arm_q), 6, name="--initial-arm-q")
        _apply_joint_positions(robot, arm_indices, q_cmd)
        for _ in range(3):
            simulation_app.update()

        initial_pose = ik.fk6(q_cmd)
        target_xyz = initial_pose[:3, 3].copy()
        target_rotation = initial_pose[:3, :3].copy()
        if args.target_xyz:
            target_xyz = _parse_vector(str(args.target_xyz), 3, name="--target-xyz")
        if args.target_rpy:
            target_rotation = _rotation_from_rpy(
                _parse_vector(str(args.target_rpy), 3, name="--target-rpy")
            )

        marker_path = None
        if bool(args.show_target_marker):
            marker_path = _create_target_marker("/World/eef_ik_target")
            _set_target_marker(marker_path, target_xyz)

        teleop = KeyboardTeleop()
        if not bool(getattr(args, "headless", False)):
            teleop.start()
            _print_controls()

        hand_alpha = 0.0
        command_dt = 1.0 / max(float(args.command_hz), 1e-6)
        last_time = time.perf_counter()
        last_command_time = 0.0
        fail_count = 0

        while simulation_app.is_running():
            now = time.perf_counter()
            dt = min(max(now - last_time, 1.0 / 240.0), 0.05)
            last_time = now

            target_xyz += teleop.target_delta(speed=float(args.eef_speed), dt=dt)
            target_xyz[2] = max(0.02, float(target_xyz[2]))
            hand_alpha = float(np.clip(hand_alpha + teleop.hand_delta(speed=1.2, dt=dt), 0.0, 1.0))

            if now - last_command_time >= command_dt:
                last_command_time = now
                q_seed = _current_arm_q(robot, arm_indices, q_cmd)
                target_pose = pose_from_xyz_rotation(target_xyz, target_rotation)
                try:
                    q_target = ik.solve6(target_pose, q_seed, psi=float(args.psi))
                    q_cmd = q_seed + np.clip(
                        q_target - q_seed,
                        -float(args.max_joint_step),
                        float(args.max_joint_step),
                    )
                    _apply_joint_positions(robot, arm_indices, q_cmd)
                    fail_count = 0
                except Exception as e:
                    fail_count += 1
                    if fail_count == 1 or fail_count % 30 == 0:
                        print(f"[warn] IK failed for target {target_xyz.tolist()}: {e}", flush=True)

                hand_q = (1.0 - hand_alpha) * HAND_OPEN_Q + hand_alpha * HAND_CLOSED_Q
                _apply_l10_hand_positions(
                    robot,
                    hand_indices,
                    derived_hand_indices,
                    hand_q,
                )
                _set_target_marker(marker_path, target_xyz)

            if _LOADER_REF is not None and _HAND_PRIMS_REF:
                try:
                    _LOADER_REF._update_hand_mimic_targets(  # type: ignore[attr-defined]
                        root_prim_path=str(args.robot_prim_path),
                        prims=_HAND_PRIMS_REF,
                    )
                except Exception:
                    pass

            simulation_app.update()
    except Exception as e:
        print(f"[error] xMate3 EEF IK teleop failed: {e}", flush=True)
    finally:
        try:
            ik.close()  # type: ignore[name-defined]
        except Exception:
            pass
        simulation_app.close()


if __name__ == "__main__":
    main()
