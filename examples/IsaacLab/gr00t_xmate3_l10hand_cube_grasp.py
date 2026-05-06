#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed-loop GR00T cube grasp demo for xMate3 + right L10 hand in Isaac Sim 5.1.

Run with Isaac Lab's Python, after starting a GR00T policy server on port 5555:

    ~/Project/IsaacLab/isaaclab.sh -p examples/IsaacLab/gr00t_xmate3_l10hand_cube_grasp.py \
        --enable_cameras \
        --policy-port 5555 \
        --instruction "pick up the cube"

The script reuses the robot/camera USD layout from:

    examples/IsaacLab/load_xmate3_with_right_l10hand_urdf.py

and the xMate3 EEF IK/L10 hand helpers from:

    examples/IsaacLab/teleop_xmate3_l10hand_eef_ik.py

Expected custom ROKAE/L10 action schema:

    action.arm_eef_pos_target: (B, T, 3), absolute XYZ target in xMate3 base/world frame
    action.hand_joint_target:  (B, T, 10), absolute L10 canonical joint targets

For DROID/Franka-style checkpoints, the script also accepts eef_9d + gripper_position
or joint_position + gripper_position and maps the scalar gripper command to the L10 hand.
"""

from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import math
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image


if TYPE_CHECKING:  # pragma: no cover
    from isaacsim.core.prims import SingleArticulation  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
LOADER_SCRIPT_PATH = REPO_ROOT / "examples" / "IsaacLab" / "load_xmate3_with_right_l10hand_urdf.py"
TELEOP_SCRIPT_PATH = REPO_ROOT / "examples" / "IsaacLab" / "teleop_xmate3_l10hand_eef_ik.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gr00t.policy.server_client import PolicyClient  # noqa: E402


DEFAULT_IMAGE_SIZE = (180, 320)
DEFAULT_CUBE_POS = (-0.5, -0.5, 0.6)
DEFAULT_CUBE_SIZE = 0.055
DEFAULT_TABLE_TOP_Z = DEFAULT_CUBE_POS[2] - DEFAULT_CUBE_SIZE * 0.5
DEFAULT_TABLE_XY_SIZE = 0.45
DEFAULT_INITIAL_EEF_XYZ = (-0.7, -0.3, 0.6)
BOTTLE_BODY_HEIGHT_RATIO = 0.72
BOTTLE_NECK_HEIGHT_RATIO = 0.22
BOTTLE_CAP_HEIGHT_RATIO = 0.06
BOTTLE_BODY_RADIUS_RATIO = 0.16
BOTTLE_NECK_RADIUS_RATIO = 0.07

WORLD_REF = None
LOADER_REF = None
TELEOP_REF = None
HAND_PRIMS_REF: dict[str, object] | None = None


def _import_module_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Module file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path.as_posix())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_vector(text: str, length: int, *, name: str) -> np.ndarray:
    values = np.asarray([float(v.strip()) for v in text.split(",") if v.strip()], dtype=np.float64)
    if values.shape != (length,):
        raise ValueError(
            f"{name} must contain {length} comma-separated floats, got {values.shape[0]}"
        )
    return values


def _infer_table_top_z(cube_position: np.ndarray, cube_size: float) -> float:
    return float(np.asarray(cube_position, dtype=np.float64).reshape(3)[2] - float(cube_size) * 0.5)


def _resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[-3:-1] == (height, width):
        return image
    original_shape = image.shape
    image = image.reshape(-1, *original_shape[-3:])
    resized = [_resize_one_with_pad(Image.fromarray(frame), height, width) for frame in image]
    return np.stack(resized).reshape(*original_shape[:-3], height, width, original_shape[-1])


def _resize_one_with_pad(image: Image.Image, height: int, width: int) -> np.ndarray:
    cur_width, cur_height = image.size
    ratio = max(cur_width / width, cur_height / height)
    resized_width = max(1, int(cur_width / ratio))
    resized_height = max(1, int(cur_height / ratio))
    resized_image = image.resize((resized_width, resized_height), resample=Image.BILINEAR)

    output = Image.new(resized_image.mode, (width, height), 0)
    pad_width = max(0, int((width - resized_width) / 2))
    pad_height = max(0, int((height - resized_height) / 2))
    output.paste(resized_image, (pad_width, pad_height))
    return np.asarray(output)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _as_hwc_uint8(value: Any, *, image_size: tuple[int, int]) -> np.ndarray:
    image = _to_numpy(value)

    if isinstance(image, np.ndarray) and image.dtype.fields is not None:
        image = image.view(np.uint8).reshape(image.shape + (-1,))
    elif isinstance(image, np.ndarray) and image.dtype == np.uint32:
        # Some Isaac/Replicator RGB buffers arrive as packed 32-bit RGBA values.
        image = image.view(np.uint8).reshape(image.shape + (4,))

    while image.ndim > 3:
        image = image[0]
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dims, got shape {image.shape}")

    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected image channel count 3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 0.0
        if max_value <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255)

    image = image.astype(np.uint8, copy=False)
    return _resize_with_pad(image, image_size[0], image_size[1])


def _describe_array(name: str, value: Any) -> str:
    arr = _to_numpy(value) if not isinstance(value, str) else np.asarray(value)
    dtype = getattr(arr, "dtype", type(value))
    shape = getattr(arr, "shape", ())
    if np.issubdtype(arr.dtype, np.number) and arr.size:
        return f"{name}: shape={shape}, dtype={dtype}, min={arr.min():.4g}, max={arr.max():.4g}"
    return f"{name}: shape={shape}, dtype={dtype}"


def _rotmat_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return rotation[:2, :].reshape(6).astype(np.float32)


def _spawn_cube(*, prim_path: str, position: np.ndarray, size: float, mass: float) -> None:
    import omni.usd  # type: ignore
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics  # type: ignore

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available.")

    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.45, 0.95)])
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    if xform.GetOrderedXformOps():
        xform.ClearXformOpOrder()
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    )
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(size), float(size), float(size))
    )

    prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.CollisionAPI.Apply(prim)
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass))
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(False)


def _set_xform_pose(
    xform: Any,
    *,
    translate: tuple[float, float, float] | np.ndarray,
    scale: tuple[float, float, float] | np.ndarray | None = None,
) -> None:
    from pxr import Gf, UsdGeom  # type: ignore

    if xform.GetOrderedXformOps():
        xform.ClearXformOpOrder()
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2]))
    )
    if scale is not None:
        xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2]))
        )


def _spawn_bottle(*, prim_path: str, position: np.ndarray, height: float, mass: float) -> None:
    import omni.usd  # type: ignore
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics  # type: ignore

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available.")

    position = np.asarray(position, dtype=np.float64).reshape(3)
    height = float(height)
    body_height = height * BOTTLE_BODY_HEIGHT_RATIO
    neck_height = height * BOTTLE_NECK_HEIGHT_RATIO
    cap_height = height * BOTTLE_CAP_HEIGHT_RATIO
    body_radius = height * BOTTLE_BODY_RADIUS_RATIO
    neck_radius = height * BOTTLE_NECK_RADIUS_RATIO

    root = UsdGeom.Xform.Define(stage, prim_path)
    _set_xform_pose(
        UsdGeom.Xformable(root.GetPrim()),
        translate=(float(position[0]), float(position[1]), float(position[2])),
    )

    body_path = f"{prim_path}/body"
    body = UsdGeom.Cylinder.Define(stage, body_path)
    body.CreateAxisAttr("Z")
    body.CreateRadiusAttr(body_radius)
    body.CreateHeightAttr(body_height)
    body.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.38, 0.9)])
    _set_xform_pose(
        UsdGeom.Xformable(body.GetPrim()),
        translate=(0.0, 0.0, -height * 0.5 + body_height * 0.5),
    )

    neck_path = f"{prim_path}/neck"
    neck = UsdGeom.Cylinder.Define(stage, neck_path)
    neck.CreateAxisAttr("Z")
    neck.CreateRadiusAttr(neck_radius)
    neck.CreateHeightAttr(neck_height)
    neck.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.38, 0.9)])
    _set_xform_pose(
        UsdGeom.Xformable(neck.GetPrim()),
        translate=(0.0, 0.0, -height * 0.5 + body_height + neck_height * 0.5),
    )

    cap_path = f"{prim_path}/cap"
    cap = UsdGeom.Cylinder.Define(stage, cap_path)
    cap.CreateAxisAttr("Z")
    cap.CreateRadiusAttr(neck_radius * 1.15)
    cap.CreateHeightAttr(cap_height)
    cap.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.95, 0.9)])
    _set_xform_pose(
        UsdGeom.Xformable(cap.GetPrim()),
        translate=(
            0.0,
            0.0,
            -height * 0.5 + body_height + neck_height + cap_height * 0.5,
        ),
    )

    root_prim = root.GetPrim()
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr(float(mass))
    PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim).CreateDisableGravityAttr(False)
    for child_path in (body_path, neck_path, cap_path):
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(child_path))


def _spawn_target_object(
    *, kind: str, prim_path: str, position: np.ndarray, size: float, mass: float
) -> None:
    if kind == "cube":
        _spawn_cube(prim_path=prim_path, position=position, size=size, mass=mass)
    elif kind == "bottle":
        _spawn_bottle(prim_path=prim_path, position=position, height=size, mass=mass)
    else:
        raise ValueError(f"Unsupported target object kind: {kind!r}")


def _spawn_table_if_requested(
    *, enabled: bool, center_xy: np.ndarray, top_z: float, table_xy_size: float
) -> None:
    if not enabled:
        return

    import omni.usd  # type: ignore
    from pxr import Gf, UsdGeom, UsdPhysics  # type: ignore

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available.")

    table_path = "/World/grasp_table"
    table_height = 0.04
    cube = UsdGeom.Cube.Define(stage, table_path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.45, 0.42)])
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(table_path))
    if xform.GetOrderedXformOps():
        xform.ClearXformOpOrder()
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), float(top_z) - table_height * 0.5)
    )
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(table_xy_size), float(table_xy_size), table_height)
    )
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(table_path))


def _get_prim_world_position(prim_path: str) -> np.ndarray | None:
    try:
        import omni.usd  # type: ignore
        from pxr import UsdGeom  # type: ignore
    except Exception:
        return None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
    t = matrix.ExtractTranslation()
    return np.asarray([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)


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
    sphere.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.2, 0.05)])
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


class CameraReader:
    def __init__(self, camera_path: str, *, image_size: tuple[int, int], name: str):
        self.camera_path = camera_path
        self.image_size = image_size
        self.name = name
        self.rep = None
        self.annotator = None
        self.render_product = None
        self.warned = False

        try:
            import omni.replicator.core as rep  # type: ignore

            self.rep = rep
            width = int(image_size[1])
            height = int(image_size[0])
            self.render_product = rep.create.render_product(camera_path, (width, height))
            self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self.annotator.attach([self.render_product])
            print(f"[camera] {name}: render product attached to {camera_path}", flush=True)
        except Exception as exc:
            print(f"[warn] {name} camera reader disabled for {camera_path}: {exc}", flush=True)

    def read(self) -> np.ndarray:
        if self.annotator is None:
            return np.zeros((*self.image_size, 3), dtype=np.uint8)

        try:
            data = self.annotator.get_data()
            if isinstance(data, dict):
                data = data.get("data", data)
            return _as_hwc_uint8(data, image_size=self.image_size)
        except Exception as exc:
            if not self.warned:
                print(f"[warn] failed to read {self.name} camera: {exc}", flush=True)
                self.warned = True
            return np.zeros((*self.image_size, 3), dtype=np.uint8)


class Gr00tObservationBuilder:
    def __init__(
        self,
        *,
        modality_config: dict[str, Any],
        exterior_reader: CameraReader,
        wrist_reader: CameraReader,
        instruction: str,
        image_size: tuple[int, int],
    ):
        self.modality_config = modality_config
        self.exterior_reader = exterior_reader
        self.wrist_reader = wrist_reader
        self.instruction = instruction
        self.image_size = image_size

        video_delta = list(self.modality_config["video"].delta_indices)
        self.video_history_len = max([abs(int(i)) for i in video_delta] + [0]) + 1
        self.frame_buffer: deque[dict[str, np.ndarray]] = deque(maxlen=self.video_history_len)

    def append_frame(self) -> None:
        self.frame_buffer.append(
            {
                "exterior": self.exterior_reader.read(),
                "wrist": self.wrist_reader.read(),
            }
        )

    def build(
        self,
        *,
        arm_q: np.ndarray,
        eef_pose: np.ndarray,
        hand_q: np.ndarray,
        hand_alpha: float,
    ) -> dict[str, Any]:
        if not self.frame_buffer:
            self.append_frame()

        return {
            "video": self._build_video_dict(),
            "state": self._build_state_dict(
                arm_q=arm_q,
                eef_pose=eef_pose,
                hand_q=hand_q,
                hand_alpha=hand_alpha,
            ),
            "language": self._build_language_dict(),
        }

    def _select_history_frames(self) -> list[dict[str, np.ndarray]]:
        delta_indices = list(self.modality_config["video"].delta_indices)
        buffer = list(self.frame_buffer)
        selected = []
        for delta in delta_indices:
            delta = int(delta)
            if delta == 0:
                selected.append(buffer[-1])
            else:
                selected.append(buffer[max(delta, -len(buffer))])
        return selected

    def _build_video_dict(self) -> dict[str, np.ndarray]:
        frames = self._select_history_frames()
        video_dict: dict[str, np.ndarray] = {}
        for key in self.modality_config["video"].modality_keys:
            lowered = str(key).lower()
            frame_key = "wrist" if ("wrist" in lowered or "hand" in lowered) else "exterior"
            stack = np.stack([frame[frame_key] for frame in frames], axis=0)
            video_dict[key] = stack[None, ...].astype(np.uint8, copy=False)
        return video_dict

    def _build_state_dict(
        self,
        *,
        arm_q: np.ndarray,
        eef_pose: np.ndarray,
        hand_q: np.ndarray,
        hand_alpha: float,
    ) -> dict[str, np.ndarray]:
        arm_q = np.asarray(arm_q, dtype=np.float32).reshape(-1)
        hand_q = np.asarray(hand_q, dtype=np.float32).reshape(-1)
        eef_xyz = np.asarray(eef_pose[:3, 3], dtype=np.float32).reshape(3)
        eef_rot6d = _rotmat_to_rot6d(eef_pose[:3, :3])

        padded_arm_7 = np.zeros(7, dtype=np.float32)
        padded_arm_7[: min(6, arm_q.size)] = arm_q[: min(6, arm_q.size)]

        source = {
            "arm_joint_pos": padded_arm_7,
            "arm_eef_pos": eef_xyz,
            "hand_joint_pos": hand_q[:10],
            "eef_9d": np.concatenate([eef_xyz, eef_rot6d]).astype(np.float32),
            "joint_position": padded_arm_7,
            "gripper_position": np.asarray([hand_alpha], dtype=np.float32),
            "single_arm": padded_arm_7,
            "gripper": np.asarray([hand_alpha], dtype=np.float32),
        }

        state_dict: dict[str, np.ndarray] = {}
        for key in self.modality_config["state"].modality_keys:
            if key not in source:
                available = ", ".join(sorted(source))
                raise KeyError(f"Cannot build state key '{key}'. Available sources: {available}")
            state_dict[key] = source[key][None, None, ...].astype(np.float32, copy=False)
        return state_dict

    def _build_language_dict(self) -> dict[str, list[list[str]]]:
        return {
            language_key: [[self.instruction]]
            for language_key in self.modality_config["language"].modality_keys
        }


class XmateL10ActionExecutor:
    def __init__(
        self,
        *,
        robot: "SingleArticulation",
        ik: Any,
        pose_from_xyz_rotation: Any,
        arm_indices: np.ndarray,
        hand_indices: np.ndarray,
        args: argparse.Namespace,
    ):
        self.robot = robot
        self.ik = ik
        self.pose_from_xyz_rotation = pose_from_xyz_rotation
        self.arm_indices = arm_indices
        self.hand_indices = hand_indices
        self.args = args
        self.teleop = _load_teleop_module()

        q_seed = _parse_vector(str(args.initial_arm_q), 6, name="--initial-arm-q")
        seed_pose = self.ik.fk6(q_seed)
        self.target_xyz = np.asarray(args.initial_eef_xyz, dtype=np.float64).reshape(3).copy()
        self.target_rotation = seed_pose[:3, :3].copy()
        try:
            initial_pose = self.pose_from_xyz_rotation(self.target_xyz, self.target_rotation)
            self.q_cmd = self.ik.solve6(initial_pose, q_seed, psi=float(self.args.psi))
            reached_xyz = self.ik.fk6(self.q_cmd)[:3, 3]
            print(
                f"[init] IK initialized EEF target={self.target_xyz.tolist()} "
                f"reached={reached_xyz.tolist()} q={self.q_cmd.tolist()}",
                flush=True,
            )
        except Exception as exc:
            self.q_cmd = q_seed
            self.target_xyz = seed_pose[:3, 3].copy()
            print(
                f"[warn] Initial EEF IK failed; using --initial-arm-q instead: {exc}",
                flush=True,
            )
        self.hand_q = self.teleop.HAND_OPEN_Q.copy()
        self.hand_alpha = 0.0
        self.target_pose = self.ik.fk6(self.q_cmd)
        self.target_rotation = self.target_pose[:3, :3].copy()
        self.last_action_keys: tuple[str, ...] = ()

        self.force_apply_current_targets()

    def current_arm_q(self) -> np.ndarray:
        return self.teleop._current_arm_q(self.robot, self.arm_indices, self.q_cmd)  # noqa: SLF001

    def current_eef_pose(self) -> np.ndarray:
        return self.ik.fk6(self.current_arm_q())

    def force_apply_current_targets(self) -> None:
        self._force_joint_positions(self.arm_indices, self.q_cmd)
        self.teleop._apply_joint_positions(self.robot, self.arm_indices, self.q_cmd)  # noqa: SLF001
        self._force_joint_positions(self.hand_indices, self.hand_q)
        self.teleop._apply_l10_hand_positions(  # noqa: SLF001
            self.robot,
            self.hand_indices,
            {},
            self.hand_q,
        )

    def _force_joint_positions(self, joint_indices: np.ndarray, q: np.ndarray) -> None:
        if joint_indices.size == 0:
            return
        values = np.asarray(q, dtype=np.float32)[: joint_indices.size]
        indices = np.asarray(joint_indices, dtype=np.int32)
        for method_name in ("set_joint_positions", "set_joint_position_targets"):
            method = getattr(self.robot, method_name, None)
            if not callable(method):
                continue
            try:
                method(values, joint_indices=indices)
                continue
            except TypeError:
                pass
            except Exception:
                continue
            try:
                method(positions=values, joint_indices=indices)
            except Exception:
                continue

    def step_action(self, action: dict[str, np.ndarray], action_index: int) -> None:
        self.last_action_keys = tuple(sorted(action.keys()))

        if "arm_eef_pos_target" in action:
            arm_target = self._action_step(action["arm_eef_pos_target"], action_index)
            self.target_xyz = np.asarray(arm_target[:3], dtype=np.float64)
        elif "eef_9d" in action:
            eef_target = self._action_step(action["eef_9d"], action_index)
            self.target_xyz = np.asarray(eef_target[:3], dtype=np.float64)
        elif all(k in action for k in ("x", "y", "z")):
            self.target_xyz = np.asarray(
                [
                    self._action_step(action["x"], action_index)[0],
                    self._action_step(action["y"], action_index)[0],
                    self._action_step(action["z"], action_index)[0],
                ],
                dtype=np.float64,
            )
        elif "joint_position" in action:
            joint_target = self._action_step(action["joint_position"], action_index)
            self._apply_direct_joint_action(joint_target)

        self.target_xyz = self._clip_xyz(self.target_xyz)
        self._solve_and_apply_arm_target()

        if "hand_joint_target" in action:
            hand_target = self._action_step(action["hand_joint_target"], action_index)
            self.hand_q = np.asarray(hand_target[:10], dtype=np.float64)
            self.hand_alpha = self._hand_q_to_alpha(self.hand_q)
        else:
            gripper_value = None
            for key in ("gripper_position", "gripper", "gripper_close"):
                if key in action:
                    gripper_value = float(self._action_step(action[key], action_index)[0])
                    break
            if gripper_value is not None:
                self.hand_alpha = float(np.clip(gripper_value, 0.0, 1.0))
                self.hand_q = (
                    1.0 - self.hand_alpha
                ) * self.teleop.HAND_OPEN_Q + self.hand_alpha * self.teleop.HAND_CLOSED_Q

        self.hand_q = self._clip_hand_q(self.hand_q)
        self.teleop._apply_l10_hand_positions(  # noqa: SLF001
            self.robot,
            self.hand_indices,
            {},
            self.hand_q,
        )

    def _action_step(self, value: np.ndarray, index: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 2:
            arr = arr[min(index, arr.shape[0] - 1)]
        return arr.reshape(-1)

    def _apply_direct_joint_action(self, joint_target: np.ndarray) -> None:
        if joint_target.size < 6:
            return
        target_q = np.asarray(joint_target[:6], dtype=np.float64)
        q_seed = self.current_arm_q()
        self.q_cmd = q_seed + np.clip(
            target_q - q_seed,
            -float(self.args.max_joint_step),
            float(self.args.max_joint_step),
        )
        self.teleop._apply_joint_positions(self.robot, self.arm_indices, self.q_cmd)  # noqa: SLF001

    def _solve_and_apply_arm_target(self) -> None:
        q_seed = self.current_arm_q()
        target_pose = self.pose_from_xyz_rotation(self.target_xyz, self.target_rotation)
        try:
            q_target = self.ik.solve6(target_pose, q_seed, psi=float(self.args.psi))
            self.q_cmd = q_seed + np.clip(
                q_target - q_seed,
                -float(self.args.max_joint_step),
                float(self.args.max_joint_step),
            )
            self.teleop._apply_joint_positions(self.robot, self.arm_indices, self.q_cmd)  # noqa: SLF001
        except Exception as exc:
            print(f"[warn] IK failed for target {self.target_xyz.tolist()}: {exc}", flush=True)

    def _clip_xyz(self, xyz: np.ndarray) -> np.ndarray:
        lower = np.asarray(self.args.workspace_min, dtype=np.float64)
        upper = np.asarray(self.args.workspace_max, dtype=np.float64)
        return np.clip(np.asarray(xyz, dtype=np.float64).reshape(3), lower, upper)

    @staticmethod
    def _clip_hand_q(hand_q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(hand_q, dtype=np.float64).reshape(10), -0.6, 1.6)

    def _hand_q_to_alpha(self, hand_q: np.ndarray) -> float:
        open_q = self.teleop.HAND_OPEN_Q
        closed_q = self.teleop.HAND_CLOSED_Q
        denom = float(np.dot(closed_q - open_q, closed_q - open_q))
        if denom < 1e-12:
            return 0.0
        alpha = float(np.dot(hand_q - open_q, closed_q - open_q) / denom)
        return float(np.clip(alpha, 0.0, 1.0))


def _load_loader_module():
    global LOADER_REF
    if LOADER_REF is None:
        LOADER_REF = _import_module_from_path(
            "load_xmate3_with_right_l10hand_urdf", LOADER_SCRIPT_PATH
        )
    return LOADER_REF


def _load_teleop_module():
    global TELEOP_REF
    if TELEOP_REF is None:
        TELEOP_REF = _import_module_from_path("teleop_xmate3_l10hand_eef_ik", TELEOP_SCRIPT_PATH)
    return TELEOP_REF


def _create_ik(args: argparse.Namespace):
    teleop = _load_teleop_module()
    return teleop._create_ik(args)  # noqa: SLF001


def _name_to_dof_index(robot: "SingleArticulation") -> dict[str, int]:
    teleop = _load_teleop_module()
    return teleop._name_to_dof_index(robot)  # noqa: SLF001


def _init_world() -> Any:
    global WORLD_REF
    if WORLD_REF is None:
        try:
            from isaacsim.core.api.world import World  # type: ignore
        except Exception:
            from isaacsim.core.world import World  # type: ignore

        WORLD_REF = World(stage_units_in_meters=1.0)
        WORLD_REF.reset()
    return WORLD_REF


def _step_simulation(simulation_app: Any, *, use_world_step: bool = False) -> None:
    if use_world_step and WORLD_REF is not None:
        try:
            WORLD_REF.step(render=True)
            return
        except Exception:
            pass
    simulation_app.update()


def _get_timeline_interface() -> Any | None:
    try:
        import omni.timeline  # type: ignore

        return omni.timeline.get_timeline_interface()
    except Exception:
        return None


def _pause_timeline() -> bool:
    timeline = _get_timeline_interface()
    if timeline is None:
        return False
    try:
        timeline.pause()
        return True
    except Exception:
        try:
            timeline.stop()
            return True
        except Exception:
            return False


def _timeline_is_playing() -> bool:
    timeline = _get_timeline_interface()
    if timeline is None:
        return True
    try:
        return bool(timeline.is_playing())
    except Exception:
        return True


def _wait_for_play_if_requested(args: argparse.Namespace, simulation_app: Any) -> None:
    if not bool(args.start_paused):
        return
    if bool(getattr(args, "headless", False)):
        print("[paused] --start-paused ignored in headless mode.", flush=True)
        return

    if _pause_timeline():
        print(
            "[paused] Scene is ready. Press Play in Isaac Sim to start GR00T control.", flush=True
        )
    else:
        print("[warn] Could not pause Isaac Sim timeline; starting immediately.", flush=True)
        return

    while simulation_app.is_running() and not _timeline_is_playing():
        _update_loader_helpers(args)
        simulation_app.update()
        time.sleep(0.05)

    print("[paused] Isaac Sim timeline is playing; starting policy loop.", flush=True)


def _spawn_cameras(loader: Any, args: argparse.Namespace) -> tuple[str, str]:
    d405 = None
    try:
        d405 = loader._load_d405_json(str(args.d405_json))  # noqa: SLF001
        print(f"Loaded D405 params: {args.d405_json}", flush=True)
    except Exception as exc:
        print(f"[warn] Failed to load D405 params, using camera defaults: {exc}", flush=True)

    attach_prim = loader._find_prim_by_name(str(args.attach_link_name))  # noqa: SLF001
    if not attach_prim:
        for candidate in ("base_link", "xMate3_base_link", "world", "xMate3_link0"):
            attach_prim = loader._find_prim_by_name(candidate)  # noqa: SLF001
            if attach_prim:
                print(
                    f"[warn] attach link '{args.attach_link_name}' not found; using {candidate}",
                    flush=True,
                )
                break
    if not attach_prim:
        raise RuntimeError(f"Could not find hand camera attach link: {args.attach_link_name}")

    mount_path = f"{attach_prim}/{args.mount_prim_name}"
    loader._ensure_xform(mount_path)  # noqa: SLF001
    theta = math.radians(-10.0)
    qx = math.sin(theta * 0.5)
    qw = math.cos(theta * 0.5)
    loader._set_local_pose(  # noqa: SLF001
        mount_path,
        pos=(0.0, -0.08, 0.0),
        quat_xyzw=(qx, 0.0, 0.0, qw),
    )
    wrist_camera_path, _ = loader._spawn_camera_and_box(mount_path=mount_path, d405=d405)  # noqa: SLF001

    fixed_rig_path, fixed_camera_path, _ = loader._spawn_fixed_world_camera_with_box()  # noqa: SLF001
    print(f"[camera] wrist camera: {wrist_camera_path}", flush=True)
    print(f"[camera] fixed camera rig: {fixed_rig_path}; camera: {fixed_camera_path}", flush=True)

    if bool(args.open_camera_viewports) and not bool(getattr(args, "headless", False)):
        loader._open_camera_viewport_window(fixed_camera_path, title="Fixed D455 Camera")  # noqa: SLF001
        loader._open_camera_viewport_window(wrist_camera_path, title="Wrist D405 Camera")  # noqa: SLF001

    return fixed_camera_path, wrist_camera_path


def _init_scene(args: argparse.Namespace) -> tuple["SingleArticulation", dict[str, int], str, str]:
    global HAND_PRIMS_REF

    loader = _load_loader_module()
    teleop = _load_teleop_module()
    world = _init_world()
    if getattr(args, "table_top_z", None) is None:
        args.table_top_z = _infer_table_top_z(args.cube_position, float(args.cube_size))

    loader._add_ground_plane()  # noqa: SLF001
    loader._add_scene_light()  # noqa: SLF001
    _spawn_table_if_requested(
        enabled=bool(args.spawn_table),
        center_xy=np.asarray(args.cube_position[:2], dtype=np.float64),
        top_z=float(args.table_top_z),
        table_xy_size=float(args.table_xy_size),
    )
    _spawn_target_object(
        kind=str(getattr(args, "target_object", "cube")),
        prim_path=str(args.cube_prim_path),
        position=np.asarray(args.cube_position, dtype=np.float64),
        size=float(args.cube_size),
        mass=float(args.cube_mass),
    )
    loader._import_urdf(  # noqa: SLF001
        urdf_path=str(loader.COMBINED_URDF_PATH),
        prim_path=str(args.robot_prim_path),
        fix_base=True,
        make_instanceable=False,
    )
    loader._apply_hand_drive_gains(root_prim_path=str(args.robot_prim_path))  # noqa: SLF001

    try:
        world.reset()
    except Exception:
        pass

    HAND_PRIMS_REF = loader._collect_named_prims_under(  # noqa: SLF001
        root_prim_path=str(args.robot_prim_path),
        names=teleop.HAND_MIMIC_JOINT_NAMES,
    )
    if HAND_PRIMS_REF:
        missing = sorted(set(teleop.HAND_MIMIC_JOINT_NAMES).difference(HAND_PRIMS_REF.keys()))
        if missing:
            print(f"[hand_mimic] missing prims: {missing}", flush=True)
        else:
            print("[hand_mimic] all required joint prims found.", flush=True)

    from isaacsim.core.prims import SingleArticulation  # type: ignore

    robot = SingleArticulation(prim_path=str(args.robot_prim_path), name="xMate3_L10Hand_GR00T")
    try:
        robot.initialize(physics_sim_view=getattr(world, "physics_sim_view", None))
    except TypeError:
        robot.initialize()

    dof_index = _name_to_dof_index(robot)
    fixed_camera_path, wrist_camera_path = _spawn_cameras(loader, args)
    return robot, dof_index, fixed_camera_path, wrist_camera_path


def _update_loader_helpers(args: argparse.Namespace) -> None:
    loader = _load_loader_module()
    if HAND_PRIMS_REF:
        try:
            loader._update_hand_mimic_targets(  # noqa: SLF001
                root_prim_path=str(args.robot_prim_path),
                prims=HAND_PRIMS_REF,
            )
        except Exception:
            pass
    try:
        loader._pin_fixed_camera_rig(quiet=True)  # noqa: SLF001
    except Exception:
        pass


def _settle_initial_pose(
    executor: XmateL10ActionExecutor,
    args: argparse.Namespace,
    simulation_app: Any,
) -> None:
    frames = max(0, int(args.initial_settle_frames))
    if frames <= 0:
        return
    print(
        f"[init] forcing initial EEF pose for {frames} frames before policy start...",
        flush=True,
    )
    for i in range(frames):
        executor.force_apply_current_targets()
        _update_loader_helpers(args)
        _step_simulation(simulation_app, use_world_step=bool(args.world_step))
        eef_xyz = executor.current_eef_pose()[:3, 3]
        print(
            f"[init] settle {i + 1}/{frames}: eef_xyz={eef_xyz.tolist()} "
            f"target={executor.target_xyz.tolist()}",
            flush=True,
        )


def _print_modality_config(modality_config: dict[str, Any]) -> None:
    print("\nGR00T modality config", flush=True)
    for modality, config in modality_config.items():
        print(
            f"  {modality}: keys={config.modality_keys}, delta_indices={config.delta_indices}",
            flush=True,
        )


def _print_observation_summary(observation: dict[str, Any], step: int) -> None:
    print(f"\n[step {step}] GR00T observation", flush=True)
    for modality in ("video", "state", "language"):
        for key, value in observation[modality].items():
            if modality == "language":
                print(f"  language.{key}: {value}", flush=True)
            else:
                print(f"  {_describe_array(f'{modality}.{key}', value)}", flush=True)


def _print_action_summary(action: dict[str, np.ndarray], step: int) -> None:
    print(f"[step {step}] GR00T action", flush=True)
    for key, value in action.items():
        print(f"  {_describe_array(key, value)}", flush=True)


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy-host", type=str, default="localhost")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--policy-api-token", type=str, default=None)
    parser.add_argument("--policy-timeout-ms", type=int, default=120000)
    parser.add_argument("--instruction", type=str, default="pick up the cube")
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--replan-horizon", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument(
        "--start-paused",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pause after scene setup and wait for Play before starting GR00T control.",
    )
    parser.add_argument(
        "--camera-warmup-frames",
        type=int,
        default=2,
        help="Render this many frames before the first policy request.",
    )
    parser.add_argument(
        "--world-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use World.step(render=True) instead of simulation_app.update().",
    )
    parser.add_argument(
        "--policy-every-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request/execute policy actions every control loop, like isaaclab_gr00t_adapter.py.",
    )
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_SIZE[1])

    parser.add_argument("--robot-prim-path", type=str, default="/World/xMate3_L10Hand")
    parser.add_argument("--attach-link-name", type=str, default="xMate3_link6")
    parser.add_argument("--mount-prim-name", type=str, default="hand_camera_mount")
    parser.add_argument("--d405-json", type=str, default=str(REPO_ROOT / "d405json.json"))
    parser.add_argument(
        "--open-camera-viewports", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument(
        "--target-object",
        choices=("cube", "bottle"),
        default="cube",
        help="Object geometry to place at --cube-prim-path.",
    )
    parser.add_argument("--cube-prim-path", type=str, default="/World/grasp_cube")
    parser.add_argument(
        "--cube-position",
        type=str,
        default=",".join(str(v) for v in DEFAULT_CUBE_POS),
        help="Cube center x,y,z in world/xMate3 base frame.",
    )
    parser.add_argument("--cube-size", type=float, default=DEFAULT_CUBE_SIZE)
    parser.add_argument("--cube-mass", type=float, default=0.08)
    parser.add_argument("--spawn-table", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--table-top-z",
        type=float,
        default=None,
        help="Tabletop z height in meters. Defaults to cube_z - cube_size / 2.",
    )
    parser.add_argument(
        "--table-xy-size",
        type=float,
        default=DEFAULT_TABLE_XY_SIZE,
        help="Square tabletop side length in meters.",
    )

    parser.add_argument(
        "--initial-arm-q",
        type=str,
        default="0.0,-0.45,0.8,0.0,0.55,0.0",
        help="IK seed and default orientation source for the initial EEF target.",
    )
    parser.add_argument(
        "--initial-eef-xyz",
        type=str,
        default=",".join(str(v) for v in DEFAULT_INITIAL_EEF_XYZ),
        help="Initial EEF XYZ target solved with IK before GR00T control starts.",
    )
    parser.add_argument(
        "--initial-settle-frames",
        type=int,
        default=8,
        help="Force the initial IK joint pose for this many rendered frames before policy starts.",
    )
    parser.add_argument("--command-hz", type=float, default=20.0)
    parser.add_argument("--max-joint-step", type=float, default=0.045)
    parser.add_argument("--psi", type=float, default=0.0)
    parser.add_argument("--show-target-marker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workspace-min", type=str, default="-0.9,-0.9,0.02")
    parser.add_argument("--workspace-max", type=str, default="0.9,0.9,1.5")

    parser.add_argument(
        "--ik-backend",
        choices=("auto", "rci", "urdf"),
        default="auto",
    )
    parser.add_argument(
        "--rci-lib-path",
        type=str,
        default=str(
            REPO_ROOT
            / "external_dependencies"
            / "rci_client"
            / "build"
            / "bindings"
            / "libxmate_ik_c.so"
        ),
    )
    parser.add_argument("--rci-ip", type=str, default="127.0.0.1")
    parser.add_argument("--rci-port", type=int, default=1337)
    parser.add_argument("--kinematics-json", type=str, default=None)


def _parse_args() -> tuple[argparse.Namespace, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_args(parser)

    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:
        raise ImportError(
            "Could not import Isaac Lab. Run this script with Isaac Lab's Python, "
            "for example `${ISAACLAB_PATH}/isaaclab.sh -p ...`."
        ) from exc

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.cube_position = _parse_vector(str(args.cube_position), 3, name="--cube-position")
    if args.table_top_z is None:
        args.table_top_z = _infer_table_top_z(args.cube_position, float(args.cube_size))
    args.initial_eef_xyz = _parse_vector(str(args.initial_eef_xyz), 3, name="--initial-eef-xyz")
    args.workspace_min = _parse_vector(str(args.workspace_min), 3, name="--workspace-min")
    args.workspace_max = _parse_vector(str(args.workspace_max), 3, name="--workspace-max")
    app_launcher = AppLauncher(args)
    return args, app_launcher.app


def main() -> None:
    args, simulation_app = _parse_args()
    ik = None
    try:
        if not bool(getattr(args, "enable_cameras", False)):
            print(
                "[warn] --enable_cameras is not set. Camera render products may be black.",
                flush=True,
            )

        policy = PolicyClient(
            host=str(args.policy_host),
            port=int(args.policy_port),
            timeout_ms=int(args.policy_timeout_ms),
            api_token=args.policy_api_token,
            strict=False,
        )
        if not policy.ping():
            raise RuntimeError(
                f"Cannot reach GR00T policy server at {args.policy_host}:{args.policy_port}"
            )
        modality_config = policy.get_modality_config()
        _print_modality_config(modality_config)

        robot, dof_index, fixed_camera_path, wrist_camera_path = _init_scene(args)
        teleop = _load_teleop_module()

        missing = [
            n for n in teleop.ARM_JOINT_NAMES + teleop.HAND_JOINT_NAMES if n not in dof_index
        ]
        if missing:
            raise RuntimeError(f"Missing expected DOFs in imported robot: {missing}")

        arm_indices = np.asarray([dof_index[n] for n in teleop.ARM_JOINT_NAMES], dtype=np.int32)
        hand_indices = np.asarray([dof_index[n] for n in teleop.HAND_JOINT_NAMES], dtype=np.int32)

        ik, pose_from_xyz_rotation = _create_ik(args)
        executor = XmateL10ActionExecutor(
            robot=robot,
            ik=ik,
            pose_from_xyz_rotation=pose_from_xyz_rotation,
            arm_indices=arm_indices,
            hand_indices=hand_indices,
            args=args,
        )
        _settle_initial_pose(executor, args, simulation_app)

        image_size = (int(args.image_height), int(args.image_width))
        exterior_reader = CameraReader(fixed_camera_path, image_size=image_size, name="fixed")
        wrist_reader = CameraReader(wrist_camera_path, image_size=image_size, name="wrist")
        obs_builder = Gr00tObservationBuilder(
            modality_config=modality_config,
            exterior_reader=exterior_reader,
            wrist_reader=wrist_reader,
            instruction=str(args.instruction),
            image_size=image_size,
        )

        marker_path = (
            _create_target_marker("/World/gr00t_eef_target") if args.show_target_marker else None
        )
        _set_target_marker(marker_path, executor.target_xyz)

        print(
            f"[scene] target_object={getattr(args, 'target_object', 'cube')} "
            f"prim={args.cube_prim_path} pos={args.cube_position.tolist()} "
            f"size={float(args.cube_size):.3f}m",
            flush=True,
        )
        print(
            f"[policy] connected to {args.policy_host}:{args.policy_port}; "
            f"instruction={args.instruction!r}",
            flush=True,
        )
        _wait_for_play_if_requested(args, simulation_app)

        action_chunk: dict[str, np.ndarray] | None = None
        action_index = 0
        command_dt = 1.0 / max(float(args.command_hz), 1e-6)
        last_command_time = 0.0
        step = 0

        print("[warmup] rendering initial camera frames...", flush=True)
        for warmup_step in range(max(0, int(args.camera_warmup_frames))):
            _update_loader_helpers(args)
            _step_simulation(simulation_app, use_world_step=bool(args.world_step))
            print(f"[warmup] frame {warmup_step + 1}/{args.camera_warmup_frames}", flush=True)

        while simulation_app.is_running():
            _update_loader_helpers(args)

            now = time.perf_counter()
            if bool(args.policy_every_step) or now - last_command_time >= command_dt:
                last_command_time = now

                need_replan = action_chunk is None or action_index >= int(args.replan_horizon)
                if need_replan:
                    if step % int(args.print_every) == 0:
                        print(f"[step {step}] Capturing camera frames...", flush=True)
                    obs_builder.append_frame()
                    if step % int(args.print_every) == 0:
                        print(f"[step {step}] Camera frames ready.", flush=True)
                    eef_pose = executor.current_eef_pose()
                    observation = obs_builder.build(
                        arm_q=executor.current_arm_q(),
                        eef_pose=eef_pose,
                        hand_q=executor.hand_q,
                        hand_alpha=executor.hand_alpha,
                    )
                    if step % int(args.print_every) == 0:
                        _print_observation_summary(observation, step)

                    request_start = time.perf_counter()
                    print(f"[step {step}] Requesting GR00T action...", flush=True)
                    action_chunk, _ = policy.get_action(observation)
                    elapsed = time.perf_counter() - request_start
                    print(f"[step {step}] GR00T action received in {elapsed:.2f}s", flush=True)
                    if step % int(args.print_every) == 0:
                        _print_action_summary(action_chunk, step)
                    action_index = 0

                assert action_chunk is not None
                executor.step_action(action_chunk, action_index)
                _set_target_marker(marker_path, executor.target_xyz)
                action_index += 1
                step += 1

                if step % int(args.print_every) == 0:
                    cube_pos = _get_prim_world_position(str(args.cube_prim_path))
                    cube_text = cube_pos.tolist() if cube_pos is not None else None
                    current_eef = executor.current_eef_pose()[:3, 3].tolist()
                    print(
                        f"[step {step}] target_xyz={executor.target_xyz.tolist()} "
                        f"eef_xyz={current_eef} hand_alpha={executor.hand_alpha:.3f} "
                        f"cube_pos={cube_text}",
                        flush=True,
                    )

                if int(args.max_steps) > 0 and step >= int(args.max_steps):
                    print(f"[done] reached max steps: {args.max_steps}", flush=True)
                    break

            _step_simulation(simulation_app, use_world_step=bool(args.world_step))

    except Exception as exc:
        print(f"[error] GR00T xMate3 L10 cube grasp failed: {exc}", flush=True)
        raise
    finally:
        try:
            if ik is not None:
                ik.close()
        except Exception:
            pass
        simulation_app.close()


if __name__ == "__main__":
    main()
