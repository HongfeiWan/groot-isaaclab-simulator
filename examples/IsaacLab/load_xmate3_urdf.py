#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal IsaacLab script: load xMate3 arm + L10 hand URDF into the stage.

运行（用 IsaacLab 的 Python）：

    ~/Project/IsaacLab/isaaclab.sh -p examples/IsaacLab/load_xmate3_urdf.py

它会把：
- `demo_data/l10_hand/xMate3_description/urdf/xMate3.urdf`（机械臂）
- `demo_data/l10_hand/right-L10hand/linkerhand_l10_right.urdf`（右手）

导入到 USD stage，并把手挂到机械臂末端 link（默认 `xMate3_link6`）下面的 `hand_mount`。
同时会弹出一个简单的 Isaac Sim UI 面板用于调 `hand_mount` 的相对位姿（pos/quat）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ARM_URDF_PATH = (
    REPO_ROOT / "demo_data" / "l10_hand" / "xMate3_description" / "urdf" / "xMate3.urdf"
)
HAND_URDF_PATH = REPO_ROOT / "demo_data" / "l10_hand" / "right-L10hand" / "linkerhand_l10_right.urdf"
_UI_REFS: list[Any] = []

# Your calibrated, fixed assembly constraint (mount local pose under the arm end link).
# Note: quaternion will be normalized internally.
FIXED_HAND_MOUNT_POS = (0.0, 0.0, 0.0)
FIXED_HAND_MOUNT_QUAT_XYZW = (0.0, 0.0, -1.0, 1.0)


def _parse_args() -> tuple[argparse.Namespace, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-urdf", type=str, default=str(ARM_URDF_PATH))
    parser.add_argument("--hand-urdf", type=str, default=str(HAND_URDF_PATH))

    parser.add_argument("--arm-prim-path", type=str, default="/World/xMate3")
    parser.add_argument("--hand-prim-path", type=str, default="/World/L10Hand")

    parser.add_argument("--arm-fix-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hand-fix-base", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--make-instanceable", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument(
        "--attach-link-name",
        type=str,
        default="xMate3_link6",
        help="Arm end-effector link name to attach the hand mount under.",
    )
    parser.add_argument(
        "--mount-prim-name",
        type=str,
        default="hand_mount",
        help="Name of the Xform prim created under the arm link for hand mounting.",
    )

    parser.add_argument(
        "--tune-mount",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show an Isaac Sim UI panel to tune the mount transform (pos/quat).",
    )
    parser.add_argument(
        "--lock-mount",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force the hand mount transform to the fixed calibrated pose every tick.",
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    return args, app


def _add_ground_plane() -> None:
    try:
        import isaaclab.sim as sim_utils

        sim_utils.GroundPlaneCfg().func("/World/groundPlane", sim_utils.GroundPlaneCfg())
    except Exception:
        # Ground plane is optional for visualization.
        pass


def _import_urdf(*, urdf_path: str, prim_path: str, fix_base: bool, make_instanceable: bool) -> None:
    # Isaac Sim URDF Importer (works inside IsaacLab runtime).
    import omni.kit.commands  # type: ignore
    import omni.usd  # type: ignore
    from pxr import Sdf, UsdGeom

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Failed to create URDF import config.")

    import_config.merge_fixed_joints = False
    import_config.fix_base = fix_base
    import_config.make_default_prim = True
    import_config.create_physics_scene = True
    if hasattr(import_config, "make_instanceable"):
        import_config.make_instanceable = make_instanceable
    elif make_instanceable:
        print("[warn] This Isaac Sim URDF importer does not expose make_instanceable.", flush=True)

    status, imported_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=urdf_path, import_config=import_config
    )
    if not status or not imported_path:
        raise RuntimeError(f"URDF import failed: status={status}, imported_path={imported_path!r}")

    stage = omni.usd.get_context().get_stage()
    target_path = Sdf.Path(prim_path)
    if target_path.pathString != str(imported_path):
        parent_path = target_path.GetParentPath()
        if parent_path != Sdf.Path.absoluteRootPath and not stage.GetPrimAtPath(parent_path):
            UsdGeom.Xform.Define(stage, parent_path)
        omni.kit.commands.execute(
            "MovePrimCommand",
            path_from=str(imported_path),
            path_to=target_path.pathString,
            keep_world_transform=True,
        )


def _find_prim_by_name(name: str) -> str | None:
    import omni.usd  # type: ignore

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def _ensure_xform(path: str) -> None:
    import omni.usd  # type: ignore
    from pxr import UsdGeom  # type: ignore

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available.")
    if stage.GetPrimAtPath(path) and stage.GetPrimAtPath(path).IsValid():
        return
    UsdGeom.Xform.Define(stage, path)


def _get_local_pose(prim_path: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return (pos xyz, quat xyzw) from local Xform matrix (assumes no scale/shear)."""
    import omni.usd  # type: ignore
    from pxr import UsdGeom  # type: ignore
    import numpy as np

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available.")
    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    local_result = xform.GetLocalTransformation()
    local = local_result[0] if isinstance(local_result, tuple) else local_result
    t = local.ExtractTranslation()
    r = local.ExtractRotationQuat()  # (w, imag xyz)
    pos = (float(t[0]), float(t[1]), float(t[2]))
    quat_xyzw = (
        float(r.GetImaginary()[0]),
        float(r.GetImaginary()[1]),
        float(r.GetImaginary()[2]),
        float(r.GetReal()),
    )
    q = np.array(quat_xyzw, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n > 1e-9:
        q = q / n
    else:
        q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    quat_xyzw = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return pos, quat_xyzw


def _set_local_pose(prim_path: str, *, pos: tuple[float, float, float], quat_xyzw: tuple[float, float, float, float]) -> None:
    import omni.usd  # type: ignore
    from pxr import Gf, UsdGeom  # type: ignore
    import numpy as np

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available.")
    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    if xform.GetOrderedXformOps():
        xform.ClearXformOpOrder()

    translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)

    translate_op.Set(Gf.Vec3d(*pos))
    x, y, z, w = quat_xyzw
    q = np.array([x, y, z, w], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n > 1e-9:
        q = q / n
    x, y, z, w = [float(v) for v in q.tolist()]
    orient_op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))


def _move_prim(path_from: str, path_to: str, *, keep_world_transform: bool) -> None:
    import omni.kit.commands  # type: ignore

    omni.kit.commands.execute(
        "MovePrimCommand",
        path_from=path_from,
        path_to=path_to,
        keep_world_transform=keep_world_transform,
    )


def _zero_local_xform_ops(prim_path: str) -> None:
    import omni.usd  # type: ignore
    from pxr import UsdGeom  # type: ignore

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        return
    xform = UsdGeom.Xformable(prim)
    if xform.GetOrderedXformOps():
        xform.ClearXformOpOrder()


def _attach_hand_to_arm(*, hand_root: str, arm_link_name: str, mount_prim_name: str) -> str:
    """Attach hand root prim under arm link and return mount prim path."""
    arm_link_prim = _find_prim_by_name(arm_link_name)
    if not arm_link_prim:
        raise RuntimeError(f"Could not find arm link prim by name '{arm_link_name}'.")

    mount_path = f"{arm_link_prim}/{mount_prim_name}"
    _ensure_xform(mount_path)

    new_hand_path = f"{mount_path}/hand"
    _move_prim(hand_root, new_hand_path, keep_world_transform=True)

    # Transfer hand local pose to mount, then zero hand local xform for easier tuning.
    pos, quat = _get_local_pose(new_hand_path)
    _set_local_pose(mount_path, pos=pos, quat_xyzw=quat)
    _zero_local_xform_ops(new_hand_path)

    return mount_path


def _run_mount_ui(*, mount_path: str) -> None:
    import omni.ui as ui  # type: ignore

    pos, quat = _get_local_pose(mount_path)
    state = {"pos": list(pos), "quat": list(quat)}

    window = ui.Window("Hand Mount Tuner (xMate3 + L10)", width=420, height=260, visible=True)
    _UI_REFS.append(window)
    with window.frame:
        with ui.VStack(spacing=10):
            ui.Label("调的是 hand_mount 的局部位姿（相对机械臂末端 link）。", height=0)
            ui.Label(f"prim: {mount_path}", word_wrap=True, height=0)

            def _apply() -> None:
                _set_local_pose(
                    mount_path,
                    pos=(float(state["pos"][0]), float(state["pos"][1]), float(state["pos"][2])),
                    quat_xyzw=(
                        float(state["quat"][0]),
                        float(state["quat"][1]),
                        float(state["quat"][2]),
                        float(state["quat"][3]),
                    ),
                )

            def _row3(label: str, values: list[float]) -> None:
                with ui.HStack(height=0):
                    ui.Label(label, width=90)
                    fields = [ui.FloatField(width=95) for _ in range(3)]
                    for f, v in zip(fields, values, strict=False):
                        f.model.set_value(float(v))

                    def _on_change(_m=None) -> None:
                        for i, f in enumerate(fields):
                            values[i] = float(f.model.get_value_as_float())
                        _apply()

                    for f in fields:
                        f.model.add_value_changed_fn(_on_change)

            def _row4(label: str, values: list[float]) -> None:
                with ui.HStack(height=0):
                    ui.Label(label, width=90)
                    fields = [ui.FloatField(width=70) for _ in range(4)]
                    for f, v in zip(fields, values, strict=False):
                        f.model.set_value(float(v))

                    def _on_change(_m=None) -> None:
                        for i, f in enumerate(fields):
                            values[i] = float(f.model.get_value_as_float())
                        _apply()

                    for f in fields:
                        f.model.add_value_changed_fn(_on_change)

            _row3("pos (m)", state["pos"])
            _row4("quat xyzw", state["quat"])
            _UI_REFS.append(state)

            with ui.HStack(height=0, spacing=8):
                def _refresh() -> None:
                    p, q = _get_local_pose(mount_path)
                    state["pos"][:] = list(p)
                    state["quat"][:] = list(q)
                    print(f"[mount] refreshed: pos={p}, quat_xyzw={q}", flush=True)

                def _print() -> None:
                    p = state["pos"]
                    q = state["quat"]
                    print("\n[hand_mount] copy these into your main script:", flush=True)
                    print(f'  --hand-mount-pos "{p[0]:.5f},{p[1]:.5f},{p[2]:.5f}"', flush=True)
                    print(f'  --hand-mount-rot "{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f}"', flush=True)

                ui.Button("Refresh", clicked_fn=_refresh)
                ui.Button("Print", clicked_fn=_print)

    print(f"[mount] UI panel opened: Hand Mount Tuner (xMate3 + L10), prim={mount_path}", flush=True)


def main() -> None:
    args, simulation_app = _parse_args()

    arm_urdf = str(Path(args.arm_urdf).expanduser())
    hand_urdf = str(Path(args.hand_urdf).expanduser())
    if not Path(arm_urdf).exists():
        raise FileNotFoundError(f"Arm URDF not found: {arm_urdf}")
    if not Path(hand_urdf).exists():
        raise FileNotFoundError(f"Hand URDF not found: {hand_urdf}")

    _add_ground_plane()
    _import_urdf(urdf_path=arm_urdf, prim_path=args.arm_prim_path, fix_base=bool(args.arm_fix_base), make_instanceable=bool(args.make_instanceable))
    print(f"Imported ARM URDF: {arm_urdf} -> {args.arm_prim_path}", flush=True)

    _import_urdf(urdf_path=hand_urdf, prim_path=args.hand_prim_path, fix_base=bool(args.hand_fix_base), make_instanceable=bool(args.make_instanceable))
    print(f"Imported HAND URDF: {hand_urdf} -> {args.hand_prim_path}", flush=True)

    mount_path = _attach_hand_to_arm(
        hand_root=args.hand_prim_path,
        arm_link_name=args.attach_link_name,
        mount_prim_name=args.mount_prim_name,
    )
    print(f"Attached hand under mount: {mount_path}", flush=True)

    # Enforce the fixed assembly pose immediately.
    _set_local_pose(mount_path, pos=FIXED_HAND_MOUNT_POS, quat_xyzw=FIXED_HAND_MOUNT_QUAT_XYZW)
    print(
        f"[hand_mount] locked pose set: pos={FIXED_HAND_MOUNT_POS}, quat_xyzw={FIXED_HAND_MOUNT_QUAT_XYZW}",
        flush=True,
    )

    if bool(args.tune_mount) and not bool(getattr(args, "headless", False)):
        _run_mount_ui(mount_path=mount_path)

    try:
        while simulation_app.is_running():
            if bool(args.lock_mount):
                # Re-apply the fixed pose to make this constraint "always true",
                # even if something else modifies the prim transform.
                try:
                    _set_local_pose(
                        mount_path,
                        pos=FIXED_HAND_MOUNT_POS,
                        quat_xyzw=FIXED_HAND_MOUNT_QUAT_XYZW,
                    )
                except Exception:
                    pass
            simulation_app.update()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
