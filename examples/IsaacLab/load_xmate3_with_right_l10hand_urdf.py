#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal IsaacLab script: load combined xMate3 + right L10 hand URDF into the stage.

运行（用 IsaacLab 的 Python）：

    ~/Project/IsaacLab/isaaclab.sh -p examples/IsaacLab/load_xmate3_with_right_l10hand_urdf.py

它会把：
- `demo_data/l10_hand/xMate3_with_right_L10hand.urdf`

直接导入到 USD stage，用于可视化检查。
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMBINED_URDF_PATH = REPO_ROOT / "demo_data" / "l10_hand" / "xMate3_with_right_L10hand.urdf"


def _parse_args() -> tuple[argparse.Namespace, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=str, default=str(COMBINED_URDF_PATH))
    parser.add_argument("--prim-path", type=str, default="/World/xMate3_L10Hand")
    parser.add_argument("--fix-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-instanceable", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument(
        "--attach-link-name",
        type=str,
        default="xMate3_link6",
        help="Attach the hand camera mount under this link prim name (searched by prim name in stage).",
    )
    parser.add_argument(
        "--mount-prim-name",
        type=str,
        default="hand_camera_mount",
        help="Name of the Xform prim created under the attach link for mounting the camera.",
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
    """Import a URDF into the current USD stage at the given prim path."""
    import omni.kit.commands  # type: ignore
    import omni.usd  # type: ignore
    from pxr import Sdf, UsdGeom  # type: ignore

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
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        return
    UsdGeom.Xform.Define(stage, path)


def _set_local_pos(prim_path: str, *, pos: tuple[float, float, float]) -> None:
    import omni.usd  # type: ignore
    from pxr import Gf, UsdGeom  # type: ignore

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
    translate_op.Set(Gf.Vec3d(*pos))


def _spawn_camera_and_box(*, mount_path: str) -> tuple[str, str]:
    """Spawn a USD Camera prim and a small box mesh under the mount.

    Returns:
        (camera_prim_path, box_prim_path)
    """
    import omni.usd  # type: ignore
    from pxr import Gf, UsdGeom  # type: ignore

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available.")

    camera_path = f"{mount_path}/camera"
    box_path = f"{mount_path}/camera_body"

    UsdGeom.Camera.Define(stage, camera_path)

    # A visible proxy for the camera in the viewport.
    cube = UsdGeom.Cube.Define(stage, box_path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(box_path))
    if xform.GetOrderedXformOps():
        xform.ClearXformOpOrder()
    # Cube is centered at origin; scale to the desired physical dimensions.
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.042, 0.042, 0.023))

    return camera_path, box_path


def main() -> None:
    args, simulation_app = _parse_args()

    urdf = str(Path(args.urdf).expanduser())
    if not Path(urdf).exists():
        raise FileNotFoundError(f"URDF not found: {urdf}")

    _add_ground_plane()
    _import_urdf(
        urdf_path=urdf,
        prim_path=args.prim_path,
        fix_base=bool(args.fix_base),
        make_instanceable=bool(args.make_instanceable),
    )
    print(f"Imported URDF: {urdf} -> {args.prim_path}", flush=True)

    attach_prim = _find_prim_by_name(str(args.attach_link_name))
    if not attach_prim:
        # Common fallbacks if the default is not found in the combined URDF.
        for candidate in ("base_link", "xMate3_base_link", "world", "xMate3_link0"):
            attach_prim = _find_prim_by_name(candidate)
            if attach_prim:
                print(
                    f"[warn] attach link '{args.attach_link_name}' not found; using '{candidate}' at {attach_prim}",
                    flush=True,
                )
                break
    if not attach_prim:
        raise RuntimeError(
            f"Could not find attach link prim by name '{args.attach_link_name}' (or common fallbacks)."
        )

    mount_path = f"{attach_prim}/{args.mount_prim_name}"
    _ensure_xform(mount_path)
    _set_local_pos(mount_path, pos=(0.0, -0.1, 0.0))
    camera_path, box_path = _spawn_camera_and_box(mount_path=mount_path)
    print(f"Created hand camera mount: {mount_path} (local +Y 0.5m)", flush=True)
    print(f"Spawned camera prim: {camera_path}", flush=True)
    print(f"Spawned camera body box: {box_path} (0.042 x 0.042 x 0.023 m)", flush=True)

    try:
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

