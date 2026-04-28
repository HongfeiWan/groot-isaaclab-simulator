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
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMBINED_URDF_PATH = REPO_ROOT / "demo_data" / "l10_hand" / "xMate3_with_right_L10hand.urdf"
D405_JSON_PATH = REPO_ROOT / "d405json.json"


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
    parser.add_argument(
        "--show-hand-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open a dedicated viewport window showing the hand camera.",
    )
    parser.add_argument(
        "--d405-json",
        type=str,
        default=str(D405_JSON_PATH),
        help="JSON file containing D405 recommended camera parameters.",
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


def _load_d405_json(path: str) -> dict:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"D405 json not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _apply_d405_params_to_usd_camera(camera, d405: dict) -> None:
    """Apply D405 recommended parameters to a UsdGeom.Camera prim."""
    from pxr import Gf  # type: ignore

    focal_m = float(d405["focal_length_m"])
    hfov_deg = float(d405["fov_degrees"]["horizontal"]["recommended"])
    vfov_deg = float(d405["fov_degrees"]["vertical"])

    # Isaac Sim/IsaacLab commonly interpret these as centimeters.
    focal_cm = focal_m * 100.0
    horiz_aperture_cm = 2.0 * focal_m * math.tan(math.radians(hfov_deg) * 0.5) * 100.0
    vert_aperture_cm = 2.0 * focal_m * math.tan(math.radians(vfov_deg) * 0.5) * 100.0

    camera.CreateFocalLengthAttr(float(focal_cm))
    camera.CreateHorizontalApertureAttr(float(horiz_aperture_cm))
    camera.CreateVerticalApertureAttr(float(vert_aperture_cm))
    camera.CreateFStopAttr(float(d405["f_stop"]))

    focus_candidates = d405.get("focus_distance_m", {}).get("recommended", [0.2])
    focus_m = float(focus_candidates[0]) if focus_candidates else 0.2
    camera.CreateFocusDistanceAttr(float(focus_m))

    clip_near = float(d405["clipping_range_m"]["near"])
    clip_far = float(d405["clipping_range_m"]["far"]["recommended"])
    camera.CreateClippingRangeAttr(Gf.Vec2f(clip_near, clip_far))


def _spawn_camera_and_box(*, mount_path: str, d405: dict | None) -> tuple[str, str]:
    """Spawn a USD Camera prim and a small box mesh under the mount.

    Returns:
        (camera_prim_path, box_prim_path)
    """
    import omni.usd  # type: ignore
    from pxr import Gf, UsdGeom  # type: ignore

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available.")

    # USD cameras typically look along -Z. We want the view to be aligned with the mount's +Z,
    # so we insert an "optical" frame that rotates 180 degrees about Y.
    optical_path = f"{mount_path}/camera_optical"
    camera_path = f"{optical_path}/camera"
    box_path = f"{mount_path}/camera_body"

    UsdGeom.Xform.Define(stage, optical_path)
    optical_xform = UsdGeom.Xformable(stage.GetPrimAtPath(optical_path))
    if optical_xform.GetOrderedXformOps():
        optical_xform.ClearXformOpOrder()
    # Place the camera on the box +Z face (box is centered at mount origin).
    # Box depth is 0.023m, so offset is 0.023/2 = 0.0115m along mount +Z.
    optical_xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0115))
    # Rotate 180 degrees around Y: (x,y,z,w) = (0,1,0,0) in quaternion.
    optical_xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(0.0, Gf.Vec3d(0.0, 1.0, 0.0)))

    camera = UsdGeom.Camera.Define(stage, camera_path)
    if d405 is not None:
        _apply_d405_params_to_usd_camera(camera, d405)

    # A visible proxy for the camera in the viewport.
    cube = UsdGeom.Cube.Define(stage, box_path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(box_path))
    if xform.GetOrderedXformOps():
        xform.ClearXformOpOrder()
    # Cube is centered at origin; scale to the desired physical dimensions.
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.042, 0.042, 0.023))

    return camera_path, box_path


def _open_camera_viewport_window(camera_prim_path: str) -> None:
    """Open a viewport window and set its active camera."""
    try:
        from omni.kit.viewport.utility import create_viewport_window, get_active_viewport  # type: ignore
    except Exception as e:
        print(f"[warn] viewport utilities not available; cannot open camera window: {e}", flush=True)
        return

    # Create a dedicated viewport window if possible.
    viewport = None
    try:
        viewport = create_viewport_window("Hand Camera", width=640, height=480)
    except Exception:
        viewport = None

    # Fallback to the currently active viewport if window creation failed.
    if viewport is None:
        try:
            viewport = get_active_viewport()
        except Exception:
            viewport = None

    if viewport is None:
        print("[warn] No viewport available; cannot bind hand camera.", flush=True)
        return

    try:
        viewport.camera_path = camera_prim_path
        print(f"[hand_camera] viewport bound to camera: {camera_prim_path}", flush=True)
    except Exception as e:
        print(f"[warn] Failed to set viewport camera_path: {e}", flush=True)


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

    d405 = None
    try:
        d405 = _load_d405_json(str(args.d405_json))
        print(f"Loaded D405 params: {args.d405_json}", flush=True)
    except Exception as e:
        print(f"[warn] Failed to load D405 params, using defaults: {e}", flush=True)

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
    _set_local_pos(mount_path, pos=(0.0, -0.08, 0.0))
    camera_path, box_path = _spawn_camera_and_box(mount_path=mount_path, d405=d405)
    print(f"Created hand camera mount: {mount_path} (local Y -0.08m)", flush=True)
    print(f"Spawned camera prim: {camera_path}", flush=True)
    print(f"Spawned camera body box: {box_path} (0.042 x 0.042 x 0.023 m)", flush=True)

    # Show the hand camera view in a dedicated viewport window (requires cameras enabled).
    if bool(args.show_hand_camera) and not bool(getattr(args, "headless", False)):
        if not bool(getattr(args, "enable_cameras", False)):
            print(
                "[warn] --enable_cameras is not set. If the camera view is black, relaunch with --enable_cameras.",
                flush=True,
            )
        _open_camera_viewport_window(camera_path)

    try:
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

