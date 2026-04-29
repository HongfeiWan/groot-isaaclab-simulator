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
RSD455_USD_RELATIVE_PATH = "Isaac/Sensors/Intel/RealSense/rsd455.usd"

# Hand joint drive gains: (stiffness, damping)
GAINS: dict[str, tuple[float, float]] = {
    "thumb_cmc_roll": (3000.0, 300.0),
    "thumb_cmc_yaw": (4000.0, 400.0),
    "thumb_cmc_pitch": (5000.0, 500.0),
    "mcp_roll": (2500.0, 250.0),
    "mcp_pitch": (5000.0, 500.0),
    "thumb_mcp": (3500.0, 350.0),
    "thumb_ip": (2500.0, 250.0),
    "pip": (3000.0, 300.0),
    "dip": (1800.0, 180.0),
}


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


def _set_local_pose(
    prim_path: str,
    *,
    pos: tuple[float, float, float],
    quat_xyzw: tuple[float, float, float, float],
) -> None:
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

    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
    x, y, z, w = quat_xyzw
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))


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


def _apply_d455_fallback_params_to_usd_camera(camera) -> None:
    """Apply D455-like camera params when the Isaac Sim D455 USD asset is unavailable."""
    from pxr import Gf  # type: ignore

    # Match the D455-like Isaac parameters used elsewhere in this script family.
    focal_m = 0.00193
    hfov_deg = 87.0
    vfov_deg = 58.0
    focal_cm = focal_m * 100.0
    horiz_aperture_cm = 2.0 * focal_m * math.tan(math.radians(hfov_deg) * 0.5) * 100.0
    vert_aperture_cm = 2.0 * focal_m * math.tan(math.radians(vfov_deg) * 0.5) * 100.0

    camera.CreateFocalLengthAttr(float(focal_cm))
    camera.CreateHorizontalApertureAttr(float(horiz_aperture_cm))
    camera.CreateVerticalApertureAttr(float(vert_aperture_cm))
    camera.CreateFStopAttr(2.0)
    camera.CreateFocusDistanceAttr(0.4)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.07, 10.0))


def _get_isaac_assets_root_path() -> str | None:
    for module_name in (
        "isaacsim.core.utils.nucleus",
        "omni.isaac.core.utils.nucleus",
    ):
        try:
            module = __import__(module_name, fromlist=["get_assets_root_path"])
            root = module.get_assets_root_path()
        except Exception:
            continue
        if root:
            return str(root).rstrip("/")
    return None


def _rsd455_usd_candidates() -> list[str]:
    candidates: list[str] = []
    assets_root = _get_isaac_assets_root_path()
    if assets_root:
        candidates.append(f"{assets_root}/{RSD455_USD_RELATIVE_PATH}")

    candidates.extend(
        [
            f"omniverse://localhost/NVIDIA/Assets/Isaac/5.1/{RSD455_USD_RELATIVE_PATH}",
            f"omniverse://localhost/NVIDIA/Assets/Isaac/5.0/{RSD455_USD_RELATIVE_PATH}",
            f"omniverse://localhost/NVIDIA/Assets/Isaac/4.5/{RSD455_USD_RELATIVE_PATH}",
        ]
    )

    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _usd_asset_exists(asset_path: str) -> bool:
    if "://" not in asset_path:
        return Path(asset_path).expanduser().exists()

    try:
        import omni.client  # type: ignore

        result, _ = omni.client.stat(asset_path)
        return result == omni.client.Result.OK
    except Exception:
        # Some Isaac Sim installs do not expose omni.client before the app is fully initialized.
        # Let USD try the reference in that case.
        return True


def _try_reference_rsd455_usd(prim_path: str) -> str | None:
    import omni.usd  # type: ignore
    from pxr import UsdGeom  # type: ignore

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available.")

    prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    for asset_path in _rsd455_usd_candidates():
        if not _usd_asset_exists(asset_path):
            continue
        try:
            prim.GetReferences().ClearReferences()
            prim.GetReferences().AddReference(asset_path)
            print(f"[fixed_camera] referenced D455 USD: {asset_path}", flush=True)
            return asset_path
        except Exception as e:
            print(f"[warn] Failed to reference D455 USD '{asset_path}': {e}", flush=True)

    print("[warn] Could not find/reference rsd455.usd; falling back to box + manual D455 camera.", flush=True)
    return None


def _find_first_camera_under(root_prim_path: str) -> str | None:
    import omni.usd  # type: ignore
    from pxr import UsdGeom  # type: ignore

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        return None

    root_prefix = str(root.GetPath())
    if not root_prefix.endswith("/"):
        root_prefix = root_prefix + "/"

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path != str(root.GetPath()) and not prim_path.startswith(root_prefix):
            continue
        if prim.IsA(UsdGeom.Camera):
            return prim_path
    return None


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


def _spawn_fixed_world_camera_with_box() -> tuple[str, str, str]:
    """Spawn a world-fixed D455 camera and its visible body.

    Camera rig pose in world:
        x=0.1, y=-0.3, z=0.6 (meters) on ``fixed_camera_rig``; child ``fixed_camera_rig/yaw``
        applies RotateXYZ=(180, 135, 0) deg for IsaacLab. Box and camera live
        under ``yaw`` so they share the same rotation without ambiguous translate/rotate op order.
    Camera body:
        Prefer Isaac Sim's D455 asset ``rsd455.usd``. If unavailable, fall back
        to a 124 mm x 29 mm x 26 mm proxy box, with thickness(26 mm) along +X
        and 124 mm along +Y: scale (x, y, z) = (0.026, 0.124, 0.029) m.
    Camera pose relative to body center:
        after yaw, move 0.013 m along body local -X to lie on the box surface,
        looking toward body local -X.
    Image axes vs box (rig-aligned) axes:
        image +u (horizontal) -> box +Y; image +v (vertical) -> box +Z.

    Returns:
        (rig_prim_path, camera_prim_path, box_prim_path)
    """
    import omni.usd  # type: ignore
    from pxr import Gf, UsdGeom  # type: ignore

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available.")

    rig_path = "/World/fixed_camera_rig"
    # Split translate vs rotation so the rig pivots at its world position.
    yaw_path = f"{rig_path}/yaw"
    body_path = f"{yaw_path}/rsd455"
    camera_offset_path = f"{yaw_path}/camera_offset"
    camera_frame_path = f"{camera_offset_path}/camera_frame"
    camera_path = f"{camera_frame_path}/camera"

    UsdGeom.Xform.Define(stage, rig_path)
    _set_local_pos(rig_path, pos=(0.1, -0.3, 0.6))

    UsdGeom.Xform.Define(stage, yaw_path)
    yaw_xform = UsdGeom.Xformable(stage.GetPrimAtPath(yaw_path))
    if yaw_xform.GetOrderedXformOps():
        yaw_xform.ClearXformOpOrder()
    rig_rotation_xyz_deg = Gf.Vec3d(180.0, 135.0, 0.0)
    yaw_xform.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(rig_rotation_xyz_deg)

    referenced_d455_usd = _try_reference_rsd455_usd(body_path)
    asset_camera_path = _find_first_camera_under(body_path) if referenced_d455_usd is not None else None
    if asset_camera_path is not None:
        print(f"[fixed_camera] using D455 USD camera prim: {asset_camera_path}", flush=True)
        return rig_path, asset_camera_path, body_path

    if referenced_d455_usd is not None:
        print("[warn] D455 USD referenced but no Camera prim was found; creating manual D455 camera.", flush=True)
    else:
        # Visible fallback body proxy.
        body = UsdGeom.Cube.Define(stage, body_path)
        body.CreateSizeAttr(1.0)
        body_xform = UsdGeom.Xformable(stage.GetPrimAtPath(body_path))
        if body_xform.GetOrderedXformOps():
            body_xform.ClearXformOpOrder()
        body_xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(0.026, 0.124, 0.029)
        )

    # USD camera: +X is image right, +Y is image up, view is -Z.
    # Map camera -> rig: +X_cam -> +Y_box, +Y_cam -> +Z_box, +Z_cam -> +X_box
    # so -Z_cam -> -X_box and image u,v align with box Y,Z.
    # Same transform as matrix [[0,0,1],[1,0,0],[0,1,0]]; pxr may not expose
    # Gf.Rotation(Matrix3d), so use axis-angle: +120 deg about (1,1,1)/sqrt(3).
    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    cam_axis = Gf.Vec3d(inv_sqrt3, inv_sqrt3, inv_sqrt3)
    cam_orient = Gf.Rotation(cam_axis, 120.0).GetQuat()
    # Offset and orientation are intentionally split:
    # - camera_offset translates along the already-yawed body -X axis to the box surface.
    # - camera_frame only orients the optical frame, so the offset is not affected by it.
    UsdGeom.Xform.Define(stage, camera_offset_path)
    camera_offset = UsdGeom.Xformable(stage.GetPrimAtPath(camera_offset_path))
    if camera_offset.GetOrderedXformOps():
        camera_offset.ClearXformOpOrder()
    camera_offset.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(-0.013, 0.0, 0.0))

    UsdGeom.Xform.Define(stage, camera_frame_path)
    camera_frame = UsdGeom.Xformable(stage.GetPrimAtPath(camera_frame_path))
    if camera_frame.GetOrderedXformOps():
        camera_frame.ClearXformOpOrder()
    camera_frame.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(cam_orient)

    camera = UsdGeom.Camera.Define(stage, camera_path)
    _apply_d455_fallback_params_to_usd_camera(camera)

    return rig_path, camera_path, body_path


def _open_camera_viewport_window(camera_prim_path: str) -> None:
    """Open a viewport window and set its active camera."""
    try:
        from omni.kit.viewport.utility import (  # type: ignore
            create_viewport_window,
            get_active_viewport,
        )
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


def _apply_hand_drive_gains(*, root_prim_path: str) -> None:
    """Apply PhysX drive stiffness/damping to hand joints under the imported robot.

    This edits USD/PhysX schema attributes directly (no articulation view required).
    """
    try:
        import omni.usd  # type: ignore
        from pxr import PhysxSchema, UsdPhysics  # type: ignore
    except Exception as e:
        print(f"[warn] PhysX schema not available; skip hand gains: {e}", flush=True)
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[warn] USD stage not available; skip hand gains.", flush=True)
        return

    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        print(f"[warn] Robot prim not found for gains: {root_prim_path}", flush=True)
        return
    root_prefix = str(root_prim.GetPath())
    if not root_prefix.endswith("/"):
        root_prefix = root_prefix + "/"

    def _pick_gain(joint_name: str) -> tuple[float, float] | None:
        if joint_name in ("thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_mcp", "thumb_ip"):
            return GAINS[joint_name]
        if joint_name.endswith("_mcp_roll"):
            return GAINS["mcp_roll"]
        if joint_name.endswith("_mcp_pitch"):
            return GAINS["mcp_pitch"]
        if joint_name.endswith("_pip"):
            return GAINS["pip"]
        if joint_name.endswith("_dip"):
            return GAINS["dip"]
        return None

    applied = 0
    skipped = 0
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path != str(root_prim.GetPath()) and not prim_path.startswith(root_prefix):
            continue
        name = prim.GetName()
        gain = _pick_gain(name)
        if gain is None:
            continue

        k, d = gain
        try:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateStiffnessAttr(float(k))
            drive.CreateDampingAttr(float(d))
            drive.CreateMaxForceAttr(1.0e6)
            # Ensure the attribute exists; targets may be overwritten per-tick for mimic joints.
            drive.CreateTargetPositionAttr(0.0)

            physx_drive = PhysxSchema.PhysxDriveAPI.Apply(prim, "angular")
            physx_drive.CreateStiffnessAttr(float(k))
            physx_drive.CreateDampingAttr(float(d))
            physx_drive.CreateMaxForceAttr(1.0e6)
            applied += 1
        except Exception:
            skipped += 1

    print(f"[hand_gains] applied={applied}, skipped={skipped}, root={root_prim_path}", flush=True)


def _collect_named_prims_under(*, root_prim_path: str, names: set[str]) -> dict[str, object]:
    """Collect first prim for each name under the given root path."""
    import omni.usd  # type: ignore

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return {}

    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        return {}

    root_prefix = str(root.GetPath())
    if not root_prefix.endswith("/"):
        root_prefix = root_prefix + "/"

    found: dict[str, object] = {}
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path != str(root.GetPath()) and not prim_path.startswith(root_prefix):
            continue
        n = prim.GetName()
        if n in names and n not in found:
            found[n] = prim
            if len(found) == len(names):
                break
    return found


def _update_hand_mimic_targets(*, root_prim_path: str, prims: dict[str, object]) -> None:
    """Update mimic joints' drive targetPosition each tick from master joints.

    Mimic mapping (your spec):
    - thumb_mcp = 1.3898 * thumb_cmc_pitch
    - thumb_ip  = 1.5080 * thumb_cmc_pitch
    - pip       = 1.3462 * mcp_pitch
    - dip       = 0.4616 * mcp_pitch
    """
    try:
        from pxr import UsdPhysics  # type: ignore
    except Exception:
        return

    def _get_target(name: str) -> float:
        prim = prims.get(name)
        if prim is None:
            return 0.0
        try:
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            attr = drive.GetTargetPositionAttr()
            v = attr.Get()
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    def _set_target(name: str, value: float) -> None:
        prim = prims.get(name)
        if prim is None:
            return
        try:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTargetPositionAttr(0.0)
            drive.GetTargetPositionAttr().Set(float(value))
        except Exception:
            return

    q_thumb_cmc_pitch = _get_target("thumb_cmc_pitch")
    q_index_mcp_pitch = _get_target("index_mcp_pitch")
    q_middle_mcp_pitch = _get_target("middle_mcp_pitch")
    q_ring_mcp_pitch = _get_target("ring_mcp_pitch")
    q_pinky_mcp_pitch = _get_target("pinky_mcp_pitch")

    _set_target("thumb_mcp", 1.3898 * q_thumb_cmc_pitch)
    _set_target("thumb_ip", 1.5080 * q_thumb_cmc_pitch)

    _set_target("index_pip", 1.3462 * q_index_mcp_pitch)
    _set_target("index_dip", 0.4616 * q_index_mcp_pitch)
    _set_target("middle_pip", 1.3462 * q_middle_mcp_pitch)
    _set_target("middle_dip", 0.4616 * q_middle_mcp_pitch)
    _set_target("ring_pip", 1.3462 * q_ring_mcp_pitch)
    _set_target("ring_dip", 0.4616 * q_ring_mcp_pitch)
    _set_target("pinky_pip", 1.3462 * q_pinky_mcp_pitch)
    _set_target("pinky_dip", 0.4616 * q_pinky_mcp_pitch)


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

    # Apply L10 hand joint stiffness/damping.
    _apply_hand_drive_gains(root_prim_path=str(args.prim_path))

    # Cache prim handles for master + mimic joints so we can update mimic targets every tick.
    _HAND_JOINT_NAMES = {
        # masters
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
        # mimics (must be updated every tick)
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
    hand_prims = _collect_named_prims_under(root_prim_path=str(args.prim_path), names=_HAND_JOINT_NAMES)
    if hand_prims:
        missing = sorted(_HAND_JOINT_NAMES.difference(hand_prims.keys()))
        if missing:
            print(f"[hand_mimic] missing prims: {missing}", flush=True)
        else:
            print("[hand_mimic] all required joint prims found.", flush=True)

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
    # Fixed mount pose: translate + rotate around local X (clockwise 10 degrees).
    theta = math.radians(-10.0)
    qx = math.sin(theta * 0.5)
    qw = math.cos(theta * 0.5)
    _set_local_pose(mount_path, pos=(0.0, -0.08, 0.0), quat_xyzw=(qx, 0.0, 0.0, qw))
    camera_path, box_path = _spawn_camera_and_box(mount_path=mount_path, d405=d405)
    print(f"Created hand camera mount: {mount_path} (local Y -0.08m, roll_x -10deg)", flush=True)
    print(f"Spawned camera prim: {camera_path}", flush=True)
    print(f"Spawned camera body box: {box_path} (0.042 x 0.042 x 0.023 m)", flush=True)

    fixed_rig_path, fixed_camera_path, fixed_box_path = _spawn_fixed_world_camera_with_box()
    print(
        f"Spawned fixed world camera rig: {fixed_rig_path} + {fixed_rig_path}/yaw "
        f"(x=0.1, y=-0.3, z=0.6 m, RotateXYZ = 180, 135, 0 deg)",
        flush=True,
    )
    print(f"Spawned fixed world camera prim: {fixed_camera_path} (look dir: -X)", flush=True)
    print(
        f"Spawned fixed world D455 body/visual prim: {fixed_box_path}",
        flush=True,
    )

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
            # Keep mimic joints' targetPosition consistent with masters (no independent control).
            if hand_prims:
                _update_hand_mimic_targets(root_prim_path=str(args.prim_path), prims=hand_prims)
            simulation_app.update()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

