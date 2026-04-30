#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Play back LeRobot v2 dataset on xMate3 + L10 hand in Isaac Sim (IsaacLab runtime).

运行（用 IsaacLab 的 Python）：

    ~/Project/IsaacLab/isaaclab.sh -p examples/IsaacLab/play_lerobot_rokae_xmate3_l10hand.py --enable_cameras

默认会加载：
- 机器人 URDF：由 `examples/IsaacLab/load_xmate3_with_right_l10hand_urdf.py` 创建的组合机器人
- 数据：`demo_data/l10_hand/lerobot_rokae_xmate3_linker_l10_groot_v1`

数据 `observation.state` (dim=20):
  [0:7)   arm_joint_pos (第7维占位，当前永远为0；播放时忽略)
  [7:10)  arm_eef_pos (XYZ, rokae_base, m)  (播放脚本不使用，仅用于调试/对齐)
  [10:20) hand_joint_pos (L10 canonical 10DoF)

数据 `action` (dim=13):
  [0:3)   arm_eef_pos_target (XYZ, rokae_base, m)
  [3:13)  hand_joint_target (L10 canonical 10DoF)

默认回放方式使用 action 里的末端 XYZ target，通过 teleop 脚本里的 IK 实时解出
xMate3 关节目标。由于当前数据集不包含 arm orientation action，默认会用
`observation.state[0:6]` 的 recorded arm joints 经同一 IK/FK 模型反算出末端姿态，
并用 recorded FK 位置对齐数据集 EEF 点和 IK 的 joint6 点，然后与 action XYZ
拼成 4x4 pose 做 solve6；相比直接播放 recorded arm joints，
这样既能保留真实示教的姿态/构型倾向，又能减少关节状态噪声造成的视觉抖动。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:  # pragma: no cover
    from isaacsim.core.prims import SingleArticulation  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "demo_data" / "l10_hand" / "lerobot_rokae_xmate3_linker_l10_groot_v1"
LOADER_SCRIPT_PATH = REPO_ROOT / "examples" / "IsaacLab" / "load_xmate3_with_right_l10hand_urdf.py"
TELEOP_SCRIPT_PATH = REPO_ROOT / "examples" / "IsaacLab" / "teleop_xmate3_l10hand_eef_ik.py"

# Keep a world reference alive for the lifetime of the script.
_WORLD_REF = None
_LOADER_REF = None
_TELEOP_REF = None
_HAND_PRIMS_REF = None
_CAMERA_PRIMS_REF = {}


def _parse_args() -> tuple[argparse.Namespace, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--episode-chunk", type=int, default=0)
    parser.add_argument("--fps", type=float, default=0.0, help="0 means use dataset fps from meta/info.json")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument(
        "--playback-clock",
        choices=("wall", "update60"),
        default="wall",
        help=(
            "wall advances dataset frames by real elapsed time, matching the source video duration. "
            "update60 advances as if every Isaac update were exactly 1/60s."
        ),
    )
    parser.add_argument(
        "--arm-source",
        choices=("action_ik", "state_joint", "state_eef_ik"),
        default="action_ik",
        help=(
            "Arm playback source. action_ik uses action[0:3] EEF target + IK; "
            "state_joint replays recorded state[0:6]; state_eef_ik uses state[7:10] + IK."
        ),
    )
    parser.add_argument(
        "--hand-source",
        choices=("action", "state"),
        default="action",
        help="Hand playback source. action uses action[3:13] target; state uses state[10:20].",
    )
    parser.add_argument("--max-joint-step", type=float, default=1.0)
    parser.add_argument("--psi", type=float, default=0.0)
    parser.add_argument(
        "--target-rotation-source",
        choices=("state_fk", "initial_fk", "current_fk"),
        default="state_fk",
        help=(
            "Rotation used for IK pose. state_fk derives each frame's EEF rotation from "
            "recorded state[0:6] joints; initial_fk keeps the first-frame rotation; "
            "current_fk keeps the simulated arm's current rotation."
        ),
    )
    parser.add_argument(
        "--target-position-frame",
        choices=("state_fk_aligned", "dataset_eef"),
        default="state_fk_aligned",
        help=(
            "Frame used for IK target position. state_fk_aligned treats action/state EEF XYZ "
            "as the dataset TCP/wrist point and shifts it into the IK FK frame using "
            "fk(recorded_state_q).xyz - observation.state[7:10]. dataset_eef sends XYZ directly."
        ),
    )
    parser.add_argument(
        "--ik-seed-source",
        choices=("state_joint", "current", "blend"),
        default="state_joint",
        help=(
            "Seed used by the IK solver. state_joint selects the branch near the recorded "
            "demonstration posture; current maximizes continuity; blend mixes both."
        ),
    )
    parser.add_argument(
        "--ik-seed-state-weight",
        type=float,
        default=0.35,
        help="When --ik-seed-source=blend, weight for recorded state joints in the IK seed.",
    )

    parser.add_argument(
        "--ik-backend",
        choices=("auto", "rci", "urdf"),
        default="auto",
        help="IK backend reused from teleop_xmate3_l10hand_eef_ik.py.",
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

    # Robot loader args (we forward a subset by simply reusing defaults).
    parser.add_argument("--robot-prim-path", type=str, default="/World/xMate3_L10Hand")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    return args, app


def _read_info_json(dataset_dir: Path) -> dict:
    import json

    p = dataset_dir / "meta" / "info.json"
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _column_to_float32_matrix(values, *, expected_dim: int, name: str) -> np.ndarray:
    if hasattr(values, "to_pylist"):
        out = np.asarray(values.to_pylist(), dtype=np.float32)
        if out.ndim != 2 or out.shape[1] != expected_dim:
            raise RuntimeError(f"Unexpected {name} shape: {out.shape}, expected (*, {expected_dim})")
        return out

    arr = values.to_numpy(zero_copy_only=False) if hasattr(values, "to_numpy") else values
    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == expected_dim:
        out = arr.astype(np.float32, copy=False)
    else:
        out = np.asarray(list(arr), dtype=np.float32)
    if out.ndim != 2 or out.shape[1] != expected_dim:
        raise RuntimeError(f"Unexpected {name} shape: {out.shape}, expected (*, {expected_dim})")
    return out


def _load_parquet_episode(
    dataset_dir: Path, *, episode_chunk: int, episode_index: int
) -> tuple[np.ndarray, np.ndarray]:
    parquet_path = dataset_dir / "data" / f"chunk-{episode_chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")

    # Backend 1: pyarrow (fast, common in data envs)
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(parquet_path.as_posix(), columns=["observation.state", "action"])
        states = _column_to_float32_matrix(table.column("observation.state"), expected_dim=20, name="observation.state")
        actions = _column_to_float32_matrix(table.column("action"), expected_dim=13, name="action")
        return states, actions
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet with pyarrow: {e}") from e

    # Backend 2: duckdb (often available even when pyarrow isn't)
    try:
        import duckdb  # type: ignore

        con = duckdb.connect(database=":memory:")
        # DuckDB returns a column of LIST<FLOAT> for this feature.
        rows = con.execute(
            'SELECT "observation.state" AS s, action AS a FROM read_parquet(?)',
            [parquet_path.as_posix()],
        ).fetchall()
        states = np.asarray([r[0] for r in rows], dtype=np.float32)
        actions = np.asarray([r[1] for r in rows], dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 20:
            raise RuntimeError(f"Unexpected observation.state shape: {states.shape}")
        if actions.ndim != 2 or actions.shape[1] != 13:
            raise RuntimeError(f"Unexpected action shape: {actions.shape}")
        return states, actions
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet with duckdb: {e}") from e

    # No backend available: fail with a clear message.
    raise RuntimeError(
        "无法读取 .parquet：当前 IsaacLab Python 环境里没有可用的 parquet 读取后端。"
        "请安装 `pyarrow`（推荐）或 `duckdb`，然后重试。"
    )


def _import_loader_module():
    """Import the robot loader script by file path (no package needed)."""
    import importlib.util

    if not LOADER_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Loader script not found: {LOADER_SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location(
        "load_xmate3_with_right_l10hand_urdf",
        LOADER_SCRIPT_PATH.as_posix(),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {LOADER_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_teleop_module():
    """Import the EEF IK teleop helpers by file path."""
    global _TELEOP_REF
    if _TELEOP_REF is not None:
        return _TELEOP_REF

    import importlib.util

    if not TELEOP_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Teleop script not found: {TELEOP_SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location(
        "teleop_xmate3_l10hand_eef_ik",
        TELEOP_SCRIPT_PATH.as_posix(),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {TELEOP_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _TELEOP_REF = module
    return module


def _spawn_playback_cameras(loader, *, prim_path: str) -> None:
    """Create the same camera rigs as the loader after articulation initialization."""
    global _CAMERA_PRIMS_REF

    # Create the hand camera as in the loader (using its defaults).
    attach_prim = loader._find_prim_by_name("xMate3_link6")  # noqa: SLF001
    if not attach_prim:
        for candidate in ("base_link", "xMate3_base_link", "world", "xMate3_link0"):
            attach_prim = loader._find_prim_by_name(candidate)  # noqa: SLF001
            if attach_prim:
                break
    if attach_prim:
        mount_path = f"{attach_prim}/hand_camera_mount"
        loader._ensure_xform(mount_path)  # noqa: SLF001
        # Keep mount pose consistent with the loader's current fixed pose.
        theta = loader.math.radians(-10.0)  # noqa: SLF001
        qx = loader.math.sin(theta * 0.5)  # noqa: SLF001
        qw = loader.math.cos(theta * 0.5)  # noqa: SLF001
        loader._set_local_pose(  # noqa: SLF001
            mount_path,
            pos=(0.0, -0.08, 0.0),
            quat_xyzw=(qx, 0.0, 0.0, qw),
        )
        d405 = None
        try:
            d405 = loader._load_d405_json(str(loader.D405_JSON_PATH))  # noqa: SLF001
        except Exception:
            d405 = None
        camera_path, _ = loader._spawn_camera_and_box(mount_path=mount_path, d405=d405)  # noqa: SLF001
        _CAMERA_PRIMS_REF["hand_camera"] = camera_path
        print(f"[playback_camera] hand camera prim: {camera_path}", flush=True)
    else:
        print(f"[warn] playback camera attach link not found under {prim_path}", flush=True)

    # Create the fixed D455/head-style camera as in the loader. This is the view that matches
    # the dataset's realsense_head -> ego_view convention.
    try:
        fixed_rig_path, fixed_camera_path, fixed_box_path = (
            loader._spawn_fixed_world_camera_with_box()  # noqa: SLF001
        )
        _CAMERA_PRIMS_REF["fixed_camera"] = fixed_camera_path
        print(
            f"[playback_camera] fixed camera rig: {fixed_rig_path}; "
            f"camera prim: {fixed_camera_path}; body prim: {fixed_box_path}",
            flush=True,
        )
        if not bool(getattr(loader, "_UI_REFS", [])) and not bool(getattr(loader, "_headless", False)):
            try:
                loader._open_camera_viewport_window(  # noqa: SLF001
                    fixed_camera_path,
                    title="Fixed D455 Camera",
                )
            except TypeError:
                loader._open_camera_viewport_window(fixed_camera_path)  # noqa: SLF001
    except Exception as e:
        print(f"[warn] failed to spawn fixed camera in playback: {e}", flush=True)

    try:
        import omni.usd  # type: ignore
        from pxr import UsdGeom  # type: ignore

        stage = omni.usd.get_context().get_stage()
        camera_prims = [
            str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)
        ]
        print(f"[playback_camera] stage camera prims: {camera_prims}", flush=True)
    except Exception as e:
        print(f"[warn] failed to list playback cameras: {e}", flush=True)


def _init_robot_scene(*, prim_path: str) -> "SingleArticulation":
    # Reuse the existing loader script to build the stage and camera rigs.
    global _LOADER_REF, _HAND_PRIMS_REF, _CAMERA_PRIMS_REF
    loader = _import_loader_module()
    _LOADER_REF = loader

    # Create a simulation World so physics views are available (Isaac Sim 5.1 requirement).
    global _WORLD_REF
    if _WORLD_REF is None:
        try:
            from isaacsim.core.api.world import World  # type: ignore
        except Exception:
            from isaacsim.core.world import World  # type: ignore

        _WORLD_REF = World(stage_units_in_meters=1.0)
        _WORLD_REF.reset()

    loader._add_ground_plane()  # noqa: SLF001
    try:
        loader._add_scene_light()  # type: ignore[attr-defined]  # noqa: SLF001
    except Exception:
        pass
    loader._import_urdf(  # noqa: SLF001
        urdf_path=str(loader.COMBINED_URDF_PATH),
        prim_path=prim_path,
        fix_base=True,
        make_instanceable=False,
    )

    # Ensure PhysX has built views for newly spawned prims.
    try:
        _WORLD_REF.reset()
    except Exception:
        pass

    # Apply the same stiff arm and hand drive gains as the loader script.
    try:
        loader._apply_arm_drive_gains(root_prim_path=str(prim_path))  # type: ignore[attr-defined]  # noqa: SLF001
    except Exception as e:
        print(f"[warn] failed to apply arm gains in playback: {e}", flush=True)
    try:
        loader._apply_hand_drive_gains(root_prim_path=str(prim_path))  # type: ignore[attr-defined]  # noqa: SLF001
    except Exception as e:
        print(f"[warn] failed to apply hand gains in playback: {e}", flush=True)

    # Cache the same master+mimic joint prims for per-tick mimic target updates (if available).
    try:
        names = {
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
            # mimics
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
        _HAND_PRIMS_REF = loader._collect_named_prims_under(  # type: ignore[attr-defined]  # noqa: SLF001
            root_prim_path=str(prim_path),
            names=names,
        )
    except Exception:
        _HAND_PRIMS_REF = None

    # Isaac Sim articulation control (stable in 5.1).
    from isaacsim.core.prims import SingleArticulation  # type: ignore

    robot = SingleArticulation(prim_path=prim_path, name="xMate3_L10Hand")
    # Pass the physics_sim_view explicitly; otherwise initialize() may see None.
    try:
        robot.initialize(physics_sim_view=getattr(_WORLD_REF, "physics_sim_view", None))
    except TypeError:
        robot.initialize()

    _spawn_playback_cameras(loader, prim_path=str(prim_path))
    return robot


def _name_to_dof_index(robot: "SingleArticulation") -> dict[str, int]:
    """Return a mapping from DOF name -> DOF index for commands.

    Isaac Sim versions differ in API surface:
    - Some expose `dof_names` + `get_dof_index(name)`
    - Others expose `get_dof_names()`
    """
    if hasattr(robot, "get_dof_index") and callable(getattr(robot, "get_dof_index")):
        # Build mapping from dof_names to be explicit (also validates names exist).
        names = []
        if hasattr(robot, "dof_names"):
            names = list(getattr(robot, "dof_names"))
        elif hasattr(robot, "get_dof_names") and callable(getattr(robot, "get_dof_names")):
            names = list(robot.get_dof_names())  # type: ignore[attr-defined]
        else:
            names = []
        if names:
            return {n: int(robot.get_dof_index(n)) for n in names}  # type: ignore[attr-defined]

    # Fallback: best-effort list of dof names
    if hasattr(robot, "dof_names"):
        names = list(getattr(robot, "dof_names"))
        return {n: i for i, n in enumerate(names)}
    if hasattr(robot, "get_dof_names") and callable(getattr(robot, "get_dof_names")):
        names = list(robot.get_dof_names())  # type: ignore[attr-defined]
        return {n: i for i, n in enumerate(names)}

    raise RuntimeError(
        "无法获取 articulation 的 DOF 名称列表：SingleArticulation API 不包含 dof_names/get_dof_names。"
    )


def _precompute_state_fk_poses(ik, states: np.ndarray, fallback_pose: np.ndarray) -> np.ndarray:
    """Use recorded arm joints as posture information by converting them to FK poses."""
    fallback = np.asarray(fallback_pose, dtype=np.float64).reshape(4, 4)
    poses = np.repeat(fallback[None, :, :], states.shape[0], axis=0)
    failures = 0

    for i, state in enumerate(states):
        q = np.asarray(state[0:6], dtype=np.float64)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            failures += 1
            continue
        try:
            pose = np.asarray(ik.fk6(q), dtype=np.float64).reshape(4, 4)
            if not np.all(np.isfinite(pose)):
                raise RuntimeError("non-finite FK pose")
            poses[i] = pose
        except Exception:
            failures += 1

    dataset_xyz = states[:, 7:10].astype(np.float64, copy=False)
    finite = np.all(np.isfinite(dataset_xyz), axis=1)
    if np.any(finite):
        err = np.linalg.norm(poses[finite, :3, 3] - dataset_xyz[finite], axis=1)
        print(
            f"[state_fk] precomputed={states.shape[0] - failures}/{states.shape[0]}, "
            f"fk_vs_state_eef_pos_mean={float(np.mean(err)):.5f}m, "
            f"max={float(np.max(err)):.5f}m",
            flush=True,
        )
    elif failures:
        print(f"[state_fk] FK failures={failures}/{states.shape[0]}", flush=True)

    return poses


def _select_ik_seed(args: argparse.Namespace, current_q: np.ndarray, recorded_q: np.ndarray) -> np.ndarray:
    current = np.asarray(current_q, dtype=np.float64).reshape(6)
    recorded = np.asarray(recorded_q, dtype=np.float64).reshape(6)
    if not np.all(np.isfinite(recorded)):
        return current

    source = str(args.ik_seed_source)
    if source == "state_joint":
        return recorded
    if source == "blend":
        weight = float(np.clip(float(args.ik_seed_state_weight), 0.0, 1.0))
        return (1.0 - weight) * current + weight * recorded
    return current


def main() -> None:
    args, simulation_app = _parse_args()
    ik = None

    try:
        dataset_dir = Path(args.dataset_dir).expanduser()
        info = _read_info_json(dataset_dir)
        dataset_fps = float(info.get("fps", 10))
        fps = float(args.fps) if float(args.fps) > 1e-6 else dataset_fps
        fps = max(1e-3, fps) * float(args.speed)

        states, actions = _load_parquet_episode(
            dataset_dir, episode_chunk=int(args.episode_chunk), episode_index=int(args.episode_index)
        )
        print(
            f"Loaded episode: T={states.shape[0]}, state_dim={states.shape[1]}, "
            f"action_dim={actions.shape[1]}, fps={fps:g}",
            flush=True,
        )

        teleop = _import_teleop_module()
        pose_from_xyz_rotation = None
        if str(args.arm_source) != "state_joint":
            ik, pose_from_xyz_rotation = teleop._create_ik(args)  # noqa: SLF001
    except Exception as e:
        # Exit gracefully without a long traceback in the IsaacLab UI.
        print(f"[error] Playback init failed: {e}", flush=True)
        try:
            if ik is not None:
                ik.close()
        except Exception:
            pass
        simulation_app.close()
        return

    try:
        # Build scene + robot.
        robot = _init_robot_scene(prim_path=str(args.robot_prim_path))
        dof_index = _name_to_dof_index(robot)

        # Keep teleop's hand helper pointed at the prim cache created by this playback scene.
        try:
            teleop._HAND_PRIMS_REF = _HAND_PRIMS_REF  # noqa: SLF001
            teleop._LOADER_REF = _LOADER_REF  # noqa: SLF001
        except Exception:
            pass

        # Joint name mapping. Prefer teleop's constants so playback and teleop stay aligned.
        arm_joint_names = list(getattr(teleop, "ARM_JOINT_NAMES", [f"xMate3_joint_{i}" for i in range(1, 7)]))
        hand_joint_names = list(
            getattr(
                teleop,
                "HAND_JOINT_NAMES",
                [
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
                ],
            )
        )
        derived_hand_names = list(getattr(teleop, "HAND_DERIVED_JOINT_NAMES", []))

        # Resolve indices (skip any missing, but print once).
        missing = [n for n in arm_joint_names + hand_joint_names if n not in dof_index]
        if missing:
            print(f"[warn] Missing joint names in articulation: {missing}", flush=True)

        arm_indices = np.array([dof_index[n] for n in arm_joint_names if n in dof_index], dtype=np.int32)
        hand_indices = np.array([dof_index[n] for n in hand_joint_names if n in dof_index], dtype=np.int32)
        derived_hand_indices = {name: int(dof_index[name]) for name in derived_hand_names if name in dof_index}

        if str(args.arm_source) != "state_joint" and arm_indices.size != 6:
            raise RuntimeError(
                f"IK playback requires all 6 xMate3 arm joints, got {arm_indices.size}: {arm_joint_names}"
            )

        q_cmd = states[0, 0:6].astype(np.float64, copy=True)
        if q_cmd.shape != (6,) or not np.all(np.isfinite(q_cmd)):
            q_cmd = np.asarray(getattr(teleop, "DEFAULT_ARM_Q"), dtype=np.float64).copy()
        if arm_indices.size > 0:
            teleop._apply_joint_positions(robot, arm_indices, q_cmd[: arm_indices.size])  # noqa: SLF001

        initial_hand = actions[0, 3:13] if str(args.hand_source) == "action" else states[0, 10:20]
        if hand_indices.size > 0 and np.all(np.isfinite(initial_hand)):
            teleop._apply_l10_hand_positions(  # noqa: SLF001
                robot,
                hand_indices,
                derived_hand_indices,
                initial_hand,
            )

        for _ in range(3):
            simulation_app.update()

        target_rotation = None
        initial_rotation = None
        state_fk_poses = None
        if str(args.arm_source) != "state_joint":
            assert ik is not None
            current_pose = ik.fk6(q_cmd)
            initial_rotation = current_pose[:3, :3].copy()
            target_rotation = initial_rotation
            if (
                str(args.target_rotation_source) == "state_fk"
                or str(args.target_position_frame) == "state_fk_aligned"
            ):
                state_fk_poses = _precompute_state_fk_poses(ik, states, current_pose)

        print(
            f"[playback] arm_source={args.arm_source}, hand_source={args.hand_source}, "
            f"position_frame={args.target_position_frame}, "
            f"rotation_source={args.target_rotation_source}, seed_source={args.ik_seed_source}, "
            f"max_joint_step={float(args.max_joint_step):g}",
            flush=True,
        )

        # Playback loop.
        dt_wall = 1.0 / fps
        render_dt = 1.0 / 60.0
        accum = 0.0
        t = 0
        fail_count = 0
        last_frame_time = time.perf_counter()

        while simulation_app.is_running():
            # Keep mimic joints' targetPosition consistent with masters (if loader helpers exist).
            if _LOADER_REF is not None and _HAND_PRIMS_REF:
                try:
                    _LOADER_REF._update_hand_mimic_targets(  # type: ignore[attr-defined]
                        root_prim_path=str(args.robot_prim_path),
                        prims=_HAND_PRIMS_REF,
                    )
                except Exception:
                    pass

            s = states[t]
            a = actions[t]

            if str(args.arm_source) == "state_joint":
                arm_q = s[0:6].astype(np.float64, copy=False)
                if arm_indices.size > 0 and np.all(np.isfinite(arm_q)):
                    q_cmd = arm_q.copy()
                    teleop._apply_joint_positions(  # noqa: SLF001
                        robot, arm_indices, q_cmd[: arm_indices.size]
                    )
            else:
                target_xyz = (
                    a[0:3].astype(np.float64, copy=False)
                    if str(args.arm_source) == "action_ik"
                    else s[7:10].astype(np.float64, copy=False)
                )
                if np.all(np.isfinite(target_xyz)):
                    if (
                        str(args.target_position_frame) == "state_fk_aligned"
                        and state_fk_poses is not None
                    ):
                        state_eef_xyz = s[7:10].astype(np.float64, copy=False)
                        if np.all(np.isfinite(state_eef_xyz)):
                            target_xyz = target_xyz + (state_fk_poses[t, :3, 3] - state_eef_xyz)

                    recorded_arm_q = s[0:6].astype(np.float64, copy=False)
                    current_q = teleop._current_arm_q(robot, arm_indices, q_cmd)  # noqa: SLF001
                    q_seed = _select_ik_seed(args, current_q, recorded_arm_q)
                    try:
                        assert ik is not None
                        assert pose_from_xyz_rotation is not None
                        assert initial_rotation is not None

                        if str(args.target_rotation_source) == "state_fk" and state_fk_poses is not None:
                            target_rotation = state_fk_poses[t, :3, :3]
                        elif str(args.target_rotation_source) == "current_fk":
                            target_rotation = ik.fk6(current_q)[:3, :3]
                        else:
                            target_rotation = initial_rotation

                        target_pose = pose_from_xyz_rotation(target_xyz, target_rotation)
                        q_target = ik.solve6(target_pose, q_seed, psi=float(args.psi))
                        if not np.all(np.isfinite(q_target)):
                            raise RuntimeError(f"non-finite IK result: {q_target}")
                        # q_seed selects the desired IK branch; the actual command is rate-limited
                        # from the current simulated joint position to avoid replaying recorded jitter.
                        q_cmd = current_q + np.clip(
                            q_target - current_q,
                            -float(args.max_joint_step),
                            float(args.max_joint_step),
                        )
                        teleop._apply_joint_positions(robot, arm_indices, q_cmd)  # noqa: SLF001
                        fail_count = 0
                    except Exception as e:
                        fail_count += 1
                        if fail_count == 1 or fail_count % 30 == 0:
                            print(
                                f"[warn] IK failed at frame {t}, target={target_xyz.tolist()}: {e}",
                                flush=True,
                            )

            hand_q = (
                a[3:13].astype(np.float64, copy=False)
                if str(args.hand_source) == "action"
                else s[10:20].astype(np.float64, copy=False)
            )
            if hand_indices.size > 0 and np.all(np.isfinite(hand_q)):
                teleop._apply_l10_hand_positions(  # noqa: SLF001
                    robot,
                    hand_indices,
                    derived_hand_indices,
                    hand_q,
                )

            if _LOADER_REF is not None:
                try:
                    _LOADER_REF._pin_fixed_camera_rig(quiet=True)  # type: ignore[attr-defined]  # noqa: SLF001
                except Exception:
                    pass

            simulation_app.update()

            if str(args.playback_clock) == "wall":
                now = time.perf_counter()
                # Cap very long stalls so a pause/window drag does not jump across the whole episode.
                accum += min(max(now - last_frame_time, 0.0), 0.5)
                last_frame_time = now
            else:
                accum += render_dt

            while accum >= dt_wall:
                accum -= dt_wall
                t += 1
                if t >= states.shape[0]:
                    if bool(args.loop):
                        t = 0
                        q_cmd = states[0, 0:6].astype(np.float64, copy=True)
                    else:
                        break
            if not bool(args.loop) and t >= states.shape[0]:
                break
    finally:
        try:
            if ik is not None:
                ik.close()
        except Exception:
            pass
        simulation_app.close()


if __name__ == "__main__":
    main()
