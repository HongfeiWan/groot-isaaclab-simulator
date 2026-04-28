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
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from isaacsim.core.prims import SingleArticulation  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "demo_data" / "l10_hand" / "lerobot_rokae_xmate3_linker_l10_groot_v1"
LOADER_SCRIPT_PATH = REPO_ROOT / "examples" / "IsaacLab" / "load_xmate3_with_right_l10hand_urdf.py"

# Keep a world reference alive for the lifetime of the script.
_WORLD_REF = None
_LOADER_REF = None
_HAND_PRIMS_REF = None


def _parse_args() -> tuple[argparse.Namespace, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--episode-chunk", type=int, default=0)
    parser.add_argument("--fps", type=float, default=0.0, help="0 means use dataset fps from meta/info.json")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")

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


def _load_parquet_states(dataset_dir: Path, *, episode_chunk: int, episode_index: int) -> np.ndarray:
    parquet_path = dataset_dir / "data" / f"chunk-{episode_chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")

    # Backend 1: pyarrow (fast, common in data envs)
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(parquet_path.as_posix(), columns=["observation.state"])
        col = table.column("observation.state")
        arr = col.to_numpy(zero_copy_only=False)
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 20:
            states = arr.astype(np.float32, copy=False)
        else:
            states = np.asarray(list(arr), dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 20:
            raise RuntimeError(f"Unexpected observation.state shape: {states.shape}")
        return states
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
            "SELECT observation.state AS s FROM read_parquet(?)",
            [parquet_path.as_posix()],
        ).fetchall()
        states = np.asarray([r[0] for r in rows], dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 20:
            raise RuntimeError(f"Unexpected observation.state shape: {states.shape}")
        return states
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


def _init_robot_scene(*, prim_path: str) -> "SingleArticulation":
    # Reuse the existing loader script to build the stage (URDF import + camera mount + viewport window).
    global _LOADER_REF, _HAND_PRIMS_REF
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

    # Apply the same hand drive stiffness/damping as the loader script.
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
        loader._set_local_pose(mount_path, pos=(0.0, -0.08, 0.0), quat_xyzw=(qx, 0.0, 0.0, qw))  # noqa: SLF001
        d405 = None
        try:
            d405 = loader._load_d405_json(str(loader.D405_JSON_PATH))  # noqa: SLF001
        except Exception:
            d405 = None
        camera_path, _ = loader._spawn_camera_and_box(mount_path=mount_path, d405=d405)  # noqa: SLF001
        if not bool(getattr(loader, "_UI_REFS", [])) and not bool(getattr(loader, "_headless", False)):
            # Bind viewport if possible.
            try:
                loader._open_camera_viewport_window(camera_path)  # noqa: SLF001
            except Exception:
                pass

    # Isaac Sim articulation control (stable in 5.1).
    from isaacsim.core.prims import SingleArticulation  # type: ignore

    robot = SingleArticulation(prim_path=prim_path, name="xMate3_L10Hand")
    # Pass the physics_sim_view explicitly; otherwise initialize() may see None.
    try:
        robot.initialize(physics_sim_view=getattr(_WORLD_REF, "physics_sim_view", None))
    except TypeError:
        robot.initialize()
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


def main() -> None:
    args, simulation_app = _parse_args()

    try:
        dataset_dir = Path(args.dataset_dir).expanduser()
        info = _read_info_json(dataset_dir)
        dataset_fps = float(info.get("fps", 10))
        fps = float(args.fps) if float(args.fps) > 1e-6 else dataset_fps
        fps = max(1e-3, fps) * float(args.speed)

        states = _load_parquet_states(
            dataset_dir, episode_chunk=int(args.episode_chunk), episode_index=int(args.episode_index)
        )
        print(f"Loaded states: T={states.shape[0]}, dim={states.shape[1]}, fps={fps:g}", flush=True)
    except Exception as e:
        # Exit gracefully without a long traceback in the IsaacLab UI.
        print(f"[error] Playback init failed: {e}", flush=True)
        simulation_app.close()
        return

    # Build scene + robot.
    robot = _init_robot_scene(prim_path=str(args.robot_prim_path))
    dof_index = _name_to_dof_index(robot)

    # Joint name mapping.
    arm_joint_names = [f"xMate3_joint_{i}" for i in range(1, 7)]
    hand_joint_names = [
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

    # Resolve indices (skip any missing, but print once).
    missing = [n for n in arm_joint_names + hand_joint_names if n not in dof_index]
    if missing:
        print(f"[warn] Missing joint names in articulation: {missing}", flush=True)

    arm_indices = np.array([dof_index[n] for n in arm_joint_names if n in dof_index], dtype=np.int32)
    hand_indices = np.array([dof_index[n] for n in hand_joint_names if n in dof_index], dtype=np.int32)

    from isaacsim.core.utils.types import ArticulationAction  # type: ignore

    # Playback loop.
    dt_wall = 1.0 / fps
    t = 0
    try:
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

            # Play one frame per wall tick (simple throttle via accumulation).
            # Using Kit time is overkill here; the dataset fps is low (10Hz).
            s = states[t]
            arm_q = s[0:6].astype(np.float32, copy=False)
            hand_q = s[10:20].astype(np.float32, copy=False)

            if arm_indices.size > 0:
                robot.apply_action(ArticulationAction(joint_positions=arm_q[: arm_indices.size], joint_indices=arm_indices))
            if hand_indices.size > 0:
                robot.apply_action(ArticulationAction(joint_positions=hand_q[: hand_indices.size], joint_indices=hand_indices))

            simulation_app.update()

            # Advance frame index based on a simple counter (dataset is 10Hz; update loop is faster).
            # We step by 1 every N sim updates to approximate dt_wall without importing extra timing utils.
            # If you need tighter sync, we can switch to carb.clock / timeline time.
            if getattr(main, "_accum", None) is None:
                main._accum = 0.0  # type: ignore[attr-defined]
            main._accum += 1.0 / 60.0  # type: ignore[attr-defined]
            if main._accum >= dt_wall:  # type: ignore[attr-defined]
                main._accum = 0.0  # type: ignore[attr-defined]
                t += 1
                if t >= states.shape[0]:
                    if bool(args.loop):
                        t = 0
                    else:
                        break
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

