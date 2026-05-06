#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Rokae-L10-Bench v0 closed-loop bottle grasp evaluation in Isaac Lab.

This script combines:

- the Isaac Sim xMate3 + right L10 hand scene/control path from
  gr00t_xmate3_l10hand_cube_grasp.py
- the local latest-checkpoint GR00T policy loading path from
  compare_l10_gr00t_zero_shot_actions.py

Run with Isaac Lab's Python:

    ~/Project/IsaacLab/isaaclab.sh -p examples/IsaacLab/evaluate_rokae_l10_bench_v0.py \
        --enable_cameras \
        --model-dir checkpoints/rokae_xmate3_l10_overfit \
        --dataset-dir outputs/IsaacLab/trimmed_l10_dataset \
        --num-trials 5

The policy observation uses a black wrist camera by default, matching runs where
the wrist view was black during training.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import importlib.util
import json
import logging
from pathlib import Path
import re
import time
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
GRASP_SCRIPT = REPO_ROOT / "examples" / "IsaacLab" / "gr00t_xmate3_l10hand_cube_grasp.py"
COMPARE_SCRIPT = REPO_ROOT / "examples" / "IsaacLab" / "compare_l10_gr00t_zero_shot_actions.py"
DEFAULT_DATASET_DIR = REPO_ROOT / "outputs" / "IsaacLab" / "trimmed_l10_dataset"
DEFAULT_MODEL_DIR = REPO_ROOT / "checkpoints" / "rokae_xmate3_l10_overfit"
DEFAULT_CONFIG_PATH = REPO_ROOT / "examples" / "IsaacLab" / "rokae_xmate3_l10_modality_config.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "IsaacLab" / "rokae_l10_bench_v0"
BENCH_NAME = "Rokae-L10-Bench-v0"
POLICY_REQUIRED_MODULES = {
    "cv2": "opencv-python-headless>=4.5,<4.13",
    "diffusers": "diffusers==0.35.1",
    "pandas": "pandas==2.2.3",
    "PIL": "Pillow",
    "scipy": "scipy==1.15.3",
    "torch": "torch==2.7.1",
    "torchvision": "torchvision==0.22.1",
    "transformers": "transformers==4.57.3",
}


def _import_module_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Module file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path.as_posix())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_grasp_module():
    return _import_module_from_path("gr00t_xmate3_l10hand_cube_grasp", GRASP_SCRIPT)


def _load_compare_module():
    return _import_module_from_path("compare_l10_gr00t_zero_shot_actions", COMPARE_SCRIPT)


def _check_policy_dependencies() -> None:
    missing = [
        (module_name, package_spec)
        for module_name, package_spec in POLICY_REQUIRED_MODULES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if not missing:
        return

    missing_modules = ", ".join(module_name for module_name, _ in missing)
    install_specs = " ".join(package_spec for _, package_spec in missing)
    raise RuntimeError(
        "Missing Python modules required for local GR00T checkpoint inference in this "
        f"IsaacLab environment: {missing_modules}. Install the missing packages in the "
        f"currently active conda env with:\n\npython -m pip install {install_specs}\n"
    )


def _checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else None


def _latest_checkpoint(model_dir: Path) -> Path:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    candidates: list[tuple[int, Path]] = []
    for child in model_dir.iterdir():
        if not child.is_dir():
            continue
        step = _checkpoint_step(child)
        if step is None:
            continue
        has_weights = (child / "model.safetensors.index.json").exists() or (
            child / "model.safetensors"
        ).exists()
        if has_weights:
            candidates.append((step, child))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-<step> model found under {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def _parse_position_list(text: str | None, grasp: Any) -> list[np.ndarray]:
    if not text:
        return []
    positions = []
    for item in text.split(";"):
        item = item.strip()
        if item:
            positions.append(grasp._parse_vector(item, 3, name="--cube-positions"))  # noqa: SLF001
    return positions


def _trial_cube_position(args: argparse.Namespace, trial_index: int) -> np.ndarray:
    explicit_positions = list(getattr(args, "cube_positions", []) or [])
    if explicit_positions:
        return np.asarray(
            explicit_positions[trial_index % len(explicit_positions)], dtype=np.float64
        )

    base = np.asarray(
        getattr(args, "base_cube_position", args.cube_position), dtype=np.float64
    ).reshape(3)
    jitter_xy = float(args.cube_jitter_xy)
    if jitter_xy <= 0.0:
        return base.copy()

    rng = np.random.default_rng(int(args.trial_seed) + int(trial_index))
    out = base.copy()
    out[:2] += rng.uniform(-jitter_xy, jitter_xy, size=2)
    return out


class BlackCameraReader:
    """CameraReader-compatible source that always returns black frames."""

    def __init__(self, *, image_size: tuple[int, int], name: str = "black_wrist"):
        self.image_size = image_size
        self.name = name

    def read(self) -> np.ndarray:
        return np.zeros((*self.image_size, 3), dtype=np.uint8)


def _reset_target_pose(
    prim_path: str, position: np.ndarray, *, target_object: str, size: float
) -> None:
    import omni.usd  # type: ignore
    from pxr import Gf, UsdGeom, UsdPhysics  # type: ignore

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available.")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Target prim not found: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    if xform.GetOrderedXformOps():
        xform.ClearXformOpOrder()
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    )
    if target_object == "cube":
        xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(float(size), float(size), float(size))
        )

    rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_body.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    rigid_body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))


def _reset_table_pose(*, center_xy: np.ndarray, top_z: float, table_xy_size: float) -> None:
    import omni.usd  # type: ignore
    from pxr import Gf, UsdGeom  # type: ignore

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available.")

    table_path = "/World/grasp_table"
    prim = stage.GetPrimAtPath(table_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Table prim not found: {table_path}")

    table_height = 0.04
    xform = UsdGeom.Xformable(prim)
    if xform.GetOrderedXformOps():
        xform.ClearXformOpOrder()
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), float(top_z) - table_height * 0.5)
    )
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(table_xy_size), float(table_xy_size), table_height)
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_policy(args: argparse.Namespace, compare: Any):
    from gr00t.data.embodiment_tags import EmbodimentTag

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    model_path = (
        Path(args.model_path).expanduser().resolve()
        if args.model_path
        else _latest_checkpoint(Path(args.model_dir).expanduser().resolve())
    )
    config_path = Path(args.modality_config_path).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    modality_config = compare._load_rokae_modality_config(config_path)  # noqa: SLF001
    statistics_episode_indices = (
        list(args.statistics_episode_indices)
        if args.statistics_episode_indices is not None
        else compare._all_episode_indices(dataset_dir)  # noqa: SLF001
    )
    statistics = compare._build_dataset_statistics(  # noqa: SLF001
        dataset_dir,
        modality_config,
        episode_indices=statistics_episode_indices,
        embodiment_value=EmbodimentTag.NEW_EMBODIMENT.value,
    )
    device = compare._resolve_device(str(args.policy_device))  # noqa: SLF001
    logging.info("bench=%s", BENCH_NAME)
    logging.info("model_path=%s", model_path)
    logging.info("dataset_dir=%s", dataset_dir)
    logging.info("statistics_episode_indices=%s", statistics_episode_indices)
    logging.info("device=%s", device)

    policy = compare.InjectedRokaePolicy(
        model_path=model_path,
        modality_config=deepcopy(modality_config),
        statistics=statistics,
        device=device,
        strict=bool(args.strict_policy),
    )
    return policy, modality_config, model_path, dataset_dir


def _make_rtc_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        rtc=bool(args.rtc),
        replan_horizon=int(args.replan_horizon),
        rtc_overlap_steps=args.rtc_overlap_steps,
        rtc_frozen_steps=int(args.rtc_frozen_steps),
        rtc_ramp_rate=float(args.rtc_ramp_rate),
    )


def _run_trial(
    *,
    trial_index: int,
    args: argparse.Namespace,
    simulation_app: Any,
    grasp: Any,
    compare: Any,
    policy: Any,
    modality_config: dict[str, Any],
    robot: Any,
    dof_index: dict[str, int],
    exterior_reader: Any,
    wrist_reader: Any,
    ik: Any,
    pose_from_xyz_rotation: Any,
    marker_path: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    teleop = grasp._load_teleop_module()  # noqa: SLF001
    arm_indices = np.asarray([dof_index[n] for n in teleop.ARM_JOINT_NAMES], dtype=np.int32)
    hand_indices = np.asarray([dof_index[n] for n in teleop.HAND_JOINT_NAMES], dtype=np.int32)

    cube_position = _trial_cube_position(args, trial_index)
    args.cube_position = cube_position
    if getattr(args, "table_top_z_auto", False):
        args.table_top_z = grasp._infer_table_top_z(  # noqa: SLF001
            cube_position, float(args.cube_size)
        )
    if bool(getattr(args, "spawn_table", True)):
        _reset_table_pose(
            center_xy=np.asarray(cube_position[:2], dtype=np.float64),
            top_z=float(args.table_top_z),
            table_xy_size=float(args.table_xy_size),
        )
    target_object = str(getattr(args, "target_object", "cube"))
    _reset_target_pose(
        str(args.cube_prim_path),
        cube_position,
        target_object=target_object,
        size=float(args.cube_size),
    )

    executor = grasp.XmateL10ActionExecutor(
        robot=robot,
        ik=ik,
        pose_from_xyz_rotation=pose_from_xyz_rotation,
        arm_indices=arm_indices,
        hand_indices=hand_indices,
        args=args,
    )
    grasp._settle_initial_pose(executor, args, simulation_app)  # noqa: SLF001

    image_size = (int(args.image_height), int(args.image_width))
    obs_builder = grasp.Gr00tObservationBuilder(
        modality_config=modality_config,
        exterior_reader=exterior_reader,
        wrist_reader=wrist_reader,
        instruction=str(args.instruction),
        image_size=image_size,
    )

    try:
        policy.reset()
    except Exception:
        pass

    grasp._set_target_marker(marker_path, executor.target_xyz)  # noqa: SLF001
    for warmup_step in range(max(0, int(args.camera_warmup_frames))):
        grasp._update_loader_helpers(args)  # noqa: SLF001
        grasp._step_simulation(simulation_app, use_world_step=bool(args.world_step))  # noqa: SLF001
        if warmup_step == 0:
            obs_builder.append_frame()

    action_horizon = len(modality_config["action"].delta_indices)
    rtc_args = _make_rtc_args(args)
    action_chunk: dict[str, np.ndarray] | None = None
    action_index = 0
    previous_action: dict[str, np.ndarray] | None = None
    inference_times: list[float] = []
    trace_rows: list[dict[str, Any]] = []
    success_streak = 0
    success = False
    first_success_step: int | None = None
    max_cube_z = float(cube_position[2])
    command_dt = 1.0 / max(float(args.command_hz), 1e-6)
    last_command_time = 0.0
    start_time = time.perf_counter()
    step = 0

    print(
        f"[trial {trial_index}] target_object={target_object} "
        f"target_position={cube_position.tolist()} "
        f"black_wrist={bool(args.black_wrist)}",
        flush=True,
    )

    while simulation_app.is_running() and step < int(args.max_steps):
        grasp._update_loader_helpers(args)  # noqa: SLF001
        now = time.perf_counter()
        if bool(args.policy_every_step) or now - last_command_time >= command_dt:
            last_command_time = now
            need_replan = action_chunk is None or action_index >= int(args.replan_horizon)

            if need_replan:
                obs_builder.append_frame()
                observation = obs_builder.build(
                    arm_q=executor.current_arm_q(),
                    eef_pose=executor.current_eef_pose(),
                    hand_q=executor.hand_q,
                    hand_alpha=executor.hand_alpha,
                )
                if step == 0 or step % int(args.print_every) == 0:
                    grasp._print_observation_summary(observation, step)  # noqa: SLF001

                rtc_options = compare._rtc_options(  # noqa: SLF001
                    args=rtc_args,
                    action_horizon=action_horizon,
                    previous_action=previous_action,
                )
                request_start = time.perf_counter()
                action_chunk, _ = policy.get_action(
                    observation,
                    options=rtc_options,
                    previous_action=previous_action if rtc_options is not None else None,
                )
                inference_s = time.perf_counter() - request_start
                inference_times.append(inference_s)
                previous_action = compare._unbatch_action_dict(action_chunk)  # noqa: SLF001
                action_index = 0
                print(
                    f"[trial {trial_index} step {step}] action in {inference_s:.2f}s "
                    f"rtc={'off' if rtc_options is None else 'on'}",
                    flush=True,
                )
                if step == 0 or step % int(args.print_every) == 0:
                    grasp._print_action_summary(action_chunk, step)  # noqa: SLF001

            assert action_chunk is not None
            executor.step_action(action_chunk, action_index)
            grasp._set_target_marker(marker_path, executor.target_xyz)  # noqa: SLF001
            action_index += 1
            step += 1

            cube_pos = grasp._get_prim_world_position(str(args.cube_prim_path))  # noqa: SLF001
            if cube_pos is not None:
                max_cube_z = max(max_cube_z, float(cube_pos[2]))
                lifted = float(cube_pos[2] - cube_position[2])
                if lifted >= float(args.success_lift_height):
                    success_streak += 1
                else:
                    success_streak = 0
                if success_streak >= int(args.success_consecutive_frames):
                    success = True
                    if first_success_step is None:
                        first_success_step = step
            else:
                lifted = 0.0

            if bool(args.save_trace):
                current_eef = executor.current_eef_pose()[:3, 3]
                trace_rows.append(
                    {
                        "trial_index": trial_index,
                        "step": step,
                        "eef_x": float(current_eef[0]),
                        "eef_y": float(current_eef[1]),
                        "eef_z": float(current_eef[2]),
                        "target_x": float(executor.target_xyz[0]),
                        "target_y": float(executor.target_xyz[1]),
                        "target_z": float(executor.target_xyz[2]),
                        "hand_alpha": float(executor.hand_alpha),
                        "cube_x": None if cube_pos is None else float(cube_pos[0]),
                        "cube_y": None if cube_pos is None else float(cube_pos[1]),
                        "cube_z": None if cube_pos is None else float(cube_pos[2]),
                        "cube_lift_m": lifted,
                        "target_object": target_object,
                        "success": int(success),
                    }
                )

            if step % int(args.print_every) == 0:
                cube_text = cube_pos.tolist() if cube_pos is not None else None
                print(
                    f"[trial {trial_index} step {step}] "
                    f"target_xyz={executor.target_xyz.tolist()} "
                    f"hand_alpha={executor.hand_alpha:.3f} "
                    f"object_pos={cube_text} lift={lifted:.3f} "
                    f"success_streak={success_streak}",
                    flush=True,
                )

            if success and bool(args.terminate_on_success):
                print(f"[trial {trial_index}] success at step {step}", flush=True)
                break

        grasp._step_simulation(simulation_app, use_world_step=bool(args.world_step))  # noqa: SLF001

    elapsed_s = time.perf_counter() - start_time
    final_cube_pos = grasp._get_prim_world_position(str(args.cube_prim_path))  # noqa: SLF001
    metric = {
        "bench": BENCH_NAME,
        "trial_index": int(trial_index),
        "target_object": target_object,
        "success": bool(success),
        "first_success_step": first_success_step,
        "steps": int(step),
        "elapsed_s": float(elapsed_s),
        "cube_start_x": float(cube_position[0]),
        "cube_start_y": float(cube_position[1]),
        "cube_start_z": float(cube_position[2]),
        "cube_final_x": None if final_cube_pos is None else float(final_cube_pos[0]),
        "cube_final_y": None if final_cube_pos is None else float(final_cube_pos[1]),
        "cube_final_z": None if final_cube_pos is None else float(final_cube_pos[2]),
        "max_cube_z": float(max_cube_z),
        "max_lift_m": float(max_cube_z - cube_position[2]),
        "success_lift_height": float(args.success_lift_height),
        "success_consecutive_frames": int(args.success_consecutive_frames),
        "policy_calls": int(len(inference_times)),
        "avg_inference_s": float(np.mean(inference_times)) if inference_times else 0.0,
        "max_inference_s": float(np.max(inference_times)) if inference_times else 0.0,
        "black_wrist": bool(args.black_wrist),
    }

    trial_prefix = output_dir / f"trial_{trial_index:03d}"
    trial_prefix.with_suffix(".metrics.json").write_text(
        json.dumps(metric, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if bool(args.save_trace):
        _write_csv(trial_prefix.with_suffix(".trace.csv"), trace_rows)
    return metric


def _summarize(
    metrics: list[dict[str, Any]], *, model_path: Path, dataset_dir: Path
) -> dict[str, Any]:
    successes = [bool(item["success"]) for item in metrics]
    return {
        "bench": BENCH_NAME,
        "model_path": str(model_path),
        "dataset_dir": str(dataset_dir),
        "num_trials": len(metrics),
        "num_success": int(sum(successes)),
        "success_rate": float(np.mean(successes)) if successes else None,
        "avg_steps": float(np.mean([m["steps"] for m in metrics])) if metrics else None,
        "avg_max_lift_m": float(np.mean([m["max_lift_m"] for m in metrics])) if metrics else None,
        "trials": metrics,
    }


def _add_args(parser: argparse.ArgumentParser, grasp: Any) -> None:
    grasp._add_args(parser)  # noqa: SLF001
    parser.set_defaults(
        max_steps=1000,
        replan_horizon=8,
        print_every=25,
        target_object="bottle",
        instruction="pick up the bottle and place it in the box",
    )

    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--modality-config-path", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--statistics-episode-indices", type=int, nargs="*", default=None)
    parser.add_argument(
        "--policy-device",
        type=str,
        default="auto",
        help="'auto', 'cuda', 'cuda:0', or 'cpu' for the GR00T policy model.",
    )
    parser.add_argument("--strict-policy", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--trial-seed", type=int, default=0)
    parser.add_argument(
        "--cube-jitter-xy",
        type=float,
        default=0.0,
        help="Uniform +/- XY jitter around --cube-position when --cube-positions is not set.",
    )
    parser.add_argument(
        "--cube-positions",
        type=str,
        default=None,
        help="Semicolon-separated x,y,z cube positions, cycled across trials.",
    )
    parser.add_argument("--success-lift-height", type=float, default=0.08)
    parser.add_argument("--success-consecutive-frames", type=int, default=5)
    parser.add_argument(
        "--terminate-on-success", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save-trace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--black-wrist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Feed black wrist-camera frames into GR00T whenever a wrist/hand video key is present.",
    )

    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rtc-overlap-steps", type=int, default=None)
    parser.add_argument("--rtc-frozen-steps", type=int, default=2)
    parser.add_argument("--rtc-ramp-rate", type=float, default=6.0)
    parser.add_argument("--log-level", type=str, default="INFO")


def _parse_args(grasp: Any) -> tuple[argparse.Namespace, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_args(parser, grasp)

    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:
        raise ImportError(
            "Could not import Isaac Lab. Run this script with Isaac Lab's Python, "
            "for example `${ISAACLAB_PATH}/isaaclab.sh -p ...`."
        ) from exc

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if str(getattr(args, "target_object", "cube")) == "bottle" and float(args.cube_size) == float(
        getattr(grasp, "DEFAULT_CUBE_SIZE", 0.055)
    ):
        args.cube_size = 0.14
        args.cube_mass = 0.06
    args.cube_position = grasp._parse_vector(str(args.cube_position), 3, name="--cube-position")  # noqa: SLF001
    args.base_cube_position = args.cube_position.copy()
    args.initial_eef_xyz = grasp._parse_vector(  # noqa: SLF001
        str(args.initial_eef_xyz), 3, name="--initial-eef-xyz"
    )
    args.workspace_min = grasp._parse_vector(str(args.workspace_min), 3, name="--workspace-min")  # noqa: SLF001
    args.workspace_max = grasp._parse_vector(str(args.workspace_max), 3, name="--workspace-max")  # noqa: SLF001
    args.cube_positions = _parse_position_list(args.cube_positions, grasp)
    args.table_top_z_auto = args.table_top_z is None
    if args.table_top_z_auto:
        table_reference_cube = args.cube_positions[0] if args.cube_positions else args.cube_position
        args.table_top_z = grasp._infer_table_top_z(  # noqa: SLF001
            table_reference_cube, float(args.cube_size)
        )
    _check_policy_dependencies()
    app_launcher = AppLauncher(args)
    return args, app_launcher.app


def main() -> None:
    grasp = _load_grasp_module()
    compare = _load_compare_module()
    args, simulation_app = _parse_args(grasp)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    ik = None
    try:
        if not bool(getattr(args, "enable_cameras", False)):
            print(
                "[warn] --enable_cameras is not set. Camera render products may be black.",
                flush=True,
            )
        if bool(args.black_wrist):
            print("[camera] policy wrist frames are forced to black.", flush=True)

        policy, modality_config, model_path, dataset_dir = _build_policy(args, compare)
        grasp._print_modality_config(modality_config)  # noqa: SLF001

        first_cube_position = _trial_cube_position(args, 0)
        args.cube_position = first_cube_position
        if getattr(args, "table_top_z_auto", False):
            args.table_top_z = grasp._infer_table_top_z(  # noqa: SLF001
                first_cube_position, float(args.cube_size)
            )
        robot, dof_index, fixed_camera_path, wrist_camera_path = grasp._init_scene(args)  # noqa: SLF001
        teleop = grasp._load_teleop_module()  # noqa: SLF001
        missing = [
            n for n in teleop.ARM_JOINT_NAMES + teleop.HAND_JOINT_NAMES if n not in dof_index
        ]
        if missing:
            raise RuntimeError(f"Missing expected DOFs in imported robot: {missing}")

        ik, pose_from_xyz_rotation = grasp._create_ik(args)  # noqa: SLF001
        marker_path = (
            grasp._create_target_marker("/World/gr00t_eef_target")  # noqa: SLF001
            if args.show_target_marker
            else None
        )
        image_size = (int(args.image_height), int(args.image_width))
        exterior_reader = grasp.CameraReader(fixed_camera_path, image_size=image_size, name="fixed")
        wrist_reader = (
            BlackCameraReader(image_size=image_size)
            if bool(args.black_wrist)
            else grasp.CameraReader(wrist_camera_path, image_size=image_size, name="wrist")
        )
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "bench": BENCH_NAME,
            "model_path": str(model_path),
            "dataset_dir": str(dataset_dir),
            "num_trials": int(args.num_trials),
            "instruction": str(args.instruction),
            "target_object": str(getattr(args, "target_object", "cube")),
            "black_wrist": bool(args.black_wrist),
            "max_steps": int(args.max_steps),
            "replan_horizon": int(args.replan_horizon),
            "cube_position": [float(v) for v in np.asarray(args.cube_position).reshape(3)],
            "cube_size": float(args.cube_size),
            "table_top_z": float(args.table_top_z),
            "table_xy_size": float(args.table_xy_size),
            "success_lift_height": float(args.success_lift_height),
            "success_consecutive_frames": int(args.success_consecutive_frames),
        }
        (output_dir / "bench_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        grasp._wait_for_play_if_requested(args, simulation_app)  # noqa: SLF001
        metrics = []
        for trial_index in range(int(args.num_trials)):
            if not simulation_app.is_running():
                break
            metrics.append(
                _run_trial(
                    trial_index=trial_index,
                    args=args,
                    simulation_app=simulation_app,
                    grasp=grasp,
                    compare=compare,
                    policy=policy,
                    modality_config=modality_config,
                    robot=robot,
                    dof_index=dof_index,
                    exterior_reader=exterior_reader,
                    wrist_reader=wrist_reader,
                    ik=ik,
                    pose_from_xyz_rotation=pose_from_xyz_rotation,
                    marker_path=marker_path,
                    output_dir=output_dir,
                )
            )

        summary = _summarize(metrics, model_path=model_path, dataset_dir=dataset_dir)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _write_csv(output_dir / "per_trial_metrics.csv", metrics)
        print(
            f"[summary] success_rate={summary['success_rate']} "
            f"num_success={summary['num_success']}/{summary['num_trials']} "
            f"saved={output_dir}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[error] {BENCH_NAME} failed: {type(exc).__name__}: {exc!r}",
            flush=True,
        )
        traceback.print_exc()
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
