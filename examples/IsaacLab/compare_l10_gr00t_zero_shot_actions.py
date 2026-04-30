#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open-loop GR00T action comparison on the Rokae xMate3 + L10 LeRobot dataset.

This follows the same idea as the README's "Zero-Shot Inference (Base Model)"
example, but targets the local L10 dataset and writes explicit predicted-vs-GT
action files.

Example:

    .venv/bin/python examples/IsaacLab/compare_l10_gr00t_zero_shot_actions.py \
        --model-path checkpoints/GR00T-N1.7-3B \
        --episode-indices 0 \
        --steps 128

Important:
    The base GR00T-N1.7-3B checkpoint is not trained on this custom xMate3+L10
    embodiment. This script injects the local Rokae modality config and dataset
    statistics into the checkpoint processor so you can run a zero-shot/open-loop
    baseline and quantify how far it is from the recorded actions.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "checkpoints" / "GR00T-N1.7-3B"
DEFAULT_DATASET_DIR = (
    REPO_ROOT / "demo_data" / "l10_hand" / "lerobot_rokae_xmate3_linker_l10_groot_v1"
)
DEFAULT_CONFIG_PATH = REPO_ROOT / "examples" / "IsaacLab" / "rokae_xmate3_l10_modality_config.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "IsaacLab" / "l10_zero_shot_action_compare"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--modality-config-path", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--episode-indices", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--statistics-episode-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Episodes used to compute NEW_EMBODIMENT normalization statistics. "
            "Defaults to all episodes listed in meta/episodes.jsonl."
        ),
    )
    parser.add_argument("--steps", type=int, default=160, help="Max frames per episode to compare.")
    parser.add_argument(
        "--replan-horizon",
        type=int,
        default=8,
        help=(
            "Stride between model calls. With the L10 action horizon of 16, the default 8 "
            "leaves 8 RTC overlap steps."
        ),
    )
    parser.add_argument(
        "--rtc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable RTC action chunk overlap by feeding the previous predicted chunk back in.",
    )
    parser.add_argument(
        "--rtc-overlap-steps",
        type=int,
        default=None,
        help=(
            "Previous chunk steps to overlap into the next DiT sample. "
            "Defaults to action_horizon - replan_horizon."
        ),
    )
    parser.add_argument(
        "--rtc-frozen-steps",
        type=int,
        default=2,
        help="Initial overlap steps kept fixed from the previous action chunk.",
    )
    parser.add_argument(
        "--rtc-ramp-rate",
        type=float,
        default=6.0,
        help="Exponential ramp rate used by the GR00T RTC inpainting logic.",
    )
    parser.add_argument(
        "--video-backend",
        choices=("ffmpeg", "torchcodec", "decord", "opencv"),
        default="ffmpeg",
        help="Backend used by gr00t.utils.video_utils.get_frames_by_indices.",
    )
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', 'cuda:0', or 'cpu'.")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--duplicate-missing-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Duplicate the first dataset video stream when the config asks for a missing view.",
    )
    parser.add_argument(
        "--strict-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Gr00tPolicy input/output validation.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _load_rokae_modality_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Modality config not found: {config_path}")
    spec = importlib.util.spec_from_file_location("rokae_xmate3_l10_modality_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import modality config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rokae_xmate3_l10_config


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _all_episode_indices(dataset_dir: Path) -> list[int]:
    episodes_file = dataset_dir / "meta" / "episodes.jsonl"
    if episodes_file.exists():
        return [int(row["episode_index"]) for row in _read_jsonl(episodes_file)]

    info = _read_json(dataset_dir / "meta" / "info.json")
    return list(range(int(info["total_episodes"])))


def _load_episode_parquet(dataset_dir: Path, episode_index: int) -> Any:
    import pandas as pd

    info = _read_json(dataset_dir / "meta" / "info.json")
    chunk_size = int(info.get("chunks_size", 1000))
    data_path = info["data_path"].format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )
    parquet_path = dataset_dir / data_path
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


def _to_matrix(column: Any, *, expected_dim: int, name: str) -> np.ndarray:
    out = np.asarray([np.asarray(v, dtype=np.float32) for v in column], dtype=np.float32)
    if out.ndim != 2 or out.shape[1] != expected_dim:
        raise RuntimeError(f"Unexpected {name} shape {out.shape}, expected (*, {expected_dim})")
    return out


def _stats_for_matrix(values: np.ndarray) -> dict[str, list[float]]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix for stats, got {arr.shape}")
    return {
        "min": np.min(arr, axis=0).astype(float).tolist(),
        "max": np.max(arr, axis=0).astype(float).tolist(),
        "mean": np.mean(arr, axis=0).astype(float).tolist(),
        "std": np.maximum(np.std(arr, axis=0), 1.0e-8).astype(float).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).astype(float).tolist(),
    }


def _build_dataset_statistics(
    dataset_dir: Path,
    modality_config: dict[str, Any],
    *,
    episode_indices: list[int],
    embodiment_value: str,
) -> dict[str, Any]:
    modality_meta = _read_json(dataset_dir / "meta" / "modality.json")
    state_chunks: dict[str, list[np.ndarray]] = {
        key: [] for key in modality_config["state"].modality_keys
    }
    action_chunks: dict[str, list[np.ndarray]] = {
        key: [] for key in modality_config["action"].modality_keys
    }

    for episode_index in episode_indices:
        df = _load_episode_parquet(dataset_dir, episode_index)
        states = _to_matrix(df["observation.state"], expected_dim=20, name="observation.state")
        actions = _to_matrix(df["action"], expected_dim=13, name="action")

        for key in state_chunks:
            sl = modality_meta["state"][key]
            state_chunks[key].append(states[:, int(sl["start"]) : int(sl["end"])])
        for key in action_chunks:
            sl = modality_meta["action"][key]
            action_chunks[key].append(actions[:, int(sl["start"]) : int(sl["end"])])

    stats: dict[str, Any] = {embodiment_value: {"state": {}, "action": {}, "relative_action": {}}}
    for key, chunks in state_chunks.items():
        stats[embodiment_value]["state"][key] = _stats_for_matrix(np.concatenate(chunks, axis=0))
    for key, chunks in action_chunks.items():
        stats[embodiment_value]["action"][key] = _stats_for_matrix(np.concatenate(chunks, axis=0))

    action_configs = modality_config["action"].action_configs or []
    for key, action_config in zip(modality_config["action"].modality_keys, action_configs):
        if str(getattr(action_config, "rep", "")).endswith("RELATIVE"):
            state_key = action_config.state_key or key
            if state_key not in state_chunks:
                continue
            action_values = np.concatenate(action_chunks[key], axis=0)
            state_values = np.concatenate(state_chunks[state_key], axis=0)
            if action_values.shape[1] == state_values.shape[1]:
                stats[embodiment_value]["relative_action"][key] = _stats_for_matrix(
                    action_values - state_values
                )

    if not stats[embodiment_value]["relative_action"]:
        stats[embodiment_value].pop("relative_action")
    return stats


class InjectedRokaePolicy:
    """A small Gr00tPolicy-compatible wrapper with injected custom modality config."""

    def __init__(
        self,
        *,
        model_path: Path,
        modality_config: dict[str, Any],
        statistics: dict[str, Any],
        device: str,
        strict: bool,
    ) -> None:
        from gr00t.data.embodiment_tags import EmbodimentTag
        import gr00t.model  # noqa: F401
        from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
        from gr00t.policy.policy import BasePolicy
        import torch
        from transformers import AutoModel, AutoProcessor

        BasePolicy.__init__(self, strict=strict)
        self.embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        self.model.to(device=device, dtype=torch.bfloat16)

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            modality_configs={self.embodiment_tag.value: modality_config},
        )
        self.processor.set_statistics(statistics, override=True)
        self.processor.eval()

        all_modality_configs = self.processor.get_modality_configs()
        self.modality_configs = {
            k: v
            for k, v in all_modality_configs[self.embodiment_tag.value].items()
            if k != "rl_info"
        }
        self.collate_fn = self.processor.collator
        self.language_key = self.modality_configs["language"].modality_keys[0]
        self._rec_to_dtype = _rec_to_dtype

        # Reuse the validated GR00T policy implementation after custom initialization.
        self._unbatch_observation = Gr00tPolicy._unbatch_observation.__get__(self)
        self.check_observation = Gr00tPolicy.check_observation.__get__(self)
        self.check_action = Gr00tPolicy.check_action.__get__(self)
        self.get_modality_config = Gr00tPolicy.get_modality_config.__get__(self)
        self.reset = Gr00tPolicy.reset.__get__(self)

    def _get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
        *,
        previous_action: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from gr00t.data.types import MessageType, VLAStepData
        import torch

        unbatched_observations = self._unbatch_observation(observation)
        if previous_action is not None and len(unbatched_observations) != 1:
            raise ValueError("RTC previous_action currently supports batch size 1.")

        processed_inputs = []
        states = []
        for obs in unbatched_observations:
            states.append(obs["state"])
            vla_step_data = VLAStepData(
                images=obs["video"],
                states=obs["state"],
                actions={} if previous_action is None else previous_action,
                text=obs["language"][self.language_key][0],
                embodiment=self.embodiment_tag,
            )
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = self._rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
        with torch.inference_mode():
            model_pred = self.model.get_action(**collated_inputs, options=options)
        normalized_action = model_pred["action_pred"].float()

        batched_states = {}
        for key in self.modality_configs["state"].modality_keys:
            batched_states[key] = np.stack([state[key] for state in states], axis=0)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )
        return {key: value.astype(np.float32) for key, value in unnormalized_action.items()}, {}

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
        *,
        previous_action: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.strict:
            self.check_observation(observation)
        action, info = self._get_action(
            observation, options, previous_action=previous_action
        )
        if self.strict:
            self.check_action(action)
        return action, info


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_video_frames(
    dataset_dir: Path,
    *,
    episode_index: int,
    video_key: str,
    frame_indices: np.ndarray,
    video_backend: str,
) -> np.ndarray:
    from gr00t.utils.video_utils import get_frames_by_indices

    info = _read_json(dataset_dir / "meta" / "info.json")
    modality_meta = _read_json(dataset_dir / "meta" / "modality.json")
    chunk_size = int(info.get("chunks_size", 1000))
    original_key = modality_meta["video"][video_key].get(
        "original_key", f"observation.images.{video_key}"
    )
    video_path = dataset_dir / info["video_path"].format(
        episode_chunk=episode_index // chunk_size,
        video_key=original_key,
        episode_index=episode_index,
    )
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    frames = get_frames_by_indices(
        str(video_path),
        np.asarray(frame_indices, dtype=np.int64),
        video_backend=video_backend,
        video_backend_kwargs={},
    )
    return np.asarray(frames, dtype=np.uint8)


def _extract_groups(
    matrix: np.ndarray,
    modality_meta: dict[str, Any],
    modality: str,
    keys: list[str],
) -> dict[str, np.ndarray]:
    out = {}
    for key in keys:
        sl = modality_meta[modality][key]
        out[key] = matrix[:, int(sl["start"]) : int(sl["end"])].astype(np.float32, copy=False)
    return out


def _get_instruction(dataset_dir: Path, df: Any, *, override: str | None) -> str:
    if override:
        return override
    tasks = _read_jsonl(dataset_dir / "meta" / "tasks.jsonl")
    task_map = {int(row["task_index"]): row["task"] for row in tasks}
    if "annotation.human.action.task_description" in df.columns:
        task_index = int(df["annotation.human.action.task_description"].iloc[0])
        return str(task_map.get(task_index, "teleop"))
    return "teleop"


def _build_observation(
    dataset_dir: Path,
    *,
    episode_index: int,
    step: int,
    states_by_key: dict[str, np.ndarray],
    modality_config: dict[str, Any],
    modality_meta: dict[str, Any],
    instruction: str,
    video_backend: str,
    duplicate_missing_video: bool,
) -> dict[str, Any]:
    obs: dict[str, Any] = {"video": {}, "state": {}, "language": {}}

    for key in modality_config["state"].modality_keys:
        delta = np.asarray(modality_config["state"].delta_indices, dtype=np.int64)
        indices = np.clip(step + delta, 0, len(states_by_key[key]) - 1)
        obs["state"][key] = states_by_key[key][indices][None, :].astype(np.float32, copy=False)

    dataset_video_keys = list(modality_meta.get("video", {}).keys())
    fallback_video_key = dataset_video_keys[0] if dataset_video_keys else None
    for key in modality_config["video"].modality_keys:
        source_key = key
        if source_key not in modality_meta.get("video", {}):
            if not duplicate_missing_video or fallback_video_key is None:
                raise KeyError(
                    f"Video key '{key}' not in dataset modality.json. "
                    f"Available: {dataset_video_keys}"
                )
            logging.warning(
                "Video key '%s' is missing in dataset; duplicating '%s'.",
                key,
                fallback_video_key,
            )
            source_key = fallback_video_key
        delta = np.asarray(modality_config["video"].delta_indices, dtype=np.int64)
        indices = np.clip(step + delta, 0, len(states_by_key[next(iter(states_by_key))]) - 1)
        frames = _load_video_frames(
            dataset_dir,
            episode_index=episode_index,
            video_key=source_key,
            frame_indices=indices,
            video_backend=video_backend,
        )
        obs["video"][key] = frames[None, :].astype(np.uint8, copy=False)

    language_key = modality_config["language"].modality_keys[0]
    obs["language"][language_key] = [[instruction]]
    return obs


def _concat_action_dict(action: dict[str, np.ndarray], action_keys: list[str]) -> np.ndarray:
    chunks = []
    for key in action_keys:
        value = np.asarray(action[key], dtype=np.float32)
        if value.ndim == 3:
            value = value[0]
        chunks.append(value)
    return np.concatenate(chunks, axis=-1)


def _unbatch_action_dict(action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    previous_action = {}
    for key, value in action.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        previous_action[key] = arr.astype(np.float32, copy=True)
    return previous_action


def _rtc_options(
    *,
    args: argparse.Namespace,
    action_horizon: int,
    previous_action: dict[str, np.ndarray] | None,
) -> dict[str, Any] | None:
    if not bool(args.rtc) or previous_action is None:
        return None

    previous_horizon = min(int(v.shape[0]) for v in previous_action.values())
    overlap_steps = (
        int(args.rtc_overlap_steps)
        if args.rtc_overlap_steps is not None
        else action_horizon - int(args.replan_horizon)
    )
    overlap_steps = max(0, min(overlap_steps, previous_horizon, action_horizon))
    if overlap_steps <= 0:
        logging.warning(
            "RTC requested but overlap is 0. Use --replan-horizon smaller than action horizon %d.",
            action_horizon,
        )
        return None

    frozen_steps = max(0, min(int(args.rtc_frozen_steps), overlap_steps))
    return {
        "action_horizon": int(previous_horizon),
        "rtc_overlap_steps": int(overlap_steps),
        "rtc_frozen_steps": int(frozen_steps),
        "rtc_ramp_rate": float(args.rtc_ramp_rate),
    }


def _plot_actions(
    *,
    output_path: Path,
    gt: np.ndarray,
    pred: np.ndarray,
    frame_indices: np.ndarray,
    action_names: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning("matplotlib unavailable; skipping plot: %s", exc)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_dim = gt.shape[1]
    fig, axes = plt.subplots(n_dim, 1, figsize=(12, max(3, 2.2 * n_dim)), sharex=True)
    if n_dim == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        label = action_names[i] if i < len(action_names) else f"action_{i}"
        ax.plot(frame_indices, gt[:, i], label="ground truth", linewidth=1.4)
        ax.plot(frame_indices, pred[:, i], label="prediction", linewidth=1.1)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("dataset frame index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _summarize_errors(gt: np.ndarray, pred: np.ndarray, action_names: list[str]) -> dict[str, Any]:
    err = pred - gt
    per_dim_mae = np.mean(np.abs(err), axis=0)
    per_dim_rmse = np.sqrt(np.mean(err * err, axis=0))
    return {
        "num_compared_steps": int(gt.shape[0]),
        "action_dim": int(gt.shape[1]),
        "mse": float(np.mean(err * err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "arm_xyz_mae": float(np.mean(np.abs(err[:, :3]))) if gt.shape[1] >= 3 else None,
        "hand_mae": float(np.mean(np.abs(err[:, 3:]))) if gt.shape[1] > 3 else None,
        "per_dim": [
            {
                "index": int(i),
                "name": action_names[i] if i < len(action_names) else f"action_{i}",
                "mae": float(per_dim_mae[i]),
                "rmse": float(per_dim_rmse[i]),
            }
            for i in range(gt.shape[1])
        ],
    }


def _action_names(dataset_dir: Path) -> list[str]:
    info = _read_json(dataset_dir / "meta" / "info.json")
    return list(info["features"]["action"].get("names") or [])


def _run_episode(
    *,
    policy: InjectedRokaePolicy,
    dataset_dir: Path,
    episode_index: int,
    modality_config: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    modality_meta = _read_json(dataset_dir / "meta" / "modality.json")
    df = _load_episode_parquet(dataset_dir, episode_index)
    states = _to_matrix(df["observation.state"], expected_dim=20, name="observation.state")
    actions = _to_matrix(df["action"], expected_dim=13, name="action")

    states_by_key = _extract_groups(
        states,
        modality_meta,
        "state",
        list(modality_config["state"].modality_keys),
    )
    actions_by_key = _extract_groups(
        actions,
        modality_meta,
        "action",
        list(modality_config["action"].modality_keys),
    )
    gt_action_full = np.concatenate(
        [actions_by_key[key] for key in modality_config["action"].modality_keys],
        axis=-1,
    )
    action_keys = list(modality_config["action"].modality_keys)
    action_names = _action_names(dataset_dir)
    instruction = _get_instruction(dataset_dir, df, override=args.instruction)

    actual_steps = min(int(args.steps), len(df))
    action_horizon = len(modality_config["action"].delta_indices)
    pred_rows: list[np.ndarray] = []
    gt_rows: list[np.ndarray] = []
    frame_rows: list[int] = []
    inference_times: list[float] = []
    previous_action: dict[str, np.ndarray] | None = None

    for step in range(0, actual_steps, int(args.replan_horizon)):
        obs = _build_observation(
            dataset_dir,
            episode_index=episode_index,
            step=step,
            states_by_key=states_by_key,
            modality_config=modality_config,
            modality_meta=modality_meta,
            instruction=instruction,
            video_backend=str(args.video_backend),
            duplicate_missing_video=bool(args.duplicate_missing_video),
        )
        rtc_options = _rtc_options(
            args=args,
            action_horizon=action_horizon,
            previous_action=previous_action,
        )
        tic = time.perf_counter()
        pred_action, _ = policy.get_action(
            obs,
            options=rtc_options,
            previous_action=previous_action if rtc_options is not None else None,
        )
        inference_times.append(time.perf_counter() - tic)
        pred_chunk = _concat_action_dict(pred_action, action_keys)
        previous_action = _unbatch_action_dict(pred_action)

        horizon = min(int(args.replan_horizon), len(pred_chunk), actual_steps - step)
        pred_rows.append(pred_chunk[:horizon])
        gt_rows.append(gt_action_full[step : step + horizon])
        frame_rows.extend(range(step, step + horizon))
        logging.info(
            "episode=%d step=%d horizon=%d rtc=%s inference=%.3fs",
            episode_index,
            step,
            horizon,
            "off" if rtc_options is None else rtc_options,
            inference_times[-1],
        )

    pred = np.concatenate(pred_rows, axis=0).astype(np.float32)
    gt = np.concatenate(gt_rows, axis=0).astype(np.float32)
    frames = np.asarray(frame_rows, dtype=np.int64)
    metrics = _summarize_errors(gt, pred, action_names)
    metrics.update(
        {
            "episode_index": int(episode_index),
            "instruction": instruction,
            "avg_inference_s": float(np.mean(inference_times)) if inference_times else 0.0,
            "max_inference_s": float(np.max(inference_times)) if inference_times else 0.0,
            "replan_horizon": int(args.replan_horizon),
            "rtc": bool(args.rtc),
            "rtc_overlap_steps": (
                int(args.rtc_overlap_steps)
                if args.rtc_overlap_steps is not None
                else max(0, action_horizon - int(args.replan_horizon))
            ),
            "rtc_frozen_steps": int(args.rtc_frozen_steps),
            "rtc_ramp_rate": float(args.rtc_ramp_rate),
        }
    )

    episode_prefix = output_dir / f"episode_{episode_index:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        episode_prefix.with_suffix(".npz"),
        frame_index=frames,
        pred_action=pred,
        gt_action=gt,
        action_names=np.asarray(action_names, dtype=str),
    )
    with episode_prefix.with_suffix(".metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if not bool(args.no_plot):
        _plot_actions(
            output_path=episode_prefix.with_suffix(".png"),
            gt=gt,
            pred=pred,
            frame_indices=frames,
            action_names=action_names,
        )
    return metrics


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    config_path = Path(args.modality_config_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_dir}")

    from gr00t.data.embodiment_tags import EmbodimentTag

    modality_config = _load_rokae_modality_config(config_path)
    statistics_episode_indices = (
        list(args.statistics_episode_indices)
        if args.statistics_episode_indices is not None
        else _all_episode_indices(dataset_dir)
    )
    statistics = _build_dataset_statistics(
        dataset_dir,
        modality_config,
        episode_indices=statistics_episode_indices,
        embodiment_value=EmbodimentTag.NEW_EMBODIMENT.value,
    )
    device = _resolve_device(str(args.device))

    logging.info("model_path=%s", model_path)
    logging.info("dataset_dir=%s", dataset_dir)
    logging.info("statistics_episodes=%s", statistics_episode_indices)
    logging.info("device=%s", device)
    if device == "cpu":
        logging.warning(
            "CUDA is not visible; loading the 3B base model on CPU will be very slow and memory heavy."
        )
    logging.info(
        "Injecting custom NEW_EMBODIMENT config into base checkpoint processor for open-loop comparison."
    )
    policy = InjectedRokaePolicy(
        model_path=model_path,
        modality_config=deepcopy(modality_config),
        statistics=statistics,
        device=device,
        strict=bool(args.strict_policy),
    )

    all_metrics = []
    for episode_index in args.episode_indices:
        all_metrics.append(
            _run_episode(
                policy=policy,
                dataset_dir=dataset_dir,
                episode_index=int(episode_index),
                modality_config=modality_config,
                output_dir=output_dir,
                args=args,
            )
        )

    summary = {
        "model_path": str(model_path),
        "dataset_dir": str(dataset_dir),
        "episodes": all_metrics,
        "average_mae": float(np.mean([m["mae"] for m in all_metrics])) if all_metrics else None,
        "average_mse": float(np.mean([m["mse"] for m in all_metrics])) if all_metrics else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info("summary saved to %s", output_dir / "summary.json")
    logging.info("average_mae=%s average_mse=%s", summary["average_mae"], summary["average_mse"])


if __name__ == "__main__":
    main()
