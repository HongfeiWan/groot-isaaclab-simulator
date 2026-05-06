#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the L10 overfit checkpoint on the curated trimmed LeRobot dataset.

This is a convenience wrapper around compare_l10_gr00t_zero_shot_actions.py.
It defaults to:

- dataset: outputs/IsaacLab/trimmed_l10_dataset
- model: latest checkpoint-* under checkpoints/rokae_xmate3_l10_overfit
- instruction: "pick up the bottle and place it in the box"

It writes per-episode prediction-vs-ground-truth npz/png/metrics files, plus a
single evaluation_summary.json and per_episode_metrics.csv.
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
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "outputs" / "IsaacLab" / "trimmed_l10_dataset"
DEFAULT_MODEL_DIR = REPO_ROOT / "checkpoints" / "rokae_xmate3_l10_overfit"
DEFAULT_CONFIG_PATH = REPO_ROOT / "examples" / "IsaacLab" / "rokae_xmate3_l10_modality_config.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "IsaacLab" / "l10_overfit_trimmed_eval"
DEFAULT_INSTRUCTION = "pick up the bottle and place it in the box"
COMPARE_SCRIPT = REPO_ROOT / "examples" / "IsaacLab" / "compare_l10_gr00t_zero_shot_actions.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Checkpoint or model directory. Defaults to the numerically latest checkpoint "
            "inside checkpoints/rokae_xmate3_l10_overfit."
        ),
    )
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--modality-config-path", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="*",
        default=None,
        help="Episodes to evaluate. Defaults to all episodes in the trimmed dataset.",
    )
    parser.add_argument(
        "--statistics-episode-indices",
        type=int,
        nargs="*",
        default=None,
        help="Episodes used for normalization statistics. Defaults to all evaluated dataset episodes.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Max frames per episode. Defaults to the longest selected episode length.",
    )
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--replan-horizon", type=int, default=8)
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rtc-overlap-steps", type=int, default=None)
    parser.add_argument("--rtc-frozen-steps", type=int, default=2)
    parser.add_argument("--rtc-ramp-rate", type=float, default=6.0)
    parser.add_argument(
        "--video-backend",
        choices=("ffmpeg", "torchcodec", "decord", "opencv"),
        default="ffmpeg",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--missing-video-mode", choices=("black", "duplicate", "error"), default="black")
    parser.add_argument("--strict-policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", type=str, default="rokae-xmate3-l10-eval")
    parser.add_argument("--swanlab-experiment-name", type=str, default=None)
    parser.add_argument("--swanlab-mode", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _load_compare_module():
    spec = importlib.util.spec_from_file_location("compare_l10_gr00t_zero_shot_actions", COMPARE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import compare script: {COMPARE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _all_episode_indices(dataset_dir: Path) -> list[int]:
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        return [int(row["episode_index"]) for row in _read_jsonl(episodes_path)]
    info = _read_json(dataset_dir / "meta" / "info.json")
    return list(range(int(info["total_episodes"])))


def _episode_lengths(dataset_dir: Path) -> dict[int, int]:
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        return {int(row["episode_index"]): int(row["length"]) for row in _read_jsonl(episodes_path)}
    lengths = {}
    info = _read_json(dataset_dir / "meta" / "info.json")
    chunk_size = int(info.get("chunks_size", 1000))
    for episode_index in range(int(info["total_episodes"])):
        parquet_path = dataset_dir / info["data_path"].format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
        )
        import pandas as pd

        lengths[episode_index] = len(pd.read_parquet(parquet_path, columns=["index"]))
    return lengths


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
        has_weights = (child / "model.safetensors.index.json").exists() or (child / "model.safetensors").exists()
        if has_weights:
            candidates.append((step, child))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-<step> model found under {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def _mean_or_none(values: list[Any]) -> float | None:
    numeric = [float(v) for v in values if v is not None]
    return float(np.mean(numeric)) if numeric else None


def _weighted_mean(metrics: list[dict[str, Any]], key: str) -> float | None:
    pairs = [
        (float(m[key]), int(m.get("num_compared_steps", 0)))
        for m in metrics
        if m.get(key) is not None and int(m.get("num_compared_steps", 0)) > 0
    ]
    if not pairs:
        return None
    total_weight = sum(weight for _, weight in pairs)
    return float(sum(value * weight for value, weight in pairs) / total_weight)


def _summarize(metrics: list[dict[str, Any]], *, model_path: Path, dataset_dir: Path) -> dict[str, Any]:
    keys = ["mae", "mse", "rmse", "arm_xyz_mae", "hand_mae", "avg_inference_s", "max_inference_s"]
    return {
        "model_path": str(model_path),
        "dataset_dir": str(dataset_dir),
        "num_episodes": len(metrics),
        "total_compared_steps": int(sum(int(m.get("num_compared_steps", 0)) for m in metrics)),
        "unweighted": {key: _mean_or_none([m.get(key) for m in metrics]) for key in keys},
        "weighted_by_steps": {key: _weighted_mean(metrics, key) for key in keys},
        "episodes": metrics,
    }


def _write_metrics_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    fields = [
        "episode_index",
        "num_compared_steps",
        "mae",
        "mse",
        "rmse",
        "arm_xyz_mae",
        "hand_mae",
        "avg_inference_s",
        "max_inference_s",
        "instruction",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            writer.writerow({field: metric.get(field) for field in fields})


def _maybe_log_swanlab(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    if not bool(args.use_swanlab):
        return
    try:
        import swanlab
    except ImportError:
        logging.warning("swanlab is not installed; skipping SwanLab evaluation logging.")
        return
    init_kwargs = {
        "project": args.swanlab_project,
        "experiment_name": args.swanlab_experiment_name or "l10-overfit-trimmed-eval",
        "config": {
            "model_path": summary["model_path"],
            "dataset_dir": summary["dataset_dir"],
            "num_episodes": summary["num_episodes"],
            "total_compared_steps": summary["total_compared_steps"],
        },
    }
    if args.swanlab_mode:
        init_kwargs["mode"] = args.swanlab_mode
    if swanlab.get_run() is None:
        swanlab.init(**init_kwargs)
    for episode in summary["episodes"]:
        step = int(episode["episode_index"])
        swanlab.log(
            {
                "eval/episode_mae": episode.get("mae"),
                "eval/episode_mse": episode.get("mse"),
                "eval/episode_rmse": episode.get("rmse"),
                "eval/episode_arm_xyz_mae": episode.get("arm_xyz_mae"),
                "eval/episode_hand_mae": episode.get("hand_mae"),
            },
            step=step,
        )
    swanlab.log(
        {
            "eval/weighted_mae": summary["weighted_by_steps"]["mae"],
            "eval/weighted_mse": summary["weighted_by_steps"]["mse"],
            "eval/weighted_rmse": summary["weighted_by_steps"]["rmse"],
            "eval/weighted_arm_xyz_mae": summary["weighted_by_steps"]["arm_xyz_mae"],
            "eval/weighted_hand_mae": summary["weighted_by_steps"]["hand_mae"],
        }
    )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    compare = _load_compare_module()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    model_path = (
        Path(args.model_path).expanduser().resolve()
        if args.model_path
        else _latest_checkpoint(Path(args.model_dir).expanduser().resolve())
    )
    config_path = Path(args.modality_config_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    episode_indices = list(args.episode_indices) if args.episode_indices else _all_episode_indices(dataset_dir)
    lengths = _episode_lengths(dataset_dir)
    steps = int(args.steps) if args.steps is not None else max(lengths[idx] for idx in episode_indices)
    statistics_episode_indices = (
        list(args.statistics_episode_indices)
        if args.statistics_episode_indices
        else _all_episode_indices(dataset_dir)
    )

    from gr00t.data.embodiment_tags import EmbodimentTag

    modality_config = compare._load_rokae_modality_config(config_path)
    statistics = compare._build_dataset_statistics(
        dataset_dir,
        modality_config,
        episode_indices=statistics_episode_indices,
        embodiment_value=EmbodimentTag.NEW_EMBODIMENT.value,
    )
    device = compare._resolve_device(str(args.device))
    logging.info("model_path=%s", model_path)
    logging.info("dataset_dir=%s", dataset_dir)
    logging.info("episode_indices=%s", episode_indices)
    logging.info("statistics_episode_indices=%s", statistics_episode_indices)
    logging.info("steps=%s", steps)
    logging.info("device=%s", device)

    policy = compare.InjectedRokaePolicy(
        model_path=model_path,
        modality_config=deepcopy(modality_config),
        statistics=statistics,
        device=device,
        strict=bool(args.strict_policy),
    )

    compare_args = SimpleNamespace(
        instruction=args.instruction,
        steps=steps,
        replan_horizon=int(args.replan_horizon),
        rtc=bool(args.rtc),
        rtc_overlap_steps=args.rtc_overlap_steps,
        rtc_frozen_steps=int(args.rtc_frozen_steps),
        rtc_ramp_rate=float(args.rtc_ramp_rate),
        video_backend=str(args.video_backend),
        missing_video_mode=str(args.missing_video_mode),
        no_plot=bool(args.no_plot),
    )

    metrics = []
    for episode_index in episode_indices:
        logging.info("evaluating episode %s length=%s", episode_index, lengths.get(episode_index))
        metrics.append(
            compare._run_episode(
                policy=policy,
                dataset_dir=dataset_dir,
                episode_index=int(episode_index),
                modality_config=modality_config,
                output_dir=output_dir,
                args=compare_args,
            )
        )

    summary = _summarize(metrics, model_path=model_path, dataset_dir=dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "evaluation_summary.json"
    csv_path = output_dir / "per_episode_metrics.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_metrics_csv(csv_path, metrics)
    _maybe_log_swanlab(args, summary)

    logging.info("summary saved to %s", summary_path)
    logging.info("per-episode csv saved to %s", csv_path)
    logging.info("weighted_mae=%s", summary["weighted_by_steps"]["mae"])
    logging.info("weighted_arm_xyz_mae=%s", summary["weighted_by_steps"]["arm_xyz_mae"])
    logging.info("weighted_hand_mae=%s", summary["weighted_by_steps"]["hand_mae"])


if __name__ == "__main__":
    main()
