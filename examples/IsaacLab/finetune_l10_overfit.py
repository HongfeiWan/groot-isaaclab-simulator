#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local overfit fine-tuning launcher for Rokae xMate3 + Linker L10 hand.

This script is intentionally more opinionated than examples/finetune.sh:

1. It prepares a training dataset under outputs/IsaacLab by symlinking the
   original data/videos and copying meta files.
2. It keeps the dataset's single ego/head camera modality unchanged, matching
   examples/IsaacLab/rokae_xmate3_l10_modality_config.py.
3. It regenerates meta/stats.json and meta/relative_stats.json for training.
4. It launches single-GPU GR00T N1.7 fine-tuning in an overfit-friendly setup.

Example smoke run:

    .venv/bin/python examples/IsaacLab/finetune_l10_overfit.py \
        --max-steps 500 \
        --global-batch-size 2 \
        --prepare-only

Example training run:

    .venv/bin/python examples/IsaacLab/finetune_l10_overfit.py \
        --max-steps 3000 \
        --global-batch-size 1 \
        --gradient-accumulation-steps 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL_PATH = REPO_ROOT / "checkpoints" / "GR00T-N1.7-3B"
DEFAULT_VLM_MODEL_PATH = REPO_ROOT / "checkpoints" / "nvidia" / "Cosmos-Reason2-2B"
DEFAULT_DATASET_DIR = (
    REPO_ROOT / "demo_data" / "l10_hand" / "lerobot_rokae_xmate3_linker_l10_groot_v1"
)
DEFAULT_PREPARED_DATASET_DIR = REPO_ROOT / "outputs" / "IsaacLab" / "l10_overfit_dataset"
DEFAULT_MODALITY_CONFIG_PATH = (
    REPO_ROOT / "examples" / "IsaacLab" / "rokae_xmate3_l10_modality_config.py"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "checkpoints" / "rokae_xmate3_l10_overfit"
DEFAULT_INSTRUCTION = "pick up the bottle and place it in the box"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=str, default=str(DEFAULT_BASE_MODEL_PATH))
    parser.add_argument("--vlm-model-path", type=str, default=str(DEFAULT_VLM_MODEL_PATH))
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument(
        "--prepared-dataset-dir", type=str, default=str(DEFAULT_PREPARED_DATASET_DIR)
    )
    parser.add_argument("--modality-config-path", type=str, default=str(DEFAULT_MODALITY_CONFIG_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--embodiment-tag", type=str, default="NEW_EMBODIMENT")
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help=(
            "Language instruction written to task_index 0 in the prepared dataset. "
            "Use an English task string for best compatibility with the base VLM."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If output_dir already contains checkpoint-<step>, resume from the largest step "
            "and continue until --max-steps. If that step is already >= --max-steps, exit."
        ),
    )
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=1,
        help=(
            "This becomes per-device microbatch size when num_gpus=1. "
            "Use gradient accumulation for a larger effective batch."
        ),
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--num-shards-per-epoch", type=int, default=128)
    parser.add_argument(
        "--episode-sampling-rate",
        type=float,
        default=1.0,
        help="Use 1.0 for overfitting all frames instead of the default 0.1 subsampling.",
    )
    parser.add_argument(
        "--video-backend",
        choices=("ffmpeg", "torchcodec", "opencv", "decord"),
        default="ffmpeg",
    )
    parser.add_argument("--cuda-visible-devices", type=str, default="0")
    parser.add_argument("--state-dropout-prob", type=float, default=0.0)
    parser.add_argument(
        "--load-bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the frozen VLM backbone in bf16 to reduce VRAM.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trade compute for lower activation memory during training.",
    )
    parser.add_argument(
        "--tune-projector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train action state/action encoder-decoder projector layers.",
    )
    parser.add_argument(
        "--tune-diffusion-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train the diffusion action model. Default freezes it because Adam states for this "
            "block are too large for many 22-24GB local GPUs."
        ),
    )
    parser.add_argument(
        "--tune-vlln",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train visual-language LayerNorm/self-attention bridge. Default freezes it for local VRAM.",
    )
    parser.add_argument(
        "--keep-color-jitter",
        action="store_true",
        help="Keep checkpoint-style color jitter. Default disables it for pure overfit.",
    )
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="finetune-gr00t-n1d7")
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", type=str, default="finetune-gr00t-n1d7")
    parser.add_argument("--swanlab-workspace", type=str, default=None)
    parser.add_argument(
        "--swanlab-mode",
        type=str,
        default=None,
        help="Optional SwanLab mode: cloud, local, offline, or disabled.",
    )
    parser.add_argument("--swanlab-logdir", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help="Hugging Face optimizer name. Try paged_adamw_8bit if bitsandbytes is installed.",
    )
    parser.add_argument(
        "--save-only-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save smaller checkpoints without optimizer state. Keep disabled for exact resume "
            "with optimizer/scheduler state."
        ),
    )
    parser.add_argument(
        "--force-regenerate-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Regenerate stats files in the prepared dataset.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare dataset/stats and validate loader, then exit before loading the model.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Avoid network access when loading local checkpoints.",
    )
    parser.add_argument(
        "--disable-deepspeed-import",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Single-GPU workaround: stop Accelerate from importing DeepSpeed during Trainer "
            "setup. This avoids Triton compiling cuda_utils, which needs Python.h."
        ),
    )
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_modality_config(config_path: Path) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"Modality config not found: {config_path}")
    sys.path.append(str(config_path.parent))
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import modality config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"[config] loaded modality config: {config_path}", flush=True)


def _copy_meta(source_dataset: Path, prepared_dataset: Path, instruction: str | None) -> None:
    source_meta = source_dataset / "meta"
    prepared_meta = prepared_dataset / "meta"
    prepared_meta.mkdir(parents=True, exist_ok=True)

    for src in source_meta.iterdir():
        if src.is_file() and src.name not in {"stats.json", "relative_stats.json"}:
            shutil.copy2(src, prepared_meta / src.name)

    modality_path = prepared_meta / "modality.json"
    modality = _read_json(modality_path)
    video = modality.get("video", {})
    if "ego_view" not in video:
        raise KeyError(f"Expected ego_view in {modality_path}, got video keys={list(video)}")

    if instruction:
        _override_primary_instruction(prepared_meta, instruction)


def _override_primary_instruction(prepared_meta: Path, instruction: str) -> None:
    tasks_path = prepared_meta / "tasks.jsonl"
    task_rows = [
        json.loads(line)
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    found_task_zero = False
    for row in task_rows:
        if int(row["task_index"]) == 0:
            row["task"] = instruction
            found_task_zero = True
            break
    if not found_task_zero:
        task_rows.insert(0, {"task_index": 0, "task": instruction})
    tasks_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in task_rows),
        encoding="utf-8",
    )

    episodes_path = prepared_meta / "episodes.jsonl"
    episode_rows = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in episode_rows:
        if row.get("tasks") == ["teleop"]:
            row["tasks"] = [instruction]
    episodes_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episode_rows),
        encoding="utf-8",
    )
    print(f"[data] instruction for task_index=0: {instruction!r}", flush=True)


def _ensure_prepared_symlink(link_path: Path, target_path: Path) -> None:
    target_path = target_path.resolve()
    if link_path.is_symlink():
        if link_path.resolve() == target_path:
            return
        link_path.unlink()
    elif link_path.exists():
        # This prepared dataset is generated by this script. Older revisions
        # created a real videos/ tree with black wrist-view files; replace it
        # so reruns match the single-camera dataset exactly.
        if link_path.is_dir() and link_path.parent.name == "l10_overfit_dataset":
            shutil.rmtree(link_path)
        else:
            raise FileExistsError(
                f"{link_path} already exists and is not the expected symlink to {target_path}"
            )
    link_path.symlink_to(target_path, target_is_directory=target_path.is_dir())


def _prepare_dataset(
    *,
    source_dataset: Path,
    prepared_dataset: Path,
    embodiment_tag: Any,
    modality_config_path: Path,
    force_regenerate_stats: bool,
    instruction: str | None,
) -> Path:
    if not source_dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {source_dataset}")
    prepared_dataset.mkdir(parents=True, exist_ok=True)
    _ensure_prepared_symlink(prepared_dataset / "data", source_dataset / "data")
    _ensure_prepared_symlink(prepared_dataset / "videos", source_dataset / "videos")
    _copy_meta(source_dataset, prepared_dataset, instruction)

    if force_regenerate_stats:
        for name in ("stats.json", "relative_stats.json"):
            path = prepared_dataset / "meta" / name
            if path.exists():
                path.unlink()

    from gr00t.data.stats import generate_rel_stats, generate_stats

    generate_stats(prepared_dataset)
    generate_rel_stats(prepared_dataset, embodiment_tag)
    print(f"[data] prepared training dataset: {prepared_dataset}", flush=True)
    return prepared_dataset


def _validate_prepared_loader(prepared_dataset: Path, embodiment_tag: Any) -> None:
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    loader = LeRobotEpisodeLoader(
        prepared_dataset,
        MODALITY_CONFIGS[embodiment_tag.value],
        video_backend="ffmpeg",
    )
    stats = loader.get_dataset_statistics()
    print(
        "[check] loader ok: "
        f"episodes={len(loader)}, "
        f"state_keys={list(stats['state'])}, "
        f"action_keys={list(stats['action'])}, "
        f"relative_keys={list(stats.get('relative_action', {}))}",
        flush=True,
    )


def _build_training_config(args: argparse.Namespace, prepared_dataset: Path, embodiment_tag: Any):
    from gr00t.configs.base_config import get_default_config
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [str(prepared_dataset)],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag.value,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = bool(args.tune_projector)
    config.model.tune_diffusion_model = bool(args.tune_diffusion_model)
    config.model.tune_vlln = bool(args.tune_vlln)
    config.model.state_dropout_prob = float(args.state_dropout_prob)
    config.model.random_rotation_angle = 0
    config.model.color_jitter_params = (
        {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}
        if args.keep_color_jitter
        else {}
    )
    config.model.extra_augmentation_config = None
    config.model.load_bf16 = bool(args.load_bf16)
    config.model.reproject_vision = False
    config.model.model_name = str(Path(args.vlm_model_path).expanduser().resolve())
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.start_from_checkpoint = str(Path(args.base_model_path).expanduser().resolve())
    config.training.resume_from_checkpoint = bool(args.auto_resume)
    config.training.optim = str(args.optim)
    config.training.global_batch_size = int(args.global_batch_size)
    config.training.gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    config.training.gradient_checkpointing = bool(args.gradient_checkpointing)
    config.training.dataloader_num_workers = int(args.dataloader_num_workers)
    config.training.learning_rate = float(args.learning_rate)
    config.training.weight_decay = float(args.weight_decay)
    config.training.warmup_ratio = float(args.warmup_ratio)
    config.training.output_dir = str(Path(args.output_dir).expanduser().resolve())
    config.training.experiment_name = args.experiment_name
    config.training.save_steps = int(args.save_steps)
    config.training.save_total_limit = int(args.save_total_limit)
    config.training.num_gpus = 1
    config.training.use_wandb = bool(args.use_wandb)
    config.training.wandb_project = str(args.wandb_project)
    config.training.use_swanlab = bool(args.use_swanlab)
    config.training.swanlab_project = str(args.swanlab_project)
    config.training.swanlab_workspace = args.swanlab_workspace
    config.training.swanlab_mode = args.swanlab_mode
    config.training.swanlab_logdir = args.swanlab_logdir
    config.training.max_steps = int(args.max_steps)
    config.training.save_only_model = bool(args.save_only_model)
    config.training.transformers_local_files_only = bool(args.local_files_only)

    config.data.video_backend = str(args.video_backend)
    config.data.episode_sampling_rate = float(args.episode_sampling_rate)
    config.data.shard_size = _safe_shard_size(
        prepared_dataset=prepared_dataset,
        modality_config=MODALITY_CONFIGS[embodiment_tag.value],
        requested_shard_size=int(args.shard_size),
        episode_sampling_rate=config.data.episode_sampling_rate,
    )
    config.data.num_shards_per_epoch = int(args.num_shards_per_epoch)

    return config


def _safe_shard_size(
    *,
    prepared_dataset: Path,
    modality_config: dict[str, Any],
    requested_shard_size: int,
    episode_sampling_rate: float,
) -> int:
    """Avoid empty shards in ShardedSingleStepDataset.

    With episode_sampling_rate=1.0, each episode contributes one shuffled segment.
    If shard_size is too small, total_steps/shard_size can exceed the number of
    available segments, leaving some shards empty and triggering an assertion.
    """
    episodes_path = prepared_dataset / "meta" / "episodes.jsonl"
    episodes = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    action_delta_indices = modality_config["action"].delta_indices
    action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1
    effective_lengths = [max(0, int(ep["length"]) - action_horizon + 1) for ep in episodes]
    total_steps = sum(effective_lengths)
    num_splits = max(1, int(1 / episode_sampling_rate))
    max_non_empty_shards = max(1, len(effective_lengths) * num_splits)
    requested_num_shards = math.ceil(total_steps / requested_shard_size)
    if requested_num_shards <= max_non_empty_shards:
        return requested_shard_size

    safe_size = math.ceil(total_steps / max_non_empty_shards)
    print(
        "[warn] requested shard_size would create empty shards: "
        f"total_steps={total_steps}, requested_shard_size={requested_shard_size}, "
        f"requested_shards={requested_num_shards}, max_non_empty_shards={max_non_empty_shards}. "
        f"Using shard_size={safe_size}.",
        flush=True,
    )
    return safe_size


def _checkpoint_step(checkpoint_dir: Path) -> int | None:
    match = re.match(r"^checkpoint-(\d+)$", checkpoint_dir.name)
    if match is None:
        return None
    return int(match.group(1))


def _training_output_dir(config: Any) -> Path:
    output_dir = Path(config.training.output_dir)
    if config.training.experiment_name is not None:
        output_dir = output_dir / str(config.training.experiment_name)
    return output_dir


def _latest_checkpoint(output_dir: Path) -> tuple[Path, int] | None:
    if not output_dir.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        step = _checkpoint_step(child)
        if step is None:
            continue
        if not (child / "trainer_state.json").exists():
            continue
        candidates.append((step, child))
    if not candidates:
        return None
    step, path = max(candidates, key=lambda item: item[0])
    return path, step


def _handle_auto_resume(args: argparse.Namespace, config: Any) -> bool:
    if not bool(args.auto_resume):
        print("[resume] auto-resume disabled; training will start from base model.", flush=True)
        return True

    output_dir = _training_output_dir(config)
    latest = _latest_checkpoint(output_dir)
    if latest is None:
        print(f"[resume] no existing checkpoint found in {output_dir}; starting fresh.", flush=True)
        return True

    checkpoint_path, checkpoint_step = latest
    target_steps = int(config.training.max_steps)
    if checkpoint_step >= target_steps:
        print(
            "[resume] latest checkpoint already reached target max_steps: "
            f"checkpoint={checkpoint_path}, step={checkpoint_step}, max_steps={target_steps}. "
            "Nothing to train.",
            flush=True,
        )
        return False

    print(
        "[resume] found latest checkpoint; training will resume until target max_steps: "
        f"checkpoint={checkpoint_path}, current_step={checkpoint_step}, max_steps={target_steps}, "
        f"remaining_steps={target_steps - checkpoint_step}",
        flush=True,
    )
    if bool(config.training.save_only_model):
        print(
            "[warn] --save-only-model is enabled. Future checkpoints may not contain optimizer/"
            "scheduler state, so resume may restart optimizer state even though model weights resume.",
            flush=True,
        )
    return True


def _disable_accelerate_deepspeed_import() -> None:
    """Avoid importing DeepSpeed on single-GPU runs.

    Accelerate's unwrap_model() imports DeepSpeed whenever the package is installed,
    even when this script is not using DeepSpeed. On minimal systems this can trigger
    Triton C extension compilation and fail if python-dev headers are absent.
    """
    try:
        import accelerate.utils.other as accelerate_other

        accelerate_other.is_deepspeed_available = lambda: False
        print("[env] disabled Accelerate DeepSpeed import for single-GPU training", flush=True)
    except Exception as exc:
        print(f"[warn] failed to disable Accelerate DeepSpeed import: {exc}", flush=True)


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.cuda_visible_devices))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from gr00t.data.embodiment_tags import EmbodimentTag

    base_model_path = Path(args.base_model_path).expanduser().resolve()
    vlm_model_path = Path(args.vlm_model_path).expanduser().resolve()
    source_dataset = Path(args.dataset_path).expanduser().resolve()
    prepared_dataset = Path(args.prepared_dataset_dir).expanduser().resolve()
    modality_config_path = Path(args.modality_config_path).expanduser().resolve()

    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")
    if not vlm_model_path.exists():
        raise FileNotFoundError(f"VLM model not found: {vlm_model_path}")

    _load_modality_config(modality_config_path)
    embodiment_tag = EmbodimentTag.resolve(str(args.embodiment_tag))
    prepared_dataset = _prepare_dataset(
        source_dataset=source_dataset,
        prepared_dataset=prepared_dataset,
        embodiment_tag=embodiment_tag,
        modality_config_path=modality_config_path,
        force_regenerate_stats=bool(args.force_regenerate_stats),
        instruction=str(args.instruction) if args.instruction else None,
    )
    _validate_prepared_loader(prepared_dataset, embodiment_tag)

    if args.prepare_only:
        print("[done] prepare-only requested; training was not started.", flush=True)
        return

    from gr00t.experiment.experiment import run

    if bool(args.disable_deepspeed_import):
        _disable_accelerate_deepspeed_import()

    config = _build_training_config(args, prepared_dataset, embodiment_tag)
    should_train = _handle_auto_resume(args, config)
    if not should_train:
        return

    print(
        "[train] starting overfit finetune: "
        f"output_dir={config.training.output_dir}, "
        f"max_steps={config.training.max_steps}, "
        f"global_batch_size={config.training.global_batch_size}, "
        f"gradient_accumulation_steps={config.training.gradient_accumulation_steps}, "
        f"video_backend={config.data.video_backend}",
        flush=True,
    )
    run(config)


if __name__ == "__main__":
    main()
