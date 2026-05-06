#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Continue L10 overfit fine-tuning on the trimmed LeRobot dataset.

This is a thin convenience launcher over finetune_l10_overfit.py. It keeps the
same training recipe and checkpoint directory, but changes the defaults to:

- dataset: outputs/IsaacLab/trimmed_l10_dataset
- prepared dataset cache: outputs/IsaacLab/trimmed_l10_overfit_dataset
- output dir: checkpoints/rokae_xmate3_l10_overfit
- max steps: 20000

Because --auto-resume is enabled by default in finetune_l10_overfit.py, this
continues from the numerically latest checkpoint-* in the output directory
instead of starting over.

Example:

    .venv/bin/python examples/IsaacLab/finetune_l10_overfit_on_trimmed_dataset.py

To only validate/prepare the trimmed dataset:

    .venv/bin/python examples/IsaacLab/finetune_l10_overfit_on_trimmed_dataset.py \
        --prepare-only
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SCRIPT = REPO_ROOT / "examples" / "IsaacLab" / "finetune_l10_overfit.py"
DEFAULT_TRIMMED_DATASET_DIR = REPO_ROOT / "outputs" / "IsaacLab" / "trimmed_l10_dataset"
DEFAULT_PREPARED_DATASET_DIR = (
    REPO_ROOT / "outputs" / "IsaacLab" / "trimmed_l10_overfit_dataset"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "checkpoints" / "rokae_xmate3_l10_overfit"


def _has_option(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def _add_default(argv: list[str], option: str, value: str) -> None:
    if not _has_option(argv, option):
        argv.extend([option, value])


def _add_trimmed_defaults() -> None:
    argv = sys.argv[1:]
    _add_default(argv, "--dataset-path", str(DEFAULT_TRIMMED_DATASET_DIR))
    _add_default(argv, "--prepared-dataset-dir", str(DEFAULT_PREPARED_DATASET_DIR))
    _add_default(argv, "--output-dir", str(DEFAULT_OUTPUT_DIR))
    _add_default(argv, "--max-steps", "20000")
    sys.argv = [sys.argv[0], *argv]


def _load_source_module():
    spec = importlib.util.spec_from_file_location("finetune_l10_overfit", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import source finetune script: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    _add_trimmed_defaults()
    module = _load_source_module()
    module.main()


if __name__ == "__main__":
    main()
