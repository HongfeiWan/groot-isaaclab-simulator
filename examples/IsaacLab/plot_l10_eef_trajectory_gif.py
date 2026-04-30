#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a 3D end-effector trajectory GIF from GR00T comparison output.

The input is the .npz file produced by compare_l10_gr00t_zero_shot_actions.py.
Only the first three action dimensions are used:
arm_eef_pos_target.x/y/z.

Example:

    .venv/bin/python examples/IsaacLab/plot_l10_eef_trajectory_gif.py \
        --input-npz outputs/IsaacLab/l10_zero_shot_action_compare/episode_000000.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_NPZ = (
    REPO_ROOT
    / "outputs"
    / "IsaacLab"
    / "l10_zero_shot_action_compare"
    / "episode_000000.npz"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=str, default=str(DEFAULT_INPUT_NPZ))
    parser.add_argument(
        "--output-gif",
        type=str,
        default=None,
        help="Defaults to <input>.eef_trajectory.gif.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth trajectory frame.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    return parser.parse_args()


def _load_eef_xyz(input_npz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not input_npz.exists():
        raise FileNotFoundError(f"Comparison npz not found: {input_npz}")

    data = np.load(input_npz, allow_pickle=True)
    required = {"gt_action", "pred_action", "frame_index"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"{input_npz} is missing keys: {sorted(missing)}")

    gt = np.asarray(data["gt_action"], dtype=np.float32)
    pred = np.asarray(data["pred_action"], dtype=np.float32)
    frames = np.asarray(data["frame_index"], dtype=np.int64)
    if gt.ndim != 2 or pred.ndim != 2 or gt.shape[1] < 3 or pred.shape[1] < 3:
        raise ValueError(f"Expected action arrays with shape (T, >=3), got {gt.shape}, {pred.shape}")

    n = min(len(gt), len(pred), len(frames))
    return frames[:n], gt[:n, :3], pred[:n, :3]


def _set_equal_xyz_limits(ax: plt.Axes, gt: np.ndarray, pred: np.ndarray) -> None:
    points = np.concatenate([gt, pred], axis=0)
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    if not np.isfinite(span) or span <= 1.0e-6:
        span = 0.1
    half = span * 0.58
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))


def _make_gif(
    *,
    output_gif: Path,
    frames: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    fps: int,
    dpi: int,
    elev: float,
    azim: float,
) -> None:
    output_gif.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    _set_equal_xyz_limits(ax, gt, pred)

    ax.set_title("L10 EEF Target Trajectory: Ground Truth vs Prediction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(True, alpha=0.25)

    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="#2563eb", alpha=0.16, linewidth=1.0)
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color="#dc2626", alpha=0.16, linewidth=1.0)

    (gt_line,) = ax.plot([], [], [], color="#2563eb", linewidth=2.4, label="ground truth")
    (pred_line,) = ax.plot([], [], [], color="#dc2626", linewidth=2.1, label="prediction")
    gt_point = ax.scatter([], [], [], color="#1d4ed8", s=36)
    pred_point = ax.scatter([], [], [], color="#b91c1c", s=36)
    frame_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    def update(i: int):
        stop = i + 1
        gt_line.set_data(gt[:stop, 0], gt[:stop, 1])
        gt_line.set_3d_properties(gt[:stop, 2])
        pred_line.set_data(pred[:stop, 0], pred[:stop, 1])
        pred_line.set_3d_properties(pred[:stop, 2])
        gt_point._offsets3d = ([gt[i, 0]], [gt[i, 1]], [gt[i, 2]])
        pred_point._offsets3d = ([pred[i, 0]], [pred[i, 1]], [pred[i, 2]])
        frame_text.set_text(f"frame: {int(frames[i])}    step: {i + 1}/{len(frames)}")
        return gt_line, pred_line, gt_point, pred_point, frame_text

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    animation.save(output_gif, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    input_npz = Path(args.input_npz).expanduser().resolve()
    output_gif = (
        Path(args.output_gif).expanduser().resolve()
        if args.output_gif
        else input_npz.with_suffix(".eef_trajectory.gif")
    )
    stride = max(1, int(args.stride))
    frames, gt, pred = _load_eef_xyz(input_npz)
    frames = frames[::stride]
    gt = gt[::stride]
    pred = pred[::stride]
    _make_gif(
        output_gif=output_gif,
        frames=frames,
        gt=gt,
        pred=pred,
        fps=int(args.fps),
        dpi=int(args.dpi),
        elev=float(args.elev),
        azim=float(args.azim),
    )
    print(f"Saved EEF trajectory GIF: {output_gif}")


if __name__ == "__main__":
    main()
