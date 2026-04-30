#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Frame-accurate LeRobot episode trim viewer.

Run with a lightweight data-cleaning environment, for example:

    streamlit run examples/IsaacLab/trim_lerobot_episode_viewer.py

Expected packages:

    pip install streamlit opencv-python pandas pyarrow

The tool is designed for LeRobot-v2 style datasets where each video frame maps
1:1 to one parquet row. It lets you inspect an episode by frame index, choose a
start/end frame, and export the selected range as a new episode in another
LeRobot dataset directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np
import pandas as pd


try:
    import streamlit as st
except ImportError:
    st = None


def _cache_data(**kwargs):
    if st is not None:
        return st.cache_data(**kwargs)

    def decorator(func):
        return func

    return decorator


DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2]
    / "demo_data"
    / "l10_hand"
    / "lerobot_rokae_xmate3_linker_l10_groot_v1"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "IsaacLab" / "trimmed_l10_dataset"


@dataclass(frozen=True)
class EpisodePaths:
    data_path: Path
    video_path: Path
    chunk_index: int
    video_original_key: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _resolve_episode_paths(dataset_dir: Path, episode_index: int, video_key: str) -> EpisodePaths:
    info = _read_json(dataset_dir / "meta" / "info.json")
    modality = _read_json(dataset_dir / "meta" / "modality.json")
    chunk_size = int(info.get("chunks_size", 1000))
    chunk_index = episode_index // chunk_size
    video_original_key = modality["video"][video_key].get(
        "original_key", f"observation.images.{video_key}"
    )
    data_path = dataset_dir / info["data_path"].format(
        episode_chunk=chunk_index,
        episode_index=episode_index,
    )
    video_path = dataset_dir / info["video_path"].format(
        episode_chunk=chunk_index,
        episode_index=episode_index,
        video_key=video_original_key,
    )
    return EpisodePaths(
        data_path=data_path,
        video_path=video_path,
        chunk_index=chunk_index,
        video_original_key=video_original_key,
    )


@_cache_data(show_spinner=False)
def _load_frame(video_path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


@_cache_data(show_spinner=False)
def _video_metadata(video_path: str) -> dict[str, int | float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    meta = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return meta


@_cache_data(show_spinner=False)
def _load_episode_table(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def _copy_base_meta(source_dataset: Path, output_dataset: Path) -> None:
    output_meta = output_dataset / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)
    for src in (source_dataset / "meta").iterdir():
        if src.is_file() and src.name not in {"episodes.jsonl", "info.json", "stats.json", "relative_stats.json"}:
            shutil.copy2(src, output_meta / src.name)
    if not (output_meta / "episodes.jsonl").exists():
        _write_jsonl(output_meta / "episodes.jsonl", [])
    if not (output_meta / "info.json").exists():
        info = _read_json(source_dataset / "meta" / "info.json")
        info["total_episodes"] = 0
        info["total_frames"] = 0
        info["total_videos"] = 0
        info["total_chunks"] = 0
        info["splits"] = {"train": "0:0"}
        _write_json(output_meta / "info.json", info)


def _next_episode_index(output_dataset: Path) -> int:
    rows = _read_jsonl(output_dataset / "meta" / "episodes.jsonl")
    if not rows:
        return 0
    return max(int(row["episode_index"]) for row in rows) + 1


def _current_total_frames(output_dataset: Path) -> int:
    rows = _read_jsonl(output_dataset / "meta" / "episodes.jsonl")
    return sum(int(row["length"]) for row in rows)


def _write_trimmed_video(
    *,
    source_video: Path,
    output_video: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    codec: str,
) -> None:
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {output_video} codec={codec}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    for frame_index in range(start_frame, end_frame + 1):
        ok, frame_bgr = cap.read()
        if not ok:
            writer.release()
            cap.release()
            raise RuntimeError(f"Could not read frame {frame_index} from {source_video}")
        writer.write(frame_bgr)

    writer.release()
    cap.release()


def _trim_dataframe(
    df: pd.DataFrame,
    *,
    start_frame: int,
    end_frame: int,
    new_episode_index: int,
    global_start_index: int,
    fps: float,
) -> pd.DataFrame:
    out = df.iloc[start_frame : end_frame + 1].copy().reset_index(drop=True)
    length = len(out)
    if "episode_index" in out.columns:
        out["episode_index"] = new_episode_index
    if "frame_index" in out.columns:
        out["frame_index"] = np.arange(length, dtype=np.int64)
    if "index" in out.columns:
        out["index"] = global_start_index + np.arange(length, dtype=np.int64)
    if "timestamp" in out.columns:
        out["timestamp"] = np.arange(length, dtype=np.float32) / float(fps)
    if "next.done" in out.columns:
        out["next.done"] = False
        out.loc[length - 1, "next.done"] = True
    return out


def _update_output_info(output_dataset: Path) -> None:
    info_path = output_dataset / "meta" / "info.json"
    info = _read_json(info_path)
    episodes = _read_jsonl(output_dataset / "meta" / "episodes.jsonl")
    total_episodes = len(episodes)
    total_frames = sum(int(row["length"]) for row in episodes)
    chunk_size = int(info.get("chunks_size", 1000))
    total_chunks = 0 if total_episodes == 0 else ((total_episodes - 1) // chunk_size + 1)
    num_video_keys = len(_read_json(output_dataset / "meta" / "modality.json").get("video", {}))

    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_videos"] = total_episodes * num_video_keys
    info["total_chunks"] = total_chunks
    info["splits"] = {"train": f"0:{total_episodes}"}
    _write_json(info_path, info)


def _append_trimmed_episode(
    *,
    source_dataset: Path,
    output_dataset: Path,
    source_episode_index: int,
    video_key: str,
    start_frame: int,
    end_frame: int,
    codec: str,
) -> dict[str, Any]:
    _copy_base_meta(source_dataset, output_dataset)

    source_info = _read_json(source_dataset / "meta" / "info.json")
    output_info = _read_json(output_dataset / "meta" / "info.json")
    source_episodes = _read_jsonl(source_dataset / "meta" / "episodes.jsonl")
    source_ep = next(
        row for row in source_episodes if int(row["episode_index"]) == int(source_episode_index)
    )

    paths = _resolve_episode_paths(source_dataset, source_episode_index, video_key)
    df = _load_episode_table(str(paths.data_path))
    fps = float(source_info.get("fps", 10))
    length = int(end_frame - start_frame + 1)

    new_episode_index = _next_episode_index(output_dataset)
    global_start_index = _current_total_frames(output_dataset)
    chunk_size = int(output_info.get("chunks_size", 1000))
    new_chunk_index = new_episode_index // chunk_size

    out_df = _trim_dataframe(
        df,
        start_frame=start_frame,
        end_frame=end_frame,
        new_episode_index=new_episode_index,
        global_start_index=global_start_index,
        fps=fps,
    )

    output_data_path = output_dataset / output_info["data_path"].format(
        episode_chunk=new_chunk_index,
        episode_index=new_episode_index,
    )
    output_video_path = output_dataset / output_info["video_path"].format(
        episode_chunk=new_chunk_index,
        episode_index=new_episode_index,
        video_key=paths.video_original_key,
    )
    output_data_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_data_path, index=False)
    _write_trimmed_video(
        source_video=paths.video_path,
        output_video=output_video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        fps=fps,
        codec=codec,
    )

    episode_row = dict(source_ep)
    episode_row["episode_index"] = new_episode_index
    episode_row["length"] = length
    if "teleop_stack_metadata" in episode_row:
        metadata = dict(episode_row["teleop_stack_metadata"])
        metadata["source_episode_index"] = source_episode_index
        metadata["source_start_frame"] = start_frame
        metadata["source_end_frame"] = end_frame
        metadata["data_path"] = str(output_data_path.relative_to(output_dataset))
        metadata["video_path"] = str(output_video_path.relative_to(output_dataset))
        episode_row["teleop_stack_metadata"] = metadata

    episodes = _read_jsonl(output_dataset / "meta" / "episodes.jsonl")
    episodes.append(episode_row)
    _write_jsonl(output_dataset / "meta" / "episodes.jsonl", episodes)
    _update_output_info(output_dataset)

    return {
        "new_episode_index": new_episode_index,
        "length": length,
        "data_path": str(output_data_path),
        "video_path": str(output_video_path),
    }


def _format_vector(value: Any, max_items: int = 6) -> str:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return str(value)
    shown = ", ".join(f"{x:.4f}" for x in arr[:max_items])
    suffix = ", ..." if arr.size > max_items else ""
    return f"[{shown}{suffix}] shape={arr.shape}"


def main() -> None:
    if st is None:
        raise SystemExit(
            "streamlit is not installed. Install it with: pip install streamlit opencv-python pandas pyarrow"
        )

    st.set_page_config(page_title="LeRobot Episode Trim Viewer", layout="wide")
    st.title("LeRobot Episode Trim Viewer")

    with st.sidebar:
        dataset_dir = Path(
            st.text_input("Dataset directory", value=str(DEFAULT_DATASET))
        ).expanduser()
        output_dir = Path(st.text_input("Output dataset directory", value=str(DEFAULT_OUTPUT))).expanduser()
        codec = st.selectbox("Output mp4 codec", ["mp4v", "avc1", "H264"], index=0)

        if not (dataset_dir / "meta" / "info.json").exists():
            st.error("Dataset meta/info.json not found.")
            st.stop()

        info = _read_json(dataset_dir / "meta" / "info.json")
        modality = _read_json(dataset_dir / "meta" / "modality.json")
        episodes = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
        episode_ids = [int(row["episode_index"]) for row in episodes]
        video_keys = list(modality.get("video", {}).keys())

        selected_episode = st.selectbox("Episode", episode_ids, index=0)
        selected_video_key = st.selectbox("Video key", video_keys, index=0)
        st.caption(f"fps={info.get('fps', 'unknown')} total_episodes={len(episodes)}")

    paths = _resolve_episode_paths(dataset_dir, int(selected_episode), selected_video_key)
    if not paths.data_path.exists():
        st.error(f"Parquet not found: {paths.data_path}")
        st.stop()
    if not paths.video_path.exists():
        st.error(f"Video not found: {paths.video_path}")
        st.stop()

    df = _load_episode_table(str(paths.data_path))
    video_meta = _video_metadata(str(paths.video_path))
    frame_count = min(len(df), int(video_meta["frame_count"]))
    if frame_count <= 0:
        st.error("No frames found.")
        st.stop()

    if "trim_start" not in st.session_state:
        st.session_state.trim_start = 0
    if "trim_end" not in st.session_state:
        st.session_state.trim_end = frame_count - 1
    if "frame_index" not in st.session_state:
        st.session_state.frame_index = 0
    st.session_state.frame_index = min(max(0, int(st.session_state.frame_index)), frame_count - 1)
    st.session_state.trim_start = min(max(0, int(st.session_state.trim_start)), frame_count - 1)
    st.session_state.trim_end = min(max(0, int(st.session_state.trim_end)), frame_count - 1)
    if st.session_state.trim_start > st.session_state.trim_end:
        st.session_state.trim_end = st.session_state.trim_start

    left, right = st.columns([1.25, 1.0])
    with left:
        st.subheader("Video Preview")
        st.video(str(paths.video_path))
        st.caption(
            "Use the exact frame slider below for frame-accurate start/end selection. "
            "The native video player is only for quick visual playback."
        )

    with right:
        st.subheader("Exact Frame")
        col_prev, col_slider, col_next = st.columns([0.12, 0.76, 0.12])
        with col_prev:
            if st.button("<", use_container_width=True):
                st.session_state.frame_index = max(0, int(st.session_state.frame_index) - 1)
        with col_next:
            if st.button(">", use_container_width=True):
                st.session_state.frame_index = min(frame_count - 1, int(st.session_state.frame_index) + 1)
        with col_slider:
            st.session_state.frame_index = st.slider(
                "Frame index",
                min_value=0,
                max_value=frame_count - 1,
                value=int(st.session_state.frame_index),
                step=1,
            )

        frame = _load_frame(str(paths.video_path), int(st.session_state.frame_index))
        st.image(frame, channels="RGB", use_container_width=True)

        row = df.iloc[int(st.session_state.frame_index)]
        st.write(
            {
                "episode": int(selected_episode),
                "frame": int(st.session_state.frame_index),
                "timestamp": float(row.get("timestamp", st.session_state.frame_index / float(info.get("fps", 10)))),
                "parquet_rows": int(len(df)),
                "video_frames": int(video_meta["frame_count"]),
            }
        )
        with st.expander("Current row preview"):
            if "observation.state" in row:
                st.text(f"state:  {_format_vector(row['observation.state'])}")
            if "action" in row:
                st.text(f"action: {_format_vector(row['action'])}")
            st.json(
                {
                    key: row[key].item() if hasattr(row[key], "item") else row[key]
                    for key in row.index
                    if key not in {"observation.state", "action"}
                }
            )

    st.divider()
    st.subheader("Trim Range")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Set Start = Current Frame", use_container_width=True):
            st.session_state.trim_start = int(st.session_state.frame_index)
            if st.session_state.trim_start > st.session_state.trim_end:
                st.session_state.trim_end = st.session_state.trim_start
    with c2:
        if st.button("Set End = Current Frame", use_container_width=True):
            st.session_state.trim_end = int(st.session_state.frame_index)
            if st.session_state.trim_end < st.session_state.trim_start:
                st.session_state.trim_start = st.session_state.trim_end
    with c3:
        st.session_state.trim_start = st.number_input(
            "Start frame", min_value=0, max_value=frame_count - 1, value=int(st.session_state.trim_start)
        )
    with c4:
        st.session_state.trim_end = st.number_input(
            "End frame", min_value=0, max_value=frame_count - 1, value=int(st.session_state.trim_end)
        )

    start_frame = int(st.session_state.trim_start)
    end_frame = int(st.session_state.trim_end)
    if start_frame > end_frame:
        st.error("Start frame must be <= end frame.")
        st.stop()

    selected_len = end_frame - start_frame + 1
    st.info(
        f"Selected frames: {start_frame}..{end_frame} inclusive, "
        f"length={selected_len}, duration={selected_len / float(info.get('fps', 10)):.2f}s"
    )

    export_col, preview_col = st.columns([0.25, 0.75])
    with export_col:
        if st.button("Export Trimmed Episode", type="primary", use_container_width=True):
            with st.spinner("Writing trimmed parquet and mp4..."):
                result = _append_trimmed_episode(
                    source_dataset=dataset_dir,
                    output_dataset=output_dir,
                    source_episode_index=int(selected_episode),
                    video_key=selected_video_key,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    codec=codec,
                )
            st.success(f"Exported episode {result['new_episode_index']} length={result['length']}")
            st.code(json.dumps(result, indent=2, ensure_ascii=False), language="json")
    with preview_col:
        st.caption(f"Output dataset: {output_dir}")


if __name__ == "__main__":
    main()
