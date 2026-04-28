# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prototype Isaac Lab <-> GR00T closed-loop adapter.

This script is intentionally narrow: one Isaac Lab environment, one task
instruction, and one robot. It validates and prints every GR00T observation and
action shape so the first integration pass can focus on interface alignment.

Run it with Isaac Lab's Python, for example:

    ${ISAACLAB_PATH}/isaaclab.sh -p examples/IsaacLab/isaaclab_gr00t_adapter.py \
        --task Isaac-Lift-Cube-Franka-v0 \
        --policy-host localhost \
        --policy-port 5555 \
        --instruction "pick up the cube" \
        --inject-franka-cameras \
        --enable_cameras \
        --external-camera-sensor camera \
        --wrist-camera-sensor wrist_camera \
        --robot-asset-name robot \
        --eef-body-name panda_hand

Start the GR00T server separately, for example:

    python gr00t/eval/run_gr00t_server.py \
        --model-path checkpoints/GR00T-N1.7-3B \
        --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
        --port 5555
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gr00t.policy.server_client import PolicyClient  # noqa: E402


DEFAULT_IMAGE_SIZE = (180, 320)
DROID_EEF_ROTATION_CORRECT = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    dtype=np.float64,
)

DEFAULT_EXTERNAL_CAM_POS = (1.0, 0.0, 0.4)
DEFAULT_EXTERNAL_CAM_ROT = (0.35355, -0.61237, -0.61237, 0.35355)
DEFAULT_WRIST_CAM_POS = (0.13, 0.0, -0.15)
DEFAULT_WRIST_CAM_ROT = (-0.70614, 0.03701, 0.03701, -0.70614)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors, lists, and arrays into a CPU numpy array."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _lookup_path(container: Any, path: str | None) -> Any:
    """Read a nested value from a dict/object using a dotted path."""
    if not path:
        raise KeyError("empty lookup path")

    if isinstance(container, dict) and path in container:
        return container[path]

    current = container
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(path)
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def _resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[-3:-1] == (height, width):
        return image

    original_shape = image.shape
    image = image.reshape(-1, *original_shape[-3:])
    resized = [_resize_one_with_pad(Image.fromarray(frame), height, width) for frame in image]
    return np.stack(resized).reshape(*original_shape[:-3], height, width, original_shape[-1])


def _resize_one_with_pad(image: Image.Image, height: int, width: int) -> np.ndarray:
    cur_width, cur_height = image.size
    ratio = max(cur_width / width, cur_height / height)
    resized_width = int(cur_width / ratio)
    resized_height = int(cur_height / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=Image.BILINEAR)

    output = Image.new(resized_image.mode, (width, height), 0)
    pad_width = max(0, int((width - resized_width) / 2))
    pad_height = max(0, int((height - resized_height) / 2))
    output.paste(resized_image, (pad_width, pad_height))
    return np.asarray(output)


def _parse_floats_csv(value: str | None, *, count: int, name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != count:
        raise ValueError(f"--{name} expects {count} comma-separated floats, got: {value!r}")
    return tuple(float(p) for p in parts)


class _CameraPreview:
    def __init__(self, enabled: bool, every: int):
        self.enabled = bool(enabled)
        self.every = max(1, int(every))
        self._backend = None
        self._cv2 = None

        if not self.enabled:
            return
        try:
            import cv2  # type: ignore

            self._cv2 = cv2
            self._backend = "cv2"
        except Exception:
            self.enabled = False
            print(
                "Camera preview disabled: OpenCV (cv2) is not available in this Isaac Lab env. "
                "Install opencv-python if you want --preview-cameras.",
                flush=True,
            )

    def show(self, step: int, *, exterior: np.ndarray, wrist: np.ndarray) -> None:
        if not self.enabled or (step % self.every != 0):
            return
        assert self._backend == "cv2" and self._cv2 is not None

        cv2 = self._cv2
        # Input is HWC RGB uint8; OpenCV expects BGR for correct colors.
        ext_bgr = cv2.cvtColor(exterior, cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)
        cv2.imshow("IsaacLab camera: exterior", ext_bgr)
        cv2.imshow("IsaacLab camera: wrist", wrist_bgr)
        cv2.waitKey(1)


def _maybe_lookup_sensor_prim_path(env: Any, sensor_name: str) -> str | None:
    """Best-effort resolve a camera sensor's USD prim path."""
    try:
        sensor = env.unwrapped.scene[sensor_name]
    except Exception:
        return None

    for attr in ("prim_path", "_prim_path", "_sensor_prim_path", "sensor_prim_path"):
        if hasattr(sensor, attr):
            value = getattr(sensor, attr)
            if isinstance(value, str) and value:
                return value

    cfg = getattr(sensor, "cfg", None)
    prim_path = getattr(cfg, "prim_path", None) if cfg is not None else None
    if isinstance(prim_path, str) and prim_path:
        return prim_path
    return None


def _resolve_stage_prim_path(
    template_path: str | None, *, env_index: int, suffix_fallback: str | None
) -> str | None:
    """Resolve camera prim path to an existing USD prim path.

    Isaac Lab sensor configs often use template paths like `{ENV_REGEX_NS}/...`.
    Viewports need a concrete prim path that exists in the current USD stage.
    """
    if not template_path and not suffix_fallback:
        return None

    try:
        import omni.usd  # type: ignore
    except Exception:
        return template_path

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return template_path

    candidates: list[str] = []
    if template_path:
        expanded = template_path.replace(
            "{ENV_REGEX_NS}", f"/World/envs/env_{int(env_index)}"
        ).replace("{ENV_NS}", f"/World/envs/env_{int(env_index)}")
        candidates.append(expanded)
        candidates.append(template_path)

    for path in candidates:
        try:
            prim = stage.GetPrimAtPath(path)
        except Exception:
            prim = None
        if prim is not None and prim.IsValid():
            return path

    # Fallback: search by suffix (e.g. "/table_cam" or "/wrist_cam").
    if suffix_fallback:
        suffix = suffix_fallback if suffix_fallback.startswith("/") else f"/{suffix_fallback}"
        try:
            for prim in stage.Traverse():
                prim_path = str(prim.GetPath())
                if prim_path.endswith(suffix):
                    return prim_path
        except Exception:
            return None

    return None


class _IsaacSimViewportPreview:
    """Preview cameras in Isaac Sim viewports (no OpenCV dependency)."""

    def __init__(self, env: Any, args: argparse.Namespace):
        self.enabled = bool(args.viewport_preview)
        self._viewports: list[Any] = []

        if not self.enabled:
            return

        try:
            from omni.kit.viewport.utility import create_viewport_window  # type: ignore
        except Exception:
            self.enabled = False
            print(
                "Viewport preview disabled: omni.kit.viewport is not available. "
                "Make sure you run inside Isaac Sim / IsaacLab.",
                flush=True,
            )
            return

        exterior_path = (
            _maybe_lookup_sensor_prim_path(env, args.external_camera_sensor)
            if args.external_camera_sensor
            else None
        )
        wrist_path = (
            _maybe_lookup_sensor_prim_path(env, args.wrist_camera_sensor)
            if args.wrist_camera_sensor
            else None
        )
        exterior_path = _resolve_stage_prim_path(
            exterior_path, env_index=args.viewport_env_index, suffix_fallback="table_cam"
        )
        wrist_path = _resolve_stage_prim_path(
            wrist_path, env_index=args.viewport_env_index, suffix_fallback="wrist_cam"
        )

        def _get_viewport_api(viewport_window: Any) -> Any | None:
            """Best-effort unwrap viewport window -> viewport API."""
            for attr in ("viewport_api", "_viewport_api"):
                api = getattr(viewport_window, attr, None)
                if api is not None:
                    return api
            getter = getattr(viewport_window, "get_viewport_api", None)
            if callable(getter):
                try:
                    api = getter()
                except Exception:
                    api = None
                if api is not None:
                    return api
            return None

        def _set_title(viewport_window: Any, title: str) -> None:
            for attr in ("window_title", "title", "name"):
                if hasattr(viewport_window, attr):
                    try:
                        setattr(viewport_window, attr, title)
                        return
                    except Exception:
                        pass
            setter = getattr(viewport_window, "set_window_title", None)
            if callable(setter):
                try:
                    setter(title)
                except Exception:
                    pass

        def _bind(viewport_window: Any, camera_path: str, *, title: str) -> bool:
            """Return True if binding succeeded."""
            _set_title(viewport_window, title)

            # Some Isaac Sim versions expose camera binding on the window object.
            for method in ("set_active_camera", "set_active_camera_path", "set_camera_path"):
                fn = getattr(viewport_window, method, None)
                if callable(fn):
                    try:
                        fn(camera_path)
                        return True
                    except Exception:
                        pass

            # Otherwise try the viewport API object.
            api = _get_viewport_api(viewport_window)
            if api is not None:
                for method in ("set_active_camera", "set_active_camera_path", "set_camera_path"):
                    fn = getattr(api, method, None)
                    if callable(fn):
                        try:
                            fn(camera_path)
                            return True
                        except Exception:
                            pass
                # Some variants expose a property.
                for attr in ("camera_path", "active_camera_path"):
                    if hasattr(api, attr):
                        try:
                            setattr(api, attr, camera_path)
                            return True
                        except Exception:
                            pass

            return False

        if args.viewport_show_exterior and exterior_path:
            vp = create_viewport_window(
                window_name="GR00T Exterior Camera",
                width=args.viewport_width,
                height=args.viewport_height,
            )
            if _bind(vp, exterior_path, title="GR00T Exterior Camera"):
                self._viewports.append(vp)
                print(f"Viewport preview bound: exterior -> {exterior_path}", flush=True)
            else:
                print(
                    "Viewport preview: could not bind exterior camera to viewport (API mismatch).",
                    flush=True,
                )
        elif args.viewport_show_exterior:
            print(
                "Viewport preview: exterior camera prim path not found; skipping exterior viewport.",
                flush=True,
            )

        if args.viewport_show_wrist and wrist_path:
            vp = create_viewport_window(
                window_name="GR00T Wrist Camera",
                width=args.viewport_width,
                height=args.viewport_height,
            )
            if _bind(vp, wrist_path, title="GR00T Wrist Camera"):
                self._viewports.append(vp)
                print(f"Viewport preview bound: wrist -> {wrist_path}", flush=True)
            else:
                print(
                    "Viewport preview: could not bind wrist camera to viewport (API mismatch).",
                    flush=True,
                )
        elif args.viewport_show_wrist:
            print(
                "Viewport preview: wrist camera prim path not found; skipping wrist viewport.",
                flush=True,
            )

        if not self._viewports:
            print(
                "Viewport preview requested, but no cameras could be bound. "
                "Ensure cameras are enabled and sensor names match.",
                flush=True,
            )

    def close(self) -> None:
        # Viewport windows are owned by Isaac Sim UI; allow them to be GC'd.
        self._viewports.clear()


def _as_hwc_uint8(value: Any, *, image_size: tuple[int, int]) -> np.ndarray:
    image = _to_numpy(value)

    while image.ndim > 3:
        image = image[0]
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dims, got shape {image.shape}")

    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected image channel count 3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 0.0
        if max_value <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255)

    image = image.astype(np.uint8, copy=False)
    return _resize_with_pad(image, image_size[0], image_size[1])


def _flatten_float32(value: Any) -> np.ndarray:
    return _to_numpy(value).astype(np.float32, copy=False).reshape(-1)


def _quat_to_rot6d(quat: np.ndarray, quat_order: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    if quat_order == "wxyz":
        quat = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
    elif quat_order != "xyzw":
        raise ValueError(f"Unsupported quaternion order: {quat_order}")

    rot_mat = Rotation.from_quat(quat).as_matrix() @ DROID_EEF_ROTATION_CORRECT
    return rot_mat[:2, :].reshape(6).astype(np.float32)


def _make_eef_9d(position: Any, quat: Any, quat_order: str) -> np.ndarray:
    xyz = _flatten_float32(position)[:3]
    rot6d = _quat_to_rot6d(_flatten_float32(quat)[:4], quat_order)
    return np.concatenate([xyz, rot6d]).astype(np.float32)


def _describe_array(name: str, value: Any) -> str:
    arr = _to_numpy(value) if not isinstance(value, str) else np.asarray(value)
    dtype = getattr(arr, "dtype", type(value))
    shape = getattr(arr, "shape", ())
    if np.issubdtype(arr.dtype, np.number) and arr.size:
        return f"{name}: shape={shape}, dtype={dtype}, min={arr.min():.4g}, max={arr.max():.4g}"
    return f"{name}: shape={shape}, dtype={dtype}"


def _print_observation_summary(observation: dict[str, Any], step: int) -> None:
    print(f"\n[step {step}] GR00T observation")
    for modality in ("video", "state", "language"):
        for key, value in observation[modality].items():
            if modality == "language":
                print(f"  language.{key}: {value}")
            else:
                print(f"  {_describe_array(f'{modality}.{key}', value)}")


def _print_action_summary(action: dict[str, np.ndarray], env_action: np.ndarray, step: int) -> None:
    print(f"[step {step}] GR00T action")
    for key, value in action.items():
        print(f"  {_describe_array(key, value)}")
    print(f"  {_describe_array('env_action', env_action)}")


def _scene_keys(env: Any) -> list[str]:
    scene = env.unwrapped.scene
    if hasattr(scene, "keys"):
        return sorted(str(key) for key in scene.keys())
    if hasattr(scene, "_entities"):
        return sorted(str(key) for key in scene._entities.keys())
    return []


class IsaacLabGr00tAdapter:
    def __init__(self, env: Any, policy: PolicyClient, args: argparse.Namespace):
        self.env = env
        self.policy = policy
        self.args = args
        self.modality_config = policy.get_modality_config()

        video_delta = self.modality_config["video"].delta_indices
        self.video_history_len = max([abs(i) for i in video_delta] + [0]) + 1
        self.frame_buffer: deque[dict[str, np.ndarray]] = deque(maxlen=self.video_history_len)
        self.action_chunk: np.ndarray | None = None
        self.action_index = 0
        self.preview = _CameraPreview(args.preview_cameras, args.preview_every)
        self.viewport_preview = _IsaacSimViewportPreview(env, args)

        print("Connected to GR00T policy server.")
        self._print_modality_config()
        print(f"\nIsaac Lab scene keys: {_scene_keys(self.env)}")

    def _print_modality_config(self) -> None:
        print("\nGR00T modality config")
        for modality, config in self.modality_config.items():
            print(
                f"  {modality}: keys={config.modality_keys}, delta_indices={config.delta_indices}"
            )

    def build_observation(self, env_obs: Any) -> dict[str, Any]:
        exterior_image, wrist_image = self._extract_images(env_obs)
        self.frame_buffer.append({"exterior": exterior_image, "wrist": wrist_image})
        # Use the caller's step counter for stable preview timing.
        # (The frame buffer index depends on modality history and resets.)

        video_dict = self._build_video_dict()
        state_dict = self._build_state_dict(env_obs)
        language_key = self.modality_config["language"].modality_keys[0]

        return {
            "video": video_dict,
            "state": state_dict,
            "language": {language_key: [[self.args.instruction]]},
        }

    def _extract_images(self, env_obs: Any) -> tuple[np.ndarray, np.ndarray]:
        image_size = (self.args.image_height, self.args.image_width)
        exterior = self._extract_image(
            env_obs,
            obs_key=self.args.external_image_key,
            sensor_name=self.args.external_camera_sensor,
            output_name=self.args.external_camera_output,
            image_size=image_size,
        )
        wrist = self._extract_image(
            env_obs,
            obs_key=self.args.wrist_image_key,
            sensor_name=self.args.wrist_camera_sensor,
            output_name=self.args.wrist_camera_output,
            image_size=image_size,
            fallback=exterior,
        )
        return exterior, wrist

    def _extract_image(
        self,
        env_obs: Any,
        *,
        obs_key: str | None,
        sensor_name: str | None,
        output_name: str,
        image_size: tuple[int, int],
        fallback: np.ndarray | None = None,
    ) -> np.ndarray:
        if obs_key:
            return _as_hwc_uint8(_lookup_path(env_obs, obs_key), image_size=image_size)

        if sensor_name:
            try:
                sensor = self.env.unwrapped.scene[sensor_name]
            except Exception as exc:
                raise KeyError(
                    f"Camera sensor '{sensor_name}' was not found. Available scene keys: "
                    f"{_scene_keys(self.env)}. If this task has cameras, also pass "
                    "--enable_cameras to isaaclab.sh."
                ) from exc
            return _as_hwc_uint8(sensor.data.output[output_name], image_size=image_size)

        if fallback is not None:
            return fallback.copy()

        raise ValueError(
            "No image source configured. Pass --external-image-key or "
            "--external-camera-sensor for the exterior image."
        )

    def _build_video_dict(self) -> dict[str, np.ndarray]:
        frames = self._select_history_frames()
        video_keys = self.modality_config["video"].modality_keys
        video_dict: dict[str, np.ndarray] = {}

        for key in video_keys:
            if "wrist" in key:
                stack = np.stack([frame["wrist"] for frame in frames], axis=0)
            else:
                stack = np.stack([frame["exterior"] for frame in frames], axis=0)
            video_dict[key] = stack[None, ...].astype(np.uint8, copy=False)
        return video_dict

    def _select_history_frames(self) -> list[dict[str, np.ndarray]]:
        if not self.frame_buffer:
            raise RuntimeError("Frame buffer is empty.")

        delta_indices = self.modality_config["video"].delta_indices
        buffer = list(self.frame_buffer)
        selected = []
        for delta in delta_indices:
            if delta == 0:
                selected.append(buffer[-1])
            else:
                selected.append(buffer[max(delta, -len(buffer))])
        return selected

    def _build_state_dict(self, env_obs: Any) -> dict[str, np.ndarray]:
        source = self._extract_state_source(env_obs)
        state_dict: dict[str, np.ndarray] = {}
        for key in self.modality_config["state"].modality_keys:
            if key not in source:
                raise KeyError(
                    f"Cannot build GR00T state '{key}'. Available state keys: {sorted(source)}"
                )
            state_dict[key] = source[key][None, None, ...].astype(np.float32, copy=False)
        return state_dict

    def _extract_state_source(self, env_obs: Any) -> dict[str, np.ndarray]:
        source: dict[str, np.ndarray] = {}

        if self.args.eef_9d_key:
            source["eef_9d"] = _flatten_float32(_lookup_path(env_obs, self.args.eef_9d_key))[:9]
        elif self.args.eef_position_key and self.args.eef_quat_key:
            source["eef_9d"] = _make_eef_9d(
                _lookup_path(env_obs, self.args.eef_position_key),
                _lookup_path(env_obs, self.args.eef_quat_key),
                self.args.quat_order,
            )

        if self.args.joint_position_key:
            source["joint_position"] = _flatten_float32(
                _lookup_path(env_obs, self.args.joint_position_key)
            )
        if self.args.gripper_position_key:
            source["gripper_position"] = _flatten_float32(
                _lookup_path(env_obs, self.args.gripper_position_key)
            )

        if self.args.robot_asset_name:
            source.update(self._extract_robot_asset_state())

        return source

    def _extract_robot_asset_state(self) -> dict[str, np.ndarray]:
        try:
            robot = self.env.unwrapped.scene[self.args.robot_asset_name]
        except Exception as exc:
            raise KeyError(
                f"Robot asset '{self.args.robot_asset_name}' was not found. Available scene keys: "
                f"{_scene_keys(self.env)}."
            ) from exc
        data = robot.data
        source: dict[str, np.ndarray] = {}

        joint_pos = _to_numpy(data.joint_pos)[0].astype(np.float32, copy=False)
        # GR00T's Franka/DROID-style joint_position modality covers the 7 arm joints.
        # Franka's two finger joints are represented separately as gripper_position.
        source.setdefault("joint_position", joint_pos[: self.args.arm_state_dim].reshape(-1))

        if self.args.gripper_joint_indices:
            indices = [int(i) for i in self.args.gripper_joint_indices.split(",")]
            source.setdefault("gripper_position", joint_pos[indices].reshape(-1))
        else:
            # GR00T's DROID-style gripper modality is a single scalar.  Franka exposes two
            # mirrored finger joints, so use one finger by default instead of sending both.
            source.setdefault("gripper_position", joint_pos[-2:-1].reshape(-1))

        if self.args.eef_body_name:
            body_names = list(getattr(robot, "body_names", []))
            if self.args.eef_body_name not in body_names:
                raise ValueError(
                    f"EEF body '{self.args.eef_body_name}' not found in robot bodies: {body_names}"
                )
            body_idx = body_names.index(self.args.eef_body_name)
            body_pos = _to_numpy(data.body_pos_w)[0, body_idx]
            body_quat = _to_numpy(data.body_quat_w)[0, body_idx]
            source.setdefault("eef_9d", _make_eef_9d(body_pos, body_quat, "wxyz"))

        return source

    def get_env_action(self, env_obs: Any, step: int) -> np.ndarray:
        if self.action_chunk is None or self.action_index >= self.args.replan_horizon:
            observation = self.build_observation(env_obs)
            if step % self.args.print_every == 0:
                _print_observation_summary(observation, step)

            request_start = time.perf_counter()
            print(f"[step {step}] Requesting GR00T action...", flush=True)
            action, _ = self.policy.get_action(observation)
            request_elapsed = time.perf_counter() - request_start
            print(f"[step {step}] GR00T action received in {request_elapsed:.2f}s", flush=True)
            self.action_chunk = self._action_dict_to_env_chunk(action)
            self.action_index = 0

            if step % self.args.print_every == 0:
                _print_action_summary(action, self.action_chunk, step)

        assert self.action_chunk is not None
        env_action = self.action_chunk[min(self.action_index, len(self.action_chunk) - 1)]
        self.action_index += 1
        return env_action[None, :].astype(np.float32, copy=False)

    def _action_dict_to_env_chunk(self, action: dict[str, np.ndarray]) -> np.ndarray:
        action_keys = [
            key.strip() for key in self.args.action_keys_to_env.split(",") if key.strip()
        ]
        if not action_keys:
            action_keys = self.modality_config["action"].modality_keys

        if action_keys == ["joint_position", "gripper_position"]:
            if "joint_position" not in action or "gripper_position" not in action:
                raise KeyError(
                    f"Expected joint and gripper actions in GR00T response: {sorted(action)}"
                )
            joint_action = action["joint_position"][0][..., : self.args.arm_action_dim]
            gripper_action = action["gripper_position"][0][..., :1]
            chunk = np.concatenate([joint_action, gripper_action], axis=-1).astype(
                np.float32, copy=False
            )
            return self._fit_action_space(chunk)

        chunks = []
        for key in action_keys:
            if key not in action:
                raise KeyError(f"Action key '{key}' not in GR00T response: {sorted(action)}")
            chunks.append(action[key][0])

        chunk = np.concatenate(chunks, axis=-1).astype(np.float32, copy=False)
        return self._fit_action_space(chunk)

    def _fit_action_space(self, chunk: np.ndarray) -> np.ndarray:
        action_shape = getattr(self.env.action_space, "shape", None)
        if action_shape is None:
            raise ValueError("Environment action_space has no shape; cannot build action tensor.")

        target_dim = int(np.prod(action_shape))
        if len(action_shape) > 1 and action_shape[0] == 1:
            target_dim = int(np.prod(action_shape[1:]))

        fitted = np.zeros((chunk.shape[0], target_dim), dtype=np.float32)
        copy_dim = min(target_dim, chunk.shape[-1])
        fitted[:, :copy_dim] = chunk[:, :copy_dim]
        if copy_dim != chunk.shape[-1]:
            print(
                f"Warning: GR00T action dim {chunk.shape[-1]} does not match env action dim "
                f"{target_dim}; copied first {copy_dim} values."
            )
        return fitted


def _add_gr00t_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", type=str, required=True, help="Isaac Lab task id.")
    parser.add_argument("--policy-host", type=str, default="localhost")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--policy-api-token", type=str, default=None)
    parser.add_argument(
        "--policy-timeout-ms",
        type=int,
        default=120000,
        help="ZMQ timeout for GR00T policy calls. First GPU inference can exceed 15s.",
    )
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--replan-horizon", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_SIZE[1])
    parser.add_argument(
        "--inject-franka-cameras",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Inject a table camera and a wrist camera into Franka manager-based tasks before "
            "gym.make(). Use with Isaac Lab's --enable_cameras flag."
        ),
    )
    parser.add_argument(
        "--external-cam-offset-pos",
        type=str,
        default=None,
        help='Override table camera offset position as "x,y,z" (meters).',
    )
    parser.add_argument(
        "--external-cam-offset-rot",
        type=str,
        default=None,
        help='Override table camera offset quaternion as "x,y,z,w" (ros convention).',
    )
    parser.add_argument(
        "--wrist-cam-offset-pos",
        type=str,
        default=None,
        help='Override wrist camera offset position as "x,y,z" (meters).',
    )
    parser.add_argument(
        "--wrist-cam-offset-rot",
        type=str,
        default=None,
        help='Override wrist camera offset quaternion as "x,y,z,w" (ros convention).',
    )
    parser.add_argument(
        "--preview-cameras",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show real-time camera RGB frames in local windows (requires opencv-python).",
    )
    parser.add_argument(
        "--preview-every",
        type=int,
        default=1,
        help="Preview every N env steps (only used when --preview-cameras).",
    )

    parser.add_argument(
        "--viewport-preview",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show camera feeds using Isaac Sim viewport windows (no OpenCV needed).",
    )
    parser.add_argument(
        "--viewport-show-exterior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --viewport-preview, show the exterior/table camera viewport.",
    )
    parser.add_argument(
        "--viewport-show-wrist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --viewport-preview, show the wrist camera viewport.",
    )
    parser.add_argument("--viewport-width", type=int, default=640)
    parser.add_argument("--viewport-height", type=int, default=360)
    parser.add_argument(
        "--viewport-env-index",
        type=int,
        default=0,
        help="When resolving {ENV_REGEX_NS} camera prim paths, use env_{index}.",
    )

    parser.add_argument("--external-image-key", type=str, default=None)
    parser.add_argument("--wrist-image-key", type=str, default=None)
    parser.add_argument("--external-camera-sensor", type=str, default="camera")
    parser.add_argument("--wrist-camera-sensor", type=str, default="wrist_camera")
    parser.add_argument("--external-camera-output", type=str, default="rgb")
    parser.add_argument("--wrist-camera-output", type=str, default="rgb")

    parser.add_argument("--robot-asset-name", type=str, default=None)
    parser.add_argument("--eef-body-name", type=str, default=None)
    parser.add_argument("--gripper-joint-indices", type=str, default=None)
    parser.add_argument("--eef-9d-key", type=str, default=None)
    parser.add_argument("--eef-position-key", type=str, default=None)
    parser.add_argument("--eef-quat-key", type=str, default=None)
    parser.add_argument("--joint-position-key", type=str, default=None)
    parser.add_argument("--gripper-position-key", type=str, default=None)
    parser.add_argument("--quat-order", choices=["wxyz", "xyzw"], default="wxyz")
    parser.add_argument(
        "--action-keys-to-env",
        type=str,
        default="joint_position,gripper_position",
        help="Comma-separated GR00T action keys to concatenate and send to Isaac Lab.",
    )
    parser.add_argument(
        "--arm-action-dim",
        type=int,
        default=7,
        help=(
            "Number of joint_position action dimensions to send before appending the scalar "
            "gripper action. Franka lift env expects 7 arm actions + 1 gripper action."
        ),
    )
    parser.add_argument(
        "--arm-state-dim",
        type=int,
        default=7,
        help=(
            "Number of robot joint_pos dimensions to expose as GR00T joint_position state. "
            "Franka has 7 arm joints plus 2 finger joints; GR00T expects the arm joints here."
        ),
    )


def _parse_args() -> tuple[argparse.Namespace, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_gr00t_args(parser)

    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:
        raise ImportError(
            "Could not import Isaac Lab. Run this script with Isaac Lab's Python, "
            "for example `${ISAACLAB_PATH}/isaaclab.sh -p ...`."
        ) from exc

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    return args, app_launcher.app


def _inject_franka_cameras(env_cfg: Any, args: argparse.Namespace) -> None:
    """Add two RGB cameras to a Franka manager-based task config."""
    from isaaclab.sensors import CameraCfg
    import isaaclab.sim as sim_utils

    external_name = args.external_camera_sensor or "camera"
    wrist_name = args.wrist_camera_sensor or "wrist_camera"
    args.external_camera_sensor = external_name
    args.wrist_camera_sensor = wrist_name

    external_pos = (
        _parse_floats_csv(args.external_cam_offset_pos, count=3, name="external-cam-offset-pos")
        or DEFAULT_EXTERNAL_CAM_POS
    )
    external_rot = (
        _parse_floats_csv(args.external_cam_offset_rot, count=4, name="external-cam-offset-rot")
        or DEFAULT_EXTERNAL_CAM_ROT
    )
    wrist_pos = (
        _parse_floats_csv(args.wrist_cam_offset_pos, count=3, name="wrist-cam-offset-pos")
        or DEFAULT_WRIST_CAM_POS
    )
    wrist_rot = (
        _parse_floats_csv(args.wrist_cam_offset_rot, count=4, name="wrist-cam-offset-rot")
        or DEFAULT_WRIST_CAM_ROT
    )

    setattr(
        env_cfg.scene,
        external_name,
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=args.image_height,
            width=args.image_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=external_pos,
                rot=external_rot,
                convention="ros",
            ),
        ),
    )
    setattr(
        env_cfg.scene,
        wrist_name,
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=args.image_height,
            width=args.image_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=wrist_pos,
                rot=wrist_rot,
                convention="ros",
            ),
        ),
    )

    env_cfg.num_rerenders_on_reset = max(getattr(env_cfg, "num_rerenders_on_reset", 0), 3)
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "render"):
        env_cfg.sim.render.antialiasing_mode = "DLAA"

    print("Injected Franka cameras into env cfg:", flush=True)
    print(f"  external sensor='{external_name}' prim_path={{ENV_REGEX_NS}}/table_cam", flush=True)
    print(f"    offset.pos={external_pos} offset.rot={external_rot} convention=ros", flush=True)
    print(
        f"  wrist sensor='{wrist_name}' prim_path={{ENV_REGEX_NS}}/Robot/panda_hand/wrist_cam",
        flush=True,
    )
    print(f"    offset.pos={wrist_pos} offset.rot={wrist_rot} convention=ros", flush=True)


def _make_env(args: argparse.Namespace) -> Any:
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    if args.inject_franka_cameras:
        _inject_franka_cameras(env_cfg, args)

    env = gym.make(args.task, cfg=env_cfg)
    print(f"Created Isaac Lab env: {args.task}")
    print(f"  observation_space={env.observation_space}")
    print(f"  action_space={env.action_space}")

    return env


def _unwrap_reset(reset_result: Any) -> Any:
    if isinstance(reset_result, tuple):
        return reset_result[0]
    return reset_result


def _unwrap_step(step_result: Any) -> tuple[Any, bool]:
    if len(step_result) == 5:
        obs, _, terminated, truncated, _ = step_result
        done = bool(_to_numpy(terminated).reshape(-1)[0] or _to_numpy(truncated).reshape(-1)[0])
        return obs, done
    if len(step_result) == 4:
        obs, _, done, _ = step_result
        return obs, bool(_to_numpy(done).reshape(-1)[0])
    raise ValueError(f"Unexpected env.step result length: {len(step_result)}")


def main() -> None:
    args, simulation_app = _parse_args()

    import torch

    env = _make_env(args)
    policy = PolicyClient(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.policy_timeout_ms,
        api_token=args.policy_api_token,
        strict=False,
    )
    adapter = IsaacLabGr00tAdapter(env, policy, args)

    env_obs = _unwrap_reset(env.reset())
    try:
        for step in range(args.max_steps):
            # Optional camera preview (real-time RGB).
            try:
                exterior_image, wrist_image = adapter._extract_images(env_obs)
                adapter.preview.show(step, exterior=exterior_image, wrist=wrist_image)
            except Exception:
                # Preview is best-effort; do not break control loop.
                pass

            env_action_np = adapter.get_env_action(env_obs, step)
            env_action = torch.tensor(
                env_action_np,
                dtype=torch.float32,
                device=getattr(env.unwrapped, "device", args.device),
            )
            env_obs, done = _unwrap_step(env.step(env_action))
            if done:
                print(f"Environment reported done at step {step}; resetting.")
                env_obs = _unwrap_reset(env.reset())
                adapter.frame_buffer.clear()
                adapter.action_chunk = None
                adapter.action_index = 0
    finally:
        env.close()
        adapter.viewport_preview.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print("\nAdapter failed with Python exception:", flush=True)
        traceback.print_exc()
        raise
