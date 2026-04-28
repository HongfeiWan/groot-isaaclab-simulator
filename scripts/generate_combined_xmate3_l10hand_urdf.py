#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a combined URDF: xMate3 arm + right L10 hand as one robot.

This avoids nested rigid-body hierarchies by defining the arm+hand in one URDF
with a fixed joint.

Inputs (repo-relative by default):
- demo_data/l10_hand/xMate3_description/urdf/xMate3.urdf
- demo_data/l10_hand/right-L10hand/linkerhand_l10_right.urdf

Output:
- demo_data/l10_hand/xMate3_with_right_L10hand.urdf
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARM = REPO_ROOT / "demo_data/l10_hand/xMate3_description/urdf/xMate3.urdf"
DEFAULT_HAND = REPO_ROOT / "demo_data/l10_hand/right-L10hand/linkerhand_l10_right.urdf"
DEFAULT_OUT = REPO_ROOT / "demo_data/l10_hand/xMate3_with_right_L10hand.urdf"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _inner_robot_xml(text: str) -> tuple[str, str]:
    """Return (robot_open_tag, inner_xml) without outer closing tag."""
    start = text.find("<robot")
    if start < 0:
        raise ValueError("No <robot ...> tag found")
    open_end = text.find(">", start)
    if open_end < 0:
        raise ValueError("Malformed <robot ...> tag")
    robot_open = text[start : open_end + 1]

    end = text.rfind("</robot>")
    if end < 0:
        raise ValueError("No </robot> closing tag found")

    inner = text[open_end + 1 : end].strip()
    return robot_open, inner


def _rewrite_hand_mesh_paths(hand_inner: str) -> str:
    # Original hand URDF uses: filename="meshes/xxx.STL"
    # We want: filename="right-L10hand/meshes/xxx.STL"
    hand_inner = hand_inner.replace('filename="meshes/', 'filename="right-L10hand/meshes/')
    hand_inner = hand_inner.replace("filename='meshes/", "filename='right-L10hand/meshes/")
    return hand_inner


def _generate(
    *,
    arm_urdf: Path,
    hand_urdf: Path,
    out_urdf: Path,
    parent_link: str,
    child_link: str,
) -> None:
    arm_text = _read_text(arm_urdf)
    hand_text = _read_text(hand_urdf)

    _, arm_inner = _inner_robot_xml(arm_text)
    _, hand_inner = _inner_robot_xml(hand_text)
    hand_inner = _rewrite_hand_mesh_paths(hand_inner)

    # Fixed assembly: quat xyzw = (0,0,-1,1) normalized => yaw = -pi/2.
    yaw = -math.pi / 2.0
    fixed_joint = f"""
  <!-- Fixed assembly: {parent_link} -> {child_link} (xyz=0,0,0; quat xyzw=0,0,-1,1 => yaw=-pi/2) -->
  <joint name="xMate3_to_l10hand_fixed" type="fixed">
    <parent link="{parent_link}"/>
    <child link="{child_link}"/>
    <origin xyz="0 0 0" rpy="0 0 {yaw}"/>
  </joint>
""".rstrip()

    robot_open = '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xMate3_with_right_L10hand">'
    combined = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!-- Auto-generated. DO NOT EDIT. -->\n"
        f"{robot_open}\n"
        "  <!-- ===== xMate3 arm (from xMate3.urdf) ===== -->\n"
        f"{arm_inner}\n\n"
        f"{fixed_joint}\n\n"
        "  <!-- ===== right L10 hand (from linkerhand_l10_right.urdf) ===== -->\n"
        "  <!-- Mesh paths rewritten to right-L10hand/meshes/... -->\n"
        f"{hand_inner}\n"
        "</robot>\n"
    )

    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    out_urdf.write_text(combined, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-urdf", type=Path, default=DEFAULT_ARM)
    parser.add_argument("--hand-urdf", type=Path, default=DEFAULT_HAND)
    parser.add_argument("--out-urdf", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--parent-link", type=str, default="xMate3_link6")
    parser.add_argument("--child-link", type=str, default="hand_base_link")
    args = parser.parse_args()

    if not args.arm_urdf.exists():
        raise FileNotFoundError(f"arm urdf not found: {args.arm_urdf}")
    if not args.hand_urdf.exists():
        raise FileNotFoundError(f"hand urdf not found: {args.hand_urdf}")

    _generate(
        arm_urdf=args.arm_urdf,
        hand_urdf=args.hand_urdf,
        out_urdf=args.out_urdf,
        parent_link=args.parent_link,
        child_link=args.child_link,
    )
    print(f"Wrote combined URDF: {args.out_urdf}")


if __name__ == "__main__":
    main()

