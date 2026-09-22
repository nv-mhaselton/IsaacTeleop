# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish the bundled SOMA motion through the SOMA body transport."""

import argparse
import logging
import math
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from isaaccapture.schema import (
    Point,
    SomaBodyJointRotationsV0,
    SomaBodyJointV0,
    SomaBodyPoseV0,
)

logger = logging.getLogger("isaaccapture.examples.soma_body_publisher")


def create_layer(data_root: Path):
    """Create the neutral SOMA-X v0.3.1 model used by the bundled motion."""
    import soma
    import torch

    if soma.__version__ != "0.3.1":
        raise RuntimeError(f"The POC requires SOMA-X 0.3.1, got {soma.__version__}")
    layer = soma.SOMALayer(
        data_root=str(data_root),
        identity_model_type="soma",
        device="cpu",
        lod="low",
        output_unit=soma.Unit.METERS,
        enable_procedural_transforms=False,
        correctives_model_path=None,
    )
    expected = [
        name.replace("_", "")
        for name, joint in sorted(
            SomaBodyJointV0.__members__.items(), key=lambda item: int(item[1])
        )
        if name != "NUM_JOINTS"
    ]
    if [name.upper() for name in layer.public_joint_names] != ["ROOT", *expected]:
        raise ValueError("SOMA layer joint order differs from the V0 FBS")
    with torch.no_grad():
        layer.prepare_identity(
            torch.zeros((1, layer.identity_model.num_identity_coeffs))
        )
    return layer


def demo_controls(path: Path, layer):
    """Normalize the bundled motion as in v0.3.1 tools/demo_soma_vis.py."""
    import torch
    from soma.geometry.rig_utils import joint_local_to_world, joint_world_to_local
    from soma.geometry.transforms import matrix_to_quaternion_xyzw

    motion = np.load(path, allow_pickle=False)
    if (
        motion.ndim != 4
        or motion.shape[1] not in (78, 94)
        or motion.shape[2:] != (4, 4)
        or not len(motion)
    ):
        raise ValueError("Expected bundled SOMA motion shaped (T, 78 or 94, 4, 4)")
    local = torch.as_tensor(motion[..., :3, :3], dtype=torch.float32)
    if local.shape[1] == 94:
        twists = (*range(40, 44), *range(72, 76), *range(81, 85), *range(90, 94))
        local = local[:, [i for i in range(94) if i not in twists]]
    orient = layer.t_pose_world[layer.public_transform_joint_indices, :3, :3]
    world = joint_local_to_world(local, layer.public_joint_parent_ids)
    relative = joint_world_to_local(
        world @ orient.transpose(-2, -1), layer.public_joint_parent_ids
    )[:, 1:]
    return matrix_to_quaternion_xyzw(relative).numpy(), motion[:, 1, :3, 3], relative


def soma_pose(quaternions, translation):
    joints = SomaBodyJointRotationsV0()
    joints.rotations[:] = quaternions
    joints.is_valid[:] = 1
    return SomaBodyPoseV0(joints, Point(*translation), True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--pusher", required=True, type=Path, help="Built soma_body_pusher executable"
    )
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate FBS packets without OpenXR",
    )
    args = parser.parse_args(argv[1:])
    if not math.isfinite(args.rate) or args.rate <= 0:
        parser.error("rate must be finite and positive")
    layer = create_layer(args.data_root)
    q, t, _ = demo_controls(args.data_root / "example_animation.npy", layer)
    command = [str(args.pusher.resolve())] + (
        ["--validate-only"] if args.validate_only else []
    )
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        interrupted = False
        try:
            start = time.monotonic()
            frame = 0
            while True:
                for rotations, translation in zip(q, t):
                    remaining = start + frame / args.rate - time.monotonic()
                    if remaining > 0 and not args.validate_only:
                        time.sleep(remaining)
                    payload = soma_pose(rotations, translation).to_bytes()
                    process.stdin.write(
                        struct.pack("<IQ", len(payload), round(frame * 1e9 / args.rate))
                    )
                    process.stdin.write(payload)
                    process.stdin.flush()
                    frame += 1
                if not args.loop or args.validate_only:
                    break
        except KeyboardInterrupt:
            interrupted = True
        finally:
            process.stdin.close()
        exit_code = process.wait()
        if exit_code != 0 and not (interrupted and exit_code == -signal.SIGINT):
            raise RuntimeError("SOMA body pusher failed; see its runtime diagnostics")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
