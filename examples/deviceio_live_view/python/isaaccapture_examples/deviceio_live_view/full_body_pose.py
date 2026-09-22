# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FullBodyPose source and skeleton layout for the DeviceIO live viewer."""

from isaaccapture.retargeting_engine.deviceio_source_nodes import FullBodySource
from isaaccapture.retargeting_engine.tensor_types import (
    BodyJointIndex,
    FullBodyInputIndex,
)

from .body_pipeline import BodyViewLayout, BodyViewPipeline


FULL_BODY_POSE_BONES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
    (1, 4),
    (4, 7),
    (7, 10),
    (2, 5),
    (5, 8),
    (8, 11),
    (12, 13),
    (13, 16),
    (16, 18),
    (18, 20),
    (20, 22),
    (12, 14),
    (14, 17),
    (17, 19),
    (19, 21),
    (21, 23),
)

FULL_BODY_POSE_LAYOUT = BodyViewLayout(
    joint_names=tuple(joint.name for joint in BodyJointIndex),
    bones=FULL_BODY_POSE_BONES,
    positions_index=int(FullBodyInputIndex.JOINT_POSITIONS),
    valid_index=int(FullBodyInputIndex.JOINT_VALID),
)


def create_full_body_pose_pipeline() -> BodyViewPipeline:
    source = FullBodySource("body")
    return BodyViewPipeline(
        output=source.output(FullBodySource.FULL_BODY),
        layout=FULL_BODY_POSE_LAYOUT,
    )
