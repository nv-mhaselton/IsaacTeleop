# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

import numpy as np
import pytest

from isaacteleop.schema import (
    DeviceDataTimestamp,
    Point,
    SomaHandednessV0,
    SomaHandJointV0,
    SomaHandJointRotationsV0,
    SomaHandPoseV0,
    SomaHandPoseV0Record,
    SomaJointRotationV0,
)


def test_soma_hand_v0_joint_order():
    expected = [
        "WRIST",
        "THUMB1",
        "THUMB2",
        "THUMB3",
        "THUMB_END",
        "INDEX1",
        "INDEX2",
        "INDEX3",
        "INDEX4",
        "INDEX_END",
        "MIDDLE1",
        "MIDDLE2",
        "MIDDLE3",
        "MIDDLE4",
        "MIDDLE_END",
        "RING1",
        "RING2",
        "RING3",
        "RING4",
        "RING_END",
        "PINKY1",
        "PINKY2",
        "PINKY3",
        "PINKY4",
        "PINKY_END",
    ]

    assert len(expected) == int(SomaHandJointV0.NUM_JOINTS) == 25
    for index, name in enumerate(expected):
        assert int(getattr(SomaHandJointV0, name)) == index


def test_soma_hand_v0_joint_rotation():
    rotation = SomaJointRotationV0(Point(0.1, 0.2, 0.3), True)

    assert rotation.axis_angle.x == pytest.approx(0.1)
    assert rotation.axis_angle.y == pytest.approx(0.2)
    assert rotation.axis_angle.z == pytest.approx(0.3)
    assert rotation.is_valid is True


def test_soma_hand_v0_rotation_views_alias_storage():
    rotations = SomaHandJointRotationsV0()

    assert rotations.axis_angles.shape == (25, 3)
    assert rotations.axis_angles.dtype == np.float32
    assert rotations.is_valid.shape == (25,)
    assert rotations.is_valid.dtype == np.uint8
    assert not rotations.axis_angles.flags.owndata

    rotations.axis_angles[SomaHandJointV0.INDEX_END] = [0.4, 0.5, 0.6]
    rotations.is_valid[SomaHandJointV0.INDEX_END] = 1

    index_end = rotations.values(int(SomaHandJointV0.INDEX_END))
    assert index_end.axis_angle.z == pytest.approx(0.6)
    assert index_end.is_valid is True


def test_soma_hand_v0_rotation_index_check():
    with pytest.raises(IndexError):
        SomaHandJointRotationsV0().values(25)


def test_soma_hand_v0_pose_construction_and_lifetime():
    rotations = SomaHandJointRotationsV0()
    rotations.axis_angles[:] = np.arange(25 * 3, dtype=np.float32).reshape(25, 3)
    rotations.is_valid[:] = 1

    pose = SomaHandPoseV0(rotations, Point(1.0, 2.0, 3.0), True, SomaHandednessV0.RIGHT)
    axis_angles = pose.joint_rotations.axis_angles
    del pose
    gc.collect()

    assert axis_angles[24, 2] == pytest.approx(74.0)


def test_soma_hand_v0_pose_defaults_and_global_translation():
    pose = SomaHandPoseV0()

    assert pose.joint_rotations is not None
    assert pose.global_translation.x == 0.0
    assert pose.global_translation_is_valid is False
    assert pose.handedness == SomaHandednessV0.UNSPECIFIED

    right = SomaHandPoseV0(
        global_translation=Point(1.0, 2.0, 3.0),
        global_translation_is_valid=True,
        handedness=SomaHandednessV0.RIGHT,
    )
    assert right.global_translation.y == pytest.approx(2.0)
    assert right.global_translation_is_valid is True
    assert right.handedness == SomaHandednessV0.RIGHT


def test_soma_hand_v0_pose_record():
    record = SomaHandPoseV0Record(
        SomaHandPoseV0(handedness=SomaHandednessV0.LEFT),
        DeviceDataTimestamp(100, 200, 300),
    )

    assert record.data is not None
    assert record.data.handedness == SomaHandednessV0.LEFT
    assert record.timestamp.sample_time_local_common_clock == 200
