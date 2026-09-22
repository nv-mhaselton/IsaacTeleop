# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

import numpy as np
import pytest

from isaacteleop.schema import (
    DeviceDataTimestamp,
    Point,
    SomaBodyJointV0,
    SomaBodyJointRotationsV0,
    SomaBodyPoseV0,
    SomaBodyPoseV0Record,
    SomaJointRotationV0,
)


def test_soma_body_v0_joint_order():
    expected = [
        "HIPS",
        "SPINE1",
        "SPINE2",
        "CHEST",
        "NECK1",
        "NECK2",
        "HEAD",
        "HEAD_END",
        "JAW",
        "LEFT_EYE",
        "RIGHT_EYE",
        "LEFT_SHOULDER",
        "LEFT_ARM",
        "LEFT_FORE_ARM",
        "LEFT_HAND",
        "LEFT_HAND_THUMB1",
        "LEFT_HAND_THUMB2",
        "LEFT_HAND_THUMB3",
        "LEFT_HAND_THUMB_END",
        "LEFT_HAND_INDEX1",
        "LEFT_HAND_INDEX2",
        "LEFT_HAND_INDEX3",
        "LEFT_HAND_INDEX4",
        "LEFT_HAND_INDEX_END",
        "LEFT_HAND_MIDDLE1",
        "LEFT_HAND_MIDDLE2",
        "LEFT_HAND_MIDDLE3",
        "LEFT_HAND_MIDDLE4",
        "LEFT_HAND_MIDDLE_END",
        "LEFT_HAND_RING1",
        "LEFT_HAND_RING2",
        "LEFT_HAND_RING3",
        "LEFT_HAND_RING4",
        "LEFT_HAND_RING_END",
        "LEFT_HAND_PINKY1",
        "LEFT_HAND_PINKY2",
        "LEFT_HAND_PINKY3",
        "LEFT_HAND_PINKY4",
        "LEFT_HAND_PINKY_END",
        "RIGHT_SHOULDER",
        "RIGHT_ARM",
        "RIGHT_FORE_ARM",
        "RIGHT_HAND",
        "RIGHT_HAND_THUMB1",
        "RIGHT_HAND_THUMB2",
        "RIGHT_HAND_THUMB3",
        "RIGHT_HAND_THUMB_END",
        "RIGHT_HAND_INDEX1",
        "RIGHT_HAND_INDEX2",
        "RIGHT_HAND_INDEX3",
        "RIGHT_HAND_INDEX4",
        "RIGHT_HAND_INDEX_END",
        "RIGHT_HAND_MIDDLE1",
        "RIGHT_HAND_MIDDLE2",
        "RIGHT_HAND_MIDDLE3",
        "RIGHT_HAND_MIDDLE4",
        "RIGHT_HAND_MIDDLE_END",
        "RIGHT_HAND_RING1",
        "RIGHT_HAND_RING2",
        "RIGHT_HAND_RING3",
        "RIGHT_HAND_RING4",
        "RIGHT_HAND_RING_END",
        "RIGHT_HAND_PINKY1",
        "RIGHT_HAND_PINKY2",
        "RIGHT_HAND_PINKY3",
        "RIGHT_HAND_PINKY4",
        "RIGHT_HAND_PINKY_END",
        "LEFT_LEG",
        "LEFT_SHIN",
        "LEFT_FOOT",
        "LEFT_TOE_BASE",
        "LEFT_TOE_END",
        "RIGHT_LEG",
        "RIGHT_SHIN",
        "RIGHT_FOOT",
        "RIGHT_TOE_BASE",
        "RIGHT_TOE_END",
    ]

    assert len(expected) == int(SomaBodyJointV0.NUM_JOINTS) == 77
    for index, name in enumerate(expected):
        assert int(getattr(SomaBodyJointV0, name)) == index


def test_soma_body_v0_joint_rotation():
    rotation = SomaJointRotationV0(Point(0.1, 0.2, 0.3), True)

    assert rotation.axis_angle.x == pytest.approx(0.1)
    assert rotation.axis_angle.y == pytest.approx(0.2)
    assert rotation.axis_angle.z == pytest.approx(0.3)
    assert rotation.is_valid is True


def test_soma_body_v0_rotation_views_alias_storage():
    rotations = SomaBodyJointRotationsV0()

    assert rotations.axis_angles.shape == (77, 3)
    assert rotations.axis_angles.dtype == np.float32
    assert rotations.is_valid.shape == (77,)
    assert rotations.is_valid.dtype == np.uint8
    assert not rotations.axis_angles.flags.owndata

    rotations.axis_angles[SomaBodyJointV0.HEAD] = [0.4, 0.5, 0.6]
    rotations.is_valid[SomaBodyJointV0.HEAD] = 1

    head = rotations.values(int(SomaBodyJointV0.HEAD))
    assert head.axis_angle.z == pytest.approx(0.6)
    assert head.is_valid is True


def test_soma_body_v0_rotation_index_check():
    with pytest.raises(IndexError):
        SomaBodyJointRotationsV0().values(77)


def test_soma_body_v0_pose_construction_and_lifetime():
    rotations = SomaBodyJointRotationsV0()
    rotations.axis_angles[:] = np.arange(77 * 3, dtype=np.float32).reshape(77, 3)
    rotations.is_valid[:] = 1

    pose = SomaBodyPoseV0(rotations, Point(1.0, 2.0, 3.0), True)
    axis_angles = pose.joint_rotations.axis_angles
    del pose
    gc.collect()

    assert axis_angles[76, 2] == pytest.approx(230.0)


def test_soma_body_v0_pose_defaults_and_global_translation():
    pose = SomaBodyPoseV0()

    assert pose.joint_rotations is not None
    assert pose.global_translation.x == 0.0
    assert pose.global_translation_is_valid is False

    translated = SomaBodyPoseV0(
        global_translation=Point(1.0, 2.0, 3.0),
        global_translation_is_valid=True,
    )
    assert translated.global_translation.y == pytest.approx(2.0)
    assert translated.global_translation_is_valid is True


def test_soma_body_v0_pose_record():
    record = SomaBodyPoseV0Record(SomaBodyPoseV0(), DeviceDataTimestamp(100, 200, 300))

    assert record.data is not None
    assert record.timestamp.sample_time_local_common_clock == 200
