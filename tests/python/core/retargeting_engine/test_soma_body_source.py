# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np
import pytest

from isaaccapture.deviceio_trackers import SomaBodyPoseV0Tracker
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    DeviceIOFullBodyPoseTracked,
    DeviceIOSomaBodyPoseV0Tracked,
    SomaBodySource,
)
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.schema import (
    BodyJoints,
    FullBodyPose,
    Point,
    SomaBodyJointRotationsV0,
    SomaBodyPoseV0,
)
from isaaccapture.teleop_session_manager.helpers import _get_trackers_from_pipeline


def pose():
    joints = SomaBodyJointRotationsV0()
    joints.rotations[:] = (0, 0, 0, 1)
    joints.is_valid[:] = 1
    return SomaBodyPoseV0(joints, Point(1, 2, 3), True)


def payload_group(group_type, data):
    group = TensorGroup(group_type)
    group[0] = data
    return group


def test_tracker_discovery_and_specs():
    source = SomaBodySource("body", "vendor.soma")
    assert isinstance(source.get_tracker(), SomaBodyPoseV0Tracker)
    assert source.get_vendor() is None
    assert _get_trackers_from_pipeline(source) == [source.get_tracker()]
    assert list(source.input_spec()) == ["deviceio_soma_body"]
    assert list(source.output_spec()) == [SomaBodySource.BODY]
    assert source.output_spec()[SomaBodySource.BODY].is_optional


def test_polling_preserves_payload_and_does_not_evaluate():
    source = SomaBodySource("body", "vendor.soma")
    tracker = MagicMock()
    source._tracker = tracker
    raw = pose()
    tracker.get_data.return_value = raw
    session = object()
    inputs = source.poll_tracker(session)
    tracker.get_data.assert_called_once_with(session)
    assert inputs["deviceio_soma_body"][0] is raw
    assert source(inputs)[SomaBodySource.BODY][0] is raw


def test_partial_validity_and_translation_are_not_reconstructed():
    source = SomaBodySource("body", "vendor.soma")
    raw = pose()
    raw.joint_rotations.is_valid[10] = 0
    raw = SomaBodyPoseV0(raw.joint_rotations, raw.global_translation, False)
    inputs = payload_group(source.input_spec()["deviceio_soma_body"], raw)
    received = source({"deviceio_soma_body": inputs})[SomaBodySource.BODY][0]
    assert received.to_bytes() == raw.to_bytes()
    assert not received.global_translation_is_valid
    assert not received.joint_rotations.is_valid[10]
    np.testing.assert_array_equal(
        received.joint_rotations.rotations, raw.joint_rotations.rotations
    )


def test_absent_sample_clears_reused_output():
    source = SomaBodySource("body", "vendor.soma")
    inputs = payload_group(source.input_spec()["deviceio_soma_body"], pose())
    outputs = {
        SomaBodySource.BODY: _make_output_group(
            source.output_spec()[SomaBodySource.BODY]
        )
    }
    source.compute({"deviceio_soma_body": inputs}, outputs)
    assert not outputs[SomaBodySource.BODY].is_none
    inputs[0] = None
    source.compute({"deviceio_soma_body": inputs}, outputs)
    assert outputs[SomaBodySource.BODY].is_none
    inputs[0] = pose()
    source.compute({"deviceio_soma_body": inputs}, outputs)
    assert not outputs[SomaBodySource.BODY].is_none


def test_soma_and_fullbody_payloads_are_distinct():
    soma = DeviceIOSomaBodyPoseV0Tracked()
    fullbody = DeviceIOFullBodyPoseTracked()
    with pytest.raises(ValueError, match="type mismatch"):
        soma.check_compatibility(fullbody)
    with pytest.raises(TypeError, match="SomaBodyPoseV0"):
        payload_group(soma, FullBodyPose(BodyJoints(), False))
    with pytest.raises(TypeError, match="FullBodyPose"):
        payload_group(fullbody, pose())


def test_empty_collection_rejected():
    with pytest.raises(ValueError, match="collection_id"):
        SomaBodySource("body", "")
