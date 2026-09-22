# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import signal
import struct
import subprocess

import numpy as np
import pytest

from isaaccapture.retargeting_engine.utilities import SomaBodyEvaluator
from isaaccapture.schema import DeviceDataTimestamp, SomaBodyPoseV0Record
from isaaccapture_examples.soma_body_publisher import publisher
from isaaccapture_examples.soma_body_publisher.publisher import (
    create_layer,
    demo_controls,
    soma_pose,
)


def test_nested_soma_serialization_uses_payload_root():
    q = np.tile([0.0, 0.0, 0.0, 1.0], (77, 1))
    pose = soma_pose(q, [1, 2, 3])
    record = SomaBodyPoseV0Record(pose, DeviceDataTimestamp(10, 20, 30))
    assert record.data.to_bytes() == pose.to_bytes()
    assert record.to_bytes() != pose.to_bytes()


@pytest.mark.parametrize("exit_code", [0, -signal.SIGINT, 1])
def test_publisher_interrupt_preserves_real_failures(monkeypatch, tmp_path, exit_code):
    from unittest.mock import MagicMock

    process = MagicMock()
    process.__enter__.return_value = process
    process.stdin.write.side_effect = KeyboardInterrupt
    process.wait.return_value = exit_code
    monkeypatch.setattr(publisher.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(publisher, "create_layer", lambda _: None)
    monkeypatch.setattr(
        publisher,
        "demo_controls",
        lambda *args: (np.tile([0, 0, 0, 1], (1, 77, 1)), [[0, 0, 0]], None),
    )
    args = ["soma_body_publisher", "--data-root", str(tmp_path), "--pusher", "pusher"]
    if exit_code == 1:
        with pytest.raises(RuntimeError, match="pusher failed"):
            publisher.main(args)
    else:
        assert publisher.main(args) == 0
    process.stdin.close.assert_called_once()


@pytest.mark.parametrize(
    "packet",
    [b"x", struct.pack("<IQ", 2049, 0), struct.pack("<IQ", 8, 0) + b"badbytes"],
)
def test_pusher_rejects_invalid_packets_without_runtime(request, packet):
    executable = request.config.getoption("--soma-pusher")
    if executable is None:
        pytest.skip("Pass --soma-pusher to exercise the native producer boundary")
    result = subprocess.run(
        [str(executable), "--validate-only"],
        input=packet,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1


def test_demo_to_fbs_to_native_soma_matches_upstream(soma_assets, request):
    import torch
    from soma.geometry.transforms import (
        matrix_to_quaternion_xyzw,
        quaternion_xyzw_to_matrix,
    )

    evaluator = SomaBodyEvaluator("soma_body_evaluator", create_layer(soma_assets))
    q, t, matrices = demo_controls(
        soma_assets / "example_animation.npy", evaluator.layer
    )
    torch.testing.assert_close(
        quaternion_xyzw_to_matrix(torch.from_numpy(q)), matrices, atol=1e-4, rtol=1e-4
    )
    with torch.no_grad():
        expected = evaluator.layer.pose(
            matrices[:1],
            transl=torch.from_numpy(t[:1]),
            pose2rot=False,
            fk_only=True,
            apply_correctives=False,
        )
    payload = soma_pose(q[0], t[0])
    record = SomaBodyPoseV0Record(payload, DeviceDataTimestamp(10, 20, 30))
    positions, orientations, valid = evaluator.evaluate(record.data)
    np.testing.assert_allclose(positions, expected["joints"][0].numpy(), atol=1e-4)
    expected_orientations = matrix_to_quaternion_xyzw(
        expected["transforms"][0, 1:, :3, :3]
    )
    actual_rotations = quaternion_xyzw_to_matrix(torch.from_numpy(orientations.copy()))
    expected_rotations = quaternion_xyzw_to_matrix(expected_orientations)
    torch.testing.assert_close(
        actual_rotations, expected_rotations, atol=1e-4, rtol=1e-4
    )
    np.testing.assert_array_equal(valid, np.ones(77, dtype=np.uint8))

    flipped_positions, _, _ = evaluator.evaluate(soma_pose(-q[0], t[0]))
    np.testing.assert_allclose(flipped_positions, positions, atol=1e-6)

    executable = request.config.getoption("--soma-pusher")
    if executable is not None:
        packets = []
        for frame, (rotations, translation) in enumerate(zip(q, t)):
            encoded = soma_pose(rotations, translation).to_bytes()
            packets.append(struct.pack("<IQ", len(encoded), frame) + encoded)
        result = subprocess.run(
            [str(executable), "--validate-only"],
            input=b"".join(packets),
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode()
