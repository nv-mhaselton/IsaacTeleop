# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np

from isaaccapture.retargeting_engine.interface.base_retargeter import (
    _make_output_group,
)
from isaaccapture.retargeting_engine.utilities import SomaBodyEvaluator


def evaluator_with_two_branches() -> SomaBodyEvaluator:
    layer = MagicMock()
    layer.public_joint_names = ("Root", *(f"Joint{index}" for index in range(77)))
    layer.output_joint_parent_ids = np.array([0, 0, 1, 1, *range(3, 77)])
    return SomaBodyEvaluator("soma_body_evaluator", layer)


def test_evaluator_exposes_native_joint_order_and_topology():
    evaluator = evaluator_with_two_branches()
    assert evaluator.joint_names[:3] == ("Joint0", "Joint1", "Joint2")
    assert evaluator.bones[:3] == ((0, 1), (0, 2), (2, 3))
    assert len(evaluator.joint_names) == 77
    assert len(evaluator.bones) == 76


def test_evaluator_propagates_validity_through_ancestors():
    evaluator = evaluator_with_two_branches()
    controls = np.ones(77, dtype=bool)
    controls[1] = False
    valid = evaluator._joint_validity(controls, translation_valid=True)
    assert valid[0]
    assert not valid[1]
    assert valid[2]
    assert valid[3]
    assert not evaluator._joint_validity(controls, translation_valid=False).any()


def test_evaluator_preserves_absent_input_without_loading_soma():
    evaluator = evaluator_with_two_branches()
    input_group = _make_output_group(evaluator.input_spec()["soma_body"])
    result = evaluator({"soma_body": input_group})
    assert result[SomaBodyEvaluator.BODY].is_none
