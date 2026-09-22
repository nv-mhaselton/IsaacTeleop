# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate transported SOMA controls into global SOMA joint poses."""

from importlib import import_module
from typing import Any

import numpy as np

from ..deviceio_source_nodes.deviceio_tensor_types import (
    DeviceIOSomaBodyPoseV0Tracked,
)
from ..interface.base_retargeter import BaseRetargeter
from ..interface.retargeter_core_types import RetargeterIO, RetargeterIOType
from ..interface.tensor_group_type import OptionalType
from ..tensor_types import SomaBodyInput, SomaBodyInputIndex


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class SomaBodyEvaluator(BaseRetargeter):
    """Run upstream SOMA FK once and expose all public transported joints."""

    BODY = "soma_body"

    def __init__(self, name: str, layer: Any) -> None:
        joint_names = tuple(str(name) for name in layer.public_joint_names)
        parent_ids = _numpy(layer.output_joint_parent_ids).astype(np.int64)
        if len(joint_names) != 78 or joint_names[0].upper() != "ROOT":
            raise ValueError("SOMA evaluator requires Root plus 77 public joints")
        if parent_ids.shape != (78,):
            raise ValueError("SOMA evaluator requires 78 public parent IDs")
        if any(
            parent < 0 or parent >= child
            for child, parent in enumerate(parent_ids[1:], 1)
        ):
            raise ValueError("SOMA public joints must follow parent-before-child order")

        self.layer = layer
        self.joint_names = joint_names[1:]
        self.parent_ids = tuple(int(parent) for parent in parent_ids)
        self.bones = tuple(
            (int(parent) - 1, child - 1)
            for child, parent in enumerate(parent_ids[1:], 1)
            if parent != 0
        )
        super().__init__(name)

    def input_spec(self) -> RetargeterIOType:
        return {"soma_body": OptionalType(DeviceIOSomaBodyPoseV0Tracked())}

    def output_spec(self) -> RetargeterIOType:
        return {self.BODY: OptionalType(SomaBodyInput())}

    def _joint_validity(
        self, control_valid: np.ndarray, translation_valid: bool
    ) -> np.ndarray:
        valid = np.zeros(78, dtype=np.uint8)
        valid[0] = translation_valid
        for child in range(1, 78):
            valid[child] = valid[self.parent_ids[child]] and control_valid[child - 1]
        return valid[1:]

    def evaluate(self, data: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        torch = import_module("torch")
        transforms = import_module("soma.geometry.transforms")

        control_valid = np.asarray(data.joint_rotations.is_valid, dtype=bool)
        rotations = np.asarray(data.joint_rotations.rotations, dtype=np.float32).copy()
        rotations[~control_valid] = (0.0, 0.0, 0.0, 1.0)
        translation_valid = bool(data.global_translation_is_valid)
        translation = data.global_translation
        translation_xyz = (
            [translation.x, translation.y, translation.z]
            if translation_valid
            else [0.0, 0.0, 0.0]
        )

        with torch.no_grad():
            output = self.layer.pose(
                transforms.quaternion_xyzw_to_matrix(
                    torch.from_numpy(rotations).unsqueeze(0)
                ),
                transl=torch.tensor([translation_xyz], dtype=torch.float32),
                pose2rot=False,
                fk_only=True,
                apply_correctives=False,
            )
            orientations = transforms.matrix_to_quaternion_xyzw(
                output["transforms"][0, 1:, :3, :3]
            )
        return (
            _numpy(output["joints"][0]).astype(np.float32, copy=False),
            _numpy(orientations).astype(np.float32, copy=False),
            self._joint_validity(control_valid, translation_valid),
        )

    def _compute_fn(
        self, inputs: RetargeterIO, outputs: RetargeterIO, context: Any
    ) -> None:
        raw = inputs["soma_body"]
        group = outputs[self.BODY]
        if raw.is_none:
            group.set_none()
            return

        positions, orientations, valid = self.evaluate(raw[0])
        group[SomaBodyInputIndex.JOINT_POSITIONS] = positions
        group[SomaBodyInputIndex.JOINT_ORIENTATIONS] = orientations
        group[SomaBodyInputIndex.JOINT_VALID] = valid
