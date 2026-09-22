# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeviceIO SOMA body source. Evaluation and compatibility mapping stay downstream."""

from typing import Any

from isaaccapture.deviceio_trackers import SomaBodyPoseV0Tracker

from .deviceio_tensor_types import DeviceIOSomaBodyPoseV0Tracked
from .interface import IDeviceIOSource
from ..interface.retargeter_core_types import RetargeterIO, RetargeterIOType
from ..interface.tensor_group import TensorGroup
from ..interface.tensor_group_type import OptionalType


class SomaBodySource(IDeviceIOSource):
    """Expose received SOMA V0 body controls without FK or validity reconstruction.

    The collection ID must match the publisher. Consumers configure their own
    identity and evaluation policy; this source requires neither SOMA-X nor Torch.
    """

    BODY = "soma_body"

    def __init__(self, name: str, collection_id: str) -> None:
        if not collection_id:
            raise ValueError("SOMA body collection_id must not be empty")
        self._tracker = SomaBodyPoseV0Tracker(collection_id)
        super().__init__(name)

    def get_tracker(self) -> SomaBodyPoseV0Tracker:
        return self._tracker

    def poll_tracker(self, deviceio_session: Any) -> RetargeterIO:
        group = TensorGroup(self.input_spec()["deviceio_soma_body"])
        group[0] = self._tracker.get_data(deviceio_session)
        return {"deviceio_soma_body": group}

    def input_spec(self) -> RetargeterIOType:
        return {"deviceio_soma_body": DeviceIOSomaBodyPoseV0Tracked()}

    def output_spec(self) -> RetargeterIOType:
        return {self.BODY: OptionalType(DeviceIOSomaBodyPoseV0Tracked())}

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        data = inputs["deviceio_soma_body"][0]
        if data is None:
            outputs[self.BODY].set_none()
        else:
            outputs[self.BODY][0] = data
