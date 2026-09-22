<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeviceIO Live View

Draws live DeviceIO human tracking — hands, head, controllers, full body — in the
browser with [viser](https://viser.studio). Trackers that are inactive or absent
are hidden rather than drawn in an error color. `CloudXRLauncher` starts the
CloudXR runtime and WSS proxy itself, so there is nothing to launch separately.

```bash
uv pip install -e ./examples/deviceio_live_view
python -m isaaccapture_examples.deviceio_live_view --accept-eula
```

Open the URL it prints (default <http://localhost:8080>). It binds every
interface, so another machine on the network can reach it at
`http://<this-host>:8080`; pass `--host 127.0.0.1` to keep it local, `--port` to
move it. Ctrl+C stops it.

## SOMA body POC

The unreleased SOMA POC uses upstream FK and renders all 77 transported joints
in SOMA's native public order. It uses a neutral identity, not subject
calibration. Separate hand tracking is unchanged.

```text
SOMA publisher -> SchemaPusher -> SomaBodyPoseV0Tracker
              -> SomaBodySource -> SomaBodyEvaluator -> native SOMA viewer
```

Use SOMA-X 0.3.1 and its neutral identity assets. For an uninstalled source
build, set `PYTHONPATH` to the built IsaacCapture package and this example's
`python/` directory.

```bash
uv pip install -e ./examples/deviceio_live_view

python -m isaaccapture_examples.deviceio_live_view --accept-eula \
  --body-schema soma \
  --soma-data-root /path/to/SOMA-X/assets
```

Use the independent [SOMA body publisher](../soma_body_publisher/README.md) to
send the bundled motion, or connect another publisher using the checked-in SOMA
body schema.

Select the body input at launch with `--body-schema full-body-pose` (the default)
or `--body-schema soma`. SOMA requires `--soma-data-root` for downstream FK.
For compatibility with earlier commands, supplying assets without `--body-schema`
also selects SOMA; an explicit schema always takes precedence. Switch sources by
restarting the viewer with the other option. Hands, head, and controllers remain
unchanged.

The SOMA viewer reads the `soma_demo` collection by default. Use
`--soma-collection-id <id>` to match another publisher's collection.

`SomaBodySource` follows the normal DeviceIO source contract: the session discovers
its tracker and polls the received payload. Its typed, optional `soma_body` output
preserves SOMA controls, translation, and validity without FK or reconstruction.
It requires neither SOMA-X nor PyTorch. The example connects the reusable
`SomaBodyEvaluator` downstream. The evaluator runs upstream FK once per graph
step and emits global positions and orientations for the 77 transported joints.
The viewer uses the joint names and hierarchy reported by the pinned SOMA layer.
It does not convert SOMA into `FullBodyPose`.

SOMA-X and PyTorch are viewer dependencies, not transport dependencies.

Missing data hides the body. Invalid translation invalidates the complete
skeleton. An invalid joint control also invalidates its descendants because
their global poses depend on that ancestor. Other valid branches remain visible.
Independent hands, subject calibration, a `FullBodyPose` compatibility
retargeter, and recording/replay examples are outside this live-view POC.

The focused tests cover upstream FK agreement, body-source selection, native
SOMA topology, and the shared renderer. Real-asset checks are optional:

```bash
python -m pytest tests/python/examples/deviceio_live_view \
  --soma-assets /path/to/SOMA-X/assets
```
