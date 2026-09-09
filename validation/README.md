# Validation evidence

This directory contains machine-readable acceptance evidence. Missing, malformed,
or failing evidence locks every output-capable hardware stage.

- `perception_report.json`: replay errors and unsupported-scene rejection result.
- `simulation_report.json`: held-out up/down completion and injected-fault stops.
- `onnx_report.json`: native/ONNX parity, tensor contract, and MuJoCo result.
- `hardware_stages.json`: signed operator/spotter reviews for completed stages.
- `bag_annotations.example.json`: deliberately unapproved example for
  timestamped, publisher-free rosbag replay.

Do not edit reports to claim success. Generate them from the corresponding test
runner and retain the raw logs and model hashes used to produce each report.
