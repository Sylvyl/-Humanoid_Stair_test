# AgiBot X2 Ultra Stair Lab

This workspace is the safe first implementation of the stair-autonomy plan. It
provides geometric stair perception, a fail-closed mission state machine,
authenticated operator UI, ROS status interfaces, PC2 compatibility checks,
rosbag recording, and a provider-neutral training contract.

**It cannot command the robot.** There is no publisher for
`/aima/mc/locomotion/velocity` or `/aima/hal/joint/*/command`. The supplied AimDK
ONNX file is a dance example validated only in simulation and is not used.

## Repository map

- `src/x2_stair_autonomy`: testable perception, authentication, dashboard API,
  runtime state, and mission supervisor.
- `static`: dependency-free learning and inspection UI.
- `ros2`: `StairEstimate`, `SafetyStatus`, `StairMission`, and a read-only ROS
  observer that only publishes `/x2/stairs/*` derived status.
- `training`: stair-domain specification, tensor manifest, cloud container,
  smoke test, ONNX validator, and an asset approval gate.
- `scripts`: read-only PC2 probe, synchronized rosbag recorder, and TLS setup.
- `deployment`: guarded shadow deployment; services are staged but not enabled.

## 1. Test locally on Windows

```powershell
cd D:\AGBX2utra_testing\x2_stair_autonomy
& '.\scripts\local.ps1' -Mode test
& '.\scripts\local.ps1' -Mode dashboard
```

Open `http://127.0.0.1:8443`. On first launch, the server creates
`config/operator-token`; read that file locally to unlock the operator panel.
Even if every box is checked, vendor/profile approval remains false and arming
is rejected.

## 2. Verify the live robot without changing it

Keep the robot stationary, retain the remote controller and hardware E-stop,
then run:

```powershell
& 'D:\AGBX2utra_testing\x2_stair_autonomy\scripts\probe-pc2.ps1'
```

Enter the `agi` password interactively. The probe checks PC2 at `10.0.1.41`,
Ubuntu/ROS/AimDK state, topics, services, QoS, and host health. It explicitly
refuses PC1 at `10.0.1.40`.

## 3. Build shadow mode on PC2

Review the deployment script, then run it explicitly:

```powershell
& 'D:\AGBX2utra_testing\x2_stair_autonomy\deployment\deploy-shadow.ps1' -Execute
```

If the first upload succeeded but its remote build failed, fix/review the
detailed error and resume without overwriting the uploaded source:

```powershell
& 'D:\AGBX2utra_testing\x2_stair_autonomy\deployment\deploy-shadow.ps1' -Execute -Resume
```

After reviewed local source changes, synchronize only project-owned files and
rebuild with:

```powershell
& 'D:\AGBX2utra_testing\x2_stair_autonomy\deployment\deploy-shadow.ps1' -Execute -Resume -SyncSource
```

`-SyncSource` first verifies the remote deployment signature. It excludes the
operator token and any real vendor approval record, then runs the full unit test
suite on PC2. It never enables a service or registers a motion input source.

The script refuses to overwrite an existing target and leaves both user
services disabled. On PC2, generate a certificate, verify its printed SHA-256
fingerprint, inspect the operator token, and only then start the services:

```bash
bash ~/x2dev/x2_stair_autonomy/scripts/generate-tls.sh ~/x2dev/x2_stair_autonomy/config/tls
systemctl --user start x2-stair-shadow.service x2-stair-dashboard.service
```

Open `https://10.0.1.41:8443`. The earlier read-only sensor dashboard remains
unchanged on port 8080.

## 4. Record and calibrate

With the robot stationary on the gantry, start the shadow observer, then:

```bash
bash ~/x2dev/x2_stair_autonomy/scripts/record-shadow-bag.sh
```

Capture labelled bags on a flat floor, unsupported objects, and measured
straight stairs. Record true riser, tread, width, camera pose, lighting, and
direction in a separate test log. Never infer safety from a single sensor frame.
The recorder first creates `validation/pc2_compatibility.json`, resolves the
actual IMU and optional power topics, and refuses to record when a required
sensor role is absent. A missing PMU topic stays visible but is never fabricated.
RGB-D projection remains disabled until `sensor_extrinsics.example.json` is
replaced by a calibrated, reviewed camera-to-base transform; LiDAR/IMU shadow
estimates must not be described as fused perception before that gate passes.

After measuring the bag timestamps, copy
`validation/bag_annotations.example.json`, fill in non-overlapping scene
intervals and expected geometry, and set each transform approval flag only after
the calibration review. Replay creates no ROS node or publisher:

```bash
python3 -m x2_stair_autonomy.bag_replay /path/to/bag annotations.json \
  --output validation/perception_report.json
```

## 5. Training workflow

Start with the contract smoke test in `training/README.md`. Cloud training is
blocked until a validated X2 USD articulation exists. The initial policy
manifest is `UNAPPROVED_SHADOW_ONLY`, has no joint order, action scale, or
command rate, and therefore cannot pass the ONNX deployment validator.
The validator requires the model, signed approval, native inference reference
arrays, and a passing MuJoCo fault-stop report; it writes but never self-approves
`validation/onnx_report.json`.

The release gate is: deterministic replay, geometric accuracy, 1,000 held-out
simulation episodes per direction with at least 95% completion, all injected
faults stopping safely, ONNX parity, AgiBot approval, then the gantry test ladder.
The machine-readable gate is available at `GET /api/v1/gates/status`. Evidence
belongs in `validation/`; missing, malformed, unsigned, or hash-mismatched
evidence locks output-capable stages. `PolicyContract` assembles tensors in the
manifest order and returns an exact zero vector for stale sensors, heartbeat
loss, locked release state, bad shapes, exceptions, or non-finite output.

## Troubleshooting

| Symptom | Meaning | Corrective action |
|---|---|---|
| PC2 offline | Direct Ethernet/SSH unavailable | Set the physical adapter to `10.0.1.2/24`; rerun the probe. |
| Sensors stale | A required topic is older than 250 ms | Check DDS environment, topic QoS, and publisher state. |
| Unsupported stairs | Geometry/confidence outside the approved profile | Measure the staircase; do not override the rejection. |
| Vendor approval missing | Hardware control contract is incomplete | Obtain signed limits, rates, stop matrix, and model hash. |
| Heartbeat fault | Hold-to-run updates stopped for >350 ms | Inspect network/UI logs and complete the checklist again. |
| Existing deployment | Safe overwrite guard triggered | Review/archive the existing directory manually; never force replace it. |

## Non-negotiable safety boundary

Unattended operation, curved/open-riser stairs, escalators, wet surfaces,
clutter, and public-network control are outside scope. A web software stop is
not a hardware E-stop. Gantry, operator, spotter, remote controller, physical
E-stop, synchronized logging, and written stage review remain required for every
real-robot experiment.
