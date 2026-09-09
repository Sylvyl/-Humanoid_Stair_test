# X2 stair policy training gate

The included `kuailechongbai.onnx` is a dance example and is never loaded here.
This folder defines the stair domain, policy tensor contract, smoke runner, and
Isaac Lab asset gate. It does **not** contain a trained stair policy.

Run the dependency-light contract check:

```bash
cd training/x2_stairs
python smoke_train.py --episodes 32 --output ../runs/smoke.json
```

Before cloud training, obtain a dynamically validated X2 USD articulation from
the supplied model and AgiBot data. Set `X2_VALIDATED_USD` only after its joint
axes, inertias, limits, actuator model, contacts, and sensor frames are reviewed.
Place a sibling `<robot>.usd.vendor-approved.json` approval document matching
`config/asset_approval.example.json`; it must include the USD SHA-256 and every
required review. Until then `isaaclab_env.py` fails closed.

The eventual training job must implement the `TaskSpec` as an Isaac Lab
`ManagerBasedRLEnvCfg`, train a privileged teacher followed by a sensor student,
export ONNX, populate the joint order/action scaling in `policy_manifest.json`,
and change its status only after vendor approval. Paid long-running jobs are not
started by any script in this repository.

Build the container only with an explicitly reviewed Isaac Lab base image, for
example `docker build --build-arg BASE_IMAGE=<approved-image> .`. No mutable or
guessed base tag is selected by default.

## Local MuJoCo simulation (no Isaac Lab / cloud training required)

`x2_stairs/mujoco_env.py` builds a stair-climbing environment straight from
the AimDK SDK's own reference RL-deployment assets -- the real X2 MuJoCo model
and the real joint order/PD gains/action-scale convention from
`extra/x2_rl_deploy/x2_rl_deploy_controller/config/motion_control.yaml`
(transcribed, with citations, in `x2_stairs/robot_reference.py`). It runs
entirely locally against a physics simulator; it never touches the real robot
and is unrelated to the Isaac Lab teacher/student path above (which stays
blocked on a vendor-approved USD asset).

Point `AIMDK_SDK_ROOT` at the *inner* extracted SDK directory (the one
containing `README.en.md`, `src/`, `extra/`, `docs/` -- note the artifacts zip
nests a duplicate directory of the same name, so this is one level deeper than
where you extracted it), then install the extra simulation dependencies:

```bash
export AIMDK_SDK_ROOT=/path/to/aimdk-aarch64-*-artifacts/aimdk-aarch64-*-artifacts
pip install -r training/requirements-sim.txt
```

### Training a policy (PPO, CPU, local)

`x2_stairs/ppo_train.py` is a from-scratch PPO (clipped surrogate + GAE(lambda))
against `StairMujocoEnv`, using the *existing* `PolicyContract.observation()`
for tensor assembly. It is not a curriculum-auto-advancing trainer --
`task_spec.domain_for_stage()` wires up three concrete stages
(`single_step_up`, `single_step_down`, `three_steps`, plus the unconstrained
`random_straight_flight`); `flat_balance` and `noisy_sensor_student` raise
`NotImplementedError` naming what's missing (a no-stairs env mode, and
observation-noise/distillation, respectively) rather than being faked.

```bash
python -m training.x2_stairs.ppo_train --stage single_step_up \
  --iterations 60 --num-envs 4 --output-dir training/runs/ppo
```

Writes `training_log.jsonl` (per-iteration reward/length/loss), periodic
`checkpoint_<n>.pt` files, and `policy.onnx` at the end (input `observation`
`(1,144)`, output `action` `(1,29)`, matching `policy_manifest.json`). A short
CPU run (a few minutes) is enough to prove the mechanics work -- rollouts
collect, the update runs, a valid ONNX exports -- **not** to master a stair.
Reward curves from a short run are expected to be noisy, not monotonic; use
`--seed` for reproducibility (it seeds both the env's domain sampling and
PyTorch's network init/action sampling).

Feed the exported policy straight into the closed-loop validator below.

Produce the closed-loop report `validate_onnx.py --mujoco-report` expects --
this is a sanity/fault-stop check (does the fail-closed `PolicyContract` gate
actually zero the command under a sensor fault, are joint limits respected),
**not** the 1000-episode/95%-success task-mastery bar, which stays a separate,
much larger `validation/simulation_report.json` gate item:

```bash
cd training/x2_stairs
python mujoco_closed_loop.py <policy>.onnx \
  --manifest ../config/policy_manifest.json \
  --episodes 4 --output ../runs/mujoco-report.json
```

After training and MuJoCo closed-loop evaluation, validate the exported model:

```bash
python validate_onnx.py policy.onnx \
  --manifest ../config/policy_manifest.json \
  --approval ../../config/vendor_approval.json \
  --reference ../runs/native-parity.npz \
  --mujoco-report ../runs/mujoco-report.json \
  --output ../../validation/onnx_report.json
```

The NPZ must contain `observations` and `native_actions`. A hash mismatch,
tensor mismatch, non-finite value, parity error, or failed fault-stop test makes
the report fail and keeps the hardware stage locked.
