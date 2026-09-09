# AgiBot X2 Ultra — Stair Climbing Research Project

Teaching a humanoid robot (the **AgiBot X2 Ultra**) to walk up and down stairs on
its own, using reinforcement learning trained in a physics simulator.

This README is written for someone starting from **zero** — it explains what
this is, what you need to install, and exactly what to type. No prior robotics
or machine-learning experience assumed.

---

## ⚠️ Read this first — safety

This project controls (or will eventually control) a **full-size humanoid robot
that weighs a lot and is designed to walk on stairs**. A fall can injure people
and destroy expensive hardware.

**The software in this repository cannot move the real robot, and that is
deliberate.** There is no code here that sends motion commands to the robot.
Everything runs in simulation. Before any real-robot motion could ever happen,
a long list of safety gates must pass (vendor approval, simulation acceptance
testing, and staged physical tests with a gantry, a spotter, and a hardware
emergency stop).

If you are working with the physical robot, read
**[docs/OPERATIONS.md](docs/OPERATIONS.md)** first — it contains the
non-negotiable safety rules.

---

## What's actually in here?

Think of it as three separate pieces:

| Piece | What it does | Touches the real robot? |
|---|---|---|
| **Simulator + training** (`training/`) | A virtual robot on virtual stairs. An AI learns to walk here by trial and error, millions of times. | ❌ No |
| **Perception + dashboard** (`src/`, `static/`) | Reads the robot's cameras/sensors, works out where the stairs are, shows a web dashboard. Read-only. | 👀 Reads only, never commands |
| **Safety gate** (`src/x2_stair_autonomy/release_gate.py`) | A checklist that refuses to unlock robot control until every safety requirement is proven. | 🔒 Blocks everything |

### Current status (honest)

- ✅ The simulator works — the virtual robot stands, balances, and walks toward a step.
- ⚠️ It **cannot climb the step yet.** Measured success rate is currently **0%** — it
  reliably falls over (`excessive_tilt`) after about 86 simulation steps. Training
  is ongoing; this is a hard skill that typically needs far more compute than has
  been spent so far.
- 🔒 Real-robot control is **fully locked** (19 unmet safety requirements) and will
  stay that way until the vendor signs off and physical testing is done.

---

## 1. What you need to install

### 1.1 Tools

| Tool | Version | Why | Where to get it |
|---|---|---|---|
| **Python** | 3.10 or newer | Everything is written in Python | [python.org/downloads](https://www.python.org/downloads/) — **tick "Add Python to PATH"** during install |
| **Git** | any recent | To download and update this code | [git-scm.com](https://git-scm.com/downloads) |

Check they work — open a terminal (PowerShell on Windows) and run:

```bash
python --version
git --version
```

You should see version numbers. If Python says "not found", reinstall it with
the *Add to PATH* box ticked.

### 1.2 The AgiBot SDK (needed for simulation only)

The robot's 3D model comes from AgiBot's own SDK, which is **not** included here
(it's their proprietary software — we only point at it, never copy it).

You need the `aimdk-aarch64-*-artifacts` package from AgiBot. Extract it
somewhere, then note the path to the folder that contains `README.en.md`,
`src/`, `extra/`, and `docs/`.

> ⚠️ The zip usually contains a folder of the *same name* inside itself. You want
> the **inner** one, e.g.
> `C:\stuff\aimdk-aarch64-a424add7-artifacts\aimdk-aarch64-a424add7-artifacts`

Official docs: <https://x2-aimdk.agibot.com>

---

## 2. Setup — step by step

### Step 1 — Download the code

```bash
git clone https://github.com/Sylvyl/-Humanoid_Stair_test.git
cd -Humanoid_Stair_test
```

### Step 2 — Create a virtual environment

A "virtual environment" keeps this project's packages separate from the rest of
your computer. Strongly recommended.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You'll know it worked when your prompt starts with `(.venv)`.

### Step 3 — Install the packages

Minimum (perception, dashboard, tests — no simulator):

```bash
pip install numpy
```

Full install (adds the physics simulator and AI training):

```bash
pip install -r training/requirements-sim.txt
```

That installs:

| Package | What it's for |
|---|---|
| `numpy` | Number crunching (the maths everything is built on) |
| `mujoco` | The physics simulator — gravity, contacts, joints, motors |
| `torch` (PyTorch) | The neural network that learns to walk (CPU-only build) |
| `onnx` / `onnxruntime` | Exports/runs the trained brain in a portable format the robot could use |
| `imageio` | Turns simulation frames into a video/GIF you can watch |

> The CPU-only PyTorch index is already configured in that requirements file, so
> you won't accidentally download a ~2.5 GB GPU build you don't need.

### Step 4 — Tell it where the SDK is

**Windows (PowerShell):**
```powershell
$env:AIMDK_SDK_ROOT = "C:\path\to\aimdk-aarch64-a424add7-artifacts\aimdk-aarch64-a424add7-artifacts"
$env:PYTHONPATH = "$PWD\src"
```

**Linux / macOS:**
```bash
export AIMDK_SDK_ROOT=/path/to/aimdk-aarch64-a424add7-artifacts/aimdk-aarch64-a424add7-artifacts
export PYTHONPATH=$PWD/src
```

> These reset every time you close the terminal. Set them again each session, or
> add them to your shell profile.

---

## 3. Running things

### 3.1 Check everything works

```bash
python -m unittest discover -s tests
```

Expected: `OK` with ~55 tests. If simulator tests say `skipped`, your
`AIMDK_SDK_ROOT` isn't set or `mujoco` isn't installed — that's fine for the
non-simulation parts.

### 3.2 Train the AI to climb

```bash
python -m training.x2_stairs.ppo_train --stage single_step_up --iterations 300 --num-envs 6 --output-dir training/runs/my_first_run
```

What the options mean:

| Option | Meaning |
|---|---|
| `--stage` | Difficulty. `single_step_up` (easiest) → `three_steps` → `random_straight_flight` (full stairs) |
| `--iterations` | How many learning rounds. More = better, but slower. 300 ≈ 45–60 min on a normal laptop CPU |
| `--num-envs` | How many virtual robots practise at once |
| `--resume-from` | Continue from a saved checkpoint instead of starting over |
| `--eval-every` | How often to measure the real success rate |

It prints one line per round and saves checkpoints as it goes. Watch
`mean_episode_reward` (higher = better) and `eval_success_rate` (the real
measure — **this is what needs to reach 0.95**).

### 3.3 Measure how good it actually is

```bash
python -m training.x2_stairs.evaluate training/runs/my_first_run/checkpoint_300.pt --episodes 50 --output training/runs/my_first_run/success.json
```

Reports `success_rate` — the fraction of attempts where the robot actually got
to the top of the step without falling.

### 3.4 Watch it (make a video)

There's no separate app — you can either render a GIF or open MuJoCo's own
interactive 3D viewer:

```bash
python -m mujoco.viewer
```

Then drag-and-drop a scene file into the window. Left-drag orbits the camera,
scroll zooms, `Ctrl`+right-drag pushes the robot around.

### 3.5 The safety dashboard (read-only)

```bash
python -m x2_stair_autonomy.server --host 127.0.0.1 --port 8443
```

Open <http://127.0.0.1:8443>. On first run it creates a login token at
`config/operator-token` — open that file and paste the contents in to log in.

> 🔐 That token file is a password. It is listed in `.gitignore` and must never
> be committed or shared.

---

## 4. Project layout

```
├── src/x2_stair_autonomy/   # Perception, safety gate, dashboard (never commands the robot)
│   ├── perception.py        #   Finds stairs in sensor data
│   ├── release_gate.py      #   The safety checklist that blocks robot control
│   ├── supervisor.py        #   Mission state machine (fails closed)
│   └── server.py            #   Web dashboard
├── training/x2_stairs/      # Simulation + AI training
│   ├── mujoco_env.py        #   The virtual robot + stairs
│   ├── terrain.py           #   Builds the virtual staircase
│   ├── ppo_train.py         #   The learning algorithm (PPO)
│   ├── evaluate.py          #   Measures real success rate
│   └── robot_reference.py   #   Real robot joint data (from the AgiBot SDK)
├── tests/                   # Automated tests
├── docs/OPERATIONS.md       # ⚠️ Robot safety + hardware procedures — read before touching hardware
└── validation/              # Evidence files required by the safety gate
```

---

## 5. Common problems

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'mujoco'` | Activate your venv, then `pip install -r training/requirements-sim.txt` |
| `AIMDK_SDK_ROOT is unset` | Do Step 4 above. Remember it resets when you close the terminal |
| `expected AimDK SDK path does not exist` | You pointed at the outer folder — go one level deeper (the one with `README.en.md`) |
| Tests say `skipped` | Normal if the simulator isn't installed — non-simulation tests still run |
| Training is very slow | Expected on CPU. Lower `--num-envs`, or leave it running in the background |
| `python` not found (Windows) | Reinstall Python with "Add Python to PATH" ticked |

---

## 6. What's not in this repository (on purpose)

- **The AgiBot SDK** — proprietary; referenced via `AIMDK_SDK_ROOT`
- **Login tokens, TLS keys, vendor approval records** — secrets
- **Robot probe output, signed test records** — real hardware evidence containing
  machine details and people's names
- **Training checkpoints and models** — large binaries; regenerate by training

See [.gitignore](.gitignore) for the full list.

---

## 7. Licence and credits

Built for the AgiBot X2 Ultra using the AgiBot AimDK SDK
(<https://x2-aimdk.agibot.com>). The SDK itself is AgiBot's property and is not
redistributed here.
