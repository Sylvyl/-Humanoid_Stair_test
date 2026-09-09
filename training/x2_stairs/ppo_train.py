"""PPO training loop against the local ``StairMujocoEnv``.

Trains a diagonal-Gaussian actor-critic MLP with clipped-surrogate PPO and
GAE(lambda) advantages, using the *existing* ``PolicyContract.observation()``
for tensor assembly (never re-implemented) and running against
``StairMujocoEnv`` -- pure local simulation, no path to the real robot.

This is a from-scratch, dependency-light PPO (PyTorch only, no RL framework),
intended to prove the training mechanics work end-to-end and produce a
checkpoint that can flow through ``mujoco_closed_loop.py`` ->
``validate_onnx.py``. It is not tuned for fast convergence or curriculum
auto-advancement -- see ``task_spec.domain_for_stage`` for which curriculum
stages are wired up.

Run as a module so the package-relative imports below resolve:

    python -m training.x2_stairs.ppo_train --stage single_step_up

Requires ``pip install -r training/requirements-sim.txt`` (mujoco + torch)
and ``AIMDK_SDK_ROOT`` set (see ``robot_reference``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _src  # noqa: F401  (adds src/ to sys.path)
from x2_stair_autonomy.policy_contract import PolicyContract

from .mujoco_env import StairMujocoEnv
from .task_spec import TaskSpec, direction_for_stage, domain_for_stage

_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "config" / "policy_manifest.json"
STAGES = ("single_step_up", "single_step_down", "three_steps", "random_straight_flight")


class ActorCritic(nn.Module):
    """Diagonal-Gaussian actor + scalar critic sharing an MLP trunk.

    The actor mean is tanh-squashed and scaled by ``action_clip`` so it
    always sits inside the manifest's declared action range; residual noise
    from the Gaussian is clamped the same way after sampling (a standard,
    slightly-approximate simplification -- no squashed-Gaussian log-prob
    correction is applied, which is fine for a first working PPO).
    """

    def __init__(self, obs_size: int, action_size: int, action_clip: float, hidden: int = 256):
        super().__init__()
        self.action_clip = action_clip
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, action_size)
        self.log_std = nn.Parameter(torch.full((action_size,), -0.5))
        self.critic = nn.Linear(hidden, 1)

    def _std(self) -> torch.Tensor:
        # Clamped so an unregularized log_std can't collapse the policy to
        # near-deterministic (which made observed PPO ratios/losses explode
        # to values in the millions during this session's own validation
        # run) or blow up into pure noise.
        return self.log_std.clamp(-2.0, 0.5).exp()

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        mean = torch.tanh(self.actor_mean(features)) * self.action_clip
        value = self.critic(features).squeeze(-1)
        return mean, value

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        std = self._std().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.sample()
        action = torch.clamp(raw_action, -self.action_clip, self.action_clip)
        log_prob = dist.log_prob(raw_action).sum(-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        std = self._std().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


class VecStairEnv:
    """``num_envs`` independent ``StairMujocoEnv`` instances stepped in a
    plain Python loop each iteration -- batches the *network* forward pass,
    not the physics (each env still steps MuJoCo on its own). Auto-resets any
    env that reports ``done`` and tracks completed-episode statistics."""

    def __init__(self, num_envs: int, stage: str, base_seed: int):
        self.stage = stage
        self.direction = direction_for_stage(stage)
        self.task_spec = TaskSpec(domain=domain_for_stage(stage))
        self.envs = [StairMujocoEnv(task_spec=self.task_spec) for _ in range(num_envs)]
        self._next_seed = [base_seed + i for i in range(num_envs)]
        self._seed_stride = num_envs
        self.episode_reward = [0.0] * num_envs
        self.episode_length = [0] * num_envs
        self.completed_episode_rewards: list[float] = []
        self.completed_episode_lengths: list[int] = []
        self.obs = [
            env.reset(seed, force_direction=self.direction)
            for env, seed in zip(self.envs, self._next_seed)
        ]

    def step(self, actions) -> tuple[list[dict], np.ndarray, np.ndarray, list[dict]]:
        obs_batch = []
        rewards = np.zeros(len(self.envs), dtype=np.float32)
        dones = np.zeros(len(self.envs), dtype=bool)
        reward_terms_batch = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            result = env.step(action)
            self.episode_reward[i] += result.reward
            self.episode_length[i] += 1
            rewards[i] = result.reward
            dones[i] = result.done
            reward_terms_batch.append(result.reward_terms)
            if result.done:
                self.completed_episode_rewards.append(self.episode_reward[i])
                self.completed_episode_lengths.append(self.episode_length[i])
                self.episode_reward[i] = 0.0
                self.episode_length[i] = 0
                self._next_seed[i] += self._seed_stride
                obs_batch.append(env.reset(self._next_seed[i], force_direction=self.direction))
            else:
                obs_batch.append(result.observation)
        self.obs = obs_batch
        return obs_batch, rewards, dones, reward_terms_batch


def flatten_obs(contract: PolicyContract, values_list: list[dict]) -> np.ndarray:
    return np.stack([contract.observation(v) for v in values_list]).astype(np.float32)


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_values: np.ndarray,
    *, gamma: float, lambda_: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``rewards``/``values``/``dones`` are (T, N); returns (advantages, returns), also (T, N)."""
    steps, _num_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(steps)):
        next_value = last_values if t == steps - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lambda_ * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def collect_rollout(vec_env: VecStairEnv, model: ActorCritic, contract: PolicyContract, rollout_steps: int, device):
    obs_buf, action_buf, logprob_buf, value_buf, reward_buf, done_buf = [], [], [], [], [], []
    reward_term_totals: dict[str, float] = {}
    reward_term_count = 0
    for _ in range(rollout_steps):
        obs_flat = flatten_obs(contract, vec_env.obs)
        obs_tensor = torch.from_numpy(obs_flat).to(device)
        with torch.no_grad():
            actions, log_probs, values = model.act(obs_tensor)
        actions_np = actions.cpu().numpy()
        obs_buf.append(obs_flat)
        action_buf.append(actions_np)
        logprob_buf.append(log_probs.cpu().numpy())
        value_buf.append(values.cpu().numpy())
        _, rewards, dones, reward_terms_batch = vec_env.step(list(actions_np))
        reward_buf.append(rewards)
        done_buf.append(dones)
        for terms in reward_terms_batch:
            reward_term_count += 1
            for name, value in terms.items():
                reward_term_totals[name] = reward_term_totals.get(name, 0.0) + value

    obs_final = flatten_obs(contract, vec_env.obs)
    with torch.no_grad():
        _, last_values = model.forward(torch.from_numpy(obs_final).to(device))
    mean_reward_terms = {
        name: total / reward_term_count for name, total in reward_term_totals.items()
    }
    return (
        np.stack(obs_buf), np.stack(action_buf), np.stack(logprob_buf),
        np.stack(value_buf), np.stack(reward_buf), np.stack(done_buf),
        last_values.cpu().numpy(), mean_reward_terms,
    )


def ppo_update(
    model: ActorCritic, optimizer: torch.optim.Optimizer,
    obs: np.ndarray, actions: np.ndarray, old_log_probs: np.ndarray,
    advantages: np.ndarray, returns: np.ndarray,
    *, epochs: int, minibatch_size: int, clip: float, entropy_coef: float, value_coef: float,
    max_grad_norm: float, device, target_kl: float | None = 0.02,
) -> tuple[float, float, float]:
    steps, num_envs = obs.shape[0], obs.shape[1]
    obs2 = obs.reshape(steps * num_envs, -1)
    actions2 = actions.reshape(steps * num_envs, -1)
    old_log_probs2 = old_log_probs.reshape(-1)
    advantages2 = advantages.reshape(-1)
    advantages2 = (advantages2 - advantages2.mean()) / (advantages2.std() + 1e-8)
    returns2 = returns.reshape(-1)

    obs_t = torch.from_numpy(obs2).to(device)
    actions_t = torch.from_numpy(actions2).to(device)
    old_log_probs_t = torch.from_numpy(old_log_probs2).to(device)
    advantages_t = torch.from_numpy(advantages2).to(device)
    returns_t = torch.from_numpy(returns2).to(device)

    total = obs_t.shape[0]
    policy_losses, value_losses, approx_kls = [], [], [0.0]
    # KL-based early stop between epochs: this session's own validation runs
    # showed policy_loss spiking into the hundreds of millions and reward
    # declining over many iterations even with LR decay and a lower entropy
    # coefficient. Summing log-probs over a 29-dim action space means a small
    # per-dimension drift compounds multiplicatively into a large ratio, and
    # naively taking multiple gradient epochs over the same rollout can then
    # keep pushing the policy further from the data it was collected under.
    # Stopping once the (old -> new) KL estimate exceeds target_kl is the
    # standard fix (used by e.g. OpenAI Spinning Up's PPO) for exactly this.
    for _epoch in range(epochs):
        permutation = torch.randperm(total)
        epoch_kls = []
        for start in range(0, total, minibatch_size):
            idx = permutation[start:start + minibatch_size]
            log_probs, entropy, values = model.evaluate(obs_t[idx], actions_t[idx])
            ratio = torch.exp(log_probs - old_log_probs_t[idx])
            surr1 = ratio * advantages_t[idx]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages_t[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns_t[idx])
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            with torch.no_grad():
                log_ratio = log_probs - old_log_probs_t[idx]
                epoch_kls.append(float(((log_ratio.exp() - 1.0) - log_ratio).mean()))
        approx_kls = epoch_kls
        if target_kl is not None and target_kl > 0 and float(np.mean(epoch_kls)) > target_kl:
            break
    return float(np.mean(policy_losses)), float(np.mean(value_losses)), float(np.mean(approx_kls))


class ActorOnly(nn.Module):
    """ONNX-exportable wrapper: deterministic mean action only, matching the
    manifest's single-input/single-output contract."""

    def __init__(self, model: ActorCritic):
        super().__init__()
        self.model = model

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        mean, _value = self.model.forward(observation)
        return mean


def export_onnx(model: ActorCritic, contract: PolicyContract, path: Path) -> None:
    model.eval()
    dummy = torch.zeros(1, contract.observation_size)
    # dynamo=False: use the legacy TorchScript-based exporter. Recent torch
    # defaults to the dynamo-based exporter, which additionally requires the
    # `onnxscript` package; the legacy path needs nothing beyond torch itself
    # and is sufficient for this plain MLP.
    torch.onnx.export(
        ActorOnly(model), dummy, str(path),
        input_names=["observation"], output_names=["action"], opset_version=17,
        dynamo=False,
    )


def train(args: argparse.Namespace) -> Path:
    # Without this, network init and action sampling draw from PyTorch's
    # unseeded global RNG regardless of --seed, so two runs with the same
    # --seed still diverge -- only the env's own domain/episode sampling was
    # actually reproducible before this fix.
    torch.manual_seed(args.seed)
    # Local import: evaluate.py imports ActorCritic/_MANIFEST_PATH from this
    # module, so importing it at module level here would be circular. By
    # call time this module is already fully initialized, so this is safe.
    from .evaluate import evaluate as evaluate_success_rate

    contract = PolicyContract.from_path(_MANIFEST_PATH)
    vec_env = VecStairEnv(args.num_envs, args.stage, args.seed)
    device = torch.device("cpu")
    model = ActorCritic(contract.observation_size, contract.action_size, contract.action_clip).to(device)
    if args.resume_from is not None:
        model.load_state_dict(torch.load(args.resume_from, map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Linear decay to a floor fraction of the initial LR, rather than a
    # constant LR for the whole run: this session's own 300-iteration
    # validation run showed reward *declining* over the second half with a
    # constant LR (27.15 -> 23.59 -> 16.92 mean reward across thirds, final
    # policy failing within 4 steps) while policy_loss stayed unstable into
    # the hundreds of millions throughout -- consistent with sustained large
    # updates causing drift rather than convergence.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda it: max(args.lr_min_frac, 1.0 - it / args.iterations)
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.jsonl"

    for iteration in range(1, args.iterations + 1):
        obs, actions, log_probs, values, rewards, dones, last_values, reward_terms = collect_rollout(
            vec_env, model, contract, args.rollout_steps, device
        )
        advantages, returns = compute_gae(rewards, values, dones, last_values, gamma=0.99, lambda_=0.95)
        policy_loss, value_loss, approx_kl = ppo_update(
            model, optimizer, obs, actions, log_probs, advantages, returns,
            epochs=args.epochs, minibatch_size=256, clip=0.2, entropy_coef=args.entropy_coef,
            value_coef=0.5, max_grad_norm=0.5, device=device, target_kl=args.target_kl,
        )
        scheduler.step()
        recent_rewards = vec_env.completed_episode_rewards[-20:]
        recent_lengths = vec_env.completed_episode_lengths[-20:]
        entry = {
            "iteration": iteration,
            "stage": args.stage,
            "mean_episode_reward": float(np.mean(recent_rewards)) if recent_rewards else None,
            "mean_episode_length": float(np.mean(recent_lengths)) if recent_lengths else None,
            "completed_episodes": len(vec_env.completed_episode_rewards),
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "approx_kl": approx_kl,
            "lr": scheduler.get_last_lr()[0],
            "reward_terms": reward_terms,
        }
        if iteration % args.eval_every == 0 or iteration == args.iterations:
            eval_report = evaluate_success_rate(
                model, contract, args.stage,
                episodes=args.eval_episodes, max_steps=args.eval_max_steps,
                base_seed=args.seed + 1_000_000,
            )
            entry["eval_success_rate"] = eval_report["success_rate"]
            entry["eval_episodes"] = eval_report["episodes"]
            entry["eval_failure_terminations"] = eval_report["failure_terminations"]
            model.train()

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(entry)
        if iteration % args.checkpoint_every == 0 or iteration == args.iterations:
            torch.save(model.state_dict(), output_dir / f"checkpoint_{iteration}.pt")

    export_onnx(model, contract, output_dir / "policy.onnx")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="single_step_up")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4, help="initial Adam learning rate")
    parser.add_argument(
        "--lr-min-frac", type=float, default=0.1,
        help="floor for the linear LR decay, as a fraction of --lr (0 = decay to zero)",
    )
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=3, help="PPO epochs per rollout")
    parser.add_argument(
        "--target-kl", type=float, default=0.02,
        help="stop taking further epochs this rollout once mean approx-KL exceeds this; <=0 disables",
    )
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="continue training from a previous checkpoint_<n>.pt instead of a fresh network",
    )
    parser.add_argument(
        "--eval-every", type=int, default=25,
        help="run a deterministic success-rate evaluation (see evaluate.py) every N iterations",
    )
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-max-steps", type=int, default=200)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "runs" / "ppo",
    )
    args = parser.parse_args()
    output_dir = train(args)
    print(f"done: {output_dir}")


if __name__ == "__main__":
    main()
