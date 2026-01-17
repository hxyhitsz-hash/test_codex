#!/usr/bin/env python3
"""CleanRL-style PPO trainer for Gymnasium environments with optional video capture."""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CleanRL-style PPO for Gymnasium")
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="Gymnasium environment id")
    parser.add_argument("--total-timesteps", type=int, default=250_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA")
    parser.add_argument("--capture-video", action="store_true", help="Record training video")
    parser.add_argument("--video-dir", type=str, default="videos", help="Directory to save videos")
    parser.add_argument("--video-step-trigger", type=int, default=10_000, help="Capture video every N steps")
    return parser.parse_args()


@dataclass
class BatchData:
    observations: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class Agent(nn.Module):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space) -> None:
        super().__init__()
        obs_dim = int(np.prod(observation_space.shape))
        action_dim = action_space.n

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def get_action_and_value(
        self, observation: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(observation)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(observation).squeeze(-1)



def make_env(env_id: str, seed: int, idx: int, capture_video: bool, video_dir: str, video_step_trigger: int):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            os.makedirs(video_dir, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_dir,
                step_trigger=lambda step: step % video_step_trigger == 0,
            )
        env.reset(seed=seed + idx)
        env.action_space.seed(seed + idx)
        return env

    return thunk



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def build_batches(
    obs: torch.Tensor,
    actions: torch.Tensor,
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    num_minibatches: int,
) -> list[BatchData]:
    batch_size = obs.shape[0]
    minibatch_size = batch_size // num_minibatches
    indices = np.arange(batch_size)
    np.random.shuffle(indices)

    batches = []
    for start in range(0, batch_size, minibatch_size):
        end = start + minibatch_size
        batch_idx = indices[start:end]
        batches.append(
            BatchData(
                observations=obs[batch_idx],
                actions=actions[batch_idx],
                logprobs=logprobs[batch_idx],
                advantages=advantages[batch_idx],
                returns=returns[batch_idx],
                values=values[batch_idx],
            )
        )
    return batches



def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed,
                idx,
                args.capture_video,
                args.video_dir,
                args.video_step_trigger,
            )
            for idx in range(args.num_envs)
        ]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only discrete actions are supported"

    agent = Agent(envs.single_observation_space, envs.single_action_space).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    global_step = 0
    start_time = time.time()

    for update in range(1, num_updates + 1):
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            next_done = torch.tensor(terminated | truncated, device=device, dtype=torch.float32)
            rewards[step] = torch.tensor(reward, device=device)
            next_obs = torch.tensor(next_obs_np, device=device, dtype=torch.float32)

            if "episode" in infos:
                for item in infos["episode"]:
                    if item is not None:
                        print(
                            f"global_step={global_step}, episode_return={item['r']:.2f}, "
                            f"episode_length={item['l']:.0f}"
                        )

        with torch.no_grad():
            next_value = agent.get_action_and_value(next_obs)[-1]
            advantages = torch.zeros_like(rewards).to(device)
            last_gae = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = rewards[t] + args.gamma * next_values * next_non_terminal - values[t]
                last_gae = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae
            returns = advantages + values

        batch_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        batch_actions = actions.reshape(-1)
        batch_logprobs = logprobs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = values.reshape(-1)

        batches = build_batches(
            batch_obs,
            batch_actions,
            batch_logprobs,
            batch_advantages,
            batch_returns,
            batch_values,
            args.num_minibatches,
        )

        for _ in range(args.update_epochs):
            for batch in batches:
                _, newlogprob, entropy, value = agent.get_action_and_value(batch.observations, batch.actions)
                logratio = newlogprob - batch.logprobs
                ratio = logratio.exp()

                advantages_normalized = (batch.advantages - batch.advantages.mean()) / (
                    batch.advantages.std() + 1e-8
                )

                pg_loss1 = -advantages_normalized * ratio
                pg_loss2 = -advantages_normalized * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((batch.returns - value) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if update % 10 == 0 or update == num_updates:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed)
            print(
                f"update={update}/{num_updates}, steps={global_step}, sps={sps}, "
                f"loss={loss.item():.3f}"
            )

    envs.close()


if __name__ == "__main__":
    main()
