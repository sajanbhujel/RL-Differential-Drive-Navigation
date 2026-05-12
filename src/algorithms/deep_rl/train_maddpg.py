#!/usr/bin/env python3

import os
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import csv
import matplotlib.pyplot as plt

import rclpy

from src.algorithms.deep_rl.networks import Actor, Critic, soft_update
from src.algorithms.deep_rl.replay_buffer import MultiAgentReplayBuffer
from src.environments.multiagent_gazebo_env import MultiAgentGazeboEnv


class MADDPGAgent:
    def __init__(
        self,
        agent_id,
        obs_dim,
        action_dim,
        global_state_dim,
        total_action_dim,
        max_action,
        device,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.01,
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.tau = tau

        self.max_action = np.array(max_action, dtype=np.float32)

        self.actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(global_state_dim, total_action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, obs, noise_std=0.1):
        obs_tensor = torch.tensor(
            obs,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]

        noise = np.random.normal(0.0, noise_std, size=action.shape)
        action = action + noise

        action[0] = np.clip(action[0], 0.0, self.max_action[0])
        action[1] = np.clip(action[1], -self.max_action[1], self.max_action[1])

        return action.astype(np.float32)


def update_maddpg(agents, replay_buffer, batch_size, device):
    batch = replay_buffer.sample(batch_size, device)

    obs = [
        batch["obs1"],
        batch["obs2"],
    ]

    next_obs = [
        batch["next_obs1"],
        batch["next_obs2"],
    ]

    global_state = batch["global_state"]
    next_global_state = batch["next_global_state"]

    actions = batch["actions"]
    rewards = batch["rewards"]
    dones = batch["dones"]

    action_dim = agents[0].action_dim

    with torch.no_grad():
        next_actions = []

        for i, agent in enumerate(agents):
            next_action_i = agent.actor_target(next_obs[i])
            next_actions.append(next_action_i)

        next_actions = torch.cat(next_actions, dim=1)

    for i, agent in enumerate(agents):
        with torch.no_grad():
            target_q = agent.critic_target(next_global_state, next_actions)

            y = rewards[:, i:i + 1] + agent.gamma * (
                1.0 - dones[:, i:i + 1]
            ) * target_q

        current_q = agent.critic(global_state, actions)

        critic_loss = nn.MSELoss()(current_q, y)

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        current_actions = []

        for j, other_agent in enumerate(agents):
            if j == i:
                current_action_j = other_agent.actor(obs[j])
                current_actions.append(current_action_j)

            else:
                start = j * action_dim
                end = start + action_dim
                current_actions.append(actions[:, start:end].detach())

        current_actions = torch.cat(current_actions, dim=1)

        actor_loss = -agent.critic(global_state, current_actions).mean()

        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        soft_update(agent.actor_target, agent.actor, agent.tau)
        soft_update(agent.critic_target, agent.critic, agent.tau)


def moving_average(values, window=20):
    values = np.array(values, dtype=np.float32)

    if len(values) == 0:
        return values

    window = min(window, len(values))
    kernel = np.ones(window) / window

    return np.convolve(values, kernel, mode="same")


def plot_training_curve(
    episode_list,
    reward_robot1_list,
    reward_robot2_list,
    avg_reward_list,
    plot_dir,
):
    if len(episode_list) == 0:
        print("No training data to plot.")
        return

    plt.figure(figsize=(10, 6))

    plt.plot(
        episode_list,
        reward_robot1_list,
        label="Robot 1 Reward",
        alpha=0.35,
    )

    plt.plot(
        episode_list,
        reward_robot2_list,
        label="Robot 2 Reward",
        alpha=0.35,
    )

    plt.plot(
        episode_list,
        moving_average(avg_reward_list, window=20),
        label="Average Reward Moving Average",
        linewidth=2,
    )

    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Training Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(plot_dir, "training_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Training curve saved at: {save_path}")


def train():
    rclpy.init()

    env = MultiAgentGazeboEnv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    obs1, obs2, global_state = env.reset()

    obs_dim = obs1.shape[0]
    action_dim = 2
    n_agents = 2

    global_state_dim = obs_dim * n_agents
    total_action_dim = action_dim * n_agents

    max_action = np.array(
        [
            env.max_linear,
            env.max_angular,
        ],
        dtype=np.float32,
    )

    agent1 = MADDPGAgent(
        agent_id=0,
        obs_dim=obs_dim,
        action_dim=action_dim,
        global_state_dim=global_state_dim,
        total_action_dim=total_action_dim,
        max_action=max_action,
        device=device,
    )

    agent2 = MADDPGAgent(
        agent_id=1,
        obs_dim=obs_dim,
        action_dim=action_dim,
        global_state_dim=global_state_dim,
        total_action_dim=total_action_dim,
        max_action=max_action,
        device=device,
    )

    agents = [
        agent1,
        agent2,
    ]

    replay_buffer = MultiAgentReplayBuffer(
        max_size=200000,
        obs_dim=obs_dim,
        global_state_dim=global_state_dim,
        action_dim=action_dim,
    )

    episodes = 1000
    batch_size = 256
    warmup_steps = 2000
    train_after_every_step = 1

    total_steps = 0

    noise_std = 0.30
    noise_decay = 0.995
    min_noise = 0.05

    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    log_dir = "training_logs"
    plot_dir = "training_plots"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    log_csv_path = os.path.join(log_dir, "training_log.csv")

    with open(log_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "episode",
            "reward_robot1",
            "reward_robot2",
            "avg_reward",
        ])

    episode_list = []
    reward_robot1_list = []
    reward_robot2_list = []
    avg_reward_list = []

    try:
        for ep in range(episodes):
            obs1, obs2, global_state = env.reset()

            ep_rewards = np.zeros(2, dtype=np.float32)
            ep_dones = np.zeros(2, dtype=np.int32)

            for step in range(env.max_steps):
                total_steps += 1

                if total_steps < warmup_steps:
                    a1 = np.array(
                        [
                            np.random.uniform(0.0, env.max_linear),
                            np.random.uniform(-env.max_angular, env.max_angular),
                        ],
                        dtype=np.float32,
                    )

                    a2 = np.array(
                        [
                            np.random.uniform(0.0, env.max_linear),
                            np.random.uniform(-env.max_angular, env.max_angular),
                        ],
                        dtype=np.float32,
                    )

                else:
                    a1 = agent1.select_action(obs1, noise_std)
                    a2 = agent2.select_action(obs2, noise_std)

                actions = np.concatenate([a1, a2]).astype(np.float32)

                next_obs1, next_obs2, next_global_state, rewards, dones = env.step(actions)

                replay_buffer.add(
                    obs1,
                    obs2,
                    global_state,
                    actions,
                    rewards,
                    next_obs1,
                    next_obs2,
                    next_global_state,
                    dones,
                )

                obs1 = next_obs1
                obs2 = next_obs2
                global_state = next_global_state

                ep_rewards += rewards
                ep_dones += dones.astype(np.int32)

                if replay_buffer.size >= batch_size and total_steps % train_after_every_step == 0:
                    update_maddpg(
                        agents,
                        replay_buffer,
                        batch_size,
                        device,
                    )

            noise_std = max(min_noise, noise_std * noise_decay)

            print(
                f"Episode {ep + 1:04d} | "
                f"R1: {ep_rewards[0]:8.2f} | "
                f"R2: {ep_rewards[1]:8.2f} | "
                f"Done R1: {ep_dones[0]:3d} | "
                f"Done R2: {ep_dones[1]:3d} | "
                f"Steps: {step + 1:4d} | "
                f"Noise: {noise_std:.3f}"
            )

            avg_reward = float((ep_rewards[0] + ep_rewards[1]) / 2.0)

            episode_list.append(ep + 1)
            reward_robot1_list.append(float(ep_rewards[0]))
            reward_robot2_list.append(float(ep_rewards[1]))
            avg_reward_list.append(avg_reward)

            with open(log_csv_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    ep + 1,
                    float(ep_rewards[0]),
                    float(ep_rewards[1]),
                    avg_reward,
                ])

            if (ep + 1) % 50 == 0:
                torch.save(
                    agent1.actor.state_dict(),
                    f"{save_dir}/agent1_actor_ep{ep + 1}.pth",
                )

                torch.save(
                    agent2.actor.state_dict(),
                    f"{save_dir}/agent2_actor_ep{ep + 1}.pth",
                )

                torch.save(
                    agent1.critic.state_dict(),
                    f"{save_dir}/agent1_critic_ep{ep + 1}.pth",
                )

                torch.save(
                    agent2.critic.state_dict(),
                    f"{save_dir}/agent2_critic_ep{ep + 1}.pth",
                )

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    finally:
        plot_training_curve(
            episode_list,
            reward_robot1_list,
            reward_robot2_list,
            avg_reward_list,
            plot_dir,
        )

        env.stop_robots()
        env.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    train()