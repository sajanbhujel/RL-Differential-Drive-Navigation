#!/usr/bin/env python3

import os
import time
import numpy as np

import torch
import rclpy

from networks import Actor
from multiagent_env import MultiAgentGazeboEnv


class TestAgent:
    def __init__(self, obs_dim, action_dim, max_action, model_path, device):
        self.device = device

        self.actor = Actor(obs_dim, action_dim, max_action).to(device)

        self.actor.load_state_dict(
            torch.load(model_path, map_location=device)
        )

        self.actor.eval()

        self.max_action = np.array(max_action, dtype=np.float32)

    def select_action(self, obs):
        obs_tensor = torch.tensor(
            obs,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]

        action[0] = np.clip(action[0], 0.0, self.max_action[0])
        action[1] = np.clip(action[1], -self.max_action[1], self.max_action[1])

        return action.astype(np.float32)


def test():
    rclpy.init()

    env = MultiAgentGazeboEnv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    obs_list, global_state = env.reset()

    obs_dim = obs_list[0].shape[0]
    action_dim = 2
    n_agents = env.n_agents

    max_action = np.array(
        [
            env.max_linear,
            env.max_angular,
        ],
        dtype=np.float32,
    )

    model_dir = "saved_models_4"

    model_episode = 100

    model_paths = [
        os.path.join(model_dir, f"agent1_actor_ep{model_episode}.pth"),
        os.path.join(model_dir, f"agent2_actor_ep{model_episode}.pth"),
        os.path.join(model_dir, f"agent3_actor_ep{model_episode}.pth"),
        os.path.join(model_dir, f"agent4_actor_ep{model_episode}.pth"),
    ]

    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model for agent {i + 1} not found: {path}")

    agents = []

    for i in range(n_agents):
        print(f"Loading Robot {i + 1} actor: {model_paths[i]}")

        agent = TestAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_action=max_action,
            model_path=model_paths[i],
            device=device,
        )

        agents.append(agent)

    test_episodes = 800

    total_test_rewards = np.zeros(n_agents, dtype=np.float32)
    total_done_counts = np.zeros(n_agents, dtype=np.int32)

    try:
        for ep in range(test_episodes):
            obs_list, global_state = env.reset()

            ep_rewards = np.zeros(n_agents, dtype=np.float32)
            ep_dones = np.zeros(n_agents, dtype=np.int32)

            print(f"\nTesting Episode {ep + 1}")

            for step in range(env.max_steps):
                action_list = []

                for i in range(n_agents):
                    action = agents[i].select_action(obs_list[i])
                    action_list.append(action)

                actions = np.concatenate(action_list).astype(np.float32)

                next_obs_list, next_global_state, rewards, dones = env.step(actions)

                obs_list = next_obs_list
                global_state = next_global_state

                ep_rewards += rewards
                ep_dones += dones.astype(np.int32)

                print_text = f"Step {step + 1:04d} | "

                for i in range(n_agents):
                    a = action_list[i]
                    print_text += f"A{i + 1}: [{a[0]:.3f}, {a[1]:.3f}] | "

                for i in range(n_agents):
                    print_text += f"R{i + 1}: {rewards[i]:7.2f} | "

                for i in range(n_agents):
                    print_text += f"D{i + 1}: {int(dones[i])} | "

                print(print_text)

                time.sleep(0.02)

            total_test_rewards += ep_rewards
            total_done_counts += ep_dones

            print_text = f"Test Episode {ep + 1:03d} finished | "

            for i in range(n_agents):
                print_text += f"R{i + 1} Total: {ep_rewards[i]:8.2f} | "

            for i in range(n_agents):
                print_text += f"Done R{i + 1}: {ep_dones[i]} | "

            print(print_text)

        print("\n========== TEST SUMMARY ==========")
        print(f"Test episodes: {test_episodes}")

        for i in range(n_agents):
            print(
                f"Average Robot {i + 1} reward: "
                f"{total_test_rewards[i] / test_episodes:.2f}"
            )

        for i in range(n_agents):
            print(
                f"Total Robot {i + 1} done count: "
                f"{total_done_counts[i]}"
            )

    except KeyboardInterrupt:
        print("Testing interrupted by user.")

    finally:
        env.stop_robots()
        env.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    test()
