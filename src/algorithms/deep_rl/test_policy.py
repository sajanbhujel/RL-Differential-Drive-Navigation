#!/usr/bin/env python3

import os
import csv
import time
import numpy as np

import torch
import rclpy

from networks import Actor
from multiagent_env import MultiAgentGazeboEnv


class ActorOnlyAgent:
    def __init__(self, obs_dim, action_dim, max_action, model_path, device):
        self.device = device
        self.max_action = np.array(max_action, dtype=np.float32)

        self.actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.actor.load_state_dict(torch.load(model_path, map_location=device))
        self.actor.eval()

    def select_action(self, local_obs):
        obs_tensor = torch.tensor(
            local_obs,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]

        action[0] = np.clip(action[0], 0.0, self.max_action[0])
        action[1] = np.clip(action[1], -self.max_action[1], self.max_action[1])

        return action.astype(np.float32)



def test_actor_only():
    rclpy.init()

    env = MultiAgentGazeboEnv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Evaluation mode: ACTOR ONLY")
    print("Centralized critic is NOT used.")
    print("Global state is NOT used for action selection.")

    obs_list, _ = env.reset()

    obs_dim = obs_list[0].shape[0]
    action_dim = 2
    n_agents = env.n_agents

    max_action = np.array(
        [env.max_linear, env.max_angular],
        dtype=np.float32
    )

    model_dir = "saved_models"
    model_episode = 1000

    agents = []

    for i in range(n_agents):
        model_path = os.path.join(
            model_dir,
            f"agent{i + 1}_actor_ep{model_episode}.pth"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading actor policy for robot {i + 1}: {model_path}")

        agents.append(
            ActorOnlyAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                max_action=max_action,
                model_path=model_path,
                device=device,
            )
        )

    test_episodes = 10

    total_rewards = np.zeros(n_agents, dtype=np.float32)
    success_counts = np.zeros(n_agents, dtype=np.int32)
    collision_counts = np.zeros(n_agents, dtype=np.int32)
    done_counts = np.zeros(n_agents, dtype=np.int32)

    episode_rows = []

    try:
        for ep in range(test_episodes):
            obs_list, _ = env.reset()

            ep_rewards = np.zeros(n_agents, dtype=np.float32)
            ep_success = np.zeros(n_agents, dtype=np.int32)
            ep_collision = np.zeros(n_agents, dtype=np.int32)
            ep_done = np.zeros(n_agents, dtype=np.int32)

            print(f"\nActor-only test episode {ep + 1}")

            for step in range(env.max_steps):
                action_list = []

                for i in range(n_agents):
                    action = agents[i].select_action(obs_list[i])
                    action_list.append(action)

                actions = np.concatenate(action_list).astype(np.float32)

                next_obs_list, _, rewards, dones = env.step(actions)

                for i in range(n_agents):
                    if dones[i] == 1.0:
                        ep_done[i] += 1
                        done_counts[i] += 1

                        if env.last_success[i] == 1:
                            ep_success[i] += 1
                            success_counts[i] += 1

                            print(f"Robot {i + 1} SUCCESS at step {step + 1}")

                        elif env.last_collision[i] == 1:
                            ep_collision[i] += 1
                            collision_counts[i] += 1

                            print(f"Robot {i + 1} COLLISION at step {step + 1}")

                        elif env.last_timeout[i] == 1:
                            print(f"Robot {i + 1} TIMEOUT at step {step + 1}")

                obs_list = next_obs_list
                ep_rewards += rewards

                time.sleep(0.02)

            total_rewards += ep_rewards

            print_text = f"Episode {ep + 1:03d} | "

            row = {"episode": ep + 1}

            for i in range(n_agents):
                row[f"robot{i + 1}_reward"] = float(ep_rewards[i])
                row[f"robot{i + 1}_done"] = int(ep_done[i])
                row[f"robot{i + 1}_success"] = int(ep_success[i])
                row[f"robot{i + 1}_collision"] = int(ep_collision[i])

                print_text += (
                    f"R{i + 1} Reward: {ep_rewards[i]:8.2f} | "
                    f"Done: {ep_done[i]} | "
                    f"Success: {ep_success[i]} | "
                    f"Collision: {ep_collision[i]} | "
                )

            episode_rows.append(row)
            print(print_text)

        print("\n========== ACTOR-ONLY TEST SUMMARY ==========")
        print(f"Test episodes: {test_episodes}")
        print("Centralized critic used during testing: NO")
        print("Global state used for action selection: NO")
        print("Each robot used only its own local observation.")

        for i in range(n_agents):
            avg_reward = total_rewards[i] / test_episodes

            if done_counts[i] > 0:
                success_rate = 100.0 * success_counts[i] / done_counts[i]
            else:
                success_rate = 0.0

            print(
                f"Robot {i + 1}: "
                f"Average reward = {avg_reward:.2f}, "
                f"Done count = {done_counts[i]}, "
                f"Success count = {success_counts[i]}, "
                f"Collision count = {collision_counts[i]}, "
                f"Success rate = {success_rate:.2f}%"
            )

        with open("actor_only_test_results.csv", "w", newline="") as f:
            fieldnames = ["episode"]

            for i in range(n_agents):
                fieldnames += [
                    f"robot{i + 1}_reward",
                    f"robot{i + 1}_done",
                    f"robot{i + 1}_success",
                    f"robot{i + 1}_collision",
                ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_rows)

        with open("actor_only_test_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "robot",
                "test_episodes",
                "average_reward",
                "done_count",
                "success_count",
                "collision_count",
                "success_rate_percent",
            ])

            for i in range(n_agents):
                avg_reward = total_rewards[i] / test_episodes

                if done_counts[i] > 0:
                    success_rate = 100.0 * success_counts[i] / done_counts[i]
                else:
                    success_rate = 0.0

                writer.writerow([
                    f"robot{i + 1}",
                    test_episodes,
                    float(avg_reward),
                    int(done_counts[i]),
                    int(success_counts[i]),
                    int(collision_counts[i]),
                    float(success_rate),
                ])

        print("\nSaved:")
        print("  actor_only_test_results.csv")
        print("  actor_only_test_summary.csv")

    except KeyboardInterrupt:
        print("Testing interrupted.")

    finally:
        env.stop_robots()
        env.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    test_actor_only()
