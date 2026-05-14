#!/usr/bin/env python3

import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import rclpy

from networks import Actor
from multiagent_env import MultiAgentGazeboEnv


class SaliencyAgent:
    def __init__(self, obs_dim, action_dim, max_action, model_path, device):
        self.device = device

        self.actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.actor.load_state_dict(torch.load(model_path, map_location=device))
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

    def compute_saliency(self, obs):
        """
        Returns saliency of each input feature for:
        1. linear velocity output
        2. angular velocity output
        3. combined output
        """


        obs_tensor = torch.tensor(
            obs,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

   
        obs_tensor.requires_grad_(True)

        action = self.actor(obs_tensor)


        self.actor.zero_grad()

        if obs_tensor.grad is not None:
            obs_tensor.grad.zero_()

        linear_output = action[0, 0]
        linear_output.backward(retain_graph=True)

        linear_saliency = obs_tensor.grad.abs().detach().cpu().numpy()[0].copy()


        self.actor.zero_grad()
        obs_tensor.grad.zero_()

        angular_output = action[0, 1]
        angular_output.backward()

        angular_saliency = obs_tensor.grad.abs().detach().cpu().numpy()[0].copy()

        # Combined saliency
        combined_saliency = linear_saliency + angular_saliency

        return linear_saliency, angular_saliency, combined_saliency


def get_feature_groups(obs_dim, lidar_bins):
    """
    For 4-agent environment:
    obs = [
        lidar_bins,
        x,
        y,
        yaw,
        dist_to_goal,
        goal_angle,
        rel_x_y to 3 other robots = 6 values,
        linear velocity,
        angular velocity
    ]

    If lidar_bins = 100, obs_dim = 113.
    """

    groups = {}

    groups["lidar"] = list(range(0, lidar_bins))

    idx = lidar_bins

    groups["x_position"] = [idx]
    idx += 1

    groups["y_position"] = [idx]
    idx += 1

    groups["yaw_heading"] = [idx]
    idx += 1

    groups["distance_to_goal"] = [idx]
    idx += 1

    groups["angle_to_goal"] = [idx]
    idx += 1

    groups["relative_other_robots"] = list(range(idx, idx + 6))
    idx += 6

    groups["linear_velocity"] = [idx]
    idx += 1

    groups["angular_velocity"] = [idx]
    idx += 1

    if idx != obs_dim:
        print(f"WARNING: Expected obs_dim {idx}, but got {obs_dim}.")
        print("Check lidar_bins or observation structure.")

    return groups


def summarize_group_saliency(saliency, groups):
    summary = {}

    for group_name, indices in groups.items():
        values = saliency[indices]
        summary[group_name] = float(np.mean(values))

    return summary


def save_group_saliency_csv(path, group_results, robot_names):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    group_names = list(group_results[robot_names[0]].keys())

    with open(path, mode="w", newline="") as file:
        writer = csv.writer(file)

        header = ["group"]

        for robot in robot_names:
            header.append(robot)

        writer.writerow(header)

        for group in group_names:
            row = [group]

            for robot in robot_names:
                row.append(group_results[robot][group])

            writer.writerow(row)

    print(f"Saved group saliency CSV: {path}")


def save_feature_saliency_csv(path, feature_saliency_results, robot_names):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    obs_dim = len(feature_saliency_results[robot_names[0]])

    with open(path, mode="w", newline="") as file:
        writer = csv.writer(file)

        header = ["feature_index"]

        for robot in robot_names:
            header.append(robot)

        writer.writerow(header)

        for i in range(obs_dim):
            row = [i]

            for robot in robot_names:
                row.append(float(feature_saliency_results[robot][i]))

            writer.writerow(row)

    print(f"Saved feature saliency CSV: {path}")


def plot_group_saliency(group_results, robot_names, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    group_names = list(group_results[robot_names[0]].keys())

    x = np.arange(len(group_names))
    width = 0.18

    plt.figure(figsize=(13, 6))

    for i, robot in enumerate(robot_names):
        values = [group_results[robot][g] for g in group_names]
        plt.bar(x + i * width, values, width, label=robot)

    plt.xticks(x + width * 1.5, group_names, rotation=35, ha="right")
    plt.ylabel("Mean Absolute Gradient Saliency")
    plt.title("Input Group Saliency for 4-Agent MADDPG Actors")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved group saliency plot: {output_path}")


def plot_lidar_saliency(feature_saliency_results, robot_names, lidar_bins, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for robot in robot_names:
        lidar_saliency = feature_saliency_results[robot][:lidar_bins]

        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(lidar_bins), lidar_saliency, linewidth=2)
        plt.xlabel("LiDAR Bin Index")
        plt.ylabel("Mean Absolute Gradient Saliency")
        plt.title(f"{robot} LiDAR Saliency")
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{robot}_lidar_saliency.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved LiDAR saliency plot: {save_path}")


def saliency_test():
    rclpy.init()

    env = MultiAgentGazeboEnv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    obs_list, global_state = env.reset()

    obs_dim = obs_list[0].shape[0]
    action_dim = 2
    n_agents = env.n_agents
    robot_names = env.robot_names
    lidar_bins = env.lidar_bins

    print(f"Number of agents: {n_agents}")
    print(f"Observation dimension: {obs_dim}")
    print(f"LiDAR bins: {lidar_bins}")

    max_action = np.array(
        [
            env.max_linear,
            env.max_angular,
        ],
        dtype=np.float32,
    )

    # ==========================
    # CHANGE THESE IF NEEDED
    # ==========================
    model_episode = 150
    model_dir = "saved_models_4"
    saliency_steps = 300
    output_dir = "saliency_results"
    # ==========================

    agents = []

    for i in range(n_agents):
        model_path = os.path.join(
            model_dir,
            f"agent{i + 1}_actor_ep{model_episode}.pth"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading {robot_names[i]} actor: {model_path}")

        agent = SaliencyAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_action=max_action,
            model_path=model_path,
            device=device,
        )

        agents.append(agent)

    feature_saliency_sum = {
        robot: np.zeros(obs_dim, dtype=np.float64)
        for robot in robot_names
    }

    linear_saliency_sum = {
        robot: np.zeros(obs_dim, dtype=np.float64)
        for robot in robot_names
    }

    angular_saliency_sum = {
        robot: np.zeros(obs_dim, dtype=np.float64)
        for robot in robot_names
    }

    try:
        obs_list, global_state = env.reset()

        for step in range(saliency_steps):
            action_list = []

            for i in range(n_agents):
                linear_sal, angular_sal, combined_sal = agents[i].compute_saliency(obs_list[i])

                robot = robot_names[i]

                linear_saliency_sum[robot] += linear_sal
                angular_saliency_sum[robot] += angular_sal
                feature_saliency_sum[robot] += combined_sal

                action = agents[i].select_action(obs_list[i])
                action_list.append(action)

            actions = np.concatenate(action_list).astype(np.float32)

            next_obs_list, next_global_state, rewards, dones = env.step(actions)

            obs_list = next_obs_list
            global_state = next_global_state

            if (step + 1) % 50 == 0:
                print(f"Collected saliency step {step + 1}/{saliency_steps}")

            time.sleep(0.01)

        feature_saliency_avg = {
            robot: feature_saliency_sum[robot] / saliency_steps
            for robot in robot_names
        }

        linear_saliency_avg = {
            robot: linear_saliency_sum[robot] / saliency_steps
            for robot in robot_names
        }

        angular_saliency_avg = {
            robot: angular_saliency_sum[robot] / saliency_steps
            for robot in robot_names
        }

        groups = get_feature_groups(obs_dim, lidar_bins)

        group_results = {
            robot: summarize_group_saliency(feature_saliency_avg[robot], groups)
            for robot in robot_names
        }

        os.makedirs(output_dir, exist_ok=True)

        save_group_saliency_csv(
            os.path.join(output_dir, "group_saliency.csv"),
            group_results,
            robot_names,
        )

        save_feature_saliency_csv(
            os.path.join(output_dir, "feature_saliency.csv"),
            feature_saliency_avg,
            robot_names,
        )

        save_feature_saliency_csv(
            os.path.join(output_dir, "linear_action_feature_saliency.csv"),
            linear_saliency_avg,
            robot_names,
        )

        save_feature_saliency_csv(
            os.path.join(output_dir, "angular_action_feature_saliency.csv"),
            angular_saliency_avg,
            robot_names,
        )

        plot_group_saliency(
            group_results,
            robot_names,
            os.path.join(output_dir, "group_saliency.png"),
        )

        plot_lidar_saliency(
            feature_saliency_avg,
            robot_names,
            lidar_bins,
            output_dir,
        )

        print("\n========== SALIENCY TEST FINISHED ==========")
        print(f"Results saved in: {output_dir}")

    except KeyboardInterrupt:
        print("Saliency test interrupted by user.")

    finally:
        env.stop_robots()
        env.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    saliency_test()
