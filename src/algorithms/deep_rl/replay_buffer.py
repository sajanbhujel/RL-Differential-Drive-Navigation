#!/usr/bin/env python3

import numpy as np
import torch


class MultiAgentReplayBuffer:
    def __init__(self, max_size, n_agents, obs_dim, global_state_dim, action_dim):
        self.max_size = max_size
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((max_size, n_agents, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, n_agents, obs_dim), dtype=np.float32)

        self.global_state = np.zeros((max_size, global_state_dim), dtype=np.float32)
        self.next_global_state = np.zeros((max_size, global_state_dim), dtype=np.float32)

        self.actions = np.zeros((max_size, action_dim * n_agents), dtype=np.float32)
        self.rewards = np.zeros((max_size, n_agents), dtype=np.float32)
        self.dones = np.zeros((max_size, n_agents), dtype=np.float32)

    def add(
        self,
        obs_list,
        global_state,
        actions,
        rewards,
        next_obs_list,
        next_global_state,
        dones,
    ):
        self.obs[self.ptr] = np.array(obs_list, dtype=np.float32)
        self.global_state[self.ptr] = global_state
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards

        self.next_obs[self.ptr] = np.array(next_obs_list, dtype=np.float32)
        self.next_global_state[self.ptr] = next_global_state
        self.dones[self.ptr] = dones

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)

        return {
            "obs": torch.tensor(self.obs[idx], dtype=torch.float32, device=device),
            "next_obs": torch.tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "global_state": torch.tensor(self.global_state[idx], dtype=torch.float32, device=device),
            "next_global_state": torch.tensor(self.next_global_state[idx], dtype=torch.float32, device=device),
            "actions": torch.tensor(self.actions[idx], dtype=torch.float32, device=device),
            "rewards": torch.tensor(self.rewards[idx], dtype=torch.float32, device=device),
            "dones": torch.tensor(self.dones[idx], dtype=torch.float32, device=device),
        }
