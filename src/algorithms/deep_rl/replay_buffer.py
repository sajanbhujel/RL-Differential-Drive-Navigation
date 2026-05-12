#!/usr/bin/env python3

import numpy as np
import torch


class MultiAgentReplayBuffer:
    def __init__(self, max_size, obs_dim, global_state_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs1 = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.obs2 = np.zeros((max_size, obs_dim), dtype=np.float32)

        self.next_obs1 = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs2 = np.zeros((max_size, obs_dim), dtype=np.float32)

        self.global_state = np.zeros((max_size, global_state_dim), dtype=np.float32)
        self.next_global_state = np.zeros((max_size, global_state_dim), dtype=np.float32)

        self.actions = np.zeros((max_size, action_dim * 2), dtype=np.float32)

        self.rewards = np.zeros((max_size, 2), dtype=np.float32)
        self.dones = np.zeros((max_size, 2), dtype=np.float32)

    def add(
        self,
        obs1,
        obs2,
        global_state,
        actions,
        rewards,
        next_obs1,
        next_obs2,
        next_global_state,
        dones,
    ):
        self.obs1[self.ptr] = obs1
        self.obs2[self.ptr] = obs2

        self.global_state[self.ptr] = global_state
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards

        self.next_obs1[self.ptr] = next_obs1
        self.next_obs2[self.ptr] = next_obs2
        self.next_global_state[self.ptr] = next_global_state

        self.dones[self.ptr] = dones

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)

        batch = {
            "obs1": torch.tensor(self.obs1[idx], dtype=torch.float32, device=device),
            "obs2": torch.tensor(self.obs2[idx], dtype=torch.float32, device=device),

            "next_obs1": torch.tensor(self.next_obs1[idx], dtype=torch.float32, device=device),
            "next_obs2": torch.tensor(self.next_obs2[idx], dtype=torch.float32, device=device),

            "global_state": torch.tensor(self.global_state[idx], dtype=torch.float32, device=device),
            "next_global_state": torch.tensor(self.next_global_state[idx], dtype=torch.float32, device=device),

            "actions": torch.tensor(self.actions[idx], dtype=torch.float32, device=device),
            "rewards": torch.tensor(self.rewards[idx], dtype=torch.float32, device=device),
            "dones": torch.tensor(self.dones[idx], dtype=torch.float32, device=device),
        }

        return batch