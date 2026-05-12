#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, max_action):
        super().__init__()

        self.register_buffer(
            "max_action",
            torch.tensor(max_action, dtype=torch.float32)
        )

        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        raw = torch.tanh(self.out(x))
        linear = (raw[:, 0:1] + 1.0) * 0.5 * self.max_action[0]
        angular = raw[:, 1:2] * self.max_action[1]

        action = torch.cat([linear, angular], dim=1)
        return action


class Critic(nn.Module):
    def __init__(self, global_state_dim, total_action_dim):
        super().__init__()

        self.fc1 = nn.Linear(global_state_dim + total_action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)

    def forward(self, global_state, actions):
        x = torch.cat([global_state, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q


def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )