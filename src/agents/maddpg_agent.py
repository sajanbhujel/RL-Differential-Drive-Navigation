#!/usr/bin/env python3

import copy
import numpy as np

import torch
import torch.optim as optim

from src.algorithms.deep_rl.networks import Actor, Critic


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