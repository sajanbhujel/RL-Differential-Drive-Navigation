from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


State = Tuple[int, int]


@dataclass
class DPConfig:
    gamma: float = 0.99
    theta: float = 1e-6         
    max_iterations: int = 10_000 
    eval_iterations: int = 200   


class DPAgent:

    def __init__(self, cfg: Optional[DPConfig] = None):
        self.cfg = cfg or DPConfig()
        self.V: Optional[np.ndarray] = None           
        self.policy: Optional[np.ndarray] = None       

  
    def value_iteration(self, env) -> Tuple[np.ndarray, np.ndarray]:
        H, W = env.height, env.width
        V = np.zeros((H, W), dtype=float)

        states = env.get_all_states()

        for it in range(self.cfg.max_iterations):
            delta = 0.0

            for (x, y) in states:
                s = (x, y)

                if env.is_terminal(s):
                    continue

                actions = env.get_actions(s)
                if not actions:
                    continue

                best_q = -np.inf
                for a in actions:
                    sp, r, done = env.transition(s, a)
                    nx, ny = sp
                    q = r + self.cfg.gamma * (0.0 if done else V[ny, nx])
                    if q > best_q:
                        best_q = q

                old = V[y, x]
                V[y, x] = best_q
                delta = max(delta, abs(old - V[y, x]))

            if delta < self.cfg.theta:
                # converged
                break

        policy = self.greedy_policy_from_value(env, V)
        self.V = V
        self.policy = policy
        return V, policy

    def greedy_policy_from_value(self, env, V: np.ndarray) -> np.ndarray:
        H, W = env.height, env.width
        policy = -np.ones((H, W), dtype=int)

        for (x, y) in env.get_all_states():
            s = (x, y)

            if env.is_terminal(s):
                policy[y, x] = -1
                continue

            actions = env.get_actions(s)
            if not actions:
                policy[y, x] = -1
                continue

            best_a = None
            best_q = -np.inf
            for a in actions:
                sp, r, done = env.transition(s, a)
                nx, ny = sp
                q = r + self.cfg.gamma * (0.0 if done else V[ny, nx])
                if q > best_q:
                    best_q = q
                    best_a = a

            policy[y, x] = int(best_a) if best_a is not None else -1

        return policy


    def policy_iteration(self, env) -> Tuple[np.ndarray, np.ndarray]:
        H, W = env.height, env.width
        V = np.zeros((H, W), dtype=float)

      
        policy = -np.ones((H, W), dtype=int)
        for (x, y) in env.get_all_states():
            s = (x, y)
            if env.is_terminal(s):
                continue
            actions = env.get_actions(s)
            if actions:
                policy[y, x] = int(np.random.choice(list(actions)))

        states = env.get_all_states()

        for it in range(self.cfg.max_iterations):
         
            for _ in range(self.cfg.eval_iterations):
                delta = 0.0
                for (x, y) in states:
                    s = (x, y)
                    if env.is_terminal(s):
                        continue
                    a = policy[y, x]
                    if a < 0:
                        continue

                    sp, r, done = env.transition(s, int(a))
                    nx, ny = sp
                    v_new = r + self.cfg.gamma * (0.0 if done else V[ny, nx])

                    old = V[y, x]
                    V[y, x] = v_new
                    delta = max(delta, abs(old - v_new))

                if delta < self.cfg.theta:
                    break

           
            stable = True
            for (x, y) in states:
                s = (x, y)
                if env.is_terminal(s):
                    continue

                old_a = policy[y, x]
                actions = env.get_actions(s)
                if not actions:
                    continue

                best_a = None
                best_q = -np.inf
                for a in actions:
                    sp, r, done = env.transition(s, a)
                    nx, ny = sp
                    q = r + self.cfg.gamma * (0.0 if done else V[ny, nx])
                    if q > best_q:
                        best_q = q
                        best_a = a

                if best_a is None:
                    continue

                policy[y, x] = int(best_a)
                if int(old_a) != int(best_a):
                    stable = False

            if stable:
                break

        self.V = V
        self.policy = policy
        return V, policy


    def plot_value(self, env, V: np.ndarray, title: str = "Value Function",
                save_path: str = None) -> None:
        fig, ax = plt.subplots(figsize=(7, 7))

        im = ax.imshow(V, origin="lower", interpolation="nearest")
        fig.colorbar(im, ax=ax)

        # obstacles
        obs_mask = np.where(env.blocked, 1.0, np.nan)
        obstacle_color = (0.4, 0.4, 0.4)
        ax.imshow(
            obs_mask,
            origin="lower",
            interpolation="nearest",
            cmap=plt.cm.colors.ListedColormap([obstacle_color]),
            vmin=0, vmax=1,
        )

        sx, sy = env.start
        gx, gy = env.goal
        ax.scatter(sx, sy, s=80, marker="o", color="blue")
        ax.scatter(gx, gy, s=120, marker="*", color="red")

        ax.set_title(title)
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_policy_arrows(self, env, policy: np.ndarray,
                        title: str = "Greedy Policy",
                        save_path: str = None) -> None:

        H, W = env.height, env.width
        fig, ax = plt.subplots(figsize=(7, 7))

        # draw obstacles
        obs_mask = np.where(env.blocked, 1.0, np.nan)
        obstacle_color = (0.4, 0.4, 0.4)
        ax.imshow(
            obs_mask,
            origin="lower",
            interpolation="nearest",
            cmap=plt.cm.colors.ListedColormap([obstacle_color]),
            vmin=0, vmax=1,
        )

        for y in range(H):
            for x in range(W):
                if env.blocked[y, x] or (x, y) == env.goal:
                    continue

                a = policy[y, x]
                if a < 0:
                    continue

                if a == 0: dx, dy = 0, 0.8
                elif a == 1: dx, dy = 0, -0.8
                elif a == 2: dx, dy = -0.8, 0
                elif a == 3: dx, dy = 0.8, 0
                else: continue

                ax.arrow(x, y, dx, dy,
                        head_width=0.6,
                        head_length=0.6,
                        length_includes_head=True)

        sx, sy = env.start
        gx, gy = env.goal
        ax.scatter(sx, sy, s=80, marker="o", color="blue")
        ax.scatter(gx, gy, s=120, marker="*", color="red")

        ax.set_title(title)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


if __name__ == "__main__":
    from src.env.grid_world import GridWorld100

    env = GridWorld100()
    agent = DPAgent(DPConfig(gamma=0.99, theta=1e-6, max_iterations=10000))

    # 1) Value Iteration
    V, pi = agent.value_iteration(env)
    agent.plot_value(env, V, title="Value Iteration: Value Function")
    agent.plot_policy_arrows(env, pi, title="Value Iteration: Greedy Policy")

    # 2) Policy Iteration (optional test)
    V2, pi2 = agent.policy_iteration(env)
    agent.plot_value(env, V2, title="Policy Iteration: Value Function")
    agent.plot_policy_arrows(env, pi2, title="Policy Iteration: Policy")

    