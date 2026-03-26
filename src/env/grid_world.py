from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RectObstacle:
    x: float   # bottom-left x
    y: float   # bottom-left y
    w: float   # width
    h: float   # height


class GridWorld100:

    # action ids
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    ACTIONS = (UP, DOWN, LEFT, RIGHT)
    ACTION_TO_DELTA = {
        UP: (0, +1),
        DOWN: (0, -1),
        LEFT: (-1, 0),
        RIGHT: (+1, 0),
    }

    def __init__(
        self,
        width: int = 100,
        height: int = 100,
        start: Tuple[int, int] = (2, 2),
        goal: Tuple[int, int] = (90, 90),
        obstacles: Optional[List[RectObstacle]] = None,
        step_reward: float = -1.0,
        goal_reward: float = 100.0,
        invalid_move_penalty: float = -5.0,
        gamma: float = 0.99,
    ):
        self.width = int(width)
        self.height = int(height)
        self.start = (int(start[0]), int(start[1]))
        self.goal = (int(goal[0]), int(goal[1]))

   
        if obstacles is None:
            obstacles = [
                RectObstacle(x=20, y=20, w=15, h=40),   
                RectObstacle(x=70, y=25, w=20, h=10),  
                RectObstacle(x=65, y=65, w=10, h=25),   
            ]
        self.obstacles = obstacles

        self.step_reward = float(step_reward)
        self.goal_reward = float(goal_reward)
        self.invalid_move_penalty = float(invalid_move_penalty)
        self.gamma = float(gamma)

    
        self.blocked = np.zeros((self.height, self.width), dtype=bool)
        self._rasterize_obstacles()

   
        if not self._in_bounds(*self.start):
            raise ValueError(f"Start {self.start} out of bounds.")
        if not self._in_bounds(*self.goal):
            raise ValueError(f"Goal {self.goal} out of bounds.")
        if self.is_blocked(*self.start):
            raise ValueError(f"Start {self.start} is inside an obstacle.")
        if self.is_blocked(*self.goal):
            raise ValueError(f"Goal {self.goal} is inside an obstacle.")

        self.state = self.start 

  
    def _rasterize_obstacles(self) -> None:
        for obs in self.obstacles:
            x0 = max(0, int(np.floor(obs.x)))
            x1 = min(self.width - 1, int(np.ceil(obs.x + obs.w)))
            y0 = max(0, int(np.floor(obs.y)))
            y1 = min(self.height - 1, int(np.ceil(obs.y + obs.h)))

            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    cx, cy = x + 0.5, y + 0.5
                    if (obs.x <= cx < obs.x + obs.w) and (obs.y <= cy < obs.y + obs.h):
                        self.blocked[y, x] = True


    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_blocked(self, x: int, y: int) -> bool:
        return bool(self.blocked[y, x])

    def is_terminal(self, s: Tuple[int, int]) -> bool:
        return s == self.goal

    def reset(self) -> Tuple[int, int]:
        self.state = self.start
        return self.state


    def get_actions(self, s: Tuple[int, int]) -> Tuple[int, ...]:
        if self.is_terminal(s):
            return tuple()
        return self.ACTIONS

    def get_all_states(self) -> List[Tuple[int, int]]:
        states: List[Tuple[int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                if not self.blocked[y, x]:
                    states.append((x, y))
        return states

    def transition(self, s: Tuple[int, int], a: int) -> Tuple[Tuple[int, int], float, bool]:
        if self.is_terminal(s):
            return s, 0.0, True

        dx, dy = self.ACTION_TO_DELTA[a]
        nx, ny = s[0] + dx, s[1] + dy

        # invalid moves: stay
        if (not self._in_bounds(nx, ny)) or self.is_blocked(nx, ny):
            return s, self.invalid_move_penalty, False

        sp = (nx, ny)
        if sp == self.goal:
            return sp, self.goal_reward, True

        return sp, self.step_reward, False


    def step(self, a: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        sp, r, done = self.transition(self.state, a)
        self.state = sp
        return sp, r, done, {}


    def render(self, values=None, title="GridWorld 100x100"):
        fig, ax = plt.subplots(figsize=(8, 4))

        obstacle_color = (0.4, 0.4, 0.4)  # legend color


        obs_mask = np.where(self.blocked, 1.0, np.nan)

        if values is not None:
            im = ax.imshow(values, origin="lower", interpolation="nearest")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


            ax.imshow(
                obs_mask,
                origin="lower",
                interpolation="nearest",
                cmap=plt.cm.colors.ListedColormap([obstacle_color]),
                vmin=0, vmax=1,
                alpha=1.0
            )
        else:

            ax.set_facecolor("white")


            ax.imshow(
                obs_mask,
                origin="lower",
                interpolation="nearest",
                cmap=plt.cm.colors.ListedColormap([obstacle_color]),
                vmin=0, vmax=1,
                alpha=1.0
            )


        sx, sy = self.start
        gx, gy = self.goal
        ax.scatter(sx, sy, s=80, marker="o", color="blue")
        ax.scatter(gx, gy, s=120, marker="*", color="red")


        import matplotlib.patches as mpatches
        start_handle = plt.Line2D([0], [0], marker="o", color="w",
                                markerfacecolor="blue", markersize=10, label="Start")
        goal_handle = plt.Line2D([0], [0], marker="*", color="w",
                                markerfacecolor="red", markersize=12, label="Goal")
        obstacle_patch = mpatches.Patch(color=obstacle_color, label="Obstacle")

        ax.legend(
            handles=[start_handle, goal_handle, obstacle_patch],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=3,
            frameon=False
        )

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    env = GridWorld100()
    env.render(title="GridWorld100: start=(2,2), goal=(90,90), 3 rectangular obstacles")