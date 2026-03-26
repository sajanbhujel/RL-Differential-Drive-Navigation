# src/env/world_2d_fixed.py

from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MplRectangle, Circle


@dataclass(frozen=True)
class RectObstacle:

    x: float   # bottom-left x
    y: float   # bottom-left y
    w: float   # width
    h: float   # height

    def contains(self, px: float, py: float) -> bool:
        return (self.x <= px <= self.x + self.w) and (self.y <= py <= self.y + self.h)


class World2D:

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        start: Tuple[float, float] = (5.0, 5.0),
        goal: Tuple[float, float] = (90.0, 90.0),
        obstacles: Optional[List[RectObstacle]] = None,
        robot_radius: float = 1.0,
    ):
        self.width = float(width)
        self.height = float(height)
        self.start = (float(start[0]), float(start[1]))
        self.goal = (float(goal[0]), float(goal[1]))
        self.robot_radius = float(robot_radius)

  
        if obstacles is None:
            obstacles = [
                RectObstacle(x=20, y=20, w=15, h=40),   # tall vertical block
                RectObstacle(x=70, y=25, w=20, h=10),   # horizontal block
                RectObstacle(x=65, y=65, w=10, h=25),   # vertical near goal area
            ]
        self.obstacles = obstacles

        self._validate_world()


    def _validate_world(self) -> None:
        sx, sy = self.start
        gx, gy = self.goal
        if not self.in_bounds(sx, sy):
            raise ValueError(f"Start {self.start} is out of bounds.")
        if not self.in_bounds(gx, gy):
            raise ValueError(f"Goal {self.goal} is out of bounds.")
        if self.point_in_obstacle(sx, sy):
            raise ValueError("Start is inside an obstacle. Move start or obstacles.")
        if self.point_in_obstacle(gx, gy):
            raise ValueError("Goal is inside an obstacle. Move goal or obstacles.")

    # ---------------------- geometry helpers ----------------------
    def in_bounds(self, x: float, y: float) -> bool:
        return (0.0 <= x <= self.width) and (0.0 <= y <= self.height)

    def point_in_obstacle(self, x: float, y: float) -> bool:
        return any(obs.contains(x, y) for obs in self.obstacles)

    def robot_collides(self, x: float, y: float) -> bool:
        r = self.robot_radius
        if (x - r < 0) or (x + r > self.width) or (y - r < 0) or (y + r > self.height):
            return True

        for obs in self.obstacles:
    
            cx = min(max(x, obs.x), obs.x + obs.w)
            cy = min(max(y, obs.y), obs.y + obs.h)
            dx = x - cx
            dy = y - cy
            if (dx * dx + dy * dy) <= (r * r):
                return True
        return False


    def plot(self, robot_pos: Optional[Tuple[float, float]] = None, title: str = "World2D (Fixed Start/Goal/Obstacles)") -> None:
        fig, ax = plt.subplots(figsize=(7, 7))

     
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

     
        for obs in self.obstacles:
            rect = MplRectangle((obs.x, obs.y), obs.w, obs.h, fill=True, alpha=0.35)
            ax.add_patch(rect)


      
        sx, sy = self.start
        gx, gy = self.goal
        ax.scatter([sx], [sy], marker="o", s=80)
        ax.text(sx + 1, sy + 1, f"Start", fontsize=10)

        ax.scatter([gx], [gy], marker="*", s=140)
        ax.text(gx + 1, gy + 1, f"Goal", fontsize=10)

 
        if robot_pos is not None:
            rx, ry = robot_pos
            circ = Circle((rx, ry), radius=self.robot_radius, fill=False, linewidth=2)
            ax.add_patch(circ)
            ax.scatter([rx], [ry], s=40)
            ax.text(rx + 1, ry + 1, f"Robot ({rx:.1f},{ry:.1f})", fontsize=10)

            coll = self.robot_collides(rx, ry)


        plt.show()


if __name__ == "__main__":
    world = World2D()
    world.plot()  

