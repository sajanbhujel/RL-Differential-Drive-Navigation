#!/usr/bin/env python3

import math
import time
import subprocess
import threading
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


GAZEBO_POSE_TOPIC = "/world/training_arena/dynamic_pose/info"


def quaternion_to_yaw(qx, qy, qz, qw):
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz)
    )
    return yaw


def get_value(block, key):
    for line in block:
        line = line.strip()

        if line.startswith(key + ":"):
            value = line.split(":", 1)[1].strip().replace('"', "")
            return float(value)

    return 0.0


def parse_pose_block(block):
    name = None

    position_block = []
    orientation_block = []

    inside_position = False
    inside_orientation = False

    for line in block:
        line = line.strip()

        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip().replace('"', "")

        elif line.startswith("position"):
            inside_position = True
            inside_orientation = False
            continue

        elif line.startswith("orientation"):
            inside_position = False
            inside_orientation = True
            continue

        elif line.startswith("}"):
            inside_position = False
            inside_orientation = False
            continue

        if inside_position:
            position_block.append(line)

        if inside_orientation:
            orientation_block.append(line)

    x = get_value(position_block, "x")
    y = get_value(position_block, "y")
    z = get_value(position_block, "z")

    qx = get_value(orientation_block, "x")
    qy = get_value(orientation_block, "y")
    qz = get_value(orientation_block, "z")
    qw = get_value(orientation_block, "w")

    yaw = quaternion_to_yaw(qx, qy, qz, qw)

    return name, x, y, z, yaw


class GazeboGlobalPoseReader:
    def __init__(self):
        self.pose = {
            "robot1": np.zeros(3, dtype=np.float32),
            "robot2": np.zeros(3, dtype=np.float32),
        }

        self.has_pose = {
            "robot1": False,
            "robot2": False,
        }

        self.lock = threading.Lock()
        self.running = True
        self.process = None

        self.thread = threading.Thread(
            target=self.read_gazebo_pose,
            daemon=True
        )
        self.thread.start()

    def read_gazebo_pose(self):
        print(f"Reading direct Gazebo global pose from: {GAZEBO_POSE_TOPIC}")

        self.process = subprocess.Popen(
            ["gz", "topic", "-e", "-t", GAZEBO_POSE_TOPIC],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        inside_pose = False
        brace_count = 0
        pose_block = []

        while self.running:
            line = self.process.stdout.readline()

            if not line:
                time.sleep(0.001)
                continue

            stripped = line.strip()

            if stripped.startswith("pose {"):
                inside_pose = True
                brace_count = 1
                pose_block = []
                continue

            if inside_pose:
                pose_block.append(line)

                brace_count += line.count("{")
                brace_count -= line.count("}")

                if brace_count == 0:
                    name, x, y, z, yaw = parse_pose_block(pose_block)

                    with self.lock:
                        if name == "edubot1":
                            self.pose["robot1"] = np.array([x, y, yaw], dtype=np.float32)
                            self.has_pose["robot1"] = True

                        elif name == "edubot2":
                            self.pose["robot2"] = np.array([x, y, yaw], dtype=np.float32)
                            self.has_pose["robot2"] = True

                    inside_pose = False
                    pose_block = []

    def get_pose(self, robot_name):
        with self.lock:
            return self.pose[robot_name].copy(), self.has_pose[robot_name]

    def stop(self):
        self.running = False

        if self.process is not None:
            self.process.terminate()

            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


class MultiAgentGazeboEnv(Node):
    def __init__(self):
        super().__init__("multiagent_rl_env")

        self.robot_names = ["robot1", "robot2"]

        self.gazebo_pose_reader = GazeboGlobalPoseReader()

        self.goals = {
            "robot1": np.array([2.5, -1.5], dtype=np.float32),
            "robot2": np.array([-2.0, -2.0], dtype=np.float32),
        }

        self.start_poses = {
            "robot1": [-2.0, 2.0, 0.15, 0.0],
            "robot2": [2.5, 2.5, 0.15, -2.0],
        }

        self.goal_radius = 0.4

        self.collision_dist = 0.18
        self.near_obstacle_dist = 0.3
        self.robot_collision_dist = 0.30

        self.max_steps = 500

        # independent step counters
        self.robot_step_count = {
            "robot1": 0,
            "robot2": 0,
        }

        self.lidar_bins = 100
        self.lidar_max = 8.0

        self.max_linear = 0.35
        self.max_angular = 0.5

        self.scans = {
            "robot1": np.ones(360, dtype=np.float32) * self.lidar_max,
            "robot2": np.ones(360, dtype=np.float32) * self.lidar_max,
        }

        self.poses = {
            "robot1": np.zeros(3, dtype=np.float32),
            "robot2": np.zeros(3, dtype=np.float32),
        }

        self.has_pose = {
            "robot1": False,
            "robot2": False,
        }

        self.vels = {
            "robot1": np.zeros(2, dtype=np.float32),
            "robot2": np.zeros(2, dtype=np.float32),
        }

        self.previous_goal_dist = {
            "robot1": None,
            "robot2": None,
        }

        self.cmd_pubs = {}

        for name in self.robot_names:
            self.cmd_pubs[name] = self.create_publisher(
                Twist,
                f"/{name}/cmd_vel",
                10,
            )

            self.create_subscription(
                LaserScan,
                f"/{name}/scan",
                lambda msg, n=name: self.scan_callback(msg, n),
                10,
            )

        time.sleep(2.0)

        self.get_logger().info("MultiAgentGazeboEnv started.")
        self.get_logger().info("Using direct Gazebo global position from Gazebo, not /odom.")

    def scan_callback(self, msg, robot_name):
        ranges = np.array(msg.ranges, dtype=np.float32)

        ranges = np.nan_to_num(
            ranges,
            nan=self.lidar_max,
            posinf=self.lidar_max,
            neginf=0.0,
        )

        ranges = np.clip(ranges, 0.0, self.lidar_max)

        self.scans[robot_name] = ranges

    def update_global_poses(self):
        for robot in self.robot_names:
            pose, has_pose = self.gazebo_pose_reader.get_pose(robot)

            self.poses[robot] = pose
            self.has_pose[robot] = has_pose

    def wait_for_global_poses(self, timeout=5.0):
        start = time.time()

        while time.time() - start < timeout:
            self.update_global_poses()

            if self.has_pose["robot1"] and self.has_pose["robot2"]:
                return True

            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().warn("Timeout waiting for direct Gazebo global pose.")
        return False

    def angle_normalize(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def downsample_scan(self, scan):
        chunks = np.array_split(scan, self.lidar_bins)
        reduced = np.array([np.min(c) for c in chunks], dtype=np.float32)
        reduced = reduced / self.lidar_max
        return reduced

    def get_obs(self, robot_name):
        self.update_global_poses()

        other = "robot2" if robot_name == "robot1" else "robot1"

        scan = self.downsample_scan(self.scans[robot_name])

        x, y, yaw = self.poses[robot_name]
        ox, oy, _ = self.poses[other]

        goal = self.goals[robot_name]

        dx = goal[0] - x
        dy = goal[1] - y

        dist_to_goal = math.sqrt(dx * dx + dy * dy)
        goal_angle = self.angle_normalize(math.atan2(dy, dx) - yaw)

        rel_x = ox - x
        rel_y = oy - y

        lin, ang = self.vels[robot_name]

        obs_extra = np.array(
            [
                x / 4.5,
                y / 4.5,
                yaw / math.pi,
                dist_to_goal / 8.0,
                goal_angle / math.pi,
                rel_x / 8.0,
                rel_y / 8.0,
                lin / self.max_linear,
                ang / self.max_angular,
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([scan, obs_extra]).astype(np.float32)
        return obs

    def get_all_obs(self):
        self.update_global_poses()

        obs1 = self.get_obs("robot1")
        obs2 = self.get_obs("robot2")

        global_state = np.concatenate([obs1, obs2]).astype(np.float32)

        return obs1, obs2, global_state

    def publish_action(self, robot_name, action):
        linear = float(np.clip(action[0], 0.0, self.max_linear))
        angular = float(np.clip(action[1], -self.max_angular, self.max_angular))

        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular

        self.cmd_pubs[robot_name].publish(msg)

        self.vels[robot_name] = np.array([linear, angular], dtype=np.float32)

    def step(self, actions):
        self.update_global_poses()

        action1 = actions[:2]
        action2 = actions[2:]

        self.publish_action("robot1", action1)
        self.publish_action("robot2", action2)

        for robot in self.robot_names:
            self.robot_step_count[robot] += 1

        start = time.time()

        while time.time() - start < 0.1:
            rclpy.spin_once(self, timeout_sec=0.01)
            self.update_global_poses()

        rewards = np.zeros(2, dtype=np.float32)
        dones = np.zeros(2, dtype=np.float32)

        for i, robot in enumerate(self.robot_names):
            x, y, _ = self.poses[robot]
            goal = self.goals[robot]

            current_pos = np.array([x, y], dtype=np.float32)
            dist_to_goal = float(np.linalg.norm(goal - current_pos))

            min_scan = float(np.min(self.scans[robot]))

            if self.previous_goal_dist[robot] is None:
                self.previous_goal_dist[robot] = dist_to_goal

            progress = self.previous_goal_dist[robot] - dist_to_goal
            self.previous_goal_dist[robot] = dist_to_goal

            
            rewards[i] = -0.01
            rewards[i] += 10.0 * progress
            rewards[i] -= 0.05 * dist_to_goal

        
            if min_scan < self.near_obstacle_dist:
                rewards[i] -= 0.2

    
            if abs(self.vels[robot][0]) < 0.01:
                rewards[i] -= 0.05


            if dist_to_goal < self.goal_radius:
                print(f"{robot} reached goal | distance={dist_to_goal:.3f}")
                rewards[i] += 500.0
                dones[i] = 1.0

            
            if min_scan < self.collision_dist:
                print(f"{robot} obstacle collision | min_scan={min_scan:.3f}")
                rewards[i] -= 100.0
                dones[i] = 1.0

        
            if self.robot_step_count[robot] >= self.max_steps:
                print(f"{robot} reached max steps")
                rewards[i] -= 50.0
                dones[i] = 1.0

        p1 = self.poses["robot1"][:2]
        p2 = self.poses["robot2"][:2]

        robot_dist = float(np.linalg.norm(p1 - p2))

        robot_robot_collision = False

        if robot_dist < self.robot_collision_dist:
            print(f"robot-robot collision | distance={robot_dist:.3f}")

            rewards[0] -= 100.0
            rewards[1] -= 100.0

            dones[0] = 1.0
            dones[1] = 1.0

            robot_robot_collision = True

        
        if robot_robot_collision:
            print("Resetting BOTH robots due to robot-robot collision")
            self.reset_both_robots()

        else:
            if dones[0] == 1.0:
                print("Resetting robot1 only")
                self.reset_one_robot("robot1")

            if dones[1] == 1.0:
                print("Resetting robot2 only")
                self.reset_one_robot("robot2")

        next_obs1, next_obs2, next_global_state = self.get_all_obs()

        return next_obs1, next_obs2, next_global_state, rewards, dones

    def reset_one_robot(self, robot_name):
        self.publish_action(robot_name, np.array([0.0, 0.0], dtype=np.float32))

        if robot_name == "robot1":
            model_name = "edubot1"
        else:
            model_name = "edubot2"

        x, y, z, yaw = self.start_poses[robot_name]

        self.set_model_pose(model_name, x, y, z, yaw)

        self.robot_step_count[robot_name] = 0
        self.previous_goal_dist[robot_name] = None

        time.sleep(0.3)

        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.02)
            self.update_global_poses()

        self.publish_action(robot_name, np.array([0.0, 0.0], dtype=np.float32))

    def reset_both_robots(self):
        self.stop_robots()

        self.set_model_pose("edubot1", -2.0, 2.0, 0.15, 0.0)
        self.set_model_pose("edubot2", 2.5, 2.5, 0.15, -2.3)

        self.robot_step_count["robot1"] = 0
        self.robot_step_count["robot2"] = 0

        self.previous_goal_dist["robot1"] = None
        self.previous_goal_dist["robot2"] = None

        time.sleep(0.5)

        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.02)
            self.update_global_poses()

        self.stop_robots()

    def reset(self):
        self.stop_robots()

        self.set_model_pose("edubot1", -2.0, 2.0, 0.15, 0.0)
        self.set_model_pose("edubot2", 2.5, 2.5, 0.15, -2.3)

        self.robot_step_count["robot1"] = 0
        self.robot_step_count["robot2"] = 0

        self.previous_goal_dist["robot1"] = None
        self.previous_goal_dist["robot2"] = None

        time.sleep(1.0)

        for _ in range(30):
            rclpy.spin_once(self, timeout_sec=0.05)
            self.update_global_poses()

        self.wait_for_global_poses(timeout=5.0)

        return self.get_all_obs()

    def stop_robots(self):
        zero = np.array([0.0, 0.0], dtype=np.float32)

        for name in self.robot_names:
            self.publish_action(name, zero)

    def set_model_pose(self, model_name, x, y, z, yaw):
        req = (
            f'name: "{model_name}" '
            f'position {{ x: {x} y: {y} z: {z} }} '
            f'orientation {{ z: {math.sin(yaw / 2.0)} w: {math.cos(yaw / 2.0)} }}'
        )

        cmd = [
            "gz",
            "service",
            "-s",
            "/world/training_arena/set_pose",
            "--reqtype",
            "gz.msgs.Pose",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "1000",
            "--req",
            req,
        ]

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def destroy_node(self):
        self.gazebo_pose_reader.stop()
        super().destroy_node()