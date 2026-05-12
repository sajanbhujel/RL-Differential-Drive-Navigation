# Technical Challenges and Development Notes

This file records the technical challenges, bugs, design decisions, and lessons learned while developing the multi-agent reinforcement learning project. The purpose of this file is to document the problems encountered during implementation so that future work can build on them more easily.


## Version 2: ROS 2/Gazebo Multi-Agent Environment

### Moving from GridWorld to Gazebo

A major challenge was moving from the simple GridWorld environment to a real Gazebo simulation.

The GridWorld environment had discrete states and actions. In Gazebo, the robot has continuous position, continuous velocity commands, LiDAR observations, and physics-based motion.

Main difficulties:

- State space became continuous
- Action space became continuous
- Collision checking became more complex
- Robot motion depended on Gazebo physics
- LiDAR data had to be processed before using it
- Resetting the robot required Gazebo service calls

This made the project much closer to real robot navigation, but also made debugging harder.

---

### ROS 2 and Gazebo Launch Setup

I had to create a ROS 2 launch file to start Gazebo, load the world, process the Xacro robot model, spawn multiple robots, and bridge topics.

Challenges encountered:

- Correctly locating package files using `get_package_share_directory`
- Processing the Xacro file for two robot namespaces
- Spawning two robots at different start locations
- Making sure the Gazebo world file loaded correctly
- Making sure the bridge connected the correct ROS 2 and Gazebo topics

Small mistakes in file paths, robot names, or topic names caused the simulation to fail or the robot topics to not appear.

---

### Multi-Robot Namespaces

Because the project uses multiple robots, I needed separate namespaces and topics for each robot.

Examples:

```text
/robot1/cmd_vel
/robot1/scan
/robot1/odom

/robot2/cmd_vel
/robot2/scan
/robot2/odom
```

Challenges encountered:

- Avoiding topic conflicts between the multiple robots
- Making sure each robot received only its own velocity command
- Making sure each robot published its own LiDAR scan
- Making sure the URDF/Xacro file used the correct namespace
- Making sure the launch file spawned both robots correctly

This was important because if both robots accidentally used the same topic, the environment would not represent a true multi-agent problem.

---

### Reading Global Robot Pose

At first, I considered using `/odom` for robot position. However, `/odom` did not always give the global position needed for reward calculation and collision checking.

For reinforcement learning, I needed reliable robot positions to compute:

- Distance to goal
- Angle to goal
- Progress reward
- Robot-robot distance
- Reset conditions

I used Gazebo world pose information to get global positions more reliably.

Challenges encountered:

- Finding the correct Gazebo pose topic
- Parsing the pose information
- Extracting robot names
- Converting quaternion orientation to yaw
- Updating the pose continuously while training
- Handling startup delays before pose data became available

This became one of the most important parts of the environment implementation.

---

### LiDAR Preprocessing

The robots use 2D LiDAR for obstacle perception. Raw LiDAR data was not always directly usable.

Problems encountered:

- Some LiDAR readings were `inf`
- Some readings could be `nan`
- Very small values could indicate collision or near-collision
- The full 360-sample scan was large
- The scan needed to be normalized before going into the neural network

The preprocessing step included:

- Replacing `nan` values
- Replacing positive infinity with maximum sensor range
- Clipping scan values
- Downsampling the scan into smaller bins
- Normalizing by maximum LiDAR range

This made the observation vector more stable for learning.

---

### Observation Vector Design

Designing the observation vector was challenging. The actor network needs enough information to navigate, but the observation should not be unnecessarily complicated.

The final observation included:

- Downsampled LiDAR scan
- Robot position
- Robot yaw
- Distance to goal
- Angle to goal
- Relative position of the other robot
- Current linear velocity
- Current angular velocity

The actor uses this local observation. The critic uses the global state created by combining the observations of both robots.

This design supports CTDE because the actor is local but the critic is centralized.

---

### MADDPG Training Stability

Training MADDPG in Gazebo was challenging because both robots learn at the same time.

Problems encountered:

- Rewards were noisy
- Exploration caused frequent collisions
- Training was slower than in GridWorld
- The critic loss could become unstable
- The robots sometimes learned unsafe behavior
- Reward shaping had a large effect on training

Stability tools used:

- Replay buffer
- Target actor networks
- Target critic networks
- Soft target updates
- Exploration noise decay
- Separate actor and critic learning rates

The training still needs tuning, but the framework is working.

---

### Reward Shaping

A sparse reward was not enough because the robots rarely reached the goal during early training.

I added reward shaping terms:

- Time penalty
- Distance-to-goal penalty
- Progress reward
- Goal bonus
- Obstacle proximity penalty
- Idle penalty
- Collision penalty

Challenges encountered:

- Too much penalty can prevent exploration
- Too little collision penalty can allow unsafe behavior
- Progress reward helps learning but can still be noisy
- Goal reward must be large enough to encourage completion

Reward design remains an important area for improvement.

---

## Environment Reset and Termination

### Individual Robot Reset

In a multi-agent environment, one robot may finish or collide before the other robot.

I implemented reset logic so that:

- If Robot 1 reaches the goal or collides with an obstacle, only Robot 1 resets.
- If Robot 2 reaches the goal or collides with an obstacle, only Robot 2 resets.
- If both robots collide with each other, both robots reset.

Challenges encountered:

- Keeping separate done flags
- Keeping separate step counters
- Resetting only one robot without disturbing the other
- Stopping robot motion before reset
- Waiting for Gazebo pose updates after reset

This logic was harder than a single-agent reset.

---

## Summary

While working on this project, the biggest technical challenges were:

- Moving from GridWorld to Gazebo
- Managing ROS 2 and Gazebo communication
- Handling multiple robot namespaces
- Reading reliable global robot poses
- Preprocessing LiDAR data
- Designing the observation vector
- Implementing CTDE correctly
- Creating a multi-agent replay buffer
- Stabilizing MADDPG training
- Designing reward shaping
- Handling individual and joint robot resets
- Organizing results and repository files

These challenges were useful for understanding the practical difficulties of applying reinforcement learning to robot navigation in simulation.