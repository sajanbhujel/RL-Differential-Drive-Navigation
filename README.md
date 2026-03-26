# Multi-Agent RL Path Planning for Differential Drive Robots (Gazebo)

## Version 1: Dynamic Programming Foundation

---

## 1. Project Overview

This project investigates reinforcement learning (RL) methods for **autonomous navigation** of differential-drive robots.

The long-term goal is to develop agents that can:
- Navigate toward assigned goals  
- Avoid obstacles and collisions  
- Operate under uncertainty (sensor noise, motion noise)  
- Scale to multi-agent environments  

This project bridges:
- **Markov Decision Processes (MDPs)**
- **Dynamic Programming (DP)**
- **Reinforcement Learning (future versions)**

---

## 2. Version 1 Scope

Version 1 focuses on building a strong theoretical and implementation foundation:

- MDP formulation of navigation  
- GridWorld abstraction of the environment  
- Dynamic Programming methods:
  - Value Iteration  
  - Policy Iteration  
- Modular agent framework  
- Visualization and result saving  

More advanced RL methods will be implemented in later versions.

---

## 3. System Setup (Target System)

### Simulation Environment (Future Work)
- ROS 2 + Gazebo  
- Differential-drive robot (EduBot platform)  
- 2D LiDAR for perception  

### Sources of Uncertainty (Future Work)
- Wheel slip (process noise)  
- LiDAR noise (measurement noise)  
- Multi-agent interaction (non-stationary dynamics)  

---

## 4. Markov Decision Process (MDP) Formulation

The navigation problem is modeled as an MDP:

\[
M = (S, A, P, R, \gamma)
\]

This formulation is critical because:
- The agent must **learn behavior through interaction**
- Transition dynamics may be **unknown or stochastic**
- Enables extension to **model-based RL and world models**

---

### 4.1 State Space (S)

In the GridWorld abstraction:

- State = (x, y) position in a 2D grid  
- Invalid states correspond to obstacles  

---

### 4.2 Action Space (A)

Discrete actions:

- UP  
- DOWN  
- LEFT  
- RIGHT  

---

### 4.3 Transition Function \( P(s' | s, a) \)

- Deterministic transitions  
- Invalid actions (collision or boundary) result in staying in place  

---

### 4.4 Reward Function \( R(s, a) \)

- +100 → Goal reached  
- -1 → Step penalty  
- -5 → Invalid move (collision/boundary)  

---

### 4.5 Terminal Condition

Episode ends when:
- Goal is reached  

---

## 5. Simplified Subtask: GridWorld for DP

To enable exact Dynamic Programming solutions, a simplified environment is used:

- 100×100 grid  
- Fixed start and goal  
- Static obstacles  
- Deterministic transitions  

This provides:
- A **well-defined MDP**
- A **debuggable environment**
- A **baseline before moving to Gazebo**

---

## 6. Implemented Algorithms (Version 1)

### Dynamic Programming

- **Value Iteration**
  - Updates value function using Bellman optimality equation  

- **Policy Iteration**
  - Alternates between:
    - Policy Evaluation  
    - Policy Improvement  

### Q-value Based Policy Improvement

Policy improvement is performed using Q-values computed from the value function:

\[
Q(s, a) = r + \gamma V(s')
\]

The optimal action is selected as:

\[
\pi(s) = \arg\max_a Q(s, a)
\]

This ensures compatibility with future RL methods such as Q-learning and Sarsa.

---

## 7. Agent Framework

A modular agent framework is implemented:

```python
class BaseAgent:
    def train(self, env):
        pass

    def evaluate(self, env):
        pass

    def select_action(self, state):
        pass
