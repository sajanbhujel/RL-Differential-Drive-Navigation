# Multi-Agent RL Path Planning for Differential Drive Robots (Gazebo)

## 1. Project Overview

This project investigates reinforcement learning (RL) methods for **multi-agent autonomous navigation** in a simulated Gazebo environment using differential-drive robots.

The goal is to develop agents that can:
- Navigate toward assigned goals  
- Avoid static obstacles  
- Avoid collisions with other agents  
- Operate under stochastic motion and sensor noise  

The system is designed to bridge:
- Classical planning (Dynamic Programming)
- Tabular RL methods
- Multi-agent reinforcement learning (future versions)

---

## 2. System Setup

### Simulation Environment
- ROS 2 + Gazebo  
- Differential-drive robots (EduBot platform)  
- 2D LiDAR sensor for perception  

### Sources of Uncertainty
- Wheel slip (process noise)  
- LiDAR noise (measurement noise)  
- Multi-agent interaction (non-stationary environment)  

---

## 3. Markov Decision Process (MDP) Formulation

The navigation problem is formulated as a Markov Decision Process:

M = (S, A, P, R, γ)

This formulation is critical because:
- The agent must **learn unknown transition dynamics**
- The agent must **infer reward structure through interaction**
- Future extensions will include **model-based RL / world models**

---

### 3.1 State Space (S)

Each agent observes:

- Robot pose estimate: (x, y, θ)  
- LiDAR scan data  
- Relative goal position (robot frame)  
- Relative positions of nearby agents (if observable)  

Thus, the RL state is:

s_t = [x, y, θ, LiDAR, Goal_relative, Agent_relative]

Note:
- The state is **partially observable**
- The environment becomes **non-stationary** in multi-agent settings  

---

### 3.2 Action Space (A)

Discrete action space (Version 1):

- Move Forward  
- Turn Left  
- Turn Right  
- Stop  

Future extensions:
- Continuous control (v, ω)

---

### 3.3 Transition Function P(s' | s, a)

State transitions follow differential-drive kinematics:

d = (Δs_r + Δs_l) / 2  
Δθ = (Δs_r − Δs_l) / b  

With stochastic wheel slip:

Δs_l' = Δs_l + ε_l  
Δs_r' = Δs_r + ε_r  

where:

ε ~ N(0, σ²)

Thus:
- Transitions are **stochastic**
- Multi-agent interactions introduce **non-stationarity**

---

### 3.4 Reward Function R(s, a)

The reward function is designed to encourage safe and efficient navigation:

- +100 → Goal reached  
- -100 → Collision (obstacle or agent)  
- -1 → Time step penalty  
- Optional shaping: distance-to-goal reduction  

---

### 3.5 Terminal Conditions

Episodes terminate when:

- Goal is reached  
- Collision occurs  
- Maximum step limit is exceeded  

---

### 3.6 Why MDP Matters

Even though the environment is implemented in Gazebo, the agent:

- Does **not know transition probabilities P(s' | s, a)**  
- Does **not know the true reward function R(s, a)**  
- Must **learn from interaction data**

This enables:
- Model-free RL (current)
- Model-based RL / world models (future versions)

---

## 4. Simplified Subtask (Grid World for DP)

To support **tabular Dynamic Programming (DP)**, a simplified environment is implemented:

### Grid World:
- Discrete 2D grid  
- Deterministic transitions  
- Obstacles and goal  

This allows implementation of:

- Policy Iteration  
- Value Iteration  
- Q-value updates  

This subtask ensures:
- Theoretical correctness
- Debugging before scaling to Gazebo

---

## 5. Implemented Algorithms (Version 1)

### Dynamic Programming
- Policy Iteration (V and Q)
- Value Iteration

### Tabular RL
- Monte Carlo Prediction
- TD(0)
- Q-learning
- Epsilon-Greedy exploration

---

## 6. Agent Framework

All agents follow a unified interface:

```python
class BaseAgent:
    def train(self, env):
        pass

    def evaluate(self, env):
        pass

    def select_action(self, state):
        pass
