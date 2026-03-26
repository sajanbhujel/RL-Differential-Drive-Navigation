# src/main.py
import argparse
import os, numpy as np
from src.env.grid_world import GridWorld100
from src.agents.dp_agent import DPAgent, DPConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["value_iteration", "policy_iteration"], default="value_iteration")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--theta", type=float, default=1e-6)
    parser.add_argument("--max_iter", type=int, default=10000)
    args = parser.parse_args()

    env = GridWorld100()
    agent = DPAgent(DPConfig(gamma=args.gamma, theta=args.theta, max_iterations=args.max_iter))

    if args.algo == "value_iteration":
        V, pi = agent.value_iteration(env)
    else:
        V, pi = agent.policy_iteration(env)

    os.makedirs("results", exist_ok=True)

    agent.plot_value(
        env, V,
        title=f"{args.algo.replace('_',' ').title()}: Value Function",
        save_path=f"results/{args.algo}_value.png"
    )

    agent.plot_policy_arrows(
        env, pi,
        title=f"{args.algo.replace('_',' ').title()}: Policy",
        save_path=f"results/{args.algo}_policy.png"
    )

    np.save("results/V.npy", V)
    np.save("results/policy.npy", pi)

if __name__ == "__main__":
    main()