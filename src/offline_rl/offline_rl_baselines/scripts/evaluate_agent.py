"""
Evaluate a trained agent in the GeMS environment.
"""
import argparse
import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from offline_rl_baselines.agents.offline.td3_bc import TD3BCAgent
from offline_rl_baselines.envs.gems_env import GeMS_Env


def evaluate_agent(
    agent,
    env,
    num_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
):
    """
    Evaluate agent in environment.

    Args:
        agent: Trained agent
        env: GeMS environment
        num_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic policy
        render: Whether to render (not implemented for GeMS)

    Returns:
        dict: Evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state, deterministic=deterministic)

            # Step environment
            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            state = next_state

            if render:
                pass  # GeMS doesn't support rendering

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward = {episode_reward:.2f}, Length = {episode_length}")

    # Compute statistics
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
    }

    return metrics, episode_rewards


def main():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--env_name", type=str, default="diffuse_topdown",
                        choices=["diffuse_topdown", "diffuse_sideview", "diffuse_birdview"])
    parser.add_argument("--ranker_type", type=str, default="GeMS",
                        choices=["GeMS", "WkNN", "Softmax"])

    # Agent
    parser.add_argument("--agent", type=str, default="td3_bc",
                        choices=["td3_bc", "cql", "iql"])
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")

    # Evaluation
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic policy")
    parser.add_argument("--seed", type=int, default=0)

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("=" * 60)
    print(f"Evaluating {args.agent.upper()} on {args.env_name}")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Ranker: {args.ranker_type}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Seed: {args.seed}")
    print()

    # Create environment
    print("Creating environment...")
    env = GeMS_Env(
        env_name=args.env_name,
        ranker_type=args.ranker_type,
        device=args.device
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print()

    # Load agent
    print("Loading agent...")
    if args.agent == "td3_bc":
        agent = TD3BCAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=args.device
        )
    else:
        raise NotImplementedError(f"Agent {args.agent} not implemented yet")

    # Load model checkpoint
    agent.load(args.model_path)
    print(f"Model loaded from {args.model_path}")
    print()

    # Evaluate
    print("=" * 60)
    print("Starting evaluation...")
    print("=" * 60)

    metrics, episode_rewards = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
    )

    # Print results
    print()
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Min reward: {metrics['min_reward']:.2f}")
    print(f"Max reward: {metrics['max_reward']:.2f}")
    print(f"Mean length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    print("=" * 60)

    # Save results
    results_dir = Path(args.model_path).parent / "evaluation"
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / f"eval_{args.ranker_type}_{args.num_episodes}eps.txt"
    with open(results_file, "w") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Environment: {args.env_name}\n")
        f.write(f"Ranker: {args.ranker_type}\n")
        f.write(f"Episodes: {args.num_episodes}\n")
        f.write(f"Deterministic: {args.deterministic}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write("\n")
        f.write(f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}\n")
        f.write(f"Min reward: {metrics['min_reward']:.2f}\n")
        f.write(f"Max reward: {metrics['max_reward']:.2f}\n")
        f.write(f"Mean length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}\n")
        f.write("\n")
        f.write("Episode rewards:\n")
        for i, r in enumerate(episode_rewards):
            f.write(f"  Episode {i+1}: {r:.2f}\n")

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
