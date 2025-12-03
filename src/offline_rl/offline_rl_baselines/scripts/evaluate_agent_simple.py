"""
Evaluate a trained agent in the GeMS environment (without gym dependency).
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


def evaluate_agent(
    agent,
    env,
    belief_encoder,
    ranker,
    num_episodes: int = 10,
    deterministic: bool = True,
    device: str = "cuda",
):
    """
    Evaluate agent in GeMS environment.

    Args:
        agent: Trained agent
        env: GeMS environment
        belief_encoder: Belief encoder for state encoding
        ranker: Ranker for action decoding
        num_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic policy
        device: Device to use

    Returns:
        dict: Evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        # Reset environment
        obs_tuple = env.reset()
        if isinstance(obs_tuple, tuple):
            obs, _ = obs_tuple
        else:
            obs = obs_tuple

        # Reset belief encoder hidden state
        for module in belief_encoder.beliefs:
            belief_encoder.hidden[module] = torch.zeros(
                1, 1, belief_encoder.hidden_dim,
                device=belief_encoder.my_device
            )

        # Get initial belief state
        belief_state = belief_encoder.forward(obs)
        state = belief_state.cpu().detach().numpy()

        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Select action (latent action)
            latent_action = agent.select_action(state, deterministic=deterministic)

            # Decode latent action to slate using ranker
            with torch.no_grad():
                latent_tensor = torch.FloatTensor(latent_action).to(device)
                slate = ranker.rank(latent_tensor)

            # Step environment
            next_obs, reward, done, info = env.step(slate)

            # Update belief state
            next_belief_tensor = belief_encoder.forward(next_obs, done=done)
            if next_belief_tensor is None:
                # If done, belief encoder may return None
                next_belief_tensor = belief_state.clone()

            next_state = next_belief_tensor.cpu().detach().numpy()

            # Convert reward to float if it's a tensor
            if isinstance(reward, torch.Tensor):
                reward = reward.item()
            episode_reward += reward
            episode_length += 1
            state = next_state
            belief_state = next_belief_tensor

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

    # Load GeMS components
    print("Loading GeMS components...")
    cwd = os.getcwd()
    try:
        from offline_data_collection.environment_factory import EnvironmentFactory
        from offline_data_collection.model_loader import ModelLoader

        # Create environment
        env_factory = EnvironmentFactory()
        env = env_factory.create_environment(args.env_name)
        print(f"✅ Environment created: {args.env_name}")

        # Load belief encoder
        model_loader = ModelLoader()
        belief_encoder = model_loader.load_belief_encoder(args.env_name)
        belief_encoder.eval()
        print(f"✅ Belief encoder loaded")

        # Load ranker
        ranker = model_loader.load_ranker(
            env_name=args.env_name,
            ranker_type=args.ranker_type,
            embedding_type="ideal"
        )
        ranker.to(args.device)  # 确保ranker在正确的设备上
        ranker.eval()
        print(f"✅ Ranker loaded: {args.ranker_type}")

    except Exception as e:
        print(f"❌ Failed to load GeMS components: {e}")
        import traceback
        traceback.print_exc()
        return

    state_dim = 20  # belief state dimension
    action_dim = 32  # latent action dimension

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
    print(f"✅ Model loaded from {args.model_path}")
    print()

    # Evaluate
    print("=" * 60)
    print("Starting evaluation...")
    print("=" * 60)

    metrics, episode_rewards = evaluate_agent(
        agent=agent,
        env=env,
        belief_encoder=belief_encoder,
        ranker=ranker,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        device=args.device,
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

    print(f"\n✅ Results saved to {results_file}")


if __name__ == "__main__":
    main()
