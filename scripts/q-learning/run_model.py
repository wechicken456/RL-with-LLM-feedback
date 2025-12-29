import torch
import numpy as np
import argparse
from utils import make_env
from models import MLPQNetwork

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument("--env-id", type=str, required=True, help="Environment ID (e.g., LunarLander-v2)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = make_env(args.env_id, args.seed, "eval", render_mode="human", record_period=1)
    
    q_network = MLPQNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q_network.load_state_dict(torch.load(args.checkpoint, weights_only=True, map_location=device))
    q_network.network.eval()
    
    print(f"Running agent from {args.checkpoint}")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            step_cnt = 0
            while not done:
                # Select action greedily (no exploration)
                with torch.no_grad():
                    obs_tensor = torch.Tensor(obs).to(device).unsqueeze(0)
                    q_values = q_network(obs_tensor)
                    action = torch.argmax(q_values, dim=1).cpu().numpy()[0]
                
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                step_cnt += 1
                done = terminated or truncated
            
            print(f"Episode: {step_cnt} steps, reward = {episode_reward:.2f}")
    
    except KeyboardInterrupt:
        # Print summary statistics
        print(f'\nEvaluation Summary:')
        print(f'Episode durations: {list(env.time_queue)}')
        print(f'Episode rewards: {list(env.return_queue)}')
        print(f'Episode lengths: {list(env.length_queue)}')

        # Calculate some useful metrics
        avg_reward = np.sum(env.return_queue)
        avg_length = np.sum(env.length_queue)
        std_reward = np.std(env.return_queue)

        print(f'\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}')
        print(f'Average episode length: {avg_length:.1f} steps')
        print(f'Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}')
        print("\nStopped by user")
    finally:
        env.close()
