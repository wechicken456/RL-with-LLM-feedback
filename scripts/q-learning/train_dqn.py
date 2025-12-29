import random
import time

import numpy as np
import torch

from utils import make_env, decode_taxi_state, taxi_state_to_text
import argparse
import os
from distutils.util import strtobool
import yaml
from datetime import date, datetime

import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from cleanrl_utils.buffers import ReplayBuffer

from models import MLPQNetwork , RewardShaper
from utils import linear_schedule
import os
import wandb
from collections import deque
from llm import LLMStructuredResponse, LLMTaxi, LLM

def load_config(config_path="config_dqn.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    """
    Parse command-line arguments, overriding defaults from config.yaml.
    """
    defaults = load_config()
    default_exp_name = "exp_" + date.today().strftime("%Y%m%d") + "_" + datetime.now().strftime("%H%M%S")
    default_wandb_run_name = default_exp_name

    parser = argparse.ArgumentParser()
    
    # Experiment 
    parser.add_argument("--exp-name", type=str, default=default_exp_name)
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--total-episodes", type=int, default=defaults["total_episodes"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--stop-reward", type=float, default=320.0)
    
    # Logging (run-specific)
    parser.add_argument("--logger", type=str, default=defaults["logger"],
                        choices=["tensorboard", "wandb", "both", "console", "none"])
    parser.add_argument("--log-dir", type=str, default=defaults["log_dir"])
    parser.add_argument("--wandb-project-name", type=str, default=defaults["wandb_project_name"])
    parser.add_argument("--wandb-run-name", type=str, default=defaults["wandb_run_name"])
    parser.add_argument("--wandb-entity", type=str, default=defaults["wandb_entity"])
    parser.add_argument("--record-period", type=int, default=defaults["record_period"])
    parser.add_argument("--log-interval-steps", type=int, default=defaults["log_interval_steps"])

    # Device and determinism
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=defaults["torch_deterministic"], nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=defaults["cuda"], nargs="?", const=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=defaults["track"], nargs="?", const=True)
    
    # Standard RL hyperparameters 
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--buffer-size", type=int, default=defaults["buffer_size"])
    parser.add_argument("--gamma", type=float, default=defaults["gamma"])
    parser.add_argument("--tau", type=float, default=defaults["tau"])
    parser.add_argument("--target-network-frequency", type=int, default=defaults["target_network_frequency"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--epsilon-start", type=float, default=defaults["epsilon_start"])
    parser.add_argument("--epsilon-end", type=float, default=defaults["epsilon_end"])
    parser.add_argument("--epsilon-decay", type=float, default=defaults["epsilon_decay"])
    parser.add_argument("--learning-starts-after-episode", type=int, default=defaults["learning_starts_after_episode"])
    parser.add_argument("--train-frequency", type=int, default=defaults["train_frequency"])
    parser.add_argument("--updates-per-step", type=int, default=defaults["updates_per_step"])

    # LLM
    parser.add_argument("--llm-provider", type=str, default=defaults["llm_provider"])
    parser.add_argument("--llm-model", type=str, default=defaults["llm_model"])

    # Reward shaping
    parser.add_argument("--use-reward-shaping", type=lambda x: bool(strtobool(x)), default=defaults["use_reward_shaping"], nargs="?", const=True)
    parser.add_argument("--shaping-lambda", type=float, default=defaults["shaping_lambda"])
    parser.add_argument("--llm-query-frequency", type=int, default=defaults.get("llm_query_frequency", 50))
    parser.add_argument("--potential-cache-size", type=int, default=defaults.get("potential_cache_size", 500))

    args = parser.parse_args()
    return args




args = parse_args()
run_name = args.wandb_run_name

run = wandb.init(
    project=args.wandb_project_name,
    entity=args.wandb_entity,
    name=run_name,
    config=vars(args),
    sync_tensorboard=False,
)

run.define_metric(name="q_losses/*", step_metric="global_step")
run.define_metric(name="model_losses/*", step_metric="episode")
run.define_metric(name="episode/*", step_metric="episode")
run.define_metric(name="shaping/*", step_metric="episode")


# --- Seeding ---
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# --- Environment Setup ---
envs = make_env(args.env_id, args.seed, run_name, train=True, record_period=args.record_period)
obs, _ = envs.reset(seed=args.seed)
encode_one_hot = False
if type(obs) == int:
    print(f"[debug] Raw observation space is a single integer: {obs}. MLPQNetwork will automatically one-hot encode it.")
    obs_shape = (envs.observation_space.n, )
    encode_one_hot = True
else:
    obs_shape = obs.shape 
action_space = getattr(envs, "action_space", None)

# --- Agent Initialization ---
print(f"obs_shape: {obs_shape}, action_space.n: {envs.action_space.n}")
q_network = MLPQNetwork(obs_shape[0], envs.action_space.n, encode_one_hot=encode_one_hot).to(device)
q_optimizer = Adam(q_network.parameters(), lr=args.learning_rate)
target_network = MLPQNetwork(obs_shape[0], envs.action_space.n, encode_one_hot=encode_one_hot).to(device)
target_network.load_state_dict(q_network.state_dict())  # Initialize target network

# --- Shaping Reward ---
llm = LLM(provider=args.llm_provider, model=args.llm_model, system_prompt="")
potential_fn = LLMTaxi(llm, cache_size=args.potential_cache_size)
reward_shaper = RewardShaper(potential_fn, gamma=args.gamma, lambda_weight=args.shaping_lambda)
# shaping gamma should match main gamma to be consistent with the MDP discount.

# --- Debug Info ---
if action_space is not None and getattr(action_space, "n", None) > 0:
    action_shape = action_space.n
else:
    raise NotImplementedError("This code only supports environments with discrete action spaces.")
batch_shape = (args.batch_size,) + tuple(obs_shape)
print("[debug] Using device:", device)
print(f"[debug] observation shape: {obs_shape}")
print(f"[debug] action shape: {action_shape}")
print(f"[debug] batch tensor shape: {batch_shape}")
rb_real = ReplayBuffer(
    args.buffer_size,
    envs.observation_space,
    envs.action_space,
    device,
    handle_timeout_termination=False,
)

# --- Training ---
global_step = 0
latest_episodes_list = []

epsilon = args.epsilon_start
reward_history = deque(maxlen=100)  # Store the last 100 episode rewards
best_reward = -float("inf")
for episode in range(args.total_episodes):
    obs, _ = envs.reset()
    if isinstance(obs, int):
        obs = np.array([obs], dtype=np.long)  # taxi environment returns a single integer observation

    #print(f"obs: {obs}, type: {type(obs)}")
    episode_transitions = {
        "observations": [],
        "actions": [],
        "next_observations": [],
        "rewards": [],
        "dones": []
    }

    # Track episode-level reward shaping stats
    episode_env_rewards = 0.0
    episode_shaping_rewards = 0.0
    episode_total_rewards = 0.0
    

    while True:
        # --- Select Action ---
        if random.random() < epsilon:
            actions = envs.action_space.sample()
        else:
            # Unsqueeze to add batch dimension: (1, obs_dim)
            obs_tensor = torch.tensor(obs, device=device).unsqueeze(0)
            # q_values shape: (1, action_shape)
            q_values = q_network(obs_tensor)
            actions = torch.argmax(q_values, dim=1).squeeze(-1).cpu().numpy() 
            actions = int(actions) # taxi environment requires a single int as action space
        
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if isinstance(next_obs, int):
            next_obs = np.array([next_obs], dtype=np.long)  # taxi environment returns a single integer observation
        if isinstance(actions, int):
            actions = np.array([actions], dtype=np.long)  # taxi environment expects a single integer observation, while cleanrl ReplayBuffer expects numpy arrays
        global_step += 1

        real_done = terminations or truncations

        # --- Apply Reward Shaping ---
        if args.use_reward_shaping and (global_step % args.llm_query_frequency == 0):
            # Extract scalar state values for potential function
            state_scalar = int(obs[0]) if isinstance(obs, np.ndarray) else int(obs)
            next_state_scalar = int(next_obs[0]) if isinstance(next_obs, np.ndarray) else int(next_obs)

            shaping_rewards = reward_shaper.get_shaping_reward(state_scalar, next_state_scalar, real_done)
            shaped_rewards = float(rewards) + shaping_rewards
            episode_env_rewards += float(rewards)
            episode_shaping_rewards += shaping_rewards
            episode_total_rewards += shaped_rewards
        else:
            shaped_rewards = float(rewards)
            episode_env_rewards += float(rewards)
            episode_total_rewards += float(rewards)

        total_rewards = float(rewards) + shaped_rewards
        rb_real.add(obs, next_obs, actions, rewards, real_done, infos)
        
        # Store transitions for world model training
        episode_transitions["observations"].append(obs)
        episode_transitions["actions"].append(actions)
        episode_transitions["next_observations"].append(next_obs)
        episode_transitions["rewards"].append(rewards)
        episode_transitions["dones"].append(real_done)
        
        # Never forgetting to do this again...
        obs = next_obs
        
        if real_done:
            break

        if episode < args.learning_starts_after_episode:
            continue

               
        # --- Update Q-Network ---
        if global_step % args.train_frequency == 0:
            total_q_loss = 0.0
            # total_q_loss_real = 0.0
            for i in range(args.updates_per_step):
                real_batch_size = args.batch_size
                # data_real shapes:
                #   observations: (real_batch_size, obs_dim)
                #   actions: (real_batch_size, 1)
                #   next_observations: (real_batch_size, obs_dim)
                #   rewards: (real_batch_size, 1)
                #   dones: (real_batch_size, 1)
                transitions_real = rb_real.sample(real_batch_size)
                
                batch_states = transitions_real.observations.to(device)
                batch_actions = transitions_real.actions.to(device)
                batch_next_states = transitions_real.next_observations.to(device)
                batch_rewards = transitions_real.rewards.to(device)
                batch_dones = transitions_real.dones.to(device)
                              
                with torch.no_grad():
                    target_q, _ = target_network(batch_next_states).max(dim=1)
                    td_target = batch_rewards.flatten() + args.gamma * target_q * (1 - batch_dones.flatten())
                current_q = q_network(batch_states).gather(1, batch_actions.long()).squeeze()
                q_loss = F.smooth_l1_loss(td_target, current_q)

                total_q_loss += q_loss.detach().cpu().item()

                q_optimizer.zero_grad()
                q_loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), 100.0)
                q_optimizer.step()


            # Soft update target network
            target_net_state_dict = target_network.state_dict()
            policy_net_state_dict = q_network.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*args.tau + target_net_state_dict[key]*(1-args.tau)
            target_network.load_state_dict(target_net_state_dict)

            # Log losses for this step
            if global_step % args.log_interval_steps == 0:
                wandb.log({"losses/total_q_loss": total_q_loss / args.updates_per_step, "global_step": global_step})

        
    # Log reward shaping stats
    if args.use_reward_shaping:
        wandb.log({
            "episode/env_rewards": episode_env_rewards,
            "episode/shaping_rewards": episode_shaping_rewards,
            "episode/total_rewards": episode_total_rewards,
            "episode": episode
        })

        # Log LLM cache stats
        cache_stats = reward_shaper.potential_fn.get_cache_stats()
        wandb.log({
            "shaping/cache_size": cache_stats["cache_size"],
            "shaping/query_count": cache_stats["query_count"],
            "shaping/cache_hit_rate": cache_stats["cache_hit_rate"],
            "episode": episode
        })

    # Log episode statistics
    #wandb.log({"episode/reward": infos["episode"]["r"], "episode": episode})
    wandb.log({"episode/length": infos["episode"]["l"], "episode": episode})
    wandb.log({"epsilon": epsilon, "episode": episode})
    epsilon = max(args.epsilon_decay * epsilon, args.epsilon_end)
    reward_history.append(infos["episode"]["r"])
    if episode % 100 == 0:
        running_avg = np.mean(reward_history)
        print(f"Episode {episode}, running avg reward : {running_avg}")
        if running_avg > best_reward:
            best_reward = running_avg
            print(f"New best reward: {best_reward}. Saving this model...")
            torch.save(q_network.state_dict(), f"best_model_{run_name}.pt")
        if running_avg >= args.stop_reward:
            print(f"Solved environment in {episode} episodes!")
            break
   

print(f"Training complete! Run name: {run_name}")
final_model_path = os.path.join(".", f"final_model_{run_name}.pt")
torch.save(q_network.state_dict(), final_model_path)
print(f"Saved final model to {final_model_path}")
