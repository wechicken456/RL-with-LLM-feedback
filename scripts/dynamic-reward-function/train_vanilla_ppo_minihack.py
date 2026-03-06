import os
import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, Any
import torch
import torch.nn as nn
import minihack

class MiniHackFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        self.char_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        char_shape = observation_space.spaces["chars"].shape
        with torch.no_grad():
            dummy_char = torch.zeros(1, 1, char_shape[0], char_shape[1])
            n_flatten = self.char_cnn(dummy_char).shape[1]

        self.char_fc = nn.Sequential(nn.Linear(n_flatten, 128), nn.ReLU())
        
        blstats_dim = observation_space.spaces["blstats"].shape[0]
        self.blstats_fc = nn.Sequential(nn.Linear(blstats_dim, 64), nn.ReLU())
        
        self.final_fc = nn.Linear(128 + 64, features_dim)

    def forward(self, observations: dict) -> torch.Tensor:
        chars = observations["chars"].unsqueeze(1).float() / 255.0 
        char_features = self.char_fc(self.char_cnn(chars))
        
        blstats = observations["blstats"].float()
        blstats_features = self.blstats_fc(blstats)
        
        combined = torch.cat((char_features, blstats_features), dim=1)
        return self.final_fc(combined)


CONFIG = {
    "env_name": "MiniHack-River-Monster-v0",
    "total_timesteps": 500_000, 
    "steps_per_chunk": 100_000,
    "n_eval_episodes": 20,
    "success_threshold": 95.0,  
    "wandb_project": "rl-llm",
    "gamma": 0.99,
}


class MissionObsWrapper(gym.ObservationWrapper):
    """
    Removes the 'mission' key from observation space.
    DummyVecEnv cannot handle Text spaces (shape=None), so we strip it.
    """
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space.spaces.copy()
        obs_space.pop("mission", None)
        self.observation_space = gym.spaces.Dict(obs_space)

    def observation(self, obs):
        obs = dict(obs)
        obs.pop("mission", None)
        return obs


class WandbEpisodeCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._ep_num = 0
        self._ep_reward = 0.0
        self._ep_len = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0])
        infos = self.locals.get("infos", [{}])

        for i, info in enumerate(infos):
            self._ep_reward += float(rewards[i])
            self._ep_len += 1

            if "episode" in info:
                ep_len = info["episode"]["l"]
                wandb.log({
                    "episode/reward_total": self._ep_reward,
                    "episode/reward_original": self._ep_reward,
                    "episode/reward_shaping": 0.0,
                    "episode/reward_per_step": self._ep_reward / max(1, ep_len),
                    "episode/length": ep_len,
                    "episode/number": self._ep_num,
                    "step": self.num_timesteps,
                })
                self._ep_num += 1
                self._ep_reward = 0.0
                self._ep_len = 0
        return True


def evaluate_policy(model, env, n_eval_episodes=10) -> Dict[str, Any]:
    episode_rewards = []
    episode_lengths = []
    episode_successes = []

    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_len = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            ep_len += 1
            
            if done:
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_len)
                is_success = False
                if isinstance(info, dict):
                    is_success = info.get("success", False)
                    if not is_success and reward > 0:
                        is_success = True
                episode_successes.append(is_success)

    mean_success = np.mean(episode_successes) * 100 if episode_successes else 0.0
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
    mean_len = np.mean(episode_lengths) if episode_lengths else 0.0

    return {
        "success_rate": mean_success,
        "mean_reward": mean_reward,
        "mean_length": mean_len,
    }


def main():
    wandb.init(
        project=CONFIG["wandb_project"],
        config=CONFIG,
        name=f"vanilla-{CONFIG['env_name']}"
    )

    def make_env(render_mode="human"):
        env = gym.make(
            CONFIG["env_name"], 
            observation_keys=("chars", "blstats"),
            render_mode=render_mode
        )
        env = MissionObsWrapper(env)
        env = Monitor(env)
        return env

    train_env = make_env(render_mode=None)
    eval_env = make_env(render_mode=None)
    
    policy_kwargs = dict(
        features_extractor_class=MiniHackFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    print("Initializing PPO (MultiInputPolicy) — Vanilla (no reward shaping)...")
    model = PPO(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,      
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs
    )
    
    total_steps = 0
    chunk_iter = 0
    
    while total_steps < CONFIG["total_timesteps"]:
        print(f"\n=== Training Chunk {chunk_iter} (Steps: {total_steps}) ===")
        
        model.learn(
            total_timesteps=CONFIG["steps_per_chunk"], 
            reset_num_timesteps=False,
            progress_bar=True,
            callback=WandbEpisodeCallback(),
        )
        total_steps += CONFIG["steps_per_chunk"]
        
        print("[?] Evaluating...")
        metrics = evaluate_policy(model, eval_env, n_eval_episodes=CONFIG["n_eval_episodes"])
        
        print(f"Metrics: Success={metrics['success_rate']:.1f}%, Return={metrics['mean_reward']:.2f}")
        wandb.log({
            "eval/success_rate": metrics["success_rate"],
            "eval/mean_reward": metrics["mean_reward"],
            "eval/mean_length": metrics["mean_length"],
            "step": total_steps
        })
        
        if metrics["success_rate"] >= CONFIG["success_threshold"]:
            print(f"[+++] Solved! Success rate {metrics['success_rate']:.1f}% >= {CONFIG['success_threshold']}%")
            break
            
        chunk_iter += 1

    print("Training finished.")
    train_env.close()
    eval_env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
