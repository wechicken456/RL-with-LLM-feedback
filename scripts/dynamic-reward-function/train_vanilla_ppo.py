import os
import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import minigrid
import torch
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "image":
                n_channels = subspace.shape[0]
                cnn = nn.Sequential(
                    nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                with torch.no_grad():
                    sample = torch.as_tensor(subspace.sample()[None]).float()
                    n_flatten = cnn(sample).shape[1]
                extractors[key] = nn.Sequential(cnn, nn.Linear(n_flatten, features_dim), nn.ReLU())
                total_concat_size += features_dim
            else:
                extractors[key] = nn.Flatten()
                total_concat_size += gym.spaces.utils.flatdim(subspace)

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded = [extractor(observations[key]) for key, extractor in self.extractors.items()]
        return torch.cat(encoded, dim=1)


CONFIG = {
    "env_name": "MiniGrid-DoorKey-8x8-v0",
    "total_timesteps": 500_000,
    "wandb_project": "rl-llm",
    "gamma": 0.99,
}


class MissionObsWrapper(gym.ObservationWrapper):
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
                    "episode/reward": self._ep_reward,
                    "episode/reward_per_step": self._ep_reward / max(1, ep_len),
                    "episode/length": ep_len,
                    "episode/number": self._ep_num,
                    "step": self.num_timesteps,
                })
                self._ep_num += 1
                self._ep_reward = 0.0
                self._ep_len = 0
        return True


def make_env():
    env = gym.make(CONFIG["env_name"], render_mode=None)
    env = MissionObsWrapper(env)
    env = Monitor(env)
    return env


def main():
    wandb.init(
        project=CONFIG["wandb_project"],
        config=CONFIG,
        name=f"vanilla-ppo-{CONFIG['env_name']}",
    )

    env = make_env()

    print("Initializing PPO...")
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=CONFIG["gamma"],
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"features_extractor_class": MinigridFeaturesExtractor,
                       "features_extractor_kwargs": {"features_dim": 128}},
    )

    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        progress_bar=True,
        callback=WandbEpisodeCallback(),
    )

    print("Training finished.")
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
