import os
import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, Tuple, Any, List
import anthropic 
import minigrid
import torch
import torch.nn as nn
import ollama
import openai
from dotenv import load_dotenv
import time

load_dotenv()


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom CombinedExtractor for Minigrid Dict obs.
    Processes 'image' (7x7x3) with a small CNN (3x3 kernels)
    and 'direction' with flatten, then concatenates.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim=1)  # dummy, updated below

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # subspace shape is (C, H, W) after VecTransposeImage
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
                # 'direction' or any other non-image key: flatten
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
    "steps_per_chunk": 100_000, # Train for ~100k steps before evaluating/refining
    "n_eval_episodes": 20,
    "max_refinements": 10,  
    "success_threshold": 95.0,  
    "llm_backend": "ollama",
    "llm_model": "qwen3-coder-next:latest",
    # "llm_backend": "anthropic",
    # "llm_model": "claude-opus-4-5",
    # "llm_backend": "openai",
    # "llm_model": "gpt-5.2-codex",
    "wandb_project": "rl-llm",
    "gamma": 0.99,
}

ollama_client = None
if CONFIG["llm_backend"] == "ollama":
    ollama_client = ollama.Client(
        host="http://10.130.127.2:11434"   
    )
elif CONFIG["llm_backend"] == "anthropic":
    anthropic_client = anthropic.Client(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )   
elif CONFIG["llm_backend"] == "openai":
    openai_client = openai.Client(
        api_key=os.environ.get("OPENAI_API_KEY")
    )
elif CONFIG["llm_backend"] == "gemini":
    gemini_client = gemini.Client(
        api_key=os.environ.get("GEMINI_API_KEY")
    )


class MissionObsWrapper(gym.ObservationWrapper):
    """
    Removes the 'mission' key from Minigrid's observation space.
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


class CustomRewardWrapper(gym.Wrapper):
    """
    Potential-Based Reward Shaping wrapper (Ng et al. 1999).
    F(s, s') = gamma * phi(s') - phi(s)
    The shaped reward is: r'(s,a,s') = r(s,a,s') + F(s, s')
    This guarantees the optimal policy is unchanged.
    """
    def __init__(self, env, reward_fn_code: str = None, gamma: float = 1.0, beta=1.0):
        super().__init__(env)
        self.reward_components = {}
        self.potential_fn = None
        self.gamma = gamma  
        self._prev_obs = None 
        self.beta = beta
        
        if reward_fn_code:
            self.update_reward_function(reward_fn_code)
    
    def _load_reward_function(self, code: str):
        """Dynamically load potential function from code string"""
        namespace = {"np": np, "gym": gym}
        try:
            exec(code, namespace)
            self.potential_fn = namespace.get("compute_potential")
            if not self.potential_fn:
                raise ValueError("Reward code must define 'compute_potential' function")
        except Exception as e:
            print(f"[!] Error loading reward function: {e}")
            self.potential_fn = None
    
    def update_reward_function(self, reward_fn_code: str):
        self._load_reward_function(reward_fn_code)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Preserve original environment reward for evaluation metrics
        info["original_reward"] = reward
        
        if self.potential_fn:
            try:
                # Potential-based shaping: F = gamma * phi(s') - phi(s)
                phi_s = self.potential_fn(self._prev_obs, info)
                phi_s_prime = self.potential_fn(obs, info)
                shaping = self.beta*(self.gamma * phi_s_prime - phi_s)
                self.reward_components = {"original": reward, "shaping": shaping, "phi_s": phi_s}
                reward = reward + shaping
            except Exception as e:
                print(f"[!] Error in reward execution: {e}")
                self.reward_components = {"error": 1.0}
        
        self._prev_obs = obs
        return obs, reward, terminated, truncated, info


class LLMCoder:
    def __init__(self, env_name: str, model: str, backend: str = "ollama"):
        self.model = model
        self.backend = backend
        self.env_name = env_name
        self.conversation_history = []
        self.system_prompt = f"""You are an expert RL reward designer.
Your goal is to write a Python potential function for potential-based reward shaping (Ng et al. 1999).
The shaping reward is computed automatically as: F(s,s') = gamma * phi(s') - phi(s)
This is ADDED to the original sparse reward.

{self.get_env_description()}

def compute_potential(obs, info) -> float:
    return potential_value

CRITICAL INSTRUCTIONS:
1. The agent is permanently anchored at coordinate (3, 6).
2. DO NOT search for the agent ID (10) in the array.
3. Use vectorized operations like np.argwhere(obs['image'][:, :, 0] == TARGET_ID) to find objects.
4. Calculate all Manhattan distances relative to the fixed agent position (3, 6).
5. BASELINE RULE: The absolute lowest potential phi(s) must be exactly 0.0.
6. SPARSITY RULE: If the agent is wandering in an empty area and not making direct progress toward a subgoal (key, door, or goal), the potential MUST evaluate to exactly 0.0. Do not assign baseline potentials like 0.1 or 0.5 for merely existing, as the F(s,s') formula will mathematically penalize the agent via decay.
7. The function MUST a float strictly bounded between 0.0 and 1.0.
8. INVENTORY MECHANICS: DO NOT check the info dictionary for carrying status. The environment automatically renders the carried object at the agent's explicit feet coordinate. The agent is carrying the key if and only if obs['image'][3, 6, 0] == 5.
9. THE PICKUP PENALTY TRAP: When the agent picks up the key, the key disappears from the general grid (len(key_pos) == 0) and moves to obs['image'][3, 6, 0]. If you do not assign a high baseline potential for simply holding the key, the potential will drop to 0.0 after pickup, resulting in a massive mathematical penalty. Holding the key must inherently provide a higher baseline potential than merely looking at it.
"""
        if backend == "anthropic":
            self.client = anthropic_client
        elif backend == "ollama":
            self.client = ollama_client
        elif backend == "openai":   
            self.client = openai_client
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def get_env_description(self) -> str:
        if "MiniGrid" in self.env_name:
            return f"# Environment Specification: {self.env_name}\n" + """ 
## 1. Overview
This environment is a sparse-reward gridworld. The agent must pick up a key to unlock a door and navigate to a green goal square. 

## 2. Action Space (Discrete)
0: left, 1: right, 2: forward, 3: pickup, 4: drop, 5: toggle, 6: done

## 3. Observation Space (Dict)
'image': (7, 7, 3) integer array representing first-person view.
'direction': integer (0-3).
'mission': string.

## 4. Observation Encoding & Spatial Mechanics
Each tile is encoded as (OBJECT_IDX, COLOR_IDX, STATE).
STATE: 0=open, 1=closed, 2=locked.
OBJECT_TO_IDX: "unseen": 0, "empty": 1, "wall": 2, "floor": 3, "door": 4, "key": 5, "ball": 6, "box": 7, "goal": 8, "lava": 9, "agent": 10.

CRITICAL RULES FOR EXTRACTING POSITIONS:
- The (3, 6) Coordinate: The agent's position in its view is hardcoded as grid.width // 2, grid.height - 1. Because the agent_view_size is 7x7, this evaluates mathematically to 7 // 2 = 3 for the x-coordinate, and 7 - 1 = 6 for the y-coordinate.
- The Missing Agent ID: The engine explicitly forces the cell at (3, 6) to be either the object the agent is carrying, or empty (object idx = 1). It never places the actual agent object (ID 10) into the grid.
- The Egocentric Shift: The grid is rotated based on the agent's direction so that "forward" is always "up" in the resulting matrix.

## 5. Original Reward
* **Success**: 1 - 0.9 * (step_count / max_steps)
* **Failure**: 0
"""

    def _call_llm(self, messages: list) -> str:
        if self.backend == "ollama":
            full_messages = [{"role": "system", "content": self.system_prompt}] + messages
            response = ollama_client.chat(model=self.model, messages=full_messages)
            return response.message.content
        elif self.backend == "anthropic":
            response = anthropic_client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=self.system_prompt,
                messages=messages
            )
            return response.content[0].text
        elif self.backend == "openai":
            response = openai_client.chat.completions.create(
                model=self.model,
                max_tokens=2000,
                messages=[{
                    "role": "system", 
                    "content": self.system_prompt
                }, 
                *messages]
            )
            return response.choices[0].message.content

    def generate_initial_reward_function(self) -> str:
        if self.client is None:
            return self._get_stub_reward()

        user_prompt = f"Generate an initial potential function for {self.env_name}."
        self.conversation_history = [{"role": "user", "content": user_prompt}]

        print("[LLM] Generating initial reward...")
        try:
            text = self._call_llm(self.conversation_history)
            code = self._extract_code(text)
            self.conversation_history.append({"role": "assistant", "content": text})
            return code
        except Exception as e:
            print(f"[!] LLM Error: {e}")
            return self._get_stub_reward()

    def refine_reward(self, process_feedback: str) -> str:
        if self.client is None:
            return self._get_stub_reward()

        prompt = f"""
We used your reward function to train a PPO agent. Here is the performance and feedback:
{process_feedback}

Based on this, refine the `compute_potential` function to improve success rate.
Focus on whether the potential is correctly capturing progress (near key, key pickup, door unlock, goal proximity).
Output ONLY the Python code.
"""
        self.conversation_history.append({"role": "user", "content": prompt})

        print("[LLM] Refining reward...")
        try:
            text = self._call_llm(self.conversation_history)
            code = self._extract_code(text)
            print("\n----------------")
            print("[+] Reward function refined:")
            print(code)
            print("----------------\n")
            self.conversation_history.append({"role": "assistant", "content": text})
            return code
        except Exception as e:
            print(f"[!] LLM Error: {e}")
            return self._get_stub_reward()
    
    def _extract_code(self, text: str) -> str:
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        elif "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text.strip()

    def _get_stub_reward(self):
        return """
def compute_potential(obs, info):
    # STUB: placeholder potential function
    # Returns 0 always - no shaping, falls back to original sparse reward
    return 0.0
"""


class TPEvaluator:
    """
    Placeholder
    """
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
    
    def collect_trajectories(self, env, model, n_episodes: int = 50):
        pass
    
    def evaluate(self, reward_fn_code: str) -> Tuple[bool, str]:
        return True, ""


class WandbEpisodeCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._ep_num = 0
        self._ep_original = 0.0
        self._ep_shaped_total = 0.0
        self._ep_len = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0])
        infos = self.locals.get("infos", [{}])

        for i, info in enumerate(infos):
            self._ep_shaped_total += float(rewards[i])
            self._ep_original += float(info.get("original_reward", 0.0))
            self._ep_len += 1

            if "episode" in info:
                ep_len = info["episode"]["l"]
                wandb.log({
                    "episode/reward_total": self._ep_shaped_total,
                    "episode/reward_original": self._ep_original,
                    "episode/reward_shaping": self._ep_shaped_total - self._ep_original,
                    "episode/reward_per_step": self._ep_shaped_total / max(1, ep_len),
                    "episode/length": ep_len,
                    "episode/number": self._ep_num,
                    "step": self.num_timesteps,
                })
                self._ep_num += 1
                self._ep_original = 0.0
                self._ep_shaped_total = 0.0
                self._ep_len = 0
        return True


def verify_reward_code(code: str, env_name: str) -> bool:
    """Test if reward code executes without errors on a dummy env"""
    try:
        env = gym.make(env_name, render_mode=None)
        env = MissionObsWrapper(env)
        wrapper = CustomRewardWrapper(env, code)
        obs, _ = wrapper.reset()
        action = wrapper.action_space.sample()
        wrapper.step(action)
        env.close()
        return True
    except Exception as e:
        print(f"[!] Reward code verification failed: {e}")
        return False


def evaluate_policy(model, env, n_eval_episodes=10) -> Dict[str, Any]:
    """
    Run evaluation episodes and return metrics.
    Only necessary metrics for REFINEMENT decisions should be here.
    """
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    components_sum = {} 
    total_steps_eval = 0

    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_len = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, info = env.reset() if isinstance(obs, tuple) else (obs, {}) # Handle VecEnv vs Gym API quirks if any
             
            obs, reward, terminated, truncated, info  = env.step(action)
            done = terminated or truncated
            
            ep_reward += info.get("original_reward", 0.0)
            ep_len += 1
            total_steps_eval += 1   
            
            if hasattr(env, "reward_components"):
                 for k, v in env.reward_components.items():
                     components_sum[k] = components_sum.get(k, 0) + v
                     
            if done:
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_len)
                is_success = False
                if isinstance(info, dict):
                    is_success = info.get("success", False)
                    if not is_success and info.get("original_reward", 0) > 0:
                        is_success = True
                
                episode_successes.append(is_success)

    mean_success = np.mean(episode_successes) * 100 if episode_successes else 0.0
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
    mean_len = np.mean(episode_lengths) if episode_lengths else 0.0
    
    # Average components per STEP
    avg_components = {k: v / max(1, total_steps_eval) for k, v in components_sum.items()}

    metrics = {
        "success_rate": mean_success,
        "mean_reward": mean_reward,
        "mean_length": mean_len,
        "components": avg_components
    }
    return metrics


def generate_process_feedback(history: List[Dict[str, Any]]) -> str:
    """Generate text feedback for the LLM based on evaluation history"""
    if not history:
        return "No history yet."
    
    latest = history[-1]
    feedback = "=== PROCESS FEEDBACK ===\n"
    feedback += f"Current Success Rate: {latest['success_rate']:.1f}%\n"
    feedback += f"Mean Reward (Total): {latest['mean_reward']:.2f}\n"
    feedback += f"Mean Episode Length: {latest['mean_length']:.1f}\n\n"
    
    feedback += "Reward Components Breakdown (Average per step):\n"
    for k, v in latest['components'].items():
        feedback += f"- {k}: {v:.4f}\n"
        
    if len(history) > 1:
        prev = history[-2]
        improve = latest['success_rate'] - prev['success_rate']
        feedback += f"\nTrend: Success rate changed by {improve:+.1f}% since last iteration.\n"
        
    return feedback


def main():
    wandb.init(
        project=CONFIG["wandb_project"],
        config=CONFIG,
        name=f"rl-llm-{CONFIG['env_name']}"
    )
    
    coder = LLMCoder(CONFIG["env_name"], CONFIG["llm_model"], backend=CONFIG["llm_backend"])
    
    print("=== Generating Initial Reward Function ===")
    current_reward_code = coder.generate_initial_reward_function()
    print(current_reward_code)
    
    if not verify_reward_code(current_reward_code, CONFIG["env_name"]):
        print("[!] Initial code invalid. Exiting.")
        return

    def make_env(render_mode="human"):
        env = gym.make(CONFIG["env_name"], render_mode=render_mode)
        env = MissionObsWrapper(env)  # Strip 'mission' Text space before Monitor/VecEnv
        env = Monitor(env) 
        env = CustomRewardWrapper(env, current_reward_code)
        return env

    train_env = make_env(render_mode=None)
    eval_env = make_env(render_mode=None)

    print("Initializing PPO (MultiInputPolicy)...")
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
        policy_kwargs={"features_extractor_class": MinigridFeaturesExtractor,
                       "features_extractor_kwargs": {"features_dim": 128}},
    )
    
    eval_history = []
    total_steps = 0
    refinement_iter = 0
    
    while total_steps < CONFIG["total_timesteps"]:
        print(f"\n=== Training Chunk {refinement_iter} (Steps: {total_steps}) ===")
        
        model.learn(
            total_timesteps=CONFIG["steps_per_chunk"], 
            reset_num_timesteps=False,
            progress_bar=True,
            callback=WandbEpisodeCallback(),
        )
        total_steps += CONFIG["steps_per_chunk"]
        
        print("[?] Evaluating...")
        metrics = evaluate_policy(model, eval_env, n_eval_episodes=CONFIG["n_eval_episodes"])
        eval_history.append(metrics)
        
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
            
        if refinement_iter >= CONFIG["max_refinements"]:
            print("[---] Max refinements reached.")
            continue

        print("[?] Querying LLM for refinement...")
        feedback = generate_process_feedback(eval_history)
        new_code = coder.refine_reward(feedback)
        
        if verify_reward_code(new_code, CONFIG["env_name"]):
            current_reward_code = new_code
            
            # train_env and eval_env are CustomRewardWrapper instances (outermost wrapper)
            train_env.update_reward_function(current_reward_code)
            eval_env.update_reward_function(current_reward_code)
            print("[+] Reward function updated.")
            
            wandb.log({
                f"reward_code_{refinement_iter}": wandb.Html(f"<pre>{current_reward_code}</pre>"),
                "step": total_steps
            })

        else:
            print("[!] Refined code failed verification. Keeping old reward.")
            
        refinement_iter += 1

    print("Training finished.")
    train_env.close()
    eval_env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
