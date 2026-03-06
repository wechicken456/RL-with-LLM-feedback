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
import minihack

load_dotenv()

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
                # Potential-based shaping: F = beta * (gamma * phi(s') - phi(s))
                phi_s = float(np.clip(self.potential_fn(self._prev_obs, info), 0.0, 1.0))
                # Zero phi(s') on TRUE termination only (not truncation)
                if terminated:
                    phi_s_prime = 0.0
                else:
                    phi_s_prime = float(np.clip(self.potential_fn(obs, info), 0.0, 1.0))
                shaping = self.beta * (self.gamma * phi_s_prime - phi_s)
                self.reward_components = {"original": reward, "shaping": shaping, "phi_s": phi_s}
                reward = reward + shaping
            except Exception as e:
                print(f"[!] Error in reward execution: {e}")
                self.reward_components = {"error": 1.0}
        
        self._prev_obs = obs
        return obs, reward, terminated, truncated, info


class LLMCoder:
    def __init__(self, env_name: str, model: str, gamma : float, beta: float, backend: str = "ollama"):
        self.model = model
        self.backend = backend
        self.env_name = env_name
        self.gamma = gamma
        self.beta = beta
        self.conversation_history = []
        self.system_prompt = f"""You are an expert RL reward designer.
Your goal is to write a Python potential function for potential-based reward shaping (Ng et al. 1999).
The shaping reward is computed automatically by the wrapper as:
  F(s,s') = beta * (gamma * phi(s') - phi(s))
where beta = {self.beta} and gamma = {self.gamma}.
This F(s,s') is ADDED to the environment's original sparse reward.

{self.get_env_description()}

You must define EXACTLY one function with this signature. Only `numpy` (imported as `np`) is available:
```python
import numpy as np
def compute_potential(obs, info) -> float:
    # obs is a dict with keys 'chars' and 'blstats'
    # Return a float: the potential value phi(s)
    ...
```

CRITICAL INSTRUCTIONS:
1. ALWAYS return a single float in [0.0, 1.0]. NEVER return values > 1.0 or < 0.0. Do NOT multiply by large constants like 50 or 100 — values outside [0, 1] are hard-clamped by the wrapper and will cause incorrect shaping gradients. Use `np.clip(value, 0.0, 1.0)` as your final return.
2. BASELINE RULE: The absolute lowest potential phi(s) must be exactly 0.0.
3. SPARSITY RULE: If the agent is not making direct progress toward a subgoal, the potential MUST be exactly 0.0. Do not assign baseline potentials like 0.1 or 0.5 for merely existing, as the F(s,s') formula will mathematically penalize the agent via gamma decay.
4. GAMMA DECAY AWARENESS: With gamma = {self.gamma}, every step the agent does NOT change state incurs a penalty of -(1 - gamma) * beta * phi(s) = -{(1 - self.gamma) * self.beta} * phi(s). Design phi so that this per-step cost is justified by actual progress signal.
5. Output ONLY the Python function definition. No explanations, no markdown, no imports.
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
        return """
# Environment Specification: MiniHack-River-Monster-v0

## 1. Overview
The agent must cross a river to reach a goal (stairs down) on the other side, while avoiding 5 hostile monsters. The agent can push boulders into the water to create dry land (a bridge). Pushing a boulder simply requires walking into it. Monsters can attack and kill the agent.

## 2. Map Layout
The map is approximately 25 columns × 7 rows. A 3-tile-wide river of water (`}`) runs vertically around column 18-20. The agent starts on the left side. The goal is on the right side at approximately column 24. There are 5 boulders and 5 monsters placed randomly.

## 3. Observation Space
* `obs['chars']`: A 2D numpy array of shape (21, 79), dtype uint8. Contains the ASCII integer values of map entities.
  **CRITICAL NUMPY INDEXING RULE:** Numpy arrays are indexed as `array[row, column]`.
* `obs['blstats']`: A 1D numpy array of agent statistics.
  - `obs['blstats'][0]` is the agent's absolute X coordinate (column index).
  - `obs['blstats'][1]` is the agent's absolute Y coordinate (row index).
  Therefore, the agent's position in the chars array is `obs['chars'][blstats[1], blstats[0]]`.

## 4. Entity ASCII Mapping
Use these exact integer values with `np.where(obs['chars'] == value)` to locate entities:
* Agent (`@`): 64
* Boulder (`` ` ``): 96
* Water (`}`): 125
* Goal / Stairs Down (`>`): 62
* Monsters: Various ASCII codes. Monsters are letters — uppercase A-Z (65-90) and lowercase a-z (97-122) represent different monster types. To find ALL monsters, check for characters in these ranges that are NOT the agent (64) or boulder (96).

Example: to find all boulder positions:
  boulder_rows, boulder_cols = np.where(obs['chars'] == 96)

Example: to find the goal position:
  goal_rows, goal_cols = np.where(obs['chars'] == 62)

Example: to get agent position:
  agent_x, agent_y = int(obs['blstats'][0]), int(obs['blstats'][1])

Example: to find all monster positions (letters excluding agent `@`=64):
  is_upper = (obs['chars'] >= 65) & (obs['chars'] <= 90)
  is_lower = (obs['chars'] >= 97) & (obs['chars'] <= 122)
  is_not_boulder = obs['chars'] != 96
  monster_rows, monster_cols = np.where((is_upper | is_lower) & is_not_boulder)

Use `np.linalg.norm([r1 - r2, c1 - c2])` to compute Euclidean distances.

## 5. Objective for the Potential Function
Higher phi(s) values mean the agent is closer to solving the task. A multi-stage heuristic works well:
1. Locate the agent via `blstats`, boulders (96), water tiles (125), the goal (62), and monsters via `np.where`.
2. Count how many water tiles remain (fewer = more bridge built = higher potential).
3. Measure the agent's distance to the nearest boulder and/or the goal.
4. Consider monster proximity as a danger factor — being close to a monster is bad.
5. Combine into a single float in [0.0, 1.0].

Do NOT mutate `obs` or `info`. Do NOT use print statements.
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

        user_prompt = f"""The agent will be trained with PPO. Generate an initial `compute_potential(obs, info)` function for {self.env_name}. 
The function should return a float in [0.0, 1.0] representing how close the agent is to solving the task.
Output ONLY the Python function definition, no explanations."""
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
We used your potential function to train a PPO agent. Here are the evaluation results:
{process_feedback}

Based on this data, refine the `compute_potential` function to improve the success rate.
Focus on whether the potential correctly captures progress toward solving the task:
- Is the agent being rewarded for approaching boulders?
- Is it rewarded when boulders fill water tiles (bridge building)?
- Is it rewarded for moving toward the goal once a path exists?
- Is the shaping reward magnitude large enough to influence learning?
Output ONLY the Python function definition. No explanations, no markdown.
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
        env = gym.make(
            env_name, 
            observation_keys=("chars", "blstats"),
            render_mode=None
        )
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
        feedback += f"- {k}: {v:.6f}\n"
    
    avg_shaping = abs(latest['components'].get('shaping', 0.0))
    if avg_shaping < 0.01:
        feedback += f"\n** WARNING: Average shaping reward magnitude per step is {avg_shaping:.8f}, which is EXTREMELY LOW. "
        feedback += "The potential function outputs are too small to meaningfully influence learning. "
        feedback += "Increase the magnitude of the potential function's return values so that the per-step shaping reward |F(s,s')| is at least 0.01. **\n"
        
    if len(history) > 1:
        prev = history[-2]
        improve = latest['success_rate'] - prev['success_rate']
        feedback += f"\nTrend: Success rate changed by {improve:+.1f}% since last iteration.\n"
        
    return feedback


def main():
    wandb.init(
        project=CONFIG["wandb_project"],
        config=CONFIG,
        name=f"{CONFIG['env_name']}"
    )
    
    coder = LLMCoder(CONFIG["env_name"], CONFIG["llm_model"], gamma=CONFIG["gamma"], beta=1.0, backend=CONFIG["llm_backend"])
    
    print("=== Generating Initial Reward Function ===")
    current_reward_code = coder.generate_initial_reward_function()
    print(current_reward_code)
    
    if not verify_reward_code(current_reward_code, CONFIG["env_name"]):
        print("[!] Initial code invalid. Exiting.")
        return

    def make_env(render_mode="human"):
        env = gym.make(
            CONFIG["env_name"], 
            observation_keys=("chars", "blstats"),
            render_mode=render_mode
        )
        env = MissionObsWrapper(env)  # Strip 'mission' Text space before Monitor/VecEnv
        env = Monitor(env) 
        env = CustomRewardWrapper(env, current_reward_code, gamma=CONFIG["gamma"], beta=1.0)
        return env

    train_env = make_env(render_mode=None)
    eval_env = make_env(render_mode=None)
    
    policy_kwargs = dict(
        features_extractor_class=MiniHackFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

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
        policy_kwargs=policy_kwargs
                       
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
