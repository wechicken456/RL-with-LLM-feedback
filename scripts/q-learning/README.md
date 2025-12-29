
# Overview

**NOTE**: Everything is tailored to the [Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/) environment. 

Standard DQN implementation enhanced with Ng et al. (1999) potential-based reward shaping. An LLM estimates state potential phi(s) to provide additional guidance rewards that accelerate learning while preserving optimal policy.

Shaping reward: F(s,a,s') = gamma * phi(s') - phi(s)
- **Note**: This is the same gamma as the one used in standard RL hyperparamters to be consistent with the MDP discount. If you use a different gamma, you distort how “future advantage” is weighted, which can bias policies.

Total reward: r_total = r_env + lambda * F(s,a,s')

# Usage

Baseline (no shaping):
```bash
python train_dqn.py --env-id Taxi-v3 --use-reward-shaping=false
```

LLM shaping:
```bash
python train_dqn.py --env-id Taxi-v3 --use-reward-shaping --shaping-lambda 5
```

## Files

- `train_dqn.py` - Main training loop with reward shaping integration
- `models.py` - DQN network architecture and reward shaper
- `llm.py` - OpenAI API interface for potential estimation. Also include system prompts for this Taxi environment
- `utils.py` - State encoding/decoding and environment utilities. Since this Taxi environment returns a single integer that encodes the observation, there are functions in this file that decode it into text to provide to the LLM.
- `config_dqn.yaml` - Hyperparameters and configuration


