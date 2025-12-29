from abc import ABC
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def scale_reward_down(r):
    return (r - (-100.0)) / (100.0 - (-100.0))  # adjust according to environment reward range

def scale_reward_up(r):
    return r * (100.0 - (-100.0)) + (-100.0) # inverse of scale_reward_down

class MLPQNetwork(ABC, nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[64, 64], encode_one_hot=False):
        """
        Args:
            obs_dim (int): If 'encode_one_hot' is False, this is the dimension of observation space. Otherwise, this is the number of possible discrete observation states.
            action_dim (int): Number of discrete actions
            hidden_dims (list[int]): List of hidden layer dimensions
            encode_one_hot (bool): if True, then the observation space is a single integer, and the forward() method will one-hot encode this integer before passing to the network.
        """
        super().__init__()
        self.encode_one_hot = encode_one_hot
        self.obs_dim = obs_dim
        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass to compute Q-values.
        
        Args:
            x: Input state tensor
               Shape: (batch_size, obs_dim)
        
        Returns:
            Q-values for each action
            Shape: (batch_size, action_dim)
        """

        if self.encode_one_hot:
            x = F.one_hot(x.squeeze(1), num_classes=self.obs_dim).float()
        return self.network(x)


class RewardShaper:
    """
    Implements potential-based reward shaping: F(s,a,s') = lambda * (gamma * phi(s') - phi(s))
    """
    
    def __init__(self, potential_fn, gamma: float, lambda_weight: float = 1.0):
        """
        Args:
            potential_fn: Function that computes phi(s)
            gamma: Discount factor
            lambda_weight: Weight for shaping reward (usually < 1.0)
        """
        self.potential_fn = potential_fn
        self.gamma = gamma
        self.lambda_weight = lambda_weight

    def get_shaping_reward(self, state: int, next_state: int, done: bool) -> float:
        """
        Compute shaping reward F(s,a,s').
        
        Args:
            state: Current state
            next_state: Next state after action
            done: Whether episode terminated
            
        Returns:
            Shaping reward
        """
        phi_s = self.potential_fn(state)
        
        # If episode done, potential of next state is 0
        phi_s_next = 0.0 if done else self.potential_fn(next_state)

        # F(s,a,s') = gamma * phi(s') - phi(s)
        shaping_reward = self.gamma * phi_s_next - phi_s
        
        return self.lambda_weight * shaping_reward

