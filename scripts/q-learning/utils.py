# Filename: utils.py
# Description: Utility functions for Dyna-Q algorithm

import gymnasium as gym


def make_env(env_id, seed, run_name, train=True, render_mode="rgb_array", record_period=100) -> gym.Env:
    """
    Creates and wraps the LunarLander environment.
    """
    if render_mode:
        env = gym.make(env_id, render_mode=render_mode)
    else:
        env = gym.make(env_id)
    
    if render_mode == "rgb_array":
        env = gym.wrappers.RecordVideo(
            env, 
            video_folder=f"./videos-{run_name}",
            name_prefix="training" if train else "eval",
            episode_trigger=lambda x: x % record_period == 0
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Calculates the epsilon for epsilon-greedy exploration."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def decode_taxi_state(state_int: int) -> dict:
    """
    Decode Taxi-v3 state, which is a single integer
    More info about the taxi state encoding: https://gymnasium.farama.org/environments/toy_text/taxi/
    
    Returns:
        dict with keys: taxi_row, taxi_col, passenger_loc, destination
    """
    taxi_row = state_int // 100
    remainder = state_int % 100
    taxi_col = remainder // 20
    remainder = remainder % 20
    passenger_loc = remainder // 4
    destination = remainder % 4
    
    return {
        "taxi_row": taxi_row,
        "taxi_col": taxi_col,
        "passenger_loc": passenger_loc,
        "destination": destination
    }


def taxi_state_to_text(state_int: int) -> str:
    """
    Convert Taxi-v3 state (which is a single integer) to natural language description.
    
    Args:
        state_int: Integer state from Taxi-v3 environment
        
    Returns:
        Natural language description of the state
    """
    state = decode_taxi_state(state_int)
    
    locations = ["Red", "Green", "Yellow", "Blue"]
    
    taxi_pos = f"({state['taxi_row']}, {state['taxi_col']})"
    dest = locations[state['destination']]
    
    if state['passenger_loc'] == 4:  # passenger in taxi
        return f"Taxi at {taxi_pos}. Passenger is IN TAXI. Destination: {dest}."
    else:
        passenger_loc_name = locations[state['passenger_loc']]
        return f"Taxi at {taxi_pos}. Passenger at {passenger_loc_name}. Destination: {dest}."

