from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from utils import taxi_state_to_text
from typing import Dict

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
ALLOWED_PROVIDERS =  ["openai", "gemini"]

class LLMStructuredResponse(BaseModel):
    reasoning: str
    potential: float # 0.0 to 1.0, progress toward goal


class LLM():
    def __init__(self, provider : str, model : str, system_prompt : str):
        provider = provider.lower()
        assert provider in ALLOWED_PROVIDERS, f"arg 'provider' must be one of: {ALLOWED_PROVIDERS}"
        
        self.client = None
        if provider == "openai":
            self.client = OpenAI(api_key = openai_api_key)
        self.provider = "openai"
        self.model = model
        self.system_prompt = system_prompt
    
    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def query(self, message: str) -> LLMStructuredResponse | None:
        if self.provider == "openai":
            response = self.client.responses.parse(
                model = self.model,
                input = [
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                text_format=LLMStructuredResponse
            )
            return response.output_parsed
        
class LLMTaxi:
    def __init__(self, llm: LLM, cache_size: int = 500):
        self.llm = llm
        self.cache: Dict[int, float] = {}
        self.cache_size = cache_size
        self.query_count = 0
        
        self.potential_prompt_template = """
You are assisting in training an RL agent in the Taxi-v3 environment.
The taxi must pick up a passenger at one of four fixed locations and
deliver them to a destination location in a 5x5 grid world.

The map of the environment is as follows: 
Map:

    +---------+
    |R: | : :G|
    | : | : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+

There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations.

Destination on the map are represented with the first letter of the color.

Passenger locations:

0: Red

1: Green

2: Yellow

3: Blue

4: In taxi

Destinations:

0: Red

1: Green

2: Yellow

3: Blue

An observation is returned as an int() that encodes the corresponding state, calculated by ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination

Your task: Evaluate ABSOLUTE PROGRESS TOWARD COMPLETING THE TASK, not the environment reward. Your job is NOT to tell the agent what to do. You must NOT evaluate actionw, Blue, or InTaxi)
- Destination location (Red, Green, Yellow, Blue)
Assign a State Potential Score F(s) between 0.0 and 1.0, where 0.0 means the objective is completely unmet and 1.0 means the objective is fully achieved. 
This score must reflect true objective progress, not whether the current strategy is good. Use the full continuous range and ensure that better states always receive higher scores than worse states.
Example:
- 0.0 = No progress (taxi far from passenger, not picked up)
- ~0.4 = Passenger picked up (halfway to goal)
- 1.0 = Task complete (passenger delivered to destination)

Consider:
1. Moving closer to the passenger before pickup
2. Successfully picking up the passenger
3. Moving closer to the destination after pickup
4. Successfully dropping the passenger at destination

Penalize:
- Moving away from current subgoal
- Illegal pickup/drop actions
- Wandering uselessly

Return a value between 0.0 and 1.0 representing overall progress, along with your reasoning."""
        llm.set_system_prompt(self.potential_prompt_template)

    def __call__(self, state_int: int) -> float:
        # return 0.5  
    
        # Check cache first
        if state_int in self.cache:
            return self.cache[state_int]

        state_description = taxi_state_to_text(state_int)
        prompt = f"Current state description: {state_description}\n\nWhat is the progress toward completing the task on a scale from 0.0 to 1.0? Provide your reasoning as well."

        try:
            response = self.llm.query(prompt)
            print(f"[debug] LLM's input prompt: {prompt}\n-==> Response:\n{response}\n")
            
            if response is not None:
                potential = response.potential
                self.query_count += 1

                # Update cache
                if len(self.cache) >= self.cache_size:
                    self.cache.pop(next(iter(self.cache)))  # Remove oldest entry
                self.cache[state_int] = potential
                
                return potential
            else:
                raise ValueError("LLM did not return a valid response.")
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return 0.5  # Neutral potential on error
    
    def get_cache_stats(self) -> dict:
        return {
            "cache_size": len(self.cache),
            "query_count": self.query_count,
            "cache_hit_rate": 1.0 - (self.query_count / max(1, self.query_count + len(self.cache)))
        }