import experiments.frozen_lake as frozen_lake

from algorithms.MCTS import MonteCarloTreeSearch

import random

def main():
    random.seed(2)
    env = frozen_lake.init_env()
    initial_state, _ = env.reset()
    steps = 10000
    monteCarloTreeSearch = MonteCarloTreeSearch(env, steps)

    monteCarloTreeSearch.run(initial_state)

    # Execute the best action in the Frozen Lake environment
    env.reset()
    env.unwrapped.s = initial_state  # Set the environment state to the root state
    new_state, reward, done, _, _ = env.step(best_action)

    # Output the result
    print(f"Initial state: {initial_state}")
    print(f"Best action: {best_action}")
    print(f"New state: {new_state}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")

if __name__ == "__main__":
   main()
