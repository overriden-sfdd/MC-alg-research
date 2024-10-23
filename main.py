import experiments.frozen_lake as frozen_lake
from algorithms.MCTS import MonteCarloTreeSearch, MCTSNode
from algorithms.utils.visualizer import generate_graph

import random
import copy

from gymnasium.spaces import Discrete

def main():    
    random.seed(42)
    env = frozen_lake.init_env()
    initial_state, _ = env.reset()
    steps = 500

    # Initialize root with 0 action, 0 reward, non-terminal state
    root = MCTSNode(initial_state, 0, 0.0, False, parent=None)
    MCTS = MonteCarloTreeSearch(root, env)
    
    for i in range(steps):
        print(f"The step #{i} out of {steps}")
        env.reset()
        node = MCTS.forward()
        reward = MCTS.simulate(node)
        MCTS.backpropagate(node, reward)

    generate_graph(root)

if __name__ == "__main__":
   main()
