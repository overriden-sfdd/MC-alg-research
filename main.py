import experiments.frozen_lake as frozen_lake
from algorithms.MCTS import MonteCarloTreeSearch, MCTSNode
from algorithms.utils.visualizer import generate_graph

import random, argparse, math
from typing import Optional

from gymnasium.spaces import Discrete

def parse_args():
    parser = argparse.ArgumentParser(description="Global constants for MCTS.")
    parser.add_argument('--iterations', type=int, default=500,
                        help="Number of MCTS iterations (default: 1000)")
    parser.add_argument('--c', type=float, default=math.sqrt(2),
                        help="Exploration constant c for UCT (default: sqrt(2))")
    parser.add_argument('--renderer_mode', type=str, default="",
                        help="Renderer mode for Gymnasium environment (e.g., 'human', 'rgb_array', or '')")
    parser.add_argument('--graph_filepath', type=str, default="",
                        help="Filepath to save the generated Graphviz tree. If the string is empty, no graph is generated (default: '')")
    parser.add_argument('--inference', type=bool, default=False,
                        help="Flag indicating whether to run inference for the resultant model or not")
    return parser.parse_args()

def main():
    args = parse_args()
    
    random.seed(42)
    frozen_lake.register_env()
    env = frozen_lake.init_env(args.renderer_mode)
    initial_state, _ = env.reset()
    steps = args.iterations

    # Initialize root with 0 action, 0 reward, non-terminal state
    root = MCTSNode(initial_state, 0, 0.0, False, parent=None)
    MCTS = MonteCarloTreeSearch(root, env, args.c)
    
    for i in range(steps):
        print(f"The step #{i} out of {steps}")
        env.reset()
        node = MCTS.forward()
        reward = MCTS.simulate(node)
        MCTS.backpropagate(node, reward)

    if (len(args.graph_filepath) > 0):
        generate_graph(root, args.graph_filepath)
        
    if (args.inference):
        env.close()
        env = frozen_lake.init_env("human")
        env.reset()
        MCTS.env = env
        MCTS.inference(root)

if __name__ == "__main__":
   main()
