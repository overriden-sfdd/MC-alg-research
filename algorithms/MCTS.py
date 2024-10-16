from .structures.node import NodeBase
from .structures.tree import TreeBase

import math
from typing import Tuple, List, Optional, Callable

import gymnasium as gym
import numpy as np

from gymnasium.spaces import Discrete
from gymnasium import Env

class MCTSNode(NodeBase):
    def __init__(self, state: int, parent: Optional[NodeBase] = None):
        super().__init__(state, parent)
        self.performance: float = 0.0

    def is_fully_expanded(self, action_space: Discrete) -> bool:
        return len(self.children) == action_space.n

    def best_child(self, score: Callable[[NodeBase], List[float]]) -> Tuple[int, NodeBase]:
        """UCT calculation to find the best child."""
        best_index = int(np.argmax(score(self)))
        return self.children[best_index]

    def expand(self, action: int, new_state: int) -> NodeBase:
        """Expand the node by adding a new child."""
        child_node = MCTSNode(new_state, parent=self)
        self.children.append((action, child_node))
        return child_node

    def update(self, reward: float) -> None:
        """Update node statistics."""
        self.visits += 1
        self.reward += reward
        self.performance = self.reward / self.visits
        
    def __str__(self):
        return "{}: (action={}, visits={}, reward={:d}, ratio={:0.4f})".format(
                                                  self.state,
                                                  self.action,
                                                  self.visits,
                                                  self.reward,
                                                  self.performance)


class MonteCarloTreeSearch(TreeBase):
    def __init__(self, env: Env[Discrete, Discrete], iterations: int = 1000):
        self.env = env
        self.iterations = iterations
        self.c: float = 1.4

    @staticmethod
    def _score(c: float) -> Callable[[NodeBase],  List[float]]:
        def score(node: NodeBase) -> List[float]:
            return [
                (child.reward / child.visits) + c * math.sqrt((2 * math.log(node.visits) / child.visits))
                if child.visits > 0 else float("-inf")
                for _, child in node.children
            ]
        return score

    def select(self, node: NodeBase) -> NodeBase:
        """Selection step: traverse the tree using UCT until a leaf node."""
        while node.children:
            print(f"Selecting best child of node with state {node.state}, visits {node.visits}, reward {node.reward}")
            _, node = node.best_child(self._score(self.c))
        return node

    def expand(self, node: NodeBase) -> Optional[NodeBase]:
        """Expand the node by adding a new child for an untried action."""
        if node.is_fully_expanded(self.env.action_space):
            return None

        # Select an untried action and create a new child node
        for action in range(self.env.action_space.n):
            if not any(a == action for a, _ in node.children):
                new_state, _, _, _, _ = self.env.step(action)
                print(f"Expanding node with state {node.state}, adding child for action {action}, new state {new_state}")
                return node.expand(action, new_state)
        return None

    def simulate(self, node: NodeBase) -> float:
        """Simulation step: run a random rollout from the node to a terminal state."""
        current_state: int = node.state
        total_reward: float = 0.0
        done: bool = False

        while not done:
            # Random policy
            action: int = self.env.action_space.sample()
            current_state, reward, done, _, _ = self.env.step(action)
            total_reward += float(reward)

        print(f"Simulating from state {node.state}, total reward: {total_reward}")

        return total_reward

    def backpropagate(self, node: NodeBase, reward: float) -> None:
        """Backpropagation step: update nodes along the path with the reward."""
        while node is not None:
            print(f"Backpropagating reward {reward} to node with state {node.state}, previous visits: {node.visits}, previous reward: {node.reward}")
            node.update(reward)
            node = node.parent

    def run(self, initial_state: int) -> int:
        """Run the MCTS algorithm from the given initial state."""
        root: MCTSNode = MCTSNode(initial_state)

        for i in range(self.iterations):
            
            print(f"\nIteration {i+1}/{self.iterations}")
            
            # Reset environment for each iteration
            self.env.reset() 
            # Set environment to the root state
            self.env.unwrapped.s = initial_state

            # Selection
            node: MCTSNode = self.select(root)

            # Expansion
            child: Optional[MCTSNode] = self.expand(node)
            if child is not None:
                node = child

            # Simulation (rollout)
            reward: float = self.simulate(node)

            # Backpropagation
            self.backpropagate(node, reward)

        # Return the best action from the root node
        best_action, _ = root.best_child(self._score(0.0))
        return best_action
