from .structures.node import NodeBase
from .structures.tree import TreeBase

import math, random
from typing import Tuple, List, Optional, Callable

import graphviz
import gymnasium as gym
import numpy as np

from gymnasium.spaces import Discrete
from gymnasium import Env

class MCTSNode(NodeBase):
    def __init__(self, state: Discrete, action: Discrete, reward: float, terminal: bool, parent: Optional[NodeBase] = None):
        super().__init__(state, action, parent)
        self.performance: float = 0.0
        self.reward = reward
        self.terminal = terminal

    def is_fully_expanded(self, action_space: Discrete) -> bool:
        return len(self.children) == action_space.n

    def best_child(self, score: Callable[[NodeBase], List[float]]) -> NodeBase:
        """UCT calculation to find the best child."""
        best_index = int(np.argmax(score(self)))
        return self.children[best_index]

    def expand(self, new_state: Discrete, action: Discrete, reward: float, terminal: bool) -> NodeBase:
        child_node = MCTSNode(new_state, action, reward, terminal, parent=self)
        self.children.append(child_node)
        return child_node

    def update(self, reward: float) -> None:
        self.visits += 1
        self.reward += reward
        self.performance = self.reward / self.visits
        
    def __str__(self):
        return "{}: (state={}, action={}, visits={}, reward={:0.4f}, ratio={:0.4f})".format(
                                                  self.uid,
                                                  self.state,
                                                  self.action,
                                                  self.visits,
                                                  self.reward,
                                                  self.performance)

class MonteCarloTreeSearch(TreeBase):
    def __init__(self, root: MCTSNode, env: Env[Discrete, Discrete], exploration_constant: float):
        self.env = env
        self.exploration_constant: float = exploration_constant
        self.root = root

    @staticmethod
    def _score(c: float) -> Callable[[NodeBase],  List[float]]:
        def score(node: NodeBase) -> List[float]:
            return [
                (child.reward / child.visits) + c * math.sqrt((2 * math.log(node.visits) / child.visits))
                if child.visits > 0 else float("-inf")
                for child in node.children
            ]
        return score

    def select(self, node: NodeBase) -> NodeBase:
        while node.children:
            node = node.best_child(self._score(self.exploration_constant))
        return node

    def expand(self, node: NodeBase) -> Optional[NodeBase]:
        actions_to_try = [action for action in range(self.env.action_space.n) if action not in [child.action for child in node.children]]
        # Select an untried action and create a new child node
        action = random.choice(actions_to_try)    
        new_state, reward, terminal, _, _ = self.env.step(action)
        return node.expand(new_state, action, float(reward), terminal)
    
    def simulate(self, node: NodeBase) -> float:
        if node.terminal:
            return node.reward
        
        while True:
            action = self.env.action_space.sample()
            _, reward, done, _, _ = self.env.step(action)
            if done:
                return float(reward)

    def forward(self) -> NodeBase:
        node = self.root
        while not node.terminal:
            if not node.is_fully_expanded(self.env.action_space):
                return self.expand(node)
            
            # Select the best path and expand
            node = self.select(node)
        
        return node

    def backpropagate(self, node: NodeBase, reward: float) -> None:
        # Update all parents with the gained reward
        while node is not None:
            node.update(reward)
            node = node.parent
            
    def inference(self, node: NodeBase) -> None:
        # Save the state to restore it later
        c = self.exploration_constant
        self.exploration_constant = 0.0
        node = self.select(node)
        print(f"node: {node}, terminal?: {node.terminal}, reward: {self.simulate(node)}")
        self.exploration_constant = c
