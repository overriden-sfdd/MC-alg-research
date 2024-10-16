from .node import NodeBase
from abc import ABC, abstractmethod

class TreeBase(ABC):
    """Abstract base class for the tree."""

    @abstractmethod
    def select(self, node: NodeBase) -> NodeBase:
        """Selection step: select a node to expand."""
        pass

    @abstractmethod
    def expand(self, node: NodeBase) -> NodeBase | None:
        """Expansion step: expand the selected node."""
        pass

    @abstractmethod
    def simulate(self, node: NodeBase) -> float:
        """Simulation step: run a rollout from the expanded node."""
        pass

    @abstractmethod
    def backpropagate(self, node: NodeBase, reward: float) -> None:
        """Backpropagation step: update the tree with the result of the simulation."""
        pass

    @abstractmethod
    def run(self, initial_state: int) -> int:
        """Run the MCTS algorithm from the initial state."""
        pass
