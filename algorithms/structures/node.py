from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Callable

class NodeBase(ABC):
    """Abstract base class for a node in the tree."""

    def __init__(self, state: int, parent: Optional['NodeBase'] = None):
        self.state: int = state
        self.parent: Optional['NodeBase'] = parent
        self.children: List[Tuple[int, 'NodeBase']] = []
        self.visits: int = 0
        self.reward: float = 0.0

    @abstractmethod
    def best_child(self, score: Callable[['NodeBase'], List[float]]) -> Tuple[int, 'NodeBase']:
        """Return the best child based on some heuristic."""
        pass

    @abstractmethod
    def expand(self, action: int, new_state: int) -> 'NodeBase':
        """Expand the node by adding a new child."""
        pass

    @abstractmethod
    def update(self, reward: float) -> None:
        """Update the node's statistics (visits, reward, etc.)."""
        pass
