import uuid
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Callable

from gymnasium.spaces import Discrete


class NodeBase(ABC):
    """Abstract base class for a node in the tree."""

    def __init__(self, state: Discrete, action: Discrete, parent: Optional['NodeBase'] = None):
        self.uid = str(uuid.uuid1())
        self.state = state
        self.action = action
        self.parent: Optional['NodeBase'] = parent
        self.children: List['NodeBase'] = []
        self.visits: int = 0
        self.reward: float = 0.0
        self.terminal: bool = False

    @abstractmethod
    def best_child(self, score: Callable[['NodeBase'], List[float]]) -> 'NodeBase':
        """Return the best child based on some heuristic."""
        pass

    @abstractmethod
    def expand(self, new_state: Discrete, action: Discrete, reward: float, terminal: bool) -> 'NodeBase':
        """Expand the node by adding a new child."""
        pass

    @abstractmethod
    def update(self, reward: float) -> None:
        """Update the node's statistics (visits, reward, etc.)."""
        pass
