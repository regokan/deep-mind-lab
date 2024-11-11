"""Policy classes."""

from .alphaMC import AlphaMCPolicy
from .base import BasePolicy
from .crossEntropy import CrossEntropyPolicy
from .discretizedSarsaMax import DiscretizedSarsaMaxPolicy
from .dqn import DQNPolicy
from .everyVisitMC import EveryVisitMCPolicy
from .firstVisitMC import FirstVisitMCPolicy
from .hillClimbing import HillClimbingPolicy
from .random import RandomPolicy
from .reinforce import ReinforcePolicy
from .sarsa import SarsaPolicy
from .sarsaExpected import SarsaExpectedPolicy
from .sarsaMax import SarsaMaxPolicy

__all__ = [
    "BasePolicy",
    "FirstVisitMCPolicy",
    "EveryVisitMCPolicy",
    "RandomPolicy",
    "SarsaPolicy",
    "SarsaMaxPolicy",
    "SarsaExpectedPolicy",
    "AlphaMCPolicy",
    "DiscretizedSarsaMaxPolicy",
    "DQNPolicy",
    "HillClimbingPolicy",
    "CrossEntropyPolicy",
    "ReinforcePolicy",
]
