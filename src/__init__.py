"""TAEG package initialization."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .data_loader import DataLoader, Event, VerseReference, Verse
from .graph_builder import TAEGGraphBuilder
from .models import TAEGModel, PegasusBaseline, PrimeraBaseline, LexRankBaseline

__all__ = [
    "DataLoader",
    "Event", 
    "VerseReference",
    "Verse",
    "TAEGGraphBuilder",
    "TAEGModel",
    "PegasusBaseline",
    "PrimeraBaseline", 
    "LexRankBaseline"
]