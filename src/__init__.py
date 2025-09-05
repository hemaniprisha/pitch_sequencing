"""
MLB Pitch Sequencing Optimization Package
Advanced baseball analytics for predictive pitch sequencing
"""

from .data_processing import DataProcessor
from .feature_engineering import FeatureEngineer
from .modeling import MLBModel
from .recommendation_engine import PitchRecommendationEngine
from .simulation import GameSimulator
from .monitoring import ModelMonitor

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    'DataProcessor',
    'FeatureEngineer', 
    'MLBModel',
    'PitchRecommendationEngine',
    'GameSimulator',
    'ModelMonitor'
]