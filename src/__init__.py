"""矿点检测系统包"""

from .data_loader import MiningDataLoader
from .models import MiningNN, MiningLSTM, TraditionalModels
from .train import DeepLearningTrainer, TraditionalModelTrainer
from .evaluate import ModelEvaluator

__version__ = '1.0.0'
__author__ = 'wxlyy2025'

__all__ = [
    'MiningDataLoader',
    'MiningNN',
    'MiningLSTM',
    'TraditionalModels',
    'DeepLearningTrainer',
    'TraditionalModelTrainer',
    'ModelEvaluator',
]
