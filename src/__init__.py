"""
AI Silent Mental Health Detector
A production-ready system for early detection of mental health decline
using NLP and LSTM temporal modeling.
"""

__version__ = "1.0.0"
__author__ = "Production ML Team"

# Use try-except to handle both relative and absolute imports
try:
    from .pipeline import MentalHealthPipeline
    from .model import EmotionDetector
    from .lstm_model import TemporalMentalHealthModel
    from .preprocessing import TextPreprocessor
except ImportError:
    # Fallback for when src is added directly to path
    from pipeline import MentalHealthPipeline
    from model import EmotionDetector
    from lstm_model import TemporalMentalHealthModel
    from preprocessing import TextPreprocessor

__all__ = [
    'MentalHealthPipeline',
    'EmotionDetector',
    'TemporalMentalHealthModel',
    'TextPreprocessor'
]
