"""
Complete pipeline for mental health detection system.
Integrates preprocessing, emotion detection, feature extraction, and LSTM prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Use relative imports for package modules
from preprocessing import TextPreprocessor
from model import EmotionDetector
from lstm_model import TemporalMentalHealthModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MentalHealthPipeline:
    """Complete pipeline for mental health detection with temporal modeling."""
    
    def __init__(
        self,
        emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        sequence_length: int = 5,
        feature_dim: int = 20
    ):
        """
        Initialize the complete pipeline.
        
        Args:
            emotion_model_name: HuggingFace model for emotion detection
            sequence_length: Number of time steps for LSTM
            feature_dim: Feature vector dimension
        """
        self.preprocessor = TextPreprocessor()
        self.emotion_detector = EmotionDetector(emotion_model_name)
        self.lstm_model = TemporalMentalHealthModel(sequence_length, feature_dim)
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        
        # Session history
        self.history: List[Dict] = []
        
        logger.info("Mental Health Pipeline initialized")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        return self.preprocessor.preprocess_text(text)
    
    def predict_emotion(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict emotion from text using transformer model.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (emotion, confidence, emotion_scores)
        """
        return self.emotion_detector.predict_emotion(text)
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic and statistical features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of features
        """
        return self.preprocessor.extract_features(text)
    
    def compute_mental_health_score(
        self,
        text: str,
        emotion: str,
        confidence: float,
        emotion_scores: Dict[str, float],
        features: Dict[str, float]
    ) -> float:
        """
        Compute mental health score combining multiple signals.
        
        Args:
            text: Input text
            emotion: Detected primary emotion
            confidence: Confidence score
            emotion_scores: All emotion probabilities
            features: Extracted features
            
        Returns:
            Mental health score (0-1, higher = better mental health)
        """
        # Start with base score
        base_score = 0.7
        
        # Emotion-based adjustment
        emotion_weights = {
            'joy': 0.15,
            'surprise': 0.05,
            'neutral': 0.0,
            'sadness': -0.20,
            'anger': -0.15,
            'fear': -0.18,
            'disgust': -0.12
        }
        
        emotion_adjustment = sum(
            emotion_scores[emo] * weight
            for emo, weight in emotion_weights.items()
        )
        
        # Feature-based adjustments
        negative_word_penalty = features['negative_word_ratio'] * -0.15
        sentiment_boost = features['sentiment_polarity'] * 0.10
        intensity_penalty = features['intensity_ratio'] * -0.08
        
        # High first-person usage can indicate rumination
        first_person_penalty = max(0, (features['first_person_ratio'] - 0.15)) * -0.10
        
        # Combine all factors
        total_score = (
            base_score +
            emotion_adjustment +
            negative_word_penalty +
            sentiment_boost +
            intensity_penalty +
            first_person_penalty
        )
        
        # Normalize to [0, 1]
        final_score = np.clip(total_score, 0.0, 1.0)
        
        return final_score
    
    def create_feature_vector(
        self,
        emotion_scores: Dict[str, float],
        features: Dict[str, float],
        mental_health_score: float
    ) -> np.ndarray:
        """
        Create feature vector for LSTM model.
        
        Args:
            emotion_scores: Emotion probabilities
            features: Extracted features
            mental_health_score: Computed mental health score
            
        Returns:
            Feature vector of size feature_dim
        """
        # Ordered emotion labels
        emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        
        # Build feature vector
        feature_vector = [mental_health_score]
        
        # Add emotion scores
        feature_vector.extend([emotion_scores[label] for label in emotion_labels])
        
        # Add key linguistic features
        feature_vector.extend([
            features['negative_word_ratio'],
            features['sentiment_polarity'],
            features['first_person_ratio'],
            features['intensity_ratio'],
            features.get('sentiment_subjectivity', 0.0),
            features['avg_sentence_length'] / 20.0,  # Normalize
            features['punctuation_ratio'],
            features['capital_ratio']
        ])
        
        # Pad or truncate to feature_dim
        if len(feature_vector) < self.feature_dim:
            feature_vector.extend([0.0] * (self.feature_dim - len(feature_vector)))
        else:
            feature_vector = feature_vector[:self.feature_dim]
        
        return np.array(feature_vector)
    
    def generate_sequence_from_history(self) -> Optional[np.ndarray]:
        """
        Generate sequence from session history for LSTM prediction.
        
        Returns:
            Sequence array or None if insufficient history
        """
        if len(self.history) < self.sequence_length:
            return None
        
        # Take last N entries
        recent_history = self.history[-self.sequence_length:]
        
        # Extract feature vectors
        sequence = np.array([entry['feature_vector'] for entry in recent_history])
        
        return sequence
    
    def process_text(self, text: str, store_history: bool = True) -> Dict:
        """
        Complete processing pipeline for a single text input.
        
        Args:
            text: Input text
            store_history: Whether to store in session history
            
        Returns:
            Dictionary with all analysis results
        """
        # Step 1: Preprocess
        cleaned_text = self.preprocess_text(text)
        
        # Step 2: Emotion detection
        emotion, confidence, emotion_scores = self.predict_emotion(cleaned_text)
        
        # Step 3: Feature extraction
        features = self.extract_features(cleaned_text)
        
        # Step 4: Compute mental health score
        mental_health_score = self.compute_mental_health_score(
            cleaned_text, emotion, confidence, emotion_scores, features
        )
        
        # Step 5: Create feature vector for LSTM
        feature_vector = self.create_feature_vector(
            emotion_scores, features, mental_health_score
        )
        
        # Compile results
        result = {
            'timestamp': datetime.now().isoformat(),
            'original_text': text,
            'cleaned_text': cleaned_text,
            'emotion': emotion,
            'emotion_confidence': confidence,
            'emotion_scores': emotion_scores,
            'features': features,
            'mental_health_score': mental_health_score,
            'feature_vector': feature_vector
        }
        
        # Store in history
        if store_history:
            self.history.append(result)
        
        return result
    
    def lstm_predict(self) -> Optional[Dict]:
        """
        Make LSTM-based risk prediction using session history.
        
        Returns:
            Dictionary with risk prediction or None if insufficient data
        """
        # Generate sequence
        sequence = self.generate_sequence_from_history()
        
        if sequence is None:
            return {
                'has_prediction': False,
                'message': f'Need at least {self.sequence_length} entries for temporal analysis'
            }
        
        # Make prediction
        if not self.lstm_model.is_trained:
            return {
                'has_prediction': False,
                'message': 'LSTM model not trained yet'
            }
        
        risk_score, risk_level = self.lstm_model.predict_risk(sequence)
        
        return {
            'has_prediction': True,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'sequence_length': len(self.history),
            'message': f'Temporal analysis based on last {self.sequence_length} entries'
        }
    
    def get_trend_data(self) -> pd.DataFrame:
        """
        Get historical trend data as DataFrame.
        
        Returns:
            DataFrame with timestamp and scores
        """
        if not self.history:
            return pd.DataFrame()
        
        data = {
            'timestamp': [entry['timestamp'] for entry in self.history],
            'mental_health_score': [entry['mental_health_score'] for entry in self.history],
            'emotion': [entry['emotion'] for entry in self.history],
            'emotion_confidence': [entry['emotion_confidence'] for entry in self.history]
        }
        
        return pd.DataFrame(data)
    
    def get_recent_scores(self, n: int = 5) -> List[float]:
        """
        Get most recent mental health scores.
        
        Args:
            n: Number of recent scores to return
            
        Returns:
            List of scores
        """
        if not self.history:
            return []
        
        recent = self.history[-n:]
        return [entry['mental_health_score'] for entry in recent]
    
    def clear_history(self):
        """Clear session history."""
        self.history = []
        logger.info("Session history cleared")
    
    def train_lstm_model(
        self,
        n_sequences: int = 1000,
        decline_ratio: float = 0.3,
        epochs: int = 50
    ):
        """
        Train the LSTM model on synthetic data.
        
        Args:
            n_sequences: Number of synthetic sequences
            decline_ratio: Ratio of declining sequences
            epochs: Training epochs
        """
        logger.info("Generating synthetic training data...")
        X, y = self.lstm_model.generate_synthetic_sequences(n_sequences, decline_ratio)
        
        logger.info("Training LSTM model...")
        history = self.lstm_model.train(X, y, epochs=epochs)
        
        logger.info("LSTM model training completed")
        return history
    
    def save_lstm_model(self, filepath: str):
        """Save trained LSTM model."""
        self.lstm_model.save_model(filepath)
    
    def load_lstm_model(self, filepath: str):
        """Load trained LSTM model."""
        self.lstm_model.load_model(filepath)
