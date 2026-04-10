"""
LSTM-based temporal model for mental health risk prediction.
Analyzes sequences of emotional states over time to detect patterns of decline.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalMentalHealthModel:
    """LSTM model for temporal pattern recognition in mental health indicators."""
    
    def __init__(self, sequence_length: int = 5, feature_dim: int = 20):
        """
        Initialize LSTM model for temporal mental health analysis.
        
        Args:
            sequence_length: Number of time steps to consider
            feature_dim: Dimension of feature vector at each time step
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def build_model(self) -> keras.Model:
        """
        Build LSTM architecture for sequence modeling.
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, self.feature_dim)),
            
            # LSTM layer with dropout for regularization
            layers.LSTM(
                64,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            
            # Second LSTM layer
            layers.LSTM(
                32,
                return_sequences=False,
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            
            # Dense layers
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            
            # Output layer - risk score between 0 and 1
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        self.model = model
        logger.info("LSTM model built successfully")
        logger.info(f"Model architecture:\n{model.summary()}")
        
        return model
    
    def generate_synthetic_sequences(
        self,
        n_sequences: int = 1000,
        decline_ratio: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data with realistic patterns.
        
        Args:
            n_sequences: Number of sequences to generate
            decline_ratio: Ratio of sequences showing mental health decline
            
        Returns:
            Tuple of (X_sequences, y_labels)
        """
        X = []
        y = []
        
        n_decline = int(n_sequences * decline_ratio)
        n_stable = n_sequences - n_decline
        
        # Generate declining sequences (high risk)
        for _ in range(n_decline):
            seq = self._generate_declining_sequence()
            X.append(seq)
            y.append(1)  # High risk
        
        # Generate stable/improving sequences (low risk)
        for _ in range(n_stable):
            seq = self._generate_stable_sequence()
            X.append(seq)
            y.append(0)  # Low risk
        
        X = np.array(X)
        y = np.array(y)
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        logger.info(f"Generated {n_sequences} synthetic sequences")
        logger.info(f"Shape: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def _generate_declining_sequence(self) -> np.ndarray:
        """Generate a sequence showing mental health decline."""
        sequence = []
        
        # Start with moderate to low scores
        initial_score = np.random.uniform(0.3, 0.6)
        
        for i in range(self.sequence_length):
            # Progressive decline with noise
            decline_factor = (i + 1) / self.sequence_length
            noise = np.random.normal(0, 0.05)
            score = max(0.1, initial_score - (decline_factor * 0.3) + noise)
            
            # Generate feature vector
            features = self._create_feature_vector(score, declining=True)
            sequence.append(features)
        
        return np.array(sequence)
    
    def _generate_stable_sequence(self) -> np.ndarray:
        """Generate a sequence showing stable or improving mental health."""
        sequence = []
        
        # Start with moderate to high scores
        initial_score = np.random.uniform(0.5, 0.8)
        trend = np.random.choice(['stable', 'improving'])
        
        for i in range(self.sequence_length):
            if trend == 'stable':
                noise = np.random.normal(0, 0.08)
                score = np.clip(initial_score + noise, 0.4, 0.9)
            else:  # improving
                improvement_factor = (i + 1) / self.sequence_length
                noise = np.random.normal(0, 0.05)
                score = min(0.9, initial_score + (improvement_factor * 0.2) + noise)
            
            # Generate feature vector
            features = self._create_feature_vector(score, declining=False)
            sequence.append(features)
        
        return np.array(sequence)
    
    def _create_feature_vector(self, base_score: float, declining: bool) -> np.ndarray:
        """Create a realistic feature vector based on mental health score."""
        features = []
        
        # Core mental health score
        features.append(base_score)
        
        # Emotion probabilities (7 emotions)
        if declining:
            # Higher negative emotions
            sadness = np.random.beta(3, 2) * (1 - base_score)
            anger = np.random.beta(2, 3) * (1 - base_score)
            fear = np.random.beta(2, 3) * (1 - base_score)
            disgust = np.random.beta(1, 4) * (1 - base_score)
            joy = np.random.beta(2, 5) * base_score
            surprise = np.random.beta(1, 3) * 0.2
            neutral = max(0, 1 - (sadness + anger + fear + disgust + joy + surprise))
        else:
            # Lower negative emotions
            sadness = np.random.beta(1, 4) * (1 - base_score)
            anger = np.random.beta(1, 5) * (1 - base_score)
            fear = np.random.beta(1, 5) * (1 - base_score)
            disgust = np.random.beta(1, 6) * (1 - base_score)
            joy = np.random.beta(3, 2) * base_score
            surprise = np.random.beta(2, 3) * 0.3
            neutral = max(0, 1 - (sadness + anger + fear + disgust + joy + surprise))
        
        features.extend([anger, disgust, fear, joy, neutral, sadness, surprise])
        
        # Linguistic features
        negative_word_ratio = (1 - base_score) * np.random.beta(2, 3)
        sentiment_polarity = (base_score - 0.5) * 2 + np.random.normal(0, 0.1)
        first_person_ratio = (1 - base_score) * np.random.beta(2, 2) * 0.3
        intensity_ratio = (1 - base_score) * np.random.beta(2, 3) * 0.2
        
        features.extend([
            negative_word_ratio,
            sentiment_polarity,
            first_person_ratio,
            intensity_ratio
        ])
        
        # Add some noise features for robustness
        features.extend(np.random.normal(0, 0.1, self.feature_dim - len(features)))
        
        return np.array(features[:self.feature_dim])
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32
    ) -> keras.callbacks.History:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Normalize features
        n_samples, seq_len, n_features = X_train.shape
        X_reshaped = X_train.reshape(-1, n_features)
        X_normalized = self.scaler.fit_transform(X_reshaped)
        X_train_scaled = X_normalized.reshape(n_samples, seq_len, n_features)
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train model
        logger.info("Starting model training...")
        history = self.model.fit(
            X_train_scaled,
            y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        self.is_trained = True
        logger.info("Model training completed")
        
        return history
    
    def predict_risk(self, sequence: np.ndarray) -> Tuple[float, str]:
        """
        Predict mental health risk from a sequence.
        
        Args:
            sequence: Input sequence of shape (sequence_length, feature_dim)
            
        Returns:
            Tuple of (risk_score, risk_level)
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning default prediction")
            return 0.5, "moderate"
        
        # Ensure correct shape
        if sequence.ndim == 2:
            sequence = sequence.reshape(1, self.sequence_length, self.feature_dim)
        
        # Normalize
        seq_reshaped = sequence.reshape(-1, self.feature_dim)
        seq_normalized = self.scaler.transform(seq_reshaped)
        seq_scaled = seq_normalized.reshape(1, self.sequence_length, self.feature_dim)
        
        # Predict
        risk_score = float(self.model.predict(seq_scaled, verbose=0)[0][0])
        
        # Classify risk level
        if risk_score < 0.3:
            risk_level = "low"
        elif risk_score < 0.6:
            risk_level = "moderate"
        else:
            risk_level = "high"
        
        return risk_score, risk_level
    
    def save_model(self, filepath: str):
        """Save model and scaler."""
        if self.model is not None:
            self.model.save(f"{filepath}_lstm.h5")
            with open(f"{filepath}_scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and scaler."""
        self.model = keras.models.load_model(f"{filepath}_lstm.h5")
        with open(f"{filepath}_scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
