"""
Emotion detection model using transformer-based architecture.
Uses j-hartmann/emotion-english-distilroberta-base for emotion classification.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionDetector:
    """Wrapper for transformer-based emotion detection model."""
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize the emotion detection model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading emotion detection model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Emotion labels for this model
            self.emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_emotion(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict emotion from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (primary_emotion, confidence_score, all_emotion_scores)
        """
        if not text or not text.strip():
            return "neutral", 0.0, {label: 0.0 for label in self.emotion_labels}
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            probs_np = probs.cpu().numpy()[0]
            
            # Get primary emotion and confidence
            primary_idx = np.argmax(probs_np)
            primary_emotion = self.emotion_labels[primary_idx]
            confidence = float(probs_np[primary_idx])
            
            # Create dictionary of all emotion scores
            emotion_scores = {
                label: float(score) 
                for label, score in zip(self.emotion_labels, probs_np)
            }
            
            return primary_emotion, confidence, emotion_scores
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "neutral", 0.0, {label: 0.0 for label in self.emotion_labels}
    
    def batch_predict(self, texts: list) -> list:
        """
        Predict emotions for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of (emotion, confidence, scores) tuples
        """
        results = []
        for text in texts:
            results.append(self.predict_emotion(text))
        return results
    
    def get_emotion_vector(self, text: str) -> np.ndarray:
        """
        Get emotion probability vector for downstream models.
        
        Args:
            text: Input text
            
        Returns:
            NumPy array of emotion probabilities
        """
        _, _, emotion_scores = self.predict_emotion(text)
        return np.array([emotion_scores[label] for label in self.emotion_labels])
