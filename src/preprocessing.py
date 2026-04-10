"""
Text preprocessing and feature extraction module.
Handles text cleaning, normalization, and feature engineering.
"""

import re
import string
import numpy as np
from typing import Dict, List
import nltk
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class TextPreprocessor:
    """Handles text preprocessing and feature extraction."""
    
    def __init__(self):
        self.negative_words = {
            'sad', 'depressed', 'lonely', 'anxious', 'worried', 'hopeless',
            'worthless', 'tired', 'exhausted', 'empty', 'numb', 'hurt',
            'pain', 'suffering', 'cry', 'tears', 'dark', 'alone', 'isolated',
            'scared', 'fear', 'panic', 'stress', 'overwhelmed', 'broken',
            'lost', 'helpless', 'desperate', 'miserable', 'unhappy', 'grief',
            'dying', 'death', 'suicide', 'kill', 'end', 'give up', 'quit',
            'hate', 'angry', 'furious', 'frustrated', 'annoyed', 'irritated',
            'disappointed', 'failed', 'failure', 'regret', 'guilty', 'shame'
        }
        
        self.intensity_words = {
            'very', 'extremely', 'incredibly', 'absolutely', 'completely',
            'totally', 'utterly', 'really', 'truly', 'deeply', 'severely'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text while preserving emotional content."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove mentions and hashtags (but keep the word)
        text = re.sub(r'[@#]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract numerical features from text for mental health assessment."""
        features = {}
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Basic text statistics
        words = cleaned_text.split()
        features['word_count'] = len(words)
        features['char_count'] = len(cleaned_text)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        
        # Sentence count and complexity
        sentences = nltk.sent_tokenize(text)
        features['sentence_count'] = len(sentences)
        features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
        
        # Negative word analysis
        negative_count = sum(1 for word in words if word in self.negative_words)
        features['negative_word_count'] = negative_count
        features['negative_word_ratio'] = negative_count / len(words) if words else 0
        
        # Intensity word analysis
        intensity_count = sum(1 for word in words if word in self.intensity_words)
        features['intensity_word_count'] = intensity_count
        features['intensity_ratio'] = intensity_count / len(words) if words else 0
        
        # First-person pronoun analysis (self-focus indicator)
        first_person = ['i', 'me', 'my', 'mine', 'myself']
        first_person_count = sum(1 for word in words if word in first_person)
        features['first_person_count'] = first_person_count
        features['first_person_ratio'] = first_person_count / len(words) if words else 0
        
        # Sentiment polarity using TextBlob
        try:
            blob = TextBlob(text)
            features['sentiment_polarity'] = blob.sentiment.polarity
            features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        except:
            features['sentiment_polarity'] = 0.0
            features['sentiment_subjectivity'] = 0.0
        
        # Punctuation analysis (emotional expression)
        exclamation_count = text.count('!')
        question_count = text.count('?')
        features['exclamation_count'] = exclamation_count
        features['question_count'] = question_count
        features['punctuation_ratio'] = (exclamation_count + question_count) / len(text) if text else 0
        
        # Capitalization (shouting/emphasis)
        capital_count = sum(1 for c in text if c.isupper())
        features['capital_ratio'] = capital_count / len(text) if text else 0
        
        return features
    
    def compute_emotional_intensity(self, features: Dict[str, float], emotion_scores: Dict[str, float]) -> float:
        """
        Compute overall emotional intensity score combining features and emotion detection.
        Returns a value between 0 and 1.
        """
        # Weight different aspects
        weights = {
            'negative_words': 0.25,
            'sentiment': 0.20,
            'intensity_words': 0.15,
            'first_person': 0.15,
            'emotion_strength': 0.25
        }
        
        # Negative word component
        negative_component = min(features['negative_word_ratio'] * 10, 1.0)
        
        # Sentiment component (inverted polarity - more negative = higher score)
        sentiment_component = (1 - (features['sentiment_polarity'] + 1) / 2)
        
        # Intensity word component
        intensity_component = min(features['intensity_ratio'] * 20, 1.0)
        
        # First person component (high self-focus)
        first_person_component = min(features['first_person_ratio'] * 5, 1.0)
        
        # Emotion strength component (from NLP model)
        negative_emotions = ['sadness', 'anger', 'fear', 'disgust']
        emotion_strength = sum(emotion_scores.get(emo, 0) for emo in negative_emotions)
        emotion_component = min(emotion_strength, 1.0)
        
        # Weighted combination
        intensity_score = (
            weights['negative_words'] * negative_component +
            weights['sentiment'] * sentiment_component +
            weights['intensity_words'] * intensity_component +
            weights['first_person'] * first_person_component +
            weights['emotion_strength'] * emotion_component
        )
        
        return intensity_score
    
    def preprocess_text(self, text: str) -> str:
        """Main preprocessing function - clean text for model input."""
        return self.clean_text(text)
