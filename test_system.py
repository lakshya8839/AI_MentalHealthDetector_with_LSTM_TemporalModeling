#!/usr/bin/env python3
"""
Quick test script to verify the mental health detection system.
Run this after installation to ensure everything is working.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("AI MENTAL HEALTH DETECTOR - SYSTEM TEST")
print("=" * 80)

# Test 1: Import modules
print("\n[1/5] Testing module imports...")
try:
    from src.preprocessing import TextPreprocessor
    from src.model import EmotionDetector
    from src.lstm_model import TemporalMentalHealthModel
    from src.pipeline import MentalHealthPipeline
    print("✅ All modules imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize preprocessor
print("\n[2/5] Testing preprocessor...")
try:
    preprocessor = TextPreprocessor()
    test_text = "I'm feeling really overwhelmed and anxious today."
    cleaned = preprocessor.clean_text(test_text)
    features = preprocessor.extract_features(test_text)
    print(f"✅ Preprocessor working")
    print(f"   Features extracted: {len(features)} dimensions")
except Exception as e:
    print(f"❌ Preprocessor failed: {e}")
    sys.exit(1)

# Test 3: Load emotion model
print("\n[3/5] Loading emotion detection model...")
print("   (This may take 1-2 minutes on first run)")
try:
    emotion_detector = EmotionDetector()
    emotion, confidence, scores = emotion_detector.predict_emotion(test_text)
    print(f"✅ Emotion model loaded and working")
    print(f"   Detected: {emotion.upper()} (confidence: {confidence:.2%})")
except Exception as e:
    print(f"❌ Emotion detection failed: {e}")
    sys.exit(1)

# Test 4: Initialize LSTM
print("\n[4/5] Testing LSTM model...")
try:
    lstm_model = TemporalMentalHealthModel(sequence_length=5, feature_dim=20)
    lstm_model.build_model()
    print("✅ LSTM model initialized")
    print(f"   Architecture: Input(5, 20) -> LSTM(64) -> LSTM(32) -> Dense(1)")
except Exception as e:
    print(f"❌ LSTM initialization failed: {e}")
    sys.exit(1)

# Test 5: Full pipeline
print("\n[5/5] Testing complete pipeline...")
try:
    pipeline = MentalHealthPipeline()
    
    # Process a few texts
    test_texts = [
        "I had a great day today! Feeling positive and energized.",
        "Work is stressful but I'm managing okay.",
        "I feel so tired and hopeless. Nothing seems to help.",
    ]
    
    print("\n   Processing sample texts:")
    for i, text in enumerate(test_texts, 1):
        result = pipeline.process_text(text)
        print(f"   {i}. Emotion: {result['emotion'].upper():8} | "
              f"Score: {result['mental_health_score']:.3f} | "
              f"Text: {text[:40]}...")
    
    print("\n✅ Complete pipeline working")
except Exception as e:
    print(f"❌ Pipeline test failed: {e}")
    sys.exit(1)

# Success!
print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nYour system is ready to use!")
print("\nNext steps:")
print("  1. Run the Streamlit app:  streamlit run app.py")
print("  2. Open Jupyter notebook:  jupyter notebook analysis.ipynb")
print("  3. Use in your code:       from src.pipeline import MentalHealthPipeline")
print("\n" + "=" * 80)
