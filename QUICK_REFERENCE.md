# 🚀 QUICK REFERENCE GUIDE

## One-Page Cheat Sheet for AI Mental Health Detector

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🏃 Quick Start

### Option 1: Run Web App
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

### Option 2: Run Analysis Notebook
```bash
jupyter notebook analysis.ipynb
```

### Option 3: Use in Python
```python
from src.pipeline import MentalHealthPipeline

pipeline = MentalHealthPipeline()
pipeline.train_lstm_model(n_sequences=1000, epochs=30)

result = pipeline.process_text("Your text here")
print(result['emotion'], result['mental_health_score'])
```

---

## 🔑 Key Functions

### Pipeline Module (`src/pipeline.py`)

```python
pipeline = MentalHealthPipeline()

# Process single text
result = pipeline.process_text(text)

# Get LSTM prediction (after 5+ entries)
lstm_result = pipeline.lstm_predict()

# Get trend data
df = pipeline.get_trend_data()

# Recent scores
scores = pipeline.get_recent_scores(n=5)

# Clear history
pipeline.clear_history()
```

### Preprocessing (`src/preprocessing.py`)

```python
preprocessor = TextPreprocessor()

# Clean text
cleaned = preprocessor.clean_text(text)

# Extract features
features = preprocessor.extract_features(text)
# Returns: negative_word_ratio, sentiment_polarity, etc.
```

### Emotion Detection (`src/model.py`)

```python
detector = EmotionDetector()

# Predict emotion
emotion, confidence, scores = detector.predict_emotion(text)
# emotion: primary emotion label
# confidence: confidence score [0, 1]
# scores: dict of all emotion probabilities
```

### LSTM Model (`src/lstm_model.py`)

```python
lstm = TemporalMentalHealthModel(sequence_length=5, feature_dim=20)

# Build model
lstm.build_model()

# Generate training data
X, y = lstm.generate_synthetic_sequences(n_sequences=1000)

# Train
history = lstm.train(X, y, epochs=50)

# Predict risk
risk_score, risk_level = lstm.predict_risk(sequence)
# risk_level: 'low', 'moderate', or 'high'
```

---

## 📊 Understanding Output

### Mental Health Score
- **0.7 - 1.0:** Healthy (green)
- **0.4 - 0.7:** Moderate concern (yellow)
- **0.0 - 0.4:** High concern (red)

### Risk Level (LSTM)
- **Low:** Pattern shows stability
- **Moderate:** Some concerning trends
- **High:** Declining pattern detected

### Emotions Detected
- anger, disgust, fear, joy, neutral, sadness, surprise

---

## 🛠️ Common Tasks

### Task 1: Analyze Text
```python
result = pipeline.process_text("I feel overwhelmed today")
print(f"Emotion: {result['emotion']}")
print(f"Score: {result['mental_health_score']:.3f}")
```

### Task 2: Track Over Time
```python
texts = ["Day 1 text", "Day 2 text", "Day 3 text", ...]
for text in texts:
    pipeline.process_text(text)

# After 5+ entries
lstm_result = pipeline.lstm_predict()
print(f"Risk: {lstm_result['risk_level']}")
```

### Task 3: Export History
```python
df = pipeline.get_trend_data()
df.to_csv('history.csv', index=False)
```

### Task 4: Retrain LSTM
```python
pipeline.train_lstm_model(
    n_sequences=1500,
    decline_ratio=0.35,
    epochs=60
)
```

---

## 🔧 Configuration

Edit `config.ini` to customize:

```ini
[Model]
sequence_length = 5        # LSTM input length
feature_dimension = 20     # Feature vector size

[Training]
n_synthetic_sequences = 1000
epochs = 50
batch_size = 32

[Scoring]
low_risk_threshold = 0.3
high_risk_threshold = 0.6
```

---

## 📈 Visualization

### In Streamlit App
- Emotion distribution chart
- Mental health score timeline
- Risk level alerts

### In Notebook
```python
import matplotlib.pyplot as plt

scores = pipeline.get_recent_scores(10)
plt.plot(scores)
plt.axhline(y=0.7, color='g', linestyle='--')
plt.axhline(y=0.4, color='r', linestyle='--')
plt.show()
```

---

## 🐛 Troubleshooting

### Issue: Model loading slow
**Solution:** First run downloads ~500MB. Subsequent runs are fast.

### Issue: LSTM not predicting
**Solution:** Need 5+ entries. Check with `len(pipeline.history)`

### Issue: Out of memory
**Solution:** Reduce batch_size in config or use CPU instead of GPU

### Issue: Import errors
**Solution:** Ensure virtual environment activated and requirements installed

---

## 📁 File Structure

```
.
├── app.py                      # Streamlit web app
├── analysis.ipynb              # Jupyter notebook
├── requirements.txt            # Dependencies
├── README.md                   # Full documentation
├── DEPLOYMENT.md              # Production guide
├── config.ini                 # Configuration
├── test_system.py             # Testing script
├── visualize_architecture.py  # Diagram generator
└── src/
    ├── __init__.py
    ├── preprocessing.py       # Text preprocessing
    ├── model.py              # Emotion detection
    ├── lstm_model.py         # Temporal modeling
    └── pipeline.py           # Complete pipeline
```

---

## 🔐 Security Notes

- By default, data is NOT persisted
- No user data leaves the system
- Configure encryption for production
- Follow DEPLOYMENT.md for HIPAA compliance

---

## 📞 Getting Help

1. **Check documentation:** README.md
2. **Run tests:** `python test_system.py`
3. **Check examples:** analysis.ipynb
4. **Deployment:** DEPLOYMENT.md

---

## 🎯 Common Use Cases

### Use Case 1: Personal Journaling
- User writes daily entries
- System tracks emotional patterns
- Alerts on declining trends

### Use Case 2: Research Study
- Collect participant text data
- Analyze emotional trends
- Export data for further analysis

### Use Case 3: Mental Health App
- Integrate as screening module
- Connect to professional review
- Trigger interventions

---

## ⚡ Performance Tips

1. **Batch Processing:** Process multiple texts together
2. **Model Caching:** Load models once, reuse
3. **GPU Usage:** Set CUDA_VISIBLE_DEVICES for GPU
4. **Feature Caching:** Cache preprocessor results

---

## 🧪 Testing

```bash
# Quick system test
python test_system.py

# Run with custom text
python -c "
from src.pipeline import MentalHealthPipeline
p = MentalHealthPipeline()
result = p.process_text('Your text here')
print(result)
"
```

---

## 📊 Example Output

```python
{
    'emotion': 'sadness',
    'emotion_confidence': 0.78,
    'mental_health_score': 0.42,
    'emotion_scores': {
        'anger': 0.05,
        'disgust': 0.02,
        'fear': 0.08,
        'joy': 0.02,
        'neutral': 0.05,
        'sadness': 0.78,
        'surprise': 0.00
    },
    'features': {
        'negative_word_ratio': 0.25,
        'sentiment_polarity': -0.6,
        'first_person_ratio': 0.15,
        ...
    }
}
```

---

## 🚀 Production Deployment

Quick deploy to cloud:

```bash
# Docker
docker build -t mental-health-detector .
docker run -p 8501:8501 mental-health-detector

# Heroku
heroku create my-mental-health-app
git push heroku main
```

See DEPLOYMENT.md for complete guide.

---

## ⚠️ Remember

- This is a screening tool, NOT a diagnostic tool
- Always involve mental health professionals
- Have crisis resources readily available
- Respect user privacy and data security

**Crisis Resources:**
- US: 988 (Suicide & Crisis Lifeline)
- Text: HOME to 741741

---

**Quick Reference v1.0 | AI Mental Health Detector**
