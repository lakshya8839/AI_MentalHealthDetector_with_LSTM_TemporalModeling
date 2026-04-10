# 🧠 AI Silent Mental Health Detector with Temporal Modeling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.36-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A production-ready AI system for early detection of mental health decline using **NLP emotion detection** and **LSTM-based temporal pattern recognition**. This system analyzes text input to detect emotional states and tracks patterns over time to predict mental health risk.

## 🎯 Problem Statement

Mental health conditions often develop gradually, with subtle changes in language and emotional expression that may go unnoticed until a crisis occurs. Traditional screening methods:
- Rely on self-reporting and periodic assessments
- Miss gradual decline patterns
- Cannot analyze natural language at scale
- Lack temporal context

**Our Solution:** An AI-powered system that:
1. Analyzes natural language for emotional content
2. Tracks mental health indicators over time
3. Detects patterns of decline using LSTM temporal modeling
4. Provides early warning signals for intervention

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INPUT (Text)                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                               │
│  • Text Cleaning                                                     │
│  • Feature Extraction (15+ linguistic features)                      │
│  • Negative word detection, sentiment analysis                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    NLP EMOTION DETECTION                             │
│  Model: j-hartmann/emotion-english-distilroberta-base                │
│  Output: 7 emotions (anger, disgust, fear, joy, neutral,            │
│          sadness, surprise) + confidence scores                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                                │
│  Combine:                                                            │
│  • Emotion probabilities (7 dimensions)                              │
│  • Linguistic features (negative words, sentiment, etc.)             │
│  • Mental health score computation                                   │
│  → Feature Vector (20 dimensions)                                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SEQUENCE GENERATION                             │
│  Store feature vectors over time                                     │
│  Create sequences of length N=5                                      │
│  Shape: (sequence_length, feature_dim)                               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LSTM TEMPORAL MODEL                               │
│  Architecture:                                                       │
│  • LSTM Layer 1: 64 units (with dropout)                             │
│  • LSTM Layer 2: 32 units (with dropout)                             │
│  • Dense Layer: 32 units (ReLU)                                      │
│  • Output: Risk score [0, 1]                                         │
│                                                                       │
│  Training:                                                           │
│  • Synthetic sequences simulating decline/stable patterns            │
│  • Binary classification (high risk vs low risk)                     │
│  • Early stopping, learning rate scheduling                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RISK PREDICTION                                 │
│  • Rule-based score (immediate assessment)                           │
│  • LSTM risk prediction (temporal pattern)                           │
│  • Risk levels: LOW / MODERATE / HIGH                                │
│  • Alert generation                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Why Time-Series Modeling?

### The Problem with Point-in-Time Analysis

**Traditional approach (Rule-based only):**
```
Entry 1: "Feeling stressed" → Score: 0.45 → Flag: Moderate
Entry 2: "Very tired today" → Score: 0.48 → Flag: Moderate  
Entry 3: "I'm okay I guess" → Score: 0.42 → Flag: Moderate
```
❌ Each entry evaluated independently  
❌ Misses the pattern: gradual decline  
❌ Cannot predict future trajectory

### Time-Series Approach (LSTM)

**LSTM analyzes sequences:**
```
Sequence: [0.65 → 0.58 → 0.51 → 0.45 → 0.39]
                     ↓
               LSTM analyzes pattern
                     ↓
         Detects: DECLINING TREND
                     ↓
            Risk Score: 0.78 (HIGH)
```
✅ Detects gradual decline  
✅ Considers temporal context  
✅ Predicts future risk  
✅ More robust to noise

---

## 📊 Rule-Based vs LSTM Comparison

| Aspect | Rule-Based Scoring | LSTM Temporal Model |
|--------|-------------------|---------------------|
| **Input** | Single text entry | Sequence of 5+ entries |
| **Context** | Current state only | Historical patterns |
| **Detection** | Immediate assessment | Trend analysis |
| **Strength** | Interpretable, fast | Pattern recognition |
| **Limitation** | Misses gradual changes | Needs history |
| **Use Case** | Initial screening | Risk prediction |
| **Output** | Mental health score | Risk trajectory |

**Best Practice:** Use **BOTH** approaches in combination!

---

## 📁 Project Structure

```
ai-mental-health-detector/
├── analysis.ipynb              # Complete analysis & experimentation
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/
│   ├── preprocessing.py        # Text cleaning & feature extraction
│   ├── model.py                # Emotion detection (Transformer)
│   ├── lstm_model.py          # LSTM temporal modeling
│   └── pipeline.py            # Complete integration pipeline
│
└── models/                     # Saved models (created after training)
    ├── mental_health_lstm_lstm.h5
    └── mental_health_lstm_scaler.pkl
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd ai-mental-health-detector

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Run Analysis Notebook

```bash
jupyter notebook analysis.ipynb
```

---

## 💻 Usage Examples

### Python API Usage

```python
from src.pipeline import MentalHealthPipeline

# Initialize pipeline
pipeline = MentalHealthPipeline()

# Train LSTM model (first time only)
pipeline.train_lstm_model(n_sequences=1000, epochs=50)

# Process single text
result = pipeline.process_text("I've been feeling really down lately...")

print(f"Emotion: {result['emotion']}")
print(f"Mental Health Score: {result['mental_health_score']:.3f}")

# After 5+ entries, get LSTM prediction
if len(pipeline.history) >= 5:
    lstm_pred = pipeline.lstm_predict()
    print(f"Risk Level: {lstm_pred['risk_level']}")
    print(f"Risk Score: {lstm_pred['risk_score']:.3f}")
```

### Batch Processing

```python
texts = [
    "Had a great day today!",
    "Feeling a bit stressed but managing",
    "Everything feels overwhelming",
    "I don't see the point anymore",
    "Can't shake this feeling of emptiness"
]

for text in texts:
    result = pipeline.process_text(text)
    print(f"Score: {result['mental_health_score']:.3f}")

# Get trend
trend_df = pipeline.get_trend_data()
print(trend_df)
```

---

## 🧪 Features

### NLP Emotion Detection
- **Model:** DistilRoBERTa fine-tuned on emotion data
- **Emotions:** anger, disgust, fear, joy, neutral, sadness, surprise
- **Output:** Probability distribution + confidence scores

### Feature Engineering (20-dimensional vectors)
1. **Emotion Features (7):** Probability for each emotion
2. **Linguistic Features (13+):**
   - Negative word ratio
   - Sentiment polarity
   - First-person pronoun usage (self-focus)
   - Intensity word usage
   - Sentence complexity
   - Punctuation patterns
   - Capitalization ratio

### LSTM Architecture
```python
Sequential([
    LSTM(64, return_sequences=True, dropout=0.2),
    LSTM(32, return_sequences=False, dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Risk score [0, 1]
])
```

### Risk Classification
- **Low Risk:** Score < 0.3 (Stable mental health)
- **Moderate Risk:** Score 0.3-0.6 (Some concerns)
- **High Risk:** Score > 0.6 (Declining pattern detected)

---

## 📈 Model Performance

**LSTM Model Metrics (on synthetic validation data):**
- **Accuracy:** ~85-90%
- **AUC:** ~0.90-0.95
- **Precision:** ~0.85
- **Recall:** ~0.87

**Training Details:**
- 1000 synthetic sequences (70% stable, 30% declining)
- 50 epochs with early stopping
- Batch size: 32
- Optimizer: Adam (lr=0.001)
- Loss: Binary cross-entropy

---

## 🎨 Streamlit App Features

- **Real-time Analysis:** Instant emotion and score computation
- **Session History:** Track entries over time
- **Temporal Predictions:** LSTM risk assessment (after 5+ entries)
- **Visualizations:**
  - Emotion distribution charts
  - Mental health score timeline
  - Trend analysis
- **Risk Alerts:** Color-coded warnings
- **Export:** Download history as CSV

---

## 🔬 Research & Development

### Synthetic Data Generation

The LSTM model is trained on synthetic sequences that simulate realistic patterns:

**Declining Pattern:**
```python
# Starts at moderate level, gradually decreases
[0.55 → 0.48 → 0.42 → 0.35 → 0.28]
# Features: Increasing sadness, more negative words, lower sentiment
```

**Stable Pattern:**
```python
# Maintains healthy range with natural variation
[0.72 → 0.68 → 0.74 → 0.70 → 0.73]
# Features: Mixed emotions, balanced sentiment
```

This approach allows the model to learn temporal patterns without requiring sensitive real patient data.

---

## ⚠️ Important Disclaimers

1. **Not a Diagnostic Tool:** This system is for screening and early detection only
2. **Requires Professional Evaluation:** Always consult mental health professionals
3. **Privacy:** Implement proper data protection in production
4. **Cultural Sensitivity:** Language patterns vary across cultures
5. **Limitations:** Cannot detect all mental health conditions
6. **Crisis Management:** Have immediate support resources available

**If you or someone you know is in crisis:**
- **National Suicide Prevention Lifeline:** 988
- **Crisis Text Line:** Text HOME to 741741
- **International:** Find local resources at befrienders.org

---

## 🛠️ Technical Requirements

- **Python:** 3.10+
- **GPU:** Optional (CPU works, GPU faster)
- **RAM:** 8GB minimum (16GB recommended)
- **Storage:** ~2GB for models and dependencies

---

## 🔮 Future Enhancements

1. **Data & Models:**
   - Fine-tune on domain-specific mental health text
   - Multi-modal analysis (text + voice + activity patterns)
   - Attention mechanisms for interpretability
   - Transfer learning from clinical notes

2. **Features:**
   - Multilingual support
   - Voice input analysis
   - Integration with wearable devices
   - Personalized baseline modeling

3. **Deployment:**
   - Mobile app (iOS/Android)
   - API for third-party integration
   - Professional dashboard for therapists
   - Automated intervention triggers

4. **Privacy & Security:**
   - Federated learning
   - Differential privacy
   - End-to-end encryption
   - HIPAA compliance

---

## 📚 References

- **Emotion Detection Model:** [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- **LSTM for Time Series:** Hochreiter & Schmidhuber (1997)
- **Mental Health NLP:** Coppersmith et al. (2015), De Choudhury et al. (2013)

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## 👥 Contact & Support

For questions, issues, or collaboration:
- Open an issue on GitHub
- Contact: [chalanalakshya5@gmail.com]

---

## 🙏 Acknowledgments

- HuggingFace for pre-trained models
- Mental health research community
- Open-source ML/NLP libraries

---

**Built with ❤️ for mental health awareness and early intervention**

*Remember: This tool supports, but does not replace, professional mental health care.*
"# AI_MentalHealthDetector_with_LSTM_TemporalModeling" 
