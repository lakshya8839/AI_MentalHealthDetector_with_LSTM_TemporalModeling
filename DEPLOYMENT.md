# Deployment Guide

## Production Deployment Checklist

### 1. Infrastructure Setup

#### Cloud Deployment (AWS/GCP/Azure)

**Recommended Specifications:**
- **CPU:** 4+ vCPUs
- **RAM:** 16GB minimum
- **Storage:** 20GB SSD
- **GPU:** Optional (T4 or better for faster inference)

**Container Setup (Docker):**

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download models (cache in container)
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('j-hartmann/emotion-english-distilroberta-base')"

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and run:**
```bash
docker build -t mental-health-detector .
docker run -p 8501:8501 mental-health-detector
```

### 2. Database Integration

For production, replace in-memory storage with a database:

**PostgreSQL Schema:**
```sql
CREATE TABLE user_sessions (
    session_id UUID PRIMARY KEY,
    user_id UUID,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE entries (
    entry_id UUID PRIMARY KEY,
    session_id UUID REFERENCES user_sessions(session_id),
    timestamp TIMESTAMP DEFAULT NOW(),
    text_encrypted TEXT,  -- Encrypted text
    emotion VARCHAR(20),
    emotion_confidence FLOAT,
    mental_health_score FLOAT,
    feature_vector JSONB
);

CREATE TABLE lstm_predictions (
    prediction_id UUID PRIMARY KEY,
    session_id UUID REFERENCES user_sessions(session_id),
    timestamp TIMESTAMP DEFAULT NOW(),
    risk_score FLOAT,
    risk_level VARCHAR(20),
    sequence_length INT
);

CREATE TABLE alerts (
    alert_id UUID PRIMARY KEY,
    session_id UUID REFERENCES user_sessions(session_id),
    timestamp TIMESTAMP DEFAULT NOW(),
    risk_level VARCHAR(20),
    alert_sent BOOLEAN DEFAULT FALSE,
    reviewed BOOLEAN DEFAULT FALSE
);
```

### 3. Security & Privacy

#### Data Encryption

```python
from cryptography.fernet import Fernet
import os

# Generate key (store securely in environment variable)
encryption_key = os.environ.get('ENCRYPTION_KEY')
cipher = Fernet(encryption_key)

def encrypt_text(text: str) -> bytes:
    return cipher.encrypt(text.encode())

def decrypt_text(encrypted: bytes) -> str:
    return cipher.decrypt(encrypted).decode()
```

#### Environment Variables

```bash
# .env file (DO NOT commit to git)
ENCRYPTION_KEY=your-encryption-key-here
DATABASE_URL=postgresql://user:pass@host:5432/dbname
SECRET_KEY=your-secret-key-here
HUGGINGFACE_TOKEN=your-hf-token-here  # If using gated models
ALERT_EMAIL_PASSWORD=your-email-password
```

### 4. API Deployment

**FastAPI Wrapper:**

```python
# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipeline import MentalHealthPipeline

app = FastAPI(title="Mental Health API")
pipeline = MentalHealthPipeline()

class TextRequest(BaseModel):
    text: str
    session_id: str

class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    mental_health_score: float
    risk_level: str = None
    risk_score: float = None

@app.post("/analyze", response_model=PredictionResponse)
async def analyze_text(request: TextRequest):
    try:
        result = pipeline.process_text(request.text)
        
        response = {
            "emotion": result["emotion"],
            "confidence": result["emotion_confidence"],
            "mental_health_score": result["mental_health_score"]
        }
        
        # LSTM prediction if available
        lstm_result = pipeline.lstm_predict()
        if lstm_result["has_prediction"]:
            response["risk_level"] = lstm_result["risk_level"]
            response["risk_score"] = lstm_result["risk_score"]
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**Run with:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 5. Monitoring & Logging

**Logging Setup:**

```python
import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

# Log all predictions
logger.info(
    "prediction_made",
    session_id=session_id,
    emotion=emotion,
    risk_level=risk_level,
    timestamp=datetime.now().isoformat()
)
```

**Metrics (Prometheus):**

```python
from prometheus_client import Counter, Histogram, Gauge

prediction_counter = Counter('predictions_total', 'Total predictions made')
risk_gauge = Gauge('current_risk_score', 'Current risk score')
inference_time = Histogram('inference_duration_seconds', 'Inference time')

# Track metrics
prediction_counter.inc()
risk_gauge.set(risk_score)
```

### 6. Load Balancing

**Nginx Configuration:**

```nginx
upstream mental_health_app {
    server localhost:8501;
    server localhost:8502;
    server localhost:8503;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://mental_health_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 7. Model Management

**Model Versioning:**

```python
import mlflow

# Track experiments
mlflow.set_experiment("mental_health_lstm")

with mlflow.start_run():
    # Train model
    history = lstm_model.train(X_train, y_train, epochs=50)
    
    # Log parameters
    mlflow.log_params({
        "sequence_length": 5,
        "lstm_units": 64,
        "epochs": 50
    })
    
    # Log metrics
    mlflow.log_metrics({
        "accuracy": history.history["val_accuracy"][-1],
        "auc": history.history["val_auc"][-1]
    })
    
    # Save model
    mlflow.tensorflow.log_model(lstm_model.model, "lstm_model")
```

### 8. Testing

**Unit Tests:**

```python
# test_pipeline.py
import pytest
from src.pipeline import MentalHealthPipeline

@pytest.fixture
def pipeline():
    return MentalHealthPipeline()

def test_emotion_detection(pipeline):
    result = pipeline.process_text("I'm feeling great today!")
    assert result["emotion"] in ["joy", "neutral", "surprise"]
    assert 0 <= result["mental_health_score"] <= 1

def test_temporal_prediction(pipeline):
    # Add 5 entries
    texts = ["Great day!"] * 5
    for text in texts:
        pipeline.process_text(text)
    
    # Should have LSTM prediction
    lstm_result = pipeline.lstm_predict()
    assert lstm_result["has_prediction"] == True
```

**Run tests:**
```bash
pytest test_pipeline.py -v
```

### 9. Compliance (HIPAA for US)

**Requirements:**
- Encrypted data storage (AES-256)
- Encrypted data transmission (TLS 1.2+)
- Access logging and audit trails
- User consent and data agreements
- Data retention and deletion policies
- Business Associate Agreements (BAA)

### 10. Scaling Strategies

**Horizontal Scaling:**
- Use Kubernetes for auto-scaling
- Implement Redis for session management
- Use message queues (RabbitMQ/Kafka) for async processing

**Model Optimization:**
- Quantization (FP16 or INT8)
- Model distillation
- ONNX conversion for faster inference
- Batch processing for multiple users

### 11. Disaster Recovery

**Backup Strategy:**
```bash
# Daily database backups
0 2 * * * pg_dump dbname > backup_$(date +\%Y\%m\%d).sql

# Weekly model backups
0 3 * * 0 aws s3 sync ./models s3://backup-bucket/models
```

**High Availability:**
- Multi-region deployment
- Database replication
- Model version rollback capability

---

## Quick Production Deploy (AWS ECS)

```bash
# 1. Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin
docker build -t mental-health-detector .
docker tag mental-health-detector:latest <account-id>.dkr.ecr.region.amazonaws.com/mental-health-detector
docker push <account-id>.dkr.ecr.region.amazonaws.com/mental-health-detector

# 2. Create ECS task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# 3. Create ECS service
aws ecs create-service --cluster production --service-name mental-health-api --task-definition mental-health-task

# 4. Configure load balancer
aws elbv2 create-load-balancer --name mental-health-lb --subnets subnet-xxx subnet-yyy
```

---

## Monitoring Dashboard

Use Grafana + Prometheus for monitoring:

**Key Metrics:**
- Requests per second
- Average inference time
- Error rate
- Risk score distribution
- LSTM prediction accuracy
- System resource usage

---

## Maintenance Schedule

- **Daily:** Check logs for errors, monitor resource usage
- **Weekly:** Review alert queue, analyze prediction patterns
- **Monthly:** Retrain LSTM model with new data, update dependencies
- **Quarterly:** Security audit, performance optimization, model evaluation

---

## Support & Escalation

**Critical Issues:**
- System downtime
- Data breach
- High error rates

**Contact:**
- On-call engineer: [phone/email]
- Security team: [contact]
- Database admin: [contact]

---

**Remember:** Mental health applications require extra care in deployment. Always prioritize user privacy, data security, and ethical considerations.
