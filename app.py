"""
Streamlit application for AI Silent Mental Health Detector.
Interactive UI for real-time mental health monitoring with temporal analysis.
STANDALONE VERSION - No complex imports needed
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Now import modules directly
import preprocessing
import model
import lstm_model
import pipeline

# Page configuration
st.set_page_config(
    page_title="AI Mental Health Detector",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .risk-moderate {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load and cache the pipeline."""
    with st.spinner("Loading AI models... This may take a minute."):
        mental_pipeline = pipeline.MentalHealthPipeline()
        # Train LSTM if not already trained
        if not mental_pipeline.lstm_model.is_trained:
            with st.spinner("Training temporal model..."):
                mental_pipeline.train_lstm_model(n_sequences=800, epochs=30)
    return mental_pipeline


def plot_emotion_distribution(emotion_scores):
    """Create bar chart for emotion distribution."""
    emotions = list(emotion_scores.keys())
    scores = list(emotion_scores.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=scores,
            marker_color=['#ff6b6b', '#ee5a6f', '#c44569', '#f7b731', 
                         '#95afc0', '#4b7bec', '#26de81']
        )
    ])
    
    fig.update_layout(
        title="Emotion Distribution",
        xaxis_title="Emotion",
        yaxis_title="Probability",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def plot_mental_health_trend(df):
    """Create line chart for mental health score trend."""
    if df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df['mental_health_score'],
        mode='lines+markers',
        name='Mental Health Score',
        line=dict(color='#4b7bec', width=3),
        marker=dict(size=8)
    ))
    
    # Add threshold lines
    fig.add_hline(y=0.7, line_dash="dash", line_color="green", 
                  annotation_text="Healthy Threshold")
    fig.add_hline(y=0.4, line_dash="dash", line_color="orange", 
                  annotation_text="Moderate Risk")
    fig.add_hline(y=0.3, line_dash="dash", line_color="red", 
                  annotation_text="High Risk")
    
    fig.update_layout(
        title="Mental Health Score Over Time",
        xaxis_title="Entry Number",
        yaxis_title="Mental Health Score (0-1)",
        height=400,
        yaxis_range=[0, 1],
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def display_risk_alert(risk_level, risk_score):
    """Display risk alert with appropriate styling."""
    if risk_level == "low":
        st.markdown(f"""
        <div class="risk-low">
            ✅ LOW RISK - Mental health indicators appear stable<br>
            Risk Score: {risk_score:.3f}
        </div>
        """, unsafe_allow_html=True)
    elif risk_level == "moderate":
        st.markdown(f"""
        <div class="risk-moderate">
            ⚠️ MODERATE RISK - Some concerning patterns detected<br>
            Risk Score: {risk_score:.3f}<br>
            Consider reaching out to a mental health professional
        </div>
        """, unsafe_allow_html=True)
    else:  # high
        st.markdown(f"""
        <div class="risk-high">
            🚨 HIGH RISK - Significant decline patterns detected<br>
            Risk Score: {risk_score:.3f}<br>
            <strong>Please seek immediate support from a mental health professional</strong>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">🧠 AI Silent Mental Health Detector</div>', 
                unsafe_allow_html=True)
    st.markdown("*Using NLP + LSTM Temporal Modeling for Early Detection*")
    
    # Initialize pipeline
    mental_pipeline = load_pipeline()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This system uses advanced AI to detect emotional decline patterns:
        
        **Technology Stack:**
        - 🤖 Transformer NLP (DistilRoBERTa)
        - 🔄 LSTM Temporal Modeling
        - 📊 Multi-feature Analysis
        
        **How it works:**
        1. Enter your thoughts/feelings
        2. AI analyzes emotional content
        3. System tracks patterns over time
        4. LSTM predicts risk based on sequences
        """)
        
        st.divider()
        
        st.header("📊 Session Stats")
        st.metric("Total Entries", len(mental_pipeline.history))
        
        if len(mental_pipeline.history) >= mental_pipeline.sequence_length:
            st.success(f"✅ Temporal analysis active ({mental_pipeline.sequence_length}+ entries)")
        else:
            remaining = mental_pipeline.sequence_length - len(mental_pipeline.history)
            st.info(f"⏳ {remaining} more entries needed for temporal analysis")
        
        st.divider()
        
        if st.button("🗑️ Clear Session History"):
            mental_pipeline.clear_history()
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💭 Share Your Thoughts")
        
        # Text input
        user_text = st.text_area(
            "How are you feeling? What's on your mind?",
            height=150,
            placeholder="Express yourself freely... The AI will analyze emotional patterns and detect concerning trends."
        )
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        with col_btn2:
            if st.button("💡 Try Example", use_container_width=True):
                example_texts = [
                    "I've been feeling really overwhelmed lately. Everything seems harder than it should be.",
                    "Had a great day today! Accomplished a lot and feeling positive about the future.",
                    "I just don't see the point anymore. Every day feels the same and empty.",
                    "Feeling anxious about work, but I know I can handle it. Just need to take it one step at a time.",
                ]
                st.session_state['example_text'] = np.random.choice(example_texts)
                st.rerun()
        
        # Use example text if set
        if 'example_text' in st.session_state:
            user_text = st.session_state['example_text']
            del st.session_state['example_text']
        
        # Analyze button logic
        if analyze_button and user_text.strip():
            with st.spinner("Analyzing..."):
                # Process text
                result = mental_pipeline.process_text(user_text)
                
                # Display results
                st.success("✅ Analysis Complete")
                
                # Metrics row
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(
                        "Primary Emotion",
                        result['emotion'].upper(),
                        f"{result['emotion_confidence']:.1%} confidence"
                    )
                
                with metric_col2:
                    score_val = result['mental_health_score']
                    score_color = "🟢" if score_val > 0.6 else "🟡" if score_val > 0.3 else "🔴"
                    st.metric(
                        "Mental Health Score",
                        f"{score_color} {score_val:.3f}",
                        None
                    )
                
                with metric_col3:
                    neg_ratio = result['features']['negative_word_ratio']
                    st.metric(
                        "Negative Content",
                        f"{neg_ratio:.1%}",
                        None
                    )
                
                # Emotion distribution
                st.plotly_chart(
                    plot_emotion_distribution(result['emotion_scores']),
                    use_container_width=True
                )
                
                # LSTM Prediction
                st.header("🔮 Temporal Risk Assessment")
                
                lstm_result = mental_pipeline.lstm_predict()
                
                if lstm_result['has_prediction']:
                    display_risk_alert(
                        lstm_result['risk_level'],
                        lstm_result['risk_score']
                    )
                    st.info(f"ℹ️ {lstm_result['message']}")
                else:
                    st.info(f"ℹ️ {lstm_result['message']}")
                
        elif analyze_button:
            st.warning("⚠️ Please enter some text to analyze")
    
    with col2:
        st.header("📈 Historical Trends")
        
        if len(mental_pipeline.history) > 0:
            # Recent scores
            recent_scores = mental_pipeline.get_recent_scores(5)
            
            st.subheader("Last 5 Scores")
            for i, score in enumerate(reversed(recent_scores), 1):
                color = "🟢" if score > 0.6 else "🟡" if score > 0.3 else "🔴"
                st.write(f"{color} Entry {len(recent_scores) - i + 1}: {score:.3f}")
            
            st.divider()
            
            # Download history
            df_history = mental_pipeline.get_trend_data()
            if not df_history.empty:
                csv = df_history.to_csv(index=False)
                st.download_button(
                    "📥 Download History (CSV)",
                    csv,
                    "mental_health_history.csv",
                    "text/csv",
                    use_container_width=True
                )
        else:
            st.info("No analysis history yet. Start by sharing your thoughts!")
    
    # Full-width trend chart
    if len(mental_pipeline.history) > 1:
        st.header("📊 Mental Health Score Timeline")
        df_trend = mental_pipeline.get_trend_data()
        fig_trend = plot_mental_health_trend(df_trend)
        if fig_trend:
            st.plotly_chart(fig_trend, use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <strong>⚠️ Disclaimer:</strong> This is an AI-based screening tool, not a substitute for professional mental health care.
        If you're experiencing a crisis, please contact a mental health professional or crisis hotline immediately.
        <br><br>
        <strong>Crisis Resources:</strong> National Suicide Prevention Lifeline: 988 | Crisis Text Line: Text HOME to 741741
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
