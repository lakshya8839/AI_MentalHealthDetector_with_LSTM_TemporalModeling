#!/usr/bin/env python3
"""
Visualize the system architecture and data flow.
Creates diagrams to understand the pipeline.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram():
    """Create a visual representation of the system architecture."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'AI Mental Health Detector Architecture', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Colors
    input_color = '#e3f2fd'
    process_color = '#fff3e0'
    model_color = '#f3e5f5'
    output_color = '#e8f5e9'
    
    # Layer 1: Input
    input_box = FancyBboxPatch((2, 10), 6, 0.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', 
                               facecolor=input_color, 
                               linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 10.4, 'User Text Input', fontsize=12, ha='center', fontweight='bold')
    
    # Arrow
    arrow1 = FancyArrowPatch((5, 10), (5, 9.2), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # Layer 2: Preprocessing
    preprocess_box = FancyBboxPatch((1.5, 8), 7, 1, 
                                   boxstyle="round,pad=0.1", 
                                   edgecolor='black', 
                                   facecolor=process_color, 
                                   linewidth=2)
    ax.add_patch(preprocess_box)
    ax.text(5, 8.7, 'Text Preprocessing', fontsize=11, ha='center', fontweight='bold')
    ax.text(5, 8.3, 'Clean • Tokenize • Feature Extraction', fontsize=9, ha='center')
    
    # Arrow
    arrow2 = FancyArrowPatch((5, 8), (5, 7.2), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # Layer 3: NLP Model
    nlp_box = FancyBboxPatch((1, 5.5), 8, 1.5, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', 
                            facecolor=model_color, 
                            linewidth=2)
    ax.add_patch(nlp_box)
    ax.text(5, 6.7, 'Emotion Detection (DistilRoBERTa)', fontsize=11, ha='center', fontweight='bold')
    ax.text(5, 6.2, '7 Emotions: anger • disgust • fear • joy • neutral • sadness • surprise', 
            fontsize=8, ha='center')
    ax.text(5, 5.8, 'Output: Probabilities + Confidence Score', fontsize=8, ha='center', style='italic')
    
    # Split arrows
    arrow3a = FancyArrowPatch((4, 5.5), (2.5, 4.7), 
                             arrowstyle='->', mutation_scale=20, 
                             linewidth=1.5, color='black')
    ax.add_patch(arrow3a)
    
    arrow3b = FancyArrowPatch((6, 5.5), (7.5, 4.7), 
                             arrowstyle='->', mutation_scale=20, 
                             linewidth=1.5, color='black')
    ax.add_patch(arrow3b)
    
    # Layer 4a: Rule-based Scoring
    rule_box = FancyBboxPatch((0.5, 3.5), 3.5, 1, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='black', 
                             facecolor=process_color, 
                             linewidth=2)
    ax.add_patch(rule_box)
    ax.text(2.25, 4.2, 'Rule-Based Scoring', fontsize=10, ha='center', fontweight='bold')
    ax.text(2.25, 3.8, 'Immediate Assessment', fontsize=8, ha='center')
    
    # Layer 4b: Feature Vector Creation
    feature_box = FancyBboxPatch((6, 3.5), 3.5, 1, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='black', 
                                facecolor=process_color, 
                                linewidth=2)
    ax.add_patch(feature_box)
    ax.text(7.75, 4.2, 'Feature Engineering', fontsize=10, ha='center', fontweight='bold')
    ax.text(7.75, 3.8, '20-dimensional vector', fontsize=8, ha='center')
    
    # Arrow from feature to sequence
    arrow4 = FancyArrowPatch((7.75, 3.5), (7.75, 2.7), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=1.5, color='black')
    ax.add_patch(arrow4)
    
    # Layer 5: Sequence Generation
    seq_box = FancyBboxPatch((6, 1.5), 3.5, 1, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', 
                            facecolor=process_color, 
                            linewidth=2)
    ax.add_patch(seq_box)
    ax.text(7.75, 2.2, 'Sequence Generation', fontsize=10, ha='center', fontweight='bold')
    ax.text(7.75, 1.8, 'Last 5 time steps', fontsize=8, ha='center')
    
    # Arrow to LSTM
    arrow5 = FancyArrowPatch((7.75, 1.5), (7.75, 0.7), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow5)
    
    # Layer 6: LSTM Model
    lstm_box = FancyBboxPatch((6, -0.5), 3.5, 1, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='black', 
                             facecolor=model_color, 
                             linewidth=2)
    ax.add_patch(lstm_box)
    ax.text(7.75, 0.2, 'LSTM Temporal Model', fontsize=10, ha='center', fontweight='bold')
    ax.text(7.75, -0.2, 'Pattern Recognition', fontsize=8, ha='center')
    
    # Merge arrows
    arrow6a = FancyArrowPatch((2.25, 3.5), (3.5, 2.5), 
                             arrowstyle='->', mutation_scale=20, 
                             linewidth=1.5, color='black')
    ax.add_patch(arrow6a)
    
    arrow6b = FancyArrowPatch((7.75, -0.5), (6.5, -1.5), 
                             arrowstyle='->', mutation_scale=20, 
                             linewidth=1.5, color='black')
    ax.add_patch(arrow6b)
    
    # Final Output
    output_box = FancyBboxPatch((2, -2.5), 6, 1.2, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', 
                               facecolor=output_color, 
                               linewidth=3)
    ax.add_patch(output_box)
    ax.text(5, -1.6, 'Risk Assessment Output', fontsize=12, ha='center', fontweight='bold')
    ax.text(5, -2.0, 'Mental Health Score • Risk Level • Alerts', fontsize=9, ha='center')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=input_color, edgecolor='black', label='Input Layer'),
        mpatches.Patch(facecolor=process_color, edgecolor='black', label='Processing Layer'),
        mpatches.Patch(facecolor=model_color, edgecolor='black', label='ML Models'),
        mpatches.Patch(facecolor=output_color, edgecolor='black', label='Output')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Add side notes
    ax.text(0.2, 6.5, 'NLP\nLayer', fontsize=9, ha='center', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(9.8, 2, 'LSTM\nLayer', fontsize=9, ha='center', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight')
    print("✅ Architecture diagram saved as 'architecture_diagram.png'")
    plt.show()


def create_data_flow_diagram():
    """Create a diagram showing how data flows through the system."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 9.5, 'Data Flow: Single Text Entry', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Timeline
    stages = [
        ("Input", 8.5, "I feel really anxious"),
        ("Clean", 7.5, "i feel really anxious"),
        ("Features", 6.5, "15+ numerical features"),
        ("Emotion", 5.5, "fear: 0.65, sadness: 0.25"),
        ("Score", 4.5, "Mental Health: 0.42"),
        ("Vector", 3.5, "[0.42, 0.08, 0.12, ...]"),
        ("Sequence", 2.5, "Last 5 vectors"),
        ("LSTM", 1.5, "Risk: 0.67 (HIGH)"),
    ]
    
    for i, (stage, y, example) in enumerate(stages):
        # Box
        box = FancyBboxPatch((1, y-0.3), 8, 0.5, 
                            boxstyle="round,pad=0.05", 
                            edgecolor='black', 
                            facecolor=plt.cm.Blues(0.3 + i*0.08), 
                            linewidth=1.5)
        ax.add_patch(box)
        
        # Text
        ax.text(1.5, y, stage, fontsize=11, fontweight='bold', va='center')
        ax.text(6, y, example, fontsize=9, va='center', style='italic')
        
        # Arrow
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((5, y-0.3), (5, stages[i+1][1]+0.2), 
                                   arrowstyle='->', mutation_scale=15, 
                                   linewidth=1.5, color='darkblue')
            ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig('data_flow_diagram.png', dpi=300, bbox_inches='tight')
    print("✅ Data flow diagram saved as 'data_flow_diagram.png'")
    plt.show()


def create_comparison_chart():
    """Create a visual comparison of Rule-based vs LSTM approaches."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Rule-based approach
    categories = ['Interpretability', 'Speed', 'Accuracy', 
                 'Pattern Detection', 'Data Requirement']
    rule_scores = [90, 95, 70, 30, 10]
    lstm_scores = [60, 70, 85, 95, 80]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.barh(x - width/2, rule_scores, width, label='Rule-Based', color='coral')
    ax1.barh(x + width/2, lstm_scores, width, label='LSTM', color='steelblue')
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(categories)
    ax1.set_xlabel('Score')
    ax1.set_title('Capability Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_xlim(0, 100)
    ax1.grid(axis='x', alpha=0.3)
    
    # Time series example
    time_steps = np.arange(10)
    declining_pattern = np.array([0.75, 0.70, 0.68, 0.62, 0.58, 0.52, 0.45, 0.40, 0.35, 0.30])
    
    ax2.plot(time_steps, declining_pattern, 'o-', linewidth=2, markersize=8, 
            label='Mental Health Score', color='darkblue')
    
    # Threshold lines
    ax2.axhline(y=0.7, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Healthy')
    ax2.axhline(y=0.4, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Moderate Risk')
    ax2.axhline(y=0.3, color='red', linestyle='--', linewidth=2, alpha=0.5, label='High Risk')
    
    # Highlight LSTM detection point
    ax2.axvspan(5, 10, alpha=0.2, color='red', label='LSTM Detects Decline')
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Mental Health Score')
    ax2.set_title('LSTM Temporal Pattern Detection', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('comparison_chart.png', dpi=300, bbox_inches='tight')
    print("✅ Comparison chart saved as 'comparison_chart.png'")
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("AI MENTAL HEALTH DETECTOR - ARCHITECTURE VISUALIZATION")
    print("=" * 70)
    print("\nGenerating diagrams...\n")
    
    try:
        create_architecture_diagram()
        create_data_flow_diagram()
        create_comparison_chart()
        
        print("\n" + "=" * 70)
        print("✅ All diagrams generated successfully!")
        print("=" * 70)
        print("\nFiles created:")
        print("  • architecture_diagram.png")
        print("  • data_flow_diagram.png")
        print("  • comparison_chart.png")
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error generating diagrams: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")
