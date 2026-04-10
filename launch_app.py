"""
Streamlit App Launcher with Path Fix
This script ensures the src module can be found before launching the app.
"""

import sys
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add it to Python path if not already there
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Verify src module exists
src_path = os.path.join(SCRIPT_DIR, 'src')
if not os.path.exists(src_path):
    print("ERROR: 'src' folder not found!")
    print(f"Expected location: {src_path}")
    print(f"\nCurrent directory contents:")
    for item in os.listdir(SCRIPT_DIR):
        print(f"  - {item}")
    sys.exit(1)

# Verify src/__init__.py exists
init_path = os.path.join(src_path, '__init__.py')
if not os.path.exists(init_path):
    print("ERROR: 'src/__init__.py' not found!")
    print("Creating it now...")
    with open(init_path, 'w') as f:
        f.write('"""AI Mental Health Detector - Source Package"""\n')
    print("✓ Created src/__init__.py")

# Test import
print("Testing imports...")
try:
    from src.preprocessing import TextPreprocessor
    from src.model import EmotionDetector
    from src.lstm_model import TemporalMentalHealthModel
    from src.pipeline import MentalHealthPipeline
    print("✓ All modules imported successfully!\n")
except ImportError as e:
    print(f"ERROR: Failed to import modules: {e}")
    print(f"\nPython path:")
    for p in sys.path:
        print(f"  - {p}")
    sys.exit(1)

# Now launch Streamlit
print("Launching Streamlit app...\n")
import subprocess
result = subprocess.run([
    sys.executable, 
    '-m', 
    'streamlit', 
    'run', 
    os.path.join(SCRIPT_DIR, 'app.py')
])

sys.exit(result.returncode)
