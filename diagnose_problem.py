"""
Emergency Diagnostic Script
Run this to identify why src module can't be found
"""

import sys
import os

print("=" * 70)
print("EMERGENCY DIAGNOSTIC FOR 'src' MODULE NOT FOUND ERROR")
print("=" * 70)

print("\n[1] PYTHON INFORMATION:")
print(f"  Python version: {sys.version}")
print(f"  Python executable: {sys.executable}")

print("\n[2] DIRECTORY INFORMATION:")
print(f"  Current working directory: {os.getcwd()}")
print(f"  Script directory: {os.path.dirname(os.path.abspath(__file__))}")

print("\n[3] PYTHON PATH:")
for i, path in enumerate(sys.path, 1):
    print(f"  {i}. {path}")

print("\n[4] CURRENT DIRECTORY CONTENTS:")
cwd = os.getcwd()
items = os.listdir(cwd)
for item in sorted(items):
    item_path = os.path.join(cwd, item)
    if os.path.isdir(item_path):
        print(f"  📁 {item}/")
    else:
        print(f"  📄 {item}")

print("\n[5] CHECKING FOR 'src' FOLDER:")
src_path = os.path.join(cwd, 'src')
if os.path.exists(src_path):
    print(f"  ✓ Found: {src_path}")
    print("\n  Contents of 'src' folder:")
    for item in os.listdir(src_path):
        print(f"    - {item}")
else:
    print(f"  ✗ NOT FOUND: {src_path}")
    print("\n  🚨 PROBLEM IDENTIFIED: 'src' folder is missing!")
    print("\n  SOLUTION: Make sure you've extracted ALL files from the ZIP")
    print("           and you're in the correct directory.")

print("\n[6] CHECKING FOR REQUIRED FILES:")
required_files = [
    'app.py',
    'requirements.txt', 
    'src/__init__.py',
    'src/pipeline.py',
    'src/model.py',
    'src/lstm_model.py',
    'src/preprocessing.py'
]

missing = []
for file in required_files:
    file_path = os.path.join(cwd, file)
    if os.path.exists(file_path):
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ {file} - MISSING!")
        missing.append(file)

if missing:
    print(f"\n  🚨 PROBLEM: {len(missing)} file(s) missing!")
    print("  SOLUTION: Re-extract all files from the ZIP archive")
else:
    print("\n  ✓ All required files present")

print("\n[7] TESTING IMPORT:")
# Add current directory to path
sys.path.insert(0, cwd)

try:
    import src
    print(f"  ✓ Successfully imported 'src' module")
    print(f"    Location: {src.__file__}")
except ImportError as e:
    print(f"  ✗ Failed to import 'src': {e}")
    
try:
    from src import pipeline
    print(f"  ✓ Successfully imported 'src.pipeline'")
except ImportError as e:
    print(f"  ✗ Failed to import 'src.pipeline': {e}")

print("\n[8] RECOMMENDATION:")

if missing:
    print("  ⚠️  FILES ARE MISSING")
    print("  Action: Re-extract the complete project ZIP file")
    print(f"  Missing files: {', '.join(missing)}")
elif not os.path.exists(src_path):
    print("  ⚠️  'src' FOLDER NOT FOUND")
    print("  Action: Make sure you're in the project root directory")
    print(f"  Try: cd G:\\LSTM_ML_PROJECT")
else:
    print("  ℹ️  Files appear to be in place")
    print("  Action: Try running the app with:")
    print("         python launch_app.py")
    print("  Or:    python -c \"import sys; sys.path.insert(0, '.'); import streamlit.web.cli as stcli; sys.argv = ['streamlit', 'run', 'app.py']; sys.exit(stcli.main())\"")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

input("\nPress Enter to exit...")
