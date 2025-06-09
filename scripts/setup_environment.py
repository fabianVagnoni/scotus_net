#!/usr/bin/env python3
"""Setup script for SCOTUS AI project environment."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def setup_environment():
    """Set up the Python environment for the SCOTUS AI project."""
    print("🚀 Setting up SCOTUS AI Project Environment")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    # Create virtual environment
    if not run_command("python -m venv venv", "Creating virtual environment"):
        return False
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
        pip_command = "venv\\Scripts\\pip"
        python_command = "venv\\Scripts\\python"
    else:  # Unix/Linux/Mac
        activate_script = "source venv/bin/activate"
        pip_command = "venv/bin/pip"
        python_command = "venv/bin/python"
    
    print(f"Virtual environment created! Activate with: {activate_script}")
    
    # Upgrade pip
    if not run_command(f"{pip_command} install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{pip_command} install -r requirements.txt", "Installing requirements"):
        return False
    
    # Install spaCy English model
    if not run_command(f"{python_command} -m spacy download en_core_web_sm", "Installing spaCy English model"):
        print("⚠️  Warning: Could not install spaCy model. You may need to install it manually.")
    
    # Install project in development mode
    if not run_command(f"{pip_command} install -e .", "Installing project in development mode"):
        return False
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "notebooks",
        ".cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Create .gitignore if it doesn't exist
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data
data/raw/
data/processed/
*.csv
*.json
*.parquet

# Models
*.pkl

# Environment
.env
.env.local
.env.*.local

# Cache
.cache/
*.cache

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
"""
    
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        with open(gitignore_path, "w") as f:
            f.write(gitignore_content)
        print("📝 Created .gitignore file")
    
    print("\n🎉 Environment setup completed successfully!")
    print("\nNext steps:")
    print(f"1. Activate the virtual environment: {activate_script}")
    print("2. Copy env.example to .env and configure your settings")
    print("3. Run the data scraper: python src/data_pipeline/scraper.py")
    print("4. Process the data: python src/data_pipeline/processor.py")
    print("5. Train the model: python src/models/model_trainer.py")
    
    return True


if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1) 