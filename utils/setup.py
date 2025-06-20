#!/usr/bin/env python3
"""
Setup script for Lex Fridman Chatbot Project
This script helps install dependencies and run initial setup.
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a shell command and handle errors."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {command}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running: {command}")
        print(f"Error: {e.stderr}")
        return None

def check_gpu():
    """Check if CUDA GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU detected: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️  No GPU detected. Model training will be slower on CPU.")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet. GPU check will be done after installation.")
        return False

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    
    # Install basic requirements first
    basic_packages = [
        "youtube-transcript-api",
        "pytube", 
        "google-api-python-client",
        "pandas",
        "requests",
        "tqdm"
    ]
    
    for package in basic_packages:
        run_command(f"pip install {package}")
    
    # Install PyTorch with CUDA support if GPU is available
    print("\n🔧 Installing PyTorch...")
    # Install PyTorch with CUDA 12.1 support (adjust based on your CUDA version)
    torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    run_command(torch_command)
    
    # Install remaining packages
    remaining_packages = [
        "transformers",
        "tokenizers", 
        "datasets",
        "accelerate",
        "bitsandbytes",
        "peft",
        "streamlit",
        "fastapi",
        "uvicorn",
        "numpy",
        "matplotlib",
        "seaborn"
    ]
    
    for package in remaining_packages:
        run_command(f"pip install {package}")

def create_project_structure():
    """Create necessary directories for the project."""
    print("📁 Creating project structure...")
    
    directories = [
        "data",
        "data/transcripts", 
        "data/metadata",
        "models",
        "notebooks",
        "src",
        "web_app"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}/")

def main():
    """Main setup function."""
    print("🚀 Setting up Lex Fridman Chatbot Project")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Create project structure
    create_project_structure()
    
    # Install requirements
    install_requirements()
    
    # Check GPU after installation
    print("\n🔍 Checking GPU availability...")
    has_gpu = check_gpu()
    
    print("\n" + "=" * 50)
    print("🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Run: python data_collection.py")
    print("2. This will collect transcripts from Lex Fridman's channel")
    print("3. The data will be saved in the 'data/' directory")
    
    if has_gpu:
        print("4. 🚀 GPU detected - ready for model fine-tuning!")
    else:
        print("4. ⚠️  Consider using a GPU for faster model training")
    
    print("\nTo start data collection now, type: python data_collection.py")

if __name__ == "__main__":
    main() 