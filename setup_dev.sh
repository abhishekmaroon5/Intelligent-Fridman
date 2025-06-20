#!/bin/bash

echo "🚀 Setting up Intelligent Fridman Development Environment"
echo "======================================================"

# Check Python version
python_version=$(python3 --version 2>&1)
echo "📍 Python version: $python_version"

# Check if Python 3.11+ is available
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    echo "❌ Python 3.11+ is required!"
    echo "💡 Please install Python 3.11 or newer"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "🔨 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Install development dependencies
echo "🛠️  Installing development dependencies..."
pip install -e .
pip install pytest pytest-cov flake8 black isort pre-commit

# Check GPU availability
echo ""
echo "🔍 Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available with {torch.cuda.device_count()} GPU(s)')
    for i in range(torch.cuda.device_count()):
        print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('❌ CUDA not available - CPU training only')
"

# Create necessary directories
echo ""
echo "📁 Creating project directories..."
mkdir -p data/transcripts
mkdir -p data/txt_files
mkdir -p data/metadata
mkdir -p processed_data/tokenized_datasets
mkdir -p models/lex_chatbot
mkdir -p logs

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Collect data: python scripts/working_transcript_downloader.py"
echo "3. Process data: python scripts/tokenizer_and_preprocessor.py"
echo "4. Train model: python scripts/model_fine_tuner.py"
echo "5. Launch app: ./run_chatbot.sh"
echo ""
echo "📚 For more information, see README.md" 