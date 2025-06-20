#!/bin/bash

# 🤖 Lex Fridman AI Chatbot Demo Launcher
# Quick demo script for the trained chatbot

echo "🚀 Lex Fridman AI Chatbot Demo"
echo "=============================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+ to continue."
    exit 1
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import torch, transformers, streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Required packages not found."
    echo "💡 Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies. Please run: pip install -r requirements.txt"
        exit 1
    fi
fi

# Check if model exists
if [ ! -d "models/lex_chatbot_simple" ]; then
    echo "❌ Trained model not found at models/lex_chatbot_simple/"
    echo "💡 Please ensure the model is properly downloaded or trained."
    exit 1
fi

echo "✅ All dependencies satisfied!"
echo ""

# Show demo options
echo "🎯 Choose demo mode:"
echo "1. 🌐 Web Interface (Recommended)"
echo "2. 💻 Command Line Interface"
echo "3. ❌ Exit"
echo ""

read -p "👉 Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🌐 Launching Streamlit Web Interface..."
        echo "📱 Access at: http://localhost:8501"
        echo "⏹️  Press Ctrl+C to stop"
        echo ""
        streamlit run web_app/lex_chatbot_app.py --server.port 8501 --server.address 0.0.0.0
        ;;
    2)
        echo ""
        echo "💻 Launching Command Line Interface..."
        echo ""
        python test_model.py
        ;;
    3)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac 