#!/bin/bash

echo "🚀 Launching Lex Fridman AI Chatbot Web Interface"
echo "=============================================="

# Check if the model exists
if [ ! -d "models/lex_chatbot" ]; then
    echo "❌ Model not found! Please train the model first:"
    echo "   python scripts/model_fine_tuner.py"
    echo ""
    echo "Or wait for the current training to complete..."
    exit 1
fi

echo "✅ Model found!"
echo "🌐 Starting web interface..."
echo ""
echo "📱 Open your browser and go to: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the application"
echo ""

# Launch Streamlit app
cd web_app && streamlit run lex_chatbot_app.py --server.port 8501 --server.address 0.0.0.0 