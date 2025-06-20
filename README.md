# 🤖 Lex Fridman AI Chatbot

**An AI chatbot trained on Lex Fridman podcast transcripts using fine-tuned DialoGPT-medium**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.0%2B-orange" alt="Transformers">
  <img src="https://img.shields.io/badge/Streamlit-1.0%2B-green" alt="Streamlit">
</p>

---

## 🎯 **Project Overview**

This project creates an AI chatbot that mimics Lex Fridman's conversational style by fine-tuning Microsoft's DialoGPT-medium model on transcripts from 6 carefully selected Lex Fridman podcast episodes covering diverse topics.

### 📊 **Dataset Statistics**
- **6 Podcast Episodes** with **8,815 conversations**
- **645,849+ words** across diverse topics
- **3.2+ million characters** of training data

### 🎙️ **Source Episodes**
| Episode | Guest | Topic | Conversations |
|---------|-------|-------|---------------|
| #467 | Tim Sweeney | Gaming, Unreal Engine | 1,190 |
| #452 | Dario Amodei | AI, AGI, Anthropic | 1,385 |
| #433 | Sara Walker | Physics, Complexity | 1,140 |
| #418 | Israel-Palestine | Geopolitics, History | 2,521 |
| #459 | DeepSeek Discussion | AI, China, Tech | 1,294 |
| #383 | Mark Zuckerberg | Social Media, Meta | 1,285 |

---

## 🚀 **Quick Demo**

### **Option 1: Web Interface** (Recommended)
```bash
# Clone the repository
git clone <your-repo-url>
cd Intelligent-Fridman

# Install dependencies
pip install -r requirements.txt

# Launch web interface
streamlit run web_app/lex_chatbot_app.py --server.port 8501 --server.address 0.0.0.0
```

**Access at:** `http://localhost:8501`

### **Option 2: Command Line Interface**
```bash
python test_model.py
```

---

## 🏗️ **Project Structure**

```
Intelligent-Fridman/
├── 📁 data/                    # Training data
│   ├── transcripts/           # Original JSON transcripts
│   └── metadata/              # Video metadata
├── 📁 trascripts/             # Text format transcripts 
├── 📁 processed_data/         # Processed training datasets
│   ├── unified_dataset.json   # Final training data
│   ├── training_data.json     # Structured conversations
│   └── dataset_analysis.json  # Dataset statistics
├── 📁 models/                 # Trained models
│   ├── lex_chatbot_simple/    # Working trained model ✅
│   ├── lex_chatbot/           # Enhanced model
│   └── lex_chatbot_enhanced/  # Advanced model
├── 📁 scripts/                # Training & utility scripts
│   ├── simple_trainer.py      # Simple training script ✅
│   ├── dataset_creator.py     # Dataset creation
│   ├── enhanced_trainer.py    # Advanced training
│   └── sample_viewer.py       # View dataset samples
├── 📁 web_app/                # Streamlit web interface
│   └── lex_chatbot_app.py     # Web app ✅
├── test_model.py              # CLI testing script ✅
├── upload_to_huggingface.py   # HF upload script
└── requirements.txt           # Dependencies
```

---

## 🔧 **Installation & Setup**

### **1. Prerequisites**
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory for training

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Download Pre-trained Model** (Optional)
```bash
# The repository includes a trained model in models/lex_chatbot_simple/
# No additional downloads needed for demo!
```

---

## 🎮 **Usage Examples**

### **Web Interface Features**
- 💬 **Interactive Chat** - Real-time conversations with AI Lex
- 📊 **Model Stats** - Performance metrics and info
- 🎨 **Beautiful UI** - Custom styled interface
- 📝 **Chat History** - Conversation tracking
- ⚙️ **Settings** - Adjust generation parameters

### **Sample Conversations**
```
👤 Human: What do you think about artificial intelligence?

🤖 Lex: It's just another human made language that doesn't understand 
what I'm saying. And if you ask humans to use it they have a hard time 
understanding the syntax of what you're saying...

👤 Human: What is consciousness?

🤖 Lex: It's the idea of all living things in this world and all living 
beings have some form of consciousness and all the knowledge that they 
have comes from all of our experiences...
```

---

## 🏋️ **Training from Scratch**

### **1. Create Dataset**
```bash
python scripts/dataset_creator.py
```

### **2. Train Model**
```bash
# Simple training (recommended)
python scripts/simple_trainer.py

# Enhanced training (requires more memory)
python scripts/enhanced_trainer.py
```

### **3. Monitor Training**
```bash
# View training logs
tensorboard --logdir=models/lex_chatbot_simple/runs
```

---

## 🧪 **Testing & Evaluation**

### **Command Line Testing**
```bash
python test_model.py
# Choose from:
# 1. Automated test questions
# 2. Interactive chat mode  
# 3. Single question test
```

### **View Dataset Samples**
```bash
python scripts/sample_viewer.py
```

### **Performance Metrics**
- **Model Size:** 354M parameters (DialoGPT-medium)
- **Training Time:** ~2 minutes (2 epochs)
- **Generation Speed:** 0.3-1.8 seconds per response
- **Final Loss:** 3.54 (good convergence)
- **Topics Covered:** 9 diverse conversation areas

---

## 🚀 **Deployment Options**

### **Local Development**
```bash
streamlit run web_app/lex_chatbot_app.py
```

### **Production Deployment**
```bash
# With custom port and host
streamlit run web_app/lex_chatbot_app.py \
  --server.port 8080 \
  --server.address 0.0.0.0 \
  --server.headless true
```

### **Docker Deployment** (Future)
```bash
# TODO: Add Dockerfile for containerized deployment
docker build -t lex-chatbot .
docker run -p 8501:8501 lex-chatbot
```

---

## 📈 **Model Performance**

### **Strengths** ✅
- Fast response generation (< 2 seconds)
- Covers diverse topics from training data
- Lex-like conversational patterns
- Stable performance with good convergence
- Memory efficient (works on single GPU)

### **Areas for Improvement** 🔄
- Longer, more coherent responses
- Better context awareness
- Reduced repetition patterns
- Extended training on larger dataset

---

## 🔮 **Future Enhancements**

### **Short Term**
- [ ] Upload model to HuggingFace Hub
- [ ] Add more conversation examples
- [ ] Improve response quality metrics
- [ ] Add conversation export feature

### **Long Term**
- [ ] Train on additional podcast episodes
- [ ] Implement memory/context system
- [ ] Add personality fine-tuning
- [ ] Multi-turn conversation improvements
- [ ] Voice interface integration

---

## 🤝 **Contributing**

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### **Development Setup**
```bash
git clone <repo-url>
cd Intelligent-Fridman
pip install -r requirements.txt
pre-commit install  # If you have pre-commit
```

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Lex Fridman** for the incredible podcast content
- **Microsoft** for the DialoGPT model
- **HuggingFace** for the Transformers library
- **OpenAI** for inspiration and techniques

---

## 📞 **Support**

- 🐛 **Bug Reports:** Open an issue
- 💡 **Feature Requests:** Start a discussion  
- 📧 **Contact:** [Your contact info]

---

<p align="center">
  <strong>Built with ❤️ for the AI community</strong>
</p> 