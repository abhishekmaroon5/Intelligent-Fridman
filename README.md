# 🤖 Intelligent Fridman - Lex Fridman AI Chatbot

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An intelligent chatbot that learns from Lex Fridman's podcast transcripts and conversations, built using state-of-the-art transformer models and fine-tuning techniques.

## 🌟 Features

- **🎯 Lex Fridman-Style Conversations**: Chat with an AI trained on Lex Fridman's speaking patterns and knowledge
- **🚀 GPU-Accelerated Training**: Optimized for NVIDIA A100 GPUs with multi-GPU support
- **💻 Beautiful Web Interface**: Streamlit-powered UI with conversation history and customizable parameters
- **📊 Comprehensive Data Pipeline**: Automated transcript collection, preprocessing, and tokenization
- **🔧 Easy Setup**: One-command installation and deployment
- **📈 Training Monitoring**: Real-time training progress and GPU utilization tracking

## 🏗️ Project Structure

```
Intelligent-Fridman/
├── 📁 data/                          # Raw and processed transcript data
│   ├── transcripts/                  # Individual podcast transcripts (JSON)
│   ├── txt_files/                    # Human-readable transcript files
│   ├── metadata/                     # Video metadata and summaries
│   └── all_transcripts.json          # Combined transcript data
├── 📁 processed_data/                # Tokenized and training-ready datasets
│   ├── tokenized_datasets/           # HuggingFace datasets format
│   └── training_config.json          # Training configuration
├── 📁 models/                        # Trained model checkpoints
│   └── lex_chatbot/                  # Fine-tuned Lex Fridman model
├── 📁 web_app/                       # Streamlit web application
│   └── lex_chatbot_app.py           # Main chatbot interface
├── 📁 scripts/                       # Core processing scripts
│   ├── working_transcript_downloader.py  # YouTube transcript collection
│   ├── tokenizer_and_preprocessor.py     # Data preprocessing pipeline
│   └── model_fine_tuner.py              # GPU-optimized model training
├── 📁 docs/                          # Documentation and resources
│   ├── Lecture_notes/                # ML/AI educational materials
│   └── LLM Roadmap from Beginner to Advanced Level.pdf
├── 📁 archive/                       # Development history and old scripts
├── 📋 requirements.txt               # Python dependencies
├── 🚀 setup.py                      # Project installation
└── 🏃 run_chatbot.sh               # Quick deployment script
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/your-username/Intelligent-Fridman.git
cd Intelligent-Fridman
pip install -r requirements.txt
```

### 2. Run Pre-trained Model (Recommended)

```bash
# Launch the chatbot interface
./run_chatbot.sh
```

### 3. Train Your Own Model (Advanced)

```bash
# Step 1: Collect transcript data
python scripts/working_transcript_downloader.py

# Step 2: Process and tokenize data
python scripts/tokenizer_and_preprocessor.py

# Step 3: Fine-tune the model (requires GPU)
python scripts/model_fine_tuner.py

# Step 4: Launch your trained model
./run_chatbot.sh
```

## 💻 Web Interface

The Streamlit web application provides:

- **🎨 Beautiful, Modern UI**: Clean design with Lex Fridman branding
- **💬 Interactive Chat**: Real-time conversation with the AI
- **📊 Conversation History**: Track your discussion topics
- **⚙️ Customizable Parameters**: Adjust temperature, max tokens, and sampling
- **📈 System Monitoring**: GPU usage and model statistics
- **💡 Sample Questions**: Get started with conversation starters

## 🤖 Model Architecture

- **Base Model**: Microsoft DialoGPT-medium (354M parameters)
- **Fine-tuning**: Supervised fine-tuning on Lex Fridman transcripts
- **Training**: Multi-GPU distributed training with BFloat16 precision
- **Optimization**: Gradient checkpointing and mixed precision training
- **Dataset**: ~48K words from high-quality podcast transcripts

## 📊 Training Performance

| Metric | Value |
|--------|-------|
| **Training Examples** | 3,100 conversation pairs |
| **Validation Examples** | 345 conversation pairs |
| **Model Parameters** | 354,823,168 |
| **GPU Memory Usage** | ~6GB per A100 GPU |
| **Training Time** | ~2 hours on 2x A100 |
| **Final Loss** | < 2.0 (conversation quality) |

## 🛠️ Technical Details

### Data Collection
- **Source**: Lex Fridman Podcast YouTube channel
- **Method**: `yt-dlp` + `youtube-transcript-api`
- **Format**: Structured JSON with metadata
- **Quality**: High-quality, manually verified transcripts

### Data Processing
- **Tokenization**: DialoGPT tokenizer (50,257 vocab)
- **Format**: "Human: [question]\n\nLex: [response]" pairs
- **Context Length**: 1024 tokens with smart truncation
- **Preprocessing**: Automated cleaning and formatting

### Model Training
- **Framework**: HuggingFace Transformers + PyTorch
- **Optimization**: AdamW with warmup and weight decay
- **Precision**: BFloat16 for A100 compatibility
- **Distributed**: Multi-GPU training with gradient accumulation
- **Monitoring**: TensorBoard logging and GPU utilization tracking

## 📋 Requirements

### Software
- Python 3.11+
- PyTorch 2.0+
- CUDA 12.0+ (for GPU training)
- 16GB+ RAM (32GB+ recommended)

### Hardware (Training)
- **Recommended**: 2x NVIDIA A100 80GB
- **Minimum**: 1x RTX 3090 24GB
- **CPU Only**: Supported but very slow

### Hardware (Inference)
- **GPU**: Any CUDA-compatible GPU with 4GB+ VRAM
- **CPU**: Any modern CPU (slower inference)

## 🔧 Configuration

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0,1  # Multi-GPU training
export TOKENIZERS_PARALLELISM=false  # Avoid warnings
```

### Training Configuration
Edit `processed_data/training_config.json`:
```json
{
  "num_epochs": 5,
  "batch_size": 4,
  "learning_rate": 3e-5,
  "warmup_steps": 100,
  "save_steps": 500
}
```

## 📈 Monitoring Training

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# TensorBoard logs
tensorboard --logdir models/lex_chatbot/logs

# Training progress
tail -f models/lex_chatbot/training.log
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Lex Fridman** for his inspiring podcast and conversations
- **HuggingFace** for the Transformers library and model hosting
- **Microsoft** for the DialoGPT model architecture
- **OpenAI** for advancing conversational AI research

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/Intelligent-Fridman/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/Intelligent-Fridman/discussions)
- **Email**: your-email@example.com

## 🔗 Links

- [Lex Fridman Podcast](https://lexfridman.com/podcast/)
- [DialoGPT Paper](https://arxiv.org/abs/1911.00536)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

*Built with ❤️ for the AI community* 