# Contributing to Intelligent Fridman

Thank you for your interest in contributing to the Intelligent Fridman project! We welcome contributions from the community and are excited to work with you.

## 🤝 How to Contribute

### 1. Fork the Repository
```bash
git clone https://github.com/your-username/Intelligent-Fridman.git
cd Intelligent-Fridman
```

### 2. Set Up Development Environment
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

## 🛠️ Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Include type hints where appropriate

### Testing
- Write tests for new features
- Ensure all existing tests pass
- Test on both CPU and GPU environments if possible

### Documentation
- Update README.md if adding new features
- Document new configuration options
- Include examples for new functionality

## 📝 Types of Contributions

### 🐛 Bug Reports
When filing a bug report, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU specs)
- Error messages and logs

### ✨ Feature Requests
For feature requests, please provide:
- Clear description of the feature
- Use case and motivation
- Proposed implementation approach
- Any relevant examples or references

### 🔧 Code Contributions
Areas where we welcome contributions:
- **Data Collection**: Improving transcript quality and coverage
- **Model Training**: Optimization and new architectures
- **Web Interface**: UI/UX improvements and new features
- **Documentation**: Better guides and examples
- **Performance**: Speed and memory optimizations
- **Testing**: More comprehensive test coverage

## 🚀 Development Workflow

### 1. Data Collection Improvements
```bash
# Test transcript collection
python scripts/working_transcript_downloader.py --test-mode

# Validate data quality
python scripts/data_validation.py
```

### 2. Model Training Changes
```bash
# Test training pipeline
python scripts/model_fine_tuner.py --dry-run

# Validate model outputs
python scripts/model_testing.py
```

### 3. Web Interface Updates
```bash
# Run local development server
streamlit run web_app/lex_chatbot_app.py --server.port 8501
```

## 📊 Performance Considerations

### GPU Memory
- Test with various GPU configurations
- Optimize batch sizes for different hardware
- Include memory usage monitoring

### Training Speed
- Profile training steps
- Consider gradient accumulation strategies
- Test multi-GPU scaling

### Inference Performance
- Benchmark response times
- Optimize model loading
- Test on different hardware configurations

## 🧪 Testing Guidelines

### Unit Tests
```bash
# Run unit tests
python -m pytest tests/unit/

# Test specific modules
python -m pytest tests/unit/test_tokenizer.py
```

### Integration Tests
```bash
# Full pipeline test
python -m pytest tests/integration/

# End-to-end test
python tests/e2e/test_complete_workflow.py
```

### GPU Tests
```bash
# GPU-specific tests
python -m pytest tests/gpu/ -k "test_gpu"
```

## 📋 Checklist Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Feature is tested on both CPU and GPU (if applicable)
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with main

## 🐛 Debugging Tips

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size in training config
2. **Tokenizer Warnings**: Set `TOKENIZERS_PARALLELISM=false`
3. **Model Loading Fails**: Check PyTorch and CUDA versions
4. **Streamlit Errors**: Ensure all dependencies are installed

### Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### GPU Debugging
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"
```

## 💬 Communication

### Getting Help
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Reviews**: Detailed feedback on pull requests

### Submitting Changes
1. Ensure your branch is up to date
2. Write clear commit messages
3. Include tests for new functionality
4. Update documentation as needed
5. Submit a pull request with detailed description

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Special mentions for major features or fixes

## 📚 Resources

- [Lex Fridman Podcast](https://lexfridman.com/podcast/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📞 Contact

For questions about contributing, please:
1. Check existing issues and discussions
2. Create a new issue with your question
3. Join the community discussions

Thank you for contributing to Intelligent Fridman! 🤖❤️ 