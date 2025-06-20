#!/usr/bin/env python3
"""
Upload trained Lex Fridman chatbot model to Hugging Face Hub
"""

import os
import json
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def upload_model_to_hf():
    """Upload the trained model to Hugging Face Hub."""
    
    print("🚀 Uploading Lex Fridman Chatbot to Hugging Face Hub")
    print("=" * 60)
    
    # Configuration
    model_path = "models/lex_chatbot"
    repo_name = "abhishekmaroon5/lex-fridman-chatbot"  # Your HF username/repo
    organization = None  # Set to your organization name if uploading to org
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("❌ Model not found! Please train the model first.")
        return False
    
    print(f"📁 Local model path: {model_path}")
    print(f"🏷️  Repository name: {repo_name}")
    
    # Load model info
    try:
        with open(f"{model_path}/training_summary.json", 'r') as f:
            training_info = json.load(f)
        print(f"📊 Training info loaded: {training_info['num_epochs']} epochs, {training_info['training_examples']} examples")
    except:
        training_info = {}
        print("⚠️  No training summary found")
    
    # Create model card
    model_card = create_model_card(training_info)
    
    try:
        # Initialize Hugging Face API
        api = HfApi()
        
        # Create repository
        repo_id = repo_name  # Already includes username
        print(f"\n🔄 Creating repository: {repo_id}")
        
        try:
            create_repo(
                repo_id=repo_id,
                token=None,  # Will use HF_TOKEN environment variable
                repo_type="model",
                exist_ok=True
            )
            print("✅ Repository created/verified")
        except Exception as e:
            print(f"⚠️  Repository creation: {e}")
        
        # Save model card
        model_card_path = f"{model_path}/README.md"
        with open(model_card_path, 'w') as f:
            f.write(model_card)
        print("✅ Model card created")
        
        # Upload model folder
        print(f"\n📤 Uploading model files to {repo_id}...")
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            token=None,  # Will use HF_TOKEN environment variable
            commit_message="Upload Lex Fridman chatbot model",
            ignore_patterns=["*.log", "logs/", "__pycache__/"]
        )
        
        print("✅ Model uploaded successfully!")
        print(f"🌐 Model available at: https://huggingface.co/{repo_id}")
        
        # Test loading from hub
        print(f"\n🧪 Testing model loading from Hub...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            model = AutoModelForCausalLM.from_pretrained(repo_id)
            print("✅ Model can be loaded from Hub successfully!")
        except Exception as e:
            print(f"⚠️  Model loading test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Hugging Face account and token set up")
        print("   2. HF_TOKEN environment variable set")
        print("   3. huggingface_hub library installed")
        return False

def create_model_card(training_info):
    """Create a model card for the Hugging Face repository."""
    
    card = f"""---
language: en
license: mit
tags:
- conversational
- chatbot
- lex-fridman
- dialogue
- pytorch
- transformers
- fine-tuned
base_model: microsoft/DialoGPT-medium
datasets:
- custom
widget:
- text: "Human: What do you think about artificial intelligence?\\n\\nLex:"
  example_title: "AI Discussion"
- text: "Human: How do you approach difficult conversations?\\n\\nLex:"
  example_title: "Conversation Approach"
- text: "Human: What's your view on the future of technology?\\n\\nLex:"
  example_title: "Future of Technology"
---

# Lex Fridman AI Chatbot

This is a fine-tuned conversational AI model that mimics the conversational style of Lex Fridman, host of the Lex Fridman Podcast. The model was fine-tuned on podcast transcript data to capture Lex's thoughtful, philosophical, and engaging conversation style.

## Model Details

- **Base Model**: microsoft/DialoGPT-medium
- **Model Type**: Conversational AI / Chatbot
- **Language**: English
- **Parameters**: ~117M
- **Training Data**: Lex Fridman podcast transcripts
- **Fine-tuning**: Custom dialogue dataset with Human-Lex conversation pairs

## Training Details

{f"- **Training Examples**: {training_info.get('training_examples', 'N/A'):,}" if training_info.get('training_examples') else ""}
{f"- **Validation Examples**: {training_info.get('validation_examples', 'N/A'):,}" if training_info.get('validation_examples') else ""}
{f"- **Training Duration**: {training_info.get('training_duration', 'N/A')}" if training_info.get('training_duration') else ""}
{f"- **Final Loss**: {training_info.get('training_loss', 'N/A'):.4f}" if training_info.get('training_loss') else ""}
{f"- **Epochs**: {training_info.get('num_epochs', 'N/A')}" if training_info.get('num_epochs') else ""}
- **Hardware**: NVIDIA A100 GPUs
- **Framework**: PyTorch + Transformers

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("your-username/lex-fridman-chatbot")
model = AutoModelForCausalLM.from_pretrained("your-username/lex-fridman-chatbot")

# Generate response
def chat_with_lex(question):
    prompt = f"Human: {{question}}\\n\\nLex:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 100,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Lex:")[-1].strip()

# Example usage
response = chat_with_lex("What do you think about artificial intelligence?")
print(response)
```

## Sample Conversations

**Human**: What do you think about artificial intelligence?
**Lex**: I think it's a big idea that we're seeing in the way computers are becoming self aware.

**Human**: How do you approach difficult conversations?
**Lex**: You approach them with your hands and you have a conversation with the viewer.

## Web Interface

This model comes with a beautiful Streamlit web interface for easy interaction:

```bash
streamlit run web_app/lex_chatbot_app.py
```

## Model Architecture

- **Architecture**: GPT-2 based transformer (DialoGPT)
- **Vocabulary Size**: 50,257 tokens
- **Context Length**: 1024 tokens
- **Precision**: FP16 for efficient inference
- **Multi-GPU**: Supports distributed inference

## Limitations and Biases

- The model is trained on a limited dataset from podcast transcripts
- May reflect biases present in the training data
- Responses are generated based on patterns learned from transcripts
- Not suitable for providing factual information or professional advice
- Best used for casual conversation and entertainment

## Ethical Considerations

- This model is for educational and entertainment purposes
- Does not represent the actual views or opinions of Lex Fridman
- Should not be used to impersonate or misrepresent the real person
- Users should be transparent about AI-generated content

## Citation

```bibtex
@misc{{lex-fridman-chatbot,
  title={{Lex Fridman AI Chatbot}},
  author={{Your Name}},
  year={{2025}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/your-username/lex-fridman-chatbot}}
}}
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Lex Fridman for the inspiring podcast content
- Microsoft for the DialoGPT base model
- Hugging Face for the transformers library
- The open-source AI community
"""
    
    return card

if __name__ == "__main__":
    print("🤖 Lex Fridman Chatbot - Hugging Face Upload")
    print("=" * 50)
    
    # Check for HF token
    if not os.getenv("HF_TOKEN"):
        print("⚠️  HF_TOKEN environment variable not set!")
        print("💡 Please set your Hugging Face token:")
        print("   export HF_TOKEN=your_token_here")
        print("   or create .env file with HF_TOKEN=your_token_here")
        print("\n🔑 Get your token at: https://huggingface.co/settings/tokens")
        
        # Ask user if they want to continue
        response = input("\n❓ Do you want to continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("👋 Exiting. Please set up your HF token first.")
            exit(1)
    
    success = upload_model_to_hf()
    
    if success:
        print("\n🎉 Upload completed successfully!")
        print("🌐 Your model is now available on Hugging Face Hub!")
    else:
        print("\n❌ Upload failed. Please check the errors above.") 