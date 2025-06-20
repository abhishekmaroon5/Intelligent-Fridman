---
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
- text: "Human: What do you think about artificial intelligence?\n\nLex:"
  example_title: "AI Discussion"
- text: "Human: How do you approach difficult conversations?\n\nLex:"
  example_title: "Conversation Approach"
- text: "Human: What's your view on the future of technology?\n\nLex:"
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

- **Training Examples**: 3,100
- **Validation Examples**: 345
- **Training Duration**: 0:20:30.474449
- **Final Loss**: 4.1058
- **Epochs**: 5
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
    prompt = f"Human: {question}\n\nLex:"
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
@misc{lex-fridman-chatbot,
  title={Lex Fridman AI Chatbot},
  author={Your Name},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/your-username/lex-fridman-chatbot}
}
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Lex Fridman for the inspiring podcast content
- Microsoft for the DialoGPT base model
- Hugging Face for the transformers library
- The open-source AI community
