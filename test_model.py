#!/usr/bin/env python3
"""
Quick test script for the trained Lex Fridman chatbot model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time

def test_model():
    """Test the trained model with sample questions."""
    
    print("🤖 Testing Lex Fridman Chatbot Model")
    print("=" * 50)
    
    model_path = "models/lex_chatbot"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("❌ Model not found! Please train the model first.")
        return False
    
    print(f"📁 Model path: {model_path}")
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🚀 GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("\n🔄 Loading model and tokenizer...")
    start_time = time.time()
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Tokenizer loaded")
        
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if torch.cuda.device_count() > 1 else None
        )
        
        if torch.cuda.device_count() == 1:
            model = model.to(device)
        
        model.eval()
        print("✅ Model loaded")
        
        load_time = time.time() - start_time
        print(f"⏱️  Loading time: {load_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test questions
    test_questions = [
        "What do you think about artificial intelligence?",
        "How do you approach difficult conversations?",
        "What's your view on the future of technology?",
        "What does consciousness mean to you?"
    ]
    
    print(f"\n🧪 Testing with {len(test_questions)} sample questions...")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n💭 Question {i}: {question}")
        print("-" * 40)
        
        try:
            # Format prompt
            prompt = f"Human: {question}\n\nLex:"
            
            # Tokenize
            inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            if device == "cuda":
                inputs = inputs.to(device)
            
            # Generate response
            start_gen = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            gen_time = time.time() - start_gen
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract Lex's response
            try:
                response_start = full_response.rfind("Lex:") + 4
                response = full_response[response_start:].strip()
                
                # Clean response
                if "Human:" in response:
                    response = response.split("Human:")[0].strip()
                
                if response and not response.endswith(('.', '!', '?')):
                    response += "."
                
                print(f"🤖 Lex: {response}")
                print(f"⏱️  Generation time: {gen_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Error processing response: {e}")
                
        except Exception as e:
            print(f"❌ Error generating response: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Model testing completed!")
    print("🚀 Ready to launch the web application!")
    
    return True

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n🌐 To launch the web app, run:")
        print("   streamlit run web_app/lex_chatbot_app.py")
        print("\n🔗 Or use the launcher script:")
        print("   ./run_chatbot.sh") 