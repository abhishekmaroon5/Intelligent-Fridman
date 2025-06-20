#!/usr/bin/env python3
"""
Test Script for Lex Fridman Chatbot
Interactive testing of the trained model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LexChatbotTester:
    def __init__(self, model_path: str = "models/lex_chatbot_simple"):
        """Initialize the chatbot tester"""
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🤖 Loading Lex Fridman Chatbot from: {model_path}")
        print(f"🔧 Device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("✅ Model loaded successfully!")
        
    def generate_response(self, prompt: str, max_length: int = 200, temperature: float = 0.8, num_return_sequences: int = 1):
        """Generate a response from the chatbot"""
        
        # Format prompt like Lex Fridman conversation
        formatted_prompt = f"Human: {prompt}\nLex:"
        
        # Tokenize input
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt", truncation=True, max_length=150)
        inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
            generation_time = time.time() - start_time
        
        # Decode response
        responses = []
        for output in outputs:
            # Remove the input prompt from the output
            response_tokens = output[inputs.shape[1]:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Clean up the response
            response = response.split("Human:")[0].strip()  # Stop at next human input
            response = response.split("\n\n")[0].strip()    # Stop at paragraph break
            
            responses.append(response)
        
        return responses, generation_time
    
    def run_test_questions(self):
        """Run a series of test questions covering different topics from our dataset"""
        
        test_questions = [
            # Gaming/Tech (Tim Sweeney topics)
            "What do you think about the future of game engines?",
            "How has programming changed over the years?",
            "What's your view on virtual reality?",
            
            # AI/Technology (Dario Amodei, DeepSeek topics)
            "What are your thoughts on artificial general intelligence?",
            "How do you see AI developing in the next decade?",
            "What concerns do you have about AI safety?",
            
            # Physics/Science (Sara Walker topics)
            "What is consciousness?",
            "Do you think we'll find alien life?",
            "How do you think about the nature of time?",
            
            # Politics/Society (Israel-Palestine, Mark Zuckerberg topics)
            "What role does technology play in society?",
            "How do you approach controversial topics?",
            "What's your view on social media?",
            
            # General Lex-style questions
            "What is the meaning of life?",
            "What advice would you give to young people?",
            "What brings you joy?"
        ]
        
        print("\n" + "="*60)
        print("🧪 TESTING LEX FRIDMAN CHATBOT")
        print("="*60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n💬 Test {i}/{ len(test_questions)}")
            print(f"❓ Question: {question}")
            print("-" * 50)
            
            try:
                responses, gen_time = self.generate_response(question)
                
                for j, response in enumerate(responses):
                    print(f"🤖 Lex: {response}")
                    if j < len(responses) - 1:
                        print("   " + "-" * 30)
                
                print(f"⏱️  Generation time: {gen_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 50)
            
            # Pause between questions
            time.sleep(1)
    
    def interactive_mode(self):
        """Interactive chat mode"""
        print("\n" + "="*60)
        print("💬 INTERACTIVE MODE - Chat with Lex!")
        print("Type 'quit' to exit")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("🤖 Lex is thinking...")
                responses, gen_time = self.generate_response(user_input)
                
                print(f"🤖 Lex: {responses[0]}")
                print(f"⏱️  ({gen_time:.2f}s)")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🚀 Lex Fridman Chatbot Tester")
    print("="*50)
    
    # Initialize tester
    try:
        tester = LexChatbotTester()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Show menu
    while True:
        print("\n📋 Choose an option:")
        print("1. Run automated test questions")
        print("2. Interactive chat mode")
        print("3. Quick single test")
        print("4. Exit")
        
        choice = input("\n👉 Enter choice (1-4): ").strip()
        
        if choice == "1":
            tester.run_test_questions()
        elif choice == "2":
            tester.interactive_mode()
        elif choice == "3":
            question = input("Enter your question: ").strip()
            if question:
                responses, gen_time = tester.generate_response(question)
                print(f"\n🤖 Lex: {responses[0]}")
                print(f"⏱️  Generation time: {gen_time:.2f}s")
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 