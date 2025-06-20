import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from datetime import datetime
import time
import re

# Page configuration
st.set_page_config(
    page_title="Lex Fridman AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online {
        background-color: #4caf50;
    }
    .status-offline {
        background-color: #f44336;
    }
</style>
""", unsafe_allow_html=True)

class LexFridmanChatbot:
    def __init__(self, model_path="models/lex_chatbot_simple"):
        """Initialize the Lex Fridman chatbot."""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 512
        self.conversation_history = []
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        try:
            if os.path.exists(self.model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                
                # Add pad token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,  # Use float32 for stability
                    device_map="auto" if torch.cuda.device_count() > 1 else None
                )
                
                if torch.cuda.device_count() == 1:
                    self.model = self.model.to(self.device)
                
                self.model.eval()
                return True
            else:
                return False
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def generate_response(self, user_input, temperature=0.8, max_new_tokens=150):
        """Generate a response from the chatbot."""
        if not self.model or not self.tokenizer:
            return "Model not loaded. Please check if the model has been trained."
        
        try:
            # Format the input as a conversation
            prompt = f"Human: {user_input}\n\nLex:"
            
            # Add conversation context if available
            if len(self.conversation_history) > 0:
                context = "\n\n".join([
                    f"Human: {conv['user']}\n\nLex: {conv['bot']}"
                    for conv in self.conversation_history[-2:]  # Last 2 exchanges
                ])
                prompt = f"{context}\n\nHuman: {user_input}\n\nLex:"
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=self.max_length, truncation=True)
            
            if self.device == "cuda":
                inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_new_tokens,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after "Lex:")
            try:
                response_start = full_response.rfind("Lex:") + 4
                response = full_response[response_start:].strip()
                
                # Clean up the response
                response = self.clean_response(response)
                
                return response
            except:
                return "I'm having trouble generating a response. Could you try rephrasing your question?"
                
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def clean_response(self, response):
        """Clean and format the generated response."""
        # Remove any remaining conversation markers
        response = re.sub(r'^(Human:|Lex:)', '', response).strip()
        
        # Stop at the next conversation turn
        if "Human:" in response:
            response = response.split("Human:")[0].strip()
        
        # Ensure the response ends with proper punctuation
        if response and not response.endswith(('.', '!', '?')):
            response += "."
        
        # Remove excessive whitespace
        response = re.sub(r'\s+', ' ', response).strip()
        
        return response
    
    def add_to_history(self, user_input, bot_response):
        """Add exchange to conversation history."""
        self.conversation_history.append({
            "user": user_input,
            "bot": bot_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 exchanges
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

def main():
    """Main Streamlit application."""
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = LexFridmanChatbot()
        st.session_state.model_loaded = False
        st.session_state.chat_history = []
    
    # Header
    st.markdown('<div class="main-header">🤖 Lex Fridman AI Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Powered by Fine-tuned Transformer Model</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("🔧 Settings")
        
        # Model status
        model_exists = os.path.exists(st.session_state.chatbot.model_path)
        if model_exists:
            status_color = "status-online" if st.session_state.model_loaded else "status-offline"
            status_text = "Loaded" if st.session_state.model_loaded else "Ready to Load"
        else:
            status_color = "status-offline"
            status_text = "Model Not Found"
        
        st.markdown(f"""
        **Model Status:** 
        <span class="status-indicator {status_color}"></span>{status_text}
        """, unsafe_allow_html=True)
        
        # Load model button
        if model_exists and not st.session_state.model_loaded:
            if st.button("🚀 Load Model", type="primary"):
                with st.spinner("Loading Lex Fridman model..."):
                    success = st.session_state.chatbot.load_model()
                    if success:
                        st.session_state.model_loaded = True
                        st.success("Model loaded successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to load model!")
        
        # Model parameters (if loaded)
        if st.session_state.model_loaded:
            st.subheader("🎛️ Generation Parameters")
            
            temperature = st.slider(
                "Temperature (Creativity)",
                min_value=0.1,
                max_value=1.5,
                value=0.8,
                step=0.1,
                help="Higher values make responses more creative but less focused"
            )
            
            max_tokens = st.slider(
                "Max Response Length",
                min_value=50,
                max_value=300,
                value=150,
                step=25
            )
            
            # Device info
            st.subheader("💻 System Info")
            device_info = "🚀 CUDA" if torch.cuda.is_available() else "💻 CPU"
            st.write(f"**Device:** {device_info}")
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                st.write(f"**GPUs:** {gpu_count}")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    st.write(f"   - GPU {i}: {gpu_name}")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.chat_history = []
            st.session_state.chatbot.conversation_history = []
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model training info
        if model_exists:
            try:
                summary_path = os.path.join(st.session_state.chatbot.model_path, "training_summary.json")
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        training_info = json.load(f)
                    
                    st.subheader("📊 Training Info")
                    st.write(f"**Training Examples:** {training_info.get('training_examples', 'N/A'):,}")
                    st.write(f"**Final Loss:** {training_info.get('training_loss', 'N/A'):.4f}")
                    st.write(f"**GPUs Used:** {training_info.get('num_gpus', 'N/A')}")
            except:
                pass
    
    # Main chat interface
    if not model_exists:
        st.error("🚫 Model not found! Please train the model first by running the fine-tuning script.")
        st.info("Expected model location: `models/lex_chatbot/`")
        return
    
    if not st.session_state.model_loaded:
        st.info("👆 Please load the model using the sidebar to start chatting!")
        return
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Conversation")
        for exchange in st.session_state.chat_history:
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong> {exchange['user']}
            </div>
            """, unsafe_allow_html=True)
            
            # Bot message
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Lex:</strong> {exchange['bot']}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    st.subheader("💭 Ask Lex Anything")
    
    # Sample questions
    with st.expander("💡 Sample Questions"):
        sample_questions = [
            "What do you think about artificial intelligence?",
            "How do you approach difficult conversations?",
            "What's your view on the future of technology?",
            "What does consciousness mean to you?",
            "How do you think about human connection?",
            "What's the most important thing in life?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            col = cols[i % 2]
            if col.button(question, key=f"sample_{i}"):
                user_input = question
                # Process this input
                with st.spinner("🤔 Lex is thinking..."):
                    response = st.session_state.chatbot.generate_response(
                        user_input,
                        temperature=0.8,
                        max_new_tokens=150
                    )
                
                # Add to history
                st.session_state.chat_history.append({
                    "user": user_input,
                    "bot": response
                })
                st.session_state.chatbot.add_to_history(user_input, response)
                st.rerun()
    
    # Text input
    user_input = st.text_input(
        "Your message:",
        placeholder="Type your question here...",
        key="user_input"
    )
    
    # Send button
    col1, col2 = st.columns([1, 5])
    
    with col1:
        send_clicked = st.button("📤 Send", type="primary")
    
    with col2:
        if st.button("🎲 Random Question"):
            sample_questions = [
                "What's your perspective on consciousness?",
                "How do you think about the future?",
                "What makes a good conversation?",
                "What's the meaning of life?",
                "How do you deal with difficult topics?"
            ]
            user_input = sample_questions[int(time.time()) % len(sample_questions)]
            send_clicked = True
    
    # Process input
    if (send_clicked or user_input) and user_input.strip():
        with st.spinner("🤔 Lex is thinking..."):
            response = st.session_state.chatbot.generate_response(
                user_input,
                temperature=temperature if 'temperature' in locals() else 0.8,
                max_new_tokens=max_tokens if 'max_tokens' in locals() else 150
            )
        
        # Add to history
        st.session_state.chat_history.append({
            "user": user_input,
            "bot": response
        })
        st.session_state.chatbot.add_to_history(user_input, response)
        
        # Clear input and refresh
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🤖 This chatbot was fine-tuned on Lex Fridman podcast transcripts<br>
        Built with Streamlit • Powered by Transformers • Accelerated by A100 GPUs
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 