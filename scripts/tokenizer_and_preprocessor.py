import os
import json
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, DatasetDict
import numpy as np
from typing import List, Dict, Optional
import re
from datetime import datetime

class LexFridmanTokenizer:
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 data_dir: str = "data",
                 output_dir: str = "processed_data",
                 max_length: int = 1024):
        """
        Initialize the tokenizer and preprocessor.
        
        Args:
            model_name: Base model to use for tokenization
            data_dir: Directory containing the collected transcripts
            output_dir: Directory to save processed data
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_length = max_length
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"🤖 Initializing tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"📝 Tokenizer vocab size: {len(self.tokenizer)}")
        print(f"🔢 Max sequence length: {max_length}")
        
    def load_transcripts(self) -> List[Dict]:
        """Load all collected transcripts."""
        transcript_file = os.path.join(self.data_dir, "all_transcripts.json")
        
        if not os.path.exists(transcript_file):
            raise FileNotFoundError(f"Transcripts file not found: {transcript_file}")
        
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcripts = json.load(f)
        
        print(f"📚 Loaded {len(transcripts)} transcripts")
        total_words = sum(t.get('word_count', 0) for t in transcripts)
        print(f"📝 Total words: {total_words:,}")
        
        return transcripts
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove or fix common transcript artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove [Music], [Applause], etc.
        text = re.sub(r'\(.*?\)', '', text)  # Remove (inaudible), etc.
        text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
        text = text.strip()
        
        # Fix common transcript issues
        text = re.sub(r'\.{2,}', '.', text)  # Multiple dots to single
        text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Space after punctuation
        
        return text
    
    def create_conversation_format(self, transcript: str, title: str) -> List[str]:
        """
        Convert transcript to conversation format for training.
        This creates training examples in a dialogue format.
        """
        # Split transcript into sentences/segments
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        conversations = []
        
        # Create question-answer pairs from consecutive sentences
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                question = sentences[i].strip()
                answer = sentences[i + 1].strip()
                
                if len(question) > 20 and len(answer) > 20:
                    # Format as conversation
                    conversation = f"Human: {question}\n\nLex: {answer}"
                    conversations.append(conversation)
        
        # Also create some context-based examples
        context_examples = []
        for i in range(len(sentences) - 2):
            context = sentences[i]
            question = sentences[i + 1]
            answer = sentences[i + 2]
            
            if all(len(s.strip()) > 15 for s in [context, question, answer]):
                example = f"Context: {context}\n\nHuman: {question}\n\nLex: {answer}"
                context_examples.append(example)
        
        return conversations + context_examples
    
    def tokenize_examples(self, examples: List[str]) -> Dict:
        """Tokenize examples for training."""
        tokenized_inputs = []
        attention_masks = []
        
        for example in examples:
            # Tokenize with truncation and padding
            encoded = self.tokenizer(
                example,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            tokenized_inputs.append(encoded['input_ids'].squeeze())
            attention_masks.append(encoded['attention_mask'].squeeze())
        
        return {
            'input_ids': torch.stack(tokenized_inputs),
            'attention_mask': torch.stack(attention_masks)
        }
    
    def process_all_transcripts(self) -> DatasetDict:
        """Process all transcripts and create training dataset."""
        print("🔄 Processing transcripts...")
        
        transcripts = self.load_transcripts()
        
        all_conversations = []
        
        for transcript_data in transcripts:
            title = transcript_data.get('title', 'Unknown')
            transcript_text = transcript_data.get('transcript', '')
            
            if not transcript_text:
                continue
                
            print(f"   📝 Processing: {title[:50]}...")
            
            # Clean the transcript
            cleaned_text = self.clean_text(transcript_text)
            
            # Create conversation examples
            conversations = self.create_conversation_format(cleaned_text, title)
            all_conversations.extend(conversations)
            
            print(f"      ✅ Created {len(conversations)} conversation examples")
        
        print(f"\n📊 Total conversation examples: {len(all_conversations)}")
        
        # Split into train/validation sets
        train_size = int(0.9 * len(all_conversations))
        train_conversations = all_conversations[:train_size]
        val_conversations = all_conversations[train_size:]
        
        print(f"🚂 Training examples: {len(train_conversations)}")
        print(f"✅ Validation examples: {len(val_conversations)}")
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'text': train_conversations
        })
        
        val_dataset = Dataset.from_dict({
            'text': val_conversations
        })
        
        # Tokenize datasets
        print("🔤 Tokenizing datasets...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
        
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        
        # Save processed datasets
        dataset_path = os.path.join(self.output_dir, "tokenized_datasets")
        dataset_dict.save_to_disk(dataset_path)
        print(f"💾 Saved tokenized datasets to: {dataset_path}")
        
        # Save processing summary
        summary = {
            'model_name': self.model_name,
            'total_transcripts': len(transcripts),
            'total_conversations': len(all_conversations),
            'train_examples': len(train_conversations),
            'validation_examples': len(val_conversations),
            'max_length': self.max_length,
            'vocab_size': len(self.tokenizer),
            'processed_at': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(self.output_dir, "processing_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Processing summary saved to: {summary_path}")
        
        return dataset_dict
    
    def create_training_config(self) -> TrainingArguments:
        """Create training configuration for fine-tuning."""
        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, "lex_chatbot_model"),
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            warmup_steps=500,
            logging_steps=100,
            save_steps=1000,
            eval_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            prediction_loss_only=True,
            dataloader_drop_last=True,
            fp16=True,  # Use mixed precision for faster training
            gradient_checkpointing=True,  # Save memory
            remove_unused_columns=True,
            report_to=None,  # Disable wandb
        )
        
        # Save training config
        config_path = os.path.join(self.output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(training_args.to_dict(), f, indent=2)
        
        print(f"⚙️  Training config saved to: {config_path}")
        
        return training_args

def main():
    """Main processing function."""
    print("🚀 Lex Fridman Transcript Tokenizer & Preprocessor")
    print("=" * 60)
    
    # Initialize tokenizer
    tokenizer = LexFridmanTokenizer(
        model_name="microsoft/DialoGPT-medium",  # Good for dialogue
        max_length=1024  # Reasonable for most conversations
    )
    
    try:
        # Process all transcripts
        dataset_dict = tokenizer.process_all_transcripts()
        
        # Create training configuration
        training_args = tokenizer.create_training_config()
        
        print("\n🎉 Tokenization Complete!")
        print("=" * 60)
        print(f"📊 Dataset Statistics:")
        print(f"   🚂 Training examples: {len(dataset_dict['train'])}")
        print(f"   ✅ Validation examples: {len(dataset_dict['validation'])}")
        print(f"   🔤 Vocab size: {len(tokenizer.tokenizer)}")
        print(f"   📏 Max length: {tokenizer.max_length}")
        
        print(f"\n📁 Files created:")
        print(f"   📦 Tokenized datasets: processed_data/tokenized_datasets/")
        print(f"   📋 Processing summary: processed_data/processing_summary.json")
        print(f"   ⚙️  Training config: processed_data/training_config.json")
        
        print(f"\n✅ Ready for Step 3: Model Fine-tuning!")
        print(f"   🚀 Your 2x A100 GPUs are ready to train!")
        print(f"   🎯 Next: Run the fine-tuning script")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure you've run the transcript collection first.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 