#!/usr/bin/env python3
"""
Simple Lex Fridman Chatbot Trainer
Ultra-conservative approach for stable training
"""

import json
import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleLexTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ultra-simple configuration
        self.config = {
            "model_name": "microsoft/DialoGPT-medium",
            "max_length": 200,  # Very short to save memory
            "batch_size": 1,
            "gradient_accumulation_steps": 8,  # Small effective batch size
            "learning_rate": 5e-5,
            "num_epochs": 2,  # Just 2 epochs for quick training
            "save_steps": 5000,  # Very infrequent saves
            "logging_steps": 50,
            "warmup_steps": 100,
            "weight_decay": 0.01
        }
        
        self.output_dir = Path("models/lex_chatbot_simple")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Simple Lex Trainer initialized")
        logger.info(f"📱 Device: {self.device}")
    
    def load_dataset(self):
        """Load and prepare a small subset of the dataset"""
        logger.info("📊 Loading dataset...")
        
        with open("processed_data/unified_dataset.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = data["conversations"]
        
        # Use only first 1000 conversations for speed
        conversations = conversations[:1000]
        logger.info(f"📉 Using {len(conversations)} conversations for simple training")
        
        # Create simple text format
        texts = []
        for conv in conversations:
            if conv["input"].strip():
                text = f"{conv['input']} {conv['output']}"
            else:
                text = f"{conv['instruction']} {conv['output']}"
            
            # Keep texts short
            if len(text) > 500:
                text = text[:500]
            
            texts.append(text)
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        
        # Simple train/test split
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        logger.info(f"✅ Dataset prepared:")
        logger.info(f"   Training: {len(split_dataset['train'])} examples")
        logger.info(f"   Validation: {len(split_dataset['test'])} examples")
        
        return split_dataset
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer"""
        logger.info(f"🔧 Setting up {self.config['model_name']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with simple settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            torch_dtype=torch.float32
        )
        
        self.model = self.model.to(self.device)
        
        logger.info(f"✅ Model loaded with {self.model.num_parameters():,} parameters")
    
    def tokenize_dataset(self, dataset):
        """Tokenize the dataset"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config['max_length']
            )
        
        logger.info("🔤 Tokenizing dataset...")
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return tokenized_dataset
    
    def train(self):
        """Simple training"""
        logger.info("🚀 Starting simple training...")
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Setup model
        self.setup_model_and_tokenizer()
        
        # Tokenize
        tokenized_dataset = self.tokenize_dataset(dataset)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Simple training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            warmup_steps=self.config['warmup_steps'],
            logging_steps=self.config['logging_steps'],
            save_steps=self.config['save_steps'],
            save_total_limit=1,
            prediction_loss_only=True,
            remove_unused_columns=True,
            dataloader_num_workers=0,
            no_cuda=False,
            fp16=False,
            report_to=None  # Disable wandb for simplicity
        )
        
        # Simple trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        logger.info(f"📊 Training on {len(tokenized_dataset['train'])} examples")
        logger.info(f"⚡ Effective batch size: {self.config['batch_size'] * self.config['gradient_accumulation_steps']}")
        
        # Train
        train_result = trainer.train()
        
        # Save
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info("✅ Simple training complete!")
        logger.info(f"📁 Model saved to: {self.output_dir}")
        logger.info(f"📊 Final training loss: {train_result.training_loss:.4f}")
        
        return train_result

def main():
    """Main execution"""
    print("🎯 Simple Lex Fridman Chatbot Training")
    print("=" * 40)
    print("📝 Ultra-conservative settings")
    print("🔧 No advanced features, just basic training")
    print("=" * 40)
    
    trainer = SimpleLexTrainer()
    
    try:
        result = trainer.train()
        print("\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {trainer.output_dir}")
        print("\n🎉 Ready to test your simple Lex Fridman AI!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 