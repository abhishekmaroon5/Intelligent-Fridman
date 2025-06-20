#!/usr/bin/env python3
"""
Enhanced Lex Fridman Chatbot Trainer
Using the comprehensive 6-transcript dataset for improved training
"""

import json
import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    get_cosine_schedule_with_warmup
)
from datasets import Dataset
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List
import wandb
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedLexTrainer:
    def __init__(self, config_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count()
        
        # Enhanced configuration for 6-transcript dataset - Fixed memory issues
        self.config = {
            "model_name": "microsoft/DialoGPT-large",  # Upgraded to large model
            "max_length": 384,  # Reduced max length to save memory
            "batch_size": 1,  # Further reduced batch size for stability
            "gradient_accumulation_steps": 32,  # Effective batch size: 1 * 32 = 32
            "learning_rate": 2e-5,  # Lower learning rate for stability
            "num_epochs": 4,  # Reduced epochs to avoid long training
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "save_steps": 1000,  # Less frequent saves
            "eval_steps": 500,  # Less frequent evaluation
            "logging_steps": 100,
            "save_total_limit": 2,  # Keep fewer checkpoints
            "load_best_model_at_end": False,  # Disable to save memory
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "early_stopping_patience": 3,
            "fp16": False,  # Disabled FP16 to fix gradient scaling issue
            "bf16": False,  # Also disable bf16 for compatibility
            "dataloader_num_workers": 0,  # No parallel loading to save memory
            "remove_unused_columns": True,  # Remove unused columns
            "push_to_hub": False,
            "report_to": "wandb",  # Enable W&B logging
            "gradient_checkpointing": True,  # Enable to save memory
            "max_grad_norm": 1.0,  # Gradient clipping for stability
            "prediction_loss_only": True,  # Only compute loss, not predictions
            "eval_accumulation_steps": 4  # Accumulate eval steps to save memory
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                self.config.update(custom_config)
        
        self.output_dir = Path("models/lex_chatbot_enhanced")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Enhanced Lex Trainer initialized")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"🔢 GPUs: {self.num_gpus}")
        logger.info(f"🏗️ Model: {self.config['model_name']}")
    
    def load_dataset(self, dataset_path: str = "processed_data/unified_dataset.json") -> Dataset:
        """Load and prepare the enhanced 6-transcript dataset"""
        logger.info(f"📊 Loading dataset from {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = data["conversations"]
        logger.info(f"📈 Loaded {len(conversations)} conversations from {len(data['metadata']['sources'])} episodes")
        
        # Create training examples in the right format
        texts = []
        for conv in conversations:
            # Create a conversational format that DialoGPT can understand
            if conv["input"].strip():
                text = f"<|endoftext|>{conv['input']}<|endoftext|>{conv['output']}<|endoftext|>"
            else:
                # For cases where input is empty, use the instruction as context
                text = f"<|endoftext|>{conv['instruction']}<|endoftext|>{conv['output']}<|endoftext|>"
            texts.append(text)
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        
        # Split into train/validation
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        logger.info(f"✅ Dataset prepared:")
        logger.info(f"   Training: {len(split_dataset['train'])} examples")
        logger.info(f"   Validation: {len(split_dataset['test'])} examples")
        
        return split_dataset
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with enhanced configuration"""
        logger.info(f"🔧 Setting up {self.config['model_name']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            padding_side="right",
            use_fast=True
        )
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with fixed dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            torch_dtype=torch.float32,  # Use float32 for stability
            device_map="auto" if self.num_gpus > 1 else None
        )
        
        # Enable gradient checkpointing if specified
        if self.config.get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
        
        # Resize token embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"✅ Model loaded with {self.model.num_parameters():,} parameters")
    
    def tokenize_dataset(self, dataset):
        """Tokenize the dataset for training"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,  # We'll pad in the data collator
                max_length=self.config['max_length'],
                return_tensors=None
            )
        
        logger.info("🔤 Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing"
        )
        
        return tokenized_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute training metrics - simplified to save memory"""
        # Since we're using prediction_loss_only=True, we don't get predictions
        # Just return empty metrics to avoid memory issues
        return {}
    
    def train(self, dataset_path: str = "processed_data/unified_dataset.json"):
        """Enhanced training with the 6-transcript dataset"""
        # Clear GPU memory before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Initialize wandb
        wandb.init(
            project="lex-fridman-chatbot-enhanced",
            name=f"enhanced-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=self.config
        )
        
        # Load and prepare dataset
        dataset = self.load_dataset(dataset_path)
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Tokenize dataset
        tokenized_dataset = self.tokenize_dataset(dataset)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=None  # No padding alignment needed for fp32
        )
        
        # Enhanced training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            warmup_ratio=self.config['warmup_ratio'],
            logging_steps=self.config['logging_steps'],
            save_steps=self.config['save_steps'],
            eval_steps=self.config['eval_steps'],
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=self.config['save_total_limit'],
            load_best_model_at_end=self.config['load_best_model_at_end'],
            metric_for_best_model=self.config['metric_for_best_model'],
            greater_is_better=self.config['greater_is_better'],
            fp16=self.config.get('fp16', False),
            bf16=self.config.get('bf16', False),
            dataloader_num_workers=self.config['dataloader_num_workers'],
            remove_unused_columns=self.config['remove_unused_columns'],
            push_to_hub=self.config['push_to_hub'],
            report_to=self.config['report_to'],
            run_name=f"lex-enhanced-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            logging_dir=str(self.output_dir / "logs"),
            ddp_find_unused_parameters=False,
            group_by_length=False,  # Disabled to avoid length column issues
            gradient_checkpointing=self.config['gradient_checkpointing'],
            max_grad_norm=self.config['max_grad_norm'],
            prediction_loss_only=self.config['prediction_loss_only'],
            eval_accumulation_steps=self.config['eval_accumulation_steps']
        )
        
        # Custom memory-optimized trainer
        class MemoryOptimizedTrainer(Trainer):
            def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
                """Memory-optimized evaluation loop"""
                # Clear cache before evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Force prediction_loss_only to True to save memory
                return super().evaluation_loop(
                    dataloader=dataloader,
                    description=description,
                    prediction_loss_only=True,  # Force to True
                    ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key_prefix
                )
        
        # Trainer with enhanced features
        trainer = MemoryOptimizedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=None,  # Disable metrics computation to save memory
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config['early_stopping_patience']
                )
            ]
        )
        
        # Start training
        logger.info("🚀 Starting enhanced training...")
        logger.info(f"📊 Training on {len(tokenized_dataset['train'])} examples")
        logger.info(f"🔍 Validating on {len(tokenized_dataset['test'])} examples")
        logger.info(f"⚡ Effective batch size: {self.config['batch_size'] * self.config['gradient_accumulation_steps'] * self.num_gpus}")
        
        train_result = trainer.train()
        
        # Save the final model
        trainer.save_model()
        trainer.save_state()
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training summary
        training_summary = {
            "dataset_info": {
                "total_conversations": len(dataset["train"]) + len(dataset["test"]),
                "training_examples": len(tokenized_dataset["train"]),
                "validation_examples": len(tokenized_dataset["test"]),
                "episodes_used": 6,
                "estimated_words": 645849  # From our dataset analysis
            },
            "model_info": {
                "base_model": self.config['model_name'],
                "parameters": self.model.num_parameters(),
                "output_dir": str(self.output_dir)
            },
            "training_config": self.config,
            "training_results": {
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.training_time,
                "train_samples_per_second": train_result.training_samples_per_second,
                "train_steps_per_second": train_result.training_steps_per_second
            }
        }
        
        with open(self.output_dir / "enhanced_training_summary.json", 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        logger.info("✅ Enhanced training complete!")
        logger.info(f"📁 Model saved to: {self.output_dir}")
        logger.info(f"📊 Final training loss: {train_result.training_loss:.4f}")
        
        # Close wandb
        wandb.finish()
        
        return trainer, train_result

def main():
    """Main training execution"""
    print("🚀 Enhanced Lex Fridman Chatbot Training")
    print("=" * 50)
    print("🎯 Using 6-transcript dataset with 645K+ words")
    print("🏗️ DialoGPT-Large with enhanced training techniques")
    print("=" * 50)
    
    trainer = EnhancedLexTrainer()
    
    try:
        model, results = trainer.train()
        print("\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {trainer.output_dir}")
        print("\n🎉 Ready to chat with your enhanced Lex Fridman AI!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 