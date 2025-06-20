#!/usr/bin/env python3
"""
Memory-Efficient Lex Fridman Chatbot Trainer
Optimized for limited GPU memory with DialoGPT-medium
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
    EarlyStoppingCallback
)
from datasets import Dataset
import logging
from pathlib import Path
import wandb
from datetime import datetime
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryEfficientLexTrainer:
    def __init__(self, config_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count()
        
        # Ultra-conservative configuration for memory efficiency
        self.config = {
            "model_name": "microsoft/DialoGPT-medium",  # Smaller model for memory
            "max_length": 256,  # Very short sequences to save memory
            "batch_size": 1,  # Minimal batch size
            "gradient_accumulation_steps": 16,  # Effective batch size: 1 * 16 = 16
            "learning_rate": 3e-5,  # Standard learning rate
            "num_epochs": 3,  # Few epochs for quick training
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
            "save_steps": 2000,  # Very infrequent saves
            "eval_steps": 1000,  # Infrequent evaluation
            "logging_steps": 200,
            "save_total_limit": 1,  # Keep only 1 checkpoint
            "load_best_model_at_end": True,  # Required for EarlyStoppingCallback
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "early_stopping_patience": 2,
            "fp16": False,  # Use fp32 for stability
            "bf16": False,
            "dataloader_num_workers": 0,  # No parallel loading
            "remove_unused_columns": True,
            "push_to_hub": False,
            "report_to": "wandb",
            "gradient_checkpointing": True,
            "max_grad_norm": 1.0,
            "prediction_loss_only": True,  # Only loss, no predictions
            "eval_accumulation_steps": 8,  # Large accumulation for eval
            "dataloader_pin_memory": False  # Don't pin memory
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                self.config.update(custom_config)
        
        self.output_dir = Path("models/lex_chatbot_memory_efficient")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Memory-Efficient Lex Trainer initialized")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"🔢 GPUs: {self.num_gpus}")
        logger.info(f"🏗️ Model: {self.config['model_name']}")
    
    def clear_memory(self):
        """Aggressive memory clearing"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def load_dataset(self, dataset_path: str = "processed_data/unified_dataset.json") -> Dataset:
        """Load and prepare the dataset with memory optimization"""
        logger.info(f"📊 Loading dataset from {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = data["conversations"]
        logger.info(f"📈 Loaded {len(conversations)} conversations")
        
        # Limit dataset size for memory efficiency
        max_conversations = min(len(conversations), 5000)  # Limit to 5K conversations
        conversations = conversations[:max_conversations]
        logger.info(f"📉 Limited to {max_conversations} conversations for memory efficiency")
        
        # Create training examples in simple format
        texts = []
        for conv in conversations:
            # Simple conversational format
            if conv["input"].strip():
                text = f"{conv['input']} {conv['output']}"
            else:
                text = f"{conv['instruction']} {conv['output']}"
            
            # Truncate very long texts
            if len(text) > 1000:
                text = text[:1000]
            
            texts.append(text)
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        
        # Split into train/validation
        split_dataset = dataset.train_test_split(test_size=0.05, seed=42)  # Smaller validation set
        
        logger.info(f"✅ Dataset prepared:")
        logger.info(f"   Training: {len(split_dataset['train'])} examples")
        logger.info(f"   Validation: {len(split_dataset['test'])} examples")
        
        return split_dataset
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with memory optimization"""
        logger.info(f"🔧 Setting up {self.config['model_name']}")
        
        # Clear memory before loading
        self.clear_memory()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            padding_side="right",
            use_fast=True
        )
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with memory optimization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            torch_dtype=torch.float32,
            device_map=None  # Load on single GPU to control memory
        )
        
        # Move to device manually
        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing
        if self.config.get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
        
        # Resize token embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"✅ Model loaded with {self.model.num_parameters():,} parameters")
        
        # Clear memory after loading
        self.clear_memory()
    
    def tokenize_dataset(self, dataset):
        """Tokenize the dataset with memory optimization"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config['max_length'],
                return_tensors=None
            )
        
        logger.info("🔤 Tokenizing dataset...")
        
        # Process in smaller batches to save memory
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=100,  # Small batch size for tokenization
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing"
        )
        
        return tokenized_dataset
    
    def train(self, dataset_path: str = "processed_data/unified_dataset.json"):
        """Memory-efficient training"""
        # Clear memory before starting
        self.clear_memory()
        
        # Initialize wandb
        wandb.init(
            project="lex-fridman-chatbot-memory-efficient",
            name=f"memory-efficient-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=self.config
        )
        
        # Load and prepare dataset
        dataset = self.load_dataset(dataset_path)
        self.clear_memory()
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        self.clear_memory()
        
        # Tokenize dataset
        tokenized_dataset = self.tokenize_dataset(dataset)
        self.clear_memory()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=None
        )
        
        # Ultra-conservative training arguments
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
            run_name=f"lex-memory-efficient-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            logging_dir=str(self.output_dir / "logs"),
            ddp_find_unused_parameters=False,
            group_by_length=False,
            gradient_checkpointing=self.config['gradient_checkpointing'],
            max_grad_norm=self.config['max_grad_norm'],
            prediction_loss_only=self.config['prediction_loss_only'],
            eval_accumulation_steps=self.config['eval_accumulation_steps'],
            dataloader_pin_memory=self.config['dataloader_pin_memory']
        )
        
        # Memory-optimized trainer
        class UltraMemoryOptimizedTrainer(Trainer):
            def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
                """Ultra memory-optimized evaluation"""
                # Clear cache before evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                
                # Force minimal evaluation
                return super().evaluation_loop(
                    dataloader=dataloader,
                    description=description,
                    prediction_loss_only=True,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key_prefix
                )
            
            def log(self, logs):
                """Clear memory after logging"""
                super().log(logs)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Create trainer
        trainer = UltraMemoryOptimizedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=None,  # No metrics to save memory
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config['early_stopping_patience']
                )
            ]
        )
        
        # Start training
        logger.info("🚀 Starting memory-efficient training...")
        logger.info(f"📊 Training on {len(tokenized_dataset['train'])} examples")
        logger.info(f"🔍 Validating on {len(tokenized_dataset['test'])} examples")
        logger.info(f"⚡ Effective batch size: {self.config['batch_size'] * self.config['gradient_accumulation_steps']}")
        
        # Clear memory before training
        self.clear_memory()
        
        train_result = trainer.train()
        
        # Save the final model
        trainer.save_model()
        trainer.save_state()
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training summary
        training_summary = {
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
        
        with open(self.output_dir / "memory_efficient_training_summary.json", 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        logger.info("✅ Memory-efficient training complete!")
        logger.info(f"📁 Model saved to: {self.output_dir}")
        logger.info(f"📊 Final training loss: {train_result.training_loss:.4f}")
        
        # Close wandb
        wandb.finish()
        
        # Final memory cleanup
        self.clear_memory()
        
        return trainer, train_result

def main():
    """Main training execution"""
    print("🔋 Memory-Efficient Lex Fridman Chatbot Training")
    print("=" * 50)
    print("💾 Optimized for limited GPU memory")
    print("🏗️ DialoGPT-Medium with ultra-conservative settings")
    print("=" * 50)
    
    trainer = MemoryEfficientLexTrainer()
    
    try:
        model, results = trainer.train()
        print("\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {trainer.output_dir}")
        print("\n🎉 Ready to chat with your memory-efficient Lex Fridman AI!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 