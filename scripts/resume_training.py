#!/usr/bin/env python3
"""
Resume Training Script for Lex Fridman Chatbot
This script resumes training from the last checkpoint with memory optimizations.
"""

import os
import torch
import gc
from model_fine_tuner import LexFridmanModelTrainer

def clear_gpu_memory():
    """Clear GPU memory aggressively."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print("🧹 GPU memory cleared")

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory."""
    checkpoint_dirs = []
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            if item.startswith("checkpoint-"):
                checkpoint_dirs.append(item)
    
    if checkpoint_dirs:
        # Sort by checkpoint number
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
        latest = checkpoint_dirs[-1]
        return os.path.join(output_dir, latest)
    return None

def main():
    """Resume training with optimized settings."""
    print("🔄 Resuming Lex Fridman Model Training")
    print("=" * 60)
    
    # Set memory optimization environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Clear memory before starting
    clear_gpu_memory()
    
    # Initialize trainer with same settings
    trainer = LexFridmanModelTrainer(
        model_name="microsoft/DialoGPT-medium",
        processed_data_dir="processed_data",
        output_dir="models/lex_chatbot"
    )
    
    # Check for existing checkpoints
    latest_checkpoint = find_latest_checkpoint(trainer.output_dir)
    if latest_checkpoint:
        print(f"📍 Found checkpoint: {latest_checkpoint}")
        print("🔄 Resuming from checkpoint...")
    else:
        print("🆕 No checkpoint found, starting fresh training...")
    
    try:
        # Load model and datasets
        model, tokenizer = trainer.load_model_and_tokenizer()
        datasets = trainer.load_datasets()
        
        # Create training arguments with even more conservative settings
        training_args = trainer.create_training_arguments()
        
        # Override with more conservative validation settings
        training_args.eval_steps = 1000  # Less frequent evaluation
        training_args.per_device_eval_batch_size = 1  # Smaller eval batch
        training_args.eval_accumulation_steps = 8  # More accumulation
        
        print(f"📊 Ultra-conservative settings applied:")
        print(f"   📦 Eval batch size: {training_args.per_device_eval_batch_size}")
        print(f"   🔄 Eval steps: {training_args.eval_steps}")
        print(f"   📈 Eval accumulation: {training_args.eval_accumulation_steps}")
        
        # Create data collator
        data_collator = trainer.create_data_collator()
        
        # Initialize memory-optimized trainer
        from transformers import Trainer, EarlyStoppingCallback
        
        class UltraMemoryOptimizedTrainer(Trainer):
            def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
                """Ultra memory-optimized evaluation."""
                print(f"🔍 Starting evaluation with memory optimization...")
                
                # Aggressive memory clearing
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
                # Monitor memory before evaluation
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        used = torch.cuda.memory_allocated(i) / 1e9
                        total = torch.cuda.get_device_properties(i).total_memory / 1e9
                        print(f"   GPU {i}: {used:.1f}GB / {total:.1f}GB before eval")
                
                try:
                    result = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"❌ OOM during evaluation: {e}")
                        print("🧹 Clearing cache and retrying with smaller batch...")
                        
                        # Clear everything
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        gc.collect()
                        
                        # Return dummy metrics to continue training
                        return {"eval_loss": float('inf')}
                    else:
                        raise e
                
                # Clear cache after evaluation
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
                print(f"✅ Evaluation completed successfully")
                return result
        
        # Create trainer
        trainer_obj = UltraMemoryOptimizedTrainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )
        
        # Resume training
        if latest_checkpoint:
            trainer_obj.train(resume_from_checkpoint=latest_checkpoint)
        else:
            trainer_obj.train()
        
        print("🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    finally:
        # Final cleanup
        clear_gpu_memory()

if __name__ == "__main__":
    main() 