import os
import json
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import load_from_disk
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Optional
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LexFridmanModelTrainer:
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 processed_data_dir: str = "processed_data",
                 output_dir: str = "models/lex_chatbot"):
        """
        Initialize the model trainer.
        
        Args:
            model_name: Base model to fine-tune
            processed_data_dir: Directory containing processed datasets
            output_dir: Directory to save the fine-tuned model
        """
        self.model_name = model_name
        self.processed_data_dir = processed_data_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Check GPU availability and setup
        self.verify_gpu_setup()
        
        # Load processing summary
        summary_path = os.path.join(processed_data_dir, "processing_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                self.processing_summary = json.load(f)
            print(f"📊 Training data: {self.processing_summary['train_examples']} examples")
        else:
            self.processing_summary = {}
    
    def verify_gpu_setup(self):
        """Verify GPU setup and configuration."""
        print(f"🚀 Lex Fridman Model Trainer Initialized")
        print(f"🔍 Checking GPU setup...")
        
        # Set memory fragmentation environment variable
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print(f"✅ Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("❌ CUDA not available! Training will be very slow on CPU.")
            self.device = "cpu"
            self.num_gpus = 0
            return
        
        self.device = "cuda"
        self.num_gpus = torch.cuda.device_count()
        
        print(f"✅ CUDA is available!")
        print(f"🚀 GPUs detected: {self.num_gpus}")
        
        # Check GPU memory and details
        total_memory = 0
        for i in range(self.num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            total_memory += gpu_memory
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Check current GPU utilization
            torch.cuda.set_device(i)
            gpu_used = torch.cuda.memory_allocated(i) / 1e9
            gpu_cached = torch.cuda.memory_reserved(i) / 1e9
            print(f"      Memory: {gpu_used:.1f}GB used, {gpu_cached:.1f}GB cached")
        
        print(f"💾 Total GPU Memory: {total_memory:.1f}GB")
        
        # Test GPU functionality
        try:
            test_tensor = torch.randn(100, 100).cuda()
            test_result = torch.matmul(test_tensor, test_tensor)
            print(f"✅ GPU functionality test passed!")
            del test_tensor, test_result
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            self.device = "cpu"
            self.num_gpus = 0
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        print(f"🤖 Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with explicit GPU configuration
        if self.device == "cuda" and self.num_gpus > 0:
            print(f"🚀 Loading model on GPU(s)...")
            
            if self.num_gpus > 1:
                # Multi-GPU setup with explicit device mapping
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,  # BFloat16 for better A100 compatibility
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                print(f"✅ Model loaded across {self.num_gpus} GPUs with device_map='auto'")
            else:
                # Single GPU setup - explicit CUDA placement
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,  # BFloat16 for better A100 compatibility
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                self.model = self.model.cuda()
                print(f"✅ Model loaded on single GPU")
        else:
            # CPU fallback
            print(f"🖥️ Loading model on CPU (warning: training will be very slow)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        
        # Resize token embeddings if necessary
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Verify model placement
        model_device = next(self.model.parameters()).device
        print(f"📍 Model loaded on device: {model_device}")
        
        # Model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Total parameters: {total_params:,}")
        print(f"🔧 Trainable parameters: {trainable_params:,}")
        
        # Check GPU memory usage after model loading
        if self.device == "cuda":
            for i in range(self.num_gpus):
                gpu_used = torch.cuda.memory_allocated(i) / 1e9
                gpu_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"💾 GPU {i} memory: {gpu_used:.1f}GB / {gpu_total:.1f}GB ({gpu_used/gpu_total*100:.1f}%)")
        
        return self.model, self.tokenizer
    
    def monitor_gpu_usage(self):
        """Monitor and print current GPU usage."""
        if self.device == "cuda" and self.num_gpus > 0:
            print(f"\n📊 GPU Usage Status:")
            for i in range(self.num_gpus):
                gpu_used = torch.cuda.memory_allocated(i) / 1e9
                gpu_cached = torch.cuda.memory_reserved(i) / 1e9
                gpu_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                utilization = gpu_used / gpu_total * 100
                
                print(f"   GPU {i}: {gpu_used:.1f}GB used / {gpu_total:.1f}GB total ({utilization:.1f}%)")
                print(f"           {gpu_cached:.1f}GB cached")
        else:
            print(f"❌ No GPUs detected or CUDA not available!")
    
    def load_datasets(self):
        """Load the processed datasets."""
        dataset_path = os.path.join(self.processed_data_dir, "tokenized_datasets")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Processed datasets not found at {dataset_path}")
        
        print(f"📂 Loading datasets from: {dataset_path}")
        datasets = load_from_disk(dataset_path)
        
        print(f"📊 Dataset sizes:")
        print(f"   🚂 Training: {len(datasets['train'])} examples")
        print(f"   ✅ Validation: {len(datasets['validation'])} examples")
        
        return datasets
    
    def create_training_arguments(self) -> TrainingArguments:
        """Create optimized training arguments for A100 GPUs."""
        
        # Calculate batch sizes based on available GPUs
        # Very conservative batch sizes to avoid validation memory issues
        if self.num_gpus >= 2:
            per_device_batch_size = 2  # Very conservative for dual A100s
            gradient_accumulation_steps = 8
        else:
            per_device_batch_size = 1
            gradient_accumulation_steps = 16
        
        effective_batch_size = per_device_batch_size * gradient_accumulation_steps * max(1, self.num_gpus)
        print(f"📊 Effective batch size: {effective_batch_size}")
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            
            # Training schedule
            num_train_epochs=5,  # More epochs for better fine-tuning
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            
            # Learning rate and optimization
            learning_rate=3e-5,  # Good for fine-tuning
            warmup_steps=100,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            
            # Evaluation and saving
            eval_strategy="steps",  # Updated parameter name
            eval_steps=500,  # Reduced frequency to save memory
            save_strategy="steps",
            save_steps=1000,  # Reduced frequency
            save_total_limit=3,  # Keep only best 3 checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Logging
            logging_steps=50,
            logging_dir=os.path.join(self.output_dir, "logs"),
            report_to=None,  # Disable wandb for now
            
            # Performance optimizations
            bf16=True,  # BFloat16 for A100 (better than fp16)
            dataloader_drop_last=True,
            gradient_checkpointing=True,  # Save memory
            dataloader_num_workers=2,  # Reduced for memory
            remove_unused_columns=True,
            
            # Mixed precision settings
            fp16_full_eval=False,
            dataloader_pin_memory=False,  # Disable to save memory
            
            # Memory optimization settings
            max_grad_norm=1.0,
            dataloader_persistent_workers=False,
            eval_accumulation_steps=4,  # Accumulate eval batches to save memory
            
            # Multi-GPU settings
            ddp_find_unused_parameters=False if self.num_gpus > 1 else None,
        )
        
        return training_args
    
    def create_data_collator(self):
        """Create data collator for language modeling."""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8 if self.device == "cuda" else None,
        )
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        
        # Calculate perplexity
        losses = []
        for i in range(len(predictions)):
            # Shift labels for causal LM
            shift_labels = labels[i][1:].tolist()
            shift_logits = predictions[i][:-1]
            
            # Calculate loss
            loss = F.cross_entropy(
                torch.tensor(shift_logits), 
                torch.tensor(shift_labels), 
                ignore_index=-100
            )
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        perplexity = np.exp(avg_loss)
        
        return {
            "perplexity": perplexity,
            "eval_loss": avg_loss
        }
    
    def train_model(self):
        """Main training function."""
        print("🚀 Starting Model Fine-tuning...")
        print("=" * 60)
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Load datasets
        datasets = self.load_datasets()
        
        # Create training arguments
        training_args = self.create_training_arguments()
        
        # Create data collator
        data_collator = self.create_data_collator()
        
        # Initialize trainer with memory optimization
        class MemoryOptimizedTrainer(Trainer):
            def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
                """Override evaluation to clear cache between batches."""
                # Clear cache before evaluation
                torch.cuda.empty_cache()
                gc.collect()
                
                result = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
                
                # Clear cache after evaluation
                torch.cuda.empty_cache()
                gc.collect()
                
                return result
        
        trainer = MemoryOptimizedTrainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Print training info
        print(f"📊 Training Configuration:")
        print(f"   🎯 Model: {self.model_name}")
        print(f"   📊 Training examples: {len(datasets['train'])}")
        print(f"   ✅ Validation examples: {len(datasets['validation'])}")
        print(f"   🔄 Epochs: {training_args.num_train_epochs}")
        print(f"   📦 Batch size per device: {training_args.per_device_train_batch_size}")
        print(f"   🚀 GPUs: {self.num_gpus}")
        print(f"   💾 Output: {self.output_dir}")
        
        # Monitor GPU before training
        self.monitor_gpu_usage()
        
        # Clear any existing cache before training
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            print("🧹 Cleared GPU cache before training")
        
        # Start training
        print(f"\n🏃‍♂️ Starting training...")
        training_start = datetime.now()
        
        try:
            train_result = trainer.train()
            
            # Training completed successfully
            training_end = datetime.now()
            training_duration = training_end - training_start
            
            print(f"\n🎉 Training completed!")
            print(f"⏱️  Training time: {training_duration}")
            print(f"📉 Final training loss: {train_result.training_loss:.4f}")
            
            # Save the final model
            print(f"💾 Saving final model...")
            trainer.save_model()
            tokenizer.save_pretrained(self.output_dir)
            
            # Save training summary
            training_summary = {
                "model_name": self.model_name,
                "training_loss": train_result.training_loss,
                "training_duration": str(training_duration),
                "num_epochs": training_args.num_train_epochs,
                "num_gpus": self.num_gpus,
                "training_examples": len(datasets['train']),
                "validation_examples": len(datasets['validation']),
                "completed_at": training_end.isoformat(),
                "output_dir": self.output_dir
            }
            
            summary_path = os.path.join(self.output_dir, "training_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(training_summary, f, indent=2)
            
            print(f"📋 Training summary saved to: {summary_path}")
            
            # Test the model
            self.test_model()
            
            return trainer
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
        
        finally:
            # Cleanup
            torch.cuda.empty_cache()
            gc.collect()
    
    def test_model(self):
        """Test the trained model with sample inputs."""
        print(f"\n🧪 Testing the trained model...")
        
        # Load the saved model
        model = AutoModelForCausalLM.from_pretrained(self.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        
        if self.device == "cuda":
            model = model.to(self.device)
        
        model.eval()
        
        # Test prompts
        test_prompts = [
            "Human: What is artificial intelligence?\n\nLex:",
            "Human: How do you think about the future of technology?\n\nLex:",
            "Human: What's your view on consciousness?\n\nLex:",
        ]
        
        print(f"🎯 Sample responses from your Lex Fridman chatbot:")
        print("-" * 60)
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n💬 Test {i+1}:")
            print(f"Prompt: {prompt}")
            
            # Generate response
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part
            generated_part = response[len(prompt):].strip()
            print(f"Response: {generated_part}")
            print("-" * 40)

def main():
    """Main training execution."""
    print("🚀 Lex Fridman Chatbot Model Trainer")
    print("Using Transformer Fine-tuning with GPU Acceleration")
    print("=" * 60)
    
    # Initialize trainer
    trainer = LexFridmanModelTrainer(
        model_name="microsoft/DialoGPT-medium",
        processed_data_dir="processed_data",
        output_dir="models/lex_chatbot"
    )
    
    # Check if we have the required data
    dataset_path = os.path.join(trainer.processed_data_dir, "tokenized_datasets")
    if not os.path.exists(dataset_path):
        print("❌ Error: Processed datasets not found!")
        print("Please run the tokenizer_and_preprocessor.py script first.")
        return
    
    try:
        # Start training
        trained_model = trainer.train_model()
        
        print(f"\n🎉 SUCCESS! Your Lex Fridman chatbot is ready!")
        print("=" * 60)
        print(f"📂 Model saved in: {trainer.output_dir}")
        print(f"🎯 Next step: Create the web interface!")
        print(f"💡 You can now use this model to chat like Lex Fridman!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print(f"💡 Check GPU memory and data availability")

if __name__ == "__main__":
    main() 