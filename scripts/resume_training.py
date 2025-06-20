#!/usr/bin/env python3
"""
Resume Training Script for Lex Fridman Chatbot
Fixed version that resolves FP16 gradient scaling issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.enhanced_trainer import EnhancedLexTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Resume training with fixed configuration"""
    print("🔧 Fixed Lex Fridman Chatbot Training")
    print("=" * 50)
    print("✅ FP16 gradient scaling issue resolved")
    print("🎯 Using 6-transcript dataset with 645K+ words")
    print("🏗️ DialoGPT-Large with stable training configuration")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists("processed_data/unified_dataset.json"):
        print("❌ Dataset not found! Please run dataset_creator.py first.")
        return
    
    trainer = EnhancedLexTrainer()
    
    try:
        print("🚀 Starting fixed training...")
        model, results = trainer.train()
        print("\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {trainer.output_dir}")
        print("\n🎉 Ready to chat with your enhanced Lex Fridman AI!")
        
        # Display some training stats
        print(f"\n📊 Training Statistics:")
        print(f"   Final Loss: {results.training_loss:.4f}")
        print(f"   Training Time: {results.training_time:.2f} seconds")
        print(f"   Samples/Second: {results.training_samples_per_second:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        print("\n🛠️ If you continue to have issues, try:")
        print("   1. Check GPU memory with: nvidia-smi")
        print("   2. Reduce batch_size in enhanced_trainer.py")
        print("   3. Disable gradient_checkpointing if memory is sufficient")
        raise

if __name__ == "__main__":
    main() 