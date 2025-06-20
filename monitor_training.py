#!/usr/bin/env python3
"""
Real-time Training Monitor for Lex Fridman Chatbot
Displays current training progress, GPU usage, and metrics
"""

import os
import time
import subprocess
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np

class TrainingMonitor:
    def __init__(self):
        self.losses = deque(maxlen=100)
        self.steps = deque(maxlen=100)
        self.gpu_utils = deque(maxlen=50)
        self.timestamps = deque(maxlen=100)
        
        # MLflow run directory (latest)
        self.mlflow_dir = "mlruns/0/6e8009f8c862456fa316f782e6293731"
        
    def get_latest_metrics(self):
        """Get the latest training metrics from MLflow"""
        try:
            # Read latest loss
            loss_file = f"{self.mlflow_dir}/metrics/loss"
            if os.path.exists(loss_file):
                with open(loss_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        parts = last_line.split()
                        if len(parts) >= 3:
                            timestamp, loss, step = parts[0], float(parts[1]), int(parts[2])
                            return {'loss': loss, 'step': step, 'timestamp': int(timestamp)}
            
            return None
        except Exception as e:
            print(f"Error reading metrics: {e}")
            return None
    
    def get_gpu_status(self):
        """Get current GPU utilization"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpus = []
                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpus.append({
                            'util': int(parts[0]),
                            'mem_used': int(parts[1]),
                            'mem_total': int(parts[2]),
                            'power': float(parts[3])
                        })
                return gpus
            return []
        except Exception as e:
            print(f"Error getting GPU status: {e}")
            return []
    
    def get_training_process_info(self):
        """Get training process information"""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'model_fine_tuner.py' in line and 'python' in line:
                        parts = line.split()
                        if len(parts) >= 11:
                            return {
                                'pid': parts[1],
                                'cpu': parts[2],
                                'mem': parts[3],
                                'time': parts[9]
                            }
            return None
        except Exception as e:
            print(f"Error getting process info: {e}")
            return None
    
    def display_status(self):
        """Display current training status"""
        os.system('clear')
        print("🚀 LEX FRIDMAN CHATBOT - TRAINING MONITOR")
        print("=" * 60)
        print(f"⏰ Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Training Process Info
        process_info = self.get_training_process_info()
        if process_info:
            print("📊 TRAINING PROCESS:")
            print(f"   PID: {process_info['pid']}")
            print(f"   CPU Usage: {process_info['cpu']}%")
            print(f"   Memory: {process_info['mem']}%")
            print(f"   Runtime: {process_info['time']}")
            print()
        
        # Latest Metrics
        metrics = self.get_latest_metrics()
        if metrics:
            print("📈 TRAINING METRICS:")
            print(f"   Current Step: {metrics['step']}")
            print(f"   Current Loss: {metrics['loss']:.4f}")
            
            # Calculate epoch progress (assuming 5 epochs, ~387 steps per epoch)
            total_steps = 5 * 387  # Approximate
            current_epoch = (metrics['step'] // 387) + 1
            epoch_progress = (metrics['step'] % 387) / 387 * 100
            overall_progress = metrics['step'] / total_steps * 100
            
            print(f"   Current Epoch: {current_epoch}/5")
            print(f"   Epoch Progress: {epoch_progress:.1f}%")
            print(f"   Overall Progress: {overall_progress:.1f}%")
            print()
        
        # GPU Status
        gpus = self.get_gpu_status()
        if gpus:
            print("🔥 GPU STATUS:")
            for i, gpu in enumerate(gpus):
                mem_percent = (gpu['mem_used'] / gpu['mem_total']) * 100
                print(f"   GPU {i}: {gpu['util']}% util | {gpu['mem_used']}/{gpu['mem_total']}MB ({mem_percent:.1f}%) | {gpu['power']:.1f}W")
            print()
        
        # Progress Bar
        if metrics:
            progress = overall_progress
            bar_length = 50
            filled_length = int(bar_length * progress / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"🎯 OVERALL PROGRESS: [{bar}] {progress:.1f}%")
            print()
        
        # Recent Loss Trend
        if len(self.losses) > 1:
            recent_losses = list(self.losses)[-10:]
            if len(recent_losses) >= 2:
                trend = "📈 Rising" if recent_losses[-1] > recent_losses[-2] else "📉 Falling"
                print(f"📊 LOSS TREND (last 10 steps): {trend}")
                print(f"   Recent losses: {[f'{l:.3f}' for l in recent_losses[-5:]]}")
                print()
        
        print("💡 TensorBoard: http://localhost:6006 (if running)")
        print("⌨️  Press Ctrl+C to stop monitoring")
        print("=" * 60)
    
    def run(self, refresh_interval=10):
        """Run the monitoring loop"""
        print("Starting training monitor...")
        print("TensorBoard should be available at http://localhost:6006")
        
        try:
            while True:
                # Get latest metrics
                metrics = self.get_latest_metrics()
                if metrics:
                    self.losses.append(metrics['loss'])
                    self.steps.append(metrics['step'])
                    self.timestamps.append(metrics['timestamp'])
                
                # Get GPU utilization
                gpus = self.get_gpu_status()
                if gpus:
                    avg_util = sum(gpu['util'] for gpu in gpus) / len(gpus)
                    self.gpu_utils.append(avg_util)
                
                # Display status
                self.display_status()
                
                # Wait for next update
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Error in monitoring loop: {e}")

if __name__ == "__main__":
    monitor = TrainingMonitor()
    monitor.run(refresh_interval=15)  # Update every 15 seconds 