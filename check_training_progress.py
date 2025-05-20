#!/usr/bin/env python3
"""
Training Progress Checker

A simple script to check the current progress of the enhanced bot training
"""

import os
import sys
import json
import glob
import psutil
import time
from datetime import datetime

def find_training_process():
    """Find the enhanced_bot_trainer.py process and get its info"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        if proc.info['cmdline'] and len(proc.info['cmdline']) > 0:
            cmd = " ".join(proc.info['cmdline'])
            if 'enhanced_bot_trainer.py' in cmd:
                # Calculate runtime
                create_time = datetime.fromtimestamp(proc.info['create_time'])
                runtime = datetime.now() - create_time
                hours, remainder = divmod(runtime.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                
                print(f"Found training process (PID: {proc.info['pid']}):")
                print(f"- Command: {cmd}")
                print(f"- Running for: {int(hours)}h {int(minutes)}m {int(seconds)}s")
                print(f"- CPU Usage: {proc.cpu_percent(interval=1.0)}%")
                print(f"- Memory Usage: {proc.memory_info().rss / (1024 * 1024):.1f} MB")
                return proc
    
    print("No enhanced_bot_trainer.py process found.")
    return None

def check_tensorflow_progress():
    """Check TensorFlow logs for training progress"""
    # Common TensorFlow log directories
    tf_log_dirs = [
        "logs/",
        "results/tensorboard/",
        "tensorboard/"
    ]
    
    latest_logs = []
    
    for log_dir in tf_log_dirs:
        if os.path.exists(log_dir):
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    if "events.out.tfevents" in file:
                        path = os.path.join(root, file)
                        latest_logs.append((os.path.getmtime(path), path))
    
    if latest_logs:
        latest_logs.sort(reverse=True)
        print("\nFound TensorFlow logs:")
        for mtime, log_path in latest_logs[:3]:  # Show 3 most recent logs
            mod_time = datetime.fromtimestamp(mtime)
            print(f"- {log_path} (Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
        
        # Try to extract metrics from TensorFlow logs
        try:
            import tensorflow as tf
            most_recent_log = latest_logs[0][1]
            print(f"\nAttempting to read metrics from: {most_recent_log}")
            
            event_acc = tf.compat.v1.train.summary_iterator(most_recent_log)
            last_step = 0
            metrics = {}
            
            for event in event_acc:
                if event.step > last_step:
                    last_step = event.step
                for value in event.summary.value:
                    if value.tag not in metrics:
                        metrics[value.tag] = []
                    metrics[value.tag].append((event.step, float(value.simple_value)))
            
            if metrics:
                print(f"Latest training step: {last_step}")
                print("\nLatest metrics:")
                for tag, values in metrics.items():
                    print(f"- {tag}: {values[-1][1]:.4f} (step {values[-1][0]})")
        except Exception as e:
            print(f"Could not extract metrics from TensorFlow logs: {str(e)}")
    else:
        print("\nNo TensorFlow logs found.")

def find_latest_checkpoint():
    """Find the latest model checkpoint"""
    checkpoint_patterns = [
        "models/*.h5",
        "models/*.keras",
        "models/*/*.h5",
        "models/*/*.keras",
        "results/*/models/*.h5",
        "results/checkpoints/*",
        "results/training/models/*"
    ]
    
    checkpoints = []
    for pattern in checkpoint_patterns:
        for file_path in glob.glob(pattern):
            if os.path.isfile(file_path):
                checkpoints.append((os.path.getmtime(file_path), file_path))
    
    if checkpoints:
        checkpoints.sort(reverse=True)
        print("\nLatest model checkpoints:")
        for mtime, file_path in checkpoints[:3]:  # Show 3 most recent
            mod_time = datetime.fromtimestamp(mtime)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"- {file_path} ({size_mb:.1f} MB, Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
    else:
        print("\nNo model checkpoints found.")

def check_training_logs():
    """Check log files for training progress"""
    log_patterns = [
        "*.log",
        "logs/*.log",
        "results/training/*.log"
    ]
    
    logs = []
    for pattern in log_patterns:
        for file_path in glob.glob(pattern):
            if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                logs.append((os.path.getmtime(file_path), file_path))
    
    if logs:
        logs.sort(reverse=True)
        print("\nLatest log files:")
        for mtime, file_path in logs[:5]:  # Show 5 most recent
            mod_time = datetime.fromtimestamp(mtime)
            size_kb = os.path.getsize(file_path) / 1024
            print(f"- {file_path} ({size_kb:.1f} KB, Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
        
        # Check the most recently modified log file with content
        recent_log = logs[0][1]
        print(f"\nChecking recent log file: {recent_log}")
        
        # Look for progress indicators in log files
        progress_indicators = ["progress", "epoch", "step", "complete", "training", "%"]
        found_progress = False
        
        with open(recent_log, 'r') as f:
            try:
                # Read the last 50 lines
                lines = f.readlines()
                recent_lines = lines[-50:] if len(lines) > 50 else lines
                
                for line in recent_lines:
                    if any(indicator in line.lower() for indicator in progress_indicators):
                        print(f"- {line.strip()}")
                        found_progress = True
                
                if not found_progress:
                    print("No clear progress indicators found in recent logs.")
            except Exception as e:
                print(f"Error reading log file: {str(e)}")
    else:
        print("\nNo log files found.")

def estimate_completion():
    """Try to estimate completion time based on available information"""
    try:
        # Check if progress_tracker.log has information
        if os.path.exists("progress_tracker.log"):
            with open("progress_tracker.log", 'r') as f:
                content = f.read()
                progress_lines = [line for line in content.split('\n') if "progress" in line.lower()]
                if progress_lines:
                    latest_progress = progress_lines[-1]
                    print(f"\nLatest recorded progress: {latest_progress}")
                    
                    # Try to extract percentage
                    import re
                    percentage_match = re.search(r"(\d+\.\d+)%", latest_progress)
                    if percentage_match:
                        percentage = float(percentage_match.group(1))
                        print(f"Progress percentage: {percentage:.1f}%")
                        
                        # Calculate estimated completion time
                        if percentage > 0:
                            # Find training process
                            for proc in psutil.process_iter(['pid', 'create_time']):
                                if proc.name() == "python" and "enhanced_bot_trainer.py" in " ".join(proc.cmdline()):
                                    create_time = datetime.fromtimestamp(proc.info['create_time'])
                                    elapsed_time = (datetime.now() - create_time).total_seconds()
                                    
                                    # Estimate remaining time
                                    if percentage < 100:
                                        remaining_seconds = (elapsed_time / percentage) * (100 - percentage)
                                        hours, remainder = divmod(remaining_seconds, 3600)
                                        minutes, seconds = divmod(remainder, 60)
                                        
                                        print(f"Estimated remaining time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
                                        print(f"Estimated completion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"Error estimating completion: {str(e)}")

def main():
    """Main function to check training progress"""
    print("=== Training Progress Check ===")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for training process
    training_process = find_training_process()
    
    # Check TensorFlow logs
    check_tensorflow_progress()
    
    # Find latest checkpoints
    find_latest_checkpoint()
    
    # Check log files
    check_training_logs()
    
    # Estimate completion
    estimate_completion()
    
    print("\n=== Progress Check Complete ===")

if __name__ == "__main__":
    main()
