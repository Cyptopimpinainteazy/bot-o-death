#!/usr/bin/env python
"""
Training Process Monitor
-----------------------
This script monitors the progress of quantum trading model training
and provides periodic performance metrics.
"""

import os
import time
import glob
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TrainingMonitor")

def find_latest_metrics_file():
    """Find the most recent metrics file"""
    metrics_dir = "results/metrics"
    if not os.path.exists(metrics_dir):
        return None
        
    files = glob.glob(f"{metrics_dir}/*.json")
    if not files:
        return None
        
    # Sort by modification time, newest first
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def find_running_processes():
    """Find Python processes that might be training models"""
    import subprocess
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    lines = result.stdout.splitlines()
    
    python_processes = []
    for line in lines:
        if "python" in line and ("train" in line or "bot" in line or "rl" in line):
            python_processes.append(line)
    
    return python_processes

def check_models_directory():
    """Check for new models in the models directory"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
        
    model_files = glob.glob(f"{models_dir}/*.zip")
    return sorted(model_files, key=os.path.getmtime)

def generate_profit_chart(metrics_files):
    """Generate a chart showing profit trends over time"""
    if not metrics_files:
        return
        
    timestamps = []
    returns = []
    win_rates = []
    
    for file in metrics_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                
            if 'avg_return_pct' in data:
                returns.append(data['avg_return_pct'])
                win_rates.append(data.get('win_rate', 0))
                timestamps.append(os.path.getmtime(file))
        except Exception as e:
            logger.error(f"Error processing file {file}: {str(e)}")
    
    if not timestamps:
        return
        
    plt.figure(figsize=(12, 6))
    
    # Plot returns
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, returns, 'b-', label='Avg Return %')
    plt.title('Trading Performance Over Time')
    plt.ylabel('Return %')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot win rate
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, win_rates, 'g-', label='Win Rate %')
    plt.xlabel('Time')
    plt.ylabel('Win Rate %')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis as dates
    from matplotlib.dates import DateFormatter
    date_format = DateFormatter('%H:%M:%S')
    plt.gca().xaxis.set_major_formatter(date_format)
    
    # Save the figure
    plt.tight_layout()
    plots_dir = "results/plots"
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f"{plots_dir}/training_progress_{timestamp}.png")
    logger.info(f"Progress chart saved to {plots_dir}/training_progress_{timestamp}.png")

def main():
    """Monitor training progress periodically"""
    logger.info("Starting training monitor")
    
    # Initial state
    last_metrics_file = find_latest_metrics_file()
    known_model_files = check_models_directory()
    last_check_time = time.time()
    
    # Create progress data collection
    evaluation_files = []
    
    logger.info(f"Initially found {len(known_model_files)} model files")
    
    try:
        while True:
            # Find running processes
            processes = find_running_processes()
            if not processes:
                logger.warning("No active training processes found")
            else:
                logger.info(f"Found {len(processes)} active training processes")
                for proc in processes:
                    logger.info(f"Active process: {proc.strip()}")
            
            # Check for new models
            current_model_files = check_models_directory()
            new_models = [f for f in current_model_files if f not in known_model_files]
            if new_models:
                logger.info(f"Found {len(new_models)} new models:")
                for model in new_models:
                    logger.info(f"  - {os.path.basename(model)}")
                known_model_files = current_model_files
            
            # Check for new metrics
            current_metrics_file = find_latest_metrics_file()
            if current_metrics_file != last_metrics_file:
                logger.info(f"New metrics file found: {os.path.basename(current_metrics_file)}")
                try:
                    with open(current_metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    # Log important metrics
                    if 'avg_return_pct' in metrics:
                        logger.info(f"Average Return: {metrics['avg_return_pct']:.2f}%")
                    if 'win_rate' in metrics:
                        logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
                        
                    evaluation_files.append(current_metrics_file)
                except Exception as e:
                    logger.error(f"Error reading metrics file: {str(e)}")
                
                last_metrics_file = current_metrics_file
                
                # Generate progress chart if we have multiple evaluation files
                if len(evaluation_files) >= 2:
                    generate_profit_chart(evaluation_files)
            
            # Sleep for a while
            time.sleep(60)  # Check every minute
    
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user")
    except Exception as e:
        logger.error(f"Error in monitor: {str(e)}")
    finally:
        logger.info("Generating final progress chart")
        generate_profit_chart(evaluation_files)
        logger.info("Monitor shutting down")

if __name__ == "__main__":
    main()
