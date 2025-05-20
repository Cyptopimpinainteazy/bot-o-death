#!/usr/bin/env python3
"""
Trading Data Collector

Collects and processes trading data from live operations for model training.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import argparse
import time

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
base_dir = os.path.dirname(parent_dir)
sys.path.append(base_dir)

# Set up directory paths
logs_dir = os.path.join(parent_dir, 'logs')
data_dir = os.path.join(parent_dir, 'data')

# Ensure all required directories exist
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Setup logging
log_file = os.path.join(logs_dir, f'data_collector_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataCollector")


class TradingDataCollector:
    """
    Collects and processes trading data from live operations for model training.
    """
    
    def __init__(self, data_dir=None, collection_interval=5):
        """
        Initialize the data collector
        
        Args:
            data_dir: Directory to store collected data
            collection_interval: How often to collect data (seconds)
        """
        self.data_dir = data_dir or os.path.join(parent_dir, 'data')
        self.collection_interval = collection_interval
        self.trades_data = []
        self.market_data = []
        self.performance_data = []
        self.last_collection_time = datetime.now()
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Generate unique dataset ID
        self.dataset_id = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.dataset_path = os.path.join(self.data_dir, f"{self.dataset_id}")
        os.makedirs(self.dataset_path, exist_ok=True)
        
        logger.info(f"Trading Data Collector initialized. Saving data to {self.dataset_path}")
    
    def collect_from_log_file(self, log_file_path):
        """
        Extract trading data from log files
        
        Args:
            log_file_path: Path to the log file to process
        
        Returns:
            Number of entries collected
        """
        try:
            if not os.path.exists(log_file_path):
                logger.error(f"Log file does not exist: {log_file_path}")
                return 0
            
            logger.info(f"Processing log file: {log_file_path}")
            
            # Parse log files for trade data
            trade_entries = []
            market_condition_entries = []
            with open(log_file_path, 'r') as f:
                for line in f:
                    # Extract trade execution data
                    if "Trade" in line and "executed successfully" in line:
                        try:
                            timestamp = line.split(' - ')[0]
                            trade_num = int(line.split("Trade ")[1].split(" executed")[0])
                            profit = float(line.split("$")[1].split(" profit")[0])
                            
                            trade_entries.append({
                                'timestamp': timestamp,
                                'trade_id': trade_num,
                                'profit': profit,
                                'source': os.path.basename(log_file_path)
                            })
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Error parsing trade line: {line}, Error: {str(e)}")
                    
                    # Extract execution strategy recommendations
                    elif "execution recommended" in line:
                        try:
                            timestamp = line.split(' - ')[0]
                            if "Flashloan execution recommended" in line:
                                strategy = "flashloan"
                                values = line.split("Flashloan execution recommended: ")[1].split(" vs standard: ")
                                flashloan_value = float(values[0])
                                standard_value = float(values[1].split(" (confidence:")[0])
                                confidence = float(values[1].split("(confidence: ")[1].split(")")[0])
                            else:
                                strategy = "standard"
                                values = line.split("Standard execution recommended: ")[1].split(" vs flashloan: ")
                                standard_value = float(values[0])
                                flashloan_value = float(values[1].split(" (confidence:")[0] if " (confidence:" in values[1] else values[1])
                                confidence = float(values[1].split("(confidence: ")[1].split(")")[0]) if " (confidence:" in values[1] else 0.5
                            
                            market_condition_entries.append({
                                'timestamp': timestamp,
                                'recommended_strategy': strategy,
                                'flashloan_value': flashloan_value,
                                'standard_value': standard_value,
                                'confidence': confidence,
                                'source': os.path.basename(log_file_path)
                            })
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Error parsing strategy line: {line}, Error: {str(e)}")
            
            # Save collected data to files
            self._save_trade_data(trade_entries)
            self._save_market_data(market_condition_entries)
            
            logger.info(f"Collected {len(trade_entries)} trade entries and {len(market_condition_entries)} market condition entries from {log_file_path}")
            return len(trade_entries) + len(market_condition_entries)
            
        except Exception as e:
            logger.error(f"Error collecting data from log file {log_file_path}: {str(e)}")
            return 0
    
    def collect_from_performance_file(self, performance_file_path):
        """
        Extract performance metrics from saved performance files
        
        Args:
            performance_file_path: Path to performance JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(performance_file_path):
                logger.error(f"Performance file does not exist: {performance_file_path}")
                return False
            
            logger.info(f"Processing performance file: {performance_file_path}")
            
            with open(performance_file_path, 'r') as f:
                performance_data = json.load(f)
            
            # Add source file info
            performance_data['source'] = os.path.basename(performance_file_path)
            self.performance_data.append(performance_data)
            
            # Save updated performance data
            self._save_performance_data()
            
            logger.info(f"Collected performance data from {performance_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting data from performance file {performance_file_path}: {str(e)}")
            return False
    
    def _save_trade_data(self, new_entries=None):
        """Save trade data to CSV file"""
        if new_entries:
            self.trades_data.extend(new_entries)
        
        if self.trades_data:
            df = pd.DataFrame(self.trades_data)
            csv_path = os.path.join(self.dataset_path, 'trades_data.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(self.trades_data)} trade entries to {csv_path}")
    
    def _save_market_data(self, new_entries=None):
        """Save market condition data to CSV file"""
        if new_entries:
            self.market_data.extend(new_entries)
        
        if self.market_data:
            df = pd.DataFrame(self.market_data)
            csv_path = os.path.join(self.dataset_path, 'market_data.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(self.market_data)} market condition entries to {csv_path}")
    
    def _save_performance_data(self):
        """Save performance data to JSON file"""
        if self.performance_data:
            json_path = os.path.join(self.dataset_path, 'performance_data.json')
            with open(json_path, 'w') as f:
                json.dump(self.performance_data, f, indent=4)
            logger.info(f"Saved {len(self.performance_data)} performance entries to {json_path}")
    
    def scan_and_collect_all(self):
        """
        Scan for all available log and performance files and collect data
        
        Returns:
            Total number of entries collected
        """
        total_collected = 0
        
        # Scan logs directory for trading logs
        logs_collected = 0
        for file in os.listdir(logs_dir):
            if file.endswith('.log') and 'trading' in file:
                log_path = os.path.join(logs_dir, file)
                logs_collected += self.collect_from_log_file(log_path)
        
        # Scan data directory for performance files
        perf_collected = 0
        for root, _, files in os.walk(parent_dir):
            for file in files:
                if file.endswith('.json') and 'performance' in file:
                    perf_path = os.path.join(root, file)
                    perf_collected += 1 if self.collect_from_performance_file(perf_path) else 0
        
        total_collected = logs_collected + perf_collected
        logger.info(f"Collection complete: processed {logs_collected} log entries and {perf_collected} performance files")
        
        return total_collected
    
    def prepare_training_dataset(self):
        """
        Prepare final training dataset by combining and processing all collected data
        
        Returns:
            Path to the prepared dataset
        """
        try:
            # Ensure we have the latest data saved
            self._save_trade_data()
            self._save_market_data()
            self._save_performance_data()
            
            # Create a metadata file with collection info
            metadata = {
                'dataset_id': self.dataset_id,
                'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'trade_entries': len(self.trades_data),
                'market_entries': len(self.market_data),
                'performance_entries': len(self.performance_data)
            }
            
            with open(os.path.join(self.dataset_path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # Create the training dataset file
            final_dataset_path = os.path.join(self.data_dir, f"{self.dataset_id}_training.csv")
            
            # If we have market data, use it as the base
            if self.market_data:
                market_df = pd.DataFrame(self.market_data)
                
                # Convert timestamps to datetime objects
                market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
                
                # If we also have trade data, we can try to correlate them
                if self.trades_data:
                    trades_df = pd.DataFrame(self.trades_data)
                    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                    
                    # Sort both dataframes by timestamp
                    market_df = market_df.sort_values('timestamp')
                    trades_df = trades_df.sort_values('timestamp')
                    
                    # Create a merged dataset with nearest trades
                    # This isn't perfect but gives us a starting point
                    from scipy.spatial.distance import cdist
                    
                    # Find the closest trade timestamp for each market entry
                    market_times = market_df['timestamp'].astype(int).values.reshape(-1, 1) // 10**9
                    trade_times = trades_df['timestamp'].astype(int).values.reshape(-1, 1) // 10**9
                    
                    # Find the closest trade timestamps (within 10 seconds)
                    if len(trade_times) > 0:
                        distances = cdist(market_times, trade_times)
                        closest_trades = np.argmin(distances, axis=1)
                        min_distances = np.min(distances, axis=1)
                        
                        # Only use trades that happened within 10 seconds of market entry
                        for i, (idx, dist) in enumerate(zip(closest_trades, min_distances)):
                            if dist <= 10:  # Within 10 seconds
                                market_df.loc[market_df.index[i], 'associated_profit'] = trades_df.iloc[idx]['profit']
                
                # Save the final dataset
                market_df.to_csv(final_dataset_path, index=False)
                logger.info(f"Training dataset prepared and saved to {final_dataset_path}")
                return final_dataset_path
            
            # If we only have trade data, use that
            elif self.trades_data:
                trades_df = pd.DataFrame(self.trades_data)
                trades_df.to_csv(final_dataset_path, index=False)
                logger.info(f"Trade-only dataset prepared and saved to {final_dataset_path}")
                return final_dataset_path
            
            else:
                logger.warning("No data collected, cannot prepare training dataset")
                return None
            
        except Exception as e:
            logger.error(f"Error preparing training dataset: {str(e)}")
            return None


def main():
    """Main function to run the data collector"""
    parser = argparse.ArgumentParser(description='Trading Data Collector')
    parser.add_argument('--scan', action='store_true', help='Scan and collect from all available log files')
    parser.add_argument('--log-file', type=str, help='Specific log file to process')
    parser.add_argument('--output-dir', type=str, help='Output directory for collected data')
    parser.add_argument('--wait', type=int, default=0, help='Continuously collect data every N seconds')
    args = parser.parse_args()
    
    # Initialize the collector
    collector = TradingDataCollector(data_dir=args.output_dir)
    
    if args.log_file:
        # Process a specific log file
        collector.collect_from_log_file(args.log_file)
    elif args.scan:
        # Scan for all available log files
        collector.scan_and_collect_all()
    elif args.wait > 0:
        # Continuously collect data
        logger.info(f"Starting continuous data collection every {args.wait} seconds. Press Ctrl+C to stop.")
        try:
            while True:
                collector.scan_and_collect_all()
                time.sleep(args.wait)
        except KeyboardInterrupt:
            logger.info("Data collection stopped by user")
    else:
        # Default behavior - scan once
        collector.scan_and_collect_all()
    
    # Prepare the final training dataset
    dataset_path = collector.prepare_training_dataset()
    if dataset_path:
        logger.info(f"Data collection complete. Training dataset available at: {dataset_path}")
    else:
        logger.warning("Data collection complete but no dataset was generated")


if __name__ == "__main__":
    main()
