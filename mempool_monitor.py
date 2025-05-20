import os
import time
import json
from web3 import Web3
from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MempoolMonitor")

# Load environment variables
load_dotenv()

class MempoolMonitor:
    """
    Real-time blockchain mempool monitoring system.
    Monitors pending transactions and identifies potential trading opportunities.
    """
    
    def __init__(self, chain="polygon"):
        """Initialize the mempool monitor for a specific chain"""
        self.chain = chain
        self.monitoring = False
        self.transaction_cache = {}  # Cache of recent transactions
        self.opportunity_callbacks = []  # Callbacks for when opportunities are found
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.lock = threading.Lock()
        
        # Configure Web3 connection based on chain
        alchemy_key = os.getenv('ALCHEMY_API_KEY')
        if chain == "polygon":
            self.w3 = Web3(Web3.HTTPProvider(f"https://polygon-mainnet.g.alchemy.com/v2/{alchemy_key}"))
        elif chain == "ethereum":
            self.w3 = Web3(Web3.HTTPProvider(f"https://eth-mainnet.g.alchemy.com/v2/{alchemy_key}"))
        else:
            self.w3 = Web3(Web3.HTTPProvider("https://bsc-dataseed.binance.org/"))
        
        # Define target contract addresses to monitor
        self.target_contracts = {
            # Add DEX router addresses
            "uniswap": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            "sushiswap": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
            "quickswap": "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff",
        }
        
        # Known method signatures for swaps
        self.swap_signatures = [
            "0x38ed1739",  # swapExactTokensForTokens
            "0x7ff36ab5",  # swapExactETHForTokens
            "0x18cbafe5",  # swapExactTokensForETH
            "0x5c11d795",  # swapExactTokensForTokensSupportingFeeOnTransferTokens
            "0xfb3bdb41",  # swapETHForExactTokens
            "0x4a25d94a",  # swapTokensForExactETH
            "0x8803dbee"   # swapTokensForExactTokens
        ]
        
        logger.info(f"Mempool monitor initialized for {chain}")
    
    def register_opportunity_callback(self, callback):
        """Register a callback function for when trading opportunities are found"""
        self.opportunity_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start monitoring the mempool"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info(f"Mempool monitoring started on {self.chain}")
    
    def stop_monitoring(self):
        """Stop monitoring the mempool"""
        self.monitoring = False
        logger.info("Mempool monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Get pending transactions from mempool
                pending_tx_count = self.w3.eth.get_block_transaction_count('pending')
                pending_block = self.w3.eth.get_block('pending', full_transactions=True)
                
                if pending_block and 'transactions' in pending_block:
                    # Process each transaction
                    for tx in pending_block['transactions']:
                        self.executor.submit(self._process_transaction, tx)
                
                logger.debug(f"Processed {pending_tx_count} pending transactions")
                
                # Clean old transactions from cache
                self._clean_transaction_cache()
                
                # Sleep to prevent excessive API calls
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in mempool monitoring: {e}")
                time.sleep(5)  # Sleep longer on error
    
    def _process_transaction(self, tx):
        """Process a single transaction to identify potential opportunities"""
        # Skip if we've seen this transaction before
        tx_hash = tx.get('hash', '').hex()
        if tx_hash in self.transaction_cache:
            return
        
        # Add to cache with timestamp
        with self.lock:
            self.transaction_cache[tx_hash] = {
                'timestamp': time.time(),
                'processed': False
            }
        
        try:
            # Check if transaction is to a target contract
            to_address = tx.get('to', '')
            if not to_address:
                return
                
            is_target = False
            for name, address in self.target_contracts.items():
                if to_address.lower() == address.lower():
                    is_target = True
                    break
            
            if not is_target:
                return
                
            # Check if this is a swap transaction
            input_data = tx.get('input', '')
            method_signature = input_data[:10] if len(input_data) >= 10 else ''
            
            if method_signature not in self.swap_signatures:
                return
                
            # Extract transaction details
            gas_price = tx.get('gasPrice', 0)
            value = tx.get('value', 0)
            
            # This is a swap transaction to a target contract
            opportunity = {
                'tx_hash': tx_hash,
                'to_address': to_address,
                'from_address': tx.get('from', ''),
                'method_signature': method_signature,
                'gas_price': gas_price,
                'value': value,
                'chain': self.chain,
                'timestamp': time.time(),
                'input_data': input_data
            }
            
            # Try to decode input data for more details
            try:
                # This is simplified - in production you'd need the full ABI
                if method_signature == "0x38ed1739":  # swapExactTokensForTokens
                    # Extract token addresses and amounts
                    opportunity['type'] = "swapExactTokensForTokens"
                    # Further decoding would be done here
            except Exception as e:
                logger.debug(f"Could not decode input data: {e}")
            
            # Call registered callbacks with the opportunity
            for callback in self.opportunity_callbacks:
                try:
                    callback(opportunity)
                except Exception as e:
                    logger.error(f"Error in opportunity callback: {e}")
            
            # Mark as processed
            with self.lock:
                if tx_hash in self.transaction_cache:
                    self.transaction_cache[tx_hash]['processed'] = True
            
            logger.info(f"Found potential trading opportunity: {tx_hash}")
            
        except Exception as e:
            logger.error(f"Error processing transaction {tx_hash}: {e}")
    
    def _clean_transaction_cache(self):
        """Clean old transactions from the cache"""
        current_time = time.time()
        with self.lock:
            to_delete = []
            for tx_hash, data in self.transaction_cache.items():
                # Remove transactions older than 5 minutes
                if current_time - data['timestamp'] > 300:
                    to_delete.append(tx_hash)
            
            for tx_hash in to_delete:
                del self.transaction_cache[tx_hash]
    
    def get_stats(self):
        """Get current monitoring stats"""
        with self.lock:
            total_tx = len(self.transaction_cache)
            processed_tx = sum(1 for data in self.transaction_cache.values() if data['processed'])
        
        return {
            "chain": self.chain,
            "monitoring": self.monitoring,
            "total_transactions": total_tx,
            "processed_transactions": processed_tx,
            "uptime": time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }


# Example usage
if __name__ == "__main__":
    # Define a callback function for opportunities
    def opportunity_handler(opportunity):
        print(f"Found opportunity: {opportunity['tx_hash']}")
    
    # Create and start monitor
    monitor = MempoolMonitor(chain="polygon")
    monitor.register_opportunity_callback(opportunity_handler)
    monitor.start_monitoring()
    
    try:
        # Run for a while
        time.sleep(60)
    finally:
        monitor.stop_monitoring()
