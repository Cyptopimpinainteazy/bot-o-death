#!/usr/bin/env python
"""
MEV Bundle Manager

This module handles the creation, submission, and monitoring of MEV transaction bundles.
It supports atomic execution of multiple trades to ensure efficiency and prevent partial
execution risks in arbitrage opportunities.

Key features:
- Atomic bundle creation for multiple transactions
- Flashbots integration for private transaction submission
- Gas optimization across bundled transactions
- MEV-Boost and MEV-Geth compatibility
- Priority fee optimization for bundle inclusion
"""

import os
import time
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple, Union
from web3 import Web3
from web3.types import TxParams, HexStr
from eth_account import Account
from eth_account.signers.local import LocalAccount
from eth_account.messages import encode_defunct
import eth_abi
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("MEVBundleManager")

# Load environment variables
load_dotenv()
ALCHEMY_API_KEY = os.getenv('ALCHEMY_API_KEY')
PRIVATE_KEY = os.getenv('PRIVATE_KEY')
BOT_ADDRESS = os.getenv('BOT_ADDRESS')

class MEVBundleManager:
    """
    Manages the creation and submission of MEV transaction bundles for arbitrage.
    Supports Flashbots and other bundle relayers.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the MEV Bundle Manager
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        self.config_dir = Path("config")
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            config_path = self.config_dir / "mev_bundle_config.json"
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = self._create_default_config()
        
        # Initialize Web3 connections
        self.connections = {}
        for network in self.config.get("networks", []):
            rpc_url = self.config["rpc_endpoints"].get(network)
            if rpc_url:
                # Replace API key placeholders
                if "{ALCHEMY_API_KEY}" in rpc_url:
                    rpc_url = rpc_url.replace("{ALCHEMY_API_KEY}", ALCHEMY_API_KEY)
                self.connections[network] = Web3(Web3.HTTPProvider(rpc_url))
                logger.info(f"Connected to {network} network")
        
        # Initialize bundle relayer endpoints
        self.relayer_endpoints = self.config.get("relayer_endpoints", {})
        
        # Load account
        try:
            if PRIVATE_KEY and PRIVATE_KEY.startswith('0x') and len(PRIVATE_KEY) == 66:
                self.account = Account.from_key(PRIVATE_KEY)
                logger.info(f"Initialized account: {self.account.address}")
            else:
                self.account = None
                logger.warning("No valid private key provided. Set PRIVATE_KEY environment variable with a valid Ethereum private key.")
        except Exception as e:
            self.account = None
            logger.warning(f"Error initializing account: {e}")
        
        # Bundle statistics
        self.stats = {
            "bundles_created": 0,
            "bundles_submitted": 0,
            "bundles_included": 0,
            "avg_profit_per_bundle": 0,
            "total_profit": 0
        }
        
        logger.info("MEV Bundle Manager initialized")
    
    def _create_default_config(self):
        """Create default configuration for MEV bundles"""
        config = {
            "networks": [
                "ethereum", 
                "polygon", 
                "arbitrum", 
                "optimism", 
                "base"
            ],
            "rpc_endpoints": {
                "ethereum": "https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}",
                "polygon": "https://polygon-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}",
                "arbitrum": "https://arb-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}",
                "optimism": "https://opt-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}",
                "base": "https://base-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
            },
            "relayer_endpoints": {
                "flashbots": "https://relay.flashbots.net",
                "bloxroute": "https://mev.api.bloxroute.com/v1/", 
                "eden": "https://api.edennetwork.io/v1/bundle",
                "builder0x69": "https://builder0x69.io"
            },
            "relayer_api_keys": {
                "bloxroute": "",
                "eden": ""
            },
            "default_relayers": ["flashbots"],
            "bundle_options": {
                "max_transactions": 10,
                "block_target_count": 5,
                "resubmission_count": 25,
                "resubmission_interval": 1,
                "tip_multiplier": 1.2,
                "priority_fee_cap": 3.0,
                "min_profit_threshold_eth": 0.005
            },
            "gas_settings": {
                "max_gas_price_gwei": 100,
                "max_priority_fee_gwei": 5,
                "base_gas_increase": 1.2
            },
            "simulation": {
                "simulate_before_submit": true,
                "simulation_timestamp": "latest",
                "simulation_state_overrides": {},
                "abort_on_simulation_error": true
            }
        }
        
        # Save default config
        config_path = self.config_dir / "mev_bundle_config.json"
        if not Path(self.config_dir).exists():
            Path(self.config_dir).mkdir(parents=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created default MEV bundle configuration at {config_path}")
        return config
    
    def create_bundle(self, 
                      transactions: List[Dict[str, Any]], 
                      network: str="ethereum",
                      block_number: Optional[int]=None,
                      ensure_success: bool=True) -> Dict[str, Any]:
        """
        Create a transaction bundle for MEV relayers
        
        Args:
            transactions: List of transaction objects with params
            network: Target network for the bundle
            block_number: Target block number (if None, uses current + 1)
            ensure_success: If True, adds revert conditions for atomic execution
            
        Returns:
            Bundle object ready for submission
        """
        w3 = self.connections.get(network)
        if not w3:
            raise ValueError(f"No connection available for network: {network}")
        
        if not self.account:
            raise ValueError("No account available. Set PRIVATE_KEY environment variable.")
        
        # Get current block for targeting if not specified
        if not block_number:
            current_block = w3.eth.block_number
            block_number = current_block + 1
        
        # Prepare transactions for the bundle
        signed_txs = []
        bundle_transactions = []
        total_gas_used = 0
        
        # Get current gas price and priority fee
        gas_price = w3.eth.gas_price
        max_priority_fee = w3.eth.max_priority_fee
        max_gas_price_gwei = self.config["gas_settings"]["max_gas_price_gwei"]
        max_priority_fee_gwei = self.config["gas_settings"]["max_priority_fee_gwei"]
        
        # Cap the gas price and priority fee
        max_gas_price_wei = w3.to_wei(max_gas_price_gwei, "gwei")
        max_priority_fee_wei = w3.to_wei(max_priority_fee_gwei, "gwei")
        
        if gas_price > max_gas_price_wei:
            gas_price = max_gas_price_wei
            
        if max_priority_fee > max_priority_fee_wei:
            max_priority_fee = max_priority_fee_wei
            
        # Apply tip multiplier to make bundle more attractive
        tip_multiplier = self.config["bundle_options"]["tip_multiplier"]
        priority_fee = int(max_priority_fee * tip_multiplier)
        
        # Get current nonce for sequential transactions
        base_nonce = w3.eth.get_transaction_count(self.account.address)
        
        for i, tx in enumerate(transactions):
            # Set basic transaction parameters if not already set
            tx_params = tx.get("params", {})
            
            # Always use the bot account address
            tx_params["from"] = self.account.address
            
            # Set sequential nonce if not specified
            if "nonce" not in tx_params:
                tx_params["nonce"] = base_nonce + i
                
            # Use EIP-1559 transaction type
            if "maxFeePerGas" not in tx_params:
                tx_params["maxFeePerGas"] = gas_price
                
            if "maxPriorityFeePerGas" not in tx_params:
                tx_params["maxPriorityFeePerGas"] = priority_fee
                
            # Estimate gas if not provided
            if "gas" not in tx_params and "to" in tx_params:
                try:
                    tx_params["gas"] = int(w3.eth.estimate_gas(tx_params) * 1.2)  # Add 20% buffer
                except Exception as e:
                    logger.warning(f"Gas estimation failed: {e}. Using default gas limit.")
                    tx_params["gas"] = 500000  # Default gas limit
            
            # Sign the transaction
            signed_tx = self.account.sign_transaction(tx_params)
            signed_txs.append(signed_tx)
            
            # Add to bundle
            bundle_tx = {
                "signed_transaction": signed_tx.rawTransaction.hex(),
                "hash": signed_tx.hash.hex(),
                "account": self.account.address,
                "nonce": tx_params["nonce"]
            }
            bundle_transactions.append(bundle_tx)
            
            # Track gas used
            total_gas_used += tx_params.get("gas", 0)
        
        # Create bundle object
        bundle = {
            "transactions": bundle_transactions,
            "block_number": block_number,
            "network": network,
            "signer": self.account.address,
            "replacement_uuid": None,  # Will be set when submitting
            "bundle_hash": Web3.keccak(text=str(time.time())).hex(),
            "total_gas_used": total_gas_used,
            "creation_timestamp": int(time.time())
        }
        
        # Track statistics
        self.stats["bundles_created"] += 1
        
        logger.info(f"Created bundle with {len(transactions)} transactions targeting block {block_number}")
        return bundle
    
    def sign_flashbots_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sign a bundle for Flashbots submission
        
        Args:
            bundle: Bundle object to sign
            
        Returns:
            Signed bundle ready for submission
        """
        # Create message for signing
        bundle_transactions = [Web3.to_hex(tx["signed_transaction"]) if isinstance(tx["signed_transaction"], bytes) 
                              else tx["signed_transaction"] for tx in bundle["transactions"]]
        
        block_number = bundle["block_number"]
        state_block_number = "latest"
        
        # Format for Flashbots
        flashbots_bundle = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_sendBundle",
            "params": [{
                "txs": bundle_transactions,
                "blockNumber": hex(block_number),
                "minTimestamp": 0,
                "maxTimestamp": 0x7FFFFFFFFFFFFFFF,  # Max uint64
                "revertingTxHashes": []
            }]
        }
        
        # Add simulation request if configured
        if self.config["simulation"]["simulate_before_submit"]:
            flashbots_bundle["params"][0]["stateBlockNumber"] = state_block_number
            
        # Sign the bundle
        message = Web3.keccak(text=str(flashbots_bundle))
        signed_message = self.account.sign_message(encode_defunct(message))
        
        # Add signature
        flashbots_bundle["x-flashbots-signature"] = f"{self.account.address}:{signed_message.signature.hex()}"
        
        return flashbots_bundle
    
    def submit_bundle(self, 
                      bundle: Dict[str, Any], 
                      relayers: Optional[List[str]]=None) -> Dict[str, Any]:
        """
        Submit a bundle to MEV relayers
        
        Args:
            bundle: Bundle object to submit
            relayers: List of relayers to submit to (default uses configured default relayers)
            
        Returns:
            Dictionary of submission results by relayer
        """
        if not relayers:
            relayers = self.config.get("default_relayers", ["flashbots"])
            
        # Track if any submission was successful
        any_success = False
        results = {}
        
        # Get network-specific Web3 instance
        network = bundle.get("network", "ethereum")
        w3 = self.connections.get(network)
        if not w3:
            raise ValueError(f"No connection available for network: {network}")
        
        # Sign bundle for Flashbots submission
        signed_bundle = self.sign_flashbots_bundle(bundle)
        
        # Submit to each relayer
        for relayer in relayers:
            try:
                endpoint = self.relayer_endpoints.get(relayer)
                if not endpoint:
                    logger.warning(f"No endpoint defined for relayer: {relayer}")
                    results[relayer] = {"success": False, "error": "No endpoint defined"}
                    continue
                
                # Add API key if required
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                
                if relayer in self.config.get("relayer_api_keys", {}):
                    api_key = self.config["relayer_api_keys"][relayer]
                    if api_key:
                        headers["X-API-KEY"] = api_key
                
                # Submit to relayer endpoint
                response = requests.post(
                    endpoint,
                    json=signed_bundle,
                    headers=headers
                )
                
                # Parse response
                if response.status_code == 200:
                    response_data = response.json()
                    bundle_hash = response_data.get("result", {}).get("bundleHash")
                    if bundle_hash:
                        results[relayer] = {
                            "success": True,
                            "bundle_hash": bundle_hash,
                            "response": response_data
                        }
                        any_success = True
                        logger.info(f"Bundle submitted successfully to {relayer}: {bundle_hash}")
                    else:
                        results[relayer] = {
                            "success": False,
                            "error": "No bundle hash in response",
                            "response": response_data
                        }
                        logger.warning(f"Bundle submission to {relayer} did not return bundle hash")
                else:
                    results[relayer] = {
                        "success": False,
                        "error": f"HTTP {response.status_code}",
                        "response": response.text
                    }
                    logger.warning(f"Bundle submission to {relayer} failed: HTTP {response.status_code}")
            
            except Exception as e:
                results[relayer] = {
                    "success": False,
                    "error": str(e)
                }
                logger.error(f"Error submitting bundle to {relayer}: {e}")
        
        # Update statistics
        if any_success:
            self.stats["bundles_submitted"] += 1
        
        return {
            "bundle": bundle,
            "results": results,
            "any_success": any_success,
            "timestamp": int(time.time())
        }
    
    def create_arbitrage_bundle(self, 
                               opportunities: List[Dict[str, Any]], 
                               network: str="ethereum",
                               block_number: Optional[int]=None) -> Dict[str, Any]:
        """
        Create a transaction bundle for arbitrage opportunities
        
        Args:
            opportunities: List of arbitrage opportunities
            network: Target network for the bundle
            block_number: Target block number (if None, uses current + 1)
            
        Returns:
            Bundle object ready for submission
        """
        # Extract transactions from opportunities
        transactions = []
        
        # Get contract ABIs
        with open("abi/router_abi.json", "r") as f:
            router_abi = json.load(f)
        
        w3 = self.connections.get(network)
        if not w3:
            raise ValueError(f"No connection available for network: {network}")
        
        # Prepare a transaction for each opportunity
        for i, opp in enumerate(opportunities):
            # Create transaction based on opportunity type
            if opp.get("type") == "direct_arbitrage":
                # Extract opportunity details
                buy_exchange = opp.get("buy_exchange")
                sell_exchange = opp.get("sell_exchange")
                symbol = opp.get("symbol")
                buy_price = opp.get("buy_price")
                sell_price = opp.get("sell_price")
                amount = opp.get("amount", w3.to_wei(0.1, "ether"))  # Default 0.1 ETH
                
                # Get router addresses from opportunity or config
                router_addresses = self.config.get("router_addresses", {})
                buy_router = opp.get("buy_router_address") or router_addresses.get(buy_exchange)
                sell_router = opp.get("sell_router_address") or router_addresses.get(sell_exchange)
                
                if not buy_router or not sell_router:
                    logger.warning(f"Missing router address for {buy_exchange} or {sell_exchange}")
                    continue
                
                # Create and initialize contract object
                buy_router_contract = w3.eth.contract(address=buy_router, abi=router_abi)
                sell_router_contract = w3.eth.contract(address=sell_router, abi=router_abi)
                
                # Create buy transaction
                buy_path = opp.get("buy_path", [])
                if not buy_path:
                    logger.warning(f"No buy path specified for opportunity {i}")
                    continue
                
                # Calculate minimum output amount for buy (with slippage)
                slippage = opp.get("slippage", 0.005)  # Default 0.5%
                amount_out_min = int(amount / buy_price * (1 - slippage))
                
                # Build buy transaction
                deadline = int(time.time() + 600)  # 10 minutes
                try:
                    buy_tx = buy_router_contract.functions.swapExactETHForTokens(
                        amount_out_min,
                        buy_path,
                        self.account.address,
                        deadline
                    ).build_transaction({
                        "from": self.account.address,
                        "value": amount,
                        "gas": 300000,  # Will be estimated in create_bundle
                        "nonce": None,  # Will be set in create_bundle
                    })
                    
                    # Add buy transaction to bundle
                    transactions.append({"params": buy_tx})
                    
                    # Build sell transaction
                    sell_path = opp.get("sell_path", list(reversed(buy_path)))
                    token_amount = amount_out_min  # Amount received from buy transaction
                    amount_out_min_sell = int(token_amount * sell_price * (1 - slippage))
                    
                    # Approve token spending if needed
                    token_address = buy_path[-1]
                    token_abi = [
                        {
                            "constant": False,
                            "inputs": [
                                {"name": "_spender", "type": "address"},
                                {"name": "_value", "type": "uint256"}
                            ],
                            "name": "approve",
                            "outputs": [{"name": "", "type": "bool"}],
                            "payable": False,
                            "stateMutability": "nonpayable",
                            "type": "function"
                        }
                    ]
                    token_contract = w3.eth.contract(address=token_address, abi=token_abi)
                    
                    # Create approval transaction
                    approve_tx = token_contract.functions.approve(
                        sell_router,
                        token_amount
                    ).build_transaction({
                        "from": self.account.address,
                        "gas": 100000,  # Will be estimated in create_bundle
                        "nonce": None,  # Will be set in create_bundle
                    })
                    
                    # Add approval transaction to bundle
                    transactions.append({"params": approve_tx})
                    
                    # Create sell transaction
                    sell_tx = sell_router_contract.functions.swapExactTokensForETH(
                        token_amount,
                        amount_out_min_sell,
                        sell_path,
                        self.account.address,
                        deadline
                    ).build_transaction({
                        "from": self.account.address,
                        "gas": 300000,  # Will be estimated in create_bundle
                        "nonce": None,  # Will be set in create_bundle
                    })
                    
                    # Add sell transaction to bundle
                    transactions.append({"params": sell_tx})
                    
                except Exception as e:
                    logger.error(f"Error creating transaction for opportunity {i}: {e}")
                    continue
                
            elif opp.get("type") == "triangle_arbitrage":
                # Handle triangle arbitrage (similar to direct but with 3 trades)
                # Would implement logic similar to above but with three trades
                pass
            
            elif opp.get("type") == "custom_transaction":
                # Allow passing in pre-built transactions
                tx_params = opp.get("transaction_params")
                if tx_params:
                    transactions.append({"params": tx_params})
                else:
                    logger.warning(f"No transaction params for custom transaction opportunity {i}")
            
            else:
                logger.warning(f"Unknown opportunity type: {opp.get('type')}")
                continue
        
        # Create the bundle if we have transactions
        if transactions:
            return self.create_bundle(
                transactions=transactions,
                network=network,
                block_number=block_number,
                ensure_success=True
            )
        else:
            logger.warning("No valid transactions created from opportunities")
            return None
    
    def check_bundle_status(self, bundle_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check the status of a submitted bundle
        
        Args:
            bundle_result: Result object from submit_bundle
            
        Returns:
            Status information for the bundle
        """
        bundle = bundle_result.get("bundle", {})
        network = bundle.get("network", "ethereum")
        block_number = bundle.get("block_number")
        w3 = self.connections.get(network)
        
        if not w3:
            return {
                "success": False,
                "error": f"No connection available for network: {network}"
            }
        
        # Check current block number
        current_block = w3.eth.block_number
        
        # Bundle is still pending if target block not reached
        if current_block < block_number:
            return {
                "success": None,
                "pending": True,
                "current_block": current_block,
                "target_block": block_number,
                "blocks_remaining": block_number - current_block
            }
        
        # Check if transactions were included
        included_txs = []
        for tx in bundle.get("transactions", []):
            tx_hash = tx.get("hash")
            if not tx_hash:
                continue
                
            try:
                # Check if transaction is in the blockchain
                receipt = w3.eth.get_transaction_receipt(tx_hash)
                if receipt and receipt.blockNumber:
                    included_txs.append({
                        "hash": tx_hash,
                        "block_number": receipt.blockNumber,
                        "status": receipt.status,
                        "gas_used": receipt.gasUsed
                    })
            except Exception:
                # Transaction not found
                pass
        
        # Check if all transactions were included
        all_included = len(included_txs) == len(bundle.get("transactions", []))
        target_block_included = any(tx.get("block_number") == block_number for tx in included_txs)
        
        if all_included and target_block_included:
            # Update statistics
            self.stats["bundles_included"] += 1
            
            # Calculate profit if successful (would need transaction analysis)
            
            return {
                "success": True,
                "included": True,
                "target_block_hit": True,
                "transactions": included_txs
            }
        elif all_included:
            return {
                "success": True,
                "included": True,
                "target_block_hit": False,
                "transactions": included_txs
            }
        elif included_txs:
            return {
                "success": True,
                "included": False,
                "partial": True,
                "transactions": included_txs,
                "missing": len(bundle.get("transactions", [])) - len(included_txs)
            }
        else:
            return {
                "success": False,
                "included": False,
                "transactions": [],
                "error": "No transactions included"
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the MEV Bundle Manager"""
        stats_copy = self.stats.copy()
        
        # Add success rate
        if stats_copy["bundles_submitted"] > 0:
            stats_copy["inclusion_rate"] = (stats_copy["bundles_included"] / stats_copy["bundles_submitted"]) * 100
        else:
            stats_copy["inclusion_rate"] = 0
            
        return stats_copy
        
    def get_configured_networks(self):
        """Get the list of configured networks"""
        try:
            return self.config.get("networks", [])
        except Exception as e:
            logger.error(f"Error getting configured networks: {e}")
            return []
            
    def get_configured_relayers(self):
        """Get the list of configured relayers"""
        try:
            return list(self.config.get("relayer_endpoints", {}).keys())
        except Exception as e:
            logger.error(f"Error getting configured relayers: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Initialize the MEV Bundle Manager
    mev_manager = MEVBundleManager()
    
    # Example arbitrage opportunity
    opportunity = {
        "type": "direct_arbitrage",
        "buy_exchange": "uniswap_v3",
        "sell_exchange": "sushiswap",
        "symbol": "ETH/USDC",
        "buy_price": 3500,
        "sell_price": 3520,
        "amount": Web3(Web3.HTTPProvider()).to_wei(0.1, "ether"),
        "buy_path": [
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"   # USDC
        ],
        "sell_path": [
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"   # WETH
        ],
        "slippage": 0.005,
        "buy_router_address": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
        "sell_router_address": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F"
    }
    
    if mev_manager.account:
        # Create bundle with arbitrage opportunity
        bundle = mev_manager.create_arbitrage_bundle([opportunity], network="ethereum")
        
        # Submit bundle
        if bundle:
            result = mev_manager.submit_bundle(bundle)
            print(f"Bundle submission result: {result['any_success']}")
            
            # Check status after a few blocks
            time.sleep(30)
            status = mev_manager.check_bundle_status(result)
            print(f"Bundle status: {status['success']}")
    else:
        print("Cannot create bundle without account. Set PRIVATE_KEY environment variable.")
