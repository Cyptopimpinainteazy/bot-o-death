from typing import List, Dict, Any, Optional
import logging
import os
import json
import yaml
import time
import asyncio
from web3 import Web3
from web3.middleware import geth_poa_middleware
from web3.providers.websocket import WebsocketProvider
from web3.exceptions import (
    TransactionNotFound,
    TimeExhausted,
    BadFunctionCallOutput,
    ValidationError,
    BlockNotFound,
    ProviderConnectionError
)

class ChainConnector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.web3_connections = {}
        self.websocket_connections = {}  # Store WebSocket connections separately
        self.contracts = {}
        self.max_retries = config.get('blockchain', {}).get('max_retries', 3)
        self.retry_delay = config.get('blockchain', {}).get('retry_delay_ms', 1000) / 1000.0  # Convert to seconds
        
        # Load contract ABIs
        self.abis = {
            'triple_flashloan': self._load_abi('TripleFlashloan'),
            'x3star_token': self._load_abi('X3STAR'),
            'mev_strategies': self._load_abi('MevStrategies')
        }

    def _load_abi(self, contract_name: str) -> Dict:
        try:
            # Try to load from artifacts directory
            artifact_path = f"artifacts/EnhancedQuantumTrading/contracts/{contract_name}.sol/{contract_name}.json"
            with open(artifact_path, 'r') as f:
                contract_json = json.load(f)
                return contract_json['abi']
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Error loading ABI for {contract_name}: {e}")
            return {}

    async def connect(self, chain: str, use_websocket: bool = False) -> bool:
        try:
            # Skip if already connected with the right type of connection
            if use_websocket and chain in self.websocket_connections and self.websocket_connections[chain].isConnected():
                return True
            elif not use_websocket and chain in self.web3_connections and self.web3_connections[chain].isConnected():
                return True
                
            if chain not in self.config['blockchain']['networks']:
                self.logger.error(f"Chain {chain} not found in config")
                return False
                
            chain_config = self.config['blockchain']['networks'][chain]
            
            # Initialize the appropriate provider
            web3 = None
            if use_websocket and 'ws_url' in chain_config and chain_config['ws_url']:
                ws_url = os.path.expandvars(chain_config['ws_url'])
                self.logger.info(f"Connecting to {chain} via WebSocket at {ws_url}")
                provider = WebsocketProvider(ws_url)
                web3 = Web3(provider)
                
                # Set up reconnection for WebSocket
                if chain in self.websocket_connections:
                    try:
                        old_provider = self.websocket_connections[chain].provider
                        if hasattr(old_provider, 'websocket_client') and old_provider.websocket_client:
                            old_provider.websocket_client.close()
                    except Exception as e:
                        self.logger.warning(f"Error closing old WebSocket connection for {chain}: {e}")
                
                self.websocket_connections[chain] = web3
            else:
                rpc_url = os.path.expandvars(chain_config['rpc_url'])
                self.logger.info(f"Connecting to {chain} via HTTP at {rpc_url}")
                web3 = Web3(Web3.HTTPProvider(rpc_url))
                self.web3_connections[chain] = web3
            
            # Add middleware for POA chains like Polygon
            web3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            if not web3.isConnected():
                self.logger.error(f"Could not connect to {chain}")
                return False
                
            # Set up contracts
            if chain not in self.contracts:
                self.contracts[chain] = {}
                
            # Load the contract instances
            for contract_type, address in chain_config['contracts'].items():
                if not address or address == "0x0000000000000000000000000000000000000000":
                    continue
                    
                contract = web3.eth.contract(
                    address=web3.toChecksumAddress(address),
                    abi=self.abis.get(contract_type, [])
                )
                self.contracts[chain][contract_type] = contract
                self.logger.info(f"Loaded {contract_type} contract at {address} on {chain}")
                
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to {chain}: {e}")
            return False
    
    def get_contract(self, chain: str, contract_type: str) -> Optional[Any]:
        """Get a specific contract instance for the specified chain"""
        if chain not in self.contracts or contract_type not in self.contracts[chain]:
            self.logger.error(f"Contract {contract_type} not available on {chain}")
            return None
        return self.contracts[chain][contract_type]
        
    def get_web3(self, chain: str, prefer_websocket: bool = True) -> Optional[Web3]:
        """Get the Web3 instance for the specified chain, preferring WebSocket if available"""
        if prefer_websocket and chain in self.websocket_connections and self.websocket_connections[chain].isConnected():
            return self.websocket_connections[chain]
        
        if chain not in self.web3_connections:
            self.logger.error(f"No Web3 connection for {chain}")
            return None
        return self.web3_connections[chain]
    
    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic for network operations"""
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except (TransactionNotFound, TimeExhausted, BadFunctionCallOutput, 
                    ValidationError, BlockNotFound, ProviderConnectionError) as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt}/{self.max_retries} failed: {e}")
                
                # If it's a connection error and we have more attempts, try to reconnect
                if isinstance(e, ProviderConnectionError) and attempt < self.max_retries:
                    chain = kwargs.get('chain', args[0] if args else None)
                    if chain:
                        self.logger.info(f"Attempting to reconnect to {chain}")
                        await self.connect(chain, chain in self.websocket_connections)
                
                # Wait before retrying (exponential backoff)
                wait_time = self.retry_delay * (2 ** (attempt - 1))
                await asyncio.sleep(wait_time)
        
        # If we get here, all attempts failed
        if last_error:
            raise last_error
        raise Exception("All retry attempts failed")

class TradingLogic:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.chain_connector = ChainConnector(config)
        self.signal_analyzer = None
        self.private_key = os.path.expandvars(config['blockchain']['wallet']['private_key'])
        self.bot_address = config['blockchain']['wallet']['bot_address']
        
        # Initialize price fetching cache
        self.price_cache = {
            'last_updated': {},
            'prices': {}
        }
        self.price_cache_ttl_seconds = config.get('blockchain', {}).get('price_cache_ttl_seconds', 60)

    async def initialize(self, use_websocket: bool = True) -> bool:
        """Initialize connections to all chains"""
        results = []
        for chain in self.config['blockchain']['networks'].keys():
            if chain != 'ethereum':  # Skip ethereum as it's commented out
                results.append(await self.chain_connector.connect(chain, use_websocket))
        return all(results)

    async def execute_triple_flashloan(self, chain: str, tokens: List[str], amounts: List[int]) -> Dict:
        """Execute a triple flashloan on the specified chain"""
        async def _execute():
            # Connect to the chain
            await self.chain_connector.connect(chain)
            
            # Get the contract
            contract = self.chain_connector.get_contract(chain, 'triple_flashloan')
            if not contract:
                return {"status": "error", "message": f"No triple_flashloan contract on {chain}"}
                
            web3 = self.chain_connector.get_web3(chain)
            if not web3:
                return {"status": "error", "message": f"No web3 connection for {chain}"}
                
            # Build transaction
            # This is a simplified version - the actual implementation would need
            # to handle the specific parameters for the flashloan function
            tx = contract.functions.executeTripleFlashloan(
                [web3.toChecksumAddress(tokens[0])],  # aaveAssets
                [amounts[0]],                          # aaveAmounts
                [web3.toChecksumAddress(tokens[1])],  # balancerAssets
                [amounts[1]],                          # balancerAmounts
                [web3.toChecksumAddress(tokens[2])],  # curveAssets
                [amounts[2]],                          # curveAmounts
                amounts[3],                           # dodoBaseAmount
                amounts[4]                            # dodoQuoteAmount
            ).buildTransaction({
                'from': self.bot_address,
                'nonce': web3.eth.getTransactionCount(self.bot_address),
                'gas': 3000000,
                'gasPrice': web3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_tx = web3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            self.logger.info(f"Triple flashloan executed on {chain}: {web3.toHex(tx_hash)}")
            return {
                "status": "success", 
                "chain": chain, 
                "tx_hash": web3.toHex(tx_hash)
            }
        
        try:
            return await self.chain_connector.execute_with_retry(_execute)
        except Exception as e:
            self.logger.error(f"Error executing triple flashloan on {chain}: {e}")
            return {"status": "error", "message": str(e)}

    async def execute_sandwich_trade(self, chain: str, token: str, amount_in: int, amount_out_min: int, deadline: int) -> Dict:
        """Execute a sandwich trade using the X3STAR contract"""
        async def _execute():
            # Connect to the chain
            await self.chain_connector.connect(chain)
            
            # Get the contract
            contract = self.chain_connector.get_contract(chain, 'x3star_token')
            if not contract:
                return {"status": "error", "message": f"No x3star_token contract on {chain}"}
                
            web3 = self.chain_connector.get_web3(chain)
            if not web3:
                return {"status": "error", "message": f"No web3 connection for {chain}"}
                
            # Example router - this should be set based on the chain
            router = self.config['blockchain']['networks'][chain]['contracts'].get('quickswap_router', "0x0000000000000000000000000000000000000000")
            
            # Build transaction
            tx = contract.functions.executeSandwichTrade(
                router,
                amount_in,
                amount_out_min,
                deadline
            ).buildTransaction({
                'from': self.bot_address,
                'nonce': web3.eth.getTransactionCount(self.bot_address),
                'gas': 500000,
                'gasPrice': web3.eth.gas_price,
                'value': amount_in  # Send ETH with the transaction
            })
            
            # Sign and send transaction
            signed_tx = web3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            self.logger.info(f"Sandwich trade executed on {chain}: {web3.toHex(tx_hash)}")
            return {
                "status": "success", 
                "chain": chain, 
                "tx_hash": web3.toHex(tx_hash)
            }
        
        try:
            return await self.chain_connector.execute_with_retry(_execute)
        except Exception as e:
            self.logger.error(f"Error executing sandwich trade on {chain}: {e}")
            return {"status": "error", "message": str(e)}
            
    async def execute_mev_strategy(self, chain: str, strategy: str, params: Dict[str, Any]) -> Dict:
        """Execute a MEV strategy using the MevStrategies contract"""
        async def _execute():
            # Connect to the chain
            await self.chain_connector.connect(chain)
            
            # Get the contract
            contract = self.chain_connector.get_contract(chain, 'mev_strategies')
            if not contract:
                return {"status": "error", "message": f"No mev_strategies contract on {chain}"}
                
            web3 = self.chain_connector.get_web3(chain)
            if not web3:
                return {"status": "error", "message": f"No web3 connection for {chain}"}
                
            # Build transaction based on strategy type
            if strategy == 'arbitrage':
                tx = contract.functions.executeCrossDexArbitrage(
                    web3.toChecksumAddress(params['token_in']),
                    web3.toChecksumAddress(params['token_out']),
                    web3.toChecksumAddress(params['source_router']),
                    web3.toChecksumAddress(params['target_router']),
                    params['amount_in'],
                    params['min_profit']
                ).buildTransaction({
                    'from': self.bot_address,
                    'nonce': web3.eth.getTransactionCount(self.bot_address),
                    'gas': 3000000,
                    'gasPrice': web3.eth.gas_price
                })
            elif strategy == 'jit_liquidity':
                tx = contract.functions.executeJitLiquidity(
                    web3.toChecksumAddress(params['liquidity_pair']),
                    web3.toChecksumAddress(params['token_a']),
                    web3.toChecksumAddress(params['token_b']),
                    params['amount_a'],
                    params['amount_b'],
                    params['target_tx'],
                    params['min_profit']
                ).buildTransaction({
                    'from': self.bot_address,
                    'nonce': web3.eth.getTransactionCount(self.bot_address),
                    'gas': 3000000,
                    'gasPrice': web3.eth.gas_price
                })
            elif strategy == 'liquidation':
                tx = contract.functions.executeLiquidation(
                    web3.toChecksumAddress(params['borrower']),
                    web3.toChecksumAddress(params['collateral_asset']),
                    web3.toChecksumAddress(params['debt_asset']),
                    params['debt_to_cover'],
                    params['min_profit']
                ).buildTransaction({
                    'from': self.bot_address,
                    'nonce': web3.eth.getTransactionCount(self.bot_address),
                    'gas': 3000000,
                    'gasPrice': web3.eth.gas_price
                })
            elif strategy == 'back_running':
                tx = contract.functions.executeBackRunning(
                    params['target_tx'],
                    web3.toChecksumAddress(params['token_in']),
                    web3.toChecksumAddress(params['token_out']),
                    web3.toChecksumAddress(params['router']),
                    params['amount_in'],
                    params['min_profit']
                ).buildTransaction({
                    'from': self.bot_address,
                    'nonce': web3.eth.getTransactionCount(self.bot_address),
                    'gas': 3000000,
                    'gasPrice': web3.eth.gas_price
                })
            else:
                return {"status": "error", "message": f"Unknown strategy: {strategy}"}
            
            # Sign and send transaction
            signed_tx = web3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            self.logger.info(f"MEV strategy {strategy} executed on {chain}: {web3.toHex(tx_hash)}")
            return {
                "status": "success", 
                "chain": chain, 
                "strategy": strategy,
                "tx_hash": web3.toHex(tx_hash)
            }
            
        try:
            return await self.chain_connector.execute_with_retry(_execute)
        except Exception as e:
            self.logger.error(f"Error executing MEV strategy {strategy} on {chain}: {e}")
            return {"status": "error", "message": str(e)}

    async def run_salmonella_trap(self, symbols: List[str], chain: str = "polygon"):
        w3 = self.chain_connector.get_web3(chain)
        for symbol in symbols:
            bait_amount = self.config['salmonella']['bait_amount']
            gas_price = w3.to_wei(self.config['salmonella']['bait_gas_price'], 'gwei')
            tx = {
                'from': self.bot_address,
                'value': bait_amount,
                'gas': 200000,
                'gasPrice': gas_price,
                'nonce': w3.eth.get_transaction_count(self.bot_address)
            }
            signed_tx = w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            self.logger.info(f"Salmonella bait deployed on {symbol}: {tx_hash.hex()}")
            await asyncio.sleep(self.config['salmonella']['monitor_interval'])
        return {"status": "success", "symbols": symbols}
    
    async def get_token_price(self, token_address: str, chain: str, force_refresh: bool = False) -> float:
        """Get token price with caching to reduce API calls"""
        cache_key = f"{chain}:{token_address}"
        current_time = time.time()
        
        # Check if we have a cached price that's still valid
        if not force_refresh and cache_key in self.price_cache['prices']:
            last_updated = self.price_cache['last_updated'].get(cache_key, 0)
            if current_time - last_updated < self.price_cache_ttl_seconds:
                self.logger.debug(f"Using cached price for {token_address} on {chain}")
                return self.price_cache['prices'][cache_key]
        
        # If we get here, we need to fetch a fresh price
        try:
            # Implementation will vary based on your price sources
            # This is a placeholder for the actual implementation
            price = await self._fetch_token_price(token_address, chain)
            
            # Cache the result
            self.price_cache['prices'][cache_key] = price
            self.price_cache['last_updated'][cache_key] = current_time
            
            return price
        except Exception as e:
            self.logger.error(f"Error fetching price for {token_address} on {chain}: {e}")
            # Return cached price if available, otherwise raise
            if cache_key in self.price_cache['prices']:
                self.logger.warning(f"Using stale cached price for {token_address} on {chain}")
                return self.price_cache['prices'][cache_key]
            raise
    
    async def _fetch_token_price(self, token_address: str, chain: str) -> float:
        """Actual implementation to fetch token price from price sources"""
        # This is where you would implement your price fetching logic
        # For example, querying CoinGecko, Chainlink, or on-chain DEX prices
        
        # Placeholder implementation that should be replaced with actual logic
        # For demo purposes, we're just returning a random price
        import random
        return random.uniform(0.1, 100.0)
