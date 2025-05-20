import os
import json
import time
import logging
import numpy as np
import pandas as pd
import requests
from web3 import Web3
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SlippageControl")

# Load environment variables
load_dotenv()

class SlippageController:
    """
    Slippage control and optimization for DEX trades.
    Handles slippage calculation, prediction, and mitigation.
    """
    
    def __init__(self):
        """Initialize the slippage controller"""
        # API keys
        self.alchemy_api_key = os.getenv("ALCHEMY_API_KEY")
        
        # Web3 providers
        self.providers = {
            "ethereum": Web3(Web3.HTTPProvider(f"https://eth-mainnet.g.alchemy.com/v2/{self.alchemy_api_key}")),
            "polygon": Web3(Web3.HTTPProvider(f"https://polygon-mainnet.g.alchemy.com/v2/{self.alchemy_api_key}"))
        }
        
        # Default DEX Router ABIs
        self.dex_abis = {
            "uniswap_v2": json.loads('''[{"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"}],"name":"getAmountsOut","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"}],"name":"getAmountsIn","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"view","type":"function"}]'''),
            "uniswap_v3": json.loads('''[{"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint24","name":"fee","type":"uint24"},{"internalType":"address","name":"tokenIn","type":"address"},{"internalType":"address","name":"tokenOut","type":"address"}],"name":"quoteExactInputSingle","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],"stateMutability":"nonpayable","type":"function"}]''')
        }
        
        # DEX Router addresses
        self.dex_routers = {
            "ethereum": {
                "uniswap_v2": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
                "sushiswap": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
                "uniswap_v3": "0xE592427A0AEce92De3Edee1F18E0157C05861564"
            },
            "polygon": {
                "quickswap": "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff",
                "sushiswap": "0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506"
            }
        }
        
        # Contract instances (will be populated as needed)
        self.router_contracts = {}
        
        # Slippage measurements
        self.slippage_history = {}
        self.max_history_items = 100
        
        # Default slippage thresholds
        self.default_slippage = 0.005  # 0.5%
        self.max_acceptable_slippage = 0.02  # 2%
        
        logger.info("Slippage controller initialized")
    
    def _get_router_contract(self, chain, dex):
        """Get or create router contract instance"""
        key = f"{chain}_{dex}"
        if key not in self.router_contracts:
            if chain in self.providers and dex in self.dex_routers.get(chain, {}):
                address = self.dex_routers[chain][dex]
                abi = self.dex_abis.get("uniswap_v2")  # Default to v2 ABI
                
                if "v3" in dex:
                    abi = self.dex_abis.get("uniswap_v3")
                
                self.router_contracts[key] = self.providers[chain].eth.contract(
                    address=Web3.to_checksum_address(address),
                    abi=abi
                )
            else:
                return None
        
        return self.router_contracts.get(key)
    
    def estimate_price_impact(self, chain, dex, token_in, token_out, amount_in):
        """
        Estimate price impact for a swap
        
        Args:
            chain: Blockchain network
            dex: DEX to use
            token_in: Input token address
            token_out: Output token address
            amount_in: Input amount in smallest units
            
        Returns:
            Estimated price impact percentage
        """
        try:
            router = self._get_router_contract(chain, dex)
            if not router:
                return None
            
            # Normalize addresses
            token_in = Web3.to_checksum_address(token_in)
            token_out = Web3.to_checksum_address(token_out)
            
            # Get quote for full amount
            full_amount_out = self._get_amount_out(router, dex, token_in, token_out, amount_in)
            if not full_amount_out:
                return None
            
            # Get quote for very small amount (to calculate spot price)
            small_amount_in = amount_in // 1000 if amount_in > 1000 else 1
            small_amount_out = self._get_amount_out(router, dex, token_in, token_out, small_amount_in)
            if not small_amount_out:
                return None
            
            # Calculate spot price and effective price
            spot_price = small_amount_out * 1000 / amount_in if amount_in > 1000 else small_amount_out
            effective_price = full_amount_out / amount_in
            
            # Calculate price impact
            price_impact = 1 - (effective_price / spot_price)
            price_impact_pct = price_impact * 100
            
            logger.info(f"Estimated price impact for {amount_in} tokens: {price_impact_pct:.4f}%")
            
            return price_impact
        
        except Exception as e:
            logger.error(f"Error estimating price impact: {e}")
            return None
    
    def _get_amount_out(self, router, dex, token_in, token_out, amount_in):
        """Get amount out from router contract"""
        try:
            if "v3" in dex:
                # For Uniswap V3, use quoteExactInputSingle with a default fee
                fee = 3000  # 0.3% fee tier
                amount_out = router.functions.quoteExactInputSingle(
                    amount_in, fee, token_in, token_out
                ).call()
            else:
                # For V2-style routers, use getAmountsOut
                amounts = router.functions.getAmountsOut(
                    amount_in, [token_in, token_out]
                ).call()
                amount_out = amounts[1]
            
            return amount_out
        
        except Exception as e:
            logger.error(f"Error getting amount out: {e}")
            return None
    
    def recommend_slippage_tolerance(self, chain, dex, token_in, token_out, amount_in, market_volatility=None):
        """
        Recommend appropriate slippage tolerance based on market conditions
        
        Args:
            chain: Blockchain network
            dex: DEX to use
            token_in: Input token address
            token_out: Output token address
            amount_in: Input amount in smallest units
            market_volatility: Optional market volatility measure (0-1)
            
        Returns:
            Recommended slippage tolerance percentage
        """
        try:
            # Calculate base slippage from price impact
            price_impact = self.estimate_price_impact(chain, dex, token_in, token_out, amount_in)
            
            if price_impact is None:
                # Use default if estimate fails
                base_slippage = self.default_slippage
            else:
                # Base slippage is 2x the estimated price impact, with minimum of default_slippage
                base_slippage = max(self.default_slippage, price_impact * 2)
            
            # Adjust for market volatility if provided
            if market_volatility is not None:
                volatility_factor = 1 + market_volatility
                base_slippage *= volatility_factor
            
            # Adjust for chain-specific factors
            chain_factor = 1.0
            if chain == "polygon":
                chain_factor = 1.2  # Polygon typically needs higher slippage
            elif chain == "arbitrum":
                chain_factor = 1.3  # Arbitrum can have higher slippage due to L2 mechanics
            
            # Adjust for token-specific factors (could be enhanced with token volatility data)
            token_factor = 1.0
            # Add token-specific adjustments here if needed
            
            # Calculate final slippage
            final_slippage = base_slippage * chain_factor * token_factor
            
            # Cap at maximum acceptable slippage
            final_slippage = min(final_slippage, self.max_acceptable_slippage)
            
            # Round to nearest 0.1%
            final_slippage = round(final_slippage * 1000) / 1000
            
            logger.info(f"Recommended slippage tolerance: {final_slippage * 100:.2f}%")
            
            return final_slippage
        
        except Exception as e:
            logger.error(f"Error recommending slippage tolerance: {e}")
            return self.default_slippage
    
    def calculate_optimal_trade_size(self, chain, dex, token_in, token_out, total_amount_in, max_price_impact=0.01):
        """
        Calculate optimal trade size to minimize slippage
        
        Args:
            chain: Blockchain network
            dex: DEX to use
            token_in: Input token address
            token_out: Output token address
            total_amount_in: Total input amount to trade
            max_price_impact: Maximum acceptable price impact per trade
            
        Returns:
            List of recommended trade sizes
        """
        try:
            # Start with a binary search to find maximum trade size within impact limit
            min_amount = total_amount_in // 100  # 1% of total
            max_amount = total_amount_in
            optimal_size = min_amount
            
            for _ in range(10):  # Max 10 iterations for binary search
                test_size = (min_amount + max_amount) // 2
                impact = self.estimate_price_impact(chain, dex, token_in, token_out, test_size)
                
                if impact is None or impact > max_price_impact:
                    # Too high impact, reduce size
                    max_amount = test_size
                else:
                    # Impact acceptable, try larger size
                    min_amount = test_size
                    optimal_size = test_size
            
            # Calculate number of trades needed
            num_trades = (total_amount_in + optimal_size - 1) // optimal_size  # Ceiling division
            
            # Distribute amount evenly
            trade_sizes = []
            remaining = total_amount_in
            
            for i in range(num_trades):
                if i == num_trades - 1:
                    # Last trade gets remainder
                    trade_sizes.append(remaining)
                else:
                    trade_sizes.append(optimal_size)
                    remaining -= optimal_size
            
            return {
                "optimal_trade_size": optimal_size,
                "number_of_trades": num_trades,
                "trade_sizes": trade_sizes,
                "estimated_price_impact_per_trade": self.estimate_price_impact(
                    chain, dex, token_in, token_out, optimal_size
                )
            }
        
        except Exception as e:
            logger.error(f"Error calculating optimal trade size: {e}")
            return {
                "optimal_trade_size": total_amount_in,
                "number_of_trades": 1,
                "trade_sizes": [total_amount_in],
                "estimated_price_impact_per_trade": None
            }
    
    def optimize_trading_route(self, chain, tokens, amount_in):
        """
        Optimize trading route to minimize slippage
        
        Args:
            chain: Blockchain network
            tokens: List of token addresses [token_in, token_out]
            amount_in: Input amount
            
        Returns:
            Optimal route information
        """
        try:
            token_in, token_out = tokens
            
            # Get available DEXes for this chain
            available_dexes = list(self.dex_routers.get(chain, {}).keys())
            if not available_dexes:
                return None
            
            # Check all DEXes and find the best rate
            best_dex = None
            best_amount_out = 0
            best_price_impact = 1.0  # 100% impact as initial value
            
            for dex in available_dexes:
                try:
                    router = self._get_router_contract(chain, dex)
                    if not router:
                        continue
                    
                    amount_out = self._get_amount_out(
                        router, dex, Web3.to_checksum_address(token_in), 
                        Web3.to_checksum_address(token_out), amount_in
                    )
                    
                    if not amount_out:
                        continue
                    
                    price_impact = self.estimate_price_impact(
                        chain, dex, token_in, token_out, amount_in
                    )
                    
                    if price_impact is None:
                        price_impact = 0.1  # Default high impact if estimation fails
                    
                    # We want highest amount out and lowest price impact
                    # Use a scoring mechanism that considers both
                    score = amount_out * (1 - price_impact)
                    
                    if best_dex is None or score > best_amount_out * (1 - best_price_impact):
                        best_dex = dex
                        best_amount_out = amount_out
                        best_price_impact = price_impact
                
                except Exception as e:
                    logger.warning(f"Error checking {dex}: {e}")
                    continue
            
            if best_dex:
                # Check for potential routing through an intermediate token
                intermediate_tokens = [
                    Web3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"),  # WETH
                    Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")   # USDC on Polygon
                ]
                
                for intermediate in intermediate_tokens:
                    if intermediate == token_in or intermediate == token_out:
                        continue
                    
                    try:
                        router = self._get_router_contract(chain, best_dex)
                        
                        # Check token_in -> intermediate
                        first_hop_out = self._get_amount_out(
                            router, best_dex, Web3.to_checksum_address(token_in), 
                            intermediate, amount_in
                        )
                        
                        if not first_hop_out:
                            continue
                        
                        # Check intermediate -> token_out
                        second_hop_out = self._get_amount_out(
                            router, best_dex, intermediate, 
                            Web3.to_checksum_address(token_out), first_hop_out
                        )
                        
                        if not second_hop_out:
                            continue
                        
                        # Compare with direct route
                        if second_hop_out > best_amount_out:
                            return {
                                "route_type": "multi_hop",
                                "dex": best_dex,
                                "path": [token_in, intermediate, token_out],
                                "expected_output": second_hop_out,
                                "improvement_over_direct": (second_hop_out / best_amount_out - 1) * 100,
                                "recommended_slippage": self.recommend_slippage_tolerance(
                                    chain, best_dex, token_in, token_out, amount_in
                                ) * 100
                            }
                    
                    except Exception as e:
                        logger.warning(f"Error checking intermediate routing: {e}")
                        continue
                
                return {
                    "route_type": "direct",
                    "dex": best_dex,
                    "path": [token_in, token_out],
                    "expected_output": best_amount_out,
                    "estimated_price_impact": best_price_impact * 100,
                    "recommended_slippage": self.recommend_slippage_tolerance(
                        chain, best_dex, token_in, token_out, amount_in
                    ) * 100
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Error optimizing trading route: {e}")
            return None
    
    def record_actual_slippage(self, trade_data):
        """
        Record actual slippage from executed trade
        
        Args:
            trade_data: Dictionary with trade details
        """
        try:
            chain = trade_data.get("chain")
            dex = trade_data.get("dex")
            token_in = trade_data.get("token_in")
            token_out = trade_data.get("token_out")
            amount_in = trade_data.get("amount_in")
            expected_out = trade_data.get("expected_out")
            actual_out = trade_data.get("actual_out")
            
            if not all([chain, dex, token_in, token_out, amount_in, expected_out, actual_out]):
                logger.warning("Missing required data for slippage recording")
                return
            
            # Calculate actual slippage
            if expected_out > 0:
                slippage = (expected_out - actual_out) / expected_out
            else:
                slippage = 0
            
            # Record in history
            key = f"{chain}_{dex}_{token_in}_{token_out}"
            if key not in self.slippage_history:
                self.slippage_history[key] = []
            
            entry = {
                "timestamp": int(time.time()),
                "amount_in": amount_in,
                "expected_out": expected_out,
                "actual_out": actual_out,
                "slippage": slippage
            }
            
            self.slippage_history[key].append(entry)
            
            # Trim history if needed
            if len(self.slippage_history[key]) > self.max_history_items:
                self.slippage_history[key].pop(0)
            
            logger.info(f"Recorded slippage of {slippage * 100:.4f}% for {key}")
            
            return slippage
        
        except Exception as e:
            logger.error(f"Error recording slippage: {e}")
            return None
    
    def get_slippage_stats(self, chain, dex, token_in, token_out):
        """Get slippage statistics for a trading pair"""
        key = f"{chain}_{dex}_{token_in}_{token_out}"
        
        if key not in self.slippage_history or not self.slippage_history[key]:
            return {
                "pair": f"{token_in}/{token_out}",
                "dex": dex,
                "chain": chain,
                "average_slippage": None,
                "median_slippage": None,
                "max_slippage": None,
                "min_slippage": None,
                "samples": 0
            }
        
        history = self.slippage_history[key]
        slippages = [entry["slippage"] for entry in history]
        
        return {
            "pair": f"{token_in}/{token_out}",
            "dex": dex,
            "chain": chain,
            "average_slippage": np.mean(slippages) * 100,
            "median_slippage": np.median(slippages) * 100,
            "max_slippage": max(slippages) * 100,
            "min_slippage": min(slippages) * 100,
            "samples": len(slippages),
            "last_updated": history[-1]["timestamp"]
        }


# Example usage
if __name__ == "__main__":
    # Initialize slippage controller
    controller = SlippageController()
    
    # Example token addresses (ETH and USDC on Ethereum)
    weth = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    usdc = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    
    # Recommend slippage tolerance
    slippage = controller.recommend_slippage_tolerance(
        "ethereum", "uniswap_v2", weth, usdc, int(1e18)  # 1 ETH
    )
    print(f"Recommended slippage tolerance: {slippage * 100:.2f}%")
    
    # Optimize trade size
    optimal_trade = controller.calculate_optimal_trade_size(
        "ethereum", "uniswap_v2", weth, usdc, int(10e18)  # 10 ETH
    )
    print(f"Optimal trade sizing: {optimal_trade}")
    
    # Find best trading route
    best_route = controller.optimize_trading_route(
        "ethereum", [weth, usdc], int(1e18)  # 1 ETH
    )
    print(f"Optimal trading route: {best_route}")
