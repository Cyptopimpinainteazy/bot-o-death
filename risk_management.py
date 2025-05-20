import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RiskManagement")

# Load environment variables
load_dotenv()

class RiskManager:
    """
    Risk management system for quantum trading.
    Implements position sizing, stop loss, take profit, and exposure limits.
    """
    
    def __init__(self, base_currency="USDC", initial_capital=None):
        """Initialize the risk manager"""
        self.base_currency = base_currency
        
        # Load initial capital from environment or use default
        if initial_capital:
            self.initial_capital = initial_capital
        else:
            env_capital = os.getenv("INITIAL_CAPITAL")
            self.initial_capital = float(env_capital) if env_capital else 10000.0
        
        # Current portfolio and risk metrics
        self.portfolio = {base_currency: self.initial_capital}
        self.positions = {}  # Current open positions
        self.trade_history = []  # History of trades
        
        # Risk parameters
        self.max_position_size_pct = 0.05  # Max 5% of portfolio per position
        self.max_daily_drawdown_pct = 0.03  # Max 3% daily drawdown
        self.max_total_exposure_pct = 0.30  # Max 30% total exposure
        self.stop_loss_pct = 0.02  # 2% stop loss per trade
        self.take_profit_pct = 0.05  # 5% take profit per trade
        
        # Quantum-adjusted risk parameters
        self.quantum_risk_adjustment = 1.0  # Multiplier for risk based on quantum signals
        
        # Performance tracking
        self.daily_pnl = {}  # Daily profit and loss
        self.total_pnl = 0  # Total profit and loss
        self.drawdowns = []  # History of drawdowns
        self.win_loss_ratio = {"wins": 0, "losses": 0}
        
        # Risk rating (1-10, where 10 is highest risk)
        self.current_risk_rating = 5
        
        logger.info(f"Risk manager initialized with {self.initial_capital} {base_currency}")
    
    def set_risk_parameters(self, params):
        """Set risk parameters"""
        if "max_position_size_pct" in params:
            self.max_position_size_pct = max(0.01, min(0.20, params["max_position_size_pct"]))
        
        if "max_daily_drawdown_pct" in params:
            self.max_daily_drawdown_pct = max(0.01, min(0.10, params["max_daily_drawdown_pct"]))
        
        if "max_total_exposure_pct" in params:
            self.max_total_exposure_pct = max(0.10, min(0.80, params["max_total_exposure_pct"]))
        
        if "stop_loss_pct" in params:
            self.stop_loss_pct = max(0.005, min(0.10, params["stop_loss_pct"]))
        
        if "take_profit_pct" in params:
            self.take_profit_pct = max(0.01, min(0.20, params["take_profit_pct"]))
        
        logger.info(f"Risk parameters updated: {params}")
    
    def adjust_quantum_risk(self, quantum_factor):
        """Adjust risk parameters based on quantum factor"""
        # Quantum factor ranges from -1 to 1
        # Use it to adjust risk parameters
        
        # Map quantum factor to risk adjustment multiplier (0.5 to 1.5)
        adjustment = 1.0 + (quantum_factor * 0.5)
        self.quantum_risk_adjustment = adjustment
        
        # Adjust risk rating based on quantum factor
        # Higher absolute value of quantum factor means higher conviction
        # Higher positive quantum factor means more bullish
        conviction = abs(quantum_factor) * 5  # 0 to 5 scale
        direction = 1 if quantum_factor > 0 else -1
        
        # Adjust risk rating (1-10 scale)
        base_rating = 5  # Neutral
        self.current_risk_rating = base_rating + (direction * conviction)
        self.current_risk_rating = max(1, min(10, round(self.current_risk_rating)))
        
        logger.info(f"Quantum risk adjustment: {adjustment}, Risk rating: {self.current_risk_rating}")
        
        return adjustment
    
    def calculate_position_size(self, symbol, price, strategy_type="standard"):
        """Calculate position size based on risk parameters and current portfolio"""
        # Get total portfolio value
        portfolio_value = self.get_portfolio_value()
        
        # Base position size on portfolio percentage
        position_size_pct = self.max_position_size_pct
        
        # Adjust based on strategy
        if strategy_type == "aggressive":
            position_size_pct *= 1.5
        elif strategy_type == "conservative":
            position_size_pct *= 0.5
        
        # Apply quantum adjustment
        position_size_pct *= self.quantum_risk_adjustment
        
        # Apply current exposure limits
        current_exposure = self.get_current_exposure()
        if current_exposure >= self.max_total_exposure_pct:
            logger.warning(f"Maximum exposure reached ({current_exposure:.2f}%), reducing position size")
            position_size_pct *= 0.5
        
        # Calculate position size in base currency
        position_value = portfolio_value * position_size_pct
        
        # Calculate position size in units of the asset
        units = position_value / price if price > 0 else 0
        
        return {
            "units": units,
            "value": position_value,
            "portfolio_pct": position_size_pct * 100
        }
    
    def validate_trade(self, trade_params):
        """Validate if a trade meets risk management criteria"""
        symbol = trade_params.get("symbol")
        price = trade_params.get("price", 0)
        units = trade_params.get("units", 0)
        side = trade_params.get("side", "buy")
        strategy = trade_params.get("strategy", "standard")
        
        if not symbol or price <= 0 or units <= 0:
            return {"valid": False, "reason": "Invalid trade parameters"}
        
        # Calculate value of the trade
        trade_value = price * units
        portfolio_value = self.get_portfolio_value()
        
        # Check if trade exceeds position size limits
        max_position_value = portfolio_value * self.max_position_size_pct
        if trade_value > max_position_value:
            return {
                "valid": False, 
                "reason": f"Position size too large. Max: {max_position_value:.2f}, Requested: {trade_value:.2f}"
            }
        
        # Check if we have enough capital for the trade
        if side == "buy":
            base_currency_balance = self.portfolio.get(self.base_currency, 0)
            if trade_value > base_currency_balance:
                return {
                    "valid": False, 
                    "reason": f"Insufficient {self.base_currency} balance"
                }
        else:  # sell
            asset_balance = self.portfolio.get(symbol, 0)
            if units > asset_balance:
                return {
                    "valid": False, 
                    "reason": f"Insufficient {symbol} balance"
                }
        
        # Check exposure limits
        current_exposure = self.get_current_exposure()
        new_exposure = current_exposure
        if side == "buy":
            new_exposure += (trade_value / portfolio_value)
        
        if new_exposure > self.max_total_exposure_pct:
            return {
                "valid": False, 
                "reason": f"Exceeds maximum exposure limit of {self.max_total_exposure_pct * 100}%"
            }
        
        # Check daily drawdown
        today = datetime.now().strftime("%Y-%m-%d")
        daily_loss = self.daily_pnl.get(today, {}).get("loss", 0)
        
        # If this is a complex strategy, factor in potential loss
        if strategy in ["sandwich", "flash", "mev"]:
            potential_loss = trade_value * 0.05  # Assume 5% potential loss for complex strategies
            if (daily_loss + potential_loss) > (portfolio_value * self.max_daily_drawdown_pct):
                return {
                    "valid": False, 
                    "reason": f"Exceeds maximum daily drawdown limit"
                }
        
        # Calculate stop loss and take profit levels
        stop_loss = price * (1 - self.stop_loss_pct) if side == "buy" else price * (1 + self.stop_loss_pct)
        take_profit = price * (1 + self.take_profit_pct) if side == "buy" else price * (1 - self.take_profit_pct)
        
        return {
            "valid": True,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": self.take_profit_pct / self.stop_loss_pct,
            "position_size": {
                "units": units,
                "value": trade_value,
                "portfolio_pct": (trade_value / portfolio_value) * 100
            }
        }
    
    def record_trade(self, trade_data):
        """Record a trade and update portfolio"""
        symbol = trade_data.get("symbol")
        price = trade_data.get("price", 0)
        units = trade_data.get("units", 0)
        side = trade_data.get("side", "buy")
        timestamp = trade_data.get("timestamp", int(time.time()))
        trade_id = trade_data.get("id", f"trade_{timestamp}")
        
        # Calculate trade value
        trade_value = price * units
        
        # Update portfolio based on trade
        if side == "buy":
            # Reduce base currency, add asset
            self.portfolio[self.base_currency] = self.portfolio.get(self.base_currency, 0) - trade_value
            self.portfolio[symbol] = self.portfolio.get(symbol, 0) + units
            
            # Add to positions
            if symbol not in self.positions:
                self.positions[symbol] = []
            
            self.positions[symbol].append({
                "id": trade_id,
                "price": price,
                "units": units,
                "value": trade_value,
                "timestamp": timestamp,
                "stop_loss": trade_data.get("stop_loss"),
                "take_profit": trade_data.get("take_profit")
            })
        
        elif side == "sell":
            # Add base currency, reduce asset
            self.portfolio[self.base_currency] = self.portfolio.get(self.base_currency, 0) + trade_value
            self.portfolio[symbol] = max(0, self.portfolio.get(symbol, 0) - units)
            
            # Calculate profit/loss if closing position
            if symbol in self.positions and self.positions[symbol]:
                # FIFO for position closing
                pnl = 0
                units_to_close = units
                positions_to_remove = []
                
                for i, position in enumerate(self.positions[symbol]):
                    if units_to_close <= 0:
                        break
                    
                    if units_to_close >= position["units"]:
                        # Close entire position
                        position_pnl = (price - position["price"]) * position["units"]
                        pnl += position_pnl
                        units_to_close -= position["units"]
                        positions_to_remove.append(i)
                    else:
                        # Partially close position
                        position_pnl = (price - position["price"]) * units_to_close
                        pnl += position_pnl
                        self.positions[symbol][i]["units"] -= units_to_close
                        self.positions[symbol][i]["value"] = self.positions[symbol][i]["units"] * position["price"]
                        units_to_close = 0
                
                # Remove closed positions
                for i in sorted(positions_to_remove, reverse=True):
                    self.positions[symbol].pop(i)
                
                # Update pnl tracking
                self.total_pnl += pnl
                
                # Update daily pnl
                today = datetime.now().strftime("%Y-%m-%d")
                if today not in self.daily_pnl:
                    self.daily_pnl[today] = {"profit": 0, "loss": 0}
                
                if pnl >= 0:
                    self.daily_pnl[today]["profit"] += pnl
                    self.win_loss_ratio["wins"] += 1
                else:
                    self.daily_pnl[today]["loss"] += abs(pnl)
                    self.win_loss_ratio["losses"] += 1
                
                trade_data["pnl"] = pnl
        
        # Add to trade history
        self.trade_history.append({
            "id": trade_id,
            "symbol": symbol,
            "side": side,
            "price": price,
            "units": units,
            "value": trade_value,
            "timestamp": timestamp,
            "pnl": trade_data.get("pnl", 0) if side == "sell" else 0
        })
        
        logger.info(f"Recorded {side} trade: {units} {symbol} at {price}")
        
        return {
            "id": trade_id,
            "status": "success",
            "portfolio": self.get_portfolio_summary()
        }
    
    def check_stop_loss_take_profit(self, current_prices):
        """Check if any positions hit stop loss or take profit levels"""
        triggered_positions = []
        
        for symbol, positions in self.positions.items():
            if not positions:
                continue
                
            current_price = current_prices.get(symbol)
            if not current_price:
                continue
            
            for position in positions:
                # Check stop loss
                stop_loss = position.get("stop_loss")
                if stop_loss and current_price <= stop_loss:
                    triggered_positions.append({
                        "symbol": symbol,
                        "units": position["units"],
                        "position_id": position["id"],
                        "type": "stop_loss",
                        "entry_price": position["price"],
                        "current_price": current_price,
                        "pnl_pct": (current_price / position["price"] - 1) * 100
                    })
                
                # Check take profit
                take_profit = position.get("take_profit")
                if take_profit and current_price >= take_profit:
                    triggered_positions.append({
                        "symbol": symbol,
                        "units": position["units"],
                        "position_id": position["id"],
                        "type": "take_profit",
                        "entry_price": position["price"],
                        "current_price": current_price,
                        "pnl_pct": (current_price / position["price"] - 1) * 100
                    })
        
        return triggered_positions
    
    def get_current_exposure(self):
        """Calculate current market exposure as percentage of portfolio"""
        portfolio_value = self.get_portfolio_value()
        
        if portfolio_value <= 0:
            return 0
        
        # Calculate total value of all positions
        position_value = 0
        for symbol, positions in self.positions.items():
            for position in positions:
                position_value += position["value"]
        
        return position_value / portfolio_value
    
    def get_portfolio_value(self, prices=None):
        """Calculate total portfolio value using current prices"""
        if not prices:
            prices = {}  # In a real system, this would fetch current prices
        
        total_value = 0
        
        # Add value of base currency
        total_value += self.portfolio.get(self.base_currency, 0)
        
        # Add value of other assets
        for symbol, amount in self.portfolio.items():
            if symbol == self.base_currency:
                continue
            
            # Use provided price or assume 1:1 if not available
            price = prices.get(symbol, 1.0)
            total_value += amount * price
        
        return total_value
    
    def get_portfolio_summary(self, prices=None):
        """Get a summary of the current portfolio"""
        if not prices:
            prices = {}
        
        portfolio_value = self.get_portfolio_value(prices)
        exposure = self.get_current_exposure()
        
        # Calculate performance metrics
        daily_pnl = 0
        today = datetime.now().strftime("%Y-%m-%d")
        if today in self.daily_pnl:
            daily_pnl = self.daily_pnl[today].get("profit", 0) - self.daily_pnl[today].get("loss", 0)
        
        # Calculate win/loss ratio
        win_rate = 0
        if self.win_loss_ratio["wins"] + self.win_loss_ratio["losses"] > 0:
            win_rate = self.win_loss_ratio["wins"] / (self.win_loss_ratio["wins"] + self.win_loss_ratio["losses"])
        
        return {
            "value": portfolio_value,
            "exposure_pct": exposure * 100,
            "assets": {symbol: amount for symbol, amount in self.portfolio.items()},
            "open_positions": sum(len(positions) for positions in self.positions.values()),
            "total_pnl": self.total_pnl,
            "daily_pnl": daily_pnl,
            "roi_pct": (portfolio_value / self.initial_capital - 1) * 100 if self.initial_capital > 0 else 0,
            "win_rate": win_rate * 100,
            "risk_rating": self.current_risk_rating
        }
    
    def adjust_for_market_conditions(self, volatility=None, trend=None):
        """Adjust risk parameters based on market conditions"""
        # Default values
        if volatility is None:
            volatility = 0.5  # Moderate volatility
        if trend is None:
            trend = 0  # Neutral trend
        
        # Volatility is between 0 (low) and 1 (high)
        # Trend is between -1 (bearish) and 1 (bullish)
        
        # Adjust position size based on volatility (reduce in high volatility)
        vol_adjustment = 1 - (volatility * 0.5)  # 0.5 to 1.0
        
        # Adjust take profit and stop loss based on volatility
        tp_adjustment = 1 + (volatility * 0.5)  # 1.0 to 1.5
        sl_adjustment = 1 + (volatility * 0.3)  # 1.0 to 1.3
        
        # Adjust exposure based on trend (increase in bullish, decrease in bearish)
        exposure_adjustment = 1 + (trend * 0.2)  # 0.8 to 1.2
        
        # Apply adjustments
        self.max_position_size_pct *= vol_adjustment
        self.take_profit_pct *= tp_adjustment
        self.stop_loss_pct *= sl_adjustment
        self.max_total_exposure_pct *= exposure_adjustment
        
        # Cap at reasonable values
        self.max_position_size_pct = max(0.01, min(0.20, self.max_position_size_pct))
        self.take_profit_pct = max(0.01, min(0.20, self.take_profit_pct))
        self.stop_loss_pct = max(0.005, min(0.10, self.stop_loss_pct))
        self.max_total_exposure_pct = max(0.10, min(0.80, self.max_total_exposure_pct))
        
        logger.info(f"Adjusted risk parameters for market conditions: "
                   f"volatility={volatility}, trend={trend}")


# Example usage
if __name__ == "__main__":
    # Initialize risk manager
    risk_manager = RiskManager(initial_capital=10000)
    
    # Adjust risk based on quantum signal
    quantum_factor = 0.6  # Strong positive signal
    risk_manager.adjust_quantum_risk(quantum_factor)
    
    # Calculate position size for ETH trade
    eth_position = risk_manager.calculate_position_size("ETH", price=3000, strategy_type="standard")
    print(f"Recommended ETH position: {eth_position}")
    
    # Validate a trade
    trade = {
        "symbol": "ETH",
        "price": 3000,
        "units": eth_position["units"],
        "side": "buy",
        "strategy": "standard"
    }
    validation = risk_manager.validate_trade(trade)
    print(f"Trade validation: {validation}")
    
    # Record the trade if valid
    if validation["valid"]:
        trade.update({
            "stop_loss": validation["stop_loss"],
            "take_profit": validation["take_profit"]
        })
        result = risk_manager.record_trade(trade)
        print(f"Trade recorded: {result}")
    
    # Check portfolio summary
    summary = risk_manager.get_portfolio_summary()
    print(f"Portfolio summary: {json.dumps(summary, indent=2)}")
