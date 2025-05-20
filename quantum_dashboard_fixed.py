#!/usr/bin/env python3
"""
Enhanced Quantum Trade AI Dashboard
A modern and interactive GUI for visualizing trading performance
"""

import tkinter as tk
from tkinter import ttk, messagebox, PhotoImage
import threading
import random
import time
import datetime
import sys
import os
import json
from collections import deque
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib
import numpy as np

# Part 1: Data Manager
#!/usr/bin/env python3
"""
Enhanced Quantum Trade AI - Data Manager
Handles data generation, wallet connections, and pricing data
"""
import os
import random
import time
import datetime
import math
import json
from collections import deque
import threading

# Configuration
CHAINS = ['ethereum', 'polygon', 'bsc', 'arbitrum_one', 'solana', 'avalanche']
TOKENS = ['ETH', 'USDC', 'WBTC', 'AAVE', 'LINK', 'UNI', 'MATIC', 'BNB', 'SOL', 'AVAX']
STRATEGIES = ['flashloan_arb', 'cross_chain_arb', 'mev_extraction', 'sandwich', 'just_in_time_liq', 'liquidation']
MARKET_CONDITIONS = ['bull', 'bear', 'sideways', 'high_volatility', 'low_volatility']

# Token icons (emoji placeholders - would be replaced with actual file paths)
TOKEN_ICONS = {
    'ETH': '🔷', 'USDC': '💵', 'WBTC': '🔶', 'AAVE': '🟣',
    'LINK': '⚓', 'UNI': '🦄', 'MATIC': '🔷', 'BNB': '🟡',
    'SOL': '☀️', 'AVAX': '❄️'
}

# Chain icons
CHAIN_ICONS = {
    'ethereum': '🔷', 'polygon': '🟣', 'bsc': '🟡',
    'arbitrum_one': '🔵', 'solana': '☀️', 'avalanche': '❄️'
}

class DataManager:
    def __init__(self):
        self.prices = {
            'ETH': 3950.42, 
            'USDC': 1.00, 
            'WBTC': 61240.78, 
            'AAVE': 92.34,
            'LINK': 15.67,
            'UNI': 7.82,
            'MATIC': 0.89,
            'BNB': 556.23,
            'SOL': 142.56,
            'AVAX': 35.78
        }
        self.price_history = {token: deque(maxlen=100) for token in TOKENS}
        for token in TOKENS:
            # Initialize with some history
            for i in range(100):
                base_price = self.prices[token]
                historical_price = base_price * (1 + random.uniform(-0.15, 0.25) * (1 - i/100))
                self.price_history[token].append(historical_price)
        
        self.trade_history = []
        self.portfolio = {token: random.uniform(0.1, 10) for token in TOKENS}
        self.portfolio_history = {token: deque(maxlen=100) for token in TOKENS}
        self.detected_market_condition = 'bull'
        self.active_strategy = 'sandwich'
        self.success_rate = 93
        self.trades_executed = 0
        self.trades_successful = 0
        self.total_profit = 0.0
        self.last_trade_time = datetime.datetime.now() - datetime.timedelta(minutes=5)
        
        # Wallet connection status
        self.wallet_connected = False
        self.wallet_address = ""
        self.wallet_balance = 0.0
        self.wallet_type = ""
        self.chain_connection_status = {chain: random.choice([True, True, True, False]) for chain in CHAINS}
        
        # Market data
        self.market_trends = {condition: random.uniform(-10, 20) for condition in MARKET_CONDITIONS}
        self.market_trends['bull'] = random.uniform(5, 20)  # Bull markets trend positive
        self.market_trends['bear'] = random.uniform(-10, -1)  # Bear markets trend negative
        
        # Liquidity pools data
        self.liquidity_pools = [
            {"name": "ETH/USDC", "platform": "Uniswap V3", "chain": "ethereum", "tvl": 120500000, "apy": 12.5},
            {"name": "WBTC/ETH", "platform": "Uniswap V3", "chain": "ethereum", "tvl": 89300000, "apy": 8.2},
            {"name": "ETH/USDC", "platform": "SushiSwap", "chain": "polygon", "tvl": 45200000, "apy": 15.7},
            {"name": "BNB/BUSD", "platform": "PancakeSwap", "chain": "bsc", "tvl": 78900000, "apy": 11.3},
            {"name": "SOL/USDC", "platform": "Raydium", "chain": "solana", "tvl": 38700000, "apy": 18.9},
            {"name": "AVAX/USDC", "platform": "TraderJoe", "chain": "avalanche", "tvl": 29800000, "apy": 16.4},
        ]
    
    def update_prices(self):
        """Update token prices with realistic movements"""
        for token in self.prices:
            # Create more volatility for some tokens
            volatility = 0.005  # Base volatility 0.5%
            if token in ['ETH', 'WBTC', 'BNB']:
                volatility = 0.008  # Higher volatility for major tokens
            
            # Adjust movement direction based on market condition
            direction_bias = 0
            if self.detected_market_condition == 'bull':
                direction_bias = 0.3
            elif self.detected_market_condition == 'bear':
                direction_bias = -0.3
            elif self.detected_market_condition == 'high_volatility':
                volatility *= 2
            
            # Calculate price change with bias
            price_change = random.normalvariate(direction_bias, volatility)
            new_price = self.prices[token] * (1 + price_change)
            
            # Ensure USDC stays close to 1
            if token == 'USDC':
                new_price = 1.0 + random.uniform(-0.001, 0.001)
                
            self.prices[token] = new_price
            self.price_history[token].append(new_price)
    
    def connect_wallet(self, wallet_type="metamask", address="0x742d35Cc6634C0532925a3b844Bc454e4438f44e"):
        """Connect to a crypto wallet"""
        self.wallet_type = wallet_type
        self.wallet_address = address
        self.wallet_connected = True
        self.wallet_balance = random.uniform(5000, 25000)
        return {
            "success": True,
            "address": address,
            "balance": self.wallet_balance,
            "connected_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def disconnect_wallet(self):
        """Disconnect wallet"""
        self.wallet_connected = False
        self.wallet_address = ""
        self.wallet_balance = 0.0
        self.wallet_type = ""
        return {"success": True}
    
    def generate_trade(self):
        """Generate a realistic trade"""
        now = datetime.datetime.now()
        
        # Don't generate trades too frequently
        if (now - self.last_trade_time).total_seconds() < 3:
            return None
        
        # Select trading pair
        base_token = random.choice(TOKENS)
        quote_token = random.choice([t for t in TOKENS if t != base_token])
        
        # Select chain
        chain = random.choice(CHAINS)
        
        # Determine strategy based on market condition
        strategy = self.active_strategy
        
        # Determine success based on strategy and market condition with optimized parameters
        # High threshold implementation - increase baseline success rate
        base_success_chance = 0.96  # Higher 96% baseline success rate
        
        # Apply strategy-specific boosts
        strategy_boost = {
            'flashloan_arb': 0.02,     # Great in volatile and bear markets
            'cross_chain_arb': 0.01,   # Good in most markets
            'mev_extraction': 0.015,   # Especially good in sideways markets
            'sandwich': 0.025,         # Best in bull markets
            'just_in_time_liq': 0.02,  # Good in high volatility
            'liquidation': 0.02       # Best in high volatility and bear markets
        }
        
        # Apply market condition modifiers
        market_modifier = {
            'bull': 0.01 if strategy == 'sandwich' else -0.005,
            'bear': 0.01 if strategy in ['flashloan_arb', 'liquidation'] else -0.005,
            'sideways': 0.01 if strategy == 'mev_extraction' else -0.002,
            'high_volatility': 0.01 if strategy in ['just_in_time_liq', 'liquidation'] else -0.005,
            'low_volatility': 0.01 if strategy == 'mev_extraction' else -0.01
        }
        
        # Calculate final success chance with all factors
        success_chance = min(0.99, base_success_chance + 
                           strategy_boost.get(strategy, 0) + 
                           market_modifier.get(self.detected_market_condition, 0))
        
        # Higher trade amounts with more significant variation
        amount = random.uniform(0.5, 3.5)  # Increased trade size
        price = self.prices[base_token]
        
        # Apply quantum optimization for higher success rate
        if random.random() < success_chance:
            success = True
            self.trades_successful += 1
            
            # Higher profit thresholds - increased from 0.1-1% to 0.5-3.5%
            # If the strategy and market condition are aligned, profit can be even higher
            base_profit_rate = random.uniform(0.005, 0.035)  # 0.5-3.5% base profit rate
            
            # Apply strategy/market alignment bonus
            if ((strategy == 'sandwich' and self.detected_market_condition == 'bull') or
                (strategy == 'flashloan_arb' and self.detected_market_condition in ['bear', 'high_volatility']) or
                (strategy == 'mev_extraction' and self.detected_market_condition == 'sideways') or
                (strategy == 'just_in_time_liq' and self.detected_market_condition == 'high_volatility') or
                (strategy == 'liquidation' and self.detected_market_condition in ['bear', 'high_volatility'])):
                # Add alignment bonus (up to additional 2%)
                alignment_bonus = random.uniform(0.005, 0.02)
                profit_rate = base_profit_rate + alignment_bonus
            else:
                profit_rate = base_profit_rate
                
            profit = amount * price * profit_rate
            self.total_profit += profit
        else:
            success = False
            profit = 0
        
        self.trades_executed += 1
        self.success_rate = (self.trades_successful / self.trades_executed) * 100 if self.trades_executed > 0 else 0
        
        # Create trade record
        trade = {
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'chain': chain,
            'strategy': strategy,
            'base_token': base_token,
            'quote_token': quote_token,
            'amount': amount,
            'price': price,
            'success': success,
            'profit': profit,
            'gas_cost': random.uniform(0.001, 0.01),
            'tx_hash': '0x' + ''.join(random.choices('0123456789abcdef', k=64)) if success else ''
        }
        
        self.trade_history.append(trade)
        self.last_trade_time = now
        
        # Update portfolio
        if success:
            self.portfolio[base_token] += amount * 0.01
            
        # Update portfolio history
        for token in self.portfolio:
            self.portfolio_history[token].append(self.portfolio[token])
            
        return trade
    
    def get_price_history(self, token, timeframe="1h"):
        """Get price history for charting"""
        if token not in self.price_history:
            return []
            
        history = list(self.price_history[token])
        # For demo, just return what we have with timestamps
        timestamps = []
        now = datetime.datetime.now()
        
        if timeframe == "1h":
            interval = 36  # seconds
        elif timeframe == "24h":
            interval = 864  # seconds
        elif timeframe == "7d":
            interval = 6048  # seconds
        else:
            interval = 36
        
        for i in range(len(history)):
            timestamps.append((now - datetime.timedelta(seconds=interval * (len(history) - i))).strftime('%H:%M:%S'))
            
        return {"timestamps": timestamps, "prices": history}
    
    def get_portfolio_allocation(self):
        """Calculate portfolio allocation percentages"""
        total_value = sum(self.portfolio[token] * self.prices[token] for token in TOKENS)
        allocation = {token: (self.portfolio[token] * self.prices[token] / total_value) * 100 for token in TOKENS}
        return allocation
        
    def get_strategy_recommendations(self):
        """Get strategy recommendations based on market condition"""
        recommendations = {
            'bull': 'sandwich',
            'bear': 'flashloan_arb',
            'sideways': 'mev_extraction',
            'high_volatility': 'just_in_time_liq',
            'low_volatility': 'mev_extraction'
        }
        return recommendations[self.detected_market_condition]


# Part 2: Dashboard UI
#!/usr/bin/env python3
"""
Enhanced Quantum Trade AI - Dashboard UI
Main GUI file with portfolio charts, wallet connection and improved UI
"""
import tkinter as tk
from tkinter import ttk, messagebox, PhotoImage
import threading
import random
import time
import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib
import numpy as np
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import data manager
from enhanced_dashboard.data_manager import DataManager, TOKENS, CHAINS, STRATEGIES, MARKET_CONDITIONS

# Configure matplotlib for dark mode
matplotlib.use("TkAgg")
style.use('dark_background')
plt.rcParams.update({
    'axes.facecolor': '#1e1e2e',
    'figure.facecolor': '#1e1e2e',
    'text.color': '#cdd6f4',
    'axes.labelcolor': '#cdd6f4',
    'xtick.color': '#cdd6f4',
    'ytick.color': '#cdd6f4',
    'grid.color': '#313244',
    'axes.edgecolor': '#45475a',
})

# Color palette based on Catppuccin Mocha
COLORS = {
    'background': '#1e1e2e',
    'surface': '#313244',
    'text': '#cdd6f4',
    'subtext': '#a6adc8',
    'primary': '#cba6f7',  # Purple/lavender
    'green': '#a6e3a1',
    'red': '#f38ba8',
    'yellow': '#f9e2af',
    'blue': '#89b4fa',
    'teal': '#94e2d5',
    'orange': '#fab387',
    'pink': '#f5c2e7',
    'mauve': '#cba6f7',
    'green_dim': '#40a02b',
    'borders': '#45475a',
}

# Token colors for charts
TOKEN_COLORS = {
    'ETH': COLORS['blue'],
    'USDC': COLORS['teal'],
    'WBTC': COLORS['orange'],
    'AAVE': COLORS['mauve'],
    'LINK': COLORS['primary'],
    'UNI': COLORS['pink'],
    'MATIC': COLORS['mauve'],
    'BNB': COLORS['yellow'],
    'SOL': COLORS['orange'],
    'AVAX': COLORS['red'],
}

class QuantumTradingDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Quantum Trade AI")
        self.root.configure(bg=COLORS['background'])
        self.root.geometry("1280x800")
        self.root.minsize(1280, 800)
        
        # Initialize data manager
        self.data_manager = DataManager()
        
        # Stop event for background threads
        self.stop_event = threading.Event()
        
        # Apply theme
        self.apply_theme()
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, style='Main.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create UI components
        self.create_header()
        self.create_dashboard()
        self.create_footer()
        
        # Start background updates in a separate thread
        self.update_thread = threading.Thread(target=self.background_updates)
        self.update_thread.daemon = True
        self.update_thread.start()
        
    def apply_theme(self):
        """Apply custom theme to widgets"""
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frames
        style.configure('Main.TFrame', background=COLORS['background'])
        style.configure('Card.TFrame', background=COLORS['surface'])
        
        # Labels
        style.configure('TLabel', background=COLORS['background'], foreground=COLORS['text'])
        style.configure('Header.TLabel', font=('Arial', 20, 'bold'), foreground=COLORS['primary'])
        style.configure('Subheader.TLabel', font=('Arial', 14, 'bold'), foreground=COLORS['text'])
        style.configure('Card.TLabel', background=COLORS['surface'], foreground=COLORS['text'])
        style.configure('Footer.TLabel', font=('Arial', 10), foreground=COLORS['subtext'])
        
        # Buttons
        style.configure('TButton', font=('Arial', 11), background=COLORS['primary'])
        style.configure('Accent.TButton', background=COLORS['primary'])
        style.configure('Secondary.TButton', background=COLORS['blue'])
        style.configure('Danger.TButton', background=COLORS['red'])
        
        # Treeview (for tables)
        style.configure('Treeview', 
                        background=COLORS['surface'], 
                        foreground=COLORS['text'],
                        fieldbackground=COLORS['surface'],
                        borderwidth=0)
        style.map('Treeview', 
                 background=[('selected', COLORS['primary'])],
                 foreground=[('selected', COLORS['background'])])
        
        style.configure('Treeview.Heading', 
                        background=COLORS['background'], 
                        foreground=COLORS['primary'],
                        font=('Arial', 10, 'bold'))
        
        # Tabs
        style.configure('TNotebook', background=COLORS['background'], tabmargins=[2, 5, 2, 0])
        style.configure('TNotebook.Tab', background=COLORS['background'], foreground=COLORS['text'],
                        padding=[10, 5], font=('Arial', 10))
        style.map('TNotebook.Tab', 
                 background=[('selected', COLORS['surface'])],
                 foreground=[('selected', COLORS['primary'])])
        
        # Entry
        style.configure('TEntry', foreground=COLORS['text'])
        
        # Combobox
        style.configure('TCombobox', 
                       fieldbackground=COLORS['surface'],
                       background=COLORS['surface'],
                       foreground=COLORS['text'])
        
        # Scrollbar
        style.configure('TScrollbar', 
                       background=COLORS['surface'],
                       troughcolor=COLORS['background'],
                       borderwidth=0,
                       arrowsize=16)
                       
    def create_header(self):
        """Create dashboard header with title and indicators"""
        header = ttk.Frame(self.main_frame, style='Main.TFrame')
        header.pack(fill=tk.X, pady=(0, 10))
        
        # Left side - Title and status
        title_frame = ttk.Frame(header, style='Main.TFrame')
        title_frame.pack(side=tk.LEFT)
        
        title = ttk.Label(title_frame, text="QUANTUM TRADE AI", style='Header.TLabel')
        title.pack(anchor=tk.W)
        
        subtitle = ttk.Label(title_frame, text="Advanced Multi-Chain Trading Dashboard", 
                            foreground=COLORS['subtext'])
        subtitle.pack(anchor=tk.W)
        
        # Right side - Wallet connection and indicators
        indicators_frame = ttk.Frame(header, style='Main.TFrame')
        indicators_frame.pack(side=tk.RIGHT)
        
        # Wallet connection status
        wallet_frame = ttk.Frame(indicators_frame, style='Main.TFrame')
        wallet_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        self.wallet_button = ttk.Button(wallet_frame, text="Connect Wallet", 
                                       command=self.toggle_wallet_connection,
                                       style='Accent.TButton')
        self.wallet_button.pack(side=tk.RIGHT)
        
        self.wallet_status = ttk.Label(wallet_frame, text="Not Connected",
                                     foreground=COLORS['red'])
        self.wallet_status.pack(side=tk.RIGHT, padx=(0, 10))
        
        # System indicators
        indicators = ttk.Frame(indicators_frame, style='Main.TFrame')
        indicators.pack(side=tk.RIGHT, padx=10)
        
        # Market condition label
        market_label = ttk.Label(indicators, text="Market:", foreground=COLORS['subtext'])
        market_label.grid(row=0, column=0, padx=(0, 5))
        
        self.market_condition = ttk.Label(indicators, text="BULL", foreground=COLORS['green'])
        self.market_condition.grid(row=0, column=1, padx=(0, 15))
        
        # Active strategy label
        strategy_label = ttk.Label(indicators, text="Strategy:", foreground=COLORS['subtext'])
        strategy_label.grid(row=0, column=2, padx=(0, 5))
        
        self.active_strategy = ttk.Label(indicators, text="SANDWICH")
        self.active_strategy.grid(row=0, column=3, padx=(0, 15))
        
        # Success rate label
        success_label = ttk.Label(indicators, text="Success Rate:", foreground=COLORS['subtext'])
        success_label.grid(row=0, column=4, padx=(0, 5))
        
        self.success_rate = ttk.Label(indicators, text="93.0%", foreground=COLORS['green'])
        self.success_rate.grid(row=0, column=5)
        
        # Date and time
        time_frame = ttk.Frame(header, style='Main.TFrame')
        time_frame.pack(side=tk.RIGHT, padx=15)
        
        self.time_label = ttk.Label(time_frame, text=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.time_label.pack()
        
    def create_dashboard(self):
        """Create main dashboard with tabs"""
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tab frames
        self.overview_frame = ttk.Frame(self.notebook, style='Main.TFrame')
        self.portfolio_frame = ttk.Frame(self.notebook, style='Main.TFrame')
        self.trades_frame = ttk.Frame(self.notebook, style='Main.TFrame')
        self.strategy_frame = ttk.Frame(self.notebook, style='Main.TFrame')
        self.pools_frame = ttk.Frame(self.notebook, style='Main.TFrame')
        
        # Add tabs to notebook
        self.notebook.add(self.overview_frame, text="Dashboard")
        self.notebook.add(self.portfolio_frame, text="Portfolio & Charts")
        self.notebook.add(self.trades_frame, text="Trade History")
        self.notebook.add(self.strategy_frame, text="Strategy Optimization")
        self.notebook.add(self.pools_frame, text="Liquidity Pools")
        
        # Create content for each tab
        self.create_overview_tab()
        self.create_portfolio_tab()
        self.create_trades_tab()
        self.create_strategy_tab()
        self.create_pools_tab()
        
    def create_overview_tab(self):
        """Create overview dashboard"""
        # Top row - Key metrics
        self.create_metrics_section()
        
        # Middle row - split between price charts and trades
        middle_frame = ttk.Frame(self.overview_frame, style='Main.TFrame')
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left side - Price charts
        charts_frame = ttk.Frame(middle_frame, style='Card.TFrame')
        charts_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        charts_label = ttk.Label(charts_frame, text="LIVE TOKEN PRICE", 
                               style='Subheader.TLabel')
        charts_label.pack(anchor=tk.W, padx=10, pady=10)
        
        # Create price chart
        self.create_price_chart(charts_frame)
        
        # Right side - Recent trades
        trades_frame = ttk.Frame(middle_frame, style='Card.TFrame')
        trades_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        trades_label = ttk.Label(trades_frame, text="RECENT TRADES", 
                              style='Subheader.TLabel')
        trades_label.pack(anchor=tk.W, padx=10, pady=10)
        
        # Create simple trades list
        self.create_simple_trades_list(trades_frame)
        
    def create_metrics_section(self):
        """Create key metrics section at the top of the dashboard"""
        # Container frame
        metrics_frame = ttk.Frame(self.overview_frame, style='Main.TFrame')
        metrics_frame.pack(fill=tk.X)
        
        # Create 4 metric cards
        self.create_metric_card(metrics_frame, "Total Profit", "$0.00", 0, 0, COLORS['green'])
        self.create_metric_card(metrics_frame, "Trades Executed", "0", 0, 1)
        self.create_metric_card(metrics_frame, "Success Rate", "0.0%", 0, 2, COLORS['green'])
        self.create_metric_card(metrics_frame, "Portfolio Value", "$0.00", 0, 3, COLORS['blue'])
        
    def create_metric_card(self, parent, title, value, row, col, color=COLORS['text']):
        """Create a metric card"""
        # Card frame
        card = ttk.Frame(parent, style='Card.TFrame')
        card.grid(row=row, column=col, padx=5, pady=5, sticky=tk.NSEW)
        parent.columnconfigure(col, weight=1)
        
        # Card title
        title_label = ttk.Label(card, text=title, style='Card.TLabel')
        title_label.pack(anchor=tk.W, padx=15, pady=(15, 5))
        
        # Card value
        value_label = ttk.Label(card, text=value, font=('Arial', 24, 'bold'), 
                              foreground=color, style='Card.TLabel')
        value_label.pack(anchor=tk.W, padx=15, pady=(0, 15))
        
        # Store reference based on title
        if title == "Total Profit":
            self.total_profit_label = value_label
        elif title == "Trades Executed":
            self.trades_executed_label = value_label
        elif title == "Success Rate":
            self.success_rate_label = value_label
        elif title == "Portfolio Value": 
            self.portfolio_value_label = value_label
    
    def create_price_chart(self, parent):
        """Create live price chart"""
        # Token selection
        selection_frame = ttk.Frame(parent, style='Card.TFrame')
        selection_frame.pack(fill=tk.X, padx=10)
        
        # Token selection
        token_label = ttk.Label(selection_frame, text="Token:", style='Card.TLabel')
        token_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.selected_token = tk.StringVar(value="ETH")
        token_combo = ttk.Combobox(selection_frame, textvariable=self.selected_token, 
                                 values=TOKENS, width=10, state="readonly")
        token_combo.pack(side=tk.LEFT, padx=(0, 15))
        
        # Timeframe selection
        timeframe_label = ttk.Label(selection_frame, text="Timeframe:", style='Card.TLabel')
        timeframe_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.selected_timeframe = tk.StringVar(value="1h")
        timeframe_combo = ttk.Combobox(selection_frame, textvariable=self.selected_timeframe, 
                                     values=["1h", "24h", "7d"], width=5, state="readonly")
        timeframe_combo.pack(side=tk.LEFT)
        
        # Bind events
        token_combo.bind("<<ComboboxSelected>>", self.update_price_chart)
        timeframe_combo.bind("<<ComboboxSelected>>", self.update_price_chart)
        
        # Create figure for the price chart
        self.price_figure = Figure(figsize=(9, 4), dpi=100, facecolor=COLORS['surface'])
        self.price_ax = self.price_figure.add_subplot(111)
        
        # Initial empty chart
        self.price_line, = self.price_ax.plot([], [], 
                                             color=COLORS['blue'], 
                                             linewidth=2)
        
        self.price_ax.set_facecolor(COLORS['surface'])
        self.price_ax.tick_params(colors=COLORS['text'])
        self.price_ax.spines['bottom'].set_color(COLORS['borders'])
        self.price_ax.spines['top'].set_color(COLORS['borders']) 
        self.price_ax.spines['right'].set_color(COLORS['borders'])
        self.price_ax.spines['left'].set_color(COLORS['borders'])
        self.price_ax.grid(True, alpha=0.3)
        
        # Create canvas
        self.price_canvas = FigureCanvasTkAgg(self.price_figure, parent)
        self.price_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initial update
        self.update_price_chart()
        
    def update_price_chart(self, event=None):
        """Update price chart with selected token and timeframe"""
        token = self.selected_token.get()
        timeframe = self.selected_timeframe.get()
        
        # Get data from data manager
        history = self.data_manager.get_price_history(token, timeframe)
        
        if not history or not history['prices']:
            return
            
        prices = history['prices']
        timestamps = history['timestamps']
        
        # Clear previous plot
        self.price_ax.clear()
        
        # Set title
        self.price_ax.set_title(f"{token} Price ({timeframe})", 
                              color=COLORS['text'], fontsize=12)
        
        # Format y-axis as currency
        self.price_ax.yaxis.set_major_formatter('${x:,.2f}')
        
        # Plot new data
        self.price_ax.plot(range(len(prices)), prices, 
                         color=TOKEN_COLORS.get(token, COLORS['blue']), 
                         linewidth=2)
        
        # Fill area under the curve with transparency
        self.price_ax.fill_between(range(len(prices)), 0, prices, 
                                 color=TOKEN_COLORS.get(token, COLORS['blue']), 
                                 alpha=0.2)
        
        # Set x-axis labels (use only a few for readability)
        tick_interval = max(1, len(timestamps) // 5)
        tick_positions = range(0, len(timestamps), tick_interval)
        self.price_ax.set_xticks(tick_positions)
        self.price_ax.set_xticklabels([timestamps[i] for i in tick_positions], rotation=30)
        
        # Style axes and grid
        self.price_ax.set_facecolor(COLORS['surface'])
        self.price_ax.tick_params(colors=COLORS['text'])
        self.price_ax.spines['bottom'].set_color(COLORS['borders'])
        self.price_ax.spines['top'].set_color(COLORS['borders']) 
        self.price_ax.spines['right'].set_color(COLORS['borders'])
        self.price_ax.spines['left'].set_color(COLORS['borders'])
        self.price_ax.grid(True, alpha=0.3)
        
        # Current price annotation
        if prices:
            current_price = prices[-1]
            self.price_ax.annotate(f"${current_price:.2f}", 
                                xy=(len(prices)-1, current_price),
                                xytext=(10, 0), 
                                textcoords="offset points",
                                color=COLORS['text'],
                                fontsize=10)
        
        # Draw canvas
        self.price_figure.tight_layout()
        self.price_canvas.draw()
        
    def create_simple_trades_list(self, parent):
        """Create a simple list of recent trades"""
        # Create frame for treeview
        tree_frame = ttk.Frame(parent, style='Card.TFrame')
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create treeview
        self.trades_tree = ttk.Treeview(tree_frame, 
                                      columns=("Time", "Chain", "Strategy", "Pair", "Amount", "Success", "Profit"),
                                      show="headings",
                                      yscrollcommand=scrollbar.set,
                                      height=8)
        
        # Configure scrollbar
        scrollbar.config(command=self.trades_tree.yview)
        
        # Configure columns
        self.trades_tree.heading("Time", text="Time")
        self.trades_tree.heading("Chain", text="Chain")
        self.trades_tree.heading("Strategy", text="Strategy")
        self.trades_tree.heading("Pair", text="Pair")
        self.trades_tree.heading("Amount", text="Amount")
        self.trades_tree.heading("Success", text="Success")
        self.trades_tree.heading("Profit", text="Profit")
        
        self.trades_tree.column("Time", width=140)
        self.trades_tree.column("Chain", width=80)
        self.trades_tree.column("Strategy", width=100)
        self.trades_tree.column("Pair", width=80)
        self.trades_tree.column("Amount", width=80)
        self.trades_tree.column("Success", width=60)
        self.trades_tree.column("Profit", width=80)
        
        self.trades_tree.pack(fill=tk.BOTH, expand=True)


# Part 3: Dashboard UI - Portfolio and Trades
def create_portfolio_tab(self):
        """Create portfolio tab with charts and allocation"""
        # Top frame - Portfolio summary and pie chart
        top_frame = ttk.Frame(self.portfolio_frame, style='Main.TFrame')
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left side - Holdings and value
        holdings_frame = ttk.Frame(top_frame, style='Card.TFrame')
        holdings_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        holdings_label = ttk.Label(holdings_frame, text="PORTFOLIO HOLDINGS", 
                                 style='Subheader.TLabel')
        holdings_label.pack(anchor=tk.W, padx=10, pady=10)
        
        # Holdings summary
        summary_frame = ttk.Frame(holdings_frame, style='Card.TFrame')
        summary_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Total value
        value_frame = ttk.Frame(summary_frame, style='Card.TFrame')
        value_frame.pack(fill=tk.X, pady=5)
        
        total_label = ttk.Label(value_frame, text="Total Portfolio Value:", 
                              style='Card.TLabel', font=('Arial', 12, 'bold'))
        total_label.pack(side=tk.LEFT, padx=10)
        
        self.portfolio_total_value = ttk.Label(value_frame, text="$0.00", 
                                            style='Card.TLabel', font=('Arial', 12, 'bold'),
                                            foreground=COLORS['blue'])
        self.portfolio_total_value.pack(side=tk.LEFT, padx=5)
        
        # Wallet status if connected
        self.wallet_info_frame = ttk.Frame(summary_frame, style='Card.TFrame')
        self.wallet_info_frame.pack(fill=tk.X, pady=5)
        
        wallet_label = ttk.Label(self.wallet_info_frame, text="Connected Wallet:", 
                              style='Card.TLabel')
        wallet_label.grid(row=0, column=0, padx=10, sticky=tk.W)
        
        self.wallet_address_label = ttk.Label(self.wallet_info_frame, text="Not Connected", 
                                          style='Card.TLabel', foreground=COLORS['red'])
        self.wallet_address_label.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        wallet_balance_label = ttk.Label(self.wallet_info_frame, text="Wallet Balance:", 
                                      style='Card.TLabel')
        wallet_balance_label.grid(row=1, column=0, padx=10, sticky=tk.W)
        
        self.wallet_balance_value = ttk.Label(self.wallet_info_frame, text="$0.00", 
                                          style='Card.TLabel')
        self.wallet_balance_value.grid(row=1, column=1, padx=5, sticky=tk.W)
        
        # Holdings grid
        self.create_holdings_grid(holdings_frame)
        
        # Right side - Pie chart
        pie_frame = ttk.Frame(top_frame, style='Card.TFrame')
        pie_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        pie_label = ttk.Label(pie_frame, text="PORTFOLIO ALLOCATION", 
                           style='Subheader.TLabel')
        pie_label.pack(anchor=tk.W, padx=10, pady=10)
        
        # Create pie chart
        self.create_portfolio_pie_chart(pie_frame)
        
        # Bottom frame - Token performance charts
        bottom_frame = ttk.Frame(self.portfolio_frame, style='Card.TFrame')
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        performance_label = ttk.Label(bottom_frame, text="TOKEN PERFORMANCE CHARTS", 
                                   style='Subheader.TLabel')
        performance_label.pack(anchor=tk.W, padx=10, pady=10)
        
        # Create token charts
        self.create_token_performance_charts(bottom_frame)
    
    def create_holdings_grid(self, parent):
        """Create a grid of holdings"""
        holdings_grid = ttk.Frame(parent, style='Card.TFrame')
        holdings_grid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create holdings display
        self.holding_frames = {}
        self.holding_labels = {}
        self.holding_values = {}
        
        # Create a grid layout
        row = 0
        col = 0
        for token in TOKENS:
            frame = ttk.Frame(holdings_grid, style='Card.TFrame')
            frame.grid(row=row, column=col, padx=5, pady=5, sticky=tk.NSEW)
            
            # Add a border (using a custom style)
            frame.configure(style='Card.TFrame')
            
            # Token name
            token_label = ttk.Label(frame, text=token, font=('Arial', 10, 'bold'), 
                                  style='Card.TLabel')
            token_label.grid(row=0, column=0, padx=10, pady=(10, 2), sticky=tk.W)
            
            # Amount
            amount = self.data_manager.portfolio.get(token, 0)
            amount_label = ttk.Label(frame, text=f"{amount:.4f}", font=('Arial', 11),
                                   style='Card.TLabel', foreground=COLORS['pink'])
            amount_label.grid(row=1, column=0, padx=10, sticky=tk.W)
            
            # USD Value
            price = self.data_manager.prices.get(token, 0)
            usd_value = amount * price
            usd_label = ttk.Label(frame, text=f"${usd_value:.2f}", font=('Arial', 10),
                                style='Card.TLabel', foreground=COLORS['green'])
            usd_label.grid(row=2, column=0, padx=10, pady=(0, 10), sticky=tk.W)
            
            # Make grid cells expandable
            holdings_grid.columnconfigure(col, weight=1)
            holdings_grid.rowconfigure(row, weight=1)
            
            # Store references for updates
            self.holding_labels[token] = amount_label
            self.holding_values[token] = usd_label
            self.holding_frames[token] = frame
            
            # Update grid position
            col += 1
            if col > 4:  # 5 columns per row
                col = 0
                row += 1
    
    def create_portfolio_pie_chart(self, parent):
        """Create portfolio allocation pie chart"""
        # Create figure
        self.pie_figure = Figure(figsize=(5, 4), dpi=100, facecolor=COLORS['surface'])
        self.pie_ax = self.pie_figure.add_subplot(111)
        
        # Create canvas
        self.pie_canvas = FigureCanvasTkAgg(self.pie_figure, parent)
        self.pie_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initial update
        self.update_portfolio_pie_chart()
    
    def update_portfolio_pie_chart(self):
        """Update the portfolio pie chart"""
        # Clear previous plot
        self.pie_ax.clear()
        
        # Calculate portfolio allocation
        allocation = self.data_manager.get_portfolio_allocation()
        
        # Filter out small allocations for better visibility
        filtered_allocation = {token: value for token, value in allocation.items() if value >= 1.0}
        other_allocation = sum(value for token, value in allocation.items() if value < 1.0)
        
        if other_allocation > 0:
            filtered_allocation['Other'] = other_allocation
        
        # Create pie chart
        tokens = list(filtered_allocation.keys())
        values = list(filtered_allocation.values())
        colors = [TOKEN_COLORS.get(token, COLORS['subtext']) for token in tokens]
        
        # Replace 'Other' color
        if 'Other' in tokens:
            colors[tokens.index('Other')] = COLORS['subtext']
        
        # Plot pie chart
        wedges, texts, autotexts = self.pie_ax.pie(
            values, 
            labels=None,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': COLORS['surface'], 'linewidth': 1.5}
        )
        
        # Style text
        for text in autotexts:
            text.set_color(COLORS['background'])
            text.set_fontsize(9)
        
        self.pie_ax.set_title("Portfolio Allocation", color=COLORS['text'], fontsize=12)
        
        # Create legend
        self.pie_ax.legend(
            wedges, 
            tokens,
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            frameon=False,
            labelcolor=COLORS['text']
        )
        
        # Set aspect ratio to be equal
        self.pie_ax.set_aspect('equal')
        
        # Update canvas
        self.pie_figure.tight_layout()
        self.pie_canvas.draw()
        
    def create_token_performance_charts(self, parent):
        """Create multi-token performance comparison chart"""
        # Selection frame
        selection_frame = ttk.Frame(parent, style='Card.TFrame')
        selection_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Create checkboxes for token selection
        self.token_vars = {}
        for i, token in enumerate(TOKENS[:5]):  # Show top 5 tokens by default
            var = tk.BooleanVar(value=True)
            self.token_vars[token] = var
            
            checkbox = ttk.Checkbutton(
                selection_frame, 
                text=token,
                variable=var,
                command=self.update_performance_chart,
                style='Card.TLabel'
            )
            checkbox.pack(side=tk.LEFT, padx=10)
        
        # Create figure
        self.performance_figure = Figure(figsize=(10, 4), dpi=100, facecolor=COLORS['surface'])
        self.performance_ax = self.performance_figure.add_subplot(111)
        
        # Create canvas
        self.performance_canvas = FigureCanvasTkAgg(self.performance_figure, parent)
        self.performance_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initial update
        self.update_performance_chart()
    
    def update_performance_chart(self):
        """Update the token performance chart"""
        # Clear previous plot
        self.performance_ax.clear()
        
        # Get selected tokens
        selected_tokens = [token for token, var in self.token_vars.items() if var.get()]
        
        if not selected_tokens:
            return
            
        # Plot each token's price history (normalized)
        for token in selected_tokens:
            history = list(self.data_manager.price_history[token])
            
            if not history:
                continue
            
            # Normalize to percentage change from first price
            first_price = history[0]
            normalized = [(price / first_price - 1) * 100 for price in history]
            
            # Plot line
            self.performance_ax.plot(
                range(len(normalized)),
                normalized,
                label=token,
                color=TOKEN_COLORS.get(token, COLORS['blue']),
                linewidth=2
            )
        
        # Style the chart
        self.performance_ax.set_title("Token Performance Comparison", color=COLORS['text'], fontsize=12)
        self.performance_ax.set_ylabel("% Change", color=COLORS['text'])
        self.performance_ax.set_facecolor(COLORS['surface'])
        self.performance_ax.tick_params(colors=COLORS['text'])
        self.performance_ax.spines['bottom'].set_color(COLORS['borders'])
        self.performance_ax.spines['top'].set_color(COLORS['borders']) 
        self.performance_ax.spines['right'].set_color(COLORS['borders'])
        self.performance_ax.spines['left'].set_color(COLORS['borders'])
        self.performance_ax.grid(True, alpha=0.3)
        
        # Create legend
        self.performance_ax.legend(
            loc="upper left",
            frameon=False,
            labelcolor=COLORS['text']
        )
        
        # Format y-axis
        self.performance_ax.yaxis.set_major_formatter('{x:,.1f}%')
        
        # Update canvas
        self.performance_figure.tight_layout()
        self.performance_canvas.draw()
        
    def create_trades_tab(self):
        """Create trades history tab"""
        # Container frame
        container = ttk.Frame(self.trades_frame, style='Card.TFrame')
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Title
        title_label = ttk.Label(container, text="TRADE HISTORY & ANALYSIS", 
                              style='Subheader.TLabel')
        title_label.pack(anchor=tk.W, padx=10, pady=10)
        
        # Controls frame
        controls = ttk.Frame(container, style='Card.TFrame')
        controls.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Filter label
        filter_label = ttk.Label(controls, text="Filter by:", style='Card.TLabel')
        filter_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Chain filter
        chain_label = ttk.Label(controls, text="Chain:", style='Card.TLabel')
        chain_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.chain_var = tk.StringVar(value="All")
        chain_combo = ttk.Combobox(controls, textvariable=self.chain_var, 
                                 values=["All"] + CHAINS, width=10, state="readonly")
        chain_combo.pack(side=tk.LEFT, padx=(0, 15))
        
        # Strategy filter
        strategy_label = ttk.Label(controls, text="Strategy:", style='Card.TLabel')
        strategy_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.strategy_var = tk.StringVar(value="All")
        strategy_combo = ttk.Combobox(controls, textvariable=self.strategy_var, 
                                    values=["All"] + STRATEGIES, width=15, state="readonly")
        strategy_combo.pack(side=tk.LEFT, padx=(0, 15))
        
        # Success filter
        success_label = ttk.Label(controls, text="Success:", style='Card.TLabel')
        success_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.success_var = tk.StringVar(value="All")
        success_combo = ttk.Combobox(controls, textvariable=self.success_var, 
                                   values=["All", "Successful", "Failed"], width=10, state="readonly")
        success_combo.pack(side=tk.LEFT)
        
        # Trades treeview
        tree_frame = ttk.Frame(container, style='Card.TFrame')
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Trades treeview
        self.full_trades_tree = ttk.Treeview(
            tree_frame, 
            columns=("Time", "Chain", "Strategy", "Pair", "Amount", "Price", "Success", "Profit", "Gas", "TxHash"),
            show="headings",
            yscrollcommand=scrollbar.set
        )
        
        # Configure scrollbar
        scrollbar.config(command=self.full_trades_tree.yview)
        
        # Configure columns
        self.full_trades_tree.heading("Time", text="Time")
        self.full_trades_tree.heading("Chain", text="Chain")
        self.full_trades_tree.heading("Strategy", text="Strategy")
        self.full_trades_tree.heading("Pair", text="Pair")
        self.full_trades_tree.heading("Amount", text="Amount")
        self.full_trades_tree.heading("Price", text="Price")
        self.full_trades_tree.heading("Success", text="Success")
        self.full_trades_tree.heading("Profit", text="Profit")
        self.full_trades_tree.heading("Gas", text="Gas Cost")
        self.full_trades_tree.heading("TxHash", text="Transaction Hash")
        
        self.full_trades_tree.column("Time", width=140)
        self.full_trades_tree.column("Chain", width=80)
        self.full_trades_tree.column("Strategy", width=100)
        self.full_trades_tree.column("Pair", width=80)
        self.full_trades_tree.column("Amount", width=80)
        self.full_trades_tree.column("Price", width=80)
        self.full_trades_tree.column("Success", width=80)
        self.full_trades_tree.column("Profit", width=80)
        self.full_trades_tree.column("Gas", width=80)
        self.full_trades_tree.column("TxHash", width=200)
        
        self.full_trades_tree.pack(fill=tk.BOTH, expand=True)
        
        # Bind double-click event to view transaction details
        self.full_trades_tree.bind("<Double-1>", self.view_transaction)
        
    def view_transaction(self, event):
        """View transaction details"""
        # Get selected item
        selection = self.full_trades_tree.selection()
        if not selection:
            return
            
        # Get transaction data
        item = self.full_trades_tree.item(selection[0])
        values = item["values"]
        
        if not values or len(values) < 10:
            return
        
        tx_hash = values[9]
        
        if not tx_hash or tx_hash == "":
            messagebox.showinfo("Transaction Details", "No transaction hash available")
            return
            
        # In a real implementation, this would open a blockchain explorer
        # For now, just show a message with transaction details
        messagebox.showinfo(
            "Transaction Details", 
            f"Transaction Hash: {tx_hash}\n\n"
            f"Chain: {values[1]}\n"
            f"Strategy: {values[2]}\n"
            f"Pair: {values[3]}\n"
            f"Amount: {values[4]}\n"
            f"Price: {values[5]}\n"
            f"Success: {values[6]}\n"
            f"Profit: {values[7]}\n"
            f"Gas Cost: {values[8]}\n"
            f"\nView on blockchain explorer: https://etherscan.io/tx/{tx_hash}"
        )


# Part 4: Dashboard UI - Strategies and Main
def create_strategy_tab(self):
        """Create strategy optimization tab"""
        # Top section - Strategy performance
        performance_frame = ttk.Frame(self.strategy_frame, style='Card.TFrame')
        performance_frame.pack(fill=tk.X, pady=(0, 10))
        
        performance_label = ttk.Label(performance_frame, text="STRATEGY PERFORMANCE", 
                                    style='Subheader.TLabel')
        performance_label.pack(anchor=tk.W, padx=10, pady=10)
        
        # Performance table frame
        table_frame = ttk.Frame(performance_frame, style='Card.TFrame')
        table_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Performance treeview
        self.performance_tree = ttk.Treeview(
            table_frame, 
            columns=("Strategy", "Market", "Success", "Profit", "Rating"),
            show="headings",
            yscrollcommand=scrollbar.set,
            height=10
        )
        
        # Configure scrollbar
        scrollbar.config(command=self.performance_tree.yview)
        
        # Configure columns
        self.performance_tree.heading("Strategy", text="Strategy")
        self.performance_tree.heading("Market", text="Market Condition")
        self.performance_tree.heading("Success", text="Success Rate")
        self.performance_tree.heading("Profit", text="Avg. Profit")
        self.performance_tree.heading("Rating", text="Rating")
        
        self.performance_tree.column("Strategy", width=150)
        self.performance_tree.column("Market", width=150)
        self.performance_tree.column("Success", width=100)
        self.performance_tree.column("Profit", width=100)
        self.performance_tree.column("Rating", width=200)
        
        self.performance_tree.pack(fill=tk.X)
        
        # Populate with initial data
        self.populate_performance_data()
        
        # Bottom section - Active strategy
        active_frame = ttk.Frame(self.strategy_frame, style='Card.TFrame')
        active_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        
        active_label = ttk.Label(active_frame, text="ACTIVE STRATEGY SETTINGS", 
                               style='Subheader.TLabel')
        active_label.pack(anchor=tk.W, padx=10, pady=10)
        
        # Strategy selection and controls
        controls_frame = ttk.Frame(active_frame, style='Card.TFrame')
        controls_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Strategy selection
        strategy_frame = ttk.Frame(controls_frame, style='Card.TFrame')
        strategy_frame.pack(fill=tk.X, pady=5)
        
        strategy_label = ttk.Label(strategy_frame, text="Select Strategy:", 
                                 style='Card.TLabel')
        strategy_label.pack(side=tk.LEFT, padx=10)
        
        self.select_strategy_var = tk.StringVar(value=self.data_manager.active_strategy)
        strategy_combo = ttk.Combobox(
            strategy_frame, 
            textvariable=self.select_strategy_var, 
            values=STRATEGIES, 
            width=20, 
            state="readonly"
        )
        strategy_combo.pack(side=tk.LEFT, padx=5)
        
        activate_button = ttk.Button(
            strategy_frame, 
            text="Activate Strategy", 
            command=self.activate_strategy,
            style='Accent.TButton'
        )
        activate_button.pack(side=tk.LEFT, padx=10)
        
        # Auto optimization
        auto_frame = ttk.Frame(controls_frame, style='Card.TFrame')
        auto_frame.pack(fill=tk.X, pady=5)
        
        self.auto_var = tk.BooleanVar(value=True)
        auto_check = ttk.Checkbutton(
            auto_frame, 
            text="Auto-select optimal strategy based on market conditions", 
            variable=self.auto_var,
            style='Card.TLabel'
        )
        auto_check.pack(anchor=tk.W, padx=10, pady=5)
        
        # Market condition selector for simulation
        market_frame = ttk.Frame(controls_frame, style='Card.TFrame')
        market_frame.pack(fill=tk.X, pady=5)
        
        market_label = ttk.Label(market_frame, text="Simulate Market Condition:", 
                               style='Card.TLabel')
        market_label.pack(side=tk.LEFT, padx=10)
        
        self.market_var = tk.StringVar(value=self.data_manager.detected_market_condition)
        market_combo = ttk.Combobox(
            market_frame, 
            textvariable=self.market_var, 
            values=MARKET_CONDITIONS, 
            width=15, 
            state="readonly"
        )
        market_combo.pack(side=tk.LEFT, padx=5)
        
        market_button = ttk.Button(
            market_frame, 
            text="Apply Market Condition", 
            command=self.apply_market_condition,
            style='Secondary.TButton'
        )
        market_button.pack(side=tk.LEFT, padx=10)
        
        # Strategy parameters
        params_frame = ttk.Frame(active_frame, style='Card.TFrame')
        params_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        params_label = ttk.Label(params_frame, text="Strategy Parameters", 
                               style='Card.TLabel', font=('Arial', 11, 'bold'))
        params_label.pack(anchor=tk.W, padx=10, pady=10)
        
        # Advanced strategy parameters in a grid
        grid_frame = ttk.Frame(params_frame, style='Card.TFrame')
        grid_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Create parameter fields with high-threshold values
        param_names = [
            "Max Gas Price (Gwei)", 
            "Min Profit Threshold ($)", 
            "Max Slippage (%)", 
            "Trade Size Multiplier", 
            "Quantum Optimization Level",
            "Success Rate Threshold (%)",
            "Position Size Limit (% of Capital)"
        ]
        # High threshold settings with more aggressive values
        param_values = [25, 2.5, 0.25, 1.5, 15, 96, 7.5]
        
        self.param_vars = []
        
        for i, (name, value) in enumerate(zip(param_names, param_values)):
            label = ttk.Label(grid_frame, text=name + ":", style='Card.TLabel')
            label.grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
            
            var = tk.DoubleVar(value=value)
            self.param_vars.append(var)
            
            entry = ttk.Entry(grid_frame, textvariable=var, width=10)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky=tk.W)
            
        # Apply button
        apply_params_button = ttk.Button(
            grid_frame, 
            text="Apply Parameters", 
            command=self.apply_parameters,
            style='Accent.TButton'
        )
        apply_params_button.grid(row=len(param_names), column=0, columnspan=2, 
                               padx=10, pady=15, sticky=tk.EW)
                               
    def populate_performance_data(self):
        """Populate the performance tree with realistic strategy data"""
        # Clear existing items
        for item in self.performance_tree.get_children():
            self.performance_tree.delete(item)
            
        # Enhanced strategy performance data with high thresholds
        data = [
            # High threshold flashloan arbitrage - excellent for bear & high volatility
            ("flashloan_arb", "bear", 97, 2.45, "HIGHLY RECOMMENDED"),
            ("flashloan_arb", "sideways", 94, 1.85, "HIGHLY RECOMMENDED"),
            ("flashloan_arb", "high_volatility", 100, 3.65, "HIGHLY RECOMMENDED"),
            ("flashloan_arb", "low_volatility", 88, 1.20, "RECOMMENDED"),
            ("flashloan_arb", "bull", 92, 1.55, "RECOMMENDED"),
            
            # Cross-chain arbitrage - steady performer
            ("cross_chain_arb", "sideways", 95, 1.45, "HIGHLY RECOMMENDED"),
            ("cross_chain_arb", "high_volatility", 98, 2.80, "HIGHLY RECOMMENDED"),
            ("cross_chain_arb", "bull", 96, 2.10, "HIGHLY RECOMMENDED"),
            ("cross_chain_arb", "bear", 92, 1.95, "RECOMMENDED"),
            ("cross_chain_arb", "low_volatility", 84, 1.25, "RECOMMENDED"),
            
            # MEV extraction - excellent for sideways
            ("mev_extraction", "bull", 97, 3.20, "HIGHLY RECOMMENDED"),
            ("mev_extraction", "bear", 94, 1.75, "HIGHLY RECOMMENDED"),
            ("mev_extraction", "sideways", 99, 2.85, "HIGHLY RECOMMENDED"),
            ("mev_extraction", "high_volatility", 97, 2.25, "HIGHLY RECOMMENDED"),
            ("mev_extraction", "low_volatility", 92, 1.45, "RECOMMENDED"),
            
            # Sandwich - best for bull markets
            ("sandwich", "bull", 99, 4.95, "HIGHLY RECOMMENDED"),
            ("sandwich", "high_volatility", 95, 3.80, "HIGHLY RECOMMENDED"),
            ("sandwich", "sideways", 88, 1.65, "RECOMMENDED"),
            ("sandwich", "bear", 82, 1.20, "RECOMMENDED"),
            ("sandwich", "low_volatility", 78, 0.95, "NOT RECOMMENDED"),
            
            # Just-in-time liquidity - great for high volatility
            ("just_in_time_liq", "bull", 100, 2.95, "HIGHLY RECOMMENDED"),
            ("just_in_time_liq", "high_volatility", 99, 4.25, "HIGHLY RECOMMENDED"),
            ("just_in_time_liq", "bear", 93, 1.85, "RECOMMENDED"),
            ("just_in_time_liq", "sideways", 92, 1.35, "RECOMMENDED"),
            ("just_in_time_liq", "low_volatility", 85, 0.90, "RECOMMENDED"),
            
            # Liquidation - high volatility specialist
            ("liquidation", "high_volatility", 99, 4.75, "HIGHLY RECOMMENDED"),
            ("liquidation", "bear", 97, 3.65, "HIGHLY RECOMMENDED"),
            ("liquidation", "bull", 93, 2.15, "RECOMMENDED"),
            ("liquidation", "sideways", 90, 1.45, "RECOMMENDED"),
            ("liquidation", "low_volatility", 82, 0.85, "NOT RECOMMENDED")
        ]
        
        # Insert data into tree
        for strategy, market, success, profit, rating in data:
            tag = "recommended"
            if rating == "HIGHLY RECOMMENDED":
                tag = "highly_recommended"
            elif rating == "NOT RECOMMENDED":
                tag = "not_recommended"
                
            self.performance_tree.insert("", "end", values=(
                strategy,
                market,
                f"{success}%",
                f"${profit}",
                rating
            ), tags=(tag,))
            
        # Configure tags for colors
        self.performance_tree.tag_configure("highly_recommended", foreground=COLORS['green'])
        self.performance_tree.tag_configure("recommended", foreground=COLORS['yellow'])
        self.performance_tree.tag_configure("not_recommended", foreground=COLORS['red'])
    
    def activate_strategy(self):
        """Activate the selected strategy"""
        strategy = self.select_strategy_var.get()
        self.data_manager.active_strategy = strategy
        self.update_active_strategy()
        
        messagebox.showinfo("Strategy Activation", f"Strategy '{strategy}' has been activated")
        
    def apply_parameters(self):
        """Apply the strategy parameters"""
        params = [var.get() for var in self.param_vars]
        
        # In a real implementation, these would affect the strategy
        # For now, just show a confirmation message
        messagebox.showinfo(
            "Parameters Applied", 
            f"Strategy parameters have been updated:\n\n"
            f"Max Gas Price: {params[0]} Gwei\n"
            f"Min Profit Threshold: ${params[1]:.2f}\n"
            f"Max Slippage: {params[2]}%\n"
            f"Trade Size Multiplier: {params[3]:.1f}x\n"
            f"Quantum Optimization Level: {int(params[4])}"
        )
        
    def apply_market_condition(self):
        """Apply the selected market condition for simulation"""
        market = self.market_var.get()
        self.data_manager.detected_market_condition = market
        
        # Update optimal strategy based on market
        if self.auto_var.get():
            recommended_strategy = self.data_manager.get_strategy_recommendations()
            self.data_manager.active_strategy = recommended_strategy
            self.select_strategy_var.set(recommended_strategy)
        
        # Update the UI
        self.update_market_condition()
        self.update_active_strategy()
        
        messagebox.showinfo(
            "Market Condition", 
            f"Market condition changed to {market.upper()}"
        )
    
    def create_pools_tab(self):
        """Create liquidity pools tab"""
        # Top frame - Pool management
        pools_frame = ttk.Frame(self.pools_frame, style='Card.TFrame')
        pools_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Title
        title_label = ttk.Label(pools_frame, text="LIQUIDITY POOLS MANAGEMENT", 
                              style='Subheader.TLabel')
        title_label.pack(anchor=tk.W, padx=10, pady=10)
        
        # Wallet status
        wallet_frame = ttk.Frame(pools_frame, style='Card.TFrame')
        wallet_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Connect wallet button if not connected
        self.pool_wallet_button = ttk.Button(
            wallet_frame, 
            text="Connect Wallet to Manage Pools", 
            command=self.toggle_wallet_connection,
            style='Accent.TButton'
        )
        self.pool_wallet_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Add liquidity button
        self.add_liquidity_button = ttk.Button(
            wallet_frame, 
            text="Add Liquidity", 
            command=self.add_liquidity,
            style='Secondary.TButton',
            state=tk.DISABLED
        )
        self.add_liquidity_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Available pools treeview
        pools_label = ttk.Label(pools_frame, text="Available Liquidity Pools", 
                              style='Card.TLabel', font=('Arial', 11, 'bold'))
        pools_label.pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        # Pools treeview frame
        tree_frame = ttk.Frame(pools_frame, style='Card.TFrame')
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Pools treeview
        self.pools_tree = ttk.Treeview(
            tree_frame, 
            columns=("Name", "Platform", "Chain", "TVL", "APY", "Action"),
            show="headings",
            yscrollcommand=scrollbar.set
        )
        
        # Configure scrollbar
        scrollbar.config(command=self.pools_tree.yview)
        
        # Configure columns
        self.pools_tree.heading("Name", text="Pool Name")
        self.pools_tree.heading("Platform", text="Platform")
        self.pools_tree.heading("Chain", text="Chain")
        self.pools_tree.heading("TVL", text="TVL")
        self.pools_tree.heading("APY", text="APY")
        self.pools_tree.heading("Action", text="Action")
        
        self.pools_tree.column("Name", width=150)
        self.pools_tree.column("Platform", width=150)
        self.pools_tree.column("Chain", width=100)
        self.pools_tree.column("TVL", width=150)
        self.pools_tree.column("APY", width=100)
        self.pools_tree.column("Action", width=150)
        
        self.pools_tree.pack(fill=tk.BOTH, expand=True)
        
        # Populate pools
        self.update_liquidity_pools()
        
    def update_liquidity_pools(self):
        """Update the liquidity pools treeview"""
        # Clear existing items
        for item in self.pools_tree.get_children():
            self.pools_tree.delete(item)
            
        # Get pool data from data manager
        pools = self.data_manager.liquidity_pools
        
        # Insert data into tree
        for pool in pools:
            self.pools_tree.insert("", "end", values=(
                pool["name"],
                pool["platform"],
                pool["chain"],
                f"${pool['tvl']:,.2f}",
                f"{pool['apy']:.1f}%",
                "Manage" if self.data_manager.wallet_connected else "Connect Wallet"
            ))
            
    def add_liquidity(self):
        """Show dialog to add liquidity"""
        if not self.data_manager.wallet_connected:
            messagebox.showinfo("Wallet Required", "Please connect your wallet first")
            return
            
        # In a real implementation, this would open a dialog to add liquidity
        # For now, just show a message
        messagebox.showinfo(
            "Add Liquidity", 
            "This would open a dialog to add liquidity to a selected pool.\n\n"
            "In a production implementation, you would be able to select tokens, "
            "amounts, and the target pool."
        )
    
    def create_footer(self):
        """Create footer with status and version info"""
        footer = ttk.Frame(self.main_frame, style='Main.TFrame')
        footer.pack(fill=tk.X, pady=(10, 0))
        
        # Status indicator
        self.status_label = ttk.Label(footer, text="System Status: ONLINE", 
                                    foreground=COLORS['green'])
        self.status_label.pack(side=tk.LEFT)
        
        # Version info
        version_label = ttk.Label(footer, text="Enhanced Quantum Trade AI v2.0.0", 
                               style='Footer.TLabel')
        version_label.pack(side=tk.RIGHT)
        
    def toggle_wallet_connection(self):
        """Toggle wallet connection status"""
        if not self.data_manager.wallet_connected:
            # Connect wallet
            result = self.data_manager.connect_wallet()
            if result["success"]:
                self.wallet_status.config(text=result["address"][:8] + "...", 
                                        foreground=COLORS['green'])
                self.wallet_button.config(text="Disconnect Wallet")
                self.wallet_address_label.config(text=result["address"][:8] + "...", 
                                              foreground=COLORS['green'])
                self.wallet_balance_value.config(text=f"${result['balance']:.2f}")
                
                # Enable pool management
                self.add_liquidity_button.config(state=tk.NORMAL)
                self.pool_wallet_button.config(text="Wallet Connected")
                
                # Update pools view
                self.update_liquidity_pools()
                
                messagebox.showinfo(
                    "Wallet Connected", 
                    f"Successfully connected to wallet\n\n"
                    f"Address: {result['address'][:8]}...{result['address'][-6:]}\n"
                    f"Balance: ${result['balance']:.2f}"
                )
        else:
            # Disconnect wallet
            result = self.data_manager.disconnect_wallet()
            if result["success"]:
                self.wallet_status.config(text="Not Connected", foreground=COLORS['red'])
                self.wallet_button.config(text="Connect Wallet")
                self.wallet_address_label.config(text="Not Connected", 
                                              foreground=COLORS['red'])
                self.wallet_balance_value.config(text="$0.00")
                
                # Disable pool management
                self.add_liquidity_button.config(state=tk.DISABLED)
                self.pool_wallet_button.config(text="Connect Wallet to Manage Pools")
                
                # Update pools view
                self.update_liquidity_pools()
    
    def background_updates(self):
        """Background thread for updating data"""
        last_trade_time = time.time()
        last_time_update = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Update time every second
                current_time = time.time()
                if current_time - last_time_update >= 1:
                    self.time_label.config(text=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    last_time_update = current_time
                
                # Update prices every 1 second
                self.data_manager.update_prices()
                self.update_prices()
                
                # Generate a new trade every 3-10 seconds
                if current_time - last_trade_time > random.uniform(3, 10):
                    trade = self.data_manager.generate_trade()
                    if trade:
                        self.update_trades(trade)
                        self.update_metrics()
                        last_trade_time = current_time
                
                # Update UI every two seconds
                self.update_market_condition()
                self.update_active_strategy()
                self.update_success_rate()
                self.update_portfolio()
                
                # Update charts every 5 seconds
                if int(current_time) % 5 == 0:
                    self.update_price_chart()
                    self.update_portfolio_pie_chart()
                    self.update_performance_chart()
                
                time.sleep(0.5)
            except Exception as e:
                print(f"Error in background thread: {e}")
                time.sleep(1)
    
    def update_prices(self):
        """Update the price displays"""
        for token, frame in self.holding_frames.items():
            # Update amount
            amount = self.data_manager.portfolio.get(token, 0)
            self.holding_labels[token].config(text=f"{amount:.4f}")
            
            # Update USD value
            price = self.data_manager.prices.get(token, 0)
            usd_value = amount * price
            self.holding_values[token].config(text=f"${usd_value:.2f}")
            
        # Update total portfolio value
        total_value = sum(self.data_manager.portfolio[token] * self.data_manager.prices[token] 
                         for token in self.data_manager.portfolio)
        self.portfolio_total_value.config(text=f"${total_value:.2f}")
        self.portfolio_value_label.config(text=f"${total_value:.2f}")
    
    def update_trades(self, trade):
        """Update the trades displays with a new trade"""
        # Format trade for display
        pair = f"{trade['base_token']}/{trade['quote_token']}"
        success_text = "✓" if trade['success'] else "✗"
        success_color = COLORS['green'] if trade['success'] else COLORS['red']
        profit_text = f"${trade['profit']:.4f}" if trade['success'] else "—"
        
        # Insert into dashboard trades list
        item = self.trades_tree.insert("", 0, values=(
            trade['timestamp'],
            trade['chain'],
            trade['strategy'],
            pair,
            f"{trade['amount']:.2f}",
            success_text,
            profit_text
        ), tags=("success" if trade['success'] else "failure"))
        
        # Configure tags for colors
        self.trades_tree.tag_configure("success", foreground=COLORS['green'])
        self.trades_tree.tag_configure("failure", foreground=COLORS['red'])
        
        # Keep only the latest 8 trades
        if len(self.trades_tree.get_children()) > 8:
            self.trades_tree.delete(self.trades_tree.get_children()[-1])
            
        # Insert into full trades list
        item = self.full_trades_tree.insert("", 0, values=(
            trade['timestamp'],
            trade['chain'],
            trade['strategy'],
            pair,
            f"{trade['amount']:.2f}",
            f"${trade['price']:.2f}",
            success_text,
            f"${trade['profit']:.4f}" if trade['success'] else "—",
            f"${trade['gas_cost']:.4f}",
            trade['tx_hash'] if trade['success'] else ""
        ), tags=("success" if trade['success'] else "failure"))
        
        # Configure tags for colors
        self.full_trades_tree.tag_configure("success", foreground=COLORS['green'])
        self.full_trades_tree.tag_configure("failure", foreground=COLORS['red'])
        
    def update_metrics(self):
        """Update the key metrics"""
        # Update total profit
        total_profit = self.data_manager.total_profit
        self.total_profit_label.config(text=f"${total_profit:.2f}")
        
        # Update trades executed
        trades_executed = self.data_manager.trades_executed
        self.trades_executed_label.config(text=str(trades_executed))
        
        # Update success rate
        success_rate = self.data_manager.success_rate
        self.success_rate_label.config(text=f"{success_rate:.1f}%")
        
        # Set color based on rate
        if success_rate >= 90:
            color = COLORS['green']
        elif success_rate >= 75:
            color = COLORS['yellow']
        else:
            color = COLORS['red']
            
        self.success_rate.config(text=f"{success_rate:.1f}%", foreground=color)
        
    def update_market_condition(self):
        """Update the market condition indicator"""
        condition = self.data_manager.detected_market_condition
        
        # Set text and color based on condition
        if condition == "bull":
            self.market_condition.config(text="BULL", foreground=COLORS['green'])
        elif condition == "bear":
            self.market_condition.config(text="BEAR", foreground=COLORS['red'])
        elif condition == "sideways":
            self.market_condition.config(text="SIDEWAYS", foreground=COLORS['teal'])
        elif condition == "high_volatility":
            self.market_condition.config(text="HIGH VOL", foreground=COLORS['yellow'])
        elif condition == "low_volatility":
            self.market_condition.config(text="LOW VOL", foreground=COLORS['mauve'])
            
    def update_active_strategy(self):
        """Update the active strategy indicator"""
        strategy = self.data_manager.active_strategy
        self.active_strategy.config(text=strategy.upper())
        
    def update_portfolio(self):
        """Update the portfolio display"""
        # Total portfolio value calculation
        total_value = sum(self.data_manager.portfolio[token] * self.data_manager.prices[token] 
                         for token in TOKENS)
        
        # If wallet is connected, add wallet balance
        if self.data_manager.wallet_connected:
            total_value += self.data_manager.wallet_balance
            
        self.portfolio_value_label.config(text=f"${total_value:.2f}")
        self.portfolio_total_value.config(text=f"${total_value:.2f}")
            
    def on_closing(self):
        """Handle window closing"""
        self.stop_event.set()
        self.root.destroy()
        

def main():
    """Main function to run the dashboard"""
    # Set up the root window
    root = tk.Tk()
    root.title("Enhanced Quantum Trade AI Dashboard")
    
    # Create the dashboard
    app = QuantumTradingDashboard(root)
    
    # Set up closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the main loop
    root.mainloop()
    

if __name__ == "__main__":
    main()


# If this file is run directly, start the dashboard
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error starting dashboard: {e}")
