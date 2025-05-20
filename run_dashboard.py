#!/usr/bin/env python3
"""
Enhanced Quantum Trade AI - Clean Dashboard Launcher
This file provides a clean implementation to launch the trading dashboard
"""
from logging_config import log_bot_activity, log_trade
import os
import sys
import time
import random
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from custom_widgets import ToggleSwitch
import threading
from collections import deque

# Constants
TOKENS = ['ETH', 'USDC', 'WBTC', 'AAVE', 'LINK', 'UNI', 'MATIC', 'BNB', 'SOL', 'AVAX']
STRATEGIES = ['sandwich', 'flashloan_arb', 'cross_chain_arb', 'mev_extraction', 'just_in_time_liq', 'liquidation']
MARKET_CONDITIONS = ['bull', 'bear', 'sideways', 'high_volatility', 'low_volatility']
CHAINS = ['ethereum', 'polygon', 'arbitrum', 'optimism', 'bsc', 'avalanche', 'solana']

# Colors
COLORS = {
    'background': '#1e1e2e',
    'card': '#313244',
    'text': '#cdd6f4',
    'accent': '#89b4fa',
    'green': '#a6e3a1',
    'red': '#f38ba8',
    'yellow': '#f9e2af',
    'blue': '#89b4fa',
    'purple': '#cba6f7',
    'teal': '#94e2d5',
    'mauve': '#cba6f7',
}

class DataGenerator:
    """Simple data generator for the dashboard"""
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
        self.active_strategies = {strategy: False for strategy in STRATEGIES}
        self.active_strategies['sandwich'] = True  # Default strategy
        self.success_rate = 95
        self.trades_executed = 0
        self.trades_successful = 0
        self.total_profit = 0.0
        self.last_trade_time = datetime.datetime.now() - datetime.timedelta(minutes=5)
        
    def update_prices(self):
        """Update token prices with stable movements"""
        # Very low volatility for stable GUI
        for token in self.prices:
            volatility = 0.0005  # Super low volatility
            
            # Very small direction bias
            direction_bias = 0
            if self.detected_market_condition == 'bull':
                direction_bias = 0.0001
            elif self.detected_market_condition == 'bear':
                direction_bias = -0.0001
            
            # Calculate price change with almost no volatility
            price_change = random.normalvariate(direction_bias, volatility)
            self.prices[token] *= (1 + price_change)
            
            # Add to price history
            self.price_history[token].append(self.prices[token])
    
    def generate_trade(self):
        # Get active strategies
        active_strats = [s for s, is_active in self.active_strategies.items() if is_active]
        if not active_strats:
            return None
            
        # Pick a random active strategy
        strategy = random.choice(active_strats)
        
        # Generate trade based on strategy
        token = random.choice(TOKENS)
        amount = random.uniform(0.1, 2.0)
        price = self.prices[token]
        profit = random.uniform(0.001, 0.05) * amount * price
        now = datetime.datetime.now()
        
        # Choose chain
        chain = random.choice(CHAINS)
        
        # Transaction details (with stable values)
        amount = random.uniform(0.5, 2.0)
        price = self.prices[token]
        
        # High success rate
        success = random.random() < 0.96
        
        if success:
            self.trades_successful += 1
            profit = amount * price * random.uniform(0.005, 0.02)
            self.total_profit += profit
        else:
            profit = 0
        
        self.trades_executed += 1
        self.success_rate = (self.trades_successful / self.trades_executed) * 100
        
        # Create trade record
        trade = {
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'chain': chain,
            'strategy': strategy,
            'base_token': token,
            'quote_token': token,
            'amount': amount,
            'price': price,
            'success': success,
            'profit': profit,
            'tx_hash': '0x' + ''.join(random.choices('0123456789abcdef', k=64)) if success else ''
        }
        
        self.trade_history.append(trade)
        self.last_trade_time = trade['timestamp']
        
        # Log the trade
        log_trade(trade)
        
        if success:
            self.portfolio[token] += amount * 0.01
            
        return trade


class TradingDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Quantum Trade AI Dashboard")
        self.root.geometry("1200x800")
        self.root.config(bg=COLORS['background'])
        
        # Setup theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        # Data source
        self.data = DataGenerator()
        
        # Create UI
        self.create_main_layout()
        
        # Start data updates
        self.stop_event = threading.Event()
        self.update_thread = threading.Thread(target=self.background_updates)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def configure_styles(self):
        """Setup ttk styles for dark theme"""
        self.style.configure('TFrame', background=COLORS['background'])
        self.style.configure('Card.TLabelframe', background=COLORS['card'],
                            foreground=COLORS['text'], borderwidth=2)
        self.style.configure('Card.TLabelframe.Label', background=COLORS['card'],
                            foreground=COLORS['text'], font=('Helvetica', 12, 'bold'))
        self.style.configure('Label.TLabel', background=COLORS['card'],
                            foreground=COLORS['text'], font=('Helvetica', 10))
        self.style.configure('Toggle.TCheckbutton', background=COLORS['card'],
                            foreground=COLORS['text'])
        self.style.configure("TFrame", background=COLORS['background'])
        self.style.configure("Card.TFrame", background=COLORS['card'])
        
        self.style.configure("TLabel", background=COLORS['background'], foreground=COLORS['text'])
        self.style.configure("Card.TLabel", background=COLORS['card'], foreground=COLORS['text'])
        self.style.configure("Header.TLabel", font=("Arial", 16, "bold"))
        self.style.configure("Subheader.TLabel", font=("Arial", 12, "bold"))
        
        self.style.configure("TButton", background=COLORS['card'], foreground=COLORS['text'])
        self.style.map("TButton", background=[("active", COLORS['accent'])])
        
        self.style.configure("TNotebook", background=COLORS['background'])
        self.style.configure("TNotebook.Tab", background=COLORS['card'], foreground=COLORS['text'], padding=[10, 2])
        self.style.map("TNotebook.Tab", background=[("selected", COLORS['accent'])], 
                       foreground=[("selected", COLORS['text'])])
        
        self.style.configure("Treeview", 
                            background=COLORS['card'],
                            foreground=COLORS['text'],
                            fieldbackground=COLORS['card'])
        self.style.map("Treeview", background=[("selected", COLORS['accent'])])
    
    def create_main_layout(self):
        """Create the main dashboard layout"""
        # Main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        self.create_header()
        
        # Notebook with tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Dashboard tab
        self.dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.dashboard_frame, text="Dashboard")
        self.create_dashboard()
        
        # Trades tab
        self.trades_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.trades_frame, text="Trades")
        self.create_trades_view()
        
        # Portfolio tab
        self.portfolio_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.portfolio_frame, text="Portfolio")
        self.create_portfolio_view()
        
        # Create footer
        self.create_footer()
    
    def create_header(self):
        """Create dashboard header"""
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left - Logo and title
        left_frame = ttk.Frame(header_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        title_label = ttk.Label(left_frame, text="ENHANCED QUANTUM TRADE AI", 
                              style="Header.TLabel")
        title_label.pack(side=tk.LEFT, padx=10)
        
        # Right - Status indicators
        right_frame = ttk.Frame(header_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Market condition indicator
        market_label = ttk.Label(right_frame, text="MARKET:", style="Card.TLabel")
        market_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.market_condition = ttk.Label(right_frame, text="BULL", foreground=COLORS['green'])
        self.market_condition.pack(side=tk.LEFT, padx=(0, 15))
        
        # Strategy indicator
        strategy_label = ttk.Label(right_frame, text="STRATEGY:", style="Card.TLabel")
        strategy_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.active_strategy = ttk.Label(right_frame, text="SANDWICH", foreground=COLORS['blue'])
        self.active_strategy.pack(side=tk.LEFT, padx=(0, 15))
        
        # Success rate indicator
        success_label = ttk.Label(right_frame, text="SUCCESS RATE:", style="Card.TLabel")
        success_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.success_rate = ttk.Label(right_frame, text="95%", foreground=COLORS['green'])
        self.success_rate.pack(side=tk.LEFT)
    
    def create_dashboard(self):
        """Create main dashboard view"""
        # Top metrics row
        metrics_frame = ttk.Frame(self.dashboard_frame, style="Card.TFrame")
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Key metrics in a grid
        metrics_frame.columnconfigure((0, 1, 2, 3), weight=1)
        
        # Portfolio value
        ttk.Label(metrics_frame, text="PORTFOLIO VALUE", style="Card.TLabel").grid(row=0, column=0, padx=10, pady=5)
        self.portfolio_value_label = ttk.Label(metrics_frame, text="$123,456.78", 
                                           font=("Arial", 16, "bold"), style="Card.TLabel")
        self.portfolio_value_label.grid(row=1, column=0, padx=10, pady=5)
        
        # 24h change
        ttk.Label(metrics_frame, text="24H CHANGE", style="Card.TLabel").grid(row=0, column=1, padx=10, pady=5)
        self.change_24h_label = ttk.Label(metrics_frame, text="+5.67%", 
                                       font=("Arial", 16, "bold"), 
                                       foreground=COLORS['green'], style="Card.TLabel")
        self.change_24h_label.grid(row=1, column=1, padx=10, pady=5)
        
        # Total profit
        ttk.Label(metrics_frame, text="TOTAL PROFIT", style="Card.TLabel").grid(row=0, column=2, padx=10, pady=5)
        self.total_profit_label = ttk.Label(metrics_frame, text="$7,890.12", 
                                         font=("Arial", 16, "bold"), 
                                         foreground=COLORS['green'], style="Card.TLabel")
        self.total_profit_label.grid(row=1, column=2, padx=10, pady=5)
        
        # Trades executed
        ttk.Label(metrics_frame, text="TRADES EXECUTED", style="Card.TLabel").grid(row=0, column=3, padx=10, pady=5)
        self.trades_executed_label = ttk.Label(metrics_frame, text="42", 
                                            font=("Arial", 16, "bold"), style="Card.TLabel")
        self.trades_executed_label.grid(row=1, column=3, padx=10, pady=5)
        
        # Prices tracking section
        prices_frame = ttk.Frame(self.dashboard_frame, style="Card.TFrame")
        prices_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(prices_frame, text="LIVE PRICES", 
               style="Subheader.TLabel").pack(anchor=tk.W, padx=10, pady=10)
        
        # Create a grid for price displays
        price_grid = ttk.Frame(prices_frame, style="Card.TFrame")
        price_grid.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Configure grid columns
        price_grid.columnconfigure((0, 1, 2, 3, 4), weight=1)
        
        # Create price displays for tokens
        self.price_labels = {}
        self.price_change_labels = {}
        
        # Create rows of token prices
        for i, token in enumerate(TOKENS[:5]):
            # Token name
            ttk.Label(price_grid, text=token, font=("Arial", 11, "bold"), 
                   style="Card.TLabel").grid(row=i*2, column=0, padx=5, pady=2, sticky=tk.W)
            
            # Current price
            self.price_labels[token] = ttk.Label(price_grid, text=f"${self.data.prices[token]:,.2f}", 
                                              style="Card.TLabel")
            self.price_labels[token].grid(row=i*2, column=1, padx=5, pady=2, sticky=tk.W)
            
            # 24h change
            change_pct = random.uniform(-5, 8)
            color = COLORS['green'] if change_pct >= 0 else COLORS['red']
            self.price_change_labels[token] = ttk.Label(price_grid, 
                                              text=f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%", 
                                              foreground=color, style="Card.TLabel")
            self.price_change_labels[token].grid(row=i*2, column=2, padx=5, pady=2, sticky=tk.W)
        
        # Second column of tokens
        for i, token in enumerate(TOKENS[5:]):
            # Token name
            ttk.Label(price_grid, text=token, font=("Arial", 11, "bold"), 
                   style="Card.TLabel").grid(row=i*2, column=3, padx=5, pady=2, sticky=tk.W)
            
            # Current price
            self.price_labels[token] = ttk.Label(price_grid, text=f"${self.data.prices[token]:,.2f}", 
                                              style="Card.TLabel")
            self.price_labels[token].grid(row=i*2, column=4, padx=5, pady=2, sticky=tk.W)
            
            # 24h change
            change_pct = random.uniform(-5, 8)
            color = COLORS['green'] if change_pct >= 0 else COLORS['red']
            self.price_change_labels[token] = ttk.Label(price_grid, 
                                              text=f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%", 
                                              foreground=color, style="Card.TLabel")
            self.price_change_labels[token].grid(row=i*2, column=5, padx=5, pady=2, sticky=tk.W)
    
    def create_trades_view(self):
        """Create the trades history view"""
        # Recent trades list
        trades_frame = ttk.Frame(self.trades_frame, style="Card.TFrame")
        trades_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        ttk.Label(trades_frame, text="RECENT TRADES", 
               style="Subheader.TLabel").pack(anchor=tk.W, padx=10, pady=10)
        
        # Create Treeview for trades
        columns = ("Time", "Strategy", "Chain", "Pair", "Amount", "Price", "Profit", "Status")
        self.trades_tree = ttk.Treeview(trades_frame, columns=columns, show="headings", height=20)
        
        # Configure columns
        for col in columns:
            self.trades_tree.heading(col, text=col)
            
        self.trades_tree.column("Time", width=150)
        self.trades_tree.column("Strategy", width=120)
        self.trades_tree.column("Chain", width=100)
        self.trades_tree.column("Pair", width=100)
        self.trades_tree.column("Amount", width=100)
        self.trades_tree.column("Price", width=100)
        self.trades_tree.column("Profit", width=100)
        self.trades_tree.column("Status", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(trades_frame, orient="vertical", command=self.trades_tree.yview)
        self.trades_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack the treeview and scrollbar
        self.trades_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=(0, 10))
    
    def create_portfolio_view(self):
        """Create the portfolio view"""
        # Holdings section
        holdings_frame = ttk.Frame(self.portfolio_frame, style="Card.TFrame")
        holdings_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        ttk.Label(holdings_frame, text="PORTFOLIO HOLDINGS", 
               style="Subheader.TLabel").pack(anchor=tk.W, padx=10, pady=10)
        
        # Create Treeview for holdings
        columns = ("Token", "Amount", "Value", "24h Change", "Portfolio %")
        self.holdings_tree = ttk.Treeview(holdings_frame, columns=columns, show="headings", height=10)
        
        # Configure columns
        for col in columns:
            self.holdings_tree.heading(col, text=col)
            
        self.holdings_tree.column("Token", width=100)
        self.holdings_tree.column("Amount", width=150)
        self.holdings_tree.column("Value", width=150)
        self.holdings_tree.column("24h Change", width=150)
        self.holdings_tree.column("Portfolio %", width=150)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(holdings_frame, orient="vertical", command=self.holdings_tree.yview)
        self.holdings_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack the treeview and scrollbar
        self.holdings_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=(0, 10))
        
        # Populate holdings data
        self.update_holdings()
    
    def create_footer(self):
        """Create the dashboard footer"""
        footer_frame = ttk.Frame(self.main_frame)
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Version info
        version_label = ttk.Label(footer_frame, text="Enhanced Quantum Trade AI v2.5.0")
        version_label.pack(side=tk.LEFT)
        
        # Status indicator
        self.status_label = ttk.Label(footer_frame, text="● READY", foreground=COLORS['green'])
        self.status_label.pack(side=tk.RIGHT)
    
    def update_holdings(self):
        """Update the portfolio holdings display"""
        # Clear existing items
        for item in self.holdings_tree.get_children():
            self.holdings_tree.delete(item)
        
        # Calculate total portfolio value
        total_value = sum(self.data.portfolio[token] * self.data.prices[token] for token in TOKENS)
        
        # Add holdings data
        for token in TOKENS:
            amount = self.data.portfolio[token]
            value = amount * self.data.prices[token]
            pct_change = random.uniform(-5, 8)
            portfolio_pct = (value / total_value) * 100 if total_value > 0 else 0
            
            # Format values
            amount_str = f"{amount:.4f}"
            value_str = f"${value:.2f}"
            change_str = f"{'+' if pct_change >= 0 else ''}{pct_change:.2f}%"
            portfolio_str = f"{portfolio_pct:.2f}%"
            
            # Insert into tree
            self.holdings_tree.insert("", tk.END, values=(token, amount_str, value_str, change_str, portfolio_str))
    
    def update_trades_display(self, trade):
        """Update the trades display with a new trade"""
        # Format trade data
        time_str = trade['timestamp']
        strategy_str = trade['strategy'].upper()
        chain_str = trade['chain'].capitalize()
        pair_str = f"{trade['base_token']}/{trade['quote_token']}"
        amount_str = f"{trade['amount']:.4f}"
        price_str = f"${trade['price']:.2f}"
        profit_str = f"${trade['profit']:.2f}"
        status_str = "SUCCESS" if trade['success'] else "FAILED"
        
        # Add to treeview
        self.trades_tree.insert("", 0, values=(time_str, strategy_str, chain_str, pair_str, 
                                             amount_str, price_str, profit_str, status_str))
        
        # Limit to 100 items
        if len(self.trades_tree.get_children()) > 100:
            self.trades_tree.delete(self.trades_tree.get_children()[-1])
    
    def update_market_condition(self):
        """Update the market condition indicator"""
        condition = self.data.detected_market_condition
        
        if condition == "bull":
            self.market_condition.config(text="BULL", foreground=COLORS['green'])
        elif condition == "bear":
            self.market_condition.config(text="BEAR", foreground=COLORS['red'])
        elif condition == "sideways":
            self.market_condition.config(text="SIDEWAYS", foreground=COLORS['blue'])
        elif condition == "high_volatility":
            self.market_condition.config(text="HIGH VOL", foreground=COLORS['yellow'])
        elif condition == "low_volatility":
            self.market_condition.config(text="LOW VOL", foreground=COLORS['purple'])
    
    def update_success_rate(self):
        """Update the success rate indicator"""
        rate = self.data.success_rate
        
        # Set color based on rate
        if rate >= 95:
            color = COLORS['green']
        elif rate >= 85:
            color = COLORS['yellow']
        else:
            color = COLORS['red']
            
        self.success_rate.config(text=f"{rate:.1f}%", foreground=color)
    
    def update_active_strategy(self):
        # Create strategy control panel
        if not hasattr(self, 'strategy_frame'):
            # Create main strategy frame with title
            strategy_container = ttk.Frame(self.root, style='Card.TFrame')
            strategy_container.pack(pady=20, padx=20, fill='x')
            
            # Add title
            title_frame = ttk.Frame(strategy_container, style='Card.TFrame')
            title_frame.pack(fill='x', padx=10, pady=(10,5))
            title_label = ttk.Label(title_frame, text="Strategy Control Panel",
                                  font=('Helvetica', 16, 'bold'),
                                  foreground=COLORS['accent'],
                                  background=COLORS['card'])
            title_label.pack(side=tk.LEFT)
            
            # Create strategy grid frame
            self.strategy_frame = ttk.Frame(strategy_container, style='Card.TFrame')
            self.strategy_frame.pack(pady=(0,10), padx=10, fill='x')
            
            # Configure grid columns
            self.strategy_frame.grid_columnconfigure(0, weight=1)
            self.strategy_frame.grid_columnconfigure(1, weight=1)
            self.strategy_frame.grid_columnconfigure(2, weight=1)
            
            # Create toggle buttons for each strategy
            self.strategy_toggles = {}
            for idx, strategy in enumerate(STRATEGIES):
                row = idx // 3
                col = idx % 3
                
                # Create strategy frame
                strategy_cell = ttk.Frame(self.strategy_frame, style='Card.TFrame')
                strategy_cell.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
                
                # Add strategy icon (you can customize these)
                icon_text = '⚡' if strategy == 'flashloan_arb' else\
                           '🔄' if strategy == 'cross_chain_arb' else\
                           '🥪' if strategy == 'sandwich' else\
                           '⚔️' if strategy == 'mev_extraction' else\
                           '⏱️' if strategy == 'just_in_time_liq' else '💰'
                
                icon_label = ttk.Label(strategy_cell, text=icon_text,
                                     font=('Helvetica', 14),
                                     background=COLORS['card'])
                icon_label.pack(side=tk.LEFT, padx=(5,10))
                
                # Strategy name
                name_label = ttk.Label(strategy_cell, text=strategy.replace('_', ' ').title(),
                                     style="Label.TLabel")
                name_label.pack(side=tk.LEFT, padx=5)
                
                # Toggle variable
                var = tk.BooleanVar(value=self.data.active_strategies[strategy])
                
                def make_toggle_callback(strat):
                    def toggle_callback():
                        self.data.active_strategies[strat] = not self.data.active_strategies[strat]
                        status = 'enabled' if self.data.active_strategies[strat] else 'disabled'
                        log_bot_activity(f"Strategy {strat} {status}")
                        self.update_strategy_status(strat)
                    return toggle_callback
                
                # Create modern toggle switch
                toggle_switch = ToggleSwitch(strategy_cell, width=60, height=30,
                                          bg_color=COLORS['background'],
                                          fg_color=COLORS['green'])
                toggle_switch.pack(side=tk.RIGHT, padx=10)
                toggle_switch.command = make_toggle_callback(strategy)
                
                self.strategy_toggles[strategy] = {
                    'switch': toggle_switch,
                    'label': name_label
                }
        
        # Update status indicators
        for strategy in STRATEGIES:
            self.update_strategy_status(strategy)
    
    def update_strategy_status(self, strategy):
        if strategy in self.strategy_toggles:
            # Update label color
            color = COLORS['green'] if self.data.active_strategies[strategy] else COLORS['text']
            self.strategy_toggles[strategy]['label'].configure(foreground=color)
            
            # Update toggle switch state
            self.strategy_toggles[strategy]['switch'].set(self.data.active_strategies[strategy])
        
        # Update active strategy label
        self.active_strategy.config(text=self.data.active_strategy.upper())
    
    def update_prices(self):
        """Update the price displays"""
        for token in TOKENS:
            # Update price
            self.price_labels[token].config(text=f"${self.data.prices[token]:,.2f}")
            
            # Update 24h change randomly (for demo)
            change_pct = random.uniform(-5, 8)
            color = COLORS['green'] if change_pct >= 0 else COLORS['red']
            self.price_change_labels[token].config(
                text=f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%",
                foreground=color
            )
    
    def update_metrics(self):
        """Update the key metrics"""
        # Calculate total portfolio value
        total_value = sum(self.data.portfolio[token] * self.data.prices[token] for token in TOKENS)
        
        # Update portfolio value
        self.portfolio_value_label.config(text=f"${total_value:,.2f}")
        
        # Update 24h change
        change_pct = random.uniform(1, 8)  # Mostly positive for demo
        color = COLORS['green'] if change_pct >= 0 else COLORS['red']
        self.change_24h_label.config(
            text=f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%",
            foreground=color
        )
        
        # Update total profit
        self.total_profit_label.config(text=f"${self.data.total_profit:,.2f}")
        
        # Update trades executed
        self.trades_executed_label.config(text=str(self.data.trades_executed))
    
    def background_updates(self):
        """Background thread for updating data"""
        last_trade_time = time.time()
        last_market_update = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Update prices every 5 seconds (very infrequently for stable display)
                self.data.update_prices()
                self.root.after(0, self.update_prices)
                
                # Update metrics
                self.root.after(0, self.update_metrics)
                
                # Update holdings occasionally
                if random.random() < 0.1:
                    self.root.after(0, self.update_holdings)
                
                # Generate a new trade every 10-20 seconds
                current_time = time.time()
                if current_time - last_trade_time > random.uniform(10, 20):
                    trade = self.data.generate_trade()
                    self.root.after(0, lambda t=trade: self.update_trades_display(t))
                    last_trade_time = current_time
                
                # Update market condition occasionally (very rarely)
                if current_time - last_market_update > random.uniform(30, 60):
                    conditions = ['bull', 'bear', 'sideways', 'high_volatility', 'low_volatility']
                    self.data.detected_market_condition = random.choice(conditions)
                    self.root.after(0, self.update_market_condition)
                    self.root.after(0, self.update_active_strategy)
                    self.root.after(0, self.update_success_rate)
                    last_market_update = current_time
                
                # Sleep between updates - long pause for stable display
                time.sleep(5)
            except Exception as e:
                print(f"Error in background thread: {e}")
                time.sleep(5)
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_event.set()
        self.root.destroy()

def main():
    """Main function to run the dashboard"""
    root = tk.Tk()
    app = TradingDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()
