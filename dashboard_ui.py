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
