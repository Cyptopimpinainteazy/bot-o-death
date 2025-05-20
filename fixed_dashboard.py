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
from data_manager import DataManager

# Constants
TOKENS = ['ETH', 'USDC', 'WBTC', 'AAVE', 'LINK', 'UNI', 'MATIC', 'BNB', 'SOL', 'AVAX']
CHAINS = ['ethereum', 'polygon', 'bsc', 'arbitrum_one']
STRATEGIES = ['sandwich', 'arbitrage', 'liquidation', 'flash_loan']
MARKET_CONDITIONS = ['bull', 'bear', 'sideways', 'volatile']

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
        
        # Trading control
        self.trading_enabled = tk.BooleanVar(value=False)
        self.trade_log_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'trade_logs',
            f'trades_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(self.trade_log_file), exist_ok=True)
        
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
        style = ttk.Style()
        style.theme_use('default')
        
        # Configure common styles
        style.configure('Main.TFrame', background=COLORS['background'])
        style.configure('TLabel', background=COLORS['background'], foreground=COLORS['text'])
        style.configure('TButton', background=COLORS['surface'], foreground=COLORS['text'])
        style.configure('Accent.TButton', background=COLORS['primary'], foreground=COLORS['background'])
        
        # Configure switch style
        style.configure('Switch.TCheckbutton',
            background=COLORS['background'],
            foreground=COLORS['text'],
            indicatorcolor=COLORS['primary'])
        style.map('Switch.TCheckbutton',
            background=[('active', COLORS['background'])],
            foreground=[('active', COLORS['primary'])])
        
    def create_header(self):
        header_frame = ttk.Frame(self.main_frame, style='Main.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_label = ttk.Label(
            header_frame,
            text="Enhanced Quantum Trade AI",
            font=('Helvetica', 24, 'bold'),
            foreground=COLORS['primary']
        )
        title_label.pack(side=tk.LEFT)
        
        # Control buttons frame
        controls_frame = ttk.Frame(header_frame, style='Main.TFrame')
        controls_frame.pack(side=tk.RIGHT)
        
        # Trading switch
        self.trading_enabled = tk.BooleanVar(value=False)
        self.trading_switch = ttk.Checkbutton(
            controls_frame,
            text="Trading",
            variable=self.trading_enabled,
            command=self.toggle_trading,
            style='Switch.TCheckbutton'
        )
        self.trading_switch.pack(side=tk.RIGHT, padx=10)
        
        # Wallet connection button
        self.wallet_button = ttk.Button(
            controls_frame,
            text="Connect Wallet",
            style='Accent.TButton',
            command=self.toggle_wallet_connection
        )
        self.wallet_button.pack(side=tk.RIGHT, padx=5)

    def create_dashboard(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.overview_tab = ttk.Frame(self.notebook, style='Main.TFrame')
        self.portfolio_tab = ttk.Frame(self.notebook, style='Main.TFrame')
        self.trades_tab = ttk.Frame(self.notebook, style='Main.TFrame')
        self.strategy_tab = ttk.Frame(self.notebook, style='Main.TFrame')
        
        self.notebook.add(self.overview_tab, text='Overview')
        self.notebook.add(self.portfolio_tab, text='Portfolio')
        self.notebook.add(self.trades_tab, text='Trades')
        self.notebook.add(self.strategy_tab, text='Strategy')
        
        # Create content for each tab
        self.create_overview_tab()
        self.create_portfolio_tab()
        self.create_trades_tab()
        self.create_strategy_tab()

    def create_overview_tab(self):
        # Create metrics section
        self.create_metrics_section(self.overview_tab)
        
        # Create charts section
        charts_frame = ttk.Frame(self.overview_tab, style='Main.TFrame')
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Price chart frame
        chart_frame = ttk.Frame(charts_frame, style='Main.TFrame')
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Token selection
        self.selected_token = tk.StringVar(value='ETH')
        token_menu = ttk.OptionMenu(
            chart_frame,
            self.selected_token,
            'ETH',
            *TOKENS,
            command=self.update_price_chart
        )
        token_menu.pack(anchor='w', padx=5, pady=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Recent trades
        self.create_simple_trades_list(charts_frame)

    def create_metrics_section(self, parent):
        metrics_frame = ttk.Frame(parent, style='Main.TFrame')
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Configure grid
        metrics_frame.grid_columnconfigure((0,1,2,3), weight=1)
        
        # Create metric cards
        self.profit_card = self.create_metric_card(
            metrics_frame, "Total Profit", "$0.00", 0, 0, COLORS['green']
        )
        self.trades_card = self.create_metric_card(
            metrics_frame, "Trades Today", "0", 0, 1
        )
        self.success_card = self.create_metric_card(
            metrics_frame, "Success Rate", "0%", 0, 2
        )
        self.strategy_card = self.create_metric_card(
            metrics_frame, "Active Strategy", "None", 0, 3, COLORS['primary']
        )

    def create_metric_card(self, parent, title, value, row, col, color=COLORS['text']):
        frame = ttk.Frame(parent, style='Main.TFrame')
        frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
        
        title_label = ttk.Label(
            frame,
            text=title,
            font=('Helvetica', 12),
            foreground=COLORS['subtext']
        )
        title_label.pack(anchor='w')
        
        value_label = ttk.Label(
            frame,
            text=value,
            font=('Helvetica', 18, 'bold'),
            foreground=color
        )
        value_label.pack(anchor='w')
        
        return value_label

    def toggle_wallet_connection(self):
        if not self.data_manager.wallet_connected:
            self.data_manager.connect_wallet()
            self.wallet_button.configure(text="Disconnect Wallet")
        else:
            self.data_manager.disconnect_wallet()
            self.wallet_button.configure(text="Connect Wallet")

    def create_footer(self):
        footer_frame = ttk.Frame(self.main_frame, style='Main.TFrame')
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        version_label = ttk.Label(
            footer_frame,
            text="v2.0.0",
            foreground=COLORS['subtext']
        )
        version_label.pack(side=tk.RIGHT)

    def background_updates(self):
        while not self.stop_event.is_set():
            try:
                # Update data
                self.data_manager.update_prices()
                
                # Update UI
                self.update_metrics()
                self.update_price_chart()
                self.update_trades()
                
                # Sleep for a bit
                time.sleep(1)
            except Exception as e:
                print(f"Error in background updates: {e}")
                time.sleep(5)
                
    def update_price_chart(self, event=None):
        token = self.selected_token.get()
        prices = list(self.data_manager.price_history[token])
        
        # Clear the axis
        self.ax.clear()
        
        # Plot the data
        self.ax.plot(range(len(prices)), prices, color=COLORS['primary'], linewidth=2)
        
        # Customize the chart
        self.ax.set_facecolor(COLORS['background'])
        self.ax.grid(True, color=COLORS['surface'])
        self.ax.set_title(f'{token} Price', color=COLORS['text'])
        self.ax.set_xlabel('Time', color=COLORS['subtext'])
        self.ax.set_ylabel('Price ($)', color=COLORS['subtext'])
        
        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()
    
    def create_simple_trades_list(self, parent):
        trades_frame = ttk.Frame(parent, style='Main.TFrame')
        trades_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Title
        title = ttk.Label(
            trades_frame,
            text="Recent Trades",
            font=('Helvetica', 14, 'bold'),
            foreground=COLORS['text']
        )
        title.pack(anchor='w', pady=(0, 10))
        
        # Create treeview
        self.trades_tree = ttk.Treeview(
            trades_frame,
            columns=('Time', 'Pair', 'Type', 'Profit'),
            show='headings',
            height=10
        )
        
        # Configure columns
        self.trades_tree.heading('Time', text='Time')
        self.trades_tree.heading('Pair', text='Pair')
        self.trades_tree.heading('Type', text='Type')
        self.trades_tree.heading('Profit', text='Profit')
        
        self.trades_tree.column('Time', width=100)
        self.trades_tree.column('Pair', width=100)
        self.trades_tree.column('Type', width=100)
        self.trades_tree.column('Profit', width=100)
        
        self.trades_tree.pack(fill=tk.BOTH, expand=True)
        
    def update_trades(self):
        # Only process trades if trading is enabled
        if not self.trading_enabled.get():
            return
            
        # Get latest trade
        if self.data_manager.trade_history:
            latest_trade = self.data_manager.trade_history[-1]
            
            # Format trade data
            time_str = latest_trade['timestamp'].strftime('%H:%M:%S')
            pair = f"{latest_trade['base_token']}/{latest_trade['quote_token']}"
            trade_type = latest_trade['type'].title()
            profit = f"${latest_trade['profit']:.2f}"
            
            # Add to treeview
            self.trades_tree.insert('', 0, values=(time_str, pair, trade_type, profit))
            
            # Keep only last 10 trades
            if len(self.trades_tree.get_children()) > 10:
                self.trades_tree.delete(self.trades_tree.get_children()[-1])
            
            # Log trade to file
            self.log_trade(latest_trade)
    
    def log_trade(self, trade):
        try:
            with open(self.trade_log_file, 'a') as f:
                timestamp = trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                pair = f"{trade['base_token']}/{trade['quote_token']}"
                trade_type = trade['type']
                profit = trade['profit']
                log_entry = f"[{timestamp}] {trade_type.upper()} {pair} - Profit: ${profit:.2f}\n"
                f.write(log_entry)
        except Exception as e:
            print(f"Error logging trade: {e}")
    
    def toggle_trading(self):
        is_enabled = self.trading_enabled.get()
        if is_enabled:
            # Start trading
            print("Trading enabled - Logging trades to:", self.trade_log_file)
            messagebox.showinfo(
                "Trading Enabled",
                f"Trading is now active\nLogs will be saved to: {os.path.basename(self.trade_log_file)}"
            )
        else:
            # Stop trading
            print("Trading disabled")
            messagebox.showinfo(
                "Trading Disabled",
                "Trading has been stopped"
            )

    def update_metrics(self):
        self.profit_card.configure(
            text=f"${self.data_manager.total_profit:,.2f}"
        )
        self.trades_card.configure(
            text=str(self.data_manager.trades_executed)
        )
        self.success_card.configure(
            text=f"{self.data_manager.success_rate}%"
        )
        self.strategy_card.configure(
            text=self.data_manager.active_strategy.title()
        )

    def create_portfolio_tab(self):
        # Create main sections
        left_frame = ttk.Frame(self.portfolio_tab, style='Main.TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(self.portfolio_tab, style='Main.TFrame')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Portfolio allocation chart
        self.create_portfolio_pie_chart(left_frame)
        
        # Holdings list
        self.create_holdings_grid(right_frame)
    
    def create_portfolio_pie_chart(self, parent):
        chart_frame = ttk.Frame(parent, style='Main.TFrame')
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create figure
        self.portfolio_fig = Figure(figsize=(6, 4), dpi=100)
        self.portfolio_ax = self.portfolio_fig.add_subplot(111)
        self.portfolio_canvas = FigureCanvasTkAgg(self.portfolio_fig, master=chart_frame)
        self.portfolio_canvas.draw()
        self.portfolio_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial plot
        self.update_portfolio_pie_chart()
    
    def update_portfolio_pie_chart(self):
        # Get portfolio data
        portfolio = self.data_manager.portfolio
        labels = []
        sizes = []
        colors = []
        
        for token, amount in portfolio.items():
            if amount > 0:
                labels.append(token)
                sizes.append(amount)
                colors.append(COLORS['primary'] if token == 'ETH' else COLORS['surface'])
        
        # Clear previous plot
        self.portfolio_ax.clear()
        
        # Create pie chart
        self.portfolio_ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        
        self.portfolio_ax.set_title('Portfolio Allocation', color=COLORS['text'])
        self.portfolio_fig.tight_layout()
        self.portfolio_canvas.draw()
    
    def create_holdings_grid(self, parent):
        # Create treeview for holdings
        self.holdings_tree = ttk.Treeview(
            parent,
            columns=('Token', 'Amount', 'Value', 'Change'),
            show='headings',
            height=10
        )
        
        # Configure columns
        self.holdings_tree.heading('Token', text='Token')
        self.holdings_tree.heading('Amount', text='Amount')
        self.holdings_tree.heading('Value', text='Value ($)')
        self.holdings_tree.heading('Change', text='24h Change')
        
        self.holdings_tree.column('Token', width=100)
        self.holdings_tree.column('Amount', width=100)
        self.holdings_tree.column('Value', width=100)
        self.holdings_tree.column('Change', width=100)
        
        self.holdings_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initial update
        self.update_holdings_grid()
    
    def update_holdings_grid(self):
        # Clear current items
        for item in self.holdings_tree.get_children():
            self.holdings_tree.delete(item)
        
        # Add updated holdings
        for token, amount in self.data_manager.portfolio.items():
            price = self.data_manager.prices[token]
            value = amount * price
            change = random.uniform(-5, 15)  # Simulated 24h change
            
            # Format values
            amount_str = f"{amount:.4f}"
            value_str = f"${value:,.2f}"
            change_str = f"{change:+.2f}%"
            
            self.holdings_tree.insert('', 'end', values=(
                token, amount_str, value_str, change_str
            ))
    
    def create_trades_tab(self):
        # Create treeview for detailed trade history
        self.detailed_trades_tree = ttk.Treeview(
            self.trades_tab,
            columns=('Time', 'Type', 'Pair', 'Amount', 'Price', 'Value', 'Profit'),
            show='headings',
            height=20
        )
        
        # Configure columns
        columns = [
            ('Time', 150),
            ('Type', 100),
            ('Pair', 100),
            ('Amount', 100),
            ('Price', 100),
            ('Value', 100),
            ('Profit', 100)
        ]
        
        for col, width in columns:
            self.detailed_trades_tree.heading(col, text=col)
            self.detailed_trades_tree.column(col, width=width)
        
        self.detailed_trades_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_strategy_tab(self):
        # Create main sections
        controls_frame = ttk.Frame(self.strategy_tab, style='Main.TFrame')
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Strategy selector
        strategy_frame = ttk.LabelFrame(controls_frame, text="Strategy", style='Main.TFrame')
        strategy_frame.pack(side=tk.LEFT, padx=10)
        
        self.strategy_var = tk.StringVar(value='sandwich')
        for strategy in STRATEGIES:
            ttk.Radiobutton(
                strategy_frame,
                text=strategy.replace('_', ' ').title(),
                value=strategy,
                variable=self.strategy_var,
                command=self.update_strategy
            ).pack(anchor='w', padx=5, pady=2)
        
        # Market condition selector
        condition_frame = ttk.LabelFrame(controls_frame, text="Market Condition", style='Main.TFrame')
        condition_frame.pack(side=tk.LEFT, padx=10)
        
        self.condition_var = tk.StringVar(value='bull')
        for condition in MARKET_CONDITIONS:
            ttk.Radiobutton(
                condition_frame,
                text=condition.replace('_', ' ').title(),
                value=condition,
                variable=self.condition_var,
                command=self.update_market_condition
            ).pack(anchor='w', padx=5, pady=2)
    
    def update_strategy(self):
        self.data_manager.active_strategy = self.strategy_var.get()
        self.update_metrics()
    
    def update_market_condition(self):
        self.data_manager.detected_market_condition = self.condition_var.get()
        self.update_metrics()
    
    def on_closing(self):
        self.stop_event.set()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = QuantumTradingDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
