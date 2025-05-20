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
