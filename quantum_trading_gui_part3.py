    def create_trades_view(self):
        """Create trades history view"""
        # Container frame
        container = ttk.Frame(self.trades_frame)
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Controls frame
        controls = ttk.Frame(container)
        controls.pack(fill=tk.X, pady=(0, 10))
        
        # Filter label
        filter_label = ttk.Label(controls, text="Filter by:")
        filter_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Chain filter
        chain_label = ttk.Label(controls, text="Chain:")
        chain_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.chain_var = tk.StringVar(value="All")
        chain_combo = ttk.Combobox(controls, textvariable=self.chain_var, values=["All"] + CHAINS, width=10, state="readonly")
        chain_combo.pack(side=tk.LEFT, padx=(0, 15))
        
        # Strategy filter
        strategy_label = ttk.Label(controls, text="Strategy:")
        strategy_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.strategy_var = tk.StringVar(value="All")
        strategy_combo = ttk.Combobox(controls, textvariable=self.strategy_var, values=["All"] + STRATEGIES, width=15, state="readonly")
        strategy_combo.pack(side=tk.LEFT, padx=(0, 15))
        
        # Success filter
        success_label = ttk.Label(controls, text="Success:")
        success_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.success_var = tk.StringVar(value="All")
        success_combo = ttk.Combobox(controls, textvariable=self.success_var, values=["All", "Successful", "Failed"], width=10, state="readonly")
        success_combo.pack(side=tk.LEFT)
        
        # Trades treeview
        tree_frame = ttk.Frame(container)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Trades treeview
        self.full_trades_tree = ttk.Treeview(tree_frame, 
                                            columns=("Time", "Chain", "Strategy", "Pair", "Amount", "Price", "Success", "Profit", "Gas"),
                                            show="headings",
                                            yscrollcommand=scrollbar.set)
        
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
        
        self.full_trades_tree.column("Time", width=140)
        self.full_trades_tree.column("Chain", width=80)
        self.full_trades_tree.column("Strategy", width=100)
        self.full_trades_tree.column("Pair", width=80)
        self.full_trades_tree.column("Amount", width=80)
        self.full_trades_tree.column("Price", width=80)
        self.full_trades_tree.column("Success", width=80)
        self.full_trades_tree.column("Profit", width=80)
        self.full_trades_tree.column("Gas", width=80)
        
        self.full_trades_tree.pack(fill=tk.BOTH, expand=True)
        
    def create_portfolio_view(self):
        """Create portfolio view"""
        # Top section - Current holdings
        holdings_label = ttk.Label(self.portfolio_frame, text="CURRENT HOLDINGS", font=("Arial", 12, "bold"))
        holdings_label.pack(anchor=tk.W, pady=(5, 10))
        
        # Holdings grid
        holdings_frame = ttk.Frame(self.portfolio_frame)
        holdings_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Create holdings display
        self.holding_frames = {}
        self.holding_labels = {}
        self.holding_values = {}
        
        row = 0
        col = 0
        for token in TOKENS:
            self.create_holding_display(holdings_frame, token, row, col)
            col += 1
            if col > 4:  # 5 columns per row
                col = 0
                row += 1
        
        # Bottom section - Allocation controls
        allocation_label = ttk.Label(self.portfolio_frame, text="PORTFOLIO ALLOCATION", font=("Arial", 12, "bold"))
        allocation_label.pack(anchor=tk.W, pady=(5, 10))
        
        allocation_frame = ttk.Frame(self.portfolio_frame)
        allocation_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - controls
        controls_frame = ttk.Frame(allocation_frame)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Market condition
        market_frame = ttk.Frame(controls_frame)
        market_frame.pack(fill=tk.X, pady=(0, 15))
        
        market_label = ttk.Label(market_frame, text="Market Condition:", font=("Arial", 10, "bold"))
        market_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.market_var = tk.StringVar(value="bull")
        for i, condition in enumerate(MARKET_CONDITIONS):
            rb = ttk.Radiobutton(market_frame, text=condition.title(), variable=self.market_var, value=condition)
            rb.pack(anchor=tk.W, pady=2)
        
        # Apply button
        apply_button = ttk.Button(controls_frame, text="Apply Optimal Allocation", command=self.apply_allocation)
        apply_button.pack(fill=tk.X, pady=(10, 0))
        
        # Right side - allocation table
        table_frame = ttk.Frame(allocation_frame)
        table_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Allocation treeview
        self.allocation_tree = ttk.Treeview(table_frame, 
                                           columns=("Token", "Current", "Suggested", "Action"),
                                           show="headings",
                                           yscrollcommand=scrollbar.set)
        
        # Configure scrollbar
        scrollbar.config(command=self.allocation_tree.yview)
        
        # Configure columns
        self.allocation_tree.heading("Token", text="Token")
        self.allocation_tree.heading("Current", text="Current %")
        self.allocation_tree.heading("Suggested", text="Suggested %")
        self.allocation_tree.heading("Action", text="Action")
        
        self.allocation_tree.column("Token", width=80)
        self.allocation_tree.column("Current", width=100)
        self.allocation_tree.column("Suggested", width=100)
        self.allocation_tree.column("Action", width=200)
        
        self.allocation_tree.pack(fill=tk.BOTH, expand=True)
        
        # Populate with initial data
        self.update_allocation_table()
        
    def create_holding_display(self, parent, token, row, col):
        """Create a holding display for a token"""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, padx=5, pady=5, sticky=tk.NSEW)
        
        # Add border and padding using a canvas
        canvas = tk.Canvas(frame, bg="#313244", bd=0, highlightthickness=1, highlightbackground="#6c7086", height=80)
        canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Token name
        token_label = tk.Label(canvas, text=token, font=("Arial", 10, "bold"), fg="#cdd6f4", bg="#313244")
        token_label.pack(anchor=tk.W, padx=10, pady=(10, 2))
        
        # Amount
        amount = self.data_generator.portfolio.get(token, 0)
        value_label = tk.Label(canvas, 
                              text=f"{amount:.4f}", 
                              font=("Arial", 12), 
                              fg="#f5c2e7", 
                              bg="#313244")
        value_label.pack(anchor=tk.W, padx=10)
        
        # USD Value
        price = self.data_generator.prices.get(token, 0)
        usd_value = amount * price
        usd_label = tk.Label(canvas, 
                            text=f"${usd_value:.2f}", 
                            font=("Arial", 9), 
                            fg="#a6e3a1", 
                            bg="#313244")
        usd_label.pack(anchor=tk.W, padx=10, pady=(0, 5))
        
        # Store references for updates
        self.holding_labels[token] = value_label
        self.holding_values[token] = usd_label
        self.holding_frames[token] = frame
        
    def create_strategy_view(self):
        """Create strategy optimization view"""
        # Top section - Strategy performance
        performance_label = ttk.Label(self.strategy_frame, text="STRATEGY PERFORMANCE", font=("Arial", 12, "bold"))
        performance_label.pack(anchor=tk.W, pady=(5, 10))
        
        # Performance table frame
        table_frame = ttk.Frame(self.strategy_frame)
        table_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Performance treeview
        self.performance_tree = ttk.Treeview(table_frame, 
                                            columns=("Strategy", "Market", "Success", "Profit", "Rating"),
                                            show="headings",
                                            yscrollcommand=scrollbar.set,
                                            height=10)
        
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
        self.performance_tree.column("Rating", width=100)
        
        self.performance_tree.pack(fill=tk.X)
        
        # Bottom section - Active strategy
        active_label = ttk.Label(self.strategy_frame, text="ACTIVE STRATEGY SETTINGS", font=("Arial", 12, "bold"))
        active_label.pack(anchor=tk.W, pady=(5, 10))
        
        active_frame = ttk.Frame(self.strategy_frame)
        active_frame.pack(fill=tk.BOTH, expand=True)
        
        # Strategy selection
        selection_frame = ttk.Frame(active_frame)
        selection_frame.pack(fill=tk.X, pady=(0, 20))
        
        strategy_label = ttk.Label(selection_frame, text="Select Strategy:")
        strategy_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.select_strategy_var = tk.StringVar(value=self.data_generator.active_strategy)
        strategy_combo = ttk.Combobox(selection_frame, textvariable=self.select_strategy_var, values=STRATEGIES, width=20, state="readonly")
        strategy_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        activate_button = ttk.Button(selection_frame, text="Activate Strategy", command=self.activate_strategy)
        activate_button.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Auto optimization
        auto_frame = ttk.Frame(active_frame)
        auto_frame.pack(fill=tk.X)
        
        self.auto_var = tk.BooleanVar(value=True)
        auto_check = ttk.Checkbutton(auto_frame, text="Auto-select optimal strategy based on market conditions", variable=self.auto_var)
        auto_check.pack(anchor=tk.W, pady=5)
        
        # Parameters frame
        params_label = ttk.Label(active_frame, text="Strategy Parameters", font=("Arial", 10, "bold"))
        params_label.pack(anchor=tk.W, pady=(20, 10))
        
        # Advanced strategy parameters
        params_frame = ttk.Frame(active_frame)
        params_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create parameter fields
        param_names = ["Max Gas Price (Gwei)", "Min Profit Threshold ($)", "Max Slippage (%)", "Trade Size Multiplier", "Quantum Optimization Level"]
        param_values = [30, 0.5, 0.5, 1.0, 10]
        
        self.param_vars = []
        
        for i, (name, value) in enumerate(zip(param_names, param_values)):
            label = ttk.Label(params_frame, text=name + ":")
            label.grid(row=i, column=0, padx=5, pady=5, sticky=tk.W)
            
            var = tk.DoubleVar(value=value)
            self.param_vars.append(var)
            
            entry = ttk.Entry(params_frame, textvariable=var, width=10)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky=tk.W)
            
        # Apply button
        apply_params_button = ttk.Button(params_frame, text="Apply Parameters", command=self.apply_parameters)
        apply_params_button.grid(row=len(param_names), column=0, columnspan=2, padx=5, pady=15, sticky=tk.EW)
        
    def create_footer(self):
        """Create footer with status and version info"""
        footer = ttk.Frame(self.main_frame)
        footer.pack(fill=tk.X, pady=(10, 0))
        
        # Status indicator
        self.status_label = ttk.Label(footer, text="System Status: ONLINE", foreground="#a6e3a1")
        self.status_label.pack(side=tk.LEFT)
        
        # Version info
        version_label = ttk.Label(footer, text="Enhanced Quantum Trade AI v1.0.0")
        version_label.pack(side=tk.RIGHT)
        
    def apply_allocation(self):
        """Apply the selected allocation"""
        market = self.market_var.get()
        self.data_generator.detected_market_condition = market
        
        # Update optimal strategy based on market
        if market == 'bull':
            self.data_generator.active_strategy = 'sandwich'
        elif market == 'bear':
            self.data_generator.active_strategy = 'flashloan_arb'
        elif market == 'sideways':
            self.data_generator.active_strategy = 'mev_extraction'
        elif market == 'high_volatility':
            self.data_generator.active_strategy = 'just_in_time_liq'
        elif market == 'low_volatility':
            self.data_generator.active_strategy = 'mev_extraction'
        
        # Update the UI
        self.update_market_condition()
        self.update_active_strategy()
        self.update_allocation_table()
        
        messagebox.showinfo("Portfolio Allocation", f"Portfolio allocation optimized for {market.upper()} market")
        
    def update_allocation_table(self):
        """Update the allocation table"""
        # Clear existing items
        for item in self.allocation_tree.get_children():
            self.allocation_tree.delete(item)
        
        # Get current allocation percentages
        total_value = sum(self.data_generator.portfolio[token] * self.data_generator.prices[token] for token in TOKENS)
        current_allocation = {token: (self.data_generator.portfolio[token] * self.data_generator.prices[token] / total_value) * 100 
                             for token in TOKENS}
        
        # Generate suggested allocation based on market condition
        market = self.data_generator.detected_market_condition
        suggested_allocation = {}
        
        if market == 'bull':
            # In bull market, overweight higher volatility assets
            suggested_allocation = {
                'ETH': 25, 'WBTC': 20, 'SOL': 15, 'AVAX': 15,
                'BNB': 10, 'MATIC': 5, 'LINK': 5, 'UNI': 3,
                'AAVE': 1, 'USDC': 1
            }
        elif market == 'bear':
            # In bear market, overweight stablecoins and blue chips
            suggested_allocation = {
                'USDC': 40, 'ETH': 20, 'WBTC': 15, 'BNB': 10,
                'SOL': 5, 'AVAX': 5, 'LINK': 2, 'MATIC': 1,
                'UNI': 1, 'AAVE': 1
            }
        elif market == 'sideways':
            # In sideways market, balanced approach
            suggested_allocation = {
                'ETH': 20, 'USDC': 20, 'WBTC': 15, 'BNB': 10,
                'SOL': 10, 'AVAX': 10, 'LINK': 5, 'MATIC': 5,
                'UNI': 3, 'AAVE': 2
            }
        elif market == 'high_volatility':
            # In high volatility, focus on pairs with arbitrage potential
            suggested_allocation = {
                'ETH': 20, 'WBTC': 20, 'SOL': 15, 'AVAX': 15,
                'BNB': 10, 'USDC': 10, 'MATIC': 5, 'LINK': 3,
                'UNI': 1, 'AAVE': 1
            }
        elif market == 'low_volatility':
            # In low volatility, focus on yield
            suggested_allocation = {
                'USDC': 30, 'ETH': 15, 'WBTC': 15, 'BNB': 10,
                'SOL': 10, 'AVAX': 10, 'LINK': 5, 'MATIC': 2,
                'UNI': 2, 'AAVE': 1
            }
            
        # Add items to the table
        for token in TOKENS:
            current = current_allocation[token]
            suggested = suggested_allocation.get(token, 0)
            diff = suggested - current
            
            if abs(diff) < 1:
                action = "No change needed"
                tag = "normal"
            elif diff > 0:
                action = f"Buy {diff:.1f}% more"
                tag = "buy"
            else:
                action = f"Sell {abs(diff):.1f}%"
                tag = "sell"
                
            self.allocation_tree.insert("", "end", values=(token, f"{current:.1f}%", f"{suggested:.1f}%", action), tags=(tag,))
            
        # Configure tags
        self.allocation_tree.tag_configure("buy", foreground="#a6e3a1")
        self.allocation_tree.tag_configure("sell", foreground="#f38ba8")
        self.allocation_tree.tag_configure("normal", foreground="#cdd6f4")
