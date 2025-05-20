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
