class QuantumTradingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Quantum Trade AI Dashboard")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e2e")  # Dark theme background
        
        # Set theme style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure dark theme colors
        self.style.configure("TFrame", background="#1e1e2e")
        self.style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4")
        self.style.configure("TButton", background="#45475a", foreground="#cdd6f4", borderwidth=1)
        self.style.map("TButton", background=[("active", "#7f849c")])
        self.style.configure("TNotebook", background="#1e1e2e", borderwidth=0)
        self.style.configure("TNotebook.Tab", background="#313244", foreground="#cdd6f4", padding=[10, 2])
        self.style.map("TNotebook.Tab", background=[("selected", "#45475a")], foreground=[("selected", "#cdd6f4")])
        self.style.configure("Treeview", background="#313244", foreground="#cdd6f4", fieldbackground="#313244")
        self.style.map("Treeview", background=[("selected", "#7f849c")])
        
        # Data generator
        self.data_generator = MockDataGenerator()
        
        # Create main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create header frame
        self.create_header()
        
        # Create main notebook with tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create dashboard tab
        self.dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.dashboard_frame, text="Dashboard")
        self.create_dashboard()
        
        # Create trades tab
        self.trades_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.trades_frame, text="Trades")
        self.create_trades_view()
        
        # Create portfolio tab
        self.portfolio_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.portfolio_frame, text="Portfolio")
        self.create_portfolio_view()
        
        # Create strategy tab
        self.strategy_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.strategy_frame, text="Strategy Optimizer")
        self.create_strategy_view()
        
        # Create footer
        self.create_footer()
        
        # Start background tasks
        self.stop_event = threading.Event()
        self.update_thread = threading.Thread(target=self.background_updates)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def create_header(self):
        """Create header with logo and status indicators"""
        header = ttk.Frame(self.main_frame)
        header.pack(fill=tk.X, pady=(0, 10))
        
        # Left side - Logo and title
        logo_frame = ttk.Frame(header)
        logo_frame.pack(side=tk.LEFT)
        
        logo_text = tk.Label(logo_frame, text="Ξ", font=("Arial", 24, "bold"), fg="#f5c2e7", bg="#1e1e2e")
        logo_text.pack(side=tk.LEFT, padx=(0, 5))
        
        title = tk.Label(logo_frame, text="ENHANCED QUANTUM TRADE AI", font=("Arial", 16, "bold"), fg="#cdd6f4", bg="#1e1e2e")
        title.pack(side=tk.LEFT)
        
        # Right side - Status indicators
        status_frame = ttk.Frame(header)
        status_frame.pack(side=tk.RIGHT)
        
        # Market condition indicator
        self.market_label = tk.Label(status_frame, text="MARKET:", fg="#cdd6f4", bg="#1e1e2e", font=("Arial", 10))
        self.market_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.market_condition = tk.Label(status_frame, text="BULL", fg="#a6e3a1", bg="#1e1e2e", font=("Arial", 10, "bold"))
        self.market_condition.pack(side=tk.LEFT, padx=(0, 15))
        
        # Strategy indicator
        self.strategy_label = tk.Label(status_frame, text="STRATEGY:", fg="#cdd6f4", bg="#1e1e2e", font=("Arial", 10))
        self.strategy_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.active_strategy = tk.Label(status_frame, text="SANDWICH", fg="#f5c2e7", bg="#1e1e2e", font=("Arial", 10, "bold"))
        self.active_strategy.pack(side=tk.LEFT, padx=(0, 15))
        
        # Success rate indicator
        self.success_label = tk.Label(status_frame, text="SUCCESS RATE:", fg="#cdd6f4", bg="#1e1e2e", font=("Arial", 10))
        self.success_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.success_rate = tk.Label(status_frame, text="93%", fg="#a6e3a1", bg="#1e1e2e", font=("Arial", 10, "bold"))
        self.success_rate.pack(side=tk.LEFT)
        
    def create_dashboard(self):
        """Create main dashboard view"""
        # Top section - Key performance metrics
        metrics_frame = ttk.Frame(self.dashboard_frame)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create metric boxes
        self.create_metric_box(metrics_frame, "Total Profit", "$0.00", "#a6e3a1", 0)
        self.create_metric_box(metrics_frame, "Trades Executed", "0", "#f5c2e7", 1)
        self.create_metric_box(metrics_frame, "Success Rate", "0%", "#89dceb", 2)
        self.create_metric_box(metrics_frame, "Active Chains", "6", "#f9e2af", 3)
        
        # Middle section - Price charts in scrollable frame
        prices_label = ttk.Label(self.dashboard_frame, text="LIVE TOKEN PRICES", font=("Arial", 12, "bold"))
        prices_label.pack(anchor=tk.W, pady=(10, 5))
        
        prices_container = ttk.Frame(self.dashboard_frame)
        prices_container.pack(fill=tk.X, pady=(0, 10))
        
        # Create price displays for each token
        self.price_frames = {}
        self.price_labels = {}
        self.price_changes = {}
        
        # Create 2 rows of 5 tokens
        row1 = ttk.Frame(prices_container)
        row1.pack(fill=tk.X)
        row2 = ttk.Frame(prices_container)
        row2.pack(fill=tk.X, pady=(10, 0))
        
        for i, token in enumerate(TOKENS[:5]):
            self.create_price_display(row1, token, i)
        
        for i, token in enumerate(TOKENS[5:]):
            self.create_price_display(row2, token, i)
        
        # Bottom section - Recent trades
        trades_label = ttk.Label(self.dashboard_frame, text="RECENT TRADES", font=("Arial", 12, "bold"))
        trades_label.pack(anchor=tk.W, pady=(10, 5))
        
        # Create trades list
        trades_container = ttk.Frame(self.dashboard_frame)
        trades_container.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(trades_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Trades treeview
        self.trades_tree = ttk.Treeview(trades_container, 
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
        self.trades_tree.column("Chain", width=100)
        self.trades_tree.column("Strategy", width=120)
        self.trades_tree.column("Pair", width=100)
        self.trades_tree.column("Amount", width=80)
        self.trades_tree.column("Success", width=80)
        self.trades_tree.column("Profit", width=80)
        
        self.trades_tree.pack(fill=tk.BOTH, expand=True)
        
    def create_metric_box(self, parent, title, value, color, col):
        """Create a metric display box"""
        frame = ttk.Frame(parent, style="TFrame")
        frame.grid(row=0, column=col, padx=5, sticky=tk.EW)
        parent.columnconfigure(col, weight=1)
        
        # Add border and padding using a canvas
        canvas = tk.Canvas(frame, bg="#313244", bd=0, highlightthickness=1, highlightbackground="#6c7086", height=100)
        canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Title
        title_label = tk.Label(canvas, text=title, font=("Arial", 10), fg="#cdd6f4", bg="#313244")
        title_label.pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        # Value
        value_label = tk.Label(canvas, text=value, font=("Arial", 18, "bold"), fg=color, bg="#313244")
        value_label.pack(anchor=tk.W, padx=10)
        
        # Store reference to value label for updates
        setattr(self, f"{title.lower().replace(' ', '_')}_label", value_label)
        
    def create_price_display(self, parent, token, col):
        """Create a price display for a token"""
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=col, padx=5, sticky=tk.EW)
        parent.columnconfigure(col, weight=1)
        
        # Add border and padding using a canvas
        canvas = tk.Canvas(frame, bg="#313244", bd=0, highlightthickness=1, highlightbackground="#6c7086", height=60)
        canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Token name
        token_label = tk.Label(canvas, text=token, font=("Arial", 10, "bold"), fg="#cdd6f4", bg="#313244")
        token_label.pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        # Price frame
        price_frame = tk.Frame(canvas, bg="#313244")
        price_frame.pack(anchor=tk.W, fill=tk.X, padx=10, pady=(2, 10))
        
        # Price
        price_label = tk.Label(price_frame, text=f"${self.data_generator.prices[token]:.2f}", font=("Arial", 12), fg="#89dceb", bg="#313244")
        price_label.pack(side=tk.LEFT)
        
        # Change
        change_label = tk.Label(price_frame, text="+0.0%", font=("Arial", 10), fg="#a6e3a1", bg="#313244")
        change_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Store references for updates
        self.price_labels[token] = price_label
        self.price_changes[token] = change_label
        self.price_frames[token] = frame
