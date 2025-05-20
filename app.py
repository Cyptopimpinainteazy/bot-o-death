import sys
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QPushButton
from PyQt5.QtCore import QTimer
from core.trading_execution import TradingLogic, ChainConnector
from core.signal_analyzer import SignalAnalyzer, TechnicalAnalysis
from core.fund_manager import FundManager
import yaml
import asyncio
from aiohttp import ClientSession

logger = logging.getLogger(__name__)

class TradingSwarm(QMainWindow):
    def __init__(self, config=None):
        if QApplication.instance() is None:
            self.app = QApplication([])
        super().__init__()
        self.setWindowTitle("Enhanced Quantum Trading - PIMP DASHBOARD")
        self.setGeometry(100, 100, 1200, 800)
        self.init_ui()
        self.config = config
        if self.config is None:
            self.load_config()
        self.setup_trading_logic()
        self.start_update_timer()

    def init_ui(self):
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Status label
        self.status_label = QLabel("Status: Initializing...", self)
        self.layout.addWidget(self.status_label)

        # Profit display
        self.profit_label = QLabel("Total Profit: $0.00", self)
        self.layout.addWidget(self.profit_label)

        # Chain balances table
        self.balances_table = QTableWidget(4, 2)  # 4 chains, 2 cols (Chain, Balance)
        self.balances_table.setHorizontalHeaderLabels(["Chain", "Balance"])
        self.layout.addWidget(self.balances_table)

        # Subgraph metrics table
        self.metrics_table = QTableWidget(12, 2)  # 12 subgraphs, 2 cols (Subgraph, Value)
        self.metrics_table.setHorizontalHeaderLabels(["Subgraph", "Latest Metric"])
        self.layout.addWidget(self.metrics_table)

        # Start/Stop button
        self.toggle_button = QPushButton("Start Trading", self)
        self.toggle_button.clicked.connect(self.toggle_trading)
        self.layout.addWidget(self.toggle_button)

    def load_config(self):
        with open('config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def setup_trading_logic(self):
        self.chain_connector = ChainConnector(self.config)
        self.signal_analyzer = SignalAnalyzer(TechnicalAnalysis())
        self.signal_analyzer.config = self.config
        self.trading_logic = TradingLogic(self.config)
        self.trading_logic.chain_connector = self.chain_connector
        self.trading_logic.signal_analyzer = self.signal_analyzer
        self.fund_manager = FundManager(self.config, self.trading_logic, self.signal_analyzer)
        self.is_trading = False
        self.total_profit = 0.0

    def start_update_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_dashboard)
        self.timer.start(5000)  # Update every 5 seconds

    def toggle_trading(self):
        if not self.is_trading:
            self.is_trading = True
            self.toggle_button.setText("Stop Trading")
            asyncio.create_task(self.run_trading())
        else:
            self.is_trading = False
            self.toggle_button.setText("Start Trading")

    async def run_trading(self):
        symbols = ["MATIC/USDC", "ETH/USDC", "BNB/BUSD"]
        chains = ["polygon", "ethereum", "bsc", "arbitrum_one"]
        while self.is_trading:
            async with ClientSession() as session:
                tasks = [self.trading_logic.execute_triple_flashloan(chain, 
                        ["0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"] * 4, 
                        [int(0.01 * 1e18)] * 4) for chain in chains]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if "executed" in str(result):
                        self.total_profit += 0.01  # Simplified profit—adjust later
            await asyncio.sleep(10)  # Trade every 10 seconds—tune this

    def update_dashboard(self):
        # Status
        self.status_label.setText(f"Status: {'Trading' if self.is_trading else 'Idle'}")

        # Profit
        self.profit_label.setText(f"Total Profit: ${self.total_profit:.2f}")

        # Chain balances
        asyncio.run(self.fund_manager.update_balances())
        balances = self.fund_manager.balances
        for row, chain in enumerate(['ethereum', 'polygon', 'bsc', 'arbitrum_one']):
            self.balances_table.setItem(row, 0, QTableWidgetItem(chain))
            self.balances_table.setItem(row, 1, QTableWidgetItem(f"{balances.get(chain, 0):.4f}"))

        # Subgraph metrics
        metrics = asyncio.run(self.signal_analyzer.analyze_market_data("MATIC/USDC", {}, "polygon"))
        indicators = metrics["technical_indicators"]
        subgraphs = [
            ("Aave V3", indicators["flashloan_opps"]),
            ("Uniswap V3", indicators["current_price"]),
            ("Yearn", indicators["yearn_yield"]),
            ("dYdX", indicators["dydx_volume"]),
            ("Convex", indicators["convex_tvl"]),
            ("Livepeer", indicators["livepeer_stake"]),
            ("PoolTogether", indicators["pooltogether_prize"]),
            ("UMA", indicators["uma_supply"]),
            ("OpenSea", indicators["opensea_volume"]),
            ("Aave V2", indicators["aave_v2_opps"]),
            ("Lido", indicators["lido_staked"]),
            ("QuickSwap", indicators["quickswap_volume"])
        ]
        for row, (name, value) in enumerate(subgraphs):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{value:.2f}"))

    async def stress_test(self, chains, symbols, rps_target=10):
        """Run a stress test with the specified parameters"""
        self.is_trading = True
        await self.run_trading()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TradingSwarm()
    window.show()
    sys.exit(app.exec_())