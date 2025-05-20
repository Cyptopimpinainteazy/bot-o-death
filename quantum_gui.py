import sys
import requests
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTextEdit, QGridLayout, QGraphicsView, 
                             QGraphicsScene, QLineEdit, QCheckBox)
from PyQt5.QtCore import QTimer, Qt, QRectF, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QPalette, QColor, QPen, QBrush, QPixmap
from web3 import Web3
import random
from playsound import playsound
from threading import Thread
from trade_execution import set_wallet, execute_trade

class QuantumLogin(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("""
            QWidget { background-color: #0a0a15; }
            QLabel { color: #00ffcc; font-family: 'Orbitron'; font-size: 20px; text-shadow: 0 0 10px #00ffcc; }
            QPushButton { 
                background-color: #1c2526; 
                color: #ff00ff; 
                border: 3px solid #00ffcc; 
                padding: 10px; 
                font-family: 'Orbitron'; 
                font-size: 16px; 
            }
            QLineEdit { 
                background-color: #0f0f1a; 
                color: #00ffcc; 
                border: 2px solid #ff00ff; 
                font-family: 'Courier New'; 
                font-size: 14px; 
            }
            QCheckBox { color: #ff00ff; font-family: 'Orbitron'; }
        """)
        
        layout = QVBoxLayout(self)
        
        header = QLabel("Quantum Keypad - Authenticate")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        self.qr_label = QLabel("Scan with MetaMask (Manual Mode Active)")
        self.qr_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.qr_label)

        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Enter Private Key")
        self.key_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.key_input)

        self.show_key = QCheckBox("Reveal Quantum Key")
        self.show_key.stateChanged.connect(self.toggle_key_visibility)
        layout.addWidget(self.show_key)

        self.connect_button = QPushButton("Engage Quantum Link")
        self.connect_button.clicked.connect(self.connect_wallet)
        layout.addWidget(self.connect_button)

        self.status = QLabel("Awaiting Authentication...")
        self.status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status)

    def toggle_key_visibility(self):
        self.key_input.setEchoMode(QLineEdit.Normal if self.show_key.isChecked() else QLineEdit.Password)

    def connect_wallet(self):
        private_key = self.key_input.text().strip()
        if private_key.startswith("0x"):
            private_key = private_key[2:]
        try:
            w3 = Web3(Web3.HTTPProvider("https://polygon-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_API_KEY"))
            account = w3.eth.account.from_key(private_key)
            balance = w3.eth.get_balance(account.address) / 1e18
            self.status.setText(f"Quantum Link Established: {account.address[:8]}... ({balance:.2f} MATIC)")
            self.parent.set_wallet(account.address, private_key)
            self.parent.show_main_gui()
        except Exception as e:
            self.status.setText(f"Quantum Link Failed: {str(e)}")

class QuantumGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bot X3 - Quantum Hypergrid")
        self.setGeometry(100, 100, 1400, 800)
        self.wallet_address = None
        self.private_key = None
        self.init_login()

    def init_login(self):
        self.login_widget = QuantumLogin(self)
        self.setCentralWidget(self.login_widget)

    def set_wallet(self, address, private_key):
        self.wallet_address = address
        self.private_key = private_key
        set_wallet(address, private_key)

    def show_main_gui(self):
        self.init_ui()
        self.start_timers()

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0a0a15; }
            QLabel { color: #00ffcc; font-family: 'Orbitron'; font-size: 18px; text-shadow: 0 0 10px #00ffcc; }
            QPushButton { 
                background-color: #1c2526; 
                color: #ff00ff; 
                border: 3px solid #00ffcc; 
                padding: 10px; 
                font-family: 'Orbitron'; 
                font-size: 16px; 
                box-shadow: 0 0 15px #ff00ff; 
            }
            QPushButton:hover { background-color: #2a4066; }
            QTextEdit { 
                background-color: #0f0f1a; 
                color: #00ffcc; 
                border: 2px solid #ff00ff; 
                font-family: 'Courier New'; 
                font-size: 14px; 
            }
            QGraphicsView { border: 2px solid #00ffcc; background-color: #0a0a15; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        header = QLabel(f"Quantum Hypergrid - Bot X3 | Pilot: {self.wallet_address[:8]}...")
        header.setFont(QFont("Orbitron", 32, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        grid_layout = QGridLayout()
        self.chain_labels = {}
        self.pulse_animations = {}
        for i, chain in enumerate(['Polygon', 'Solana']):
            self.chain_labels[chain] = {
                'price': QLabel(f"{chain} Price: $0.00"),
                'depth': QLabel(f"{chain} Depth: 0"),
                'volume': QLabel(f"{chain} Volume: 0")
            }
            for key, label in self.chain_labels[chain].items():
                grid_layout.addWidget(label, i, list(self.chain_labels[chain].keys()).index(key))
                anim = QPropertyAnimation(label, b"styleSheet")
                anim.setDuration(1000)
                anim.setStartValue(f"color: #00ffcc; font-size: 18px;")
                anim.setEndValue(f"color: #ff00ff; font-size: 20px;")
                anim.setEasingCurve(QEasingCurve.InOutSine)
                anim.setLoopCount(-1)
                self.pulse_animations[f"{chain}_{key}"] = anim

        self.core_view = QGraphicsView()
        self.core_scene = QGraphicsScene()
        self.core_view.setScene(self.core_scene)
        self.particles = []
        self.update_quantum_core()
        grid_layout.addWidget(self.core_view, 0, 3, 2, 1)

        main_layout.addLayout(grid_layout)

        self.status = QLabel("Quantum Hypergrid: Disengaged")
        self.status.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status)

        control_layout = QHBoxLayout()
        self.trade_button = QPushButton("Execute Quantum Leap")
        self.trade_button.clicked.connect(self.execute_trade)
        control_layout.addWidget(self.trade_button)
        main_layout.addLayout(control_layout)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        main_layout.addWidget(self.log)

    def update_quantum_core(self):
        self.core_scene.clear()
        pen = QPen(QColor("#00ffcc"), 2)
        brush = QBrush(QColor("#ff00ff"))
        for i in range(3):
            y = 50 + i * 60
            self.core_scene.addLine(0, y, 300, y, pen)
            if random.random() > 0.5:
                self.core_scene.addRect(80, y-15, 30, 30, pen, brush)
            if random.random() > 0.5:
                self.core_scene.addEllipse(200, y-15, 30, 30, pen, brush)
        for _ in range(10):
            x, y = random.randint(0, 300), random.randint(0, 200)
            particle = self.core_scene.addEllipse(x, y, 5, 5, QPen(QColor("#00ffcc")), QBrush(QColor("#ff00ff")))
            self.particles.append(particle)
        self.core_view.setSceneRect(QRectF(0, 0, 300, 200))

    def animate_particles(self):
        for particle in self.particles:
            x = particle.rect().x() + random.randint(-5, 5)
            y = particle.rect().y() + random.randint(-5, 5)
            particle.setRect(x, y, 5, 5)
            if x < 0 or x > 295 or y < 0 or y > 195:
                particle.setRect(random.randint(0, 295), random.randint(0, 195), 5, 5)

    def start_timers(self):
        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self.update_data)
        self.data_timer.start(5000)
        self.particle_timer = QTimer()
        self.particle_timer.timeout.connect(self.animate_particles)
        self.particle_timer.start(100)

    def update_data(self):
        try:
            rl_response = requests.get("http://localhost:5000/rl").json()
            action = rl_response["action"]
            timestamp = rl_response["timestamp"]

            chain_data = requests.get("http://localhost:5000/quantum").json()
            if "action" not in chain_data:
                for chain in self.chain_labels:
                    self.chain_labels[chain]['price'].setText(f"{chain} Price: ${chain_data[chain]['price']:.2f}")
                    self.chain_labels[chain]['depth'].setText(f"{chain} Depth: {chain_data[chain]['depth']:.0f}")
                    self.chain_labels[chain]['volume'].setText(f"{chain} Volume: {chain_data[chain]['volume']:.0f}")
                    self.pulse_animations[f"{chain}_price"].start()

            w3 = Web3(Web3.HTTPProvider("https://polygon-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_API_KEY"))
            balance = w3.eth.get_balance(self.wallet_address) / 1e18
            x3star_balance = w3.eth.contract(address=CONTRACT_ADDRESS, abi=X3STAR_ABI).functions.balanceOf(self.wallet_address).call() / 1e18
            self.status.setText(f"Quantum Hypergrid: Engaged - Vector {action} | MATIC: {balance:.2f} | $X3STAR: {x3star_balance:.2f}")

            if action == 1:
                log_msg = f"[Grid {timestamp}] AI Buy: Polygon"
            elif action == 2:
                log_msg = f"[Grid {timestamp}] AI Sell: Polygon"
            elif action == 3:
                log_msg = f"[Grid {timestamp}] AI Sandwich Trade: Polygon"
            else:
                log_msg = f"[Grid {timestamp}] Stabilizing Hypergrid"
            self.log.append(log_msg)

        except Exception as e:
            self.log.append(f"[Grid Collapse] Anomaly: {str(e)}")

    def execute_trade(self):
        self.log.append("[Hypergrid Command] Engaging Quantum Leap...")
        Thread(target=lambda: playsound("quantum_leap.wav")).start()
        rl_response = requests.get("http://localhost:5000/rl").json()
        action = rl_response["action"]
        if action < 3:
            buy_idx, sell_idx = divmod(action, 1)
            sell_idx = sell_idx + (sell_idx >= buy_idx)
            buy_chain = ['Polygon', 'Solana'][buy_idx]
            sell_chain = ['Polygon', 'Solana'][sell_idx]
            if execute_trade(buy_chain, sell_chain):
                self.log.append(f"[Quantum Triumph] Leap Successful: {buy_chain} -> {sell_chain}")
            else:
                self.log.append("[Quantum Rift] Leap Failed")
        else:
            self.log.append("[Quantum Hold] No Leap Vector")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Courier New", 12))
    window = QuantumGUI()
    window.show()
    sys.exit(app.exec_())
