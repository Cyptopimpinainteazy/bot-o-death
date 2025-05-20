# Enhanced Quantum Trade AI

Advanced trading platform integrating quantum computations, AI, blockchain technologies and real-time market data.

## Features

- **Multi-Source Data Collection**: Fetch market data from CEX, DEX and Chainlink price feeds
- **Quantum-Enhanced Algorithms**: Leverage quantum computing for portfolio optimization
- **Smart Contract Integration**: Deploy and interact with DeFi protocols
- **Real-Time Dashboard**: Monitor markets and trading performance
- **Web3 Wallet Connection**: Connect to MetaMask and other wallets

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/EnhancedQuantumTradeAI.git
   cd EnhancedQuantumTradeAI
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables (create `.env` file):
   ```
   # Blockchain RPC URLs
   POLYGON_RPC_URL=https://polygon-rpc.com
   
   # API Keys
   POLYGON_API_KEY=your_polygon_api_key
   
   # Chainlink feed addresses (comma-separated)
   CHAINLINK_FEEDS=0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419,0x8fFfFfd4AfB6115b954Bd326cbe7B4BA576818f6
   
   # Default symbols
   DEFAULT_SYMBOLS=ETH/USDT,BTC/USDT
   ```

## Usage

### Web Dashboard

Run the Flask web server:
```
python server.py
```

Access the dashboard at `http://localhost:5000`.

### Terminal Trading

Run the trading application:
```
python EnhancedQuantumTrading/run_enhanced_trader.py
```

### Smart Contract Deployment

1. Install Hardhat or Truffle
2. Deploy contracts:
   ```
   cd EnhancedQuantumTrading/contracts
   npx hardhat compile
   npx hardhat deploy --network polygon
   ```

## Architecture

- **Core**: Trading logic, data collection, and analysis
- **Dashboard**: Real-time monitoring and controls
- **Models**: Machine learning and quantum models
- **Contracts**: Smart contracts for on-chain operations

## Technologies

- **Python**: Core trading, ML, and data processing
- **Solidity**: Smart contracts (OpenZeppelin, Chainlink)
- **Web3.py**: Blockchain interaction
- **Flask**: Web dashboard backend
- **Lightweight Charts**: Technical chart visualization
- **Ethers.js**: Web3 integration
- **Polygon API**: Market data
- **CCXT**: Exchange integration
- **Quantum Libraries**: PennyLane for quantum algorithms

## License

MIT 