import os
import sys
from dotenv import load_dotenv
import logging
from trade_execution import set_wallet, execute_trade
from quantum import quantum_trade_strategy
from web3 import Web3

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BotX3-CLI")

# Load environment variables
load_dotenv()

def main():
    """Simple CLI version of the Quantum Bot X3 application"""
    print("=" * 50)
    print("QUANTUM BOT X3 - COMMAND LINE INTERFACE")
    print("=" * 50)
    
    # Get wallet details from .env file
    private_key = os.getenv('PRIVATE_KEY')
    bot_address = os.getenv('BOT_ADDRESS')
    
    if not private_key or not bot_address or private_key == "0xyour_private_key":
        print("No wallet configuration found.")
        private_key = input("Enter your private key: ")
        
        # Connect to Alchemy API
        alchemy_key = os.getenv('ALCHEMY_API_KEY', '')
        w3 = Web3(Web3.HTTPProvider(f"https://polygon-mainnet.g.alchemy.com/v2/{alchemy_key}"))
        
        try:
            # Remove 0x prefix if present
            if private_key.startswith("0x"):
                private_key = private_key[2:]
                
            # Generate account from private key
            account = w3.eth.account.from_key(private_key)
            bot_address = account.address
            
            print(f"Successfully connected to wallet: {bot_address[:8]}...")
            set_wallet(bot_address, private_key)
        except Exception as e:
            print(f"Error connecting to wallet: {e}")
            sys.exit(1)
    else:
        print(f"Using wallet from .env: {bot_address[:8]}...")
        set_wallet(bot_address, private_key)
    
    # Display menu
    while True:
        print("\nQUANTUM BOT X3 OPERATIONS:")
        print("1. Execute Quantum Analysis")
        print("2. Execute Trade on Polygon")
        print("3. Execute Cross-Chain Trade")
        print("4. Exit")
        
        choice = input("Select an option (1-4): ")
        
        if choice == "1":
            print("Running quantum analysis...")
            # Create sample chain data for demonstration
            chain_data = {
                "polygon": {"price": 1.2, "depth": 5000, "volume": 10000},
                "ethereum": {"price": 4500, "depth": 15000, "volume": 50000},
                "solana": {"price": 150, "depth": 8000, "volume": 25000}
            }
            # Try different strategy types
            strategies = ["sandwich", "mev", "flash"]
            for strategy in strategies:
                result = quantum_trade_strategy(chain_data, strategy)
                print(f"Quantum analysis result for {strategy} strategy: {result}")
        elif choice == "2":
            print("Preparing trade on Polygon...")
            amount = float(input("Enter trade amount (in ETH): "))
            print(f"Executing trade with {amount} ETH...")
            try:
                execute_trade("polygon", "polygon", amount_in=amount * 1e18)
                print("Trade simulation completed. This is running in demo mode.")
            except Exception as e:
                print(f"Trade error: {e}")
        elif choice == "3":
            print("Preparing cross-chain trade...")
            amount = float(input("Enter trade amount (in ETH): "))
            source = input("Source chain (polygon/bsc): ").lower()
            target = input("Target chain (polygon/bsc): ").lower()
            print(f"Executing cross-chain trade with {amount} ETH from {source} to {target}...")
            try:
                execute_trade(source, target, amount_in=amount * 1e18)
                print("Cross-chain trade simulation completed. This is running in demo mode.")
            except Exception as e:
                print(f"Trade error: {e}")
        elif choice == "4":
            print("Exiting Quantum Bot X3. Goodbye!")
            break
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
