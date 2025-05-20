import os
from dotenv import load_dotenv
import requests
from stable_baselines3 import DQN
from rl_agent import TradingEnv
from quantum import quantum_trade_strategy
from web3 import Web3
import logging

logger = logging.getLogger("BotX3")
load_dotenv()

# Web3.py for Polygon Mumbai (testing)
ALCHEMY_API_KEY = os.getenv('ALCHEMY_API_KEY', '83UihM_ylIp0XVxt7OUvJ-G--DDTnrn5')
w3_polygon = Web3(Web3.HTTPProvider(f"https://polygon-mumbai.g.alchemy.com/v2/{ALCHEMY_API_KEY}"))
w3_bsc = Web3(Web3.HTTPProvider("https://bsc-dataseed.binance.org/"))
BOT_ADDRESS = None
PRIVATE_KEY = None
CONTRACT_ADDRESS = "0xDeployedX3STARAddress"  # Update after deployment
# Sample minimal ABI for demonstration purposes
X3STAR_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    }
]

# Initialize contract in try/except block to handle errors gracefully
try:
    contract = w3_polygon.eth.contract(address=CONTRACT_ADDRESS, abi=X3STAR_ABI)
except Exception as e:
    logger.warning(f"Failed to initialize contract: {e}")
    contract = None
WMATIC = "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270"
USDC = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

def set_wallet(address, private_key):
    global BOT_ADDRESS, PRIVATE_KEY
    BOT_ADDRESS = address
    PRIVATE_KEY = private_key

def monitor_mempool():
    pending = w3_polygon.eth.get_block("pending", full_transactions=True)["transactions"]
    for tx in pending:
        if tx["to"] == "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff":
            return tx["value"], tx["input"]
    return None, None

def fetch_cross_chain_prices():
    polygon_price = float(requests.get("http://localhost:5000/quantum").json()["Polygon"]["price"])
    bsc_price = polygon_price * 1.05
    return {"Polygon": polygon_price, "BSC": bsc_price}

def execute_buy(amount_in=1e18, token=USDC):
    tx = contract.functions.executeSandwichTrade(token, amount_in, 0, 0).build_transaction({
        "from": BOT_ADDRESS,
        "value": amount_in,
        "nonce": w3_polygon.eth.get_transaction_count(BOT_ADDRESS),
        "gas": 400000,
        "gasPrice": w3_polygon.to_wei("50", "gwei")
    })
    signed_tx = w3_polygon.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3_polygon.eth.send_raw_transaction(signed_tx.rawTransaction)
    receipt = w3_polygon.eth.wait_for_transaction_receipt(tx_hash)
    return receipt.status == 1

def execute_sell(amount_in=1e18, token=USDC):
    tx = contract.functions.executeMEVTrade([token, WMATIC], amount_in, 0).build_transaction({
        "from": BOT_ADDRESS,
        "value": 0,
        "nonce": w3_polygon.eth.get_transaction_count(BOT_ADDRESS),
        "gas": 350000,
        "gasPrice": w3_polygon.to_wei("50", "gwei")
    })
    signed_tx = w3_polygon.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3_polygon.eth.send_raw_transaction(signed_tx.rawTransaction)
    receipt = w3_polygon.eth.wait_for_transaction_receipt(tx_hash)
    return receipt.status == 1

def execute_cross_chain_trade(amount_in=1e18):
    prices = fetch_cross_chain_prices()
    if prices["Polygon"] < prices["BSC"]:
        tx = contract.functions.depositCrossChainProfit("Polygon").build_transaction({
            "from": BOT_ADDRESS,
            "value": amount_in,
            "nonce": w3_polygon.eth.get_transaction_count(BOT_ADDRESS),
            "gas": 50000,
            "gasPrice": w3_polygon.to_wei("30", "gwei")
        })
        signed_tx = w3_polygon.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3_polygon.eth.send_raw_transaction(signed_tx.rawTransaction)
        receipt = w3_polygon.eth.wait_for_transaction_receipt(tx_hash)
        return receipt.status == 1
    return False

def execute_triangle_trade(amount_in=1e18):
    tx = contract.functions.executeTriangleTrade(WMATIC, USDC, amount_in).build_transaction({
        "from": BOT_ADDRESS,
        "value": amount_in,
        "nonce": w3_polygon.eth.get_transaction_count(BOT_ADDRESS),
        "gas": 300000,
        "gasPrice": w3_polygon.to_wei("30", "gwei")
    })
    signed_tx = w3_polygon.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3_polygon.eth.send_raw_transaction(signed_tx.rawTransaction)
    receipt = w3_polygon.eth.wait_for_transaction_receipt(tx_hash)
    return receipt.status == 1

def execute_sandwich_trade(amount_in=1e18):
    chain_data = fetch_uniswap_data(CHAINS)
    if quantum_trade_strategy(chain_data, "sandwich"):
        value, _ = monitor_mempool()
        if value and value > 2e18:
            amount_out_min_frontrun = int(amount_in * 0.98)
            amount_out_min_backrun = int(amount_in * 0.95)
            tx = contract.functions.executeSandwichTrade(
                USDC, amount_in, amount_out_min_frontrun, amount_out_min_backrun
            ).build_transaction({
                "from": BOT_ADDRESS,
                "value": amount_in,
                "nonce": w3_polygon.eth.get_transaction_count(BOT_ADDRESS),
                "gas": 400000,
                "gasPrice": w3_polygon.to_wei("50", "gwei")
            })
            signed_tx = w3_polygon.eth.account.sign_transaction(tx, PRIVATE_KEY)
            tx_hash = w3_polygon.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = w3_polygon.eth.wait_for_transaction_receipt(tx_hash)
            return receipt.status == 1
    return False

def execute_mev_trade():
    chain_data = fetch_uniswap_data(CHAINS)
    if quantum_trade_strategy(chain_data, "mev"):
        path = [WMATIC, USDC, WMATIC]
        amount_in = 1e18
        amount_out_min = int(amount_in * 1.01)
        tx = contract.functions.executeMEVTrade(
            path, amount_in, amount_out_min
        ).build_transaction({
            "from": BOT_ADDRESS,
            "value": amount_in,
            "nonce": w3_polygon.eth.get_transaction_count(BOT_ADDRESS),
            "gas": 350000,
            "gasPrice": w3_polygon.to_wei("50", "gwei")
        })
        signed_tx = w3_polygon.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3_polygon.eth.send_raw_transaction(signed_tx.rawTransaction)
        receipt = w3_polygon.eth.wait_for_transaction_receipt(tx_hash)
        return receipt.status == 1
    return False

def execute_flash_loan_trade(amount_in=1e18):
    chain_data = fetch_uniswap_data(CHAINS)
    if quantum_trade_strategy(chain_data, "flash"):
        tx = contract.functions.executeFlashLoanArbitrage(USDC, amount_in).build_transaction({
            "from": BOT_ADDRESS,
            "nonce": w3_polygon.eth.get_transaction_count(BOT_ADDRESS),
            "gas": 500000,
            "gasPrice": w3_polygon.to_wei("50", "gwei")
        })
        signed_tx = w3_polygon.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3_polygon.eth.send_raw_transaction(signed_tx.rawTransaction)
        receipt = w3_polygon.eth.wait_for_transaction_receipt(tx_hash)
        return receipt.status == 1
    return False

def execute_trade(buy_chain, sell_chain, amount_in=1e18):
    env = TradingEnv(CHAINS)
    model = DQN.load("models/dqn_buy_sell", env=env)
    obs = env.reset()
    action, _ = model.predict(obs)

    if action == 1:
        return execute_buy(amount_in, USDC if buy_chain == "Polygon" else WMATIC)
    elif action == 2:
        return execute_sell(amount_in, USDC if sell_chain == "Polygon" else WMATIC)
    elif action == 3:
        return execute_sandwich_trade(amount_in)
    elif buy_chain != sell_chain:
        return execute_cross_chain_trade(amount_in)
    else:
        return execute_triangle_trade(amount_in)
    return False
