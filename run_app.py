#!/usr/bin/env python3
"""
Run script for Enhanced Quantum Trade AI

This script helps run the trading application by ensuring the correct Python path
"""
import sys
import os
import subprocess

# Add user's site-packages to path
home_dir = os.path.expanduser("~")
site_packages = os.path.join(home_dir, ".local/lib/python3.11/site-packages")
sys.path.append(site_packages)

# Check if we can now import web3
try:
    import web3
    print(f"Found web3 version: {web3.__version__}")
except ImportError:
    print("Web3 module still not found. Trying to install it...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "web3", "--user"], check=True)
        print("Web3 installed successfully")
    except Exception as e:
        print(f"Error installing web3: {str(e)}")

# Create swarm placeholder class to avoid ImportError
class Swarm:
    def __init__(self):
        print("Using placeholder Swarm class")

# Place this in sys.modules to avoid import errors
sys.modules['swarm'] = type('swarm', (), {'Swarm': Swarm})

# Run the app
print("Running Enhanced Quantum Trade AI application...")
os.environ['PYTHONPATH'] = f"{os.getcwd()}:{site_packages}"

# Run the app with command line arguments
args = "sync_blockchain --symbols ETH USDC --mode paper --chain ethereum"
cmd = f"{sys.executable} app.py {args}"
print(f"Running command: {cmd}")
result = subprocess.run(cmd, shell=True)
print(f"Process finished with exit code {result.returncode}")
