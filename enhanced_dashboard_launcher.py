#!/usr/bin/env python3
"""
Enhanced Quantum Trade AI Dashboard - Main Launcher
Combines all dashboard components and launches the dashboard
"""
import os
import sys
import tkinter as tk

# Make sure the enhanced_dashboard package is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import required modules
    from enhanced_dashboard.data_manager import DataManager, TOKENS, CHAINS, STRATEGIES, MARKET_CONDITIONS
    from enhanced_dashboard.dashboard_ui import *
    from enhanced_dashboard.dashboard_ui_part2 import *
    from enhanced_dashboard.dashboard_ui_part3 import *
    
    print("Starting Enhanced Quantum Trade AI Dashboard...")
    
    # Create and run the dashboard
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you have matplotlib installed. Try: pip install matplotlib")
    
except Exception as e:
    print(f"Error starting dashboard: {e}")
