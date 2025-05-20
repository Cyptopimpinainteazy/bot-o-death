#!/bin/bash

# Enhanced Quantum Trade AI Startup Script

# Load environment variables
set -a
source .env
set +a

# Check for Python virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]]; then
    # Windows
    source .venv/Scripts/activate
else
    # Linux/Mac
    source .venv/bin/activate
fi

# Install or update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the web dashboard
echo "Starting Enhanced Quantum Trade AI Dashboard..."
python server.py 