#!/usr/bin/env python
import os
import sys
import re
import shutil
from pathlib import Path

"""
This script updates the Flask server to serve the React app.
It adds new API endpoints for the UI to interact with the trading bot.
"""

def backup_server_file():
    """Create a backup of the original server.py file"""
    if os.path.exists('server.py'):
        shutil.copy('server.py', 'server.py.bak')
        print("✅ Backed up original server.py to server.py.bak")
    else:
        print("❌ server.py not found!")
        sys.exit(1)

def update_server_file():
    """Update server.py to serve the React app and add new API endpoints"""
    with open('server.py', 'r') as f:
        content = f.read()
    
    # Add new imports
    imports_pattern = r'from flask import Flask, render_template, jsonify, request'
    new_imports = 'from flask import Flask, render_template, jsonify, request, send_from_directory\nimport asyncio\nfrom concurrent.futures import ThreadPoolExecutor'
    
    if 'send_from_directory' not in content:
        content = re.sub(imports_pattern, new_imports, content)
    
    # Add new API endpoints
    endpoint_section = """
# Portfolio management API endpoints
@app.route('/api/portfolio/value')
def portfolio_value():
    # This would integrate with your portfolio manager in a real implementation
    return jsonify({
        'total': 125750.42,
        'change_24h': 2450.75,
        'change_24h_percent': 1.99,
        'cash_balance': 45320.18
    })

@app.route('/api/portfolio/positions')
def portfolio_positions():
    # This would integrate with your portfolio manager in a real implementation
    return jsonify([
        {
            'id': 1,
            'symbol': 'ETH/USD',
            'position_type': 'LONG',
            'entry_price': 3245.50,
            'current_price': 3318.75,
            'quantity': 15.5,
            'value': 51440.63,
            'unrealized_pnl': 1135.38,
            'unrealized_pnl_percent': 2.26,
            'stop_loss': 3100.00,
            'take_profit': 3500.00
        },
        {
            'id': 2,
            'symbol': 'BTC/USD',
            'position_type': 'LONG',
            'entry_price': 42100.25,
            'current_price': 43250.80,
            'quantity': 0.65,
            'value': 28113.02,
            'unrealized_pnl': 747.86,
            'unrealized_pnl_percent': 2.73,
            'stop_loss': 40000.00,
            'take_profit': 45000.00
        },
        {
            'id': 3,
            'symbol': 'SOL/USD',
            'position_type': 'SHORT',
            'entry_price': 105.75,
            'current_price': 101.20,
            'quantity': 85.0,
            'value': 8602.00,
            'unrealized_pnl': 386.75,
            'unrealized_pnl_percent': 4.71,
            'stop_loss': 115.00,
            'take_profit': 90.00
        }
    ])

@app.route('/api/portfolio/metrics')
def portfolio_metrics():
    # This would integrate with your portfolio manager in a real implementation
    return jsonify({
        'sharpe_ratio': 1.85,
        'sortino_ratio': 2.12,
        'max_drawdown': -12.4,
        'win_rate': 62.5,
        'profit_factor': 1.75,
        'avg_win': 3.2,
        'avg_loss': -1.8,
        'total_trades': 48,
        'profitable_trades': 30,
        'loss_trades': 18
    })

# Trading execution API endpoints
@app.route('/api/execute_flashloan', methods=['POST'])
def execute_flashloan():
    data = request.json
    
    if not data:
        return jsonify({'status': 'error', 'message': 'No data provided'}), 400
    
    # In a real implementation, this would call the trading_execution module
    try:
        # Create a ThreadPoolExecutor to run the async function
        with ThreadPoolExecutor() as executor:
            # Run the async function in the executor
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # This is a placeholder that would actually call your trading logic
            # result = loop.run_until_complete(
            #     trading_logic.execute_triple_flashloan(
            #         data['chain'],
            #         data['tokens'],
            #         data['amounts']
            #     )
            # )
            
            # Mock successful result for the frontend demo
            result = {
                'status': 'success',
                'chain': data.get('chain', 'polygon'),
                'tx_hash': '0x' + '0123456789abcdef' * 4
            }
            
            loop.close()
            
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/execute_sandwich', methods=['POST'])
def execute_sandwich():
    data = request.json
    
    if not data:
        return jsonify({'status': 'error', 'message': 'No data provided'}), 400
    
    # Mock successful result for the frontend demo
    result = {
        'status': 'success',
        'chain': data.get('chain', 'polygon'),
        'tx_hash': '0x' + '0123456789abcdef' * 4
    }
    
    return jsonify(result)

@app.route('/api/execute_mev', methods=['POST'])
def execute_mev():
    data = request.json
    
    if not data:
        return jsonify({'status': 'error', 'message': 'No data provided'}), 400
    
    # Mock successful result for the frontend demo
    result = {
        'status': 'success',
        'chain': data.get('chain', 'polygon'),
        'strategy': data.get('strategy', 'arbitrage'),
        'tx_hash': '0x' + '0123456789abcdef' * 4
    }
    
    return jsonify(result)

# Serve React app in production
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join('frontend/build', path)):
        return send_from_directory('frontend/build', path)
    else:
        return send_from_directory('frontend/build', 'index.html')
"""
    
    # Add the new endpoints before the if __name__ == '__main__': block
    if 'if __name__ == \'__main__\':' in content and endpoint_section not in content:
        content = content.replace('if __name__ == \'__main__\':', endpoint_section + '\nif __name__ == \'__main__\':')
    
    with open('server.py', 'w') as f:
        f.write(content)
    
    print("✅ Updated server.py with new API endpoints and React app serving")

def create_build_frontend_script():
    """Create a script to build the React frontend"""
    build_script = """#!/bin/bash
# Script to build the React frontend and copy to the correct location

set -e

echo "Building React frontend..."
cd frontend
npm install
npm run build

echo "Copying build files to static directory..."
rm -rf ../static/react
mkdir -p ../static/react
cp -r build/* ../static/react/

echo "Frontend build complete!"
"""
    
    with open('build_frontend.sh', 'w') as f:
        f.write(build_script)
    
    os.chmod('build_frontend.sh', 0o755)
    print("✅ Created build_frontend.sh script")

def main():
    """Main function to update the server and create build script"""
    print("🚀 Updating Flask server to serve React app...")
    
    backup_server_file()
    update_server_file()
    create_build_frontend_script()
    
    print("\n🎉 All done! Next steps:")
    print("1. Build the React frontend with: ./build_frontend.sh")
    print("2. Start the server with: python server.py")
    print("3. Access the application at: http://localhost:5000")

if __name__ == "__main__":
    main() 