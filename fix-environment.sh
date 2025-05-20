#!/bin/bash
# Script to fix environment issues for Enhanced Quantum Trade AI deployment
set -e

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Enhanced Quantum Trade AI - Environment Fix Tool ===${NC}"
echo -e "${YELLOW}This script will attempt to fix issues found during the environment check.${NC}"
echo

# 1. Fix Node.js dependencies
echo -e "${BLUE}Step 1: Installing Node.js dependencies...${NC}"
if [ ! -d "node_modules" ]; then
  echo -e "${YELLOW}Running npm install...${NC}"
  npm install
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Node.js dependencies installed successfully.${NC}"
  else
    echo -e "${RED}✗ Failed to install Node.js dependencies.${NC}"
    echo -e "${YELLOW}Please try running 'npm install' manually.${NC}"
  fi
else
  echo -e "${GREEN}✓ node_modules directory already exists.${NC}"
fi

# 2. Test and fix Polygon RPC URL
echo -e "\n${BLUE}Step 2: Testing Polygon RPC URL...${NC}"

# Check if .env file exists
if [ ! -f ".env" ]; then
  echo -e "${RED}✗ .env file not found!${NC}"
  echo -e "${YELLOW}Creating a template .env file...${NC}"
  
  cat > .env << EOF
# Wallet Configuration - REPLACE WITH YOUR ACTUAL VALUES
PRIVATE_KEY=your_private_key_here
WALLET_ADDRESS=your_wallet_address_here

# Primary Network Endpoints
POLYGON_RPC_URL=https://polygon-rpc.com
POLYGON_WSS_URL=wss://polygon-rpc.com

# API Keys
POLYGONSCAN_API_KEY=your_polygonscan_api_key_here
EOF

  echo -e "${YELLOW}Template .env file created. Please edit it with your actual values.${NC}"
  echo -e "${YELLOW}Then run this script again.${NC}"
  exit 1
fi

# Source the .env file
source .env

# Check if POLYGON_RPC_URL is set
if [ -z "$POLYGON_RPC_URL" ]; then
  echo -e "${RED}✗ POLYGON_RPC_URL is not set in .env file.${NC}"
  echo -e "${YELLOW}Here are some common Polygon RPC URLs you can use:${NC}"
  echo -e "  - https://polygon-rpc.com"
  echo -e "  - https://rpc-mainnet.matic.network"
  echo -e "  - https://rpc-mainnet.maticvigil.com"
  echo -e "${YELLOW}Please add one of these to your .env file.${NC}"
else
  # Test the Polygon RPC URL
  echo -e "${YELLOW}Testing connection to $POLYGON_RPC_URL...${NC}"
  response=$(curl -s -X POST -H "Content-Type: application/json" --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' $POLYGON_RPC_URL)
  
  if echo "$response" | grep -q "result"; then
    echo -e "${GREEN}✓ Successfully connected to Polygon network!${NC}"
    block_number=$(echo "$response" | grep -o '"result":"0x[^"]*"' | cut -d'"' -f4)
    echo -e "${GREEN}  Current block number: $block_number${NC}"
    
    # Test gas price API
    gas_response=$(curl -s -X POST -H "Content-Type: application/json" --data '{"jsonrpc":"2.0","method":"eth_gasPrice","params":[],"id":1}' $POLYGON_RPC_URL)
    
    if echo "$gas_response" | grep -q "result"; then
      echo -e "${GREEN}✓ Gas price API is working.${NC}"
    else
      echo -e "${RED}✗ Failed to query gas price.${NC}"
      echo -e "${YELLOW}This might be an issue with the RPC provider. Consider using a different provider.${NC}"
    fi
  else
    echo -e "${RED}✗ Failed to connect to Polygon network!${NC}"
    echo -e "${YELLOW}Response: $response${NC}"
    echo -e "${YELLOW}The POLYGON_RPC_URL in your .env file might be incorrect or the service might be down.${NC}"
    echo -e "${YELLOW}Please update it with one of the URLs mentioned above.${NC}"
  fi
fi

# 3. Check if there are any other issues
echo -e "\n${BLUE}Step 3: Running full environment check again...${NC}"
./deployment/check-environment.sh

if [ $? -eq 0 ]; then
  echo -e "\n${GREEN}All environment issues have been fixed!${NC}"
  echo -e "${YELLOW}You can now run the deployment script:${NC}"
  echo -e "  ./run-deploy.sh"
else
  echo -e "\n${YELLOW}Some issues still need to be resolved manually.${NC}"
  echo -e "${YELLOW}Please fix the remaining issues and run the environment check again.${NC}"
fi 