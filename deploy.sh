#!/bin/bash
# Wrapper script for Enhanced Quantum Trade AI deployment
set -e

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Enhanced Quantum Trade AI - Deployment Tool ===${NC}"

# Check if deployment directory exists
if [ ! -d "deployment" ]; then
  echo -e "${RED}Error: Deployment directory not found!${NC}"
  echo -e "${YELLOW}Make sure you're running this script from the project root.${NC}"
  exit 1
fi

# Check for required files
for file in "deployment/deployment-tracker.sh" "deployment/deploy-to-mainnet.sh" "deployment/security-checklist.md"; do
  if [ ! -f "$file" ]; then
    echo -e "${RED}Error: Required file $file not found!${NC}"
    echo -e "${YELLOW}Please ensure all deployment files are present.${NC}"
    exit 1
  fi
done

# Make the deployment scripts executable
chmod +x deployment/deployment-tracker.sh
chmod +x deployment/deploy-to-mainnet.sh

# Print a helpful message
echo -e "${GREEN}Starting the deployment process...${NC}"
echo -e "${YELLOW}You'll be guided through the checklist to ensure all necessary steps are completed.${NC}"
echo -e "${YELLOW}Critical security items must be completed before deployment can proceed.${NC}"
echo ""

# Run the deployment tracker
./deployment/deployment-tracker.sh

# If the tracker exits with an error, show a message
if [ $? -ne 0 ]; then
  echo -e "${RED}The deployment process encountered an error.${NC}"
  echo -e "${YELLOW}Please check the logs and try again.${NC}"
  exit 1
fi 