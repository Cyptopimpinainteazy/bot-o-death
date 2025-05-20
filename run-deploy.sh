#!/bin/bash
# One-step deployment script - checks environment and starts deployment
set -e

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Enhanced Quantum Trade AI - One-Step Deployment ===${NC}"
echo -e "${YELLOW}This script will check your environment and start the deployment process.${NC}"
echo

# Make scripts executable
chmod +x deployment/check-environment.sh
chmod +x deploy.sh

# Step 1: Check the environment
echo -e "${BLUE}Step 1: Checking environment...${NC}"
if ! ./deployment/check-environment.sh; then
  echo -e "${RED}Environment check failed. Please fix the issues before continuing.${NC}"
  exit 1
fi

# Ask for confirmation
echo
echo -e "${YELLOW}The environment check passed. Do you want to continue with deployment? (y/n)${NC}"
read -p "Continue? " confirm

if [[ $confirm != "y" && $confirm != "Y" ]]; then
  echo -e "${YELLOW}Deployment cancelled.${NC}"
  exit 0
fi

# Step 2: Start the deployment process
echo -e "${BLUE}Step 2: Starting deployment process...${NC}"
if ! ./deploy.sh; then
  echo -e "${RED}Deployment failed. Please check the logs for details.${NC}"
  exit 1
fi

echo -e "${GREEN}Deployment process finished successfully!${NC}"
exit 0 