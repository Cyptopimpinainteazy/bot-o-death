#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const dotenv = require('dotenv');
const ethers = require('ethers');
const Web3 = require('web3');

// Load environment variables
dotenv.config();

console.log('Quantum AI Arbitrage Bot - Setup & Verification');
console.log('=============================================');

// Check for critical dependencies
const dependencies = [
  { name: 'ethers.js', check: () => typeof ethers !== 'undefined' },
  { name: 'web3.js', check: () => typeof Web3 !== 'undefined' },
  { name: 'dotenv', check: () => typeof dotenv !== 'undefined' },
  { name: 'TensorFlow', check: () => {
    try {
      require('@tensorflow/tfjs-node');
      return true;
    } catch (e) {
      return false;
    }
  }}
];

console.log('\nChecking dependencies...');
let missingDeps = false;
dependencies.forEach(dep => {
  const installed = dep.check();
  console.log(`${dep.name}: ${installed ? '✓' : '✗'}`);
  if (!installed) missingDeps = true;
});

if (missingDeps) {
  console.log('\nSome dependencies are missing. Installing...');
  try {
    execSync('npm install', { stdio: 'inherit' });
    console.log('Dependencies installed successfully.');
  } catch (error) {
    console.error('Failed to install dependencies:', error.message);
    process.exit(1);
  }
}

// Check for .env file
console.log('\nChecking environment configuration...');
if (!fs.existsSync(path.join(__dirname, '.env'))) {
  console.error('Error: .env file not found!');
  console.log('Please create a .env file with required configuration.');
  console.log('Example:');
  console.log(`
PRIVATE_KEY=your_private_key
WALLET_ADDRESS=your_wallet_address
OPTIMISM_RPC_URL=https://mainnet.optimism.io
POLYGON_RPC_URL=https://polygon-rpc.com
ARBITRUM_RPC_URL=https://arb1.arbitrum.io/rpc
BSC_RPC_URL=https://bsc-dataseed.binance.org
OPTIMISM_WSS_URL=wss://ws-mainnet.optimism.io
POLYGON_WSS_URL=wss://ws-polygon-mainnet.chainstacklabs.com
ARBITRUM_WSS_URL=wss://arb1.arbitrum.io/ws
BSC_WSS_URL=wss://bsc-ws-node.nariox.org
OPTIMISM_API_KEY=your_optimism_api_key
POLYGONSCAN_API_KEY=your_polygonscan_api_key
ARBISCAN_API_KEY=your_arbiscan_api_key
BSCSCAN_API_KEY=your_bscscan_api_key
MAX_GAS_PRICE=100
PRIORITY_FEE=2
API_KEY=your_api_key
API_SECRET=your_api_secret
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASS=password
DB_NAME=quantum_arbitrage`);
  process.exit(1);
}

// Validate critical environment variables
const requiredEnvVars = [
  'PRIVATE_KEY',
  'WALLET_ADDRESS',
  'OPTIMISM_RPC_URL',
  'POLYGON_RPC_URL'
];

const missingEnvVars = requiredEnvVars.filter(varName => !process.env[varName]);
if (missingEnvVars.length > 0) {
  console.error(`Error: Missing required environment variables: ${missingEnvVars.join(', ')}`);
  process.exit(1);
}

// Validate private key
try {
  const wallet = new ethers.Wallet(process.env.PRIVATE_KEY);
  console.log(`Wallet address from private key: ${wallet.address}`);
  
  if (wallet.address.toLowerCase() !== process.env.WALLET_ADDRESS.toLowerCase()) {
    console.warn('Warning: WALLET_ADDRESS in .env does not match the address derived from PRIVATE_KEY');
  } else {
    console.log('Wallet address verified ✓');
  }
} catch (error) {
  console.error('Error: Invalid private key:', error.message);
  process.exit(1);
}

// Test RPC connections
console.log('\nTesting network connections...');
const networks = [
  { name: 'Optimism', rpcUrl: process.env.OPTIMISM_RPC_URL, chainId: 10 },
  { name: 'Polygon', rpcUrl: process.env.POLYGON_RPC_URL, chainId: 137 },
  { name: 'Arbitrum', rpcUrl: process.env.ARBITRUM_RPC_URL, chainId: 42161 },
  { name: 'BSC', rpcUrl: process.env.BSC_RPC_URL, chainId: 56 }
];

async function testRpcConnection(network) {
  try {
    const provider = new ethers.providers.JsonRpcProvider(network.rpcUrl, { chainId: network.chainId });
    const blockNumber = await provider.getBlockNumber();
    console.log(`${network.name}: Connected ✓ (Block #${blockNumber})`);
    return true;
  } catch (error) {
    console.error(`${network.name}: Failed to connect ✗ (${error.message})`);
    return false;
  }
}

// Create directories if they don't exist
console.log('\nChecking directory structure...');
const requiredDirs = [
  'logs',
  'ai/models',
  'ai/models/price_prediction_model',
  'contracts/artifacts',
  'contracts/dexes',
  'contracts/interfaces'
];

requiredDirs.forEach(dir => {
  const dirPath = path.join(__dirname, dir);
  if (!fs.existsSync(dirPath)) {
    console.log(`Creating directory: ${dir}`);
    fs.mkdirSync(dirPath, { recursive: true });
  }
});

// Check for AI model
const modelPath = path.join(__dirname, 'ai/models/price_prediction_model/model.json');
if (!fs.existsSync(modelPath)) {
  console.warn('Warning: AI model not found at ai/models/price_prediction_model/model.json');
  console.log('Bot will use fallback prediction model.');
} else {
  console.log('AI model found ✓');
}

// Check for contract ABIs
const requiredABIs = [
  'contracts/artifacts/X3STAR.json',
  'contracts/artifacts/MevStrategies.json',
  'contracts/artifacts/TripleFlashloan.json'
];

const missingABIs = requiredABIs.filter(abiPath => !fs.existsSync(path.join(__dirname, abiPath)));
if (missingABIs.length > 0) {
  console.error(`Error: Missing contract ABIs: ${missingABIs.join(', ')}`);
  process.exit(1);
} else {
  console.log('Contract ABIs verified ✓');
}

// Check for DEX router ABIs
const requiredDEXs = [
  'contracts/dexes/uniswaprouter.json',
  'contracts/dexes/sushiswaprouter.json',
  'contracts/dexes/quickswaprouter.json',
  'contracts/dexes/pancakeswaprouter.json',
  'contracts/dexes/velodromerouter.json'
];

const missingDEXs = requiredDEXs.filter(dexPath => !fs.existsSync(path.join(__dirname, dexPath)));
if (missingDEXs.length > 0) {
  console.warn(`Warning: Missing DEX router ABIs: ${missingDEXs.join(', ')}`);
  console.log('Some DEXes may not be available for arbitrage.');
} else {
  console.log('DEX router ABIs verified ✓');
}

// Run network tests
async function runNetworkTests() {
  console.log('\nRunning network connection tests...');
  const results = await Promise.all(networks.map(network => testRpcConnection(network)));
  const allConnected = results.every(result => result);
  
  if (!allConnected) {
    console.warn('Warning: Some networks could not be connected. The bot will skip these networks during operation.');
  }
  
  // Final setup message
  console.log('\nSetup complete!');
  console.log('To start the bot, run: node run.js');
  console.log('To start the server and dashboard, run: node server.js');
  console.log('\nFor test mode (no real transactions): node run.js --test');
  console.log('To scan for opportunities only: node run.js --opportunitiesOnly');
  console.log('To specify a network: node run.js --network=polygon');
}

runNetworkTests(); 