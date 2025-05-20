#!/usr/bin/env node

const { initializeBot, monitorOpportunities, stopMonitoring } = require('./QuantumArbitrageBot');
const config = require('./config');
const fs = require('fs');
const path = require('path');
const { program } = require('commander');

// Parse command line arguments
program
  .option('-n, --network <network>', 'Network to monitor (optimism, polygon, arbitrum, bsc)', 'optimism')
  .option('-s, --strategy <strategy>', 'Strategy (cross-dex, sandwich, jit, liquidation, back-running)', 'cross-dex')
  .option('-d, --debug', 'Enable debug mode')
  .option('-t, --test', 'Run in test mode (no real transactions)')
  .option('-o, --opportunitiesOnly', 'Only scan for opportunities, do not execute trades')
  .parse(process.argv);

const options = program.opts();

// Enable test mode if specified
if (options.test) {
  config.botConfig.liveTradingEnabled = false;
  console.log('Running in TEST MODE - No real transactions will be executed');
}

// Set up logging
const logDir = path.join(__dirname, 'logs');
if (!fs.existsSync(logDir)) {
  fs.mkdirSync(logDir);
}

const logStream = fs.createWriteStream(
  path.join(logDir, `bot_${options.network}_${new Date().toISOString().replace(/:/g, '-')}.log`),
  { flags: 'a' }
);

// Redirect console output to log file
const originalConsoleLog = console.log;
const originalConsoleError = console.error;
console.log = function() {
  const args = Array.from(arguments);
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] ${args.join(' ')}`;
  
  logStream.write(logMessage + '\n');
  originalConsoleLog.apply(console, [logMessage]);
};

console.error = function() {
  const args = Array.from(arguments);
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] ERROR: ${args.join(' ')}`;
  
  logStream.write(logMessage + '\n');
  originalConsoleError.apply(console, [logMessage]);
};

// Handle process signals
process.on('SIGINT', async () => {
  console.log('Gracefully shutting down...');
  await stopMonitoring();
  logStream.end();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('Termination signal received, shutting down...');
  await stopMonitoring();
  logStream.end();
  process.exit(0);
});

// Log startup information
console.log(`Starting Quantum AI Arbitrage Bot on ${options.network}`);
console.log(`Strategy: ${options.strategy}`);
console.log(`Live trading: ${config.botConfig.liveTradingEnabled ? 'ENABLED' : 'DISABLED'}`);
console.log(`Minimum profit threshold: $${config.botConfig.minProfitUsd}`);
console.log(`Monitoring interval: ${config.botConfig.monitoringIntervalMs / 1000}s`);

// Initialize the bot and start monitoring
async function main() {
  try {
    await initializeBot(options.network);
    
    console.log('Bot initialized successfully');
    
    // Start monitoring with the specified strategy
    const strategies = options.opportunitiesOnly ? [] : [options.strategy];
    await monitorOpportunities(options.network, strategies);
    
    console.log('Monitoring started successfully');
  } catch (error) {
    console.error(`Error starting bot: ${error.message}`);
    logStream.end();
    process.exit(1);
  }
}

// Start the bot
main(); 