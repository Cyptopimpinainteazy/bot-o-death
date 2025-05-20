#!/usr/bin/env node

const { initializeBot, monitorOpportunities, stopMonitoring } = require('./QuantumArbitrageBot');
const config = require('./config');
const fs = require('fs');
const path = require('path');
const { program } = require('commander');

// File to store opportunities
const DEFAULT_OUTPUT_FILE = 'opportunities.json';

// Parse command line arguments
program
  .option('-n, --network <network>', 'Network to monitor (optimism, polygon, arbitrum, bsc)', 'optimism')
  .option('-s, --strategy <strategy>', 'Strategy to scan for (cross-dex, sandwich, jit, liquidation, back-running)', 'cross-dex')
  .option('-t, --time <minutes>', 'How long to scan in minutes (0 = indefinite)', '60')
  .option('-o, --output <file>', 'Output file for detected opportunities', DEFAULT_OUTPUT_FILE)
  .option('-w, --websocket', 'Use WebSocket connection for real-time updates', true)
  .option('-m, --min-profit <usd>', 'Minimum profit threshold in USD', '10')
  .option('-v, --verbose', 'Show detailed logs')
  .parse(process.argv);

const options = program.opts();

// Setup logging
const logLevel = options.verbose ? 'debug' : 'info';
console.log(`Log level set to ${logLevel}`);

// Validate network
if (!config.networks[options.network]) {
  console.error(`Error: Network "${options.network}" not found in configuration`);
  process.exit(1);
}

// Convert scanning time to milliseconds
const scanDuration = parseInt(options.time, 10);
const scanTimeMs = scanDuration > 0 ? scanDuration * 60 * 1000 : 0;

// Set minimum profit threshold
config.botConfig.minProfitUsd = parseFloat(options.minProfit);

// Store opportunities
let opportunities = [];

// Function to save opportunities to file
function saveOpportunities() {
  const outputPath = path.resolve(options.output);
  try {
    fs.writeFileSync(outputPath, JSON.stringify(opportunities, null, 2));
    console.log(`Saved ${opportunities.length} opportunities to ${outputPath}`);
  } catch (error) {
    console.error(`Error saving opportunities: ${error.message}`);
  }
}

// Handle graceful shutdown
process.on('SIGINT', async () => {
  console.log('Stopping opportunity scanning...');
  await stopMonitoring();
  saveOpportunities();
  console.log('Scan complete. Exiting...');
  process.exit(0);
});

// Main function to start scanning
async function startScanning() {
  console.log(`Starting Quantum AI Arbitrage Bot in SCAN ONLY mode`);
  console.log(`Network: ${options.network}`);
  console.log(`Strategy: ${options.strategy}`);
  console.log(`Scanning duration: ${scanDuration > 0 ? `${scanDuration} minutes` : 'indefinite'}`);
  console.log(`Minimum profit threshold: $${config.botConfig.minProfitUsd}`);
  console.log(`Using ${options.websocket ? 'WebSocket' : 'HTTP'} connection`);
  
  try {
    // Initialize the bot with websocket if specified
    const bot = await initializeBot(options.network, options.websocket);
    console.log(`Bot initialized on ${options.network}`);
    
    // Override the executeArbitrage function to just log and store opportunities
    const originalExecuteArbitrage = require('./QuantumArbitrageBot').executeArbitrage;
    require('./QuantumArbitrageBot').executeArbitrage = async (opportunity) => {
      console.log(`[SCAN ONLY] Found opportunity: ${opportunity.type} on ${opportunity.network} with expected profit: $${opportunity.expectedProfitUsd}`);
      
      // Add timestamp to opportunity
      opportunity.timestamp = new Date().toISOString();
      
      // Store the opportunity
      opportunities.push(opportunity);
      
      // Save periodically (every 10 opportunities)
      if (opportunities.length % 10 === 0) {
        saveOpportunities();
      }
      
      // Return simulated success result
      return {
        status: 'simulated',
        opportunity
      };
    };
    
    // Start monitoring for opportunities - but don't execute
    config.botConfig.liveTradingEnabled = false;
    await monitorOpportunities(options.network, [options.strategy], options.websocket);
    console.log(`Opportunity monitoring started`);
    
    // Set up timer to stop scanning if specified
    if (scanTimeMs > 0) {
      console.log(`Will scan for ${scanDuration} minutes`);
      setTimeout(async () => {
        console.log(`Scan time (${scanDuration} minutes) reached. Stopping...`);
        await stopMonitoring();
        saveOpportunities();
        console.log('Scan complete. Exiting...');
        process.exit(0);
      }, scanTimeMs);
    }
  } catch (error) {
    console.error(`Error starting scan: ${error.message}`);
    process.exit(1);
  }
}

// Start the scanning process
startScanning().catch((error) => {
  console.error(`Fatal error: ${error.message}`);
  process.exit(1);
}); 