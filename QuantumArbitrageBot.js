const ethers = require('ethers');
const config = require('./config');
const fs = require('fs');
const path = require('path');

// Load ABIs
const X3STARABI = require('./artifacts/EnhancedQuantumTrading/contracts/X3STAR.sol/X3STAR.json');
const MevStrategiesABI = require('./artifacts/EnhancedQuantumTrading/contracts/MevStrategies.sol/MevStrategies.json');
const TripleFlashloanABI = require('./artifacts/EnhancedQuantumTrading/contracts/TripleFlashloan.sol/TripleFlashloan.json');

// Global variables
let isMonitoring = false;
let monitoringInterval = null;
let connections = {};
let wsConnections = {}; // WebSocket connections
let contracts = {};
let wallets = {};
let priceData = {
  tokens: {},
  lastUpdated: {}
};

// Counter for websocket reconnection attempts
let wsReconnectAttempts = {};
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_INTERVAL_MS = 2000;

/**
 * Initialize the bot for a specific network
 * @param {string} networkName - Network to initialize (optimism, polygon, arbitrum, bsc)
 * @param {boolean} useWebsocket - Whether to use WebSocket connection (default: true)
 */
async function initializeBot(networkName, useWebsocket = true) {
  try {
    console.log(`Initializing bot for ${networkName}...`);
    
    // Get network configuration
    const networkConfig = config.networks[networkName];
    if (!networkConfig) {
      throw new Error(`Network ${networkName} not configured`);
    }
    
    // Initialize providers and wallet
    let provider;
    
    if (useWebsocket && networkConfig.wsUrl) {
      // Initialize WebSocket provider
      provider = new ethers.providers.WebSocketProvider(networkConfig.wsUrl, {
        chainId: networkConfig.chainId
      });
      
      setupWebSocketListeners(networkName, provider);
      wsConnections[networkName] = provider;
      
      console.log(`WebSocket connection established for ${networkName}`);
    } else {
      // Fallback to HTTP provider
      provider = new ethers.providers.JsonRpcProvider(networkConfig.rpcUrl, {
        chainId: networkConfig.chainId
      });
      console.log(`HTTP provider initialized for ${networkName}`);
    }
    
    const wallet = new ethers.Wallet(process.env.PRIVATE_KEY, provider);
    
    // Initialize contracts
    contracts[networkName] = {
      tripleFlashloan: new ethers.Contract(
        networkConfig.contracts.tripleFlashloan,
        TripleFlashloanABI.abi,
        wallet
      ),
      mevStrategies: new ethers.Contract(
        networkConfig.contracts.mevStrategies,
        MevStrategiesABI.abi,
        wallet
      ),
      x3star: new ethers.Contract(
        networkConfig.contracts.x3star,
        X3STARABI.abi,
        wallet
      )
    };
    
    // Store connection and wallet
    connections[networkName] = provider;
    wallets[networkName] = wallet;
    
    // Verify wallet balance
    const balance = await provider.getBalance(wallet.address);
    console.log(`Wallet ${wallet.address} initialized with ${ethers.utils.formatEther(balance)} ETH`);
    
    return { 
      provider,
      wallet,
      contracts: contracts[networkName]
    };
  } catch (error) {
    console.error(`Failed to initialize bot for ${networkName}: ${error.message}`);
    throw error;
  }
}

/**
 * Setup event listeners for WebSocket provider
 * @param {string} networkName - Network name
 * @param {WebSocketProvider} provider - WebSocket provider
 */
function setupWebSocketListeners(networkName, provider) {
  // Reset reconnect attempts counter for this network
  wsReconnectAttempts[networkName] = 0;
  
  // Listen for WebSocket disconnection
  provider._websocket.on('close', async (code) => {
    console.warn(`WebSocket connection closed for ${networkName} with code ${code}`);
    
    // Clean up old connection
    if (wsConnections[networkName]) {
      try {
        wsConnections[networkName].removeAllListeners();
        delete wsConnections[networkName];
      } catch (err) {
        console.error(`Error cleaning up WebSocket for ${networkName}: ${err.message}`);
      }
    }
    
    // Attempt to reconnect if we haven't reached max attempts
    if (wsReconnectAttempts[networkName] < MAX_RECONNECT_ATTEMPTS) {
      wsReconnectAttempts[networkName]++;
      
      console.log(`Attempting to reconnect WebSocket for ${networkName} (attempt ${wsReconnectAttempts[networkName]}/${MAX_RECONNECT_ATTEMPTS})`);
      
      // Wait before reconnecting
      await new Promise(resolve => setTimeout(resolve, RECONNECT_INTERVAL_MS));
      
      try {
        // Re-initialize with WebSocket
        await initializeBot(networkName, true);
        console.log(`Successfully reconnected WebSocket for ${networkName}`);
      } catch (error) {
        console.error(`Failed to reconnect WebSocket for ${networkName}: ${error.message}`);
        
        // If we still have more attempts, we'll try again on the next close event
        if (wsReconnectAttempts[networkName] >= MAX_RECONNECT_ATTEMPTS) {
          console.warn(`Max reconnect attempts reached for ${networkName}, falling back to HTTP provider`);
          
          // Fall back to HTTP provider
          try {
            await initializeBot(networkName, false);
            console.log(`Successfully fell back to HTTP provider for ${networkName}`);
          } catch (fallbackError) {
            console.error(`Failed to initialize fallback HTTP provider for ${networkName}: ${fallbackError.message}`);
          }
        }
      }
    }
  });
  
  // Handle WebSocket errors
  provider._websocket.on('error', (error) => {
    console.error(`WebSocket error for ${networkName}: ${error.message}`);
  });
  
  // Set up subscription for new blocks
  provider.on('block', (blockNumber) => {
    console.log(`New block on ${networkName}: ${blockNumber}`);
    // You could trigger opportunity scanning here instead of using setInterval
  });
}

/**
 * Monitor for arbitrage opportunities on the specified network
 * @param {string} networkName - Network to monitor
 * @param {string[]} strategies - Strategies to execute
 * @param {boolean} useWebsocket - Whether to use WebSocket connection (default: true)
 */
async function monitorOpportunities(networkName, strategies = ['cross-dex'], useWebsocket = true) {
  if (isMonitoring) {
    console.log('Bot is already monitoring opportunities');
    return;
  }
  
  isMonitoring = true;
  console.log(`Starting to monitor ${networkName} for opportunities (strategies: ${strategies.join(', ')})`);
  
  // Ensure bot is initialized with websocket if specified
  if (!connections[networkName]) {
    await initializeBot(networkName, useWebsocket);
  }
  
  // If using websocket, leverage block notifications for scanning
  if (useWebsocket && wsConnections[networkName]) {
    // Remove existing handler if any
    wsConnections[networkName].removeAllListeners('block');
    
    // Setup new block handler for opportunity scanning
    wsConnections[networkName].on('block', async (blockNumber) => {
      try {
        console.log(`Scanning for opportunities on ${networkName} (block ${blockNumber})...`);
        
        // Scan for opportunities
        const opportunities = await scanForOpportunities(networkName, strategies);
        
        if (opportunities.length > 0) {
          console.log(`Found ${opportunities.length} opportunities on block ${blockNumber}`);
          
          // Execute opportunities if live trading is enabled
          if (config.botConfig.liveTradingEnabled) {
            for (const opportunity of opportunities) {
              try {
                await executeArbitrage(opportunity);
              } catch (error) {
                console.error(`Failed to execute opportunity: ${error.message}`);
              }
            }
          } else {
            console.log('Live trading disabled, skipping execution');
          }
        }
      } catch (error) {
        console.error(`Error monitoring opportunities on block ${blockNumber}: ${error.message}`);
      }
    });
    
    console.log(`WebSocket block monitoring enabled for ${networkName}`);
  } else {
    // Fall back to interval-based monitoring
    monitoringInterval = setInterval(async () => {
      try {
        // Scan for opportunities
        const opportunities = await scanForOpportunities(networkName, strategies);
        
        if (opportunities.length > 0) {
          console.log(`Found ${opportunities.length} opportunities`);
          
          // Execute opportunities if live trading is enabled
          if (config.botConfig.liveTradingEnabled) {
            for (const opportunity of opportunities) {
              try {
                await executeArbitrage(opportunity);
              } catch (error) {
                console.error(`Failed to execute opportunity: ${error.message}`);
              }
            }
          } else {
            console.log('Live trading disabled, skipping execution');
          }
        } else {
          console.log('No profitable opportunities found');
        }
      } catch (error) {
        console.error(`Error monitoring opportunities: ${error.message}`);
      }
    }, config.botConfig.monitoringIntervalMs);
    
    console.log(`Interval-based monitoring enabled for ${networkName} (${config.botConfig.monitoringIntervalMs}ms)`);
  }
}

/**
 * Stop monitoring for opportunities
 */
async function stopMonitoring() {
  if (!isMonitoring) {
    console.log('Bot is not monitoring');
    return;
  }
  
  // Clear interval if it exists
  if (monitoringInterval) {
    clearInterval(monitoringInterval);
    monitoringInterval = null;
  }
  
  // Remove WebSocket event listeners
  for (const networkName in wsConnections) {
    try {
      wsConnections[networkName].removeAllListeners('block');
      console.log(`Stopped WebSocket monitoring for ${networkName}`);
    } catch (error) {
      console.error(`Error removing WebSocket listeners for ${networkName}: ${error.message}`);
    }
  }
  
  isMonitoring = false;
  console.log('Stopped monitoring for opportunities');
}

// Get the network provider, preferring WebSocket if available
function getNetworkProvider(networkName) {
  // Prefer WebSocket connection if available
  if (wsConnections[networkName]) {
    return wsConnections[networkName];
  }
  // Fall back to HTTP provider
  return connections[networkName];
}

/**
 * Scan for arbitrage opportunities
 * @param {string} networkName - Network to scan
 * @param {string[]} strategies - Strategies to scan for
 * @returns {Array} Array of opportunity objects
 */
async function scanForOpportunities(networkName, strategies) {
  const opportunities = [];
  
  for (const strategy of strategies) {
    try {
      switch (strategy) {
        case 'cross-dex':
          const crossDexOpps = await scanCrossDexOpportunities(networkName);
          opportunities.push(...crossDexOpps);
          break;
        case 'sandwich':
          const sandwichOpps = await scanSandwichOpportunities(networkName);
          opportunities.push(...sandwichOpps);
          break;
        case 'jit':
          const jitOpps = await scanJITOpportunities(networkName);
          opportunities.push(...jitOpps);
          break;
        case 'liquidation':
          const liquidationOpps = await scanLiquidationOpportunities(networkName);
          opportunities.push(...liquidationOpps);
          break;
        case 'back-running':
          const backRunningOpps = await scanBackRunningOpportunities(networkName);
          opportunities.push(...backRunningOpps);
          break;
        default:
          console.log(`Strategy ${strategy} not implemented for scanning`);
      }
    } catch (error) {
      console.error(`Error scanning for ${strategy} opportunities: ${error.message}`);
    }
  }
  
  // Filter for minimum profit
  return opportunities.filter(opportunity => 
    opportunity.expectedProfitUsd >= config.botConfig.minProfitUsd
  );
}

/**
 * Execute arbitrage opportunity
 * @param {Object} opportunity - Opportunity to execute
 */
async function executeArbitrage(opportunity) {
  console.log(`Executing ${opportunity.type} arbitrage on ${opportunity.network}...`);
  
  try {
    switch (opportunity.type) {
      case 'cross-dex':
        await executeCrossDexArbitrage(opportunity);
        break;
      case 'sandwich':
        await executeSandwichAttack(opportunity);
        break;
      case 'jit':
        await executeJitLiquidity(opportunity);
        break;
      case 'liquidation':
        await executeLiquidation(opportunity);
        break;
      case 'back-running':
        await executeBackRunning(opportunity);
        break;
      case 'flashloan':
        await executeFlashloanArbitrage(opportunity);
        break;
      default:
        console.log(`Strategy ${opportunity.type} not implemented for execution`);
        return;
    }
  } catch (error) {
    console.error(`Error executing arbitrage: ${error.message}`);
    throw error;
  }
}

/**
 * Execute cross-DEX arbitrage
 * @param {Object} opportunity - Opportunity to execute
 */
async function executeCrossDexArbitrage(opportunity) {
  const connection = getNetworkConnection(opportunity.network);
  const contract = getContracts(opportunity.network).mevStrategies;
  const wallet = getWallet(opportunity.network);
  
  // Prepare transaction parameters
  const tokenIn = opportunity.tokenIn;
  const tokenOut = opportunity.tokenOut;
  const sourceRouter = opportunity.sourceRouter;
  const targetRouter = opportunity.targetRouter;
  const amountIn = ethers.utils.parseUnits(
    opportunity.amountIn.toString(), 
    config.tokenDecimals[opportunity.tokenInSymbol] || 18
  );
  const minProfit = ethers.utils.parseUnits(
    (opportunity.expectedProfitUsd * 0.8).toString(), // 80% of expected profit as minimum
    config.tokenDecimals[opportunity.tokenInSymbol] || 18
  );
  
  // Get current gas price and optimize it
  const gasPrice = await connection.getGasPrice();
  const optimalGasPrice = gasPrice.mul(150).div(100); // 1.5x current gas price
  
  // Execute transaction
  console.log(`Executing cross-DEX arbitrage: ${opportunity.tokenInSymbol} -> ${opportunity.tokenOutSymbol}`);
  
  const tx = await contract.executeCrossDexArbitrage(
    tokenIn,
    tokenOut,
    sourceRouter,
    targetRouter,
    amountIn,
    minProfit,
    { 
      gasPrice: optimalGasPrice,
      gasLimit: 1000000 
    }
  );
  
  console.log(`Transaction sent: ${tx.hash}`);
  
  // Wait for transaction to be mined
  const receipt = await tx.wait();
  
  console.log(`Transaction confirmed! Gas used: ${receipt.gasUsed.toString()}`);
  
  return receipt;
}

/**
 * Execute sandwich attack
 * @param {Object} opportunity - Opportunity to execute
 */
async function executeSandwichAttack(opportunity) {
  const connection = getNetworkConnection(opportunity.network);
  const contract = getContracts(opportunity.network).x3star;
  const wallet = getWallet(opportunity.network);
  
  // Prepare transaction params
  const router = opportunity.targetRouter;
  const amountIn = ethers.utils.parseUnits(
    opportunity.amountIn.toString(), 18
  );
  const amountOutMin = opportunity.amountOutMin || 0;
  const deadline = opportunity.deadline || Math.floor(Date.now() / 1000) + 300;
  
  // Gas optimization: 2x current gas price for frontrunning
  const gasPrice = await connection.getGasPrice();
  const optimalGasPrice = gasPrice.mul(200).div(100);
  
  console.log(`Executing sandwich attack: ${opportunity.network}`);
  const tx = await contract.executeSandwichTrade(
    router,
    amountIn,
    amountOutMin,
    deadline,
    {
      value: amountIn,
      gasPrice: optimalGasPrice,
      gasLimit: 500000
    }
  );
  console.log(`Sandwich tx sent: ${tx.hash}`);
  const receipt = await tx.wait();
  console.log(`Sandwich confirmed, gas used: ${receipt.gasUsed}`);
  return receipt;
}

/**
 * Execute JIT liquidity provision
 * @param {Object} opportunity
 */
async function executeJitLiquidity(opportunity) {
  const connection = getNetworkConnection(opportunity.network);
  const contract = getContracts(opportunity.network).mevStrategies;
  const wallet = getWallet(opportunity.network);

  const liquidityPair = opportunity.liquidityPair;
  const tokenA = opportunity.tokenA;
  const tokenB = opportunity.tokenB;
  const amountA = ethers.utils.parseUnits(
    opportunity.amountA.toString(),
    opportunity.tokenADecimals || 18
  );
  const amountB = ethers.utils.parseUnits(
    opportunity.amountB.toString(),
    opportunity.tokenBDecimals || 18
  );
  const targetTx = opportunity.targetTx;
  const minProfit = ethers.utils.parseUnits(
    opportunity.expectedProfitUsd.toString(),
    18
  );

  const gasPrice = await connection.getGasPrice();
  const optimalGasPrice = gasPrice.mul(180).div(100);
  
  console.log(`Executing JIT liquidity on ${liquidityPair}`);
  const tx = await contract.executeJitLiquidity(
    liquidityPair,
    tokenA,
    tokenB,
    amountA,
    amountB,
    targetTx,
    minProfit,
    {
      gasPrice: optimalGasPrice,
      gasLimit: 800000
    }
  );
  console.log(`JIT tx sent: ${tx.hash}`);
  const receipt = await tx.wait();
  console.log(`JIT confirmed, gas used: ${receipt.gasUsed}`);
  return receipt;
}

/**
 * Execute liquidation arbitrage
 * @param {Object} opportunity
 */
async function executeLiquidation(opportunity) {
  const connection = getNetworkConnection(opportunity.network);
  const contract = getContracts(opportunity.network).mevStrategies;
  const wallet = getWallet(opportunity.network);

  const borrower = opportunity.borrower;
  const collateralAsset = opportunity.collateralAsset;
  const debtAsset = opportunity.debtAsset;
  const debtToCover = ethers.utils.parseUnits(
    opportunity.debtToCover.toString(),
    opportunity.debtAssetDecimals || 18
  );
  const minProfit = ethers.utils.parseUnits(
    opportunity.expectedProfitUsd.toString(),
    18
  );

  const gasPrice = await connection.getGasPrice();
  const optimalGasPrice = gasPrice.mul(150).div(100);
  
  console.log(`Executing liquidation for ${borrower}`);
  const tx = await contract.executeLiquidation(
    borrower,
    collateralAsset,
    debtAsset,
    debtToCover,
    minProfit,
    {
      gasPrice: optimalGasPrice,
      gasLimit: 1200000
    }
  );
  console.log(`Liquidation tx sent: ${tx.hash}`);
  const receipt = await tx.wait();
  console.log(`Liquidation confirmed, gas used: ${receipt.gasUsed}`);
  return receipt;
}

/**
 * Execute back-running
 * @param {Object} opportunity
 */
async function executeBackRunning(opportunity) {
  const connection = getNetworkConnection(opportunity.network);
  const contract = getContracts(opportunity.network).mevStrategies;
  const wallet = getWallet(opportunity.network);

  const targetTx = opportunity.targetTx;
  const tokenIn = opportunity.tokenIn;
  const tokenOut = opportunity.tokenOut;
  const router = opportunity.router;
  const amountIn = ethers.utils.parseUnits(
    opportunity.amountIn.toString(),
    opportunity.tokenInDecimals || 18
  );
  const minProfit = ethers.utils.parseUnits(
    opportunity.expectedProfitUsd.toString(),
    opportunity.tokenInDecimals || 18
  );

  const gasPrice = await connection.getGasPrice();
  const optimalGasPrice = gasPrice.mul(120).div(100);
  
  console.log(`Executing back-running on ${opportunity.network}`);
  const tx = await contract.executeBackRunning(
    targetTx,
    tokenIn,
    tokenOut,
    router,
    amountIn,
    minProfit,
    {
      gasPrice: optimalGasPrice,
      gasLimit: 800000
    }
  );
  console.log(`Back-running tx sent: ${tx.hash}`);
  const receipt = await tx.wait();
  console.log(`Back-running confirmed, gas used: ${receipt.gasUsed}`);
  return receipt;
}

/**
 * Execute flashloan arbitrage
 * @param {Object} opportunity
 */
async function executeFlashloanArbitrage(opportunity) {
  const connection = getNetworkConnection(opportunity.network);
  const contract = getContracts(opportunity.network).tripleFlashloan;
  const wallet = getWallet(opportunity.network);

  const aaveAssets = [opportunity.tokenIn];
  const aaveAmounts = [ethers.utils.parseUnits(
    opportunity.aaveFlashloanAmount.toString(),
    opportunity.tokenInDecimals || 18
  )];

  const balancerAssets = opportunity.balancerAssets || [];
  const balancerAmounts = (opportunity.balancerAmounts || []).map((amt, i) =>
    ethers.utils.parseUnits(
      amt.toString(),
      opportunity.balancerDecimals?.[i] || 18
  ));

  const curveAssets = opportunity.curveAssets || [];
  const curveAmounts = (opportunity.curveAmounts || []).map((amt, i) =>
    ethers.utils.parseUnits(
      amt.toString(),
      opportunity.curveDecimals?.[i] || 18
  ));

  const dodoBaseAmt = opportunity.dodoBaseAmount
    ? ethers.utils.parseUnits(
      opportunity.dodoBaseAmount.toString(),
      opportunity.dodoBaseDecimals || 18
    ) : 0;
  const dodoQuoteAmt = opportunity.dodoQuoteAmount
    ? ethers.utils.parseUnits(
      opportunity.dodoQuoteAmount.toString(),
      opportunity.dodoQuoteDecimals || 18
    ) : 0;

  const gasPrice = await connection.getGasPrice();
  const optimalGasPrice = gasPrice.mul(150).div(100);

  console.log(`Executing flashloan arbitrage on ${opportunity.network}`);
  const tx = await contract.executeTripleFlashloan(
    aaveAssets,
    aaveAmounts,
    balancerAssets,
    balancerAmounts,
    curveAssets,
    curveAmounts,
    dodoBaseAmt,
    dodoQuoteAmt,
    {
      gasPrice: optimalGasPrice,
      gasLimit: 2000000
    }
  );
  console.log(`Flashloan tx sent: ${tx.hash}`);
  const receipt = await tx.wait();
  console.log(`Flashloan confirmed, gas used: ${receipt.gasUsed}`);
  return receipt;
}

/**
 * Helper functions to get network connection, contracts, and wallet
 */
function getNetworkConnection(networkName) {
  // Prefer WebSocket connection if available
  if (wsConnections[networkName]) {
    return wsConnections[networkName];
  }
  
  if (!connections[networkName]) {
    throw new Error(`Network ${networkName} not initialized`);
  }
  return connections[networkName];
}

function getContracts(networkName) {
  if (!contracts[networkName]) {
    throw new Error(`Contracts for ${networkName} not initialized`);
  }
  return contracts[networkName];
}

function getWallet(networkName) {
  if (!wallets[networkName]) {
    throw new Error(`Wallet for ${networkName} not initialized`);
  }
  return wallets[networkName];
}

/**
 * Scan functions (stubs for demo purposes)
 */
async function scanCrossDexOpportunities(networkName) {
  // Implementation would involve checking price differences across DEXes
  return [];
}

async function scanSandwichOpportunities(networkName) {
  // Implementation would involve monitoring mempool for large swaps
  return [];
}

async function scanJITOpportunities(networkName) {
  // Implementation would involve monitoring for impending large trades
  return [];
}

async function scanLiquidationOpportunities(networkName) {
  // Implementation would involve checking lending pools for near-liquidation positions
  return [];
}

async function scanBackRunningOpportunities(networkName) {
  // Implementation would involve monitoring mempool for opportunities to back-run
  return [];
}

/**
 * Make an RPC call with retry logic
 * @param {function} callFn - Function that makes the RPC call
 * @param {number} maxRetries - Maximum number of retries
 * @param {number} backoffMs - Initial backoff in milliseconds
 * @returns {Promise} Promise that resolves with the RPC call result
 */
async function makeRpcCall(callFn, maxRetries = 3, backoffMs = 1000) {
  let lastError;
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await callFn();
    } catch (error) {
      lastError = error;
      console.warn(`RPC call failed, retry ${i+1}/${maxRetries}: ${error.message}`);
      
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, backoffMs * Math.pow(2, i)));
    }
  }
  
  throw lastError;
}

module.exports = {
  initializeBot,
  monitorOpportunities,
  stopMonitoring,
  executeArbitrage,
  executeCrossDexArbitrage,
  executeSandwichAttack,
  executeJitLiquidity,
  executeLiquidation,
  executeBackRunning,
  executeFlashloanArbitrage
}; 