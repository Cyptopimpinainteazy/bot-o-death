#!/usr/bin/env node

const { initializeBot, executeCrossDexArbitrage, executeSandwichAttack, executeJitLiquidity, executeLiquidation, executeBackRunning, executeFlashloanArbitrage } = require('./QuantumArbitrageBot');
const { program } = require('commander');

program
  .requiredOption('-n, --network <network>', 'Network to use (optimism, polygon, arbitrum, bsc)')
  .requiredOption('-s, --strategy <strategy>', 'Strategy: flashloan|cross-dex|sandwich|jit|liquidation|back-running')
  .requiredOption('-p, --params <json>', 'JSON string of strategy parameters')
  .parse(process.argv);

const { network, strategy, params } = program.opts();

(async () => {
  try {
    console.log(`Initializing bot on ${network}...`);
    await initializeBot(network);
    console.log(`Executing strategy: ${strategy}`);
    const paramObj = JSON.parse(params);

    let result;
    switch (strategy) {
      case 'flashloan':
        result = await executeFlashloanArbitrage({ network, ...paramObj });
        break;
      case 'cross-dex':
        result = await executeCrossDexArbitrage({ network, ...paramObj });
        break;
      case 'sandwich':
        result = await executeSandwichAttack({ network, ...paramObj });
        break;
      case 'jit':
        result = await executeJitLiquidity({ network, ...paramObj });
        break;
      case 'liquidation':
        result = await executeLiquidation({ network, ...paramObj });
        break;
      case 'back-running':
        result = await executeBackRunning({ network, ...paramObj });
        break;
      default:
        throw new Error(`Unknown strategy: ${strategy}`);
    }

    console.log('Result:', result);
    process.exit(0);
  } catch (error) {
    console.error('Error executing strategy:', error);
    process.exit(1);
  }
})(); 