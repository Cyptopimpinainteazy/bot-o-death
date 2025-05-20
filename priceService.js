const axios = require('axios');
const ethers = require('ethers');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// Token price cache
const priceCache = {
  tokens: {},
  lastUpdated: {},
  cacheTTLMs: 60000 // 1 minute default cache time-to-live
};

// ERC20 ABI for token interactions
const ERC20_ABI = [
  'function name() view returns (string)',
  'function symbol() view returns (string)',
  'function decimals() view returns (uint8)',
  'function balanceOf(address) view returns (uint)'
];

// Price pair ABI (like Uniswap pairs)
const PAIR_ABI = [
  'function token0() external view returns (address)',
  'function token1() external view returns (address)',
  'function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast)'
];

/**
 * Get token price from multiple sources with caching
 * @param {string} tokenAddress - Token address
 * @param {string} network - Network name
 * @param {object} options - Options for price fetching
 * @returns {Promise<number>} Token price in USD
 */
async function getTokenPrice(tokenAddress, network, options = {}) {
  const {
    forceRefresh = false,
    cacheTTLMs = priceCache.cacheTTLMs,
    decimals = null,
    providers = null
  } = options;

  // Normalize token address
  const normalizedAddress = tokenAddress.toLowerCase();
  
  // Generate cache key
  const cacheKey = `${network}:${normalizedAddress}`;

  // Check cache if not forcing refresh
  if (!forceRefresh && 
      priceCache.tokens[cacheKey] && 
      (Date.now() - priceCache.lastUpdated[cacheKey]) < cacheTTLMs) {
    return priceCache.tokens[cacheKey];
  }

  // Initialize price sources based on the network
  const priceSourcesForNetwork = getPriceSourcesForNetwork(network, providers);
  
  // Try each price source in sequence
  let price = null;
  let errors = [];

  for (const source of priceSourcesForNetwork) {
    try {
      price = await source.getPrice(normalizedAddress, decimals);
      if (price !== null && price > 0) {
        break;
      }
    } catch (error) {
      errors.push(`${source.name}: ${error.message}`);
    }
  }

  if (price === null) {
    console.error(`Failed to get price for ${normalizedAddress} on ${network}. Errors: ${errors.join(', ')}`);
    
    // If we have a cached price, return that with a warning
    if (priceCache.tokens[cacheKey]) {
      console.warn(`Using cached price for ${normalizedAddress} on ${network}`);
      return priceCache.tokens[cacheKey];
    }
    
    throw new Error(`Could not get price from any source: ${errors.join(', ')}`);
  }

  // Update cache
  priceCache.tokens[cacheKey] = price;
  priceCache.lastUpdated[cacheKey] = Date.now();

  return price;
}

/**
 * Get multiple token prices in a batch to reduce API calls
 * @param {string[]} tokenAddresses - Array of token addresses
 * @param {string} network - Network name
 * @param {object} options - Options for price fetching
 * @returns {Promise<Object>} Object mapping token addresses to prices
 */
async function getBatchTokenPrices(tokenAddresses, network, options = {}) {
  const results = {};
  const fetchPromises = [];
  const uniqueAddresses = [...new Set(tokenAddresses.map(addr => addr.toLowerCase()))];

  // Try to use CoinGecko or other API for batch fetching if possible
  if (uniqueAddresses.length > 3) {
    try {
      const batchResults = await fetchCoinGeckoBatchPrices(uniqueAddresses, network);
      if (batchResults && Object.keys(batchResults).length > 0) {
        // Store results in cache
        for (const [address, price] of Object.entries(batchResults)) {
          const cacheKey = `${network}:${address.toLowerCase()}`;
          priceCache.tokens[cacheKey] = price;
          priceCache.lastUpdated[cacheKey] = Date.now();
          results[address] = price;
        }
        
        // Get any missing prices individually
        const missingAddresses = uniqueAddresses.filter(addr => !results[addr]);
        if (missingAddresses.length > 0) {
          for (const addr of missingAddresses) {
            fetchPromises.push(
              getTokenPrice(addr, network, options)
                .then(price => { results[addr] = price; })
                .catch(error => {
                  console.error(`Error fetching price for ${addr}: ${error.message}`);
                  results[addr] = null;
                })
            );
          }
          await Promise.all(fetchPromises);
        }
        
        return results;
      }
    } catch (error) {
      console.warn(`Batch price fetching failed: ${error.message}, falling back to individual requests`);
    }
  }

  // Fallback to individual requests
  for (const addr of uniqueAddresses) {
    fetchPromises.push(
      getTokenPrice(addr, network, options)
        .then(price => { results[addr] = price; })
        .catch(error => {
          console.error(`Error fetching price for ${addr}: ${error.message}`);
          results[addr] = null;
        })
    );
  }

  await Promise.all(fetchPromises);
  return results;
}

/**
 * Get price sources configured for a specific network
 * @param {string} network - Network name
 * @param {object} providersOverride - Optional providers to use instead of creating new ones
 * @returns {Array} Array of price sources
 */
function getPriceSourcesForNetwork(network, providersOverride = null) {
  const networkConfig = require('./config').networks[network] || {};
  const chainId = networkConfig.chainId || getChainIdForNetwork(network);
  
  // Use provided web3 providers if available, or create new ones
  const providers = providersOverride || {
    http: new ethers.providers.JsonRpcProvider(networkConfig.rpcUrl),
    ws: networkConfig.wsUrl ? new ethers.providers.WebSocketProvider(networkConfig.wsUrl) : null
  };

  // Get preferred DEXes for the network
  const dexes = networkConfig.dexes || {
    uniswapV2Router: networkConfig.contracts?.uniswapRouter || null,
    sushiswapRouter: networkConfig.contracts?.sushiswapRouter || null,
    quickswapRouter: networkConfig.contracts?.quickswapRouter || null
  };

  // Create array of price sources in priority order
  const priceSources = [
    new CoinGeckoSource(chainId),
    new CoinMarketCapSource(chainId),
    new ChainlinkSource(providers.http, chainId),
    new UniswapSource(providers.http, dexes),
    new DexScreenerSource(),
    // Fallback source
    new ZeroXAPISource(chainId)
  ];

  return priceSources;
}

/**
 * Map network name to chain ID
 * @param {string} network - Network name
 * @returns {number} Chain ID
 */
function getChainIdForNetwork(network) {
  const chainIdMap = {
    'ethereum': 1,
    'optimism': 10,
    'bsc': 56,
    'polygon': 137,
    'arbitrum': 42161,
    'avalanche': 43114
  };
  
  return chainIdMap[network.toLowerCase()] || 1; // Default to Ethereum
}

/**
 * Get the current price of native token (ETH, MATIC, etc.)
 * @param {string} network - Network name
 * @returns {Promise<number>} Native token price in USD
 */
async function getNativeTokenPrice(network) {
  const nativeTokenMap = {
    'ethereum': 'ethereum',
    'optimism': 'ethereum',
    'bsc': 'binancecoin',
    'polygon': 'matic-network',
    'arbitrum': 'ethereum',
    'avalanche': 'avalanche-2'
  };
  
  const coinId = nativeTokenMap[network.toLowerCase()];
  if (!coinId) return null;

  const cacheKey = `${network}:native`;
  
  // Check cache
  if (priceCache.tokens[cacheKey] && 
     (Date.now() - priceCache.lastUpdated[cacheKey]) < priceCache.cacheTTLMs) {
    return priceCache.tokens[cacheKey];
  }

  try {
    const response = await axios.get(
      `https://api.coingecko.com/api/v3/simple/price?ids=${coinId}&vs_currencies=usd`
    );
    
    if (response.data && response.data[coinId] && response.data[coinId].usd) {
      const price = response.data[coinId].usd;
      
      // Update cache
      priceCache.tokens[cacheKey] = price;
      priceCache.lastUpdated[cacheKey] = Date.now();
      
      return price;
    }
  } catch (error) {
    console.error(`Error fetching native token price for ${network}: ${error.message}`);
  }
  
  // Fallback to cached price if available
  return priceCache.tokens[cacheKey] || null;
}

/**
 * Clear price cache entries older than specified time
 * @param {number} maxAgeMs - Maximum age in milliseconds
 */
function clearOldCache(maxAgeMs = 3600000) { // Default 1 hour
  const now = Date.now();
  
  Object.keys(priceCache.lastUpdated).forEach(key => {
    if (now - priceCache.lastUpdated[key] > maxAgeMs) {
      delete priceCache.tokens[key];
      delete priceCache.lastUpdated[key];
    }
  });
}

/**
 * Save price cache to disk
 * @param {string} filePath - Path to cache file
 */
function saveCacheToDisk(filePath = 'price-cache.json') {
  try {
    fs.writeFileSync(
      filePath, 
      JSON.stringify({
        tokens: priceCache.tokens,
        lastUpdated: priceCache.lastUpdated,
        savedAt: Date.now()
      }, null, 2)
    );
  } catch (error) {
    console.error(`Error saving price cache to disk: ${error.message}`);
  }
}

/**
 * Load price cache from disk
 * @param {string} filePath - Path to cache file
 * @param {number} maxAgeMs - Maximum age of cache entries to load
 */
function loadCacheFromDisk(filePath = 'price-cache.json', maxAgeMs = 3600000) {
  try {
    if (fs.existsSync(filePath)) {
      const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      const now = Date.now();
      
      Object.keys(data.lastUpdated).forEach(key => {
        if (now - data.lastUpdated[key] <= maxAgeMs) {
          priceCache.tokens[key] = data.tokens[key];
          priceCache.lastUpdated[key] = data.lastUpdated[key];
        }
      });
      
      console.log(`Loaded ${Object.keys(priceCache.tokens).length} token prices from cache`);
    }
  } catch (error) {
    console.error(`Error loading price cache from disk: ${error.message}`);
  }
}

/**
 * CoinGecko price source
 */
class CoinGeckoSource {
  constructor(chainId) {
    this.name = 'CoinGecko';
    this.chainId = chainId;
    this.platformId = this.getPlatformId(chainId);
    this.rateLimitWindowMs = 60000; // 1 minute window
    this.maxRequestsPerWindow = 10; // Free API limit
    this.requestTimestamps = [];
  }
  
  getPlatformId(chainId) {
    const platformMap = {
      1: 'ethereum',
      10: 'optimistic-ethereum',
      56: 'binance-smart-chain',
      137: 'polygon-pos',
      42161: 'arbitrum-one',
      43114: 'avalanche'
    };
    return platformMap[chainId] || 'ethereum';
  }
  
  async getPrice(tokenAddress) {
    // Check rate limit
    this.enforceRateLimit();
    
    try {
      const response = await axios.get(
        `https://api.coingecko.com/api/v3/coins/${this.platformId}/contract/${tokenAddress}`
      );
      
      if (response.data && response.data.market_data && response.data.market_data.current_price) {
        return response.data.market_data.current_price.usd;
      }
    } catch (error) {
      throw new Error(`CoinGecko API error: ${error.message}`);
    }
    
    return null;
  }
  
  enforceRateLimit() {
    const now = Date.now();
    
    // Remove timestamps outside the window
    this.requestTimestamps = this.requestTimestamps.filter(
      timestamp => now - timestamp < this.rateLimitWindowMs
    );
    
    // Check if we've hit the limit
    if (this.requestTimestamps.length >= this.maxRequestsPerWindow) {
      const oldestTimestamp = this.requestTimestamps[0];
      const waitTime = this.rateLimitWindowMs - (now - oldestTimestamp);
      
      if (waitTime > 0) {
        throw new Error(`Rate limit exceeded. Try again in ${Math.ceil(waitTime / 1000)} seconds.`);
      }
    }
    
    // Add current timestamp
    this.requestTimestamps.push(now);
  }
}

/**
 * Fetch batch prices from CoinGecko
 * @param {string[]} addresses - Token addresses
 * @param {string} network - Network name
 * @returns {Promise<Object>} Prices by address
 */
async function fetchCoinGeckoBatchPrices(addresses, network) {
  const chainId = getChainIdForNetwork(network);
  const platformId = new CoinGeckoSource(chainId).platformId;
  
  // CoinGecko has a limit on the number of addresses per request
  const BATCH_SIZE = 25;
  const results = {};

  // Process in batches
  for (let i = 0; i < addresses.length; i += BATCH_SIZE) {
    const batch = addresses.slice(i, i + BATCH_SIZE);
    const addressesParam = batch.join(',');
    
    try {
      const response = await axios.get(
        `https://api.coingecko.com/api/v3/simple/token_price/${platformId}` +
        `?contract_addresses=${addressesParam}&vs_currencies=usd`
      );
      
      if (response.data) {
        // Add to results
        Object.entries(response.data).forEach(([address, data]) => {
          if (data.usd) {
            results[address.toLowerCase()] = data.usd;
          }
        });
      }
      
      // Respect rate limits
      if (i + BATCH_SIZE < addresses.length) {
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    } catch (error) {
      console.error(`CoinGecko batch price error: ${error.message}`);
      // Continue with other batches
    }
  }
  
  return results;
}

/**
 * CoinMarketCap price source
 */
class CoinMarketCapSource {
  constructor(chainId) {
    this.name = 'CoinMarketCap';
    this.chainId = chainId;
    // CoinMarketCap API key should be in .env or config
    this.apiKey = process.env.COINMARKETCAP_API_KEY || '';
  }
  
  async getPrice(tokenAddress) {
    if (!this.apiKey) {
      throw new Error('CoinMarketCap API key not configured');
    }
    
    try {
      const response = await axios.get(
        'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest', {
          headers: {
            'X-CMC_PRO_API_KEY': this.apiKey
          },
          params: {
            address: tokenAddress,
            platform_id: this.getPlatformId(this.chainId)
          }
        }
      );
      
      if (response.data && response.data.data) {
        const tokenData = Object.values(response.data.data)[0];
        if (tokenData && tokenData.quote && tokenData.quote.USD) {
          return tokenData.quote.USD.price;
        }
      }
    } catch (error) {
      throw new Error(`CoinMarketCap API error: ${error.message}`);
    }
    
    return null;
  }
  
  getPlatformId(chainId) {
    const platformMap = {
      1: 1, // Ethereum
      10: 12, // Optimism
      56: 2, // BSC
      137: 3, // Polygon
      42161: 11, // Arbitrum
      43114: 5 // Avalanche
    };
    return platformMap[chainId] || 1;
  }
}

/**
 * Chainlink price source
 */
class ChainlinkSource {
  constructor(provider, chainId) {
    this.name = 'Chainlink';
    this.provider = provider;
    this.chainId = chainId;
    this.feedRegistry = this.getFeedRegistryAddress(chainId);
    this.feedRegistryABI = [
      'function latestAnswer() external view returns (int256)',
      'function decimals() external view returns (uint8)'
    ];
  }
  
  getFeedRegistryAddress(chainId) {
    // Chainlink Feed Registry addresses by network
    const registryMap = {
      1: '0x47Fb2585D2C56Fe188D0E6ec628a38b74fCeeeDf', // Ethereum
      56: '0xF0014A5A2EB159ff80CAf2B861105ce7De36E6dE', // BSC
      137: '0xAB594600376Ec9fD91F8e885dADF0CE036862dE0', // Polygon
      // Add other networks as needed
    };
    return registryMap[chainId] || null;
  }
  
  async getPrice(tokenAddress) {
    if (!this.feedRegistry) {
      throw new Error(`Chainlink Feed Registry not available for chain ID ${this.chainId}`);
    }
    
    try {
      // Note: This is a simplified implementation.
      // A complete implementation would need to look up the correct feed based on token address
      // and handle different feed formats and aggregators.
      
      // For the purposes of this example, we assume the tokenAddress is a feed address
      const feedContract = new ethers.Contract(
        tokenAddress,
        this.feedRegistryABI,
        this.provider
      );
      
      const [latestPrice, decimals] = await Promise.all([
        feedContract.latestAnswer(),
        feedContract.decimals()
      ]);
      
      // Convert to decimal
      return parseFloat(ethers.utils.formatUnits(latestPrice, decimals));
    } catch (error) {
      throw new Error(`Chainlink error: ${error.message}`);
    }
  }
}

/**
 * Uniswap style DEX price source
 */
class UniswapSource {
  constructor(provider, dexConfig) {
    this.name = 'Uniswap';
    this.provider = provider;
    this.dexConfig = dexConfig;
    this.WETH_ADDRESSES = {
      1: '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', // Ethereum
      10: '0x4200000000000000000000000000000000000006', // Optimism
      56: '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c', // BSC (WBNB)
      137: '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270', // Polygon (WMATIC)
      42161: '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1', // Arbitrum
      43114: '0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7'  // Avalanche (WAVAX)
    };
  }
  
  async getPrice(tokenAddress, decimals = null) {
    const chainId = await this.provider.getNetwork().then(network => network.chainId);
    const nativeTokenAddress = this.WETH_ADDRESSES[chainId];
    
    if (!nativeTokenAddress) {
      throw new Error(`Native token address not configured for chain ID ${chainId}`);
    }
    
    if (!decimals) {
      // Get token decimals
      try {
        const tokenContract = new ethers.Contract(
          tokenAddress,
          ERC20_ABI,
          this.provider
        );
        decimals = await tokenContract.decimals();
      } catch (error) {
        throw new Error(`Error getting token decimals: ${error.message}`);
      }
    }
    
    // Try to find a pair with native token or USDC/USDT
    try {
      // Find the factory address
      const routerAddress = this.getRouterAddress(chainId);
      if (!routerAddress) {
        throw new Error(`No DEX router configured for chain ID ${chainId}`);
      }
      
      // Get native token price first if the token is not a stablecoin
      const nativeTokenPriceUsd = await this.getNativeTokenPrice(chainId);
      
      // Find pair address
      const pairAddress = await this.findPairAddress(
        tokenAddress,
        nativeTokenAddress,
        routerAddress
      );
      
      if (!pairAddress) {
        throw new Error(`No pair found for ${tokenAddress} with native token`);
      }
      
      // Get reserves
      const pairContract = new ethers.Contract(
        pairAddress,
        PAIR_ABI,
        this.provider
      );
      
      const [token0, token1, reserves] = await Promise.all([
        pairContract.token0(),
        pairContract.token1(),
        pairContract.getReserves()
      ]);
      
      // Determine which token is which in the pair
      const isToken0 = token0.toLowerCase() === tokenAddress.toLowerCase();
      const tokenReserve = isToken0 ? reserves[0] : reserves[1];
      const nativeTokenReserve = isToken0 ? reserves[1] : reserves[0];
      
      // Calculate price
      if (tokenReserve.isZero()) {
        throw new Error('Token reserve is zero');
      }
      
      const priceInNative = nativeTokenReserve.mul(ethers.BigNumber.from(10).pow(decimals))
        .div(tokenReserve);
      
      // Convert to USD
      return parseFloat(ethers.utils.formatUnits(priceInNative, 18)) * nativeTokenPriceUsd;
    } catch (error) {
      throw new Error(`Uniswap price error: ${error.message}`);
    }
  }
  
  getRouterAddress(chainId) {
    // Try to use configured routers from dexConfig
    if (this.dexConfig) {
      for (const routerType of ['uniswapV2Router', 'sushiswapRouter', 'quickswapRouter']) {
        if (this.dexConfig[routerType]) {
          return this.dexConfig[routerType];
        }
      }
    }
    
    // Fallback to known router addresses
    const routerMap = {
      1: '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D', // Uniswap V2
      10: '0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45', // Optimism Uniswap
      56: '0x10ED43C718714eb63d5aA57B78B54704E256024E', // PancakeSwap
      137: '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff', // QuickSwap
      42161: '0xd0a1e359811322d97991e03f863a0c30c2cf029c', // Arbitrum SushiSwap
      43114: '0x60aE616a2155Ee3d9A68541Ba4544862310933d4'  // Avalanche Trader Joe
    };
    
    return routerMap[chainId] || null;
  }
  
  async getNativeTokenPrice(chainId) {
    // This would normally get the native token price from a reliable source
    // For simplicity, we'll just use hardcoded values for example purposes
    const nativePrices = {
      1: 2000,  // ETH
      10: 2000, // ETH on Optimism
      56: 300,  // BNB
      137: 0.7, // MATIC
      42161: 2000, // ETH on Arbitrum
      43114: 20 // AVAX
    };
    
    return nativePrices[chainId] || 2000; // Default to ETH price
  }
  
  async findPairAddress(tokenA, tokenB, routerAddress) {
    // This is a simplified placeholder that would normally interact with the factory
    // to find the pair address. For a real implementation, you'd need to:
    // 1. Get the factory address from the router
    // 2. Call factory.getPair(tokenA, tokenB)
    
    // For simplicity, we'll just return null, which will cause the price fetch to fail
    return null;
  }
}

/**
 * DexScreener price source
 */
class DexScreenerSource {
  constructor() {
    this.name = 'DexScreener';
    this.baseUrl = 'https://api.dexscreener.com/latest/dex';
  }
  
  async getPrice(tokenAddress) {
    try {
      const response = await axios.get(`${this.baseUrl}/tokens/${tokenAddress}`);
      
      if (response.data && response.data.pairs && response.data.pairs.length > 0) {
        // Sort by liquidity to get the most liquid pair
        const sortedPairs = response.data.pairs.sort((a, b) => 
          parseFloat(b.liquidity?.usd || 0) - parseFloat(a.liquidity?.usd || 0)
        );
        
        // Get price from the most liquid pair
        const topPair = sortedPairs[0];
        if (topPair && topPair.priceUsd) {
          return parseFloat(topPair.priceUsd);
        }
      }
    } catch (error) {
      throw new Error(`DexScreener error: ${error.message}`);
    }
    
    return null;
  }
}

/**
 * 0x API price source (fallback)
 */
class ZeroXAPISource {
  constructor(chainId) {
    this.name = '0x API';
    this.chainId = chainId;
  }
  
  async getPrice(tokenAddress) {
    try {
      // 0x API uses different base URLs for different chains
      const baseUrl = this.getBaseUrl(this.chainId);
      if (!baseUrl) {
        throw new Error(`Unsupported chain ID for 0x API: ${this.chainId}`);
      }
      
      // Get a quote for selling 1 unit of the token for USDC
      const USDC_ADDRESS = this.getUsdcAddress(this.chainId);
      
      const response = await axios.get(`${baseUrl}/price`, {
        params: {
          sellToken: tokenAddress,
          buyToken: USDC_ADDRESS,
          sellAmount: '1000000000000000000' // 1 token in wei (assuming 18 decimals)
        }
      });
      
      if (response.data && response.data.price) {
        return parseFloat(response.data.price);
      }
    } catch (error) {
      throw new Error(`0x API error: ${error.message}`);
    }
    
    return null;
  }
  
  getBaseUrl(chainId) {
    const urls = {
      1: 'https://api.0x.org/swap/v1',
      10: 'https://optimism.api.0x.org/swap/v1',
      56: 'https://bsc.api.0x.org/swap/v1',
      137: 'https://polygon.api.0x.org/swap/v1',
      42161: 'https://arbitrum.api.0x.org/swap/v1',
      43114: 'https://avalanche.api.0x.org/swap/v1'
    };
    return urls[chainId] || null;
  }
  
  getUsdcAddress(chainId) {
    const addresses = {
      1: '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
      10: '0x7F5c764cBc14f9669B88837ca1490cCa17c31607',
      56: '0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d',
      137: '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
      42161: '0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8',
      43114: '0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E'
    };
    return addresses[chainId] || addresses[1]; // Default to Ethereum USDC
  }
}

// Automatic cache clearing
setInterval(() => {
  clearOldCache();
}, 3600000); // Clear old cache entries every hour

// Initialize by loading cache from disk if available
loadCacheFromDisk();

// Save cache to disk on exit
process.on('exit', () => {
  saveCacheToDisk();
});

// Handle SIGINT (Ctrl+C)
process.on('SIGINT', () => {
  saveCacheToDisk();
  process.exit(0);
});

module.exports = {
  getTokenPrice,
  getBatchTokenPrices,
  getNativeTokenPrice,
  clearOldCache,
  saveCacheToDisk,
  loadCacheFromDisk
}; 