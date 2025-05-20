import requests
import random
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BotX3")

SUBGRAPH_URL = "https://gateway.thegraph.com/api/44577482a091e48ca832f7d938f06d82/subgraphs/id/5Tf9s7syYLHQzhmtjukjTjmhFwx7c3hrdVxy4jo3TgCC"

def fetch_uniswap_data(chains):
    """Fetch liquidity data with retry logic."""
    data = {}
    query = """
    {
      pools(first: 1, orderBy: liquidity, orderDirection: desc) {
        id
        token0 { id symbol }
        token1 { id symbol }
        liquidity
        volumeUSD
      }
    }
    """
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        response = session.post(SUBGRAPH_URL, json={'query': query}, timeout=60)
        response.raise_for_status()
        json_data = response.json()
        pools = json_data['data'].get('pools', [])
        pool = pools[0] if pools else None
        
        for chain in chains:
            if chain['name'] == 'Polygon' and pool:
                data[chain['name']] = {
                    'depth': min(float(pool['liquidity']) / 1e18, 1e6),  # Normalize to reasonable scale
                    'volume': float(pool['volumeUSD']) / 24,
                    'price': 1.0
                }
            else:
                data[chain['name']] = {
                    'depth': random.uniform(100, 1000),
                    'volume': random.uniform(50, 500),
                    'price': 1 + random.uniform(0, 0.1)
                }
            logger.info(f"{chain['name']} - Depth: ${data[chain['name']]['depth']:.2f}, Volume: ${data[chain['name']]['volume']:.2f}")
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        for chain in chains:
            data[chain['name']] = {
                'depth': random.uniform(100, 1000),
                'volume': random.uniform(50, 500),
                'price': 1 + random.uniform(0, 0.1)
            }
    return data

def enrich_data(chains, historical_window=60):
    """Enrich data with features for ML/RL."""
    chain_data = fetch_uniswap_data(chains)
    historical_data = {chain['name']: [] for chain in chains}
    
    for _ in range(historical_window):
        temp_data = fetch_uniswap_data(chains)
        for chain in chains:
            historical_data[chain['name']].append({
                'price': temp_data[chain['name']]['price'],
                'depth': temp_data[chain['name']]['depth'],
                'volume': temp_data[chain['name']]['volume']
            })
    
    df = pd.DataFrame()
    for chain in chains:
        chain_df = pd.DataFrame(historical_data[chain['name']])
        chain_df['chain'] = chain['name']
        chain_df['volatility'] = chain_df['price'].rolling(window=5).std()
        chain_df['spread'] = chain_df['price'] - chain_df['price'].mean()
        chain_df['timestamp'] = pd.date_range(end=pd.Timestamp.now(), periods=len(chain_df), freq='1min')
        chain_df['hour'] = chain_df['timestamp'].dt.hour
        df = pd.concat([df, chain_df])
    
    df['future_price'] = df.groupby('chain')['price'].shift(-1)
    df['profit'] = (df['future_price'] / df['price']) - 1
    df['target'] = (df['profit'] > 0.02).astype(int)
    return df.dropna()
