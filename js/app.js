// static/js/app.js
(async () => {
    // Chart options with dark theme
    const chartOptions = {
        layout: {
            background: { color: '#252535' },
            textColor: '#f8fafc',
        },
        grid: {
            vertLines: { color: 'rgba(255, 255, 255, 0.1)' },
            horzLines: { color: 'rgba(255, 255, 255, 0.1)' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: {
                color: 'rgba(99, 102, 241, 0.3)',
                style: LightweightCharts.LineStyle.Dashed,
            },
            horzLine: {
                color: 'rgba(99, 102, 241, 0.3)',
                style: LightweightCharts.LineStyle.Dashed,
            },
        },
        timeScale: {
            borderColor: 'rgba(255, 255, 255, 0.2)',
        },
    };
    // Create and configure chart
    const chart = LightweightCharts.createChart(
        document.getElementById('chart'), 
        chartOptions
    );
    const lineSeries = chart.addLineSeries({
        color: '#6366f1',
        lineWidth: 2,
        crosshairMarkerVisible: true,
        crosshairMarkerRadius: 4,
    });
    
    // Load symbols and timeframes
    async function loadAvailableSymbols() {
        try {
            const res = await fetch('/api/available_symbols');
            const data = await res.json();
            
            const symbolSelect = document.getElementById('symbol-select');
            symbolSelect.innerHTML = '';
            
            // Add CEX symbols
            if (data.cex && data.cex.length > 0) {
                const cexGroup = document.createElement('optgroup');
                cexGroup.label = 'CEX';
                data.cex.forEach(symbol => {
                    const option = document.createElement('option');
                    option.value = symbol;
                    option.textContent = symbol;
                    cexGroup.appendChild(option);
                });
                symbolSelect.appendChild(cexGroup);
            }
            
            // Add DEX symbols
            if (data.dex && data.dex.length > 0) {
                const dexGroup = document.createElement('optgroup');
                dexGroup.label = 'DEX';
                data.dex.forEach(symbol => {
                    const option = document.createElement('option');
                    option.value = symbol;
                    option.textContent = symbol;
                    dexGroup.appendChild(option);
                });
                symbolSelect.appendChild(dexGroup);
            }
            
            return data.cex[0] || data.dex[0] || 'AAPL';  // Default symbol
        } catch (err) {
            console.error('Failed to load symbols:', err);
            return 'AAPL';
        }
    }
    
    async function loadAvailableTimeframes() {
        try {
            const res = await fetch('/api/timeframes');
            const data = await res.json();
            
            const timeframeSelect = document.getElementById('timeframe-select');
            timeframeSelect.innerHTML = '';
            
            Object.entries(data).forEach(([value, label]) => {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = label;
                timeframeSelect.appendChild(option);
            });
            
            return Object.keys(data)[0] || '1d';  // Default timeframe
        } catch (err) {
            console.error('Failed to load timeframes:', err);
            return '1d';
        }
    }
    
    // Function to load price data for a symbol
    async function loadSymbolData(symbol, timeframe = '1d') {
        const res = await fetch(`/api/market_data?symbols=${symbol}&timeframe=${timeframe}`);
        const data = await res.json();
        const keyPrefix = symbol.includes('/') ? 'cex_' : 'cex_';
        const key = `${keyPrefix}${symbol}`;
        
        if (data[key] && data[key].length) {
            const seriesData = data[key].map(d => ({ 
                time: Math.floor(new Date(d.timestamp).getTime()/1000), 
                value: d.close || d.price || 0
            }));
            lineSeries.setData(seriesData);
            chart.timeScale().fitContent();
            
            // Update statistics
            loadStatistics(symbol, timeframe);
            
            return true;
        }
        return false;
    }
    
    // Load statistics for the selected symbol
    async function loadStatistics(symbol, timeframe = '1d') {
        try {
            const res = await fetch(`/api/statistics?symbol=${symbol}&timeframe=${timeframe}`);
            const stats = await res.json();
            
            if (stats.error) {
                console.error('Statistics error:', stats.error);
                return;
            }
            
            const statsContainer = document.getElementById('market-stats');
            statsContainer.innerHTML = '';
            
            // Create stat items
            const statsToDisplay = [
                { label: 'Last Price', value: formatPrice(stats.last_price) },
                { label: 'Change (24h)', value: formatChange(stats.change_24h, stats.change_24h_percent) },
                { label: 'High (24h)', value: formatPrice(stats.high_24h) },
                { label: 'Low (24h)', value: formatPrice(stats.low_24h) }
            ];
            
            if (stats.volume_24h) {
                statsToDisplay.push({ label: 'Volume (24h)', value: formatVolume(stats.volume_24h) });
            }
            
            statsToDisplay.forEach(stat => {
                const statItem = document.createElement('div');
                statItem.className = 'stat-item';
                statItem.innerHTML = `
                    <div class="stat-label">${stat.label}</div>
                    <div class="stat-value">${stat.value}</div>
                `;
                statsContainer.appendChild(statItem);
            });
        } catch (err) {
            console.error('Failed to load statistics:', err);
        }
    }
    
    // Formatting helpers
    function formatPrice(price) {
        return price !== null ? `$${parseFloat(price).toFixed(2)}` : 'N/A';
    }
    
    function formatChange(change, changePercent) {
        if (change === null || changePercent === null) return 'N/A';
        
        const changeStr = `$${Math.abs(parseFloat(change)).toFixed(2)}`;
        const percentStr = `${Math.abs(parseFloat(changePercent)).toFixed(2)}%`;
        const direction = change >= 0 ? 'price-up' : 'price-down';
        const arrow = change >= 0 ? '↑' : '↓';
        
        return `<span class="${direction}">${arrow} ${changeStr} (${percentStr})</span>`;
    }
    
    function formatVolume(volume) {
        if (volume === null) return 'N/A';
        
        if (volume >= 1e9) {
            return `$${(volume / 1e9).toFixed(2)}B`;
        } else if (volume >= 1e6) {
            return `$${(volume / 1e6).toFixed(2)}M`;
        } else if (volume >= 1e3) {
            return `$${(volume / 1e3).toFixed(2)}K`;
        } else {
            return `$${volume.toFixed(2)}`;
        }
    }
    
    // Initialize the UI - load symbols, timeframes, and initial data
    async function initializeUI() {
        const defaultSymbol = await loadAvailableSymbols();
        const defaultTimeframe = await loadAvailableTimeframes();
        await loadSymbolData(defaultSymbol, defaultTimeframe);
    }
    
    // Initialize
    await initializeUI();
    
    // Setup symbol and timeframe selectors
    const symbolSelect = document.getElementById('symbol-select');
    const timeframeSelect = document.getElementById('timeframe-select');
    
    symbolSelect.addEventListener('change', () => {
        loadSymbolData(symbolSelect.value, timeframeSelect.value);
    });
    
    timeframeSelect.addEventListener('change', () => {
        loadSymbolData(symbolSelect.value, timeframeSelect.value);
    });
    
    // Load and display Chainlink price feeds
    async function loadChainlinkPrices() {
        try {
            const res = await fetch('/api/chainlink_prices');
            const prices = await res.json();
            
            const priceList = document.getElementById('chainlink-prices');
            priceList.innerHTML = '';
            
            if (Object.keys(prices).length === 0) {
                priceList.innerHTML = '<li class="price-item">No Chainlink feeds configured</li>';
                return;
            }
            
            for (const [address, price] of Object.entries(prices)) {
                const shortAddress = `${address.substring(0, 6)}...${address.substring(address.length - 4)}`;
                const item = document.createElement('li');
                item.className = 'price-item';
                item.innerHTML = `
                    <div class="token-name">
                        <div class="token-icon">CL</div>
                        ${shortAddress}
                    </div>
                    <div class="price-value">$${parseFloat(price).toFixed(2)}</div>
                `;
                priceList.appendChild(item);
            }
        } catch (err) {
            console.error('Failed to load Chainlink prices:', err);
        }
    }
    
    // Update trade signals panel after wallet connection
    function updateTradeSignals(isConnected, address) {
        const tradeSignals = document.getElementById('trade-signals');
        
        if (isConnected) {
            // Generate some mock trade signals based on current symbol 
            const symbol = symbolSelect.value;
            tradeSignals.innerHTML = `
                <div class="signal-item">
                    <div class="signal-header">
                        <span class="signal-type">AI Quantum Analysis</span>
                        <span class="signal-confidence price-up">72% Confidence</span>
                    </div>
                    <div class="signal-action">BUY ${symbol} on next 4-hour candle close</div>
                    <div class="signal-targets">
                        <div>Entry: $${(Math.random() * 1000 + 100).toFixed(2)}</div>
                        <div>Target: $${(Math.random() * 2000 + 500).toFixed(2)}</div>
                        <div>Stop: $${(Math.random() * 50 + 80).toFixed(2)}</div>
                    </div>
                </div>
                <div class="signal-item">
                    <div class="signal-header">
                        <span class="signal-type">Sentiment Analysis</span>
                        <span class="signal-confidence">NEUTRAL</span>
                    </div>
                    <div class="signal-action">Market sentiment shifting bullish on ${symbol}</div>
                </div>
            `;
        } else {
            tradeSignals.innerHTML = `
                <p>Connect your wallet to view AI-powered trade signals and portfolio recommendations.</p>
            `;
        }
    }
    
    // Load Chainlink prices initially and every 30 seconds
    loadChainlinkPrices();
    setInterval(loadChainlinkPrices, 30000);

    // Ethers.js wallet connect
    const provider = new ethers.providers.Web3Provider(window.ethereum, 'any');
    document.getElementById('connectButton').onclick = async () => {
        try {
            await provider.send('eth_requestAccounts', []);
            const signer = provider.getSigner();
            const address = await signer.getAddress();
            const addressElement = document.getElementById('address');
            addressElement.innerText = `${address.substring(0, 6)}...${address.substring(address.length - 4)}`;
            document.getElementById('connectButton').innerText = 'Connected';
            
            // Update trade signals with personalized content
            updateTradeSignals(true, address);
        } catch (err) {
            console.error('Failed to connect wallet:', err);
        }
    };
    
    // Add some styling for trade signals
    const style = document.createElement('style');
    style.textContent = `
    .signal-item {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 15px;
    }
    .signal-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .signal-type {
        font-weight: 600;
    }
    .signal-confidence {
        font-size: 0.9em;
        opacity: 0.9;
    }
    .signal-action {
        font-size: 1.1em;
        margin-bottom: 10px;
    }
    .signal-targets {
        display: flex;
        justify-content: space-between;
        font-size: 0.9em;
        opacity: 0.8;
    }
    `;
    document.head.appendChild(style);
})(); 