"""
AI-Optimized Trading Strategy

This module integrates reinforcement learning with the quantum trading system
to optimize decision-making for trade execution.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import matplotlib.pyplot as plt

# Add parent directory to path to import trading modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_optimization.reinforcement_trainer import ReinforcementTrainer
from ai_optimization.feature_engineering import TradingFeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AIOptimizedTrader:
    """
    Integrates AI optimization with the quantum trading system to
    enhance decision-making and improve profitability.
    """
    
    def __init__(self, input_features=10, model_path=None, risk_level='moderate'):
        """
        Initialize the AI-optimized trader with enhanced risk management
        
        Args:
            input_features: Number of input features for the model
            model_path: Path to pre-trained model (if available)
            risk_level: Risk tolerance level ('conservative', 'moderate', 'aggressive')
        """
        self.trainer = ReinforcementTrainer(input_dim=input_features)
        self.feature_engineer = TradingFeatureEngineer()
        self.input_features = input_features
        self.trained = False
        self.model_load_attempts = 0
        self.model_load_success = False
        self.model_path = model_path
        
        # Risk management settings
        self.risk_level = risk_level
        self.risk_settings = self._initialize_risk_settings(risk_level)
        
        # Performance tracking
        self.performance_history = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'standard_trades': 0,
            'flashloan_trades': 0,
            'total_profit': 0.0,
            'peak_capital': 0.0,
            'drawdown': 0.0,
            'trade_history': []
        }
        
        # Try to load the model with fallbacks
        self._load_model_with_fallbacks()
        
        logging.info("AI Optimized Trader initialized with risk level: " + risk_level)
        if self.trained:
            logging.info(f"Loaded pre-trained model from {model_path}")
    
    def train_on_simulation_data(self, data_path, epochs=100):
        """
        Train the AI model on data from trading simulations
        
        Args:
            data_path: Path to simulation data file
            epochs: Number of training epochs
            
        Returns:
            Training history metrics
        """
        logging.info(f"Training AI model on simulation data: {data_path}")
        
        # Load and preprocess the data
        df = self.feature_engineer.load_data(data_path)
        if df.empty:
            logging.error("Failed to load simulation data")
            return None
        
        # Train the model
        history = self.trainer.train_on_dataset(data_path, num_epochs=epochs)
        
        if history:
            self.trained = True
            
            # Save the model and feature configuration
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs('models', exist_ok=True)
            
            model_path = f"models/trading_rl_model_{timestamp}.h5"
            feature_config_path = f"models/trading_rl_features_{timestamp}.json"
            
            self.trainer.save_model(model_path)
            self.feature_engineer.save_feature_config(feature_config_path)
            
        return history
    
    def _initialize_risk_settings(self, risk_level):
        """
        Initialize risk management settings based on risk level
        
        Args:
            risk_level: Risk tolerance level ('conservative', 'moderate', 'aggressive')
            
        Returns:
            Dictionary of risk settings
        """
        if risk_level == 'conservative':
            return {
                'max_trade_size': 0.05,  # 5% of capital per trade
                'max_daily_trades': 20,
                'min_profit_threshold': 0.005,  # 0.5% minimum profit
                'max_risk_score': 0.4,  # Only low-risk trades
                'flashloan_confidence_threshold': 0.8,  # High confidence for flashloans
                'max_drawdown': 0.05,  # 5% maximum drawdown
                'gas_price_limit': 100,  # Max gas price in Gwei
                'diversification_required': True,  # Require diverse assets
                'stop_loss_percent': 0.02  # 2% stop loss
            }
        elif risk_level == 'aggressive':
            return {
                'max_trade_size': 0.15,  # 15% of capital per trade
                'max_daily_trades': 50,
                'min_profit_threshold': 0.001,  # 0.1% minimum profit
                'max_risk_score': 0.8,  # Allow higher risk trades
                'flashloan_confidence_threshold': 0.4,  # Lower bar for flashloans
                'max_drawdown': 0.15,  # 15% maximum drawdown
                'gas_price_limit': 300,  # Higher gas price tolerance
                'diversification_required': False,  # Don't require diversification
                'stop_loss_percent': 0.05  # 5% stop loss
            }
        else:  # 'moderate' - default
            return {
                'max_trade_size': 0.10,  # 10% of capital per trade
                'max_daily_trades': 30,
                'min_profit_threshold': 0.002,  # 0.2% minimum profit
                'max_risk_score': 0.6,  # Moderate risk tolerance
                'flashloan_confidence_threshold': 0.6,  # Moderate confidence required
                'max_drawdown': 0.10,  # 10% maximum drawdown
                'gas_price_limit': 200,  # Moderate gas price limit
                'diversification_required': True,  # Prefer diversification
                'stop_loss_percent': 0.03  # 3% stop loss
            }
    
    def _load_model_with_fallbacks(self):
        """
        Try to load the model with multiple fallback options for robustness
        """
        # Try primary model path first
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.trained = self.trainer.load_model(self.model_path)
                if self.trained:
                    self.model_load_success = True
                    # Load feature configuration if available
                    feature_config_path = self.model_path.replace('.h5', '_features.json')
                    if os.path.exists(feature_config_path):
                        self.feature_engineer.load_feature_config(feature_config_path)
                    return True
            except Exception as e:
                logging.error(f"Error loading model {self.model_path}: {str(e)}")
                self.model_load_attempts += 1
        
        # If primary model failed, try searching for alternatives
        if not self.model_load_success:
            try:
                models_dir = 'models'
                if os.path.exists(models_dir):
                    # Find all model files
                    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
                    if model_files:
                        # Sort by creation time, newest first
                        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
                        
                        # Try up to 3 most recent models
                        for model_file in model_files[:3]:
                            try:
                                backup_path = os.path.join(models_dir, model_file)
                                logging.info(f"Attempting to load alternative model: {backup_path}")
                                self.trained = self.trainer.load_model(backup_path)
                                if self.trained:
                                    self.model_load_success = True
                                    self.model_path = backup_path
                                    # Try to load feature config
                                    feature_config_path = backup_path.replace('.h5', '_features.json')
                                    if os.path.exists(feature_config_path):
                                        self.feature_engineer.load_feature_config(feature_config_path)
                                    logging.info(f"Successfully loaded alternative model: {model_file}")
                                    return True
                            except Exception as e:
                                logging.warning(f"Failed to load alternative model {model_file}: {str(e)}")
                                self.model_load_attempts += 1
            except Exception as e:
                logging.error(f"Error during model fallback process: {str(e)}")
        
        logging.warning("Could not load any model, using untrained model for now")
        return False

    def optimize_execution_strategy(self, opportunity):
        """
        Determine the optimal execution strategy for a trading opportunity
        with advanced risk management
        
        Args:
            opportunity: Trading opportunity data
            
        Returns:
            action: 0 for standard execution, 1 for flashloan execution
            confidence: Confidence score for the prediction
            metadata: Additional decision metadata for monitoring
        """
        # Start with risk assessment
        risk_score = opportunity.get('risk_score', 0.5)
        gas_price_gwei = opportunity.get('gas_price_gwei', 50)
        
        # Risk management checks
        if risk_score > self.risk_settings['max_risk_score']:
            logging.info(f"Trade rejected: risk score {risk_score} exceeds maximum {self.risk_settings['max_risk_score']}")
            return 0, 0.0, {'decision': 'rejected', 'reason': 'risk_exceeded', 'risk_score': risk_score}
        
        if gas_price_gwei > self.risk_settings['gas_price_limit']:
            logging.info(f"Trade caution: gas price {gas_price_gwei} exceeds limit {self.risk_settings['gas_price_limit']}")
            # Don't reject, but adjust profit expectations
        
        # If we have direct profit calculations, use those first
        if 'standard_profit' in opportunity and 'flashloan_profit' in opportunity:
            standard_profit = opportunity.get('standard_profit', 0)
            flashloan_profit = opportunity.get('flashloan_profit', 0)
            
            # Apply risk adjustments
            risk_adjustment = 1.0 - (risk_score * 0.3)  # Higher risk reduces expected profit
            gas_adjustment = 1.0 - min(1.0, (gas_price_gwei / self.risk_settings['gas_price_limit']) * 0.2)
            
            # Apply adjustments to profits
            adjusted_standard_profit = standard_profit * risk_adjustment * gas_adjustment
            adjusted_flashloan_profit = flashloan_profit * risk_adjustment * gas_adjustment * 0.95  # Extra 5% caution for flashloans
            
            # Minimum profit check
            min_profit = self.risk_settings['min_profit_threshold']
            if max(adjusted_standard_profit, adjusted_flashloan_profit) < min_profit:
                logging.info(f"Trade rejected: expected profit below minimum threshold of {min_profit}")
                return 0, 0.0, {'decision': 'rejected', 'reason': 'profit_too_low', 
                               'adjusted_profits': [adjusted_standard_profit, adjusted_flashloan_profit]}
            
            # Calculate profit improvement for confidence
            baseline = abs(adjusted_standard_profit) if adjusted_standard_profit != 0 else 0.0001
            profit_diff = adjusted_flashloan_profit - adjusted_standard_profit
            confidence = min(0.99, max(0.01, abs(profit_diff / baseline)))  # Normalize to 0.01-0.99
            
            # Default confidence if we couldn't calculate
            if pd.isna(confidence):
                confidence = 0.5
            
            # Decision metadata for monitoring
            metadata = {
                'original_standard_profit': standard_profit,
                'original_flashloan_profit': flashloan_profit,
                'adjusted_standard_profit': adjusted_standard_profit,
                'adjusted_flashloan_profit': adjusted_flashloan_profit,
                'risk_score': risk_score,
                'gas_price_gwei': gas_price_gwei,
                'risk_adjustment': risk_adjustment,
                'gas_adjustment': gas_adjustment,
                'confidence': confidence,
                'decision': 'accepted'
            }
            
            # Make the decision with enhanced risk awareness
            if adjusted_flashloan_profit > adjusted_standard_profit:
                # Check confidence threshold for flashloans
                if confidence >= self.risk_settings['flashloan_confidence_threshold']:
                    # Flashloan is better with sufficient confidence
                    action = 1
                    logging.info(f"Flashloan execution recommended: {adjusted_flashloan_profit:.6f} vs standard: {adjusted_standard_profit:.6f} (confidence: {confidence:.2f})")
                    metadata['strategy'] = 'flashloan'
                else:
                    # Not enough confidence for flashloan despite higher potential
                    action = 0
                    logging.info(f"Standard execution selected despite lower profit due to confidence threshold: {confidence:.2f} < {self.risk_settings['flashloan_confidence_threshold']}")
                    metadata['strategy'] = 'standard'
                    metadata['reason'] = 'low_confidence'
            else:
                # Standard is better
                action = 0
                logging.info(f"Standard execution recommended: {adjusted_standard_profit:.6f} vs flashloan: {adjusted_flashloan_profit:.6f}")
                metadata['strategy'] = 'standard'
            
            # Update performance tracking
            self.performance_history['total_trades'] += 1
            if action == 1:
                self.performance_history['flashloan_trades'] += 1
            else:
                self.performance_history['standard_trades'] += 1
            
            return action, confidence, metadata
        elif self.trained:
            # Use the trained model if available
            action, confidence = self.trainer.predict_best_action(opportunity)
            
            strategy_name = "flashloan" if action == 1 else "standard"
            logging.info(f"AI recommends {strategy_name} execution with {confidence:.2f} confidence")
            
            return action, confidence
        else:
            # Fallback to a simple heuristic
            expected_profit = opportunity.get('expected_profit', 0)
            risk_score = opportunity.get('risk_score', 0.5)
            
            # Use standard for small profits and high risk, flashloan otherwise
            if expected_profit < 0.01 or risk_score > 0.7:
                return 0, 0.7  # Standard with moderate confidence
            else:
                return 1, 0.6  # Flashloan with moderate confidence
    
    def enhance_opportunity(self, opportunity):
        """
        Enhance a trading opportunity with AI-optimized execution strategy
        and risk management checks
        
        Args:
            opportunity: Trading opportunity dictionary
            
        Returns:
            Enhanced opportunity with optimal execution strategy or None if rejected
        """
        # Apply circuit breaker if active
        if self._check_circuit_breaker():
            logging.warning("Circuit breaker active - rejecting all trades")
            return None
            
        # Make a copy to avoid modifying the original
        enhanced = opportunity.copy()
        
        # Predict optimal execution strategy with risk management
        action, confidence, metadata = self.optimize_execution_strategy(opportunity)
        
        # Check if trade was rejected by risk management
        if metadata.get('decision') == 'rejected':
            logging.info(f"Trade rejected by risk management: {metadata.get('reason')}")
            return None
        
        # Add AI recommendations to the opportunity
        enhanced['ai_recommended_strategy'] = 'flashloan' if action == 1 else 'standard'
        enhanced['ai_confidence'] = float(confidence)
        enhanced['risk_metadata'] = metadata
        
        # Set the execution strategy based on AI recommendation
        enhanced['execution_strategy'] = enhanced['ai_recommended_strategy']
        
        return enhanced
    
    def _check_circuit_breaker(self):
        """
        Check if circuit breaker conditions are met and trading should be paused
        
        Returns:
            True if circuit breaker is active, False otherwise
        """
        # Check for drawdown circuit breaker
        if hasattr(self, 'initial_capital') and self.initial_capital > 0:
            current_capital = self.initial_capital + self.performance_history['total_profit']
            drawdown = 1.0 - (current_capital / self.performance_history['peak_capital']) if self.performance_history['peak_capital'] > 0 else 0
            
            if drawdown > self.risk_settings['max_drawdown']:
                logging.warning(f"Circuit breaker activated: drawdown {drawdown:.2%} exceeds max {self.risk_settings['max_drawdown']:.2%}")
                return True
            
        # Check for consecutive loss circuit breaker
        if len(self.performance_history['trade_history']) >= 5:
            recent_trades = self.performance_history['trade_history'][-5:]
            if all(trade['profit'] < 0 for trade in recent_trades):
                logging.warning("Circuit breaker activated: 5 consecutive losing trades")
                return True
                
        # Check for rapid trade frequency circuit breaker
        daily_trade_limit = self.risk_settings['max_daily_trades']
        if hasattr(self, 'today_trade_count') and self.today_trade_count > daily_trade_limit:
            logging.warning(f"Circuit breaker activated: daily trade limit {daily_trade_limit} exceeded")
            return True
            
        return False
        
    def optimize_trading_batch(self, opportunities):
        """
        Apply AI optimization to a batch of trading opportunities
        with risk management filters
        
        Args:
            opportunities: List of trading opportunities
            
        Returns:
            Enhanced opportunities with optimal execution strategies
        """
        enhanced_opportunities = []
        rejected_count = 0
        
        for opportunity in opportunities:
            enhanced = self.enhance_opportunity(opportunity)
            if enhanced:  # Only add non-rejected opportunities
                enhanced_opportunities.append(enhanced)
            else:
                rejected_count += 1
            
        accepted_count = len(enhanced_opportunities)
        logging.info(f"Processed {len(opportunities)} opportunities: {accepted_count} accepted, {rejected_count} rejected")
        
        # Generate summary statistics
        if enhanced_opportunities:
            strategy_counts = {
                'standard': sum(1 for opp in enhanced_opportunities 
                               if opp['ai_recommended_strategy'] == 'standard'),
                'flashloan': sum(1 for opp in enhanced_opportunities 
                                if opp['ai_recommended_strategy'] == 'flashloan')
            }
            logging.info(f"Strategy distribution: {strategy_counts['standard']} standard, {strategy_counts['flashloan']} flashloan")
        
        # Check portfolio diversity if required
        if self.risk_settings['diversification_required'] and len(enhanced_opportunities) > 3:
            asset_counts = {}
            for opp in enhanced_opportunities:
                assets = opp.get('assets', [])
                for asset in assets:
                    asset_counts[asset] = asset_counts.get(asset, 0) + 1
            
            # If any asset represents more than 40% of opportunities, reduce its presence
            total_opps = len(enhanced_opportunities)
            for asset, count in asset_counts.items():
                if count / total_opps > 0.4:
                    logging.warning(f"Diversity warning: {asset} represents {count/total_opps:.1%} of trades, reducing exposure")
                    # Keep only half of the opportunities with this asset
                    to_remove = count // 2
                    for i in range(len(enhanced_opportunities)-1, -1, -1):
                        if to_remove <= 0:
                            break
                        if asset in enhanced_opportunities[i].get('assets', []):
                            enhanced_opportunities.pop(i)
                            to_remove -= 1
        
        return enhanced_opportunities
        
        # This seems to be a duplicate return statement that should be removed
    
    def analyze_model_performance(self, test_data):
        """
        Analyze model performance with advanced risk analytics
        
        Args:
            test_data: Path to test data file, DataFrame, or list of trade opportunities
            
        Returns:
            Dictionary with performance metrics and visualization data
        """
        # Load and process test data based on input type
        if isinstance(test_data, str):
            # It's a file path, load it
            try:
                if test_data.endswith('.json'):
                    with open(test_data, 'r') as f:
                        opportunities = json.load(f)
                    df = pd.DataFrame(opportunities)
                elif test_data.endswith('.csv'):
                    df = pd.read_csv(test_data)
                else:
                    df = self.feature_engineer.load_data(test_data)
            except Exception as e:
                logging.error(f"Failed to load test data: {str(e)}")
                return None
        elif isinstance(test_data, list):
            # It's a list of opportunities
            df = pd.DataFrame(test_data)
        elif isinstance(test_data, pd.DataFrame):
            # It's already a DataFrame
            df = test_data.copy()
        else:
            logging.error(f"Invalid test data format: {type(test_data)}")
            return None
            
        if df.empty:
            logging.error("No valid test data available")
            return None
        
        # Prepare metrics tracking
        results = {
            'total_opportunities': len(df),
            'accepted_count': 0,
            'rejected_count': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'false_positives': 0,  # Chose flashloan when standard was better
            'false_negatives': 0,  # Chose standard when flashloan was better
            'risk_distribution': {'low': 0, 'medium': 0, 'high': 0},
            'strategy_distribution': {'standard': 0, 'flashloan': 0, 'rejected': 0},
            'profits': {
                'total_optimal': 0.0,
                'total_model': 0.0,
                'total_standard': 0.0,
                'total_flashloan': 0.0,
                'risk_adjusted': 0.0
            },
            'positions': []
        }
        
        # For creating profit curve visualization
        cumulative = {
            'optimal': 0.0,
            'model': 0.0,
            'standard': 0.0,
            'flashloan': 0.0
        }
        profit_curves = {'optimal': [], 'model': [], 'standard': [], 'flashloan': []}
        
        # Track individual trades for detailed analysis
        trade_records = []
        
        # Process each opportunity
        for idx, row in df.iterrows():
            opportunity = row.to_dict()
            
            # Apply feature engineering if needed for consistency
            if 'features' not in opportunity and self.feature_engineer:
                opportunity = self.feature_engineer.extract_features(opportunity)
            
            # Check for required profit fields
            if 'standard_profit' not in opportunity or 'flashloan_profit' not in opportunity:
                logging.warning(f"Skipping opportunity {idx}: missing profit fields")
                continue
                
            # Get ground truth (which strategy was actually better)
            standard_profit = opportunity.get('standard_profit', 0)
            flashloan_profit = opportunity.get('flashloan_profit', 0)
            optimal_strategy = 1 if flashloan_profit > standard_profit else 0
            optimal_profit = max(standard_profit, flashloan_profit)
            
            # Get AI recommendation with risk management
            action, confidence, metadata = self.optimize_execution_strategy(opportunity)
            
            # Determine if trade was accepted or rejected
            if metadata.get('decision') == 'rejected':
                results['rejected_count'] += 1
                results['strategy_distribution']['rejected'] += 1
                
                trade_record = {
                    'id': idx,
                    'status': 'rejected',
                    'reason': metadata.get('reason', 'unknown'),
                    'standard_profit': standard_profit,
                    'flashloan_profit': flashloan_profit,
                    'risk_score': opportunity.get('risk_score', 0.5)
                }
                trade_records.append(trade_record)
                continue
                
            # Trade was accepted, track results
            results['accepted_count'] += 1
            
            # Get actual profit based on AI decision
            model_profit = flashloan_profit if action == 1 else standard_profit
            
            # Track which strategy was chosen
            if action == 1:
                results['strategy_distribution']['flashloan'] += 1
            else:
                results['strategy_distribution']['standard'] += 1
                
            # Check if prediction matches optimal
            correct = (action == optimal_strategy)
            if correct:
                results['correct_predictions'] += 1
            else:
                results['incorrect_predictions'] += 1
                # Track error types
                if action == 1 and optimal_strategy == 0:
                    results['false_positives'] += 1  # Wrong flashloan recommendation
                else:
                    results['false_negatives'] += 1  # Missed flashloan opportunity
            
            # Track risk category
            risk_score = opportunity.get('risk_score', 0.5)
            if risk_score < 0.3:
                results['risk_distribution']['low'] += 1
            elif risk_score < 0.7:
                results['risk_distribution']['medium'] += 1
            else:
                results['risk_distribution']['high'] += 1
                
            # Track profits
            results['profits']['total_optimal'] += optimal_profit
            results['profits']['total_model'] += model_profit
            results['profits']['total_standard'] += standard_profit
            results['profits']['total_flashloan'] += flashloan_profit
            
            # Calculate risk-adjusted profit
            risk_adjustment = 1.0 - (risk_score * 0.3)  # Higher risk reduces expected profit
            risk_adjusted_profit = model_profit * risk_adjustment
            results['profits']['risk_adjusted'] += risk_adjusted_profit
            
            # Update profit curves for visualization
            cumulative['optimal'] += optimal_profit
            cumulative['model'] += model_profit
            cumulative['standard'] += standard_profit
            cumulative['flashloan'] += flashloan_profit
            
            profit_curves['optimal'].append(cumulative['optimal'])
            profit_curves['model'].append(cumulative['model'])
            profit_curves['standard'].append(cumulative['standard'])
            profit_curves['flashloan'].append(cumulative['flashloan'])
            
            # Record detailed trade information
            trade_record = {
                'id': idx,
                'status': 'accepted',
                'predicted_strategy': 'flashloan' if action == 1 else 'standard',
                'optimal_strategy': 'flashloan' if optimal_strategy == 1 else 'standard',
                'correct': correct,
                'confidence': confidence,
                'standard_profit': standard_profit,
                'flashloan_profit': flashloan_profit,
                'model_profit': model_profit,
                'optimal_profit': optimal_profit,
                'risk_score': risk_score,
                'risk_adjusted_profit': risk_adjusted_profit,
                'metadata': metadata
            }
            trade_records.append(trade_record)
            
            # Record position data for portfolio analysis
            if 'position_size' in opportunity and 'asset' in opportunity:
                results['positions'].append({
                    'asset': opportunity['asset'],
                    'size': opportunity['position_size'],
                    'profit': model_profit,
                    'strategy': 'flashloan' if action == 1 else 'standard'
                })      
    def _generate_performance_visualization(self, results, profit_curves, trades):
        """
        Generate performance visualization charts
        
        Args:
            results: Performance metrics dictionary
            profit_curves: Cumulative profit data for different strategies
            trades: Detailed trade records
        """
        try:
            # Create directory for reports if needed
            os.makedirs('reports', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = f'reports/trading_performance_{timestamp}'
            
            # Create figure with subplots
            plt.figure(figsize=(16, 12))
            
            # 1. Cumulative Profit Curves
            plt.subplot(2, 2, 1)
            x = range(len(profit_curves['model']))
            plt.plot(x, profit_curves['model'], label='AI Model', linewidth=2)
            plt.plot(x, profit_curves['optimal'], label='Optimal', linewidth=2, linestyle='--')
            plt.plot(x, profit_curves['standard'], label='Standard Only', linewidth=1.5)
            plt.plot(x, profit_curves['flashloan'], label='Flashloan Only', linewidth=1.5)
            plt.title('Cumulative Profit Comparison', fontsize=14)
            plt.xlabel('Trade Number', fontsize=12)
            plt.ylabel('Cumulative Profit', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            
            # 2. Strategy Distribution Pie Chart
            plt.subplot(2, 2, 2)
            strategy_labels = ['Standard', 'Flashloan', 'Rejected']
            strategy_sizes = [
                results['strategy_distribution']['standard'],
                results['strategy_distribution']['flashloan'],
                results['rejected_count']
            ]
            plt.pie(strategy_sizes, labels=strategy_labels, autopct='%1.1f%%', 
                   startangle=90, colors=['#66b3ff', '#99ff99', '#ff9999'])
            plt.title('Trade Strategy Distribution', fontsize=14)
            
            # 3. Confidence vs Profit Scatter
            plt.subplot(2, 2, 3)
            accepted_trades = [t for t in trades if t['status'] == 'accepted']
            if accepted_trades:
                profits = [t['model_profit'] for t in accepted_trades]
                confidences = [t['confidence'] for t in accepted_trades]
                colors = ['green' if t['correct'] else 'red' for t in accepted_trades]
                plt.scatter(confidences, profits, c=colors, alpha=0.7)
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                plt.title('Trade Confidence vs Profit', fontsize=14)
                plt.xlabel('Model Confidence', fontsize=12)
                plt.ylabel('Profit', fontsize=12)
                plt.grid(True, alpha=0.3)
            
            # 4. Performance Metrics Bar Chart
            plt.subplot(2, 2, 4)
            metrics_to_show = ['accuracy', 'opportunity_capture', 'risk_adjusted_ratio', 
                              'flashloan_preference', 'rejection_rate']
            metric_labels = ['Accuracy', 'Opportunity\nCapture', 'Risk-Adjusted\nRatio', 
                           'Flashloan\nPreference', 'Rejection\nRate']
            metric_values = [results.get(m, 0) for m in metrics_to_show]
            bars = plt.bar(metric_labels, metric_values, color='#66b3ff')
            plt.title('Key Performance Metrics', fontsize=14)
            plt.ylim(0, 1.2)
            plt.grid(axis='y', alpha=0.3)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'{report_path}.png', dpi=300)
            plt.close()
            logging.info(f"Performance visualization saved to {report_path}.png")
            
            # Generate JSON report with detailed metrics
            with open(f'{report_path}.json', 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error generating visualization: {str(e)}")
        
        # Calculate overall performance metrics
        evaluated_count = results['accepted_count']
        if evaluated_count > 0:
            # Accuracy of predictions on accepted trades
            results['accuracy'] = results['correct_predictions'] / evaluated_count
            
            # Economic impact metrics
            standard_baseline = results['profits']['total_standard']
            if standard_baseline != 0:
                results['improvement_over_standard'] = (results['profits']['total_model'] - standard_baseline) / abs(standard_baseline)
            else:
                results['improvement_over_standard'] = 0.0
                
            # Opportunity capture rate
            results['opportunity_capture'] = results['profits']['total_model'] / results['profits']['total_optimal'] if results['profits']['total_optimal'] > 0 else 0.0
            
            # Risk-adjusted performance
            results['risk_adjusted_ratio'] = results['profits']['risk_adjusted'] / results['profits']['total_model'] if results['profits']['total_model'] > 0 else 0.0
            
            # Rejection rate
            total = results['accepted_count'] + results['rejected_count']
            results['rejection_rate'] = results['rejected_count'] / total if total > 0 else 0.0
            
            # Strategy preference
            results['flashloan_preference'] = results['strategy_distribution']['flashloan'] / evaluated_count if evaluated_count > 0 else 0.0
            
            # Decision confidence
            results['avg_confidence'] = sum(t['confidence'] for t in trade_records if t['status'] == 'accepted') / evaluated_count if evaluated_count > 0 else 0.0
        
        # Calculate Sharpe ratio if we have enough data points
        if len(trade_records) > 5:
            profits = [t['model_profit'] for t in trade_records if t['status'] == 'accepted']
            mean_profit = np.mean(profits) if profits else 0
            std_profit = np.std(profits) if profits else 1
            results['sharpe_ratio'] = mean_profit / std_profit if std_profit > 0 else 0
        else:
            results['sharpe_ratio'] = 0
            
        # Generate visualization
        try:
            self._generate_performance_visualization(results, profit_curves, trade_records)
        except Exception as e:
            logging.warning(f"Failed to generate visualization: {str(e)}")
            
        # Log performance summary
        if evaluated_count > 0:
            logging.info(f"Performance summary: {results['accuracy']:.2f} accuracy, " +
                        f"{results['improvement_over_standard']:.2f}x improvement over standard, " + 
                        f"${results['profits']['total_model']:.2f} total profit")
            
        return {
            'metrics': results,
            'profit_curves': profit_curves,
            'trades': trade_records
        }
        
        results['total_standard_profit'] = standard_profit
        results['total_ai_profit'] = ai_profit
        
        logging.info(f"Model accuracy: {results['accuracy']:.4f}")
        logging.info(f"Profit improvement: {results['profit_improvement']:.4f}")
        
        # Generate analysis plots
        self._generate_performance_plots(results)
        
        return results
    
    def _generate_performance_plots(self, results):
        """
        Generate plots to visualize model performance
        
        Args:
            results: Performance analysis results
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = 'results/ai_performance'
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Convert predictions to DataFrame for easier manipulation
            df_pred = pd.DataFrame(results['predictions'])
            
            # Plot accuracy by confidence level
            plt.figure(figsize=(10, 6))
            
            # Create confidence bins
            df_pred['confidence_bin'] = pd.cut(df_pred['confidence'], bins=10)
            accuracy_by_confidence = df_pred.groupby('confidence_bin')['correct'].mean()
            
            plt.bar(range(len(accuracy_by_confidence)), accuracy_by_confidence.values)
            plt.xticks(range(len(accuracy_by_confidence)), 
                      [f"{b.left:.1f}-{b.right:.1f}" for b in accuracy_by_confidence.index], 
                      rotation=45)
            
            plt.title('Prediction Accuracy by Confidence Level')
            plt.xlabel('Confidence Range')
            plt.ylabel('Accuracy')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/accuracy_by_confidence_{timestamp}.png")
            
            # Plot strategy distribution
            plt.figure(figsize=(8, 6))
            strategy_counts = df_pred['ai_recommendation'].value_counts()
            plt.pie(strategy_counts, labels=strategy_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=['#ff9999','#66b3ff'])
            plt.title('AI Strategy Recommendations')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/strategy_distribution_{timestamp}.png")
            
            # Plot profit comparison
            plt.figure(figsize=(10, 6))
            
            # Group by opportunity type if available
            if 'type' in df_pred.columns:
                opportunity_types = df_pred['type'].unique()
                
                # Calculate profits by type
                profit_by_type = {}
                for opp_type in opportunity_types:
                    type_mask = df_pred['type'] == opp_type
                    profit_by_type[opp_type] = {
                        'ai': df_pred[type_mask & (df_pred['ai_recommendation'] == 'flashloan')]['profit_difference'].sum(),
                        'standard': 0
                    }
                
                # Plot as grouped bar chart
                labels = list(profit_by_type.keys())
                ai_profits = [profit_by_type[t]['ai'] for t in labels]
                standard_profits = [profit_by_type[t]['standard'] for t in labels]
                
                x = np.arange(len(labels))
                width = 0.35
                
                fig, ax = plt.subplots(figsize=(10, 6))
                rects1 = ax.bar(x - width/2, standard_profits, width, label='Standard')
                rects2 = ax.bar(x + width/2, ai_profits, width, label='AI Optimized')
                
                ax.set_title('Profit Comparison by Opportunity Type')
                ax.set_xlabel('Opportunity Type')
                ax.set_ylabel('Profit')
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.legend()
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/profit_by_type_{timestamp}.png")
            
            # Otherwise plot overall profit comparison
            else:
                profits = [results['total_standard_profit'], results['total_ai_profit']]
                labels = ['Standard', 'AI Optimized']
                
                plt.bar(labels, profits, color=['#ff9999','#66b3ff'])
                plt.title('Total Profit Comparison')
                plt.ylabel('Total Profit')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/profit_comparison_{timestamp}.png")
            
            logging.info(f"Performance plots saved to {output_dir}")
            
        except Exception as e:
            logging.error(f"Error generating performance plots: {str(e)}")
    
    def optimize_portfolio_allocation(self, opportunities, total_capital):
        """
        Optimize capital allocation across multiple trading opportunities
        
        Args:
            opportunities: List of trading opportunities
            total_capital: Total available capital
            
        Returns:
            List of opportunities with optimized allocations
        """
        if not opportunities:
            return []
        
        # Score each opportunity based on expected profit and AI confidence
        scored_opps = []
        
        for opportunity in opportunities:
            # Get AI recommendation
            action, confidence = self.optimize_execution_strategy(opportunity)
            
            # Calculate opportunity score based on profit and confidence
            expected_profit = opportunity.get('expected_profit', 0)
            risk_score = opportunity.get('risk_score', 0.5)
            
            # Higher score for higher profit, higher confidence, lower risk
            score = expected_profit * confidence / (risk_score + 0.1)
            
            scored_opps.append({
                'opportunity': opportunity,
                'score': score,
                'action': action,
                'confidence': confidence
            })
        
        # Sort by score
        scored_opps.sort(key=lambda x: x['score'], reverse=True)
        
        # Allocate capital proportionally to scores
        total_score = sum(opp['score'] for opp in scored_opps)
        
        enhanced_opportunities = []
        
        if total_score > 0:
            for opp in scored_opps:
                # Original opportunity
                opportunity = opp['opportunity'].copy()
                
                # Allocate capital proportionally to score
                allocation = (opp['score'] / total_score) * total_capital
                
                # Add AI enhancements
                opportunity['ai_score'] = float(opp['score'])
                opportunity['ai_recommended_strategy'] = 'flashloan' if opp['action'] == 1 else 'standard'
                opportunity['ai_confidence'] = float(opp['confidence'])
                opportunity['ai_allocation'] = float(allocation)
                opportunity['execution_strategy'] = opportunity['ai_recommended_strategy']
                
                enhanced_opportunities.append(opportunity)
        
        logging.info(f"Optimized capital allocation across {len(enhanced_opportunities)} opportunities")
        
        return enhanced_opportunities
