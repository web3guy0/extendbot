"""
Strategy Manager
Coordinates trading strategies - SWING ONLY for sustainable profits
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import asyncio

# Import strategies - SWING ONLY (scalping removed - fees kill profits)
from app.strategies.rule_based.swing_strategy import SwingStrategy

# Import adaptive components for BTC correlation
from app.strategies.adaptive import MultiAssetCorrelationAnalyzer

logger = logging.getLogger(__name__)


class StrategyManager:
    """
    Manages trading strategy execution
    - SWING ONLY for sustainable profits
    - Scalping removed (fees kill profits)
    - Tracks performance
    """
    
    def __init__(self, symbol: str, config: Dict[str, Any] = None):
        """
        Initialize strategy manager
        
        Args:
            symbol: Trading symbol
            config: Strategy configuration
        """
        self.symbol = symbol
        self.config = config if config is not None else {}
        
        # SWING ONLY - Scalping disabled (fees kill profits)
        # WHY SWING ONLY:
        # - Scalping: 0.4% target - 0.04% fees = 10% of profit gone
        # - Swing: 2% target - 0.04% fees = 2% of profit gone
        # - Fewer trades = fewer fees = more $$ in YOUR pocket
        self.strategies = {
            'swing': SwingStrategy(symbol, config),
        }
        self.strategy_stats = {
            'swing': {'signals': 0, 'trades': 0, 'execution_rate': 0},
        }
        
        logger.info(f"🌟 Strategy Manager initialized for {symbol}")
        logger.info(f"   • SwingStrategy: Adaptive regime, SMC, MTF, OrderFlow")
        logger.info(f"   💰 SWING ONLY - Quality trades for real profits")
        
        # BTC Correlation analyzer (for altcoins)
        self.correlation_analyzer = MultiAssetCorrelationAnalyzer() if symbol != 'BTC' else None
        self.btc_candles: List[Dict] = []
        
        # Execution mode
        self.execution_mode = 'first_signal'
        self.last_strategy_used = None
        
        logger.info(f"   Active Strategies: {len(self.strategies)}")
        logger.info(f"   BTC Correlation: {'Enabled' if self.correlation_analyzer else 'Disabled (is BTC)'}")
    
    async def generate_signal(self, market_data: Dict[str, Any],
                             account_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate signal from all strategies
        Returns first valid signal found
        
        Args:
            market_data: Current market state
            account_state: Current account state
            
        Returns:
            Trading signal or None
        """
        # Check if already in position
        positions = account_state.get('positions', [])
        if any(p['symbol'] == self.symbol for p in positions):
            return None  # Already in position, don't generate new signals
        
        # Run all strategies in parallel
        signals = await self._run_all_strategies(market_data, account_state)
        
        # Filter valid signals
        valid_signals = [s for s in signals if s is not None]
        
        if not valid_signals:
            return None
        
        # Select signal based on execution mode - SWING ONLY
        if self.execution_mode == 'first_signal':
            # Just take the first valid swing signal
            selected_signal = valid_signals[0]
            
        elif self.execution_mode == 'round_robin':
            # Rotate between strategies
            selected_signal = self._round_robin_select(valid_signals)
            
        elif self.execution_mode == 'priority':
            # Priority order: world_class first
            selected_signal = self._priority_select(valid_signals)
            
        else:
            selected_signal = valid_signals[0]
        
        # BTC Correlation check for altcoins
        if self.correlation_analyzer and self.btc_candles and selected_signal:
            asset_candles = market_data.get('candles', [])
            direction = selected_signal.get('direction')
            
            if asset_candles and direction:
                correlation_result = self.correlation_analyzer.analyze(
                    self.symbol,
                    asset_candles,
                    self.btc_candles,
                    direction
                )
                
                if not correlation_result.should_trade:
                    logger.info(f"⚠️ Signal filtered by BTC correlation: {correlation_result.notes}")
                    return None
                
                # Adjust position size based on correlation confidence
                size_adj = self.correlation_analyzer.get_position_size_adjustment(correlation_result)
                selected_signal['position_size_pct'] *= float(size_adj)
                selected_signal['correlation_analysis'] = {
                    'correlation': float(correlation_result.correlation),
                    'state': correlation_result.state.value,
                    'relative_strength': correlation_result.relative_strength.value,
                    'btc_trend': correlation_result.btc_trend,
                    'confidence': float(correlation_result.confidence),
                }
        
        # Update statistics
        strategy_name = selected_signal['strategy']
        if strategy_name in self.strategy_stats:
            self.strategy_stats[strategy_name]['signals'] += 1
        
        self.last_strategy_used = strategy_name
        
        logger.info(f"📊 Signal selected: {strategy_name} ({len(valid_signals)} strategies triggered)")
        
        return selected_signal
    
    def update_btc_candles(self, candles: List[Dict]):
        """
        Update BTC candles for correlation analysis.
        
        Args:
            candles: BTC price candles
        """
        self.btc_candles = candles
    
    async def _run_all_strategies(self, market_data: Dict[str, Any],
                                   account_state: Dict[str, Any]) -> List[Optional[Dict[str, Any]]]:
        """
        Run all strategies in parallel
        
        Args:
            market_data: Market data
            account_state: Account state
            
        Returns:
            List of signals (may contain None)
        """
        tasks = []
        strategy_names = []
        
        for name, strategy in self.strategies.items():
            task = strategy.generate_signal(market_data, account_state)
            tasks.append(task)
            strategy_names.append(name)
        
        # Execute all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        signals = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Strategy {strategy_names[i]} error: {result}")
                signals.append(None)
            else:
                signals.append(result)
        
        return signals
    
    def _round_robin_select(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Round-robin selection between strategies
        NOTE: With SWING ONLY mode, this just returns the first signal.
        Kept for backwards compatibility if more strategies are added.
        
        Args:
            signals: List of valid signals
            
        Returns:
            Selected signal
        """
        if not signals:
            return None
        
        # With only swing strategy, just return first signal
        if len(self.strategies) == 1:
            return signals[0]
        
        if not self.last_strategy_used:
            return signals[0]
        
        # Find next strategy in rotation based on registered strategies
        strategy_names = list(self.strategies.keys())
        
        try:
            last_idx = strategy_names.index(self.last_strategy_used)
            next_idx = (last_idx + 1) % len(strategy_names)
            next_strategies = strategy_names[next_idx:] + strategy_names[:next_idx]
            
            # Find first signal from next strategies
            for strategy_name in next_strategies:
                for signal in signals:
                    if signal['strategy'] == strategy_name:
                        return signal
        except (ValueError, IndexError):
            pass
        
        return signals[0]
    
    def _priority_select(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Priority-based selection - returns first/best signal
        Only SwingStrategy is active.
        
        Args:
            signals: List of valid signals
            
        Returns:
            Highest priority signal
        """
        # SwingStrategy is the only active strategy
        return signals[0]
    
    def record_trade_execution(self, signal: Dict[str, Any], result: Dict[str, Any]):
        """
        Record trade execution for a strategy
        
        Args:
            signal: Original signal
            result: Execution result
        """
        # Defensive: ensure signal and result are dicts
        if not isinstance(signal, dict):
            logger.warning(f"record_trade_execution: signal is not a dict: {type(signal)}")
            return
        if not isinstance(result, dict):
            logger.warning(f"record_trade_execution: result is not a dict: {type(result)}")
            return
        
        strategy_name = signal.get('strategy')
        
        if strategy_name in self.strategy_stats:
            self.strategy_stats[strategy_name]['trades'] += 1
        
        # Pass to individual strategy
        strategy_key = self._get_strategy_key(strategy_name)
        if strategy_key and strategy_key in self.strategies:
            self.strategies[strategy_key].record_trade_execution(signal, result)
    
    def revalidate_signal(self, signal: Dict[str, Any], current_price) -> bool:
        """
        Revalidate signal before execution
        
        Args:
            signal: Original signal
            current_price: Current market price
            
        Returns:
            True if signal still valid
        """
        strategy_name = signal.get('strategy')
        strategy_key = self._get_strategy_key(strategy_name)
        
        if strategy_key and strategy_key in self.strategies:
            strategy = self.strategies[strategy_key]
            # Check if strategy has revalidation method
            if hasattr(strategy, 'revalidate_signal'):
                return strategy.revalidate_signal(signal, current_price)
        
        # Default: accept signal if no revalidation available
        return True
    
    def _get_strategy_key(self, strategy_class_name: str) -> Optional[str]:
        """
        Map strategy class name to internal key
        
        Args:
            strategy_class_name: Class name
            
        Returns:
            Internal key
        """
        mapping = {
            'SwingStrategy': 'swing',
        }
        return mapping.get(strategy_class_name)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregate statistics for all strategies
        
        Returns:
            Statistics dictionary
        """
        total_signals = sum(s['signals'] for s in self.strategy_stats.values())
        total_trades = sum(s['trades'] for s in self.strategy_stats.values())
        
        stats = {
            'manager': 'StrategyManager',
            'symbol': self.symbol,
            'total_signals': total_signals,
            'total_trades': total_trades,
            'execution_rate': total_trades / total_signals if total_signals > 0 else 0,
            'execution_mode': self.execution_mode,
            'last_strategy_used': self.last_strategy_used,
            'strategy_breakdown': {}
        }
        
        # Add per-strategy stats
        for name, strategy in self.strategies.items():
            strategy_stats = strategy.get_statistics()
            stats['strategy_breakdown'][name] = {
                'signals': self.strategy_stats[name]['signals'],
                'trades': self.strategy_stats[name]['trades'],
                'execution_rate': (self.strategy_stats[name]['trades'] / 
                                  self.strategy_stats[name]['signals'] 
                                  if self.strategy_stats[name]['signals'] > 0 else 0)
            }
        
        return stats
    
    def log_statistics(self):
        """Log current statistics"""
        stats = self.get_statistics()
        
        logger.info(f"📊 Strategy Manager Statistics:")
        logger.info(f"   Total Signals: {stats['total_signals']}")
        logger.info(f"   Total Trades: {stats['total_trades']}")
        logger.info(f"   Execution Rate: {stats['execution_rate']:.1%}")
        logger.info(f"   Last Used: {stats['last_strategy_used']}")
        
        for name, breakdown in stats['strategy_breakdown'].items():
            logger.info(f"   {name}: {breakdown['signals']} signals, "
                       f"{breakdown['trades']} trades ({breakdown['execution_rate']:.1%})")
