#!/usr/bin/env python3
"""
Backtesting Framework
=====================

Tests trading strategies on historical data before live deployment.
Uses the same strategy classes as the live bot for consistency.

Features:
- Historical data loading (candles from exchange or CSV)
- Strategy simulation with realistic fills
- Performance metrics (Sharpe, Sortino, max drawdown, etc.)
- Trade-by-trade analysis
- Visualization support
"""

import asyncio
import logging
from decimal import Decimal
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a simulated trade"""
    id: int
    symbol: str
    side: str  # 'long' or 'short'
    entry_time: datetime
    entry_price: Decimal
    size: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    
    # Exit info (filled when trade closes)
    exit_time: Optional[datetime] = None
    exit_price: Optional[Decimal] = None
    exit_reason: Optional[str] = None  # 'tp_hit', 'sl_hit', 'signal', 'eod'
    
    # P&L
    pnl: Decimal = Decimal('0')
    pnl_pct: Decimal = Decimal('0')
    
    # Strategy info
    strategy_name: str = ''
    signal_score: int = 0


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    # Configuration
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_balance: Decimal
    leverage: int
    
    # Performance metrics
    final_balance: Decimal = Decimal('0')
    total_pnl: Decimal = Decimal('0')
    total_pnl_pct: Decimal = Decimal('0')
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: Decimal = Decimal('0')
    
    # Risk metrics
    max_drawdown: Decimal = Decimal('0')
    max_drawdown_pct: Decimal = Decimal('0')
    avg_win: Decimal = Decimal('0')
    avg_loss: Decimal = Decimal('0')
    profit_factor: Decimal = Decimal('0')
    
    # Advanced metrics
    sharpe_ratio: Decimal = Decimal('0')
    sortino_ratio: Decimal = Decimal('0')
    calmar_ratio: Decimal = Decimal('0')
    
    # Time analysis
    avg_trade_duration: timedelta = field(default_factory=lambda: timedelta(0))
    max_trade_duration: timedelta = field(default_factory=lambda: timedelta(0))
    
    # Trade list
    trades: List[BacktestTrade] = field(default_factory=list)
    
    # Daily returns (for Sharpe/Sortino)
    daily_returns: List[Decimal] = field(default_factory=list)
    
    # Equity curve
    equity_curve: List[Tuple[datetime, Decimal]] = field(default_factory=list)


class Backtester:
    """
    Backtests trading strategies on historical data
    
    Usage:
        backtester = Backtester(strategy, initial_balance=10000, leverage=5)
        result = await backtester.run(candles, symbol='SOL')
        print(result.total_pnl, result.win_rate)
    """
    
    def __init__(
        self,
        strategy,  # StrategyManager or individual strategy
        initial_balance: float = 10000.0,
        leverage: int = 5,
        commission_rate: float = 0.0004,  # 0.04% taker fee
        slippage_pct: float = 0.01  # 0.01% slippage simulation
    ):
        """
        Initialize backtester
        
        Args:
            strategy: Strategy instance to test
            initial_balance: Starting account balance
            leverage: Trading leverage
            commission_rate: Commission per trade (fraction)
            slippage_pct: Slippage simulation (fraction)
        """
        self.strategy = strategy
        self.initial_balance = Decimal(str(initial_balance))
        self.leverage = leverage
        self.commission_rate = Decimal(str(commission_rate))
        self.slippage_pct = Decimal(str(slippage_pct))
        
        # State during backtest
        self.balance = self.initial_balance
        self.position = None  # Current open position (BacktestTrade)
        self.trades: List[BacktestTrade] = []
        self.trade_id = 0
        
        # Equity tracking
        self.equity_curve: List[Tuple[datetime, Decimal]] = []
        self.peak_equity = self.initial_balance
        self.max_drawdown = Decimal('0')
        
        logger.info(f"🧪 Backtester initialized")
        logger.info(f"   Initial Balance: ${initial_balance:,.2f}")
        logger.info(f"   Leverage: {leverage}x")
        logger.info(f"   Commission: {commission_rate*100:.3f}%")
    
    async def run(
        self,
        candles: List[Dict[str, Any]],
        symbol: str = 'SOL',
        start_idx: int = 100,  # Start after enough history for indicators
        end_idx: Optional[int] = None
    ) -> BacktestResult:
        """
        Run backtest on historical candles
        
        Args:
            candles: List of OHLCV candles
            symbol: Trading symbol
            start_idx: Start index (skip early candles for indicator warmup)
            end_idx: End index (None = end of data)
        
        Returns:
            BacktestResult with performance metrics
        """
        if not candles:
            raise ValueError("No candles provided")
        
        end_idx = end_idx or len(candles)
        
        logger.info(f"🧪 Starting backtest for {symbol}")
        logger.info(f"   Candles: {len(candles)} ({start_idx} to {end_idx})")
        
        # Reset state
        self.balance = self.initial_balance
        self.position = None
        self.trades = []
        self.trade_id = 0
        self.equity_curve = []
        self.peak_equity = self.initial_balance
        self.max_drawdown = Decimal('0')
        
        # Simulate bar-by-bar
        for i in range(start_idx, end_idx):
            current_candle = candles[i]
            historical_candles = candles[:i+1]  # All candles up to current
            
            await self._process_bar(current_candle, historical_candles, symbol)
        
        # Close any remaining position at end
        if self.position:
            final_candle = candles[end_idx - 1]
            self._close_position(
                exit_time=datetime.fromtimestamp(final_candle['time'] / 1000, tz=timezone.utc),
                exit_price=Decimal(str(final_candle['close'])),
                reason='eod'  # End of data
            )
        
        # Calculate results
        return self._calculate_results(symbol, candles, start_idx, end_idx)
    
    async def _process_bar(
        self,
        candle: Dict[str, Any],
        historical: List[Dict[str, Any]],
        symbol: str
    ):
        """Process a single bar (candle)"""
        bar_time = datetime.fromtimestamp(candle['time'] / 1000, tz=timezone.utc)
        bar_high = Decimal(str(candle['high']))
        bar_low = Decimal(str(candle['low']))
        bar_close = Decimal(str(candle['close']))
        
        # Check existing position for SL/TP hits
        if self.position:
            self._check_stops(bar_time, bar_high, bar_low)
        
        # Skip signal generation if already in position
        if self.position:
            self._update_equity(bar_time, bar_close)
            return
        
        # Build market data for strategy
        market_data = {
            'symbol': symbol,
            'price': float(bar_close),
            'candles': historical[-150:],  # Last 150 candles
            'time': bar_time.isoformat()
        }
        
        # Build mock account state
        account_state = {
            'account_value': float(self.balance),
            'positions': [],
            'margin_used': 0
        }
        
        # Generate signal
        try:
            signal = await self.strategy.generate_signal(market_data, account_state)
            
            if signal:
                self._open_position(signal, bar_time, bar_close)
        except Exception as e:
            logger.debug(f"Signal generation error: {e}")
        
        # Update equity curve
        self._update_equity(bar_time, bar_close)
    
    def _open_position(
        self,
        signal: Dict[str, Any],
        entry_time: datetime,
        current_price: Decimal
    ):
        """Open a new position based on signal"""
        self.trade_id += 1
        
        # Apply slippage
        side = 'long' if signal['side'].lower() == 'buy' else 'short'
        if side == 'long':
            entry_price = current_price * (1 + self.slippage_pct / 100)
        else:
            entry_price = current_price * (1 - self.slippage_pct / 100)
        
        # Calculate position size
        size = Decimal(str(signal.get('size', 1.0)))
        
        # Commission on entry
        entry_value = size * entry_price
        commission = entry_value * self.commission_rate
        self.balance -= commission
        
        # Create position
        self.position = BacktestTrade(
            id=self.trade_id,
            symbol=signal['symbol'],
            side=side,
            entry_time=entry_time,
            entry_price=entry_price,
            size=size,
            stop_loss=Decimal(str(signal['stop_loss'])),
            take_profit=Decimal(str(signal['take_profit'])),
            strategy_name=signal.get('strategy', 'unknown'),
            signal_score=signal.get('score', 0)
        )
        
        logger.debug(f"📈 Opened {side.upper()} @ ${entry_price:.4f}")
    
    def _check_stops(self, bar_time: datetime, bar_high: Decimal, bar_low: Decimal):
        """Check if SL or TP was hit during this bar"""
        if not self.position:
            return
        
        if self.position.side == 'long':
            # Long: SL hit if low <= SL, TP hit if high >= TP
            if bar_low <= self.position.stop_loss:
                self._close_position(bar_time, self.position.stop_loss, 'sl_hit')
            elif bar_high >= self.position.take_profit:
                self._close_position(bar_time, self.position.take_profit, 'tp_hit')
        else:
            # Short: SL hit if high >= SL, TP hit if low <= TP
            if bar_high >= self.position.stop_loss:
                self._close_position(bar_time, self.position.stop_loss, 'sl_hit')
            elif bar_low <= self.position.take_profit:
                self._close_position(bar_time, self.position.take_profit, 'tp_hit')
    
    def _close_position(
        self,
        exit_time: datetime,
        exit_price: Decimal,
        reason: str
    ):
        """Close the current position"""
        if not self.position:
            return
        
        # Apply slippage on exit
        if self.position.side == 'long':
            exit_price = exit_price * (1 - self.slippage_pct / 100)
        else:
            exit_price = exit_price * (1 + self.slippage_pct / 100)
        
        # Calculate P&L
        if self.position.side == 'long':
            pnl = (exit_price - self.position.entry_price) * self.position.size * self.leverage
        else:
            pnl = (self.position.entry_price - exit_price) * self.position.size * self.leverage
        
        pnl_pct = (pnl / (self.position.entry_price * self.position.size)) * 100
        
        # Commission on exit
        exit_value = self.position.size * exit_price
        commission = exit_value * self.commission_rate
        pnl -= commission
        
        # Update position
        self.position.exit_time = exit_time
        self.position.exit_price = exit_price
        self.position.exit_reason = reason
        self.position.pnl = pnl
        self.position.pnl_pct = pnl_pct
        
        # Update balance
        self.balance += pnl
        
        # Store trade
        self.trades.append(self.position)
        
        emoji = "✅" if pnl > 0 else "❌"
        logger.debug(f"{emoji} Closed {self.position.side.upper()} @ ${exit_price:.4f} | P&L: ${pnl:+.2f} ({reason})")
        
        # Clear position
        self.position = None
    
    def _update_equity(self, timestamp: datetime, current_price: Decimal):
        """Update equity curve and drawdown"""
        # Calculate current equity
        if self.position:
            if self.position.side == 'long':
                unrealized = (current_price - self.position.entry_price) * self.position.size * self.leverage
            else:
                unrealized = (self.position.entry_price - current_price) * self.position.size * self.leverage
            equity = self.balance + unrealized
        else:
            equity = self.balance
        
        # Track peak and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        drawdown = (self.peak_equity - equity) / self.peak_equity * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        # Store equity point
        self.equity_curve.append((timestamp, equity))
    
    def _calculate_results(
        self,
        symbol: str,
        candles: List[Dict],
        start_idx: int,
        end_idx: int
    ) -> BacktestResult:
        """Calculate final backtest metrics"""
        start_time = datetime.fromtimestamp(candles[start_idx]['time'] / 1000, tz=timezone.utc)
        end_time = datetime.fromtimestamp(candles[end_idx-1]['time'] / 1000, tz=timezone.utc)
        
        result = BacktestResult(
            symbol=symbol,
            start_date=start_time,
            end_date=end_time,
            initial_balance=self.initial_balance,
            leverage=self.leverage,
            final_balance=self.balance,
            total_pnl=self.balance - self.initial_balance,
            total_pnl_pct=(self.balance - self.initial_balance) / self.initial_balance * 100,
            total_trades=len(self.trades),
            max_drawdown=self.peak_equity - min(eq for _, eq in self.equity_curve) if self.equity_curve else Decimal('0'),
            max_drawdown_pct=self.max_drawdown,
            trades=self.trades,
            equity_curve=self.equity_curve
        )
        
        if self.trades:
            # Win/loss stats
            winners = [t for t in self.trades if t.pnl > 0]
            losers = [t for t in self.trades if t.pnl <= 0]
            
            result.winning_trades = len(winners)
            result.losing_trades = len(losers)
            result.win_rate = Decimal(str(len(winners) / len(self.trades) * 100))
            
            if winners:
                result.avg_win = sum(t.pnl for t in winners) / len(winners)
            if losers:
                result.avg_loss = sum(t.pnl for t in losers) / len(losers)
            
            # Profit factor
            gross_profit = sum(t.pnl for t in winners) if winners else Decimal('0')
            gross_loss = abs(sum(t.pnl for t in losers)) if losers else Decimal('1')
            result.profit_factor = gross_profit / gross_loss if gross_loss != 0 else Decimal('0')
            
            # Duration stats
            durations = [(t.exit_time - t.entry_time) for t in self.trades if t.exit_time]
            if durations:
                result.avg_trade_duration = sum(durations, timedelta()) / len(durations)
                result.max_trade_duration = max(durations)
            
            # Calculate daily returns for Sharpe/Sortino
            result.daily_returns = self._calculate_daily_returns()
            
            if result.daily_returns and len(result.daily_returns) > 1:
                # Sharpe Ratio (assuming 0% risk-free rate)
                import statistics
                returns = [float(r) for r in result.daily_returns]
                avg_return = statistics.mean(returns)
                std_return = statistics.stdev(returns)
                if std_return > 0:
                    result.sharpe_ratio = Decimal(str(avg_return / std_return * (365 ** 0.5)))
                
                # Sortino Ratio (downside deviation only)
                downside = [r for r in returns if r < 0]
                if downside:
                    downside_std = statistics.stdev(downside) if len(downside) > 1 else abs(downside[0])
                    if downside_std > 0:
                        result.sortino_ratio = Decimal(str(avg_return / downside_std * (365 ** 0.5)))
                
                # Calmar Ratio
                if float(result.max_drawdown_pct) > 0:
                    annual_return = float(result.total_pnl_pct) * 365 / max(1, (end_time - start_time).days)
                    result.calmar_ratio = Decimal(str(annual_return / float(result.max_drawdown_pct)))
        
        return result
    
    def _calculate_daily_returns(self) -> List[Decimal]:
        """Calculate daily returns from equity curve"""
        if len(self.equity_curve) < 2:
            return []
        
        daily_returns = []
        prev_equity = self.equity_curve[0][1]
        prev_date = self.equity_curve[0][0].date()
        
        for timestamp, equity in self.equity_curve[1:]:
            if timestamp.date() != prev_date:
                # New day - calculate return
                daily_return = (equity - prev_equity) / prev_equity * 100
                daily_returns.append(daily_return)
                prev_equity = equity
                prev_date = timestamp.date()
        
        return daily_returns
    
    def print_summary(self, result: BacktestResult):
        """Print a formatted summary of backtest results"""
        print("\n" + "=" * 60)
        print("🧪 BACKTEST RESULTS")
        print("=" * 60)
        print(f"Symbol: {result.symbol}")
        print(f"Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
        print(f"Leverage: {result.leverage}x")
        print()
        
        print("💰 PERFORMANCE")
        print("-" * 40)
        print(f"Initial Balance:  ${float(result.initial_balance):,.2f}")
        print(f"Final Balance:    ${float(result.final_balance):,.2f}")
        print(f"Total P&L:        ${float(result.total_pnl):+,.2f} ({float(result.total_pnl_pct):+.2f}%)")
        print()
        
        print("📊 TRADE STATISTICS")
        print("-" * 40)
        print(f"Total Trades:     {result.total_trades}")
        print(f"Winning Trades:   {result.winning_trades}")
        print(f"Losing Trades:    {result.losing_trades}")
        print(f"Win Rate:         {float(result.win_rate):.1f}%")
        print(f"Avg Win:          ${float(result.avg_win):+.2f}")
        print(f"Avg Loss:         ${float(result.avg_loss):.2f}")
        print(f"Profit Factor:    {float(result.profit_factor):.2f}")
        print()
        
        print("📉 RISK METRICS")
        print("-" * 40)
        print(f"Max Drawdown:     ${float(result.max_drawdown):,.2f} ({float(result.max_drawdown_pct):.2f}%)")
        print(f"Sharpe Ratio:     {float(result.sharpe_ratio):.2f}")
        print(f"Sortino Ratio:    {float(result.sortino_ratio):.2f}")
        print(f"Calmar Ratio:     {float(result.calmar_ratio):.2f}")
        print()
        
        print("⏱️ TIME ANALYSIS")
        print("-" * 40)
        print(f"Avg Trade Duration: {result.avg_trade_duration}")
        print(f"Max Trade Duration: {result.max_trade_duration}")
        print("=" * 60)


async def run_backtest_example():
    """Example usage of the backtester"""
    from app.strategies.strategy_manager import StrategyManager
    from app.hl.hl_client import HyperLiquidClient
    import os
    
    # Load credentials
    account_address = os.getenv('ACCOUNT_ADDRESS')
    api_secret = os.getenv('API_SECRET')
    
    # Initialize client
    client = HyperLiquidClient(account_address, api_secret, api_secret, testnet=True)
    
    # Fetch historical candles
    print("📥 Fetching historical data...")
    candles = client.get_candles('SOL', '1m', 10000)  # ~1 week of 1m candles
    print(f"   Loaded {len(candles)} candles")
    
    # Initialize strategy
    strategy = StrategyManager('SOL')
    
    # Run backtest
    backtester = Backtester(
        strategy=strategy,
        initial_balance=10000,
        leverage=5
    )
    
    result = await backtester.run(candles, symbol='SOL')
    
    # Print results
    backtester.print_summary(result)
    
    return result


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    asyncio.run(run_backtest_example())
