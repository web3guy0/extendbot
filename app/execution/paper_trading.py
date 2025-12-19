"""
Paper Trading Mode - Simulate Trades Without Real Execution

Allows strategy validation without risking real money.
Tracks simulated positions, P&L, and performance metrics.
"""

import os
import logging
from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import deque
import json

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """Simulated position."""
    symbol: str
    side: str  # 'long' or 'short'
    size: Decimal
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    entry_time: datetime
    leverage: int = 5  # Store leverage for margin calculation
    unrealized_pnl: Decimal = Decimal('0')
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    realized_pnl: Optional[Decimal] = None


@dataclass
class PaperTrade:
    """Completed paper trade record."""
    symbol: str
    side: str
    size: Decimal
    entry_price: Decimal
    exit_price: Decimal
    entry_time: datetime
    exit_time: datetime
    exit_reason: str
    pnl: Decimal
    pnl_pct: Decimal
    duration_seconds: int


class PaperTradingEngine:
    """
    Paper Trading Engine
    
    Simulates real trading without execution:
    - Tracks virtual positions
    - Simulates SL/TP hits
    - Calculates P&L
    - Records trade history
    - Provides performance metrics
    """
    
    def __init__(self, initial_balance: Decimal = Decimal('1000')):
        """
        Initialize paper trading engine.
        
        Args:
            initial_balance: Starting virtual balance
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        
        # Active positions (symbol -> PaperPosition)
        self.positions: Dict[str, PaperPosition] = {}
        
        # Trade history
        self.trade_history: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = Decimal('0')
        self.max_drawdown = Decimal('0')
        self.peak_equity = initial_balance
        
        # Session stats
        self.session_start = datetime.now(timezone.utc)
        self.session_pnl = Decimal('0')
        
        logger.info("📝 Paper Trading Engine initialized")
        logger.info(f"   Initial Balance: ${initial_balance}")
        logger.info(f"   Mode: SIMULATION (no real trades)")
    
    def open_position(
        self,
        symbol: str,
        side: str,
        size: Decimal,
        entry_price: Decimal,
        stop_loss: Decimal,
        take_profit: Decimal,
        leverage: int = 1,
    ) -> Dict[str, Any]:
        """
        Open a simulated position.
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            size: Position size in base currency (e.g., 0.1 BTC)
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            leverage: Leverage used (for margin calculation)
        
        Returns:
            Dict with position details
        """
        if symbol in self.positions:
            return {'success': False, 'error': f'Position already open for {symbol}'}
        
        # Calculate notional and margin
        notional = size * entry_price
        margin_required = notional / Decimal(str(leverage))
        
        if margin_required > self.balance:
            return {'success': False, 'error': f'Insufficient balance. Need ${margin_required:.2f}, have ${self.balance:.2f}'}
        
        # Create position
        position = PaperPosition(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(timezone.utc),
            leverage=leverage,
        )
        
        self.positions[symbol] = position
        self.balance -= margin_required  # Reserve margin
        
        logger.info(f"📝 [PAPER] Opened {side.upper()} {symbol}")
        logger.info(f"   Entry: ${entry_price:.4f} | Size: {size}")
        logger.info(f"   SL: ${stop_loss:.4f} | TP: ${take_profit:.4f}")
        
        return {
            'success': True,
            'position': {
                'symbol': symbol,
                'side': side,
                'size': float(size),
                'entry_price': float(entry_price),
                'stop_loss': float(stop_loss),
                'take_profit': float(take_profit),
                'margin_used': float(margin_required),
            }
        }
    
    def update_positions(self, current_prices: Dict[str, Decimal]) -> List[PaperTrade]:
        """
        Update all positions with current prices.
        Check for SL/TP hits.
        
        Args:
            current_prices: Dict of symbol -> current_price
        
        Returns:
            List of closed trades
        """
        closed_trades = []
        
        for symbol, position in list(self.positions.items()):
            current_price = current_prices.get(symbol)
            if not current_price:
                continue
            
            # Calculate unrealized P&L
            if position.side == 'long':
                pnl = (current_price - position.entry_price) * position.size
                hit_tp = current_price >= position.take_profit
                hit_sl = current_price <= position.stop_loss
            else:
                pnl = (position.entry_price - current_price) * position.size
                hit_tp = current_price <= position.take_profit
                hit_sl = current_price >= position.stop_loss
            
            position.unrealized_pnl = pnl
            
            # Check for SL/TP hit
            if hit_tp:
                trade = self._close_position(symbol, position.take_profit, 'take_profit')
                if trade:
                    closed_trades.append(trade)
            elif hit_sl:
                trade = self._close_position(symbol, position.stop_loss, 'stop_loss')
                if trade:
                    closed_trades.append(trade)
        
        # Update equity
        self._update_equity()
        
        return closed_trades
    
    def close_position(
        self,
        symbol: str,
        exit_price: Decimal,
        reason: str = 'manual',
    ) -> Optional[PaperTrade]:
        """
        Manually close a position.
        
        Args:
            symbol: Symbol to close
            exit_price: Exit price
            reason: Close reason
        
        Returns:
            PaperTrade record or None
        """
        if symbol not in self.positions:
            return None
        
        return self._close_position(symbol, exit_price, reason)
    
    def _close_position(
        self,
        symbol: str,
        exit_price: Decimal,
        reason: str,
    ) -> PaperTrade:
        """Internal position close."""
        position = self.positions.pop(symbol)
        exit_time = datetime.now(timezone.utc)
        
        # Calculate P&L
        if position.side == 'long':
            pnl = (exit_price - position.entry_price) * position.size
            pnl_pct = ((exit_price / position.entry_price) - 1) * 100
        else:
            pnl = (position.entry_price - exit_price) * position.size
            pnl_pct = ((position.entry_price / exit_price) - 1) * 100
        
        # Update balance
        notional = position.size * position.entry_price
        margin_returned = notional / Decimal(str(position.leverage))  # Use position's leverage
        self.balance += margin_returned + pnl
        
        # Update stats
        self.total_trades += 1
        self.total_pnl += pnl
        self.session_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Duration
        duration = (exit_time - position.entry_time).total_seconds()
        
        # Create trade record
        trade = PaperTrade(
            symbol=symbol,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=exit_time,
            exit_reason=reason,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_seconds=int(duration),
        )
        
        self.trade_history.append(trade)
        
        # Log
        emoji = "🟢" if pnl > 0 else "🔴"
        logger.info(f"📝 [PAPER] {emoji} Closed {position.side.upper()} {symbol}")
        logger.info(f"   Entry: ${position.entry_price:.4f} → Exit: ${exit_price:.4f}")
        logger.info(f"   P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | Reason: {reason}")
        
        return trade
    
    def _update_equity(self):
        """Update equity and drawdown."""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        self.equity = self.balance + unrealized
        
        # Track peak and drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        drawdown = (self.peak_equity - self.equity) / self.peak_equity * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # Calculate average win/loss
        wins = [t for t in self.trade_history if t.pnl > 0]
        losses = [t for t in self.trade_history if t.pnl <= 0]
        
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else Decimal('0')
        avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else Decimal('0')
        
        profit_factor = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses))) if losses and sum(t.pnl for t in losses) != 0 else Decimal('0')
        
        return {
            'mode': 'PAPER TRADING',
            'initial_balance': float(self.initial_balance),
            'current_balance': float(self.balance),
            'equity': float(self.equity),
            'total_pnl': float(self.total_pnl),
            'total_pnl_pct': float((self.equity / self.initial_balance - 1) * 100),
            'session_pnl': float(self.session_pnl),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'max_drawdown_pct': float(self.max_drawdown),
            'peak_equity': float(self.peak_equity),
            'open_positions': len(self.positions),
        }
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        return [
            {
                'symbol': p.symbol,
                'side': p.side,
                'size': float(p.size),
                'entry_price': float(p.entry_price),
                'stop_loss': float(p.stop_loss),
                'take_profit': float(p.take_profit),
                'unrealized_pnl': float(p.unrealized_pnl),
                'entry_time': p.entry_time.isoformat(),
            }
            for p in self.positions.values()
        ]
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades."""
        trades = list(self.trade_history)[-limit:]
        return [
            {
                'symbol': t.symbol,
                'side': t.side,
                'entry_price': float(t.entry_price),
                'exit_price': float(t.exit_price),
                'pnl': float(t.pnl),
                'pnl_pct': float(t.pnl_pct),
                'exit_reason': t.exit_reason,
                'duration_seconds': t.duration_seconds,
                'exit_time': t.exit_time.isoformat(),
            }
            for t in trades
        ]
    
    def format_status(self) -> str:
        """Format status for Telegram."""
        stats = self.get_statistics()
        
        return f"""📝 **PAPER TRADING MODE**

💰 **Balance**: ${stats['current_balance']:.2f}
📊 **Equity**: ${stats['equity']:.2f}
📈 **Total P&L**: ${stats['total_pnl']:+.2f} ({stats['total_pnl_pct']:+.2f}%)

🎯 **Performance**:
• Trades: {stats['total_trades']}
• Win Rate: {stats['win_rate']:.1f}%
• Profit Factor: {stats['profit_factor']:.2f}
• Max Drawdown: {stats['max_drawdown_pct']:.2f}%

📂 **Open Positions**: {stats['open_positions']}

⚠️ _No real trades executed_"""


def is_paper_trading_enabled() -> bool:
    """Check if paper trading mode is enabled."""
    return os.getenv('PAPER_TRADING', 'false').lower() in ('true', '1', 'yes')


def get_paper_trading_balance() -> Decimal:
    """Get paper trading starting balance."""
    return Decimal(os.getenv('PAPER_TRADING_BALANCE', '1000'))
