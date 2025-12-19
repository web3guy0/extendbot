"""
Kill Switch - Emergency stop mechanism
Multiple safety triggers to halt trading immediately
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class TriggerReason(Enum):
    """Kill switch trigger reasons"""
    MANUAL = "manual_stop"
    DAILY_LOSS = "daily_loss_limit"
    DRAWDOWN = "drawdown_limit"
    MARGIN_CALL = "margin_call"
    CONNECTION_LOSS = "connection_loss"
    ERROR_RATE = "high_error_rate"
    POSITION_LOSS = "position_loss_limit"
    SYSTEM_ERROR = "system_error"


class KillSwitch:
    """
    Emergency stop mechanism with multiple triggers
    """
    
    def __init__(self, account_manager, position_manager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize kill switch
        
        Args:
            account_manager: AccountManager instance
            position_manager: PositionManager instance
            config: Kill switch configuration
        """
        self.account_manager = account_manager
        self.position_manager = position_manager
        
        # Load configuration
        cfg = config or {}
        
        # Kill switch state
        self.is_triggered = False
        self.trigger_reason: Optional[TriggerReason] = None
        self.trigger_time: Optional[datetime] = None
        self.trigger_details: Optional[str] = None
        
        # Trigger thresholds
        self.daily_loss_trigger_pct = Decimal(str(cfg.get('daily_loss_trigger_pct', 10)))
        self.drawdown_trigger_pct = Decimal(str(cfg.get('drawdown_trigger_pct', 15)))
        self.margin_call_trigger_pct = Decimal(str(cfg.get('margin_call_trigger_pct', 90)))
        self.position_loss_trigger_pct = Decimal(str(cfg.get('position_loss_trigger_pct', 8)))
        self.error_rate_threshold = cfg.get('error_rate_threshold', 0.5)  # 50% error rate
        
        # Error tracking
        self.recent_errors = []
        self.recent_trades = []
        self.max_recent_items = 20
        
        # Callbacks (Python 3.8+ compatible)
        self.on_trigger_callbacks: List[Callable] = []
        
        # Auto-reset configuration
        self.auto_reset_enabled = cfg.get('auto_reset_enabled', False)
        self.auto_reset_minutes = cfg.get('auto_reset_minutes', 60)
        
        logger.info("🔴 Kill Switch initialized")
        logger.info(f"   Daily Loss Trigger: {self.daily_loss_trigger_pct}%")
        logger.info(f"   Drawdown Trigger: {self.drawdown_trigger_pct}%")
        logger.info(f"   Margin Call Trigger: {self.margin_call_trigger_pct}%")
        logger.info(f"   Auto-reset: {'Enabled' if self.auto_reset_enabled else 'Disabled'}")
    
    def check_triggers(self) -> bool:
        """
        Check all kill switch triggers
        
        Returns:
            True if kill switch should be triggered
        """
        if self.is_triggered:
            return True
        
        # Check auto-reset
        if self.auto_reset_enabled and self._should_auto_reset():
            self.reset()
        
        # 1. Check daily loss trigger
        if self.account_manager.session_pnl < 0:
            # Prevent division by zero if session_start_equity is 0
            if self.account_manager.session_start_equity > 0:
                daily_loss_pct = abs(self.account_manager.session_pnl / self.account_manager.session_start_equity * 100)
            else:
                daily_loss_pct = Decimal('0')
            if daily_loss_pct >= self.daily_loss_trigger_pct:
                self.trigger(
                    TriggerReason.DAILY_LOSS,
                    f"Daily loss {daily_loss_pct:.2f}% >= {self.daily_loss_trigger_pct}%"
                )
                return True
        
        # 2. Check drawdown trigger
        equity = self.account_manager.current_equity
        if self.account_manager.peak_equity > 0:  # Prevent division by zero
            drawdown_pct = (self.account_manager.peak_equity - equity) / self.account_manager.peak_equity * 100
            if drawdown_pct >= self.drawdown_trigger_pct:
                self.trigger(
                    TriggerReason.DRAWDOWN,
                    f"Drawdown {drawdown_pct:.2f}% >= {self.drawdown_trigger_pct}%"
                )
                return True
        
        # 3. Check margin call
        if equity > 0:  # Prevent division by zero
            margin_usage_pct = (self.account_manager.margin_used / equity) * 100
            if margin_usage_pct >= self.margin_call_trigger_pct:
                self.trigger(
                    TriggerReason.MARGIN_CALL,
                    f"Margin usage {margin_usage_pct:.2f}% >= {self.margin_call_trigger_pct}%"
                )
                return True
        
        # 4. Check position losses
        for symbol, position in self.position_manager.open_positions.items():
            if position.unrealized_pnl < 0:
                # Guard against division by zero
                position_value = position.size * position.entry_price
                if position_value > 0:
                    loss_pct = abs(position.unrealized_pnl / position_value * 100)
                else:
                    loss_pct = Decimal('0')
                if loss_pct >= self.position_loss_trigger_pct:
                    self.trigger(
                        TriggerReason.POSITION_LOSS,
                        f"{symbol} loss {loss_pct:.2f}% >= {self.position_loss_trigger_pct}%"
                    )
                    return True
        
        # 5. Check error rate
        if len(self.recent_trades) >= 10:
            error_rate = len(self.recent_errors) / len(self.recent_trades)
            if error_rate >= self.error_rate_threshold:
                self.trigger(
                    TriggerReason.ERROR_RATE,
                    f"Error rate {error_rate*100:.1f}% >= {self.error_rate_threshold*100:.1f}%"
                )
                return True
        
        return False
    
    def trigger(self, reason: TriggerReason, details: str):
        """
        Trigger the kill switch
        
        Args:
            reason: Reason for triggering
            details: Additional details
        """
        if self.is_triggered:
            return  # Already triggered
        
        self.is_triggered = True
        self.trigger_reason = reason
        self.trigger_time = datetime.now(timezone.utc)
        self.trigger_details = details
        
        logger.critical("🚨 KILL SWITCH TRIGGERED 🚨")
        logger.critical(f"   Reason: {reason.value}")
        logger.critical(f"   Details: {details}")
        logger.critical(f"   Time: {self.trigger_time.isoformat()}")
        
        # Execute callbacks
        for callback in self.on_trigger_callbacks:
            try:
                callback(reason, details)
            except Exception as e:
                logger.error(f"Kill switch callback error: {e}")
    
    def manual_trigger(self, details: str = "User requested"):
        """Manually trigger kill switch"""
        self.trigger(TriggerReason.MANUAL, details)
    
    def reset(self):
        """Reset kill switch (use with caution!)"""
        if not self.is_triggered:
            return
        
        logger.warning("⚠️ Kill switch RESET - resuming trading")
        logger.warning(f"   Previous trigger: {self.trigger_reason.value if self.trigger_reason else 'None'}")
        logger.warning(f"   Previous details: {self.trigger_details}")
        
        self.is_triggered = False
        self.trigger_reason = None
        self.trigger_time = None
        self.trigger_details = None
    
    def _should_auto_reset(self) -> bool:
        """Check if auto-reset should occur"""
        if not self.trigger_time:
            return False
        
        elapsed_minutes = (datetime.now(timezone.utc) - self.trigger_time).total_seconds() / 60
        return elapsed_minutes >= self.auto_reset_minutes
    
    def record_trade(self, success: bool):
        """
        Record trade result for error tracking
        
        CRITICAL FIX: Track errors with timestamps, not just counts.
        This ensures error rate is calculated over the same time window as trades.
        
        Args:
            success: Whether trade was successful
        """
        now = datetime.now(timezone.utc)
        
        self.recent_trades.append({
            'time': now,
            'success': success
        })
        
        if not success:
            self.recent_errors.append({
                'time': now
            })
        
        # CRITICAL FIX: Trim BOTH lists together based on time, not count
        # This prevents error rate calculation from being skewed by independent trimming
        cutoff_time = now - timedelta(minutes=30)  # Keep last 30 minutes of data
        
        self.recent_trades = [t for t in self.recent_trades if t['time'] > cutoff_time]
        self.recent_errors = [e for e in self.recent_errors if e['time'] > cutoff_time]
        
        # Also apply max count limit as a safety
        if len(self.recent_trades) > self.max_recent_items:
            self.recent_trades = self.recent_trades[-self.max_recent_items:]
        
        # CRITICAL FIX: Only keep errors that have a corresponding trade in recent_trades
        # This ensures error rate is accurate after trimming
        earliest_trade_time = self.recent_trades[0]['time'] if self.recent_trades else now
        self.recent_errors = [e for e in self.recent_errors if e['time'] >= earliest_trade_time]
    
    def record_connection_loss(self):
        """Record connection loss"""
        self.trigger(
            TriggerReason.CONNECTION_LOSS,
            "Lost connection to exchange"
        )
    
    def record_system_error(self, error: str):
        """
        Record system error
        
        Args:
            error: Error description
        """
        self.trigger(
            TriggerReason.SYSTEM_ERROR,
            f"System error: {error}"
        )
    
    def add_trigger_callback(self, callback: Callable):
        """
        Add callback to execute when kill switch triggers
        
        Args:
            callback: Function to call with (reason, details)
        """
        self.on_trigger_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get kill switch status"""
        return {
            'is_triggered': self.is_triggered,
            'trigger_reason': self.trigger_reason.value if self.trigger_reason else None,
            'trigger_time': self.trigger_time.isoformat() if self.trigger_time else None,
            'trigger_details': self.trigger_details,
            'recent_errors': len(self.recent_errors),
            'recent_trades': len(self.recent_trades),
            'error_rate': len(self.recent_errors) / len(self.recent_trades) if self.recent_trades else 0,
            'auto_reset_enabled': self.auto_reset_enabled,
            'minutes_until_reset': self._minutes_until_reset() if self.auto_reset_enabled and self.is_triggered else None
        }
    
    def _minutes_until_reset(self) -> Optional[float]:
        """Get minutes until auto-reset"""
        if not self.trigger_time:
            return None
        
        elapsed = (datetime.now(timezone.utc) - self.trigger_time).total_seconds() / 60
        remaining = self.auto_reset_minutes - elapsed
        return max(0, remaining)
    
    def get_thresholds(self) -> Dict[str, Any]:
        """Get configured thresholds"""
        return {
            'daily_loss_trigger_pct': float(self.daily_loss_trigger_pct),
            'drawdown_trigger_pct': float(self.drawdown_trigger_pct),
            'margin_call_trigger_pct': float(self.margin_call_trigger_pct),
            'position_loss_trigger_pct': float(self.position_loss_trigger_pct),
            'error_rate_threshold': self.error_rate_threshold
        }
