"""
Drawdown Monitor - Real-time drawdown tracking and alerting
Tracks peak equity and current drawdown with multiple alert levels
"""

import logging
from typing import Dict, Any, Optional, Callable, Tuple
from decimal import Decimal
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DrawdownLevel(Enum):
    """Drawdown severity levels"""
    NORMAL = "normal"      # < 5%
    WARNING = "warning"    # 5-10%
    CRITICAL = "critical"  # 10-15%
    EMERGENCY = "emergency"  # > 15%


@dataclass
class DrawdownSnapshot:
    """Snapshot of drawdown at a point in time"""
    timestamp: datetime
    peak_equity: Decimal
    current_equity: Decimal
    drawdown_amount: Decimal
    drawdown_pct: Decimal
    level: DrawdownLevel
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'peak_equity': float(self.peak_equity),
            'current_equity': float(self.current_equity),
            'drawdown_amount': float(self.drawdown_amount),
            'drawdown_pct': float(self.drawdown_pct),
            'level': self.level.value
        }


class DrawdownMonitor:
    """
    Real-time drawdown monitoring with alerts
    """
    
    def __init__(self, account_manager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize drawdown monitor
        
        Args:
            account_manager: AccountManager instance
            config: Configuration
        """
        self.account_manager = account_manager
        
        # Load configuration
        cfg = config or {}
        
        # Alert thresholds
        self.warning_threshold_pct = Decimal(str(cfg.get('warning_threshold_pct', 5)))
        self.critical_threshold_pct = Decimal(str(cfg.get('critical_threshold_pct', 10)))
        self.emergency_threshold_pct = Decimal(str(cfg.get('emergency_threshold_pct', 15)))
        
        # Auto-pause configuration
        self.auto_pause_enabled = cfg.get('auto_pause_enabled', True)
        self.auto_pause_threshold_pct = Decimal(str(cfg.get('auto_pause_threshold_pct', 12)))
        
        # State
        self.is_paused = False
        self.pause_time: Optional[datetime] = None
        self.pause_reason: Optional[str] = None
        
        # History
        self.snapshots: list[DrawdownSnapshot] = []
        self.max_snapshots = cfg.get('max_snapshots', 1000)
        
        # Current state
        self.current_level = DrawdownLevel.NORMAL
        self.last_alert_level: Optional[DrawdownLevel] = None
        
        # Statistics
        self.max_drawdown_ever_pct = Decimal('0')
        self.max_drawdown_ever_amount = Decimal('0')
        self.max_drawdown_time: Optional[datetime] = None
        self.recovery_count = 0
        
        # Callbacks
        self.on_level_change_callbacks: list[Callable] = []
        self.on_pause_callbacks: list[Callable] = []
        
        logger.info("📉 Drawdown Monitor initialized")
        logger.info(f"   Warning: {self.warning_threshold_pct}%")
        logger.info(f"   Critical: {self.critical_threshold_pct}%")
        logger.info(f"   Emergency: {self.emergency_threshold_pct}%")
        logger.info(f"   Auto-pause: {'Enabled' if self.auto_pause_enabled else 'Disabled'} at {self.auto_pause_threshold_pct}%")
    
    def update(self):
        """Update drawdown status"""
        peak = self.account_manager.peak_equity
        current = self.account_manager.current_equity
        
        # Calculate drawdown
        drawdown_amount = peak - current
        drawdown_pct = (drawdown_amount / peak * 100) if peak > 0 else Decimal('0')
        
        # Determine level
        level = self._calculate_level(drawdown_pct)
        
        # Create snapshot
        snapshot = DrawdownSnapshot(
            timestamp=datetime.now(timezone.utc),
            peak_equity=peak,
            current_equity=current,
            drawdown_amount=drawdown_amount,
            drawdown_pct=drawdown_pct,
            level=level
        )
        
        # Store snapshot
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]
        
        # Update max drawdown
        if drawdown_pct > self.max_drawdown_ever_pct:
            self.max_drawdown_ever_pct = drawdown_pct
            self.max_drawdown_ever_amount = drawdown_amount
            self.max_drawdown_time = snapshot.timestamp
            logger.warning(f"📉 New maximum drawdown: {drawdown_pct:.2f}%")
        
        # Check for recovery
        if self.current_level != DrawdownLevel.NORMAL and level == DrawdownLevel.NORMAL:
            self.recovery_count += 1
            logger.info(f"📈 Recovered from drawdown (recovery #{self.recovery_count})")
        
        # Check for level change
        if level != self.current_level:
            self._handle_level_change(self.current_level, level, drawdown_pct)
            self.current_level = level
        
        # Check auto-pause
        if self.auto_pause_enabled and not self.is_paused:
            if drawdown_pct >= self.auto_pause_threshold_pct:
                self.pause(f"Auto-pause: drawdown {drawdown_pct:.2f}% >= {self.auto_pause_threshold_pct}%")
    
    def _calculate_level(self, drawdown_pct: Decimal) -> DrawdownLevel:
        """Calculate drawdown level"""
        if drawdown_pct >= self.emergency_threshold_pct:
            return DrawdownLevel.EMERGENCY
        elif drawdown_pct >= self.critical_threshold_pct:
            return DrawdownLevel.CRITICAL
        elif drawdown_pct >= self.warning_threshold_pct:
            return DrawdownLevel.WARNING
        else:
            return DrawdownLevel.NORMAL
    
    def _handle_level_change(self, old_level: DrawdownLevel, new_level: DrawdownLevel, 
                            drawdown_pct: Decimal):
        """Handle drawdown level change"""
        logger.warning(f"⚠️ Drawdown level changed: {old_level.value} → {new_level.value} ({drawdown_pct:.2f}%)")
        
        # Execute callbacks
        for callback in self.on_level_change_callbacks:
            try:
                callback(old_level, new_level, drawdown_pct)
            except Exception as e:
                logger.error(f"Level change callback error: {e}")
    
    def pause(self, reason: str):
        """
        Pause trading due to drawdown
        
        Args:
            reason: Reason for pause
        """
        if self.is_paused:
            return
        
        self.is_paused = True
        self.pause_time = datetime.now(timezone.utc)
        self.pause_reason = reason
        
        logger.critical(f"⏸️  TRADING PAUSED - {reason}")
        
        # Execute callbacks
        for callback in self.on_pause_callbacks:
            try:
                callback(reason)
            except Exception as e:
                logger.error(f"Pause callback error: {e}")
    
    def resume(self):
        """Resume trading after pause"""
        if not self.is_paused:
            return
        
        logger.info("▶️  Trading RESUMED")
        logger.info(f"   Was paused: {self.pause_reason}")
        
        self.is_paused = False
        self.pause_time = None
        self.pause_reason = None
    
    def add_level_change_callback(self, callback: Callable):
        """
        Add callback for level changes
        
        Args:
            callback: Function(old_level, new_level, drawdown_pct)
        """
        self.on_level_change_callbacks.append(callback)
    
    def add_pause_callback(self, callback: Callable):
        """
        Add callback for pause events
        
        Args:
            callback: Function(reason)
        """
        self.on_pause_callbacks.append(callback)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current drawdown status"""
        peak = self.account_manager.peak_equity
        current = self.account_manager.current_equity
        drawdown_amount = peak - current
        drawdown_pct = (drawdown_amount / peak * 100) if peak > 0 else Decimal('0')
        
        return {
            'current_drawdown_pct': float(drawdown_pct),
            'current_drawdown_amount': float(drawdown_amount),
            'peak_equity': float(peak),
            'current_equity': float(current),
            'level': self.current_level.value,
            'is_paused': self.is_paused,
            'pause_reason': self.pause_reason,
            'pause_time': self.pause_time.isoformat() if self.pause_time else None
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get drawdown statistics"""
        return {
            'max_drawdown_pct': float(self.max_drawdown_ever_pct),
            'max_drawdown_amount': float(self.max_drawdown_ever_amount),
            'max_drawdown_time': self.max_drawdown_time.isoformat() if self.max_drawdown_time else None,
            'recovery_count': self.recovery_count,
            'snapshots_count': len(self.snapshots)
        }
    
    def get_recent_snapshots(self, count: int = 10) -> list[Dict[str, Any]]:
        """
        Get recent drawdown snapshots
        
        Args:
            count: Number of snapshots to return
            
        Returns:
            List of snapshot dictionaries
        """
        recent = self.snapshots[-count:] if len(self.snapshots) > count else self.snapshots
        return [s.to_dict() for s in recent]
    
    def get_thresholds(self) -> Dict[str, Any]:
        """Get configured thresholds"""
        return {
            'warning_threshold_pct': float(self.warning_threshold_pct),
            'critical_threshold_pct': float(self.critical_threshold_pct),
            'emergency_threshold_pct': float(self.emergency_threshold_pct),
            'auto_pause_enabled': self.auto_pause_enabled,
            'auto_pause_threshold_pct': float(self.auto_pause_threshold_pct)
        }
    
    def get_recovery_mode_multiplier(self) -> float:
        """
        Get position size multiplier based on drawdown recovery mode.
        
        Recovery mode reduces position sizes after drawdown to:
        1. Limit further losses
        2. Allow gradual recovery with reduced risk
        3. Prevent emotional revenge trading
        
        Returns:
            Multiplier (0.0 to 1.0) to apply to position sizes
            - NORMAL: 1.0 (100% of normal size)
            - WARNING: 0.7 (70% - moderate reduction)
            - CRITICAL: 0.4 (40% - significant reduction)
            - EMERGENCY: 0.0 (0% - no trading)
        """
        multipliers = {
            DrawdownLevel.NORMAL: 1.0,
            DrawdownLevel.WARNING: 0.7,
            DrawdownLevel.CRITICAL: 0.4,
            DrawdownLevel.EMERGENCY: 0.0,
        }
        
        multiplier = multipliers.get(self.current_level, 1.0)
        
        if multiplier < 1.0:
            logger.info(f"🔄 Recovery Mode: {self.current_level.value} - position size at {multiplier*100:.0f}%")
        
        return multiplier
    
    def is_trading_allowed(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on drawdown status.
        
        Returns:
            (allowed, reason)
        """
        if self.is_paused:
            return False, f"Trading paused: {self.pause_reason}"
        
        if self.current_level == DrawdownLevel.EMERGENCY:
            return False, f"Emergency drawdown level - trading suspended"
        
        return True, "OK"
