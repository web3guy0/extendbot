"""
Risk Engine - Pre-trade and real-time risk validation
Enterprise-grade risk management with multiple safety layers
"""

import logging
from typing import Dict, Any, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class RiskLimits:
    """Risk limit configuration"""
    def __init__(self, config: Dict[str, Any]):
        # Position limits
        self.max_position_size_pct = Decimal(str(config.get('max_position_size_pct', 50)))
        self.max_positions = config.get('max_positions', 5)
        self.max_symbol_exposure_pct = Decimal(str(config.get('max_symbol_exposure_pct', 30)))
        
        # Leverage limits
        self.max_leverage = Decimal(str(config.get('max_leverage', 5)))
        self.max_margin_usage_pct = Decimal(str(config.get('max_margin_usage_pct', 80)))
        
        # P&L limits
        self.max_daily_loss_pct = Decimal(str(config.get('max_daily_loss_pct', 5)))
        self.max_drawdown_pct = Decimal(str(config.get('max_drawdown_pct', 10)))
        self.max_loss_per_trade_pct = Decimal(str(config.get('max_loss_per_trade_pct', 2)))
        
        # Trade frequency limits
        self.max_trades_per_hour = config.get('max_trades_per_hour', 10)
        self.max_trades_per_day = config.get('max_trades_per_day', 50)
        
        # Correlation limits
        self.max_correlated_exposure_pct = Decimal(str(config.get('max_correlated_exposure_pct', 60)))


class RiskEngine:
    """
    Enterprise risk engine for comprehensive risk management
    """
    
    def __init__(self, account_manager, position_manager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize risk engine
        
        Args:
            account_manager: AccountManager instance
            position_manager: PositionManager instance
            config: Risk configuration
        """
        self.account_manager = account_manager
        self.position_manager = position_manager
        
        # Load risk limits
        risk_config = config or {}
        self.limits = RiskLimits(risk_config)
        
        # Risk state
        self.is_enabled = True
        self.risk_violations: list = []
        self.daily_trades = 0
        self.hourly_trades = 0
        self.last_trade_time = None
        self.hour_reset_time = datetime.now(timezone.utc)
        self.day_reset_time = datetime.now(timezone.utc)
        
        # Risk scoring
        self.current_risk_score = 0  # 0-100, higher = more risky
        
        logger.info("🛡️ Risk Engine initialized")
        logger.info(f"   Max Position Size: {self.limits.max_position_size_pct}%")
        logger.info(f"   Max Leverage: {self.limits.max_leverage}x")
        logger.info(f"   Max Daily Loss: {self.limits.max_daily_loss_pct}%")
        logger.info(f"   Max Drawdown: {self.limits.max_drawdown_pct}%")
    
    def enable(self):
        """Enable risk checks"""
        self.is_enabled = True
        logger.info("✅ Risk engine enabled")
    
    def disable(self):
        """Disable risk checks (dangerous!)"""
        self.is_enabled = False
        logger.warning("⚠️ Risk engine DISABLED - trading without protection!")
    
    def validate_pre_trade(self, symbol: str, side: str, size: Decimal,
                          price: Decimal) -> Tuple[bool, Optional[str]]:
        """
        Validate a trade before execution
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Order size
            price: Order price
            
        Returns:
            (is_valid, rejection_reason)
        """
        if not self.is_enabled:
            return True, None
        
        # Reset counters if needed
        self._reset_counters()
        
        # 1. Check position count limit
        if len(self.position_manager.open_positions) >= self.limits.max_positions:
            return False, f"Max positions limit reached ({self.limits.max_positions})"
        
        # 2. Check position size limit (based on collateral, not leveraged notional)
        trade_value = size * price
        equity = self.account_manager.current_equity
        
        # Guard against division by zero
        if equity <= 0:
            return False, "Invalid equity: cannot calculate position size"
        if self.limits.max_leverage <= 0:
            return False, "Invalid leverage configuration"
            
        # Calculate collateral needed (trade value divided by leverage)
        collateral_pct = (trade_value / self.limits.max_leverage / equity) * 100
        
        # Use >= for limit check (allow exactly at limit)
        if collateral_pct > self.limits.max_position_size_pct + Decimal('0.1'):  # Small buffer
            return False, f"Position size {collateral_pct:.1f}% exceeds limit {self.limits.max_position_size_pct}%"
        
        # 3. Check margin availability
        required_margin = trade_value / self.limits.max_leverage
        if required_margin > self.account_manager.current_balance:
            return False, f"Insufficient margin: need ${required_margin:.2f}, have ${self.account_manager.current_balance:.2f}"
        
        # 4. Check total margin usage
        total_margin = self.account_manager.margin_used + required_margin
        margin_usage_pct = (total_margin / equity) * 100
        
        logger.info(f"🔍 Margin check: current={self.account_manager.margin_used:.2f}, required={required_margin:.2f}, total={total_margin:.2f}, equity={equity:.2f}, usage={margin_usage_pct:.1f}%")
        
        if margin_usage_pct > self.limits.max_margin_usage_pct:
            return False, f"Margin usage {margin_usage_pct:.1f}% exceeds limit {self.limits.max_margin_usage_pct}%"
        
        # 5. Check symbol exposure limit
        existing_position = self.position_manager.get_position(symbol)
        if existing_position:
            return False, f"Already have position in {symbol}"
        
        # 6. Check daily loss limit
        session_start_equity = self.account_manager.session_start_equity
        if session_start_equity > 0:
            daily_loss_pct = abs(self.account_manager.session_pnl / session_start_equity * 100)
            if self.account_manager.session_pnl < 0 and daily_loss_pct >= self.limits.max_daily_loss_pct:
                return False, f"Daily loss limit reached: {daily_loss_pct:.2f}% >= {self.limits.max_daily_loss_pct}%"
        
        # 7. Check drawdown limit
        peak_equity = self.account_manager.peak_equity
        if peak_equity > 0:
            current_drawdown = (peak_equity - equity) / peak_equity * 100
            if current_drawdown >= self.limits.max_drawdown_pct:
                return False, f"Drawdown limit reached: {current_drawdown:.2f}% >= {self.limits.max_drawdown_pct}%"
        
        # 8. Check trade frequency
        if self.hourly_trades >= self.limits.max_trades_per_hour:
            return False, f"Hourly trade limit reached ({self.limits.max_trades_per_hour})"
        
        if self.daily_trades >= self.limits.max_trades_per_day:
            return False, f"Daily trade limit reached ({self.limits.max_trades_per_day})"
        
        # All checks passed
        return True, None
    
    def record_trade(self):
        """Record a trade execution"""
        self.daily_trades += 1
        self.hourly_trades += 1
        self.last_trade_time = datetime.now(timezone.utc)
    
    def _reset_counters(self):
        """Reset hourly and daily counters if needed"""
        now = datetime.now(timezone.utc)
        
        # Reset hourly counter
        if (now - self.hour_reset_time).total_seconds() >= 3600:
            self.hourly_trades = 0
            self.hour_reset_time = now
        
        # Reset daily counter
        if now.date() > self.day_reset_time.date():
            self.daily_trades = 0
            self.day_reset_time = now
    
    def calculate_risk_score(self) -> int:
        """
        Calculate overall risk score (0-100)
        Higher score = higher risk
        """
        score = 0
        
        # Factor 1: Margin usage (0-25 points)
        current_equity = self.account_manager.current_equity
        if current_equity > 0:
            margin_usage = float(self.account_manager.margin_used / current_equity * 100)
            score += min(25, margin_usage / 4)
        
        # Factor 2: Drawdown (0-25 points)
        peak_equity = self.account_manager.peak_equity
        if peak_equity > 0:
            drawdown = float((peak_equity - current_equity) / peak_equity * 100)
            score += min(25, max(0, drawdown * 2.5))  # Ensure non-negative
        
        # Factor 3: Position count (0-25 points)
        if self.limits.max_positions > 0:
            position_count_pct = (len(self.position_manager.open_positions) / self.limits.max_positions) * 100
            score += min(25, position_count_pct / 4)
        
        # Factor 4: Daily P&L (0-25 points)
        session_start_equity = self.account_manager.session_start_equity
        if self.account_manager.session_pnl < 0 and session_start_equity > 0:
            daily_loss_pct = abs(float(self.account_manager.session_pnl / session_start_equity * 100))
            score += min(25, daily_loss_pct * 5)
        
        self.current_risk_score = int(score)
        return self.current_risk_score
    
    def get_risk_assessment(self) -> Dict[str, Any]:
        """Get comprehensive risk assessment"""
        risk_score = self.calculate_risk_score()
        
        # Risk level classification
        if risk_score < 30:
            risk_level = "LOW"
        elif risk_score < 60:
            risk_level = "MEDIUM"
        elif risk_score < 80:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        equity = self.account_manager.current_equity
        margin_used = self.account_manager.margin_used
        peak_equity = self.account_manager.peak_equity
        session_start_equity = self.account_manager.session_start_equity
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'is_enabled': self.is_enabled,
            'margin_usage_pct': float(margin_used / equity * 100) if equity > 0 else 0,
            'current_drawdown_pct': float((peak_equity - equity) / peak_equity * 100) if peak_equity > 0 else 0,
            'daily_loss_pct': float(abs(self.account_manager.session_pnl / session_start_equity * 100)) if self.account_manager.session_pnl < 0 and session_start_equity > 0 else 0,
            'open_positions': len(self.position_manager.open_positions),
            'daily_trades': self.daily_trades,
            'hourly_trades': self.hourly_trades,
            'can_trade': risk_level != "CRITICAL",
            'warnings': self._get_warnings()
        }
    
    def _get_warnings(self) -> list:
        """Get active risk warnings"""
        warnings = []
        
        equity = self.account_manager.current_equity
        peak_equity = self.account_manager.peak_equity
        session_start_equity = self.account_manager.session_start_equity
        
        # Check margin usage
        margin_usage_pct = float(self.account_manager.margin_used / equity * 100) if equity > 0 else 0
        if margin_usage_pct > 70:
            warnings.append(f"High margin usage: {margin_usage_pct:.1f}%")
        
        # Check drawdown (guard for peak_equity > 0)
        if peak_equity > 0:
            drawdown_pct = float((peak_equity - equity) / peak_equity * 100)
            if drawdown_pct > 7:
                warnings.append(f"High drawdown: {drawdown_pct:.1f}%")
        
        # Check daily loss (guard for session_start_equity > 0)
        if self.account_manager.session_pnl < 0 and session_start_equity > 0:
            daily_loss_pct = abs(float(self.account_manager.session_pnl / session_start_equity * 100))
            if daily_loss_pct > 3:
                warnings.append(f"Daily loss approaching limit: {daily_loss_pct:.1f}%")
        
        # Check position count
        if len(self.position_manager.open_positions) >= self.limits.max_positions * 0.8:
            warnings.append(f"High position count: {len(self.position_manager.open_positions)}/{self.limits.max_positions}")
        
        return warnings
    
    def get_limits(self) -> Dict[str, Any]:
        """Get configured risk limits"""
        return {
            'max_position_size_pct': float(self.limits.max_position_size_pct),
            'max_positions': self.limits.max_positions,
            'max_leverage': float(self.limits.max_leverage),
            'max_margin_usage_pct': float(self.limits.max_margin_usage_pct),
            'max_daily_loss_pct': float(self.limits.max_daily_loss_pct),
            'max_drawdown_pct': float(self.limits.max_drawdown_pct),
            'max_trades_per_hour': self.limits.max_trades_per_hour,
            'max_trades_per_day': self.limits.max_trades_per_day
        }
