"""
Session Manager - Time-Based Trading Intelligence
Adjusts strategy parameters based on trading session.
"""

import os
import logging
from decimal import Decimal
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


class TradingSession(Enum):
    """Major trading sessions."""
    ASIAN = "asian"           # 00:00 - 08:00 UTC
    LONDON = "london"         # 08:00 - 12:00 UTC
    US_MORNING = "us_morning" # 12:00 - 16:00 UTC
    US_AFTERNOON = "us_afternoon"  # 16:00 - 21:00 UTC
    OFF_HOURS = "off_hours"   # 21:00 - 00:00 UTC


class SessionManager:
    """
    Trading Session Manager
    
    Crypto markets have distinct session characteristics:
    - Asian: Low volume, range-bound, mean reversion works
    - London Open: First breakout attempts, volatility pickup
    - US Morning: Main moves, highest volume, trend following
    - US Afternoon: Continuation or reversal patterns
    - Off Hours: Low liquidity, avoid large positions
    
    Adjusts:
    - Position sizes
    - Strategy aggressiveness
    - TP/SL distances
    - Trading frequency
    """
    
    def __init__(self):
        """Initialize session manager."""
        # Session definitions (UTC hours)
        self.sessions = {
            TradingSession.ASIAN: (0, 8),
            TradingSession.LONDON: (8, 12),
            TradingSession.US_MORNING: (12, 16),
            TradingSession.US_AFTERNOON: (16, 21),
            TradingSession.OFF_HOURS: (21, 24),
        }
        
        # Session parameter adjustments
        self.session_params = {
            TradingSession.ASIAN: {
                'aggression': 0.6,        # Conservative
                'position_size_mult': 0.7,
                'prefer_strategy': 'mean_reversion',
                'volatility_expected': 'low',
                'tp_multiplier': 0.7,     # Tighter TP in ranges
                'sl_multiplier': 0.6,     # Tighter SL
                'max_trades_per_hour': 2,
            },
            TradingSession.LONDON: {
                'aggression': 0.9,
                'position_size_mult': 0.9,
                'prefer_strategy': 'breakout',
                'volatility_expected': 'increasing',
                'tp_multiplier': 1.2,
                'sl_multiplier': 1.0,
                'max_trades_per_hour': 4,
            },
            TradingSession.US_MORNING: {
                'aggression': 1.2,        # Most aggressive
                'position_size_mult': 1.2,
                'prefer_strategy': 'trend_following',
                'volatility_expected': 'high',
                'tp_multiplier': 1.5,     # Let winners run
                'sl_multiplier': 1.0,
                'max_trades_per_hour': 6,
            },
            TradingSession.US_AFTERNOON: {
                'aggression': 1.0,
                'position_size_mult': 1.0,
                'prefer_strategy': 'momentum',
                'volatility_expected': 'medium',
                'tp_multiplier': 1.2,
                'sl_multiplier': 1.0,
                'max_trades_per_hour': 4,
            },
            TradingSession.OFF_HOURS: {
                'aggression': 0.4,        # Very conservative
                'position_size_mult': 0.5,
                'prefer_strategy': 'none',  # Avoid trading
                'volatility_expected': 'very_low',
                'tp_multiplier': 0.6,
                'sl_multiplier': 0.5,
                'max_trades_per_hour': 1,
            },
        }
        
        # High-impact event windows (UTC hours) - reduce risk
        self.high_impact_hours = [
            14,  # US Stock Open
            18,  # FOMC typical time
            20,  # US Stock Close
        ]
        
        logger.info("⏰ Session Manager initialized")
        logger.info(f"   Sessions: {[s.value for s in TradingSession]}")
    
    def get_current_session(self) -> TradingSession:
        """Get current trading session."""
        hour = datetime.now(timezone.utc).hour
        
        for session, (start, end) in self.sessions.items():
            if start <= hour < end:
                return session
        
        return TradingSession.OFF_HOURS
    
    def get_session_params(self, session: Optional[TradingSession] = None) -> Dict[str, Any]:
        """Get trading parameters for session."""
        if session is None:
            session = self.get_current_session()
        
        params = self.session_params[session].copy()
        
        # Check for high-impact hour
        hour = datetime.now(timezone.utc).hour
        if hour in self.high_impact_hours:
            params['aggression'] *= 0.5
            params['position_size_mult'] *= 0.5
            params['is_high_impact_hour'] = True
            logger.info(f"⚠️ High-impact hour ({hour}:00 UTC) - reducing exposure")
        else:
            params['is_high_impact_hour'] = False
        
        params['session'] = session.value
        return params
    
    def should_trade(self, session: Optional[TradingSession] = None) -> Tuple[bool, str]:
        """
        Check if we should trade in current session.
        
        Returns:
            Tuple of (should_trade, reason)
        """
        if session is None:
            session = self.get_current_session()
        
        params = self.session_params[session]
        
        if params['prefer_strategy'] == 'none':
            return False, f"Off-hours session ({session.value}) - avoid trading"
        
        if params['aggression'] < 0.5:
            return True, f"Low aggression session ({session.value}) - trade cautiously"
        
        return True, f"Active session ({session.value}) - trading enabled"
    
    def get_optimal_trade_time(self) -> Dict[str, Any]:
        """Get info about optimal trading times."""
        now = datetime.now(timezone.utc)
        current_session = self.get_current_session()
        
        # Find next high-aggression session
        best_sessions = [TradingSession.US_MORNING, TradingSession.LONDON]
        
        next_best = None
        hours_until = None
        
        for session in best_sessions:
            start, end = self.sessions[session]
            if now.hour < start:
                hours_until = start - now.hour
                next_best = session
                break
            elif now.hour >= end:
                # Next day
                hours_until = 24 - now.hour + start
                next_best = session
        
        return {
            'current_session': current_session.value,
            'current_aggression': self.session_params[current_session]['aggression'],
            'next_optimal_session': next_best.value if next_best else None,
            'hours_until_optimal': hours_until,
            'is_optimal_now': current_session in best_sessions,
        }
    
    def get_position_size_adjustment(self) -> Decimal:
        """Get position size multiplier for current session."""
        params = self.get_session_params()
        return Decimal(str(params['position_size_mult']))
    
    def get_tp_sl_adjustment(self) -> Tuple[Decimal, Decimal]:
        """Get TP/SL multipliers for current session."""
        params = self.get_session_params()
        return (
            Decimal(str(params['tp_multiplier'])),
            Decimal(str(params['sl_multiplier'])),
        )
    
    def time_until_session(self, target_session: TradingSession) -> timedelta:
        """Calculate time until a specific session starts."""
        now = datetime.now(timezone.utc)
        start, _ = self.sessions[target_session]
        
        if now.hour < start:
            # Session starts later today
            hours_until = start - now.hour
            minutes_until = 60 - now.minute if now.minute > 0 else 0
            if minutes_until > 0:
                hours_until -= 1
            return timedelta(hours=hours_until, minutes=minutes_until)
        else:
            # Session starts tomorrow
            hours_until = 24 - now.hour + start
            minutes_until = 60 - now.minute if now.minute > 0 else 0
            if minutes_until > 0:
                hours_until -= 1
            return timedelta(hours=hours_until, minutes=minutes_until)
    
    def log_session_info(self):
        """Log current session information."""
        session = self.get_current_session()
        params = self.get_session_params()
        
        logger.info(f"📅 Current Session: {session.value}")
        logger.info(f"   Aggression: {params['aggression']:.1f}x")
        logger.info(f"   Position Size: {params['position_size_mult']:.1f}x")
        logger.info(f"   Preferred Strategy: {params['prefer_strategy']}")
        logger.info(f"   Expected Volatility: {params['volatility_expected']}")
