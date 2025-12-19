"""
Swing Trading Strategy
Institutional-grade strategy with adaptive risk and smart money concepts.

Features:
- ATR-based dynamic TP/SL (not fixed percentages)
- Market regime detection (trending/ranging/volatile)
- Smart Money Concepts (FVG, Order Blocks, Liquidity Sweeps, BoS)
- Multi-timeframe alignment
- Session-aware trading
- Order flow integration
- VWAP confluence (institutional fair value)
- Divergence detection (RSI/MACD)
- Funding rate filter (avoid squeezes)
- Volume confirmation (filter fake moves)
- Adaptive position sizing

Target: 65%+ win rate with 3:1+ R:R in favorable conditions.
"""

import os
import logging
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from datetime import datetime, timezone, timedelta

from app.strategies.adaptive.market_regime import MarketRegimeDetector, MarketRegime
from app.strategies.adaptive.smart_money import SmartMoneyAnalyzer
from app.strategies.adaptive.multi_timeframe import MultiTimeframeAnalyzer
from app.strategies.adaptive.order_flow import OrderFlowAnalyzer
from app.strategies.adaptive.session_manager import SessionManager
from app.strategies.adaptive.adaptive_risk import AdaptiveRiskManager
from app.strategies.adaptive.pro_filters import ProTradingFilters
from app.strategies.adaptive.vwap import VWAPCalculator
from app.strategies.adaptive.divergence import DivergenceDetector
from app.strategies.adaptive.funding_rate import FundingRateFilter
from app.strategies.adaptive.supertrend import SupertrendIndicator, SupertrendDirection
from app.strategies.adaptive.donchian import DonchianChannel, DonchianPosition
# NEW: Professional momentum and volume indicators
from app.strategies.adaptive.stoch_rsi import StochRSICalculator
from app.strategies.adaptive.obv import OBVCalculator
from app.strategies.adaptive.cmf import ChaikinMoneyFlow

logger = logging.getLogger(__name__)


class SwingStrategy:
    """
    Institutional-Grade Swing Trading Strategy
    
    Combines multiple professional trading concepts:
    1. Adaptive Risk (ATR-based TP/SL)
    2. Market Regime Detection
    3. Smart Money Concepts
    4. Multi-Timeframe Analysis
    5. Session-Aware Trading
    6. Order Flow Analysis
    
    Signal Score System (0-10):
    - Technical indicators: 4 points max
    - SMC alignment: 2 points max
    - HTF alignment: 2 points max
    - Order flow: 2 points max
    
    Entry threshold: 6/10 (60%) minimum
    """
    
    def __init__(self, symbol: str, config: Dict[str, Any] = None):
        """Initialize world-class strategy."""
        self.symbol = symbol
        self.config = config or {}
        
        # Core parameters from environment
        self.leverage = int(os.getenv('MAX_LEVERAGE', '5'))
        self.base_position_size = Decimal(os.getenv('POSITION_SIZE_PCT', '50'))
        
        # TP/SL is calculated dynamically by AdaptiveRiskManager using ATR
        # See ATR_SL_MULTIPLIER and ATR_TP_MULTIPLIER in .env
        
        # ==================== SCORING SYSTEM ====================
        # Full theoretical max: ~25 points (all indicators perfectly aligned)
        # Score breakdown:
        #   Base (tech+SMC+HTF+OF+BoS): ~12 points
        #   Enhanced (regime+supertrend+donchian+vwap+div+vol+stoch+obv+cmf): ~13 points
        #
        # Penalty System (critical failures subtract points):
        #   - Counter-trend trade (regime mismatch): -5 points
        #   - Against Supertrend: -3 points  
        #   - Weak volume (below average): -2 points
        #   - OBV/CMF divergence against direction: -2 points
        #   - HTF misalignment: -3 points
        #
        # ADAPTIVE THRESHOLD based on regime:
        #   - TRENDING: Higher threshold (safer in strong moves)
        #   - RANGING: Lower threshold (mean reversion opportunities)
        self.min_signal_score = int(os.getenv('MIN_SIGNAL_SCORE', '12'))
        self.ranging_threshold_reduction = int(os.getenv('RANGING_THRESHOLD_REDUCTION', '4'))  # Lower by 4 in ranging
        self.max_signal_score = 25  # Full theoretical range
        
        # Penalty thresholds for critical failures
        # STRICT PENALTIES for high win rate - reject marginal trades
        self.regime_penalty = float(os.getenv('REGIME_PENALTY', '5.0'))      # Counter-trend is dangerous
        self.supertrend_penalty = float(os.getenv('SUPERTREND_PENALTY', '3.0'))  # Against major trend
        self.volume_penalty = float(os.getenv('VOLUME_PENALTY', '2.0'))      # No volume = fake move
        self.htf_penalty = float(os.getenv('HTF_PENALTY', '3.0'))         # Against higher timeframe
        
        # Technical indicator periods
        self.rsi_period = 14
        self.ema_fast = 21
        self.ema_slow = 50
        self.adx_period = 14
        self.atr_period = 14
        self.bb_period = 20
        
        # RSI thresholds (adaptive based on regime)
        self.rsi_oversold_base = 30
        self.rsi_overbought_base = 70
        
        # Volume confirmation threshold
        self.volume_multiplier = Decimal(os.getenv('VOLUME_CONFIRMATION_MULT', '1.2'))  # Require 20% above average
        
        # Initialize adaptive components
        self.regime_detector = MarketRegimeDetector()
        self.smc_analyzer = SmartMoneyAnalyzer()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.order_flow = OrderFlowAnalyzer()
        self.session_manager = SessionManager()
        self.risk_manager = AdaptiveRiskManager()
        self.pro_filters = ProTradingFilters(symbol)  # Professional trading filters
        
        # NEW: Pro-level enhancements
        self.vwap_calculator = VWAPCalculator()
        self.divergence_detector = DivergenceDetector()
        self.funding_filter = FundingRateFilter()
        
        # NEW: Trend-following indicators (from TradingView setup)
        # Supertrend: Clear trend direction with built-in volatility adjustment
        self.supertrend = SupertrendIndicator(
            period=int(os.getenv('SUPERTREND_PERIOD', '10')),
            multiplier=float(os.getenv('SUPERTREND_MULTIPLIER', '2.0'))
        )
        # Donchian Channel: Breakout detection and trend bias
        self.donchian = DonchianChannel(
            period=int(os.getenv('DONCHIAN_PERIOD', '50')),
            offset=0
        )
        
        # NEW: Professional momentum and volume indicators
        # StochRSI: More sensitive than RSI for overbought/oversold extremes
        self.stoch_rsi = StochRSICalculator(
            rsi_period=int(os.getenv('STOCH_RSI_PERIOD', '14')),
            stoch_period=14,  # Use same as RSI period
            k_smooth=int(os.getenv('STOCH_RSI_K', '3')),
            d_smooth=int(os.getenv('STOCH_RSI_D', '3'))
        )
        # OBV: Volume-price confirmation, detects accumulation/distribution
        self.obv_calculator = OBVCalculator()
        # CMF: Chaikin Money Flow - institutional buying/selling pressure
        self.cmf_calculator = ChaikinMoneyFlow(
            period=int(os.getenv('CMF_PERIOD', '20'))
        )
        
        # State tracking - BE PATIENT, DON'T OVERTRADE
        self.last_signal_time: Optional[datetime] = None
        self.signal_cooldown_seconds = int(os.getenv('SWING_COOLDOWN', '600'))  # 10 min between signals (was 5)
        self.recent_prices: deque = deque(maxlen=200)
        
        # ==================== WHIPSAW PROTECTION ====================
        # Prevents rapid direction changes that destroy accounts
        
        # Signal confirmation - requires signal to persist across multiple scans
        self.signal_confirmation_required = int(os.getenv('SIGNAL_CONFIRMATION_SCANS', '3'))  # Need 3 consecutive confirmations
        self._pending_signal: Optional[Dict] = None  # Direction waiting for confirmation
        self._pending_signal_time: Optional[datetime] = None  # CRITICAL FIX: Track when signal started
        self._confirmation_count = 0  # How many times we've seen this direction
        self._last_confirmed_direction: Optional[str] = None  # Last direction we actually traded
        
        # CRITICAL FIX: Signal expiry timeout (prevents stale signals after gaps)
        self.signal_expiry_seconds = int(os.getenv('SIGNAL_EXPIRY_SECONDS', '300'))  # 5 min default
        
        # Direction lock - prevent flipping too quickly after a signal
        self.direction_lock_seconds = int(os.getenv('DIRECTION_LOCK_SECONDS', '900'))  # 15 min lock after signal
        self._direction_lock_until: Optional[datetime] = None
        self._locked_direction: Optional[str] = None
        
        # Score stability - track score history to detect erratic signals
        self._score_history: deque = deque(maxlen=10)  # Last 10 scores per direction
        self._long_score_history: deque = deque(maxlen=5)
        self._short_score_history: deque = deque(maxlen=5)
        
        # Indicator cache
        self._indicator_cache = {}
        self._cache_timestamp: Optional[datetime] = None
        
        # RSI smoothing state
        self.rsi_avg_gain: Optional[Decimal] = None
        self.rsi_avg_loss: Optional[Decimal] = None
        
        # RSI/MACD history for divergence detection
        self.rsi_history: deque = deque(maxlen=30)
        self.macd_history: deque = deque(maxlen=30)
        
        # Statistics
        self.signals_generated = 0
        self.trades_taken = 0
        
        # Log initialization
        logger.info(f"🌟 WORLD-CLASS Swing Strategy initialized for {symbol}")
        logger.info(f"   Leverage: {self.leverage}x")
        logger.info(f"   Base Position: {self.base_position_size}%")
        logger.info(f"   Signal Threshold: {self.min_signal_score}/{self.max_signal_score}")
        logger.info(f"   Components: Regime, SMC, MTF, OrderFlow, Sessions, AdaptiveRisk")
    
    async def generate_signal(
        self,
        market_data: Dict[str, Any],
        account_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal with full analysis.
        
        Args:
            market_data: Market data containing candles
            account_state: Current account state
        
        Returns:
            Signal dict or None if no trade
        """
        # Extract candles from market_data
        candles = market_data.get('candles', [])
        htf_candles = market_data.get('htf_candles')
        
        if not candles or len(candles) < 100:
            return None
        
        # Check cooldown
        if not self._check_cooldown():
            return None
        
        # Extract prices
        prices = [Decimal(str(c.get('close', c.get('c', 0)))) for c in candles]
        current_price = prices[-1]
        
        # ==================== ANALYSIS LAYERS ====================
        
        # 1. Calculate technical indicators
        indicators = self._calculate_indicators(candles)
        if not indicators:
            return None
        
        # Store indicator history for divergence detection
        rsi = indicators.get('rsi')
        macd = indicators.get('macd', {})
        if rsi:
            self.rsi_history.append(rsi)
        if macd:
            self.macd_history.append(macd)
        
        # 2. Detect market regime
        regime, regime_confidence, regime_params = self.regime_detector.detect_regime(
            candles,
            adx=indicators.get('adx'),
            atr=indicators.get('atr'),
            bb_bandwidth=indicators.get('bb_bandwidth'),
            ema_fast=indicators.get('ema_fast'),
            ema_slow=indicators.get('ema_slow'),
        )
        
        # CRITICAL: Don't trade when regime is UNKNOWN (insufficient data/analysis)
        from app.strategies.adaptive.market_regime import MarketRegime
        if regime == MarketRegime.UNKNOWN:
            logger.debug(f"⏸️ Regime UNKNOWN for {self.symbol} - skipping signal generation")
            return None
        
        # 3. Smart Money Concepts analysis
        smc_analysis = self.smc_analyzer.analyze(candles)
        
        # 4. Multi-timeframe analysis
        if htf_candles:
            for interval, htf_data in htf_candles.items():
                if htf_data:
                    self.mtf_analyzer.analyze_timeframe(htf_data, interval)
        
        # 5. Order flow analysis
        of_analysis = self.order_flow.analyze_from_candles(candles)
        
        # 6. Session parameters
        session_params = self.session_manager.get_session_params()
        
        # ==================== SIGNAL GENERATION ====================
        
        # Calculate directional scores
        long_score = self._calculate_signal_score(
            direction='long',
            indicators=indicators,
            regime=regime,
            regime_params=regime_params,
            smc_analysis=smc_analysis,
            of_analysis=of_analysis,
            current_price=current_price,
        )
        
        short_score = self._calculate_signal_score(
            direction='short',
            indicators=indicators,
            regime=regime,
            regime_params=regime_params,
            smc_analysis=smc_analysis,
            of_analysis=of_analysis,
            current_price=current_price,
        )
        
        # Apply enhanced scoring (VWAP, Divergence, Volume)
        long_enhanced, long_details = self._calculate_enhanced_score('long', candles, indicators, long_score)
        short_enhanced, short_details = self._calculate_enhanced_score('short', candles, indicators, short_score)
        
        # Track score history for stability analysis
        self._long_score_history.append(long_enhanced)
        self._short_score_history.append(short_enhanced)
        
        # ==================== ADAPTIVE THRESHOLD ====================
        # Lower threshold in ranging markets (mean reversion is more reliable)
        # Higher threshold in trending markets (need stronger confirmation)
        from app.strategies.adaptive.market_regime import MarketRegime
        if regime == MarketRegime.RANGING:
            effective_threshold = max(8, self.min_signal_score - self.ranging_threshold_reduction)
        elif regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            effective_threshold = self.min_signal_score + 2  # Stricter in trends
        else:
            effective_threshold = self.min_signal_score
        
        # OPTIMIZED LOGGING: Only log when scores change significantly or approaching threshold
        # This prevents log spam (was logging every 2 seconds!)
        score_changed = (
            not hasattr(self, '_last_logged_scores') or
            abs(long_enhanced - self._last_logged_scores.get('long', 0)) >= 2 or
            abs(short_enhanced - self._last_logged_scores.get('short', 0)) >= 2 or
            long_enhanced >= effective_threshold - 3 or  # Getting close to threshold
            short_enhanced >= effective_threshold - 3
        )
        
        if score_changed and (long_enhanced > 5 or short_enhanced > 5):
            logger.info(f"📊 Scores: LONG={long_enhanced}/{self.max_signal_score} | SHORT={short_enhanced}/{self.max_signal_score} | Regime={regime.value} | Threshold={effective_threshold}")
            self._last_logged_scores = {'long': long_enhanced, 'short': short_enhanced}
        
        # Determine best direction
        if long_enhanced >= effective_threshold and long_enhanced > short_enhanced:
            direction = 'long'
            score = long_enhanced
            score_details = long_details
            logger.info(f"✅ LONG wins: {long_enhanced} vs SHORT {short_enhanced} (threshold={effective_threshold})")
        elif short_enhanced >= effective_threshold and short_enhanced > long_enhanced:
            direction = 'short'
            score = short_enhanced
            score_details = short_details
            logger.info(f"✅ SHORT wins: {short_enhanced} vs LONG {long_enhanced} (threshold={effective_threshold})")
        else:
            # No valid signal - reset confirmation
            self._pending_signal = None
            self._confirmation_count = 0
            if long_enhanced > 0 or short_enhanced > 0:
                logger.debug(f"⏳ No signal: LONG={long_enhanced}/{effective_threshold}, SHORT={short_enhanced}/{effective_threshold} (need {effective_threshold}+)")
            return None
        
        # ==================== CRITICAL: HARD COUNTER-TREND BLOCK ====================
        # This is a HARD REJECTION, not just a penalty. Counter-trend trades are the
        # primary cause of losses in trending markets.
        # NOTE: This does NOT apply in RANGING - ranging allows both directions
        if direction == 'long' and regime == MarketRegime.TRENDING_DOWN:
            logger.warning(f"🚫 HARD BLOCK: Cannot LONG in TRENDING_DOWN regime - rejecting signal")
            self._pending_signal = None
            self._confirmation_count = 0
            return None
        elif direction == 'short' and regime == MarketRegime.TRENDING_UP:
            logger.warning(f"🚫 HARD BLOCK: Cannot SHORT in TRENDING_UP regime - rejecting signal")
            self._pending_signal = None
            self._confirmation_count = 0
            return None
        
        # ==================== SUPERTREND HARD BLOCK ====================
        # If supertrend is strongly against the trade direction, reject
        st_result = self.supertrend.calculate(candles)
        if st_result:
            st_against = (
                (direction == 'long' and st_result.direction == SupertrendDirection.BEARISH) or
                (direction == 'short' and st_result.direction == SupertrendDirection.BULLISH)
            )
            if st_against and st_result.strength > 1.5:  # Strong trend against us
                logger.warning(f"🚫 HARD BLOCK: {direction.upper()} against strong Supertrend ({st_result.direction.value}, strength={st_result.strength:.1f})")
                self._pending_signal = None
                self._confirmation_count = 0
                return None
        
        # ==================== WHIPSAW PROTECTION ====================
        
        # 1. Direction Lock Check - adaptive based on volatility
        now = datetime.now(timezone.utc)
        if self._direction_lock_until and now < self._direction_lock_until:
            if self._locked_direction and direction != self._locked_direction:
                # Check if we should override lock due to high volatility reversal
                override_lock = self._should_override_direction_lock(
                    direction, score, indicators, current_price
                )
                if not override_lock:
                    remaining = (self._direction_lock_until - now).total_seconds()
                    logger.debug(f"🔒 Direction locked to {self._locked_direction} for {remaining:.0f}s more - ignoring {direction}")
                    return None
                else:
                    logger.info(f"⚡ Direction lock OVERRIDDEN due to high volatility reversal signal")
        
        # 2. Score Stability Check - detect erratic signals
        if not self._check_score_stability(direction, score):
            logger.debug(f"📉 Score unstable for {direction} - waiting for consistency")
            return None
        
        # 3. Signal Confirmation - require consistent signals across multiple scans
        confirmed_signal = self._confirm_signal(direction, score, current_price, indicators)
        if not confirmed_signal:
            return None  # Still building confirmation
        
        # Check HTF alignment
        htf_aligned, htf_score, htf_reason = self.mtf_analyzer.should_take_trade(direction)
        if not htf_aligned and htf_score < 0.4:
            logger.debug(f"❌ Signal rejected: HTF misalignment ({htf_reason})")
            return None
        
        # Check session
        should_trade, session_reason = self.session_manager.should_trade()
        if not should_trade:
            logger.debug(f"❌ Signal rejected: {session_reason}")
            return None
        
        # ==================== PRO TRADING FILTERS ====================
        # Final quality gate - professional-level confirmation
        
        btc_candles = market_data.get('btc_candles')  # BTC correlation check
        pro_result = self.pro_filters.check_all(
            direction=direction,
            candles=candles,
            indicators=indicators,
            btc_candles=btc_candles,
        )
        
        if not pro_result.passed:
            logger.debug(f"❌ Signal rejected by pro filter: {pro_result.reason}")
            return None
        
        logger.info(f"✅ Pro filters passed: {pro_result.reason} (confidence: {pro_result.confidence:.1%})")
        
        # ==================== RISK CALCULATION ====================
        
        atr = indicators.get('atr')
        if atr is None or atr <= 0:
            logger.warning(f"⚠️ Invalid ATR for {self.symbol}, using fallback")
            atr = current_price * Decimal('0.01')  # 1% of price as fallback
        
        # Ensure ATR is Decimal
        atr = Decimal(str(atr)) if not isinstance(atr, Decimal) else atr
        
        # Get adaptive TP/SL levels
        risk_levels = self.risk_manager.calculate_adaptive_levels(
            entry_price=Decimal(str(current_price)),
            direction=direction,
            atr=atr,
            regime_params=regime_params,
            session_params=session_params,
        )
        
        # Apply session aggression to position size
        aggression = Decimal(str(session_params.get('aggression', 1.0)))
        position_size = self.base_position_size * aggression
        
        # Apply regime position size adjustment
        regime_size_mult = Decimal(str(regime_params.get('position_size_mult', 1.0)))
        position_size *= regime_size_mult
        
        # Cap at maximum
        max_size = Decimal(os.getenv('MAX_POSITION_SIZE_PCT', '55'))
        position_size = min(position_size, max_size)
        
        # ==================== BUILD SIGNAL ====================
        
        # Calculate size in tokens (approximate)
        account_value = Decimal(str(account_state.get('account_value', 1000)))
        size_usd = account_value * position_size / 100 * self.leverage
        size_tokens = size_usd / current_price
        
        signal = {
            'symbol': self.symbol,
            'direction': direction,
            'side': 'buy' if direction == 'long' else 'sell',  # For compatibility
            'signal_type': f'{direction.upper()} (Swing)',
            'entry_price': float(current_price),
            'stop_loss': risk_levels['stop_loss'],
            'take_profit': risk_levels['take_profit'],
            'position_size_pct': float(position_size),
            'size': float(size_tokens),  # Token size for order execution
            'leverage': self.leverage,
            
            # Scoring
            'signal_score': score,
            'max_score': self.max_signal_score,
            'score_pct': score / self.max_signal_score * 100,
            
            # Analysis results
            'regime': regime.value,
            'regime_confidence': regime_confidence,
            'htf_alignment': htf_score,
            'htf_reason': htf_reason,
            'smc_bias': smc_analysis.get('bias'),
            'order_flow_bias': of_analysis.get('bias'),
            'session': session_params.get('session'),
            
            # Risk metrics
            'sl_pct': risk_levels['sl_pct'],
            'tp_pct': risk_levels['tp_pct'],
            'rr_ratio': risk_levels['rr_ratio'],
            'atr': float(atr),
            
            # Telegram display
            'reason': f"Regime: {regime.value}, Score: {score}/{self.max_signal_score}",
            
            # Metadata
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'strategy': 'Swing',
        }
        
        # Update state
        self.last_signal_time = datetime.now(timezone.utc)
        self.signals_generated += 1
        
        logger.info(f"🎯 SIGNAL: {direction.upper()} {self.symbol}")
        logger.info(f"   Score: {score}/{self.max_signal_score} ({score/self.max_signal_score*100:.0f}%)")
        logger.info(f"   Entry: ${current_price:.4f}")
        logger.info(f"   SL: ${risk_levels['stop_loss']:.4f} ({risk_levels['sl_pct']:.2f}%)")
        logger.info(f"   TP: ${risk_levels['take_profit']:.4f} ({risk_levels['tp_pct']:.2f}%)")
        logger.info(f"   R:R: {risk_levels['rr_ratio']:.1f}:1")
        logger.info(f"   Regime: {regime.value} | HTF: {htf_score:.0%} | SMC: {smc_analysis.get('bias')}")
        
        return signal
    
    def _calculate_signal_score(
        self,
        direction: str,
        indicators: Dict,
        regime: MarketRegime,
        regime_params: Dict,
        smc_analysis: Dict,
        of_analysis: Dict,
        current_price: Decimal,
    ) -> int:
        """
        Calculate comprehensive signal score.
        
        Scoring (0-10):
        - Technical: 0-4 points
        - SMC: 0-2 points
        - HTF: 0-2 points
        - Order Flow: 0-2 points
        """
        score = 0
        
        # ========== TECHNICAL (0-4 points) ==========
        rsi = indicators.get('rsi')
        ema_fast = indicators.get('ema_fast')
        ema_slow = indicators.get('ema_slow')
        adx = indicators.get('adx')
        macd = indicators.get('macd', {})
        
        if direction == 'long':
            # RSI condition (0-1 point) - STRICT for high win rate
            # Only give full points for genuinely favorable conditions
            if rsi and rsi < 40:
                score += 1  # Truly oversold - excellent entry
            elif rsi and rsi < 55:
                score += 0.5  # Mildly oversold - acceptable
            # RSI > 55 = no points (too extended for quality entry)
            
            # EMA alignment (0-1 point)
            if ema_fast and ema_slow and ema_fast > ema_slow:
                score += 1
            
            # ADX trend strength (0-1 point)
            if adx and adx > 25:  # Raised from 20 to 25 for stronger trends
                if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                    score += 1
                else:
                    score += 0.5
            
            # MACD (0-1 point)
            if macd.get('histogram') and macd['histogram'] > 0:
                score += 1
            elif macd.get('macd') and macd.get('signal') and macd['macd'] > macd['signal']:
                score += 0.5
        
        else:  # short
            # RSI condition - STRICT for high win rate
            if rsi and rsi > 60:
                score += 1  # Truly overbought - excellent short entry
            elif rsi and rsi > 45:
                score += 0.5  # Mildly overbought - acceptable
            # RSI < 45 = no points (too extended for quality short)
            
            # EMA alignment
            if ema_fast and ema_slow and ema_fast < ema_slow:
                score += 1
            
            # ADX trend strength
            if adx and adx > 20:
                if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                    score += 1
                else:
                    score += 0.5
            
            # MACD
            if macd.get('histogram') and macd['histogram'] < 0:
                score += 1
            elif macd.get('macd') and macd.get('signal') and macd['macd'] < macd['signal']:
                score += 0.5
        
        # ========== SMART MONEY (0-2 points) ==========
        smc_bias = smc_analysis.get('bias')
        smc_signals = smc_analysis.get('signals', [])
        
        # Bias alignment
        if (direction == 'long' and smc_bias == 'bullish') or \
           (direction == 'short' and smc_bias == 'bearish'):
            score += 1
        
        # SMC signals
        for sig in smc_signals:
            if sig.get('direction') == direction:
                if sig.get('type') == 'liquidity_sweep':
                    score += 1  # Strongest signal
                elif sig.get('type') in ['fvg_fill', 'order_block']:
                    score += 0.5
        
        # Cap SMC at 2
        score = min(score, 6)  # Tech(4) + SMC(2)
        
        # ========== HTF ALIGNMENT (0-2 points) ==========
        htf_score, _ = self.mtf_analyzer.get_alignment_score(direction)
        score += htf_score * 2  # Scale 0-1 to 0-2
        
        # ========== ORDER FLOW (0-2 points) ==========
        of_bias = of_analysis.get('bias')
        
        if (direction == 'long' and of_bias == 'bullish') or \
           (direction == 'short' and of_bias == 'bearish'):
            score += 1
        
        # POC proximity bonus
        if of_analysis.get('poc_distance_pct') is not None:
            poc_dist = abs(of_analysis['poc_distance_pct'])
            if poc_dist < 0.5:  # Within 0.5% of POC
                score += 1
        
        # Whale activity
        whale_bias, buy_count, sell_count = self.order_flow.get_whale_bias()
        if (direction == 'long' and whale_bias == 'bullish') or \
           (direction == 'short' and whale_bias == 'bearish'):
            score += 0.5
        
        # ========== NEW: BREAK OF STRUCTURE (0-1.5 points) ==========
        bos_score, bos_reason = self.smc_analyzer.get_bos_signal(direction)
        if bos_score > 0:
            score += bos_score
            logger.debug(f"   BoS: +{bos_score:.1f} ({bos_reason})")
        elif bos_score < 0:
            score += bos_score  # Penalty
            logger.debug(f"   BoS: {bos_score:.1f} ({bos_reason})")
        
        return int(score)
    
    def _calculate_enhanced_score(
        self,
        direction: str,
        candles: List[Dict],
        indicators: Dict,
        base_score: int,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Calculate enhanced score with pro-level indicators.
        
        Scoring Layers:
        - Supertrend alignment: 0-2 points (trend direction), -1.5 against
        - Donchian position: 0-1.5 points (breakout/trend)
        - VWAP confluence: 0-1.5 points
        - Divergence (RSI/MACD): 0-2 points
        - Volume confirmation: 0-1 point, -0.5 weak
        - StochRSI: 0-1.5 points (sensitive overbought/oversold)
        - OBV: 0-1.5 points, -1 for divergence
        - CMF: 0-1.5 points (institutional money flow)
        
        Total max additional: ~13 points
        
        PENALTY SYSTEM (critical failures):
        - Counter-trend (regime): -5 points
        - Against Supertrend: -3 points
        - Weak volume: -2 points
        - HTF misalignment: -3 points
        
        Args:
            direction: 'long' or 'short'
            candles: OHLCV candles
            indicators: Calculated indicators
            base_score: Score from _calculate_signal_score
            
        Returns:
            Tuple of (enhanced_score, details)
        """
        score = base_score
        details = {'base_score': base_score, 'penalties': []}
        
        # ========== REGIME ALIGNMENT CHECK (CRITICAL) ==========
        # Counter-trend trading is DANGEROUS - heavy penalty
        from app.strategies.adaptive.market_regime import MarketRegime
        regime_result = self.regime_detector.detect_regime(candles)
        # detect_regime returns (regime_enum, confidence, params) tuple
        regime = regime_result[0] if isinstance(regime_result, tuple) else regime_result
        
        if direction == 'long' and regime == MarketRegime.TRENDING_DOWN:
            score -= self.regime_penalty  # -5 points
            details['penalties'].append({'type': 'regime', 'score': -self.regime_penalty, 'reason': 'LONG in downtrend'})
            logger.debug(f"   ⛔ REGIME PENALTY: -{self.regime_penalty} (LONG against TRENDING_DOWN)")
        elif direction == 'short' and regime == MarketRegime.TRENDING_UP:
            score -= self.regime_penalty  # -5 points
            details['penalties'].append({'type': 'regime', 'score': -self.regime_penalty, 'reason': 'SHORT in uptrend'})
            logger.debug(f"   ⛔ REGIME PENALTY: -{self.regime_penalty} (SHORT against TRENDING_UP)")
        elif (direction == 'long' and regime == MarketRegime.TRENDING_UP) or \
             (direction == 'short' and regime == MarketRegime.TRENDING_DOWN):
            score += 2.0  # Bonus for trend alignment
            details['regime_bonus'] = {'score': 2.0, 'reason': 'Trading WITH trend'}
            logger.debug(f"   ✅ Regime: +2.0 (Trading WITH {regime.value})")
        
        # ========== SUPERTREND (CRITICAL) ==========
        # This is a key trend filter - trading against supertrend is risky
        st_result = self.supertrend.calculate(candles)
        if st_result:
            st_aligned = (
                (direction == 'long' and st_result.direction == SupertrendDirection.BULLISH) or
                (direction == 'short' and st_result.direction == SupertrendDirection.BEARISH)
            )
            
            if st_aligned:
                # Aligned with supertrend - bonus points
                st_score = 1.5
                if st_result.changed:  # Fresh flip = extra confidence
                    st_score += 0.5
                if st_result.strength > 1.0:  # Strong trend
                    st_score += 0.5
                score += st_score
                details['supertrend'] = {
                    'aligned': True, 
                    'direction': st_result.direction.value,
                    'score': st_score,
                    'strength': st_result.strength
                }
                logger.debug(f"   ✅ Supertrend: +{st_score:.1f} ({st_result.direction.value}, strength={st_result.strength:.1f}%)")
            else:
                # Against supertrend - HEAVY PENALTY (this is dangerous!)
                score -= self.supertrend_penalty  # -3 points
                details['penalties'].append({'type': 'supertrend', 'score': -self.supertrend_penalty, 'reason': f'Against {st_result.direction.value}'})
                details['supertrend'] = {
                    'aligned': False,
                    'direction': st_result.direction.value,
                    'score': -self.supertrend_penalty,
                    'warning': 'Trading against trend!'
                }
                logger.debug(f"   ⛔ SUPERTREND PENALTY: -{self.supertrend_penalty} (AGAINST {st_result.direction.value} trend!)")
        
        # ========== DONCHIAN CHANNEL (0-1.5 points) ==========
        dc_result = self.donchian.calculate(candles)
        if dc_result:
            dc_score = 0.0
            dc_reason = ""
            
            # Breakout signals are very strong
            if dc_result.breakout == 'bullish' and direction == 'long':
                dc_score = 1.5
                dc_reason = "Bullish breakout!"
            elif dc_result.breakout == 'bearish' and direction == 'short':
                dc_score = 1.5
                dc_reason = "Bearish breakdown!"
            # Position-based scoring
            elif direction == 'long':
                if dc_result.position == DonchianPosition.UPPER_ZONE:
                    dc_score = 0.5
                    dc_reason = "Upper zone (bullish)"
                elif dc_result.position == DonchianPosition.ABOVE_UPPER:
                    dc_score = 1.0
                    dc_reason = "Above upper band"
                elif dc_result.position == DonchianPosition.LOWER_ZONE:
                    dc_score = -0.5
                    dc_reason = "Lower zone (bearish bias)"
            else:  # short
                if dc_result.position == DonchianPosition.LOWER_ZONE:
                    dc_score = 0.5
                    dc_reason = "Lower zone (bearish)"
                elif dc_result.position == DonchianPosition.BELOW_LOWER:
                    dc_score = 1.0
                    dc_reason = "Below lower band"
                elif dc_result.position == DonchianPosition.UPPER_ZONE:
                    dc_score = -0.5
                    dc_reason = "Upper zone (bullish bias)"
            
            # Squeeze detection - volatility expansion coming
            if dc_result.squeeze:
                dc_score += 0.5
                dc_reason += " [SQUEEZE]"
            
            if dc_score != 0:
                score += dc_score
                details['donchian'] = {
                    'score': dc_score,
                    'position': dc_result.position.value,
                    'reason': dc_reason,
                    'squeeze': dc_result.squeeze,
                    'width_pct': dc_result.width_pct
                }
                logger.debug(f"   Donchian: {dc_score:+.1f} ({dc_reason})")
        
        # ========== VWAP CONFLUENCE (0-1.5 points) ==========
        vwap_analysis = self.vwap_calculator.calculate_from_candles(candles)
        vwap_score, vwap_reason = self.vwap_calculator.get_vwap_signal(direction, vwap_analysis)
        if vwap_score != 0:
            score += vwap_score
            details['vwap'] = {'score': vwap_score, 'reason': vwap_reason}
            logger.debug(f"   VWAP: +{vwap_score:.1f} ({vwap_reason})")
        
        # ========== DIVERGENCE (0-2 points) ==========
        if len(self.rsi_history) >= 15 and len(self.macd_history) >= 15:
            div_analysis = self.divergence_detector.detect_all(
                candles, 
                list(self.rsi_history), 
                list(self.macd_history)
            )
            div_score, div_reason = self.divergence_detector.get_divergence_score(direction)
            if div_score != 0:
                score += div_score
                details['divergence'] = {'score': div_score, 'reason': div_reason}
                logger.debug(f"   Divergence: +{div_score:.1f} ({div_reason})")
        
        # ========== VOLUME CONFIRMATION (CRITICAL) ==========
        # "Volume is truth" - no volume = fake move
        volume_ok, volume_ratio = self._check_volume_confirmation(candles)
        if volume_ok:
            score += 1.5  # Increased bonus for volume confirmation
            details['volume'] = {'confirmed': True, 'ratio': volume_ratio}
            logger.debug(f"   ✅ Volume: +1.5 (ratio: {volume_ratio:.1f}x)")
        else:
            # Weak volume is a serious warning - PENALTY
            score -= self.volume_penalty  # -2 points
            details['penalties'].append({'type': 'volume', 'score': -self.volume_penalty, 'reason': f'Weak volume ({volume_ratio:.1f}x)'})
            details['volume'] = {'confirmed': False, 'ratio': volume_ratio}
            logger.debug(f"   ⛔ VOLUME PENALTY: -{self.volume_penalty} (weak: {volume_ratio:.1f}x)")
        
        # ========== STOCH RSI (0-1.5 points) ==========
        # More sensitive than regular RSI for detecting extreme conditions
        stoch_result = self.stoch_rsi.calculate(candles)
        if stoch_result:
            stoch_score = 0.0
            stoch_reason = ""
            
            # Score based on zone and crossover alignment with direction
            if direction == 'long':
                if stoch_result.zone == 'oversold':
                    stoch_score = 1.0
                    stoch_reason = f"Oversold (K={stoch_result.k_line:.1f})"
                    if stoch_result.crossover == 'bullish':
                        stoch_score = 1.5
                        stoch_reason += " + bullish crossover"
                elif stoch_result.zone == 'overbought':
                    stoch_score = -0.5
                    stoch_reason = f"Overbought warning (K={stoch_result.k_line:.1f})"
            else:  # short
                if stoch_result.zone == 'overbought':
                    stoch_score = 1.0
                    stoch_reason = f"Overbought (K={stoch_result.k_line:.1f})"
                    if stoch_result.crossover == 'bearish':
                        stoch_score = 1.5
                        stoch_reason += " + bearish crossover"
                elif stoch_result.zone == 'oversold':
                    stoch_score = -0.5
                    stoch_reason = f"Oversold warning (K={stoch_result.k_line:.1f})"
            
            if stoch_score != 0:
                score += stoch_score
                details['stoch_rsi'] = {
                    'score': stoch_score,
                    'k': stoch_result.k_line,
                    'd': stoch_result.d_line,
                    'zone': stoch_result.zone,
                    'crossover': stoch_result.crossover,
                    'reason': stoch_reason
                }
                logger.debug(f"   StochRSI: {stoch_score:+.1f} ({stoch_reason})")
        
        # ========== OBV - On Balance Volume (0-1.5 points, -1 for divergence) ==========
        # Volume-price confirmation from institutional trading
        obv_result = self.obv_calculator.calculate(candles)
        if obv_result:
            obv_score = 0.0
            obv_reason = ""
            
            # Divergence is a strong reversal signal
            if obv_result.divergence == 'bullish' and direction == 'long':
                obv_score = 1.5
                obv_reason = "Bullish OBV divergence (accumulation)"
            elif obv_result.divergence == 'bearish' and direction == 'short':
                obv_score = 1.5
                obv_reason = "Bearish OBV divergence (distribution)"
            # Divergence against our direction is a warning
            elif obv_result.divergence == 'bullish' and direction == 'short':
                obv_score = -1.0
                obv_reason = "⚠️ Bullish divergence vs short"
            elif obv_result.divergence == 'bearish' and direction == 'long':
                obv_score = -1.0
                obv_reason = "⚠️ Bearish divergence vs long"
            # Trend confirmation
            elif direction == 'long' and obv_result.trend == 'rising':
                obv_score = 1.0 if obv_result.strength > 0.5 else 0.5
                obv_reason = f"OBV rising (strength={obv_result.strength:.1f})"
            elif direction == 'short' and obv_result.trend == 'falling':
                obv_score = 1.0 if obv_result.strength > 0.5 else 0.5
                obv_reason = f"OBV falling (strength={obv_result.strength:.1f})"
            # Against trend
            elif direction == 'long' and obv_result.trend == 'falling':
                obv_score = -0.5
                obv_reason = "OBV falling vs long"
            elif direction == 'short' and obv_result.trend == 'rising':
                obv_score = -0.5
                obv_reason = "OBV rising vs short"
            
            if obv_score != 0:
                score += obv_score
                details['obv'] = {
                    'score': obv_score,
                    'trend': obv_result.trend,
                    'strength': obv_result.strength,
                    'divergence': obv_result.divergence,
                    'reason': obv_reason
                }
                if obv_score > 0:
                    logger.debug(f"   OBV: {obv_score:+.1f} (trend={obv_result.trend}, {obv_reason})")
                else:
                    logger.debug(f"   OBV: {obv_score:+.1f} ⚠️ ({obv_reason})")
        
        # ========== CMF - Chaikin Money Flow (0-1.5 points) ==========
        # Institutional buying/selling pressure
        cmf_result = self.cmf_calculator.calculate(candles)
        if cmf_result:
            cmf_score = 0.0
            cmf_reason = ""
            
            # Divergence is a strong signal
            if cmf_result.divergence == 'bullish' and direction == 'long':
                cmf_score = 1.5
                cmf_reason = "Bullish CMF divergence"
            elif cmf_result.divergence == 'bearish' and direction == 'short':
                cmf_score = 1.5
                cmf_reason = "Bearish CMF divergence"
            # Zone-based scoring
            elif direction == 'long':
                if cmf_result.zone == 'strong_buy':
                    cmf_score = 1.0
                    cmf_reason = f"Strong buying ({cmf_result.cmf:.2f})"
                elif cmf_result.zone == 'buy':
                    cmf_score = 0.5
                    cmf_reason = f"Buying pressure ({cmf_result.cmf:.2f})"
                elif cmf_result.zone in ['sell', 'strong_sell']:
                    cmf_score = -0.5
                    cmf_reason = f"Against money flow ({cmf_result.cmf:.2f})"
            else:  # short
                if cmf_result.zone == 'strong_sell':
                    cmf_score = 1.0
                    cmf_reason = f"Strong selling ({cmf_result.cmf:.2f})"
                elif cmf_result.zone == 'sell':
                    cmf_score = 0.5
                    cmf_reason = f"Selling pressure ({cmf_result.cmf:.2f})"
                elif cmf_result.zone in ['buy', 'strong_buy']:
                    cmf_score = -0.5
                    cmf_reason = f"Against money flow ({cmf_result.cmf:.2f})"
            
            if cmf_score != 0:
                score += cmf_score
                details['cmf'] = {
                    'score': cmf_score,
                    'cmf': cmf_result.cmf,
                    'zone': cmf_result.zone,
                    'trend': cmf_result.trend,
                    'divergence': cmf_result.divergence,
                    'reason': cmf_reason
                }
                logger.debug(f"   CMF: {cmf_score:+.1f} ({cmf_reason})")
        
        # ========== FINAL SCORE CALCULATION ==========
        # Floor at 0 (can't go negative) and cap at max_signal_score
        final_score = max(0, min(int(score), self.max_signal_score))
        
        # Calculate total penalties for logging
        total_penalties = sum(p['score'] for p in details.get('penalties', []))
        details['total_penalties'] = total_penalties
        details['raw_score'] = int(score)
        details['final_score'] = final_score
        
        # Only log score summary at debug level (too verbose for info)
        logger.debug(f"   📊 Score: {base_score} (base) + bonuses - {abs(total_penalties):.0f} (penalties) = {final_score}/{self.max_signal_score}")
        
        if final_score > self.max_signal_score:
            logger.debug(f"   Score capped: {int(score)} → {final_score}")
        
        return final_score, details
    
    def _check_volume_confirmation(self, candles: List[Dict]) -> Tuple[bool, float]:
        """
        Check if current volume is above average (confirms move is real).
        
        VOLUME IS TRUTH: No volume = fake move.
        
        Args:
            candles: OHLCV candles
            
        Returns:
            Tuple of (is_confirmed, volume_ratio)
        """
        if len(candles) < 20:
            return True, 1.0  # Not enough data, pass
        
        volumes = [float(c.get('volume', c.get('v', 0))) for c in candles]
        avg_volume = sum(volumes[-20:]) / 20
        current_volume = volumes[-1]
        
        if avg_volume <= 0:
            return True, 1.0
        
        volume_ratio = current_volume / avg_volume
        is_confirmed = Decimal(str(volume_ratio)) >= self.volume_multiplier
        
        return is_confirmed, volume_ratio
    
    def _calculate_indicators(self, candles: List[Dict]) -> Optional[Dict]:
        """Calculate all technical indicators."""
        prices = [Decimal(str(c.get('close', c.get('c', 0)))) for c in candles]
        
        if len(prices) < 50:
            return None
        
        return {
            'rsi': self._calculate_rsi(prices),
            'ema_fast': self._calculate_ema(prices, self.ema_fast),
            'ema_slow': self._calculate_ema(prices, self.ema_slow),
            'adx': self._calculate_adx(candles),
            'atr': self._calculate_atr(candles),
            'macd': self._calculate_macd(prices),
            'bb_bandwidth': self._calculate_bb_bandwidth(prices),
        }
    
    def _calculate_rsi(self, prices: List[Decimal], period: int = 14) -> Optional[Decimal]:
        """Calculate RSI with Wilder's smoothing."""
        if len(prices) < period + 1:
            return None
        
        # Ensure prices are Decimal
        decimal_prices = [Decimal(str(p)) if not isinstance(p, Decimal) else p for p in prices]
        period_dec = Decimal(str(period))
        
        current_change = decimal_prices[-1] - decimal_prices[-2]
        current_gain = max(current_change, Decimal('0'))
        current_loss = abs(min(current_change, Decimal('0')))
        
        if self.rsi_avg_gain is None:
            changes = [decimal_prices[i] - decimal_prices[i-1] for i in range(-period, 0)]
            gains = [max(c, Decimal('0')) for c in changes]
            losses = [abs(min(c, Decimal('0'))) for c in changes]
            self.rsi_avg_gain = sum(gains) / period_dec
            self.rsi_avg_loss = sum(losses) / period_dec
        else:
            self.rsi_avg_gain = (self.rsi_avg_gain * (period_dec - Decimal('1')) + current_gain) / period_dec
            self.rsi_avg_loss = (self.rsi_avg_loss * (period_dec - Decimal('1')) + current_loss) / period_dec
        
        if self.rsi_avg_loss == 0:
            return Decimal('100')
        
        rs = self.rsi_avg_gain / self.rsi_avg_loss
        return Decimal('100') - (Decimal('100') / (Decimal('1') + rs))
    
    def _calculate_ema(self, prices: List[Decimal], period: int) -> Optional[Decimal]:
        """Calculate EMA."""
        if len(prices) < period:
            return None
        
        # Ensure all arithmetic uses Decimal to avoid float/Decimal mixing
        multiplier = Decimal('2') / Decimal(str(period + 1))
        
        # Ensure prices are all Decimal
        decimal_prices = [Decimal(str(p)) if not isinstance(p, Decimal) else p for p in prices]
        
        ema = sum(decimal_prices[:period]) / Decimal(str(period))
        
        for price in decimal_prices[period:]:
            ema = (price * multiplier) + (ema * (Decimal('1') - multiplier))
        
        return ema
    
    def _calculate_macd(self, prices: List[Decimal]) -> Dict:
        """Calculate MACD with proper signal line (EMA of MACD values)."""
        # Need at least 26 + 9 = 35 prices to calculate signal line
        if len(prices) < 35:
            # Fallback: calculate just MACD line without proper signal
            ema_12 = self._calculate_ema(prices, 12)
            ema_26 = self._calculate_ema(prices, 26)
            if not ema_12 or not ema_26:
                return {}
            macd_line = ema_12 - ema_26
            return {
                'macd': macd_line,
                'signal': macd_line,  # No proper signal available
                'histogram': Decimal('0'),
            }
        
        # Calculate MACD values for the last 9 periods to build signal line
        macd_values = []
        for i in range(9):
            # Use prices up to position -(8-i) from the end
            # i=0: prices[:-8], i=1: prices[:-7], ... i=8: prices[:] (all)
            end_idx = len(prices) - (8 - i) if i < 8 else len(prices)
            price_slice = prices[:end_idx]
            
            ema_12 = self._calculate_ema(price_slice, 12)
            ema_26 = self._calculate_ema(price_slice, 26)
            
            if ema_12 and ema_26:
                macd_values.append(ema_12 - ema_26)
        
        if len(macd_values) < 9:
            return {}
        
        # Current MACD line is the last value
        macd_line = macd_values[-1]
        
        # Signal line is EMA(9) of MACD values
        signal_line = self._calculate_ema(macd_values, 9) or macd_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line,
        }
    
    def _calculate_adx(self, candles: List[Dict], period: int = 14) -> Optional[Decimal]:
        """Calculate ADX with proper True Range (using H/L/C, not just close)."""
        if len(candles) < period * 2:
            return None
        
        tr_list, plus_dm_list, minus_dm_list = [], [], []
        
        for i in range(1, len(candles)):
            # Get OHLC data
            high = Decimal(str(candles[i].get('high', candles[i].get('h', 0))))
            low = Decimal(str(candles[i].get('low', candles[i].get('l', 0))))
            prev_close = Decimal(str(candles[i-1].get('close', candles[i-1].get('c', 0))))
            prev_high = Decimal(str(candles[i-1].get('high', candles[i-1].get('h', 0))))
            prev_low = Decimal(str(candles[i-1].get('low', candles[i-1].get('l', 0))))
            
            # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
            
            # Directional Movement using high/low
            up_move = high - prev_high
            down_move = prev_low - low
            
            # +DM: up move is greater than down move and positive
            plus_dm = up_move if (up_move > down_move and up_move > 0) else Decimal('0')
            # -DM: down move is greater than up move and positive
            minus_dm = down_move if (down_move > up_move and down_move > 0) else Decimal('0')
            
            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)
        
        if len(tr_list) < period:
            return None
        
        period_dec = Decimal(str(period))
        atr = sum(tr_list[-period:]) / period_dec
        plus_dm = sum(plus_dm_list[-period:]) / period_dec
        minus_dm = sum(minus_dm_list[-period:]) / period_dec
        
        plus_di = (plus_dm / atr * Decimal('100')) if atr > 0 else Decimal('0')
        minus_di = (minus_dm / atr * Decimal('100')) if atr > 0 else Decimal('0')
        
        di_sum = plus_di + minus_di
        di_diff = abs(plus_di - minus_di)
        
        return (di_diff / di_sum * Decimal('100')) if di_sum > 0 else Decimal('0')
    
    def _calculate_atr(self, candles: List[Dict], period: int = 14) -> Optional[Decimal]:
        """Calculate ATR."""
        if len(candles) < period:
            return None
        
        tr_list = []
        for i in range(1, len(candles)):
            high = Decimal(str(candles[i].get('high', candles[i].get('h', 0))))
            low = Decimal(str(candles[i].get('low', candles[i].get('l', 0))))
            prev_close = Decimal(str(candles[i-1].get('close', candles[i-1].get('c', 0))))
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
        
        return sum(tr_list[-period:]) / Decimal(str(period))
    
    def _calculate_bb_bandwidth(self, prices: List[Decimal], period: int = 20) -> Optional[Decimal]:
        """Calculate Bollinger Band bandwidth."""
        if len(prices) < period:
            return None
        
        recent = prices[-period:]
        period_dec = Decimal(str(period))
        sma = sum(recent) / period_dec
        variance = sum((p - sma) ** 2 for p in recent) / period_dec
        std = variance ** Decimal('0.5')
        
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        
        return ((upper - lower) / sma) * 100
    
    def _check_cooldown(self) -> bool:
        """Check if signal cooldown has passed."""
        if not self.last_signal_time:
            return True
        
        elapsed = (datetime.now(timezone.utc) - self.last_signal_time).total_seconds()
        return elapsed >= self.signal_cooldown_seconds
    
    def invalidate_indicator_cache(self):
        """Invalidate cache when new candle arrives."""
        self._indicator_cache = {}
        self._cache_timestamp = None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics for Telegram /stats command."""
        return {
            'strategy': 'Swing',
            'signals': self.signals_generated,
            'trades': self.signals_generated,  # Approximate
            'current_regime': self.regime_detector.current_regime.value if hasattr(self, 'regime_detector') else 'unknown',
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status for monitoring."""
        return {
            'strategy': 'Swing',
            'symbol': self.symbol,
            'signals_generated': self.signals_generated,
            'current_regime': self.regime_detector.current_regime.value,
            'regime_confidence': self.regime_detector.regime_confidence,
            'htf_bias': self.mtf_analyzer.get_htf_bias(),
            'session': self.session_manager.get_current_session().value,
            'cooldown_remaining': max(0, self.signal_cooldown_seconds - 
                (datetime.now(timezone.utc) - self.last_signal_time).total_seconds()
                if self.last_signal_time else 0),
        }
    
    def record_trade_execution(self, signal: Dict[str, Any], result: Dict[str, Any]):
        """
        Record a trade execution for strategy performance tracking.
        
        Args:
            signal: The signal that was executed
            result: The execution result from the order manager
        """
        # Defensive: ensure signal and result are dicts
        if not isinstance(signal, dict):
            logger.warning(f"record_trade_execution: signal is not a dict: {type(signal)}")
            return
        if not isinstance(result, dict):
            logger.warning(f"record_trade_execution: result is not a dict: {type(result)}")
            return
        
        # Track execution
        success = result.get('success', False) or result.get('status') == 'ok'
        
        if success:
            logger.info(f"✅ Trade executed for {signal.get('symbol', self.symbol)}: "
                       f"{signal.get('side', 'unknown').upper()} @ {signal.get('entry_price', 0)}")
        else:
            logger.warning(f"⚠️ Trade execution issue for {signal.get('symbol', self.symbol)}: {result}")
        
        # Could add more tracking here (trade history, performance metrics, etc.)
    
    # ==================== WHIPSAW PROTECTION METHODS ====================
    
    def _should_override_direction_lock(
        self, 
        direction: str, 
        score: float, 
        indicators: Dict,
        current_price: Decimal
    ) -> bool:
        """
        Check if direction lock should be overridden due to market conditions.
        
        Override conditions (more realistic):
        1. High volatility (ATR > 2%)
        2. Score meets threshold (7+) - not waiting for unicorn 9+
        3. RSI in favorable zone
        4. MACD confirms direction
        
        The idea: Lock is protection, but don't be stubborn when market clearly reversed.
        
        Args:
            direction: New signal direction
            score: New signal score
            indicators: Current indicators
            current_price: Current price
            
        Returns:
            True if lock should be overridden
        """
        # Score must at least meet our normal threshold
        if score < self.min_signal_score:
            return False
        
        override_reasons = []
        override_score = 0  # Need 3+ points to override
        
        # Check ATR for volatility (1 point)
        atr = indicators.get('atr')
        if atr:
            atr_pct = float(atr) / float(current_price) * 100
            if atr_pct >= 2.0:  # Elevated volatility
                override_score += 1
                override_reasons.append(f"ATR={atr_pct:.1f}%")
            if atr_pct >= 3.0:  # High volatility - extra point
                override_score += 1
                override_reasons.append("HIGH_VOL")
        
        # Check RSI for extreme conditions (1-2 points)
        rsi = indicators.get('rsi')
        if rsi:
            rsi_val = float(rsi)
            if direction == 'long':
                if rsi_val < 30:  # Very oversold
                    override_score += 2
                    override_reasons.append(f"RSI={rsi_val:.0f}(oversold)")
                elif rsi_val < 40:  # Somewhat oversold
                    override_score += 1
                    override_reasons.append(f"RSI={rsi_val:.0f}")
            else:  # short
                if rsi_val > 70:  # Very overbought
                    override_score += 2
                    override_reasons.append(f"RSI={rsi_val:.0f}(overbought)")
                elif rsi_val > 60:  # Somewhat overbought
                    override_score += 1
                    override_reasons.append(f"RSI={rsi_val:.0f}")
        
        # Check MACD direction (1 point)
        macd = indicators.get('macd', {})
        if macd:
            histogram = macd.get('histogram', 0)
            if histogram:
                if direction == 'long' and histogram > 0:
                    override_score += 1
                    override_reasons.append("MACD+")
                elif direction == 'short' and histogram < 0:
                    override_score += 1
                    override_reasons.append("MACD-")
        
        # Strong signal score bonus (1 point if score is 8+)
        if score >= 8:
            override_score += 1
            override_reasons.append(f"Score={score}")
        
        # Need at least 3 points to override lock
        if override_score >= 3:
            logger.info(f"⚡ Lock override: {' | '.join(override_reasons)} (override_score={override_score})")
            return True
        
        logger.debug(f"🔒 Lock maintained: override_score={override_score}/3 ({', '.join(override_reasons) or 'no signals'})")
        return False
    
    def _confirm_signal(self, direction: str, score: float, current_price: Decimal, indicators: Dict = None) -> bool:
        """
        Require signals to persist across multiple scans before triggering.
        This prevents acting on noise/whipsaw movements.
        
        In high volatility (ATR 2x+), reduce confirmation requirement.
        
        CRITICAL FIX: Signals expire after signal_expiry_seconds to prevent stale signals
        after market gaps or long pauses between scans.
        
        Args:
            direction: 'long' or 'short'
            score: Current signal score
            current_price: Current price
            indicators: Current indicators (for volatility check)
            
        Returns:
            True if signal is confirmed, False if still building confirmation
        """
        now = datetime.now(timezone.utc)
        
        # CRITICAL FIX: Check for signal expiry FIRST
        # Prevents confirming stale signals after market gaps
        if self._pending_signal is not None and self._pending_signal_time is not None:
            signal_age = (now - self._pending_signal_time).total_seconds()
            if signal_age > self.signal_expiry_seconds:
                logger.warning(f"⏰ Signal expired: {self._pending_signal.get('direction', 'unknown').upper()} "
                             f"was {signal_age:.0f}s old (max {self.signal_expiry_seconds}s)")
                self._pending_signal = None
                self._pending_signal_time = None
                self._confirmation_count = 0
        
        # Adaptive confirmation based on volatility
        confirmations_needed = self.signal_confirmation_required
        if indicators:
            atr = indicators.get('atr')
            if atr and current_price > 0:
                atr_pct = float(atr) / float(current_price) * 100
                if atr_pct > 3.0:  # High volatility
                    confirmations_needed = max(1, self.signal_confirmation_required - 1)
                    logger.debug(f"⚡ High volatility ({atr_pct:.2f}%) - reduced confirmations to {confirmations_needed}")
        
        # Check if this is a new direction or continuation
        if self._pending_signal is None or self._pending_signal.get('direction') != direction:
            # New direction - start confirmation process
            self._pending_signal = {
                'direction': direction,
                'first_seen': now,
                'price_at_first': float(current_price),
                'scores': [score],
            }
            self._pending_signal_time = now  # CRITICAL FIX: Track start time for expiry
            self._confirmation_count = 1
            logger.info(f"🔄 Signal confirmation started: {direction.upper()} (1/{confirmations_needed})")
            return False
        
        # Same direction - increment confirmation
        self._confirmation_count += 1
        self._pending_signal['scores'].append(score)
        
        # Check if price moved too much during confirmation (invalidates signal)
        first_price = Decimal(str(self._pending_signal['price_at_first']))
        price_change_pct = abs(current_price - first_price) / first_price * 100
        
        # Adaptive price threshold based on volatility
        max_price_move = Decimal('1.0')
        if indicators:
            atr = indicators.get('atr')
            if atr and current_price > 0:
                atr_pct = Decimal(str(float(atr) / float(current_price) * 100))
                max_price_move = min(Decimal('2.0'), atr_pct)  # Allow up to ATR% move, max 2%
        
        if price_change_pct > max_price_move:
            logger.info(f"❌ Confirmation reset: price moved {price_change_pct:.2f}% > {max_price_move:.2f}% during confirmation")
            self._pending_signal = None
            self._confirmation_count = 0
            return False
        
        # Check if we have enough confirmations (use adaptive count)
        if self._confirmation_count >= confirmations_needed:
            # Check average score during confirmation (guard against empty list)
            scores = self._pending_signal.get('scores', [])
            if not scores:
                logger.warning("⚠️ No scores in confirmation - resetting")
                self._pending_signal = None
                self._confirmation_count = 0
                return False
            
            avg_score = sum(scores) / len(scores)
            if avg_score < self.min_signal_score * 0.9:  # Must maintain 90% of threshold
                logger.info(f"❌ Confirmation failed: avg score {avg_score:.1f} < {self.min_signal_score * 0.9:.1f}")
                self._pending_signal = None
                self._confirmation_count = 0
                return False
            
            # Signal confirmed!
            logger.info(f"✅ Signal CONFIRMED: {direction.upper()} after {self._confirmation_count} scans (avg score: {avg_score:.1f})")
            
            # Set adaptive direction lock based on volatility
            # High volatility = shorter lock (market moves fast)
            lock_seconds = self.direction_lock_seconds
            if indicators:
                atr = indicators.get('atr')
                if atr and current_price > 0:
                    atr_pct = float(atr) / float(current_price) * 100
                    if atr_pct > 3.0:  # High volatility
                        lock_seconds = max(300, lock_seconds // 2)  # Reduce lock, min 5 min
                        logger.debug(f"⚡ High volatility - reduced direction lock to {lock_seconds}s")
            
            self._direction_lock_until = now + timedelta(seconds=lock_seconds)
            self._locked_direction = direction
            self._last_confirmed_direction = direction
            
            # Reset confirmation state (including time)
            self._pending_signal = None
            self._pending_signal_time = None
            self._confirmation_count = 0
            
            return True
        
        # Still building confirmation
        logger.info(f"🔄 Signal confirmation building: {direction.upper()} ({self._confirmation_count}/{self.signal_confirmation_required})")
        return False
    
    def _check_score_stability(self, direction: str, current_score: float) -> bool:
        """
        Check if scores are stable (not erratic).
        Erratic signals often indicate choppy/ranging markets that whipsaw.
        
        Args:
            direction: 'long' or 'short'
            current_score: Current signal score
            
        Returns:
            True if scores are stable, False if too erratic
        """
        # Get relevant history
        history = self._long_score_history if direction == 'long' else self._short_score_history
        
        if len(history) < 3:
            return True  # Not enough data, allow signal
        
        recent_scores = list(history)[-3:]
        
        # Check for large swings (volatility in scores = volatility in market)
        score_range = max(recent_scores) - min(recent_scores)
        if score_range > 4:  # More than 4 point swing in scores
            logger.debug(f"⚠️ Score instability detected: range={score_range:.1f} in last 3 scans")
            return False
        
        # Check if opposite direction was stronger recently
        opposite_history = self._short_score_history if direction == 'long' else self._long_score_history
        if len(opposite_history) >= 2:
            opposite_recent = list(opposite_history)[-2:]
            if max(opposite_recent) > current_score:
                # Opposite direction was stronger very recently - whipsaw risk
                logger.debug(f"⚠️ Whipsaw risk: opposite direction scored {max(opposite_recent):.1f} vs current {current_score:.1f}")
                return False
        
        return True
    
    def reset_whipsaw_protection(self):
        """Reset whipsaw protection state (call after position closes)."""
        self._pending_signal = None
        self._confirmation_count = 0
        self._direction_lock_until = None
        self._locked_direction = None
        self._long_score_history.clear()
        self._short_score_history.clear()
        logger.debug("🔄 Whipsaw protection reset")
    
    def revalidate_signal(self, signal: Dict[str, Any], current_price: Decimal) -> bool:
        """
        Revalidate a signal before execution.
        Ensures market conditions haven't changed significantly.
        
        Args:
            signal: The original signal
            current_price: Current market price
            
        Returns:
            True if signal is still valid, False if conditions changed
        """
        entry_price = Decimal(str(signal.get('entry_price', 0)))
        # Handle both 'side' and 'direction' keys, and both naming conventions
        side = signal.get('side', signal.get('direction', '')).lower()
        
        if entry_price <= 0:
            return True  # Can't validate without entry price
        
        # Calculate price deviation
        deviation_pct = abs(current_price - entry_price) / entry_price * 100
        
        # Allow up to 0.5% deviation for swing trades
        max_deviation = Decimal('0.5')
        
        if deviation_pct > max_deviation:
            logger.warning(f"Signal invalidated: price moved {deviation_pct:.2f}% from entry")
            return False
        
        # For longs/buys, price shouldn't have dropped too much (might indicate reversal)
        # For shorts/sells, price shouldn't have risen too much
        if side in ('buy', 'long') and current_price < entry_price * Decimal('0.995'):
            logger.warning(f"Long signal invalidated: price dropped below entry")
            return False
        elif side in ('sell', 'short') and current_price > entry_price * Decimal('1.005'):
            logger.warning(f"Short signal invalidated: price rose above entry")
            return False
        
        return True
