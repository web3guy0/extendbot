"""
Smart Money Concepts Analyzer
Detects institutional trading patterns:
- Fair Value Gaps (FVG)
- Order Blocks
- Liquidity Sweeps
- Imbalance Zones
- Break of Structure (BoS) - NEW
- Change of Character (ChoCh) - NEW
"""

import os
import logging
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ZoneType(Enum):
    """Types of smart money zones."""
    BULLISH_FVG = "bullish_fvg"
    BEARISH_FVG = "bearish_fvg"
    BULLISH_OB = "bullish_order_block"
    BEARISH_OB = "bearish_order_block"
    LIQUIDITY_LOW = "liquidity_low"
    LIQUIDITY_HIGH = "liquidity_high"
    IMBALANCE = "imbalance"


@dataclass
class BreakOfStructure:
    """Represents a Break of Structure (BoS) event."""
    bos_type: str  # 'bullish_bos' or 'bearish_bos'
    level: Decimal
    broken_at: datetime
    strength: float  # 0-1
    confirmed: bool = False


@dataclass
class SmartMoneyZone:
    """Represents a smart money zone on the chart."""
    zone_type: ZoneType
    high: Decimal
    low: Decimal
    created_time: datetime
    strength: float  # 0-1 strength score
    times_tested: int = 0
    invalidated: bool = False


class SmartMoneyAnalyzer:
    """
    Smart Money Concepts (SMC) Analyzer
    
    Identifies institutional footprints:
    1. Fair Value Gaps (FVG) - Price imbalances where institutions entered
    2. Order Blocks - Zones where big players accumulated/distributed
    3. Liquidity Sweeps - Stop hunts before real moves
    4. Imbalance Zones - Areas of aggressive buying/selling
    5. Break of Structure (BoS) - Structure confirmation
    6. Change of Character (ChoCh) - Trend reversal confirmation
    
    Trading Edge:
    - Enter at FVGs when price returns
    - Trade with order blocks, not against
    - Fade liquidity sweeps
    - Target liquidity pools
    - Only enter after BoS confirmation
    """
    
    def __init__(self):
        """Initialize SMC analyzer."""
        # Configuration from env
        self.fvg_min_gap_pct = Decimal(os.getenv('FVG_MIN_GAP_PCT', '0.3'))
        self.ob_lookback = int(os.getenv('ORDER_BLOCK_LOOKBACK', '50'))
        self.liquidity_lookback = int(os.getenv('LIQUIDITY_LOOKBACK', '20'))
        self.zone_expiry_bars = int(os.getenv('SMC_ZONE_EXPIRY_BARS', '100'))
        self.bos_lookback = int(os.getenv('BOS_LOOKBACK', '20'))
        
        # Active zones
        self.fvg_zones: List[SmartMoneyZone] = []
        self.order_blocks: List[SmartMoneyZone] = []
        self.liquidity_levels: List[SmartMoneyZone] = []
        
        # Sweep detection
        self.recent_sweeps: deque = deque(maxlen=10)
        
        # Break of Structure tracking
        self.recent_bos: deque = deque(maxlen=5)
        self.market_structure: str = 'unknown'  # 'bullish', 'bearish', 'unknown'
        
        logger.info("💰 Smart Money Analyzer initialized")
        logger.info(f"   FVG Min Gap: {self.fvg_min_gap_pct}%")
        logger.info(f"   Order Block Lookback: {self.ob_lookback}")
        logger.info(f"   Liquidity Lookback: {self.liquidity_lookback}")
        logger.info(f"   BoS Lookback: {self.bos_lookback}")
    
    def analyze(self, candles: List[Dict]) -> Dict[str, Any]:
        """
        Full SMC analysis on price data.
        
        Returns:
            Dict with zones, signals, and bias
        """
        if len(candles) < 50:
            return {'bias': None, 'zones': [], 'signals': [], 'bos': None}
        
        # Detect all patterns
        fvgs = self.detect_fair_value_gaps(candles)
        obs = self.detect_order_blocks(candles)
        liquidity = self.detect_liquidity_levels(candles)
        sweeps = self.detect_liquidity_sweeps(candles)
        bos = self.detect_break_of_structure(candles)
        
        # Update internal state
        self.fvg_zones = fvgs
        self.order_blocks = obs
        self.liquidity_levels = liquidity
        
        # Determine bias
        current_price = Decimal(str(candles[-1].get('close', candles[-1].get('c', 0))))
        bias = self._determine_bias(current_price, fvgs, obs, sweeps)
        
        # Generate signals
        signals = self._generate_signals(current_price, fvgs, obs, sweeps)
        
        return {
            'bias': bias,
            'fvg_zones': fvgs,
            'order_blocks': obs,
            'liquidity_levels': liquidity,
            'recent_sweeps': list(sweeps),
            'signals': signals,
            'bos': bos,
            'market_structure': self.market_structure,
            'structure_confirmed': bos is not None and bos.confirmed,
        }
    
    def detect_break_of_structure(self, candles: List[Dict]) -> Optional[BreakOfStructure]:
        """
        Detect Break of Structure (BoS).
        
        SMC CORE: Don't enter until structure confirms direction.
        
        - Bullish BoS: Price closes above previous swing high
        - Bearish BoS: Price closes below previous swing low
        
        Returns:
            BreakOfStructure if detected, None otherwise
        """
        if len(candles) < self.bos_lookback:
            return None
        
        lookback_candles = candles[-self.bos_lookback:]
        current_candle = candles[-1]
        current_close = Decimal(str(current_candle.get('close', current_candle.get('c', 0))))
        
        # Get swing highs and lows from lookback period (excluding last 3 candles)
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(lookback_candles) - 3):
            high = Decimal(str(lookback_candles[i].get('high', lookback_candles[i].get('h', 0))))
            low = Decimal(str(lookback_candles[i].get('low', lookback_candles[i].get('l', 0))))
            
            # Check if swing high (local maximum)
            is_swing_high = True
            for j in range(-2, 3):
                if j == 0:
                    continue
                neighbor_high = Decimal(str(lookback_candles[i+j].get('high', lookback_candles[i+j].get('h', 0))))
                if neighbor_high >= high:
                    is_swing_high = False
                    break
            if is_swing_high:
                swing_highs.append(high)
            
            # Check if swing low (local minimum)
            is_swing_low = True
            for j in range(-2, 3):
                if j == 0:
                    continue
                neighbor_low = Decimal(str(lookback_candles[i+j].get('low', lookback_candles[i+j].get('l', 0))))
                if neighbor_low <= low:
                    is_swing_low = False
                    break
            if is_swing_low:
                swing_lows.append(low)
        
        if not swing_highs or not swing_lows:
            return None
        
        # Get the most recent significant swing points
        recent_high = max(swing_highs[-5:]) if len(swing_highs) >= 1 else swing_highs[-1] if swing_highs else None
        recent_low = min(swing_lows[-5:]) if len(swing_lows) >= 1 else swing_lows[-1] if swing_lows else None
        
        if not recent_high or not recent_low:
            return None
        
        # Check for Bullish BoS: Close above swing high
        if current_close > recent_high:
            bos = BreakOfStructure(
                bos_type='bullish_bos',
                level=recent_high,
                broken_at=datetime.now(timezone.utc),
                strength=min(1.0, float((current_close - recent_high) / recent_high * 100)),
                confirmed=True,
            )
            self.recent_bos.append(bos)
            self.market_structure = 'bullish'
            logger.info(f"📈 BULLISH BoS: Closed above swing high ${recent_high}")
            return bos
        
        # Check for Bearish BoS: Close below swing low
        elif current_close < recent_low:
            bos = BreakOfStructure(
                bos_type='bearish_bos',
                level=recent_low,
                broken_at=datetime.now(timezone.utc),
                strength=min(1.0, float((recent_low - current_close) / recent_low * 100)),
                confirmed=True,
            )
            self.recent_bos.append(bos)
            self.market_structure = 'bearish'
            logger.info(f"📉 BEARISH BoS: Closed below swing low ${recent_low}")
            return bos
        
        return None
    
    def get_bos_signal(self, direction: str) -> Tuple[float, str]:
        """
        Get signal score modifier based on Break of Structure.
        
        Args:
            direction: 'long' or 'short'
            
        Returns:
            Tuple of (score_modifier, reason)
        """
        if not self.recent_bos:
            return 0.0, "No recent BoS"
        
        latest_bos = self.recent_bos[-1]
        
        # BoS alignment with trade direction
        if direction == 'long' and latest_bos.bos_type == 'bullish_bos':
            return 1.5 * latest_bos.strength, "Bullish BoS confirms long"
        elif direction == 'short' and latest_bos.bos_type == 'bearish_bos':
            return 1.5 * latest_bos.strength, "Bearish BoS confirms short"
        
        # BoS against trade direction = penalty
        elif direction == 'long' and latest_bos.bos_type == 'bearish_bos':
            return -1.0, "Bearish BoS contradicts long"
        elif direction == 'short' and latest_bos.bos_type == 'bullish_bos':
            return -1.0, "Bullish BoS contradicts short"
        
        return 0.0, "No BoS signal"
    
    def is_structure_confirmed(self, direction: str) -> bool:
        """
        Check if market structure confirms trade direction.
        
        Pro rule: Don't enter until structure confirms.
        """
        if direction == 'long':
            return self.market_structure == 'bullish'
        elif direction == 'short':
            return self.market_structure == 'bearish'
        return False
    
    def detect_fair_value_gaps(self, candles: List[Dict]) -> List[SmartMoneyZone]:
        """
        Detect Fair Value Gaps (price imbalances).
        
        FVG occurs when:
        - Bullish: Candle N-1 high < Candle N+1 low (gap up)
        - Bearish: Candle N-1 low > Candle N+1 high (gap down)
        """
        fvgs = []
        
        for i in range(1, len(candles) - 1):
            prev = candles[i - 1]
            curr = candles[i]
            next_c = candles[i + 1]
            
            prev_high = Decimal(str(prev.get('high', prev.get('h', 0))))
            prev_low = Decimal(str(prev.get('low', prev.get('l', 0))))
            next_high = Decimal(str(next_c.get('high', next_c.get('h', 0))))
            next_low = Decimal(str(next_c.get('low', next_c.get('l', 0))))
            curr_close = Decimal(str(curr.get('close', curr.get('c', 0))))
            
            # Bullish FVG: Gap between prev high and next low
            if next_low > prev_high:
                gap = next_low - prev_high
                gap_pct = (gap / curr_close) * 100
                
                if gap_pct >= self.fvg_min_gap_pct:
                    fvgs.append(SmartMoneyZone(
                        zone_type=ZoneType.BULLISH_FVG,
                        high=next_low,
                        low=prev_high,
                        created_time=datetime.now(timezone.utc),
                        strength=min(1.0, float(gap_pct) / 0.5),
                    ))
            
            # Bearish FVG: Gap between prev low and next high
            elif prev_low > next_high:
                gap = prev_low - next_high
                gap_pct = (gap / curr_close) * 100
                
                if gap_pct >= self.fvg_min_gap_pct:
                    fvgs.append(SmartMoneyZone(
                        zone_type=ZoneType.BEARISH_FVG,
                        high=prev_low,
                        low=next_high,
                        created_time=datetime.now(timezone.utc),
                        strength=min(1.0, float(gap_pct) / 0.5),
                    ))
        
        # Keep only recent unfilled FVGs (last 20)
        return fvgs[-20:] if len(fvgs) > 20 else fvgs
    
    def detect_order_blocks(self, candles: List[Dict]) -> List[SmartMoneyZone]:
        """
        Detect Order Blocks (institutional accumulation/distribution zones).
        
        Bullish OB: Last bearish candle before strong bullish move
        Bearish OB: Last bullish candle before strong bearish move
        """
        obs = []
        lookback = min(self.ob_lookback, len(candles) - 5)
        
        for i in range(lookback, len(candles) - 3):
            curr = candles[i]
            curr_open = Decimal(str(curr.get('open', curr.get('o', 0))))
            curr_close = Decimal(str(curr.get('close', curr.get('c', 0))))
            curr_high = Decimal(str(curr.get('high', curr.get('h', 0))))
            curr_low = Decimal(str(curr.get('low', curr.get('l', 0))))
            
            # Check next 3 candles for strong move
            next_closes = [
                Decimal(str(candles[i+j].get('close', candles[i+j].get('c', 0))))
                for j in range(1, 4) if i+j < len(candles)
            ]
            
            if not next_closes:
                continue
            
            move_pct = (next_closes[-1] - curr_close) / curr_close * 100
            
            # Bullish Order Block: Bearish candle followed by strong bullish move
            if curr_close < curr_open and move_pct > Decimal('1.0'):
                obs.append(SmartMoneyZone(
                    zone_type=ZoneType.BULLISH_OB,
                    high=curr_high,
                    low=curr_low,
                    created_time=datetime.now(timezone.utc),
                    strength=min(1.0, float(abs(move_pct)) / 3.0),
                ))
            
            # Bearish Order Block: Bullish candle followed by strong bearish move
            elif curr_close > curr_open and move_pct < Decimal('-1.0'):
                obs.append(SmartMoneyZone(
                    zone_type=ZoneType.BEARISH_OB,
                    high=curr_high,
                    low=curr_low,
                    created_time=datetime.now(timezone.utc),
                    strength=min(1.0, float(abs(move_pct)) / 3.0),
                ))
        
        return obs[-10:] if len(obs) > 10 else obs
    
    def detect_liquidity_levels(self, candles: List[Dict]) -> List[SmartMoneyZone]:
        """
        Detect liquidity pools (stop-loss clusters).
        
        Equal highs/lows = liquidity (stops sitting above/below)
        Swing highs/lows = liquidity targets
        """
        levels = []
        lookback = min(self.liquidity_lookback, len(candles))
        
        highs = [Decimal(str(c.get('high', c.get('h', 0)))) for c in candles[-lookback:]]
        lows = [Decimal(str(c.get('low', c.get('l', 0)))) for c in candles[-lookback:]]
        
        # Find swing highs (local maxima)
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                levels.append(SmartMoneyZone(
                    zone_type=ZoneType.LIQUIDITY_HIGH,
                    high=highs[i] * Decimal('1.001'),  # Slightly above
                    low=highs[i],
                    created_time=datetime.now(timezone.utc),
                    strength=0.7,
                ))
        
        # Find swing lows (local minima)
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                levels.append(SmartMoneyZone(
                    zone_type=ZoneType.LIQUIDITY_LOW,
                    high=lows[i],
                    low=lows[i] * Decimal('0.999'),  # Slightly below
                    created_time=datetime.now(timezone.utc),
                    strength=0.7,
                ))
        
        return levels
    
    def detect_liquidity_sweeps(self, candles: List[Dict]) -> deque:
        """
        Detect liquidity sweeps (stop hunts).
        
        Sweep = Price breaks level but closes back inside
        Strong signal for reversal in opposite direction
        """
        sweeps = deque(maxlen=5)
        
        if len(candles) < 10:
            return sweeps
        
        # Get recent swing points
        recent = candles[-20:]
        highs = [Decimal(str(c.get('high', c.get('h', 0)))) for c in recent[:-3]]
        lows = [Decimal(str(c.get('low', c.get('l', 0)))) for c in recent[:-3]]
        
        swing_high = max(highs) if highs else Decimal('0')
        swing_low = min(lows) if lows else Decimal('0')
        
        # Check last candle for sweep
        last = candles[-1]
        last_high = Decimal(str(last.get('high', last.get('h', 0))))
        last_low = Decimal(str(last.get('low', last.get('l', 0))))
        last_close = Decimal(str(last.get('close', last.get('c', 0))))
        last_open = Decimal(str(last.get('open', last.get('o', 0))))
        
        # Bullish sweep: Wick below swing low, close back inside
        if last_low < swing_low and last_close > swing_low:
            sweeps.append({
                'type': 'bullish_sweep',
                'level': float(swing_low),
                'wick_low': float(last_low),
                'close': float(last_close),
                'strength': min(1.0, float((swing_low - last_low) / swing_low) * 100),
            })
            logger.info(f"🎯 BULLISH SWEEP detected at {swing_low}")
        
        # Bearish sweep: Wick above swing high, close back inside
        if last_high > swing_high and last_close < swing_high:
            sweeps.append({
                'type': 'bearish_sweep',
                'level': float(swing_high),
                'wick_high': float(last_high),
                'close': float(last_close),
                'strength': min(1.0, float((last_high - swing_high) / swing_high) * 100),
            })
            logger.info(f"🎯 BEARISH SWEEP detected at {swing_high}")
        
        self.recent_sweeps = sweeps
        return sweeps
    
    def _determine_bias(
        self, 
        price: Decimal, 
        fvgs: List[SmartMoneyZone], 
        obs: List[SmartMoneyZone],
        sweeps: deque,
    ) -> Optional[str]:
        """Determine overall market bias from SMC analysis."""
        bullish_score = 0
        bearish_score = 0
        
        # FVG bias
        for fvg in fvgs[-5:]:
            if fvg.zone_type == ZoneType.BULLISH_FVG and price > fvg.low:
                bullish_score += fvg.strength
            elif fvg.zone_type == ZoneType.BEARISH_FVG and price < fvg.high:
                bearish_score += fvg.strength
        
        # Order block bias
        for ob in obs[-5:]:
            if ob.zone_type == ZoneType.BULLISH_OB and price > ob.low:
                bullish_score += ob.strength * 1.5
            elif ob.zone_type == ZoneType.BEARISH_OB and price < ob.high:
                bearish_score += ob.strength * 1.5
        
        # Sweep bias (strong signal)
        for sweep in sweeps:
            if sweep['type'] == 'bullish_sweep':
                bullish_score += sweep['strength'] * 2
            elif sweep['type'] == 'bearish_sweep':
                bearish_score += sweep['strength'] * 2
        
        if bullish_score > bearish_score * 1.2:
            return 'bullish'
        elif bearish_score > bullish_score * 1.2:
            return 'bearish'
        return 'neutral'
    
    def _generate_signals(
        self,
        price: Decimal,
        fvgs: List[SmartMoneyZone],
        obs: List[SmartMoneyZone],
        sweeps: deque,
    ) -> List[Dict]:
        """Generate trading signals from SMC patterns."""
        signals = []
        
        # FVG fill signals
        for fvg in fvgs[-10:]:
            if fvg.zone_type == ZoneType.BULLISH_FVG:
                if fvg.low <= price <= fvg.high:
                    signals.append({
                        'type': 'fvg_fill',
                        'direction': 'long',
                        'entry': float(fvg.low),
                        'strength': fvg.strength,
                        'reason': 'Bullish FVG fill',
                    })
            elif fvg.zone_type == ZoneType.BEARISH_FVG:
                if fvg.low <= price <= fvg.high:
                    signals.append({
                        'type': 'fvg_fill',
                        'direction': 'short',
                        'entry': float(fvg.high),
                        'strength': fvg.strength,
                        'reason': 'Bearish FVG fill',
                    })
        
        # Order block test signals
        for ob in obs[-5:]:
            if ob.zone_type == ZoneType.BULLISH_OB:
                if ob.low <= price <= ob.high:
                    signals.append({
                        'type': 'order_block',
                        'direction': 'long',
                        'entry': float(ob.low),
                        'strength': ob.strength,
                        'reason': 'Bullish Order Block test',
                    })
            elif ob.zone_type == ZoneType.BEARISH_OB:
                if ob.low <= price <= ob.high:
                    signals.append({
                        'type': 'order_block',
                        'direction': 'short',
                        'entry': float(ob.high),
                        'strength': ob.strength,
                        'reason': 'Bearish Order Block test',
                    })
        
        # Liquidity sweep signals (strongest)
        for sweep in sweeps:
            if sweep['type'] == 'bullish_sweep':
                signals.append({
                    'type': 'liquidity_sweep',
                    'direction': 'long',
                    'entry': sweep['close'],
                    'strength': sweep['strength'],
                    'reason': 'Bullish Liquidity Sweep - stops hunted',
                })
            elif sweep['type'] == 'bearish_sweep':
                signals.append({
                    'type': 'liquidity_sweep',
                    'direction': 'short',
                    'entry': sweep['close'],
                    'strength': sweep['strength'],
                    'reason': 'Bearish Liquidity Sweep - stops hunted',
                })
        
        return signals
    
    def is_price_at_zone(self, price: Decimal, tolerance_pct: Decimal = Decimal('0.2')) -> Optional[SmartMoneyZone]:
        """Check if price is at any significant SMC zone."""
        for zone in self.fvg_zones + self.order_blocks:
            if zone.invalidated:
                continue
            
            zone_mid = (zone.high + zone.low) / 2
            tolerance = zone_mid * tolerance_pct / 100
            
            if zone.low - tolerance <= price <= zone.high + tolerance:
                return zone
        
        return None
