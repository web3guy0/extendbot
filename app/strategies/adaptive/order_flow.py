"""
Order Flow Analysis Module
Whale detection, volume profile, trade flow analysis.
"""

import os
import logging
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


@dataclass
class LargeTrade:
    """Represents a detected large trade."""
    timestamp: datetime
    side: str  # 'buy' or 'sell'
    size: Decimal
    price: Decimal
    size_usd: Decimal
    is_whale: bool


class OrderFlowAnalyzer:
    """
    Order Flow Analysis - Trade Flow Intelligence
    
    Detects institutional activity through:
    1. Large trade detection (whale trades)
    2. Volume profile (POC - Point of Control)
    3. Delta analysis (buy/sell imbalance)
    4. CVD (Cumulative Volume Delta)
    
    Trading edge: Follow the whales, fade retail.
    """
    
    def __init__(self):
        """Initialize order flow analyzer."""
        # Configuration from env
        self.whale_threshold_usd = Decimal(os.getenv('WHALE_THRESHOLD_USD', '50000'))
        self.large_trade_threshold_usd = Decimal(os.getenv('LARGE_TRADE_THRESHOLD_USD', '10000'))
        self.volume_profile_bins = int(os.getenv('VOLUME_PROFILE_BINS', '20'))
        self.volume_profile_bin_pct = Decimal(os.getenv('VOLUME_PROFILE_BIN_PCT', '0.1'))  # 0.1% bins
        
        # State tracking
        self.recent_trades: deque = deque(maxlen=1000)
        self.large_trades: deque = deque(maxlen=100)
        self.whale_trades: deque = deque(maxlen=50)
        
        # Volume profile
        self.volume_profile: Dict[Decimal, Decimal] = {}
        self.poc_price: Optional[Decimal] = None  # Point of Control
        self.value_area_high: Optional[Decimal] = None
        self.value_area_low: Optional[Decimal] = None
        
        # Delta tracking
        self.cumulative_delta: Decimal = Decimal('0')
        self.session_buy_volume: Decimal = Decimal('0')
        self.session_sell_volume: Decimal = Decimal('0')
        
        logger.info("📈 Order Flow Analyzer initialized")
        logger.info(f"   Whale Threshold: ${self.whale_threshold_usd}")
        logger.info(f"   Large Trade Threshold: ${self.large_trade_threshold_usd}")
    
    def process_trade(
        self,
        price: Decimal,
        size: Decimal,
        side: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[LargeTrade]:
        """
        Process incoming trade data.
        
        Args:
            price: Trade price
            size: Trade size
            side: 'buy' or 'sell'
            timestamp: Trade timestamp
        
        Returns:
            LargeTrade if this is a significant trade, None otherwise
        """
        timestamp = timestamp or datetime.now(timezone.utc)
        size_usd = price * size
        
        # Update delta
        if side == 'buy':
            self.cumulative_delta += size_usd
            self.session_buy_volume += size_usd
        else:
            self.cumulative_delta -= size_usd
            self.session_sell_volume += size_usd
        
        # Update volume profile
        self._update_volume_profile(price, size)
        
        # Check if large trade
        is_whale = size_usd >= self.whale_threshold_usd
        is_large = size_usd >= self.large_trade_threshold_usd
        
        if is_large:
            trade = LargeTrade(
                timestamp=timestamp,
                side=side,
                size=size,
                price=price,
                size_usd=size_usd,
                is_whale=is_whale,
            )
            
            self.large_trades.append(trade)
            if is_whale:
                self.whale_trades.append(trade)
                logger.info(f"🐋 WHALE {'BUY' if side == 'buy' else 'SELL'}: "
                          f"{size} @ ${price} (${size_usd:,.0f})")
            
            return trade
        
        return None
    
    def process_trades_batch(self, trades: List[Dict]) -> List[LargeTrade]:
        """Process multiple trades from WebSocket feed."""
        large_trades = []
        
        for trade in trades:
            price = Decimal(str(trade.get('px', trade.get('price', 0))))
            size = Decimal(str(trade.get('sz', trade.get('size', 0))))
            side = trade.get('side', 'buy' if trade.get('isBuy', True) else 'sell')
            
            result = self.process_trade(price, size, side)
            if result:
                large_trades.append(result)
        
        return large_trades
    
    def analyze_from_candles(self, candles: List[Dict]) -> Dict[str, Any]:
        """
        Analyze order flow from candle data (when trade feed not available).
        
        Uses volume and price action to infer order flow.
        """
        if len(candles) < 20:
            return {'bias': None}
        
        # Reset volume profile
        self.volume_profile = {}
        
        # Build volume profile from candles
        for candle in candles:
            high = Decimal(str(candle.get('high', candle.get('h', 0))))
            low = Decimal(str(candle.get('low', candle.get('l', 0))))
            close = Decimal(str(candle.get('close', candle.get('c', 0))))
            open_p = Decimal(str(candle.get('open', candle.get('o', 0))))
            volume = Decimal(str(candle.get('volume', candle.get('v', 0))))
            
            # Distribute volume across price range
            if high != low:
                mid = (high + low) / 2
                self._update_volume_profile(mid, volume)
        
        # Calculate POC and value area
        self._calculate_poc_and_value_area()
        
        # Estimate delta from candle closes
        total_bullish_vol = Decimal('0')
        total_bearish_vol = Decimal('0')
        
        for candle in candles[-20:]:
            close = Decimal(str(candle.get('close', candle.get('c', 0))))
            open_p = Decimal(str(candle.get('open', candle.get('o', 0))))
            volume = Decimal(str(candle.get('volume', candle.get('v', 0))))
            
            if close > open_p:
                total_bullish_vol += volume
            else:
                total_bearish_vol += volume
        
        # Determine bias
        total_vol = total_bullish_vol + total_bearish_vol
        buy_ratio = float(total_bullish_vol / total_vol) if total_vol > 0 else 0.5
        
        if buy_ratio > 0.6:
            bias = 'bullish'
        elif buy_ratio < 0.4:
            bias = 'bearish'
        else:
            bias = 'neutral'
        
        # Get current price position relative to POC
        current_price = Decimal(str(candles[-1].get('close', candles[-1].get('c', 0))))
        poc_distance = None
        if self.poc_price:
            poc_distance = float((current_price - self.poc_price) / self.poc_price * 100)
        
        return {
            'bias': bias,
            'buy_volume_ratio': buy_ratio,
            'poc_price': float(self.poc_price) if self.poc_price else None,
            'value_area_high': float(self.value_area_high) if self.value_area_high else None,
            'value_area_low': float(self.value_area_low) if self.value_area_low else None,
            'poc_distance_pct': poc_distance,
            'cumulative_delta': float(self.cumulative_delta),
            'whale_count_1h': self._count_recent_whales(hours=1),
        }
    
    def get_whale_bias(self, lookback_minutes: int = 60) -> Tuple[Optional[str], int, int]:
        """
        Get whale activity bias.
        
        Returns:
            Tuple of (bias, buy_count, sell_count)
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        
        buy_count = 0
        sell_count = 0
        buy_volume = Decimal('0')
        sell_volume = Decimal('0')
        
        for trade in self.whale_trades:
            if trade.timestamp >= cutoff:
                if trade.side == 'buy':
                    buy_count += 1
                    buy_volume += trade.size_usd
                else:
                    sell_count += 1
                    sell_volume += trade.size_usd
        
        if buy_volume > sell_volume * Decimal('1.5'):
            return 'bullish', buy_count, sell_count
        elif sell_volume > buy_volume * Decimal('1.5'):
            return 'bearish', buy_count, sell_count
        return 'neutral', buy_count, sell_count
    
    def get_delta_signal(self) -> Dict[str, Any]:
        """
        Get trading signal from delta analysis.
        
        Divergence between price and delta = potential reversal
        """
        if self.session_buy_volume == 0 and self.session_sell_volume == 0:
            return {'signal': None, 'strength': 0}
        
        total = self.session_buy_volume + self.session_sell_volume
        buy_pct = self.session_buy_volume / total * 100
        sell_pct = self.session_sell_volume / total * 100
        
        # Strong buying pressure
        if buy_pct > 65:
            return {
                'signal': 'long',
                'strength': float((buy_pct - 50) / 50),
                'buy_pct': float(buy_pct),
                'reason': f'Strong buying pressure ({buy_pct:.1f}%)',
            }
        
        # Strong selling pressure
        if sell_pct > 65:
            return {
                'signal': 'short',
                'strength': float((sell_pct - 50) / 50),
                'sell_pct': float(sell_pct),
                'reason': f'Strong selling pressure ({sell_pct:.1f}%)',
            }
        
        return {'signal': None, 'strength': 0}
    
    def is_price_at_poc(self, price: Decimal, tolerance_pct: Decimal = Decimal('0.3')) -> bool:
        """Check if price is near Point of Control."""
        if not self.poc_price:
            return False
        
        distance = abs(price - self.poc_price) / self.poc_price * 100
        return distance <= tolerance_pct
    
    def is_price_at_value_area_edge(
        self,
        price: Decimal,
        tolerance_pct: Decimal = Decimal('0.3'),
    ) -> Optional[str]:
        """
        Check if price is at Value Area edge (support/resistance).
        
        Returns:
            'vah' if at Value Area High, 'val' if at Value Area Low, None otherwise
        """
        if not self.value_area_high or not self.value_area_low:
            return None
        
        vah_distance = abs(price - self.value_area_high) / self.value_area_high * 100
        val_distance = abs(price - self.value_area_low) / self.value_area_low * 100
        
        if vah_distance <= tolerance_pct:
            return 'vah'
        if val_distance <= tolerance_pct:
            return 'val'
        return None
    
    def reset_session(self):
        """Reset session-based tracking (call at session start)."""
        self.cumulative_delta = Decimal('0')
        self.session_buy_volume = Decimal('0')
        self.session_sell_volume = Decimal('0')
        logger.info("📊 Order flow session reset")
    
    def _update_volume_profile(self, price: Decimal, volume: Decimal):
        """Update volume profile with new data."""
        # Round price to bin (configurable bin size)
        bin_size = self.volume_profile_bin_pct
        binned_price = round(price / bin_size) * bin_size
        
        if binned_price in self.volume_profile:
            self.volume_profile[binned_price] += volume
        else:
            self.volume_profile[binned_price] = volume
    
    def _calculate_poc_and_value_area(self):
        """Calculate POC and Value Area from volume profile."""
        if not self.volume_profile:
            return
        
        # POC = price level with highest volume
        self.poc_price = max(self.volume_profile, key=self.volume_profile.get)
        
        # Value Area = 70% of volume centered on POC
        total_volume = sum(self.volume_profile.values())
        target_volume = total_volume * Decimal('0.7')
        
        # Sort by distance from POC
        sorted_prices = sorted(
            self.volume_profile.keys(),
            key=lambda p: abs(p - self.poc_price)
        )
        
        cumulative = Decimal('0')
        included_prices = []
        
        for price in sorted_prices:
            cumulative += self.volume_profile[price]
            included_prices.append(price)
            if cumulative >= target_volume:
                break
        
        if included_prices:
            self.value_area_high = max(included_prices)
            self.value_area_low = min(included_prices)
    
    def _count_recent_whales(self, hours: int = 1) -> int:
        """Count whale trades in last N hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        return sum(1 for t in self.whale_trades if t.timestamp >= cutoff)
