#!/usr/bin/env python3
"""
Multi-Asset Trading Manager
===========================

Enables simultaneous trading across multiple assets (BTC, ETH, SOL, etc.)
with proper position management and risk allocation per asset.

Features:
- Independent candle caches per symbol
- Strategy instances per symbol
- Per-asset position limits
- Portfolio-level risk management
- Rotation-based signal scanning
"""

import asyncio
import logging
from decimal import Decimal
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


@dataclass
class AssetState:
    """State tracking for a single asset"""
    symbol: str
    is_enabled: bool = True
    has_position: bool = False
    position_side: Optional[str] = None  # 'long' or 'short'
    position_size: Decimal = Decimal('0')
    entry_price: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    
    # Candle cache (LTF - 1m, 5m)
    candles: List[Dict[str, Any]] = field(default_factory=list)
    last_candle_fetch: Optional[datetime] = None
    candle_update_pending: bool = False
    
    # HTF candle cache (15m, 1h, 4h) for multi-timeframe analysis
    htf_candles: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  # {interval: candles}
    last_htf_fetch: Optional[datetime] = None
    
    # Signal cooldown (prevent over-trading)
    last_signal_time: Optional[datetime] = None
    signal_cooldown_seconds: int = 60  # Min time between signals
    
    # Stats
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    pnl_today: Decimal = Decimal('0')


class MultiAssetManager:
    """
    Manages multi-asset trading with independent tracking per symbol
    
    Default assets: BTC, ETH, SOL (top 3 by volume)
    Can be extended to any HyperLiquid perpetual
    """
    
    # Default enabled assets (most liquid, most signals)
    DEFAULT_ASSETS = ['BTC', 'ETH', 'SOL']
    
    # Maximum simultaneous positions
    DEFAULT_MAX_POSITIONS = 3
    
    def __init__(
        self,
        enabled_assets: Optional[List[str]] = None,
        max_positions: int = DEFAULT_MAX_POSITIONS,
        position_size_pct_per_asset: float = 20.0  # % of account per trade
    ):
        """
        Initialize multi-asset manager
        
        Args:
            enabled_assets: List of symbols to trade (default: BTC, ETH, SOL)
            max_positions: Max simultaneous open positions
            position_size_pct_per_asset: Account % to risk per trade
        """
        self.enabled_assets = enabled_assets or self.DEFAULT_ASSETS
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct_per_asset
        
        # Asset states
        self.assets: Dict[str, AssetState] = {}
        for symbol in self.enabled_assets:
            self.assets[symbol] = AssetState(symbol=symbol)
        
        # Rotation index for fair signal scanning
        self._scan_index = 0
        
        # Portfolio-level tracking
        self.total_positions = 0
        self.portfolio_exposure_pct = Decimal('0')
        
        logger.info(f"🌐 Multi-Asset Manager initialized")
        logger.info(f"   Assets: {', '.join(self.enabled_assets)}")
        logger.info(f"   Max positions: {max_positions}")
        logger.info(f"   Size per trade: {position_size_pct_per_asset}%")
    
    def get_enabled_assets(self) -> List[str]:
        """Get list of enabled trading assets"""
        return [s for s, state in self.assets.items() if state.is_enabled]
    
    def get_assets_with_positions(self) -> List[str]:
        """Get list of assets that have open positions"""
        return [s for s, state in self.assets.items() if state.has_position]
    
    def get_assets_without_positions(self) -> List[str]:
        """Get list of assets available for new positions"""
        return [s for s, state in self.assets.items() 
                if state.is_enabled and not state.has_position]
    
    def can_open_new_position(self) -> bool:
        """Check if we can open a new position (portfolio level)"""
        return self.total_positions < self.max_positions
    
    def can_trade_asset(self, symbol: str) -> tuple[bool, str]:
        """
        Check if an asset can be traded right now
        
        Returns:
            (can_trade, reason)
        """
        if symbol not in self.assets:
            return False, f"Asset {symbol} not in managed list"
        
        state = self.assets[symbol]
        
        if not state.is_enabled:
            return False, f"{symbol} is disabled"
        
        if state.has_position:
            return False, f"{symbol} already has open position"
        
        if not self.can_open_new_position():
            return False, f"Max positions ({self.max_positions}) reached"
        
        # Check signal cooldown
        if state.last_signal_time:
            elapsed = (datetime.now(timezone.utc) - state.last_signal_time).total_seconds()
            if elapsed < state.signal_cooldown_seconds:
                remaining = state.signal_cooldown_seconds - elapsed
                return False, f"{symbol} on cooldown ({remaining:.0f}s remaining)"
        
        return True, "OK"
    
    def get_next_asset_to_scan(self) -> Optional[str]:
        """
        Get next asset to scan for signals (round-robin)
        
        Returns:
            Symbol to scan, or None if none available
        """
        available = self.get_assets_without_positions()
        
        if not available:
            return None
        
        if not self.can_open_new_position():
            return None
        
        # Round-robin selection
        self._scan_index = self._scan_index % len(available)
        symbol = available[self._scan_index]
        self._scan_index += 1
        
        return symbol
    
    def update_position_state(
        self,
        symbol: str,
        has_position: bool,
        side: Optional[str] = None,
        size: Optional[Decimal] = None,
        entry_price: Optional[Decimal] = None,
        unrealized_pnl: Optional[Decimal] = None
    ):
        """
        Update position state for an asset
        
        Args:
            symbol: Asset symbol
            has_position: Whether position is open
            side: 'long' or 'short'
            size: Position size
            entry_price: Entry price
            unrealized_pnl: Current unrealized P&L
        """
        if symbol not in self.assets:
            return
        
        state = self.assets[symbol]
        was_open = state.has_position
        
        state.has_position = has_position
        state.position_side = side if has_position else None
        state.position_size = size or Decimal('0')
        state.entry_price = entry_price or Decimal('0')
        state.unrealized_pnl = unrealized_pnl or Decimal('0')
        
        # Track position count changes
        if has_position and not was_open:
            self.total_positions += 1
            logger.info(f"📈 Position opened: {symbol} {side} | Total: {self.total_positions}/{self.max_positions}")
        elif not has_position and was_open:
            self.total_positions = max(0, self.total_positions - 1)
            logger.info(f"📉 Position closed: {symbol} | Total: {self.total_positions}/{self.max_positions}")
    
    def update_from_account_state(self, account_state: Dict[str, Any]):
        """
        Update all asset states from exchange account state
        
        Args:
            account_state: Account state from exchange
        """
        positions = account_state.get('positions', [])
        
        # Build set of symbols with open positions
        open_positions: Dict[str, Dict] = {}
        for pos in positions:
            size = float(pos.get('size', 0))
            if size != 0:
                symbol = pos['symbol']
                open_positions[symbol] = pos
        
        # Update each managed asset
        for symbol, state in self.assets.items():
            if symbol in open_positions:
                pos = open_positions[symbol]
                size = Decimal(str(pos.get('size', 0)))
                self.update_position_state(
                    symbol=symbol,
                    has_position=True,
                    side='long' if size > 0 else 'short',
                    size=abs(size),
                    entry_price=Decimal(str(pos.get('entry_price', 0))),
                    unrealized_pnl=Decimal(str(pos.get('unrealized_pnl', 0)))
                )
            else:
                self.update_position_state(symbol=symbol, has_position=False)
    
    def record_trade(self, symbol: str, pnl: Decimal, is_win: bool):
        """Record completed trade for an asset"""
        if symbol not in self.assets:
            return
        
        state = self.assets[symbol]
        state.trades_today += 1
        state.pnl_today += pnl
        
        if is_win:
            state.wins_today += 1
        else:
            state.losses_today += 1
    
    def record_signal(self, symbol: str):
        """Record that a signal was generated (for cooldown)"""
        if symbol not in self.assets:
            return
        
        self.assets[symbol].last_signal_time = datetime.now(timezone.utc)
    
    def update_candles(self, symbol: str, candles: List[Dict[str, Any]]):
        """Update candle cache for an asset"""
        if symbol not in self.assets:
            return
        
        state = self.assets[symbol]
        state.candles = candles
        state.last_candle_fetch = datetime.now(timezone.utc)
        state.candle_update_pending = False
    
    def get_candles(self, symbol: str) -> List[Dict[str, Any]]:
        """Get cached candles for an asset"""
        if symbol not in self.assets:
            return []
        return self.assets[symbol].candles
    
    def needs_candle_refresh(self, symbol: str, max_age_seconds: int = 120) -> bool:
        """Check if candles need refresh for an asset"""
        if symbol not in self.assets:
            return True
        
        state = self.assets[symbol]
        
        if not state.candles or not state.last_candle_fetch:
            return True
        
        if state.candle_update_pending:
            return True
        
        age = (datetime.now(timezone.utc) - state.last_candle_fetch).total_seconds()
        return age > max_age_seconds
    
    def mark_candle_update_pending(self, symbol: str):
        """Mark that candles need update (WebSocket new candle)"""
        if symbol in self.assets:
            self.assets[symbol].candle_update_pending = True
    
    # ==================== HTF CANDLE MANAGEMENT ====================
    def update_htf_candles(self, symbol: str, htf_candles: Dict[str, List[Dict[str, Any]]]):
        """Update HTF candles for an asset"""
        if symbol not in self.assets:
            return
        
        state = self.assets[symbol]
        state.htf_candles = htf_candles
        state.last_htf_fetch = datetime.now(timezone.utc)
    
    def get_htf_candles(self, symbol: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get cached HTF candles for an asset"""
        if symbol not in self.assets:
            return {}
        return self.assets[symbol].htf_candles
    
    def needs_htf_refresh(self, symbol: str, max_age_seconds: int = 900) -> bool:
        """Check if HTF candles need refresh for an asset (default 15 min)"""
        if symbol not in self.assets:
            return True
        
        state = self.assets[symbol]
        
        if not state.htf_candles or not state.last_htf_fetch:
            return True
        
        age = (datetime.now(timezone.utc) - state.last_htf_fetch).total_seconds()
        return age > max_age_seconds
    
    def enable_asset(self, symbol: str) -> bool:
        """Enable trading for an asset"""
        if symbol not in self.assets:
            # Add new asset
            self.assets[symbol] = AssetState(symbol=symbol, is_enabled=True)
            self.enabled_assets.append(symbol)
            logger.info(f"✅ Added and enabled {symbol}")
            return True
        
        self.assets[symbol].is_enabled = True
        logger.info(f"✅ Enabled {symbol}")
        return True
    
    def disable_asset(self, symbol: str) -> bool:
        """Disable trading for an asset"""
        if symbol not in self.assets:
            return False
        
        self.assets[symbol].is_enabled = False
        logger.info(f"🚫 Disabled {symbol}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get portfolio-wide stats"""
        total_pnl = sum(s.pnl_today for s in self.assets.values())
        total_trades = sum(s.trades_today for s in self.assets.values())
        total_wins = sum(s.wins_today for s in self.assets.values())
        
        return {
            'enabled_assets': self.get_enabled_assets(),
            'positions_open': self.total_positions,
            'max_positions': self.max_positions,
            'total_trades_today': total_trades,
            'total_wins_today': total_wins,
            'win_rate': (total_wins / total_trades * 100) if total_trades > 0 else 0,
            'total_pnl_today': float(total_pnl),
            'per_asset': {
                symbol: {
                    'enabled': state.is_enabled,
                    'has_position': state.has_position,
                    'position_side': state.position_side,
                    'trades': state.trades_today,
                    'pnl': float(state.pnl_today)
                }
                for symbol, state in self.assets.items()
            }
        }
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at start of trading day)"""
        for state in self.assets.values():
            state.trades_today = 0
            state.wins_today = 0
            state.losses_today = 0
            state.pnl_today = Decimal('0')
        
        logger.info("📊 Daily stats reset for all assets")


# Singleton instance
_multi_asset_manager: Optional[MultiAssetManager] = None


def get_multi_asset_manager(
    enabled_assets: Optional[List[str]] = None,
    max_positions: int = 3
) -> MultiAssetManager:
    """
    Get or create the multi-asset manager singleton
    
    Args:
        enabled_assets: List of symbols to trade
        max_positions: Max simultaneous positions
    
    Returns:
        MultiAssetManager instance
    """
    global _multi_asset_manager
    
    if _multi_asset_manager is None:
        _multi_asset_manager = MultiAssetManager(
            enabled_assets=enabled_assets,
            max_positions=max_positions
        )
    
    return _multi_asset_manager
