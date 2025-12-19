#!/usr/bin/env python3
"""
Dynamic Symbol Manager for HyperLiquid
======================================

Manages trading symbols dynamically using HyperLiquid's official API methods.
Allows easy switching between trading pairs without hardcoding.

Features:
- Auto-discovery of available markets
- Symbol validation and metadata
- Market selection and switching
- Real-time market status updates
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from hyperliquid.info import Info
from hyperliquid.utils import constants

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    asset_id: int
    sz_decimals: int
    max_leverage: float
    only_isolated: bool
    is_active: bool
    day_ntl_vlm: float = 0.0
    prev_day_px: float = 0.0
    mark_px: float = 0.0

@dataclass
class MarketStats:
    """Market statistics"""
    symbol: str
    volume_24h: float
    price_change_24h: float
    price_change_pct: float
    high_24h: float
    low_24h: float
    trades_24h: int
    open_interest: float

class SymbolManager:
    """
    Manages trading symbols and market selection dynamically
    """
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize symbol manager
        
        Args:
            base_url: HyperLiquid API base URL (None for mainnet)
        """
        self.info = Info(base_url or constants.MAINNET_API_URL, skip_ws=True)
        self.markets: Dict[str, MarketData] = {}
        self.stats: Dict[str, MarketStats] = {}
        self.last_update = None
        self.update_interval = timedelta(minutes=5)
        
        # Current trading configuration
        self.active_symbol: Optional[str] = None
        self.symbol_blacklist: Set[str] = set()
        self.min_volume_threshold = 1000000.0  # $1M min daily volume
        
        logger.info("Symbol Manager initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize symbol manager and load market data
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Loading market data from HyperLiquid...")
            
            # Load universe data
            await self.update_markets()
            
            if not self.markets:
                logger.error("No markets loaded")
                return False
            
            logger.info(f"Loaded {len(self.markets)} markets")
            
            # Load market statistics
            await self.update_market_stats()
            
            return True
            
        except Exception as e:
            logger.error(f"Symbol manager initialization failed: {e}")
            return False
    
    async def update_markets(self) -> None:
        """
        Update market data from HyperLiquid API
        """
        try:
            meta = self.info.meta()
            universe = meta.get('universe', [])
            
            self.markets = {}
            
            for asset in universe:
                market_data = MarketData(
                    symbol=asset['name'],
                    asset_id=asset.get('assetId', 0),
                    sz_decimals=asset.get('szDecimals', 4),
                    max_leverage=float(asset.get('maxLeverage', 50)),
                    only_isolated=asset.get('onlyIsolated', False),
                    is_active=True  # Assume active if in universe
                )
                
                self.markets[asset['name']] = market_data
            
            self.last_update = datetime.now()
            logger.debug(f"Updated {len(self.markets)} markets")
            
        except Exception as e:
            logger.error(f"Failed to update markets: {e}")
    
    async def update_market_stats(self) -> None:
        """
        Update market statistics for better symbol selection
        """
        try:
            # Get all tickers for volume and price data
            all_mids = self.info.all_mids()
            
            for symbol in self.markets.keys():
                try:
                    # Get 24h statistics
                    # Note: HyperLiquid may not have all stats APIs, using available data
                    mid_price = all_mids.get(symbol, 0.0)
                    
                    # For now, create basic stats
                    stats = MarketStats(
                        symbol=symbol,
                        volume_24h=0.0,  # Would need candle data to calculate
                        price_change_24h=0.0,
                        price_change_pct=0.0,
                        high_24h=mid_price,
                        low_24h=mid_price,
                        trades_24h=0,
                        open_interest=0.0
                    )
                    
                    self.stats[symbol] = stats
                    
                except Exception as e:
                    logger.debug(f"Could not load stats for {symbol}: {e}")
            
            logger.debug(f"Updated stats for {len(self.stats)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to update market stats: {e}")
    
    def get_available_symbols(self, min_volume: Optional[float] = None) -> List[str]:
        """
        Get list of available trading symbols
        
        Args:
            min_volume: Minimum 24h volume filter
            
        Returns:
            List of available symbols
        """
        symbols = []
        volume_filter = min_volume or self.min_volume_threshold
        
        for symbol, market in self.markets.items():
            # Skip blacklisted symbols
            if symbol in self.symbol_blacklist:
                continue
            
            # Skip isolated-only markets if not desired
            if market.only_isolated:
                continue
            
            # Check volume if stats available
            stats = self.stats.get(symbol)
            if stats and stats.volume_24h < volume_filter:
                continue
            
            symbols.append(symbol)
        
        return sorted(symbols)
    
    def get_top_symbols_by_volume(self, count: int = 10) -> List[str]:
        """
        Get top symbols by trading volume
        
        Args:
            count: Number of symbols to return
            
        Returns:
            List of top symbols by volume
        """
        # Sort by volume (descending)
        sorted_symbols = sorted(
            self.stats.items(),
            key=lambda x: x[1].volume_24h,
            reverse=True
        )
        
        return [symbol for symbol, _ in sorted_symbols[:count]]
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if symbol is available for trading
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            True if symbol is valid and available
        """
        if symbol not in self.markets:
            return False
        
        if symbol in self.symbol_blacklist:
            return False
        
        market = self.markets[symbol]
        if not market.is_active:
            return False
        
        return True
    
    def get_market_info(self, symbol: str) -> Optional[MarketData]:
        """
        Get market information for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            MarketData if available, None otherwise
        """
        return self.markets.get(symbol)
    
    def get_market_stats(self, symbol: str) -> Optional[MarketStats]:
        """
        Get market statistics for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            MarketStats if available, None otherwise
        """
        return self.stats.get(symbol)
    
    def set_active_symbol(self, symbol: str) -> bool:
        """
        Set the active trading symbol
        
        Args:
            symbol: Symbol to set as active
            
        Returns:
            True if symbol set successfully
        """
        if not self.validate_symbol(symbol):
            logger.error(f"Cannot set invalid symbol as active: {symbol}")
            return False
        
        old_symbol = self.active_symbol
        self.active_symbol = symbol
        
        logger.info(f"Active symbol changed: {old_symbol} -> {symbol}")
        
        # Log market info
        market = self.markets[symbol]
        logger.info(f"{symbol} Info:")
        logger.info(f"   Asset ID: {market.asset_id}")
        logger.info(f"   Decimals: {market.sz_decimals}")
        logger.info(f"   Max Leverage: {market.max_leverage}x")
        logger.info(f"   Isolated Only: {market.only_isolated}")
        
        return True
    
    def get_active_symbol(self) -> Optional[str]:
        """
        Get the currently active trading symbol
        
        Returns:
            Active symbol or None if not set
        """
        return self.active_symbol
    
    def add_to_blacklist(self, symbol: str) -> None:
        """
        Add symbol to blacklist
        
        Args:
            symbol: Symbol to blacklist
        """
        self.symbol_blacklist.add(symbol)
        logger.info(f"Added {symbol} to blacklist")
    
    def remove_from_blacklist(self, symbol: str) -> None:
        """
        Remove symbol from blacklist
        
        Args:
            symbol: Symbol to remove from blacklist
        """
        self.symbol_blacklist.discard(symbol)
        logger.info(f"Removed {symbol} from blacklist")
    
    def suggest_symbols(self, criteria: Dict[str, Any]) -> List[str]:
        """
        Suggest trading symbols based on criteria
        
        Args:
            criteria: Selection criteria
                - min_volume: Minimum 24h volume
                - max_leverage: Minimum max leverage
                - exclude_isolated: Skip isolated-only markets
                - top_n: Return top N symbols
                
        Returns:
            List of suggested symbols
        """
        suggested = []
        
        min_vol = criteria.get('min_volume', self.min_volume_threshold)
        min_leverage = criteria.get('max_leverage', 10.0)
        exclude_isolated = criteria.get('exclude_isolated', True)
        top_n = criteria.get('top_n', 10)
        
        for symbol, market in self.markets.items():
            # Apply filters
            if symbol in self.symbol_blacklist:
                continue
            
            if exclude_isolated and market.only_isolated:
                continue
            
            if market.max_leverage < min_leverage:
                continue
            
            stats = self.stats.get(symbol)
            if stats and stats.volume_24h < min_vol:
                continue
            
            suggested.append(symbol)
        
        # Sort by volume if stats available
        if self.stats:
            suggested.sort(
                key=lambda s: self.stats.get(s, MarketStats(s, 0, 0, 0, 0, 0, 0, 0)).volume_24h,
                reverse=True
            )
        
        return suggested[:top_n]
    
    async def should_update(self) -> bool:
        """
        Check if markets should be updated
        
        Returns:
            True if update is needed
        """
        if not self.last_update:
            return True
        
        return datetime.now() - self.last_update > self.update_interval
    
    async def auto_update(self) -> None:
        """
        Auto-update markets if needed
        """
        if await self.should_update():
            await self.update_markets()
            await self.update_market_stats()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get symbol manager summary
        
        Returns:
            Summary dictionary
        """
        return {
            'total_markets': len(self.markets),
            'available_symbols': len(self.get_available_symbols()),
            'active_symbol': self.active_symbol,
            'blacklisted': list(self.symbol_blacklist),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'top_symbols': self.get_top_symbols_by_volume(5),
            'markets_with_stats': len(self.stats)
        }
