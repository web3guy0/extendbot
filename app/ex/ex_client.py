"""
Extended Exchange Client - SDK Wrapper
Direct SDK methods with minimal wrapper overhead.
Uses x10xchange Python SDK for all trading operations.

SDK Reference: https://github.com/x10xchange/python_sdk
API Docs: https://api.docs.extended.exchange/
"""
import os
import time
import asyncio
from functools import lru_cache
from typing import Optional, Dict, Any, List
from decimal import Decimal
from datetime import datetime, timedelta, timezone

from x10.perpetual.accounts import StarkPerpetualAccount
from x10.perpetual.configuration import MAINNET_CONFIG, TESTNET_CONFIG, EndpointConfig
from x10.perpetual.trading_client import PerpetualTradingClient
from x10.perpetual.orders import OrderSide, TimeInForce
from x10.perpetual.stream_client import PerpetualStreamClient
from x10.perpetual.markets import MarketModel

from app.utils.trading_logger import TradingLogger

logger = TradingLogger("ex_client")


class ExtendedClient:
    """
    Extended Exchange SDK wrapper - exposes SDK methods directly.
    
    Uses x10xchange Python SDK for:
    - REST API trading operations
    - Account management
    - Market data retrieval
    - Position management
    """
    
    def __init__(
        self, 
        api_key: str, 
        private_key: str, 
        public_key: str,
        vault_id: int,
        testnet: bool = False
    ):
        """
        Initialize Extended Exchange client.
        
        Args:
            api_key: API key from Extended API management page
            private_key: Stark private key
            public_key: Stark public key
            vault_id: Account vault ID
            testnet: Use testnet if True (Starknet Sepolia)
        """
        # Select endpoint configuration
        self.config: EndpointConfig = TESTNET_CONFIG if testnet else MAINNET_CONFIG
        self.testnet = testnet
        
        # Create Stark account for signing
        self.stark_account = StarkPerpetualAccount(
            vault=vault_id,
            private_key=private_key,
            public_key=public_key,
            api_key=api_key,
        )
        
        # Direct SDK instances
        self.trading_client: Optional[PerpetualTradingClient] = None
        self.stream_client: Optional[PerpetualStreamClient] = None
        
        # Market data cache
        self._markets_cache: Optional[Dict[str, MarketModel]] = None
        self._meta_cache: Optional[Dict] = None
        self._last_markets_fetch: Optional[datetime] = None
        self._markets_cache_ttl = 300  # 5 minutes
        
        # WebSocket reference (set by bot.py after initialization)
        self.websocket = None
        
        # Rate limit monitoring
        self._rate_limit_warning_threshold = 0.8  # Warn at 80% usage
        self._last_rate_limit_check: float = 0
        self._rate_limit_check_interval = 60.0
        
        logger.info(f"ExtendedClient initialized (testnet={testnet})")
    
    async def initialize(self) -> bool:
        """
        Initialize the trading client (must be called before trading).
        
        Returns:
            True if initialization successful
        """
        try:
            # Create trading client
            self.trading_client = PerpetualTradingClient(
                endpoint_config=self.config,
                stark_account=self.stark_account,
            )
            
            # Create stream client for WebSocket
            self.stream_client = PerpetualStreamClient(api_url=self.config.stream_url)
            
            # Pre-fetch markets to cache
            await self._refresh_markets_cache()
            
            logger.info("✅ ExtendedClient initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ExtendedClient: {e}")
            return False
    
    @property
    def _loop(self):
        """Get the running event loop lazily."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.get_event_loop()
    
    async def _refresh_markets_cache(self):
        """Refresh markets cache from API."""
        try:
            if not self.trading_client:
                return
            
            markets_response = await self.trading_client.markets_info.get_markets()
            if markets_response.data:
                self._markets_cache = {m.name: m for m in markets_response.data}
                self._last_markets_fetch = datetime.now(timezone.utc)
                logger.debug(f"Cached {len(self._markets_cache)} markets")
        except Exception as e:
            logger.warning(f"Failed to refresh markets cache: {e}")
    
    async def get_markets(self) -> Dict[str, MarketModel]:
        """Get cached markets, refresh if stale."""
        now = datetime.now(timezone.utc)
        
        if (not self._markets_cache or 
            not self._last_markets_fetch or
            (now - self._last_markets_fetch).total_seconds() > self._markets_cache_ttl):
            await self._refresh_markets_cache()
        
        return self._markets_cache or {}
    
    # ==================== MARKET DATA ====================
    def get_meta(self) -> Dict:
        """Get perpetuals metadata (cached)."""
        return self._meta_cache or {}
    
    async def get_market(self, symbol: str) -> Optional[MarketModel]:
        """Get market info for a symbol (e.g., BTC-USD)."""
        markets = await self.get_markets()
        return markets.get(symbol)
    
    def get_sz_decimals(self, symbol: str) -> int:
        """Get size decimals for proper rounding."""
        if self._markets_cache and symbol in self._markets_cache:
            market = self._markets_cache[symbol]
            return market.asset_precision
        return 6  # Default
    
    def get_price_decimals(self, symbol: str) -> int:
        """Get price decimals for proper rounding."""
        if self._markets_cache and symbol in self._markets_cache:
            market = self._markets_cache[symbol]
            return market.collateral_asset_precision
        return 6  # Default
    
    def round_price(self, symbol: str, price: float) -> float:
        """Round price to valid tick size for the asset."""
        decimals = self.get_price_decimals(symbol)
        return round(float(f'{price:.5g}'), decimals)
    
    def round_size(self, symbol: str, size: float) -> float:
        """Round size to valid step size for the asset."""
        decimals = self.get_sz_decimals(symbol)
        return round(size, decimals)
    
    # ==================== ACCOUNT DATA ====================
    async def get_balance(self) -> float:
        """Get available balance for trading."""
        if not self.trading_client:
            return 0.0
        try:
            response = await self.trading_client.account.get_balance()
            if response.data:
                return float(response.data.available_for_trade or 0)
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
        return 0.0
    
    async def get_equity(self) -> float:
        """Get account equity."""
        if not self.trading_client:
            return 0.0
        try:
            response = await self.trading_client.account.get_balance()
            if response.data:
                return float(response.data.equity or 0)
        except Exception as e:
            logger.error(f"Failed to get equity: {e}")
        return 0.0
    
    async def get_account_state(self) -> Dict[str, Any]:
        """
        Get comprehensive account state.
        Returns dict with account_value, margin_used, positions, etc.
        """
        if not self.trading_client:
            return {}
        
        try:
            # Get balance
            balance_response = await self.trading_client.account.get_balance()
            balance = balance_response.data
            
            # Get positions
            positions_response = await self.trading_client.account.get_positions()
            
            positions = []
            if positions_response.data:
                for pos in positions_response.data:
                    positions.append({
                        'symbol': pos.market,
                        'size': float(pos.size),
                        'side': pos.side.lower() if pos.side else 'long',
                        'entry_price': float(pos.open_price or 0),
                        'mark_price': float(pos.mark_price or 0),
                        'unrealized_pnl': float(pos.unrealised_pnl or 0),
                        'position_value': float(pos.value or 0),
                        'leverage': float(pos.leverage or 1),
                        'liquidation_price': float(pos.liquidation_price) if pos.liquidation_price else None,
                    })
            
            return {
                'account_value': float(balance.equity) if balance else 0,
                'margin_used': float(balance.initial_margin) if balance else 0,
                'available_margin': float(balance.available_for_trade) if balance else 0,
                'positions': positions,
                'unrealized_pnl': float(balance.unrealised_pnl) if balance else 0,
            }
        except Exception as e:
            logger.error(f"Failed to get account state: {e}")
            return {}
    
    # ==================== POSITION DATA ====================
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for symbol."""
        if not self.trading_client:
            return None
        try:
            response = await self.trading_client.account.get_positions(market_names=[symbol])
            if response.data and len(response.data) > 0:
                pos = response.data[0]
                return {
                    'symbol': pos.market,
                    'size': float(pos.size),
                    'side': pos.side.lower() if pos.side else 'long',
                    'entry_price': float(pos.open_price or 0),
                    'mark_price': float(pos.mark_price or 0),
                    'unrealized_pnl': float(pos.unrealised_pnl or 0),
                    'leverage': float(pos.leverage or 1),
                }
        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}")
        return None
    
    async def get_all_positions(self) -> List[Dict]:
        """Get all open positions."""
        state = await self.get_account_state()
        return state.get('positions', [])
    
    async def get_open_positions(self) -> List[Dict]:
        """Get all open positions (alias for get_all_positions)."""
        return await self.get_all_positions()
    
    # ==================== MARKET PRICE DATA ====================
    async def get_mid_price(self, symbol: str) -> float:
        """Get current mid price (average of bid/ask)."""
        if not self.trading_client:
            return 0.0
        try:
            markets = await self.trading_client.markets_info.get_markets(market_names=[symbol])
            if markets.data and len(markets.data) > 0:
                market = markets.data[0]
                stats = market.market_stats
                if stats:
                    bid = float(stats.bid_price or 0)
                    ask = float(stats.ask_price or 0)
                    return (bid + ask) / 2 if bid and ask else float(stats.last_price or 0)
        except Exception as e:
            logger.error(f"Failed to get mid price for {symbol}: {e}")
        return 0.0
    
    async def get_market_price(self, symbol: str) -> float:
        """Get current market price (alias for get_mid_price)."""
        return await self.get_mid_price(symbol)
    
    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate for a symbol."""
        try:
            markets = await self.trading_client.markets_info.get_markets(market_names=[symbol])
            if markets.data and len(markets.data) > 0:
                stats = markets.data[0].market_stats
                if stats and stats.funding_rate:
                    return float(stats.funding_rate)
        except Exception as e:
            logger.warning(f"Failed to get funding rate for {symbol}: {e}")
        return None
    
    async def get_all_funding_rates(self) -> Dict[str, float]:
        """Get funding rates for all assets."""
        result = {}
        try:
            markets = await self.trading_client.markets_info.get_markets()
            if markets.data:
                for market in markets.data:
                    if market.market_stats and market.market_stats.funding_rate:
                        result[market.name] = float(market.market_stats.funding_rate)
        except Exception as e:
            logger.warning(f"Failed to get funding rates: {e}")
        return result
    
    # ==================== HISTORICAL DATA ====================
    async def get_candles(self, symbol: str, interval: str = "1m", limit: int = 100) -> List[Dict]:
        """
        Get historical candles from API.
        
        Args:
            symbol: Trading pair symbol (e.g., BTC-USD)
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch
        
        Returns:
            List of candle dicts with open, high, low, close, volume, time
        """
        if not self.trading_client:
            return []
        
        try:
            # Map interval to Extended candle type
            # Extended uses: PT1M, PT5M, PT15M, PT30M, PT1H, PT4H, PT24H
            interval_map = {
                '1m': 'PT1M',
                '5m': 'PT5M',
                '15m': 'PT15M',
                '30m': 'PT30M',
                '1h': 'PT1H',
                '4h': 'PT4H',
                '1d': 'PT24H',
            }
            candle_interval = interval_map.get(interval, 'PT1M')
            
            # Calculate time range
            now_ms = int(time.time() * 1000)
            interval_ms_map = {
                '1m': 60 * 1000,
                '5m': 5 * 60 * 1000,
                '15m': 15 * 60 * 1000,
                '30m': 30 * 60 * 1000,
                '1h': 60 * 60 * 1000,
                '4h': 4 * 60 * 60 * 1000,
                '1d': 24 * 60 * 60 * 1000,
            }
            interval_ms = interval_ms_map.get(interval, 60 * 1000)
            start_time = now_ms - (limit * interval_ms)
            
            # Fetch candles using REST API
            response = await self.trading_client.info.get_candles(
                market=symbol,
                candle_type='trades',
                interval=candle_interval,
                start_time=start_time,
                end_time=now_ms,
                limit=limit,
            )
            
            candles = []
            if response.data:
                for c in response.data:
                    candles.append({
                        'open': float(c.open or 0),
                        'high': float(c.high or 0),
                        'low': float(c.low or 0),
                        'close': float(c.close or 0),
                        'volume': float(c.volume or 0) if hasattr(c, 'volume') else 0,
                        'time': c.timestamp,
                    })
            return candles
            
        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol}: {e}")
            return []
    
    # ==================== ORDERS ====================
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders, optionally filtered by symbol."""
        if not self.trading_client:
            return []
        try:
            market_names = [symbol] if symbol else None
            response = await self.trading_client.account.get_open_orders(market_names=market_names)
            orders = []
            if response.data:
                for o in response.data:
                    orders.append({
                        'id': o.id,
                        'external_id': o.external_id,
                        'symbol': o.market,
                        'side': o.side,
                        'type': o.type,
                        'status': o.status,
                        'price': float(o.price or 0),
                        'qty': float(o.qty or 0),
                        'filled_qty': float(o.filled_qty or 0),
                        'created_time': o.created_time,
                    })
            return orders
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
        return []
    
    # ==================== LEVERAGE ====================
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        if not self.trading_client:
            return False
        try:
            await self.trading_client.account.update_leverage(
                market_name=symbol,
                leverage=Decimal(str(leverage))
            )
            logger.info(f"Set leverage to {leverage}x for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")
            return False
    
    async def get_leverage(self, symbol: str) -> int:
        """Get current leverage for a symbol."""
        if not self.trading_client:
            return 1
        try:
            response = await self.trading_client.account.get_leverage(market_names=[symbol])
            if response.data and len(response.data) > 0:
                return int(response.data[0].leverage)
        except Exception as e:
            logger.error(f"Failed to get leverage for {symbol}: {e}")
        return 1
    
    async def close(self):
        """Clean up resources."""
        if self.trading_client:
            await self.trading_client.close()
        logger.info("ExtendedClient closed")


def create_client(
    api_key: str,
    private_key: str,
    public_key: str,
    vault_id: int,
    testnet: bool = False
) -> ExtendedClient:
    """Create ExtendedClient instance."""
    return ExtendedClient(api_key, private_key, public_key, vault_id, testnet)
