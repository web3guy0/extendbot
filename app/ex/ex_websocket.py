"""
Extended Exchange WebSocket - Real-time Data Streaming
Uses x10xchange Python SDK PerpetualStreamClient for all real-time data.

Subscription types (SDK native):
- Orderbooks: Market depth data
- Public Trades: Trade feed
- Candles: Candlestick data
- Funding Rates: Funding rate updates
- Account Updates: Orders, trades, positions, balance

SDK Reference: https://github.com/x10xchange/python_sdk
API Docs: https://api.docs.extended.exchange/#public-websocket-streams
"""
import asyncio
import time
from typing import Optional, Dict, Any, Callable, List, Set

from x10.perpetual.configuration import MAINNET_CONFIG, TESTNET_CONFIG, EndpointConfig
from x10.perpetual.stream_client import PerpetualStreamClient
from x10.perpetual.stream_client.perpetual_stream_connection import PerpetualStreamConnection
from x10.perpetual.accounts import AccountStreamDataModel
from x10.utils.http import WrappedStreamResponse

from app.utils.trading_logger import TradingLogger

logger = TradingLogger("ex_websocket")


class ExtendedWebSocket:
    """
    Extended Exchange WebSocket using SDK PerpetualStreamClient.
    
    Provides real-time streaming for:
    - Order book updates (public)
    - Trade feed (public)
    - Candle updates (public)
    - Account updates (private - orders, trades, positions)
    """
    
    def __init__(self, api_key: str, testnet: bool = False):
        """
        Initialize WebSocket client.
        
        Args:
            api_key: API key for private subscriptions
            testnet: Use testnet if True
        """
        self.api_key = api_key
        self.testnet = testnet
        
        # Select endpoint configuration
        self.config: EndpointConfig = TESTNET_CONFIG if testnet else MAINNET_CONFIG
        
        # SDK Stream client
        self.stream_client = PerpetualStreamClient(api_url=self.config.stream_url)
        
        # Active connections
        self._connections: Dict[str, PerpetualStreamConnection] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._active_subs: Set[str] = set()
        self._running = False
        
        # Reconnection state
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_base_delay = 1.0
        self._reconnect_max_delay = 60.0
        self._last_message_time: Optional[float] = None
        self._heartbeat_timeout = 60.0
        
        # Market data cache
        self._market_data: Dict[str, Dict[str, Any]] = {}
        self._orderbooks: Dict[str, Dict] = {}
        
        logger.info(f"ExtendedWebSocket initialized (testnet={testnet})")
    
    # ==================== CALLBACK MANAGEMENT ====================
    def add_callback(self, sub_type: str, callback: Callable):
        """Register callback for subscription type."""
        if sub_type not in self._callbacks:
            self._callbacks[sub_type] = []
        self._callbacks[sub_type].append(callback)
    
    def remove_callback(self, sub_type: str, callback: Callable):
        """Remove callback for subscription type."""
        if sub_type in self._callbacks and callback in self._callbacks[sub_type]:
            self._callbacks[sub_type].remove(callback)
    
    def _dispatch(self, sub_type: str, data: Any):
        """Dispatch data to registered callbacks."""
        self._last_message_time = time.time()
        self._reconnect_attempts = 0
        
        for cb in self._callbacks.get(sub_type, []):
            try:
                cb(data)
            except Exception as e:
                logger.error(f"Callback error for {sub_type}: {e}")
    
    # ==================== PUBLIC SUBSCRIPTIONS ====================
    async def subscribe_orderbook(
        self, 
        symbol: str, 
        callback: Optional[Callable] = None,
        depth: Optional[int] = None
    ):
        """
        Subscribe to order book updates for a symbol.
        
        Args:
            symbol: Market symbol (e.g., BTC-USD)
            callback: Function to call with orderbook updates
            depth: Depth of orderbook (1 for BBO only)
        """
        sub_key = f"orderbook:{symbol}"
        
        if callback:
            self.add_callback(sub_key, callback)
        
        try:
            connection = await self.stream_client.subscribe_to_orderbooks(
                market_name=symbol,
                depth=depth
            )
            self._connections[sub_key] = connection
            self._active_subs.add(sub_key)
            
            # Start message handler
            asyncio.create_task(self._handle_orderbook_messages(symbol, connection))
            
            logger.info(f"Subscribed to orderbook:{symbol}")
        except Exception as e:
            logger.error(f"Failed to subscribe to orderbook:{symbol}: {e}")
    
    async def _handle_orderbook_messages(
        self, 
        symbol: str, 
        connection: PerpetualStreamConnection
    ):
        """Handle incoming orderbook messages."""
        sub_key = f"orderbook:{symbol}"
        try:
            async for msg in connection:
                if msg.data:
                    # Update cache
                    self._orderbooks[symbol] = {
                        'bids': [(float(b.price), float(b.qty)) for b in (msg.data.bids or [])],
                        'asks': [(float(a.price), float(a.qty)) for a in (msg.data.asks or [])],
                    }
                    self._dispatch(sub_key, msg.data)
        except Exception as e:
            logger.error(f"Orderbook stream error for {symbol}: {e}")
            await self._handle_reconnect(sub_key)
    
    async def subscribe_trades(self, symbol: str, callback: Optional[Callable] = None):
        """
        Subscribe to public trade feed for a symbol.
        
        Args:
            symbol: Market symbol (e.g., BTC-USD)
            callback: Function to call with trade updates
        """
        sub_key = f"trades:{symbol}"
        
        if callback:
            self.add_callback(sub_key, callback)
        
        try:
            connection = await self.stream_client.subscribe_to_public_trades(market_name=symbol)
            self._connections[sub_key] = connection
            self._active_subs.add(sub_key)
            
            asyncio.create_task(self._handle_trades_messages(symbol, connection))
            
            logger.info(f"Subscribed to trades:{symbol}")
        except Exception as e:
            logger.error(f"Failed to subscribe to trades:{symbol}: {e}")
    
    async def _handle_trades_messages(
        self, 
        symbol: str, 
        connection: PerpetualStreamConnection
    ):
        """Handle incoming trade messages."""
        sub_key = f"trades:{symbol}"
        try:
            async for msg in connection:
                if msg.data:
                    self._dispatch(sub_key, msg.data)
        except Exception as e:
            logger.error(f"Trades stream error for {symbol}: {e}")
            await self._handle_reconnect(sub_key)
    
    async def subscribe_candles(
        self, 
        symbol: str, 
        interval: str = "1m",
        callback: Optional[Callable] = None
    ):
        """
        Subscribe to candlestick data.
        
        Args:
            symbol: Market symbol (e.g., BTC-USD)
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
            callback: Function to call with candle updates
        """
        sub_key = f"candle:{symbol}:{interval}"
        
        # Map interval to SDK format
        interval_map = {
            '1m': 'PT1M',
            '5m': 'PT5M', 
            '15m': 'PT15M',
            '30m': 'PT30M',
            '1h': 'PT1H',
            '4h': 'PT4H',
            '1d': 'PT24H',
        }
        sdk_interval = interval_map.get(interval, 'PT1M')
        
        if callback:
            self.add_callback(sub_key, callback)
        
        try:
            connection = await self.stream_client.subscribe_to_candles(
                market_name=symbol,
                candle_type='trades',
                interval=sdk_interval
            )
            self._connections[sub_key] = connection
            self._active_subs.add(sub_key)
            
            asyncio.create_task(self._handle_candle_messages(symbol, interval, connection))
            
            logger.info(f"Subscribed to candle:{symbol}:{interval}")
        except Exception as e:
            logger.error(f"Failed to subscribe to candles:{symbol}: {e}")
    
    async def _handle_candle_messages(
        self, 
        symbol: str, 
        interval: str,
        connection: PerpetualStreamConnection
    ):
        """Handle incoming candle messages."""
        sub_key = f"candle:{symbol}:{interval}"
        try:
            async for msg in connection:
                if msg.data:
                    # Convert to standard candle format
                    for candle_data in msg.data if isinstance(msg.data, list) else [msg.data]:
                        candle = {
                            'symbol': symbol,
                            'interval': interval,
                            'open': float(candle_data.open or 0),
                            'high': float(candle_data.high or 0),
                            'low': float(candle_data.low or 0),
                            'close': float(candle_data.close or 0),
                            'volume': float(candle_data.volume or 0) if hasattr(candle_data, 'volume') else 0,
                            'time': candle_data.timestamp,
                        }
                        self._dispatch(sub_key, (symbol, candle))
        except Exception as e:
            logger.error(f"Candle stream error for {symbol}: {e}")
            await self._handle_reconnect(sub_key)
    
    async def subscribe_funding_rates(
        self, 
        symbol: Optional[str] = None,
        callback: Optional[Callable] = None
    ):
        """
        Subscribe to funding rate updates.
        
        Args:
            symbol: Optional market symbol (None for all markets)
            callback: Function to call with funding rate updates
        """
        sub_key = f"funding:{symbol or 'all'}"
        
        if callback:
            self.add_callback(sub_key, callback)
        
        try:
            connection = await self.stream_client.subscribe_to_funding_rates(market_name=symbol)
            self._connections[sub_key] = connection
            self._active_subs.add(sub_key)
            
            asyncio.create_task(self._handle_funding_messages(symbol, connection))
            
            logger.info(f"Subscribed to funding:{symbol or 'all'}")
        except Exception as e:
            logger.error(f"Failed to subscribe to funding rates: {e}")
    
    async def _handle_funding_messages(
        self, 
        symbol: Optional[str],
        connection: PerpetualStreamConnection
    ):
        """Handle incoming funding rate messages."""
        sub_key = f"funding:{symbol or 'all'}"
        try:
            async for msg in connection:
                if msg.data:
                    self._dispatch(sub_key, msg.data)
        except Exception as e:
            logger.error(f"Funding stream error: {e}")
            await self._handle_reconnect(sub_key)
    
    # ==================== PRIVATE SUBSCRIPTIONS ====================
    async def subscribe_account_updates(self, callback: Optional[Callable] = None):
        """
        Subscribe to account updates (orders, trades, positions, balance).
        
        This is a private subscription requiring API key authentication.
        
        Args:
            callback: Function to call with account updates
        """
        sub_key = "account_updates"
        
        if callback:
            self.add_callback(sub_key, callback)
        
        try:
            connection = await self.stream_client.subscribe_to_account_updates(self.api_key)
            self._connections[sub_key] = connection
            self._active_subs.add(sub_key)
            
            asyncio.create_task(self._handle_account_messages(connection))
            
            logger.info("Subscribed to account updates")
        except Exception as e:
            logger.error(f"Failed to subscribe to account updates: {e}")
    
    async def _handle_account_messages(self, connection: PerpetualStreamConnection):
        """Handle incoming account update messages."""
        sub_key = "account_updates"
        try:
            async for msg in connection:
                if msg.data:
                    # Dispatch different event types
                    if hasattr(msg.data, 'orders') and msg.data.orders:
                        self._dispatch("order_updates", msg.data.orders)
                    if hasattr(msg.data, 'trades') and msg.data.trades:
                        self._dispatch("user_fills", msg.data.trades)
                    if hasattr(msg.data, 'positions') and msg.data.positions:
                        self._dispatch("position_updates", msg.data.positions)
                    if hasattr(msg.data, 'balance') and msg.data.balance:
                        self._dispatch("balance_updates", msg.data.balance)
                    
                    # Also dispatch raw message
                    self._dispatch(sub_key, msg.data)
        except Exception as e:
            logger.error(f"Account updates stream error: {e}")
            await self._handle_reconnect(sub_key)
    
    def subscribe_user_fills(self, callback: Optional[Callable] = None):
        """Register callback for user fill events (from account updates)."""
        if callback:
            self.add_callback("user_fills", callback)
    
    def subscribe_order_updates(self, callback: Optional[Callable] = None):
        """Register callback for order status updates (from account updates)."""
        if callback:
            self.add_callback("order_updates", callback)
    
    # ==================== CONNECTION MANAGEMENT ====================
    async def _handle_reconnect(self, sub_key: str):
        """Handle reconnection with exponential backoff."""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error(f"Max reconnect attempts reached for {sub_key}")
            return
        
        self._reconnect_attempts += 1
        delay = min(
            self._reconnect_base_delay * (2 ** self._reconnect_attempts),
            self._reconnect_max_delay
        )
        
        logger.warning(f"Reconnecting {sub_key} in {delay:.1f}s (attempt {self._reconnect_attempts})")
        await asyncio.sleep(delay)
        
        # Re-subscribe based on sub_key type
        if sub_key.startswith("orderbook:"):
            symbol = sub_key.split(":")[1]
            await self.subscribe_orderbook(symbol)
        elif sub_key.startswith("trades:"):
            symbol = sub_key.split(":")[1]
            await self.subscribe_trades(symbol)
        elif sub_key.startswith("candle:"):
            parts = sub_key.split(":")
            symbol, interval = parts[1], parts[2]
            await self.subscribe_candles(symbol, interval)
        elif sub_key == "account_updates":
            await self.subscribe_account_updates()
    
    def start(self):
        """Start WebSocket processing (for compatibility)."""
        self._running = True
        logger.info("ExtendedWebSocket started")
    
    async def stop(self):
        """Stop all WebSocket connections."""
        self._running = False
        
        for sub_key, connection in self._connections.items():
            try:
                await connection.close()
            except Exception as e:
                logger.warning(f"Error closing connection {sub_key}: {e}")
        
        self._connections.clear()
        self._active_subs.clear()
        logger.info("ExtendedWebSocket stopped")
    
    # ==================== CACHED DATA ACCESS ====================
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get cached orderbook for symbol."""
        return self._orderbooks.get(symbol)
    
    def get_best_bid(self, symbol: str) -> Optional[float]:
        """Get best bid price for symbol."""
        ob = self._orderbooks.get(symbol)
        if ob and ob.get('bids'):
            return ob['bids'][0][0]
        return None
    
    def get_best_ask(self, symbol: str) -> Optional[float]:
        """Get best ask price for symbol."""
        ob = self._orderbooks.get(symbol)
        if ob and ob.get('asks'):
            return ob['asks'][0][0]
        return None
    
    def get_cached_state(self) -> Optional[Dict]:
        """Get cached account state if available."""
        # This is for compatibility with hl_client
        return None
