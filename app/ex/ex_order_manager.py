"""
Extended Exchange Order Manager - Order Execution and Management
Handles order placement, cancellation, TP/SL with proper Stark signatures.

Uses x10xchange Python SDK for:
- Order placement (limit, market, conditional)
- Take Profit / Stop Loss orders
- Order cancellation
- Position management

SDK Reference: https://github.com/x10xchange/python_sdk
API Docs: https://api.docs.extended.exchange/#create-or-edit-order
"""
import asyncio
import uuid
import time
from typing import Optional, Dict, Any, Callable, List, Literal
from decimal import Decimal
from datetime import datetime, timedelta, timezone

from x10.perpetual.orders import (
    OrderSide,
    TimeInForce,
    OrderTpslType,
    OrderPriceType,
    OrderTriggerPriceType,
)
from x10.perpetual.order_object import OrderTpslTriggerParam, create_order_object

from app.ex.ex_client import ExtendedClient
from app.utils.trading_logger import TradingLogger

logger = TradingLogger("ex_order_manager")


class ExtendedOrderManager:
    """
    Extended Exchange order manager using SDK for all operations.
    
    Features:
    - Market and limit order placement
    - Take Profit / Stop Loss with proper signatures
    - Order cancellation (single and mass)
    - Position tracking
    """
    
    def __init__(self, client: ExtendedClient, on_fill: Optional[Callable] = None):
        """
        Initialize order manager.
        
        Args:
            client: ExtendedClient instance
            on_fill: Optional callback for fill events
        """
        self.client = client
        self.on_fill = on_fill
        
        # Position tracking for trailing stops
        self.position_orders: Dict[str, Dict[str, Any]] = {}
        
        # Markets cache
        self._markets_cache: Dict[str, Any] = {}
    
    @property
    def _loop(self):
        """Get the running event loop lazily."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.get_event_loop()
    
    def _gen_external_id(self) -> str:
        """Generate unique external order ID."""
        return str(uuid.uuid4())
    
    async def _get_market(self, symbol: str) -> Optional[Any]:
        """Get market info with caching."""
        if symbol not in self._markets_cache:
            market = await self.client.get_market(symbol)
            if market:
                self._markets_cache[symbol] = market
        return self._markets_cache.get(symbol)
    
    # ==================== LEVERAGE ====================
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        return await self.client.set_leverage(symbol, leverage)
    
    async def get_leverage(self, symbol: str) -> int:
        """Get current leverage for a symbol."""
        return await self.client.get_leverage(symbol)
    
    # ==================== MARKET ORDERS ====================
    async def market_buy(
        self,
        symbol: str,
        size: float,
        reduce_only: bool = False,
        slippage: float = 0.01,
    ) -> Dict:
        """
        Place a market buy order.
        
        Args:
            symbol: Trading pair (e.g., BTC-USD)
            size: Order size in base asset
            reduce_only: If True, only reduce existing position
            slippage: Max slippage percentage (default 1%)
        
        Returns:
            Order result dict
        """
        return await self._place_market_order(
            symbol=symbol,
            size=size,
            is_buy=True,
            reduce_only=reduce_only,
            slippage=slippage,
        )
    
    async def market_sell(
        self,
        symbol: str,
        size: float,
        reduce_only: bool = False,
        slippage: float = 0.01,
    ) -> Dict:
        """
        Place a market sell order.
        
        Args:
            symbol: Trading pair (e.g., BTC-USD)
            size: Order size in base asset
            reduce_only: If True, only reduce existing position
            slippage: Max slippage percentage (default 1%)
        
        Returns:
            Order result dict
        """
        return await self._place_market_order(
            symbol=symbol,
            size=size,
            is_buy=False,
            reduce_only=reduce_only,
            slippage=slippage,
        )
    
    async def _place_market_order(
        self,
        symbol: str,
        size: float,
        is_buy: bool,
        reduce_only: bool = False,
        slippage: float = 0.01,
    ) -> Dict:
        """Internal market order placement."""
        if not self.client.trading_client:
            return {"status": "error", "error": "Client not initialized"}
        
        try:
            # Get current price for slippage calculation
            mid_price = await self.client.get_mid_price(symbol)
            if mid_price <= 0:
                return {"status": "error", "error": "Failed to get market price"}
            
            # Calculate limit price with slippage (market orders need a price on Extended)
            if is_buy:
                price = mid_price * (1 + slippage)
            else:
                price = mid_price * (1 - slippage)
            
            # Round price and size
            price = self.client.round_price(symbol, price)
            size = self.client.round_size(symbol, size)
            
            # Place order using SDK
            response = await self.client.trading_client.place_order(
                market_name=symbol,
                amount_of_synthetic=Decimal(str(size)),
                price=Decimal(str(price)),
                side=OrderSide.BUY if is_buy else OrderSide.SELL,
                post_only=False,  # Market order
                time_in_force=TimeInForce.IOC,  # Immediate or cancel
                reduce_only=reduce_only,
            )
            
            if response.status == "OK" and response.data:
                order_id = response.data.id
                logger.info(f"✅ Market {'BUY' if is_buy else 'SELL'} {size} {symbol} @ ~{price}")
                return {
                    "status": "ok",
                    "order_id": order_id,
                    "data": {
                        "symbol": symbol,
                        "side": "buy" if is_buy else "sell",
                        "size": size,
                        "price": price,
                        "type": "market",
                    }
                }
            else:
                error = response.error if hasattr(response, 'error') else "Unknown error"
                logger.error(f"❌ Market order failed: {error}")
                return {"status": "error", "error": str(error)}
                
        except Exception as e:
            logger.error(f"Market order error: {e}")
            return {"status": "error", "error": str(e)}
    
    # ==================== LIMIT ORDERS ====================
    async def limit_buy(
        self,
        symbol: str,
        size: float,
        price: float,
        post_only: bool = False,
        reduce_only: bool = False,
        expire_hours: int = 24,
    ) -> Dict:
        """Place a limit buy order."""
        return await self._place_limit_order(
            symbol=symbol,
            size=size,
            price=price,
            is_buy=True,
            post_only=post_only,
            reduce_only=reduce_only,
            expire_hours=expire_hours,
        )
    
    async def limit_sell(
        self,
        symbol: str,
        size: float,
        price: float,
        post_only: bool = False,
        reduce_only: bool = False,
        expire_hours: int = 24,
    ) -> Dict:
        """Place a limit sell order."""
        return await self._place_limit_order(
            symbol=symbol,
            size=size,
            price=price,
            is_buy=False,
            post_only=post_only,
            reduce_only=reduce_only,
            expire_hours=expire_hours,
        )
    
    async def _place_limit_order(
        self,
        symbol: str,
        size: float,
        price: float,
        is_buy: bool,
        post_only: bool = False,
        reduce_only: bool = False,
        expire_hours: int = 24,
    ) -> Dict:
        """Internal limit order placement."""
        if not self.client.trading_client:
            return {"status": "error", "error": "Client not initialized"}
        
        try:
            # Round price and size
            price = self.client.round_price(symbol, price)
            size = self.client.round_size(symbol, size)
            
            # Calculate expiry time
            expire_time = datetime.now(timezone.utc) + timedelta(hours=expire_hours)
            
            # Place order using SDK
            response = await self.client.trading_client.place_order(
                market_name=symbol,
                amount_of_synthetic=Decimal(str(size)),
                price=Decimal(str(price)),
                side=OrderSide.BUY if is_buy else OrderSide.SELL,
                post_only=post_only,
                time_in_force=TimeInForce.GTT,
                reduce_only=reduce_only,
                expire_time=expire_time,
            )
            
            if response.status == "OK" and response.data:
                order_id = response.data.id
                logger.info(f"✅ Limit {'BUY' if is_buy else 'SELL'} {size} {symbol} @ {price}")
                return {
                    "status": "ok",
                    "order_id": order_id,
                    "data": {
                        "symbol": symbol,
                        "side": "buy" if is_buy else "sell",
                        "size": size,
                        "price": price,
                        "type": "limit",
                    }
                }
            else:
                error = response.error if hasattr(response, 'error') else "Unknown error"
                logger.error(f"❌ Limit order failed: {error}")
                return {"status": "error", "error": str(error)}
                
        except Exception as e:
            logger.error(f"Limit order error: {e}")
            return {"status": "error", "error": str(e)}
    
    # ==================== MARKET ENTRY WITH TP/SL ====================
    async def market_entry_with_tpsl(
        self,
        symbol: str,
        is_buy: bool,
        size: float,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        slippage: float = 0.01,
        entry_price: Optional[float] = None,
    ) -> Dict:
        """
        Market entry with Take Profit and Stop Loss.
        
        Extended Exchange supports TPSL orders attached to the main order.
        
        Args:
            symbol: Trading pair (e.g., BTC-USD)
            is_buy: True for long, False for short
            size: Position size
            tp_price: Take profit trigger price
            sl_price: Stop loss trigger price
            slippage: Max slippage for entry
            entry_price: Specific entry price (optional)
        
        Returns:
            Order result dict
        """
        if not self.client.trading_client:
            return {"status": "error", "error": "Client not initialized"}
        
        try:
            # Get market info
            market = await self._get_market(symbol)
            if not market:
                return {"status": "error", "error": f"Market {symbol} not found"}
            
            # Get current price
            if entry_price:
                price = entry_price
            else:
                mid_price = await self.client.get_mid_price(symbol)
                if mid_price <= 0:
                    return {"status": "error", "error": "Failed to get market price"}
                price = mid_price * (1 + slippage) if is_buy else mid_price * (1 - slippage)
            
            # Round values
            price = self.client.round_price(symbol, price)
            size = self.client.round_size(symbol, size)
            
            # Build TP/SL parameters
            take_profit = None
            stop_loss = None
            
            if tp_price:
                tp_price = self.client.round_price(symbol, tp_price)
                take_profit = OrderTpslTriggerParam(
                    trigger_price=Decimal(str(tp_price)),
                    trigger_price_type=OrderTriggerPriceType.LAST,
                    price=Decimal(str(tp_price)),
                    price_type=OrderPriceType.MARKET,
                )
            
            if sl_price:
                sl_price = self.client.round_price(symbol, sl_price)
                stop_loss = OrderTpslTriggerParam(
                    trigger_price=Decimal(str(sl_price)),
                    trigger_price_type=OrderTriggerPriceType.LAST,
                    price=Decimal(str(sl_price)),
                    price_type=OrderPriceType.MARKET,
                )
            
            # Place order with TPSL
            response = await self.client.trading_client.place_order(
                market_name=symbol,
                amount_of_synthetic=Decimal(str(size)),
                price=Decimal(str(price)),
                side=OrderSide.BUY if is_buy else OrderSide.SELL,
                post_only=False,
                time_in_force=TimeInForce.IOC,
                tp_sl_type=OrderTpslType.ORDER if (tp_price or sl_price) else None,
                take_profit=take_profit,
                stop_loss=stop_loss,
            )
            
            if response.status == "OK" and response.data:
                order_id = response.data.id
                
                # Track position orders
                self.position_orders[symbol] = {
                    "entry_order_id": order_id,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "side": "long" if is_buy else "short",
                    "size": size,
                }
                
                logger.info(f"✅ Entry {'LONG' if is_buy else 'SHORT'} {size} {symbol} @ ~{price}")
                if tp_price:
                    logger.info(f"   TP: {tp_price}")
                if sl_price:
                    logger.info(f"   SL: {sl_price}")
                
                return {
                    "status": "ok",
                    "order_id": order_id,
                    "data": {
                        "symbol": symbol,
                        "side": "buy" if is_buy else "sell",
                        "size": size,
                        "price": price,
                        "tp_price": tp_price,
                        "sl_price": sl_price,
                    }
                }
            else:
                error = response.error if hasattr(response, 'error') else "Unknown error"
                logger.error(f"❌ Entry with TPSL failed: {error}")
                return {"status": "error", "error": str(error)}
                
        except Exception as e:
            logger.error(f"Entry with TPSL error: {e}")
            return {"status": "error", "error": str(e)}
    
    # Alias for compatibility
    atomic_market_entry_with_tpsl = market_entry_with_tpsl
    
    # ==================== CLOSE POSITION ====================
    async def close_position(
        self,
        symbol: str,
        size: Optional[float] = None,
        slippage: float = 0.01,
    ) -> Dict:
        """
        Close a position (full or partial).
        
        Args:
            symbol: Trading pair
            size: Size to close (None for full position)
            slippage: Max slippage
        
        Returns:
            Order result dict
        """
        try:
            # Get current position
            position = await self.client.get_position(symbol)
            if not position or position.get('size', 0) == 0:
                logger.info(f"No position to close for {symbol}")
                return {"status": "ok", "message": "No position"}
            
            pos_size = abs(float(position['size']))
            is_long = position['side'] == 'long'
            
            # Determine close size
            close_size = size if size else pos_size
            close_size = min(close_size, pos_size)  # Can't close more than we have
            
            # Close by placing opposite order
            if is_long:
                result = await self.market_sell(
                    symbol=symbol,
                    size=close_size,
                    reduce_only=True,
                    slippage=slippage,
                )
            else:
                result = await self.market_buy(
                    symbol=symbol,
                    size=close_size,
                    reduce_only=True,
                    slippage=slippage,
                )
            
            # Clear position tracking
            if result.get("status") == "ok":
                if symbol in self.position_orders:
                    del self.position_orders[symbol]
            
            return result
            
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return {"status": "error", "error": str(e)}
    
    # ==================== CANCEL ORDERS ====================
    async def cancel_order(self, order_id: int) -> Dict:
        """Cancel a single order by ID."""
        if not self.client.trading_client:
            return {"status": "error", "error": "Client not initialized"}
        
        try:
            response = await self.client.trading_client.orders.cancel_order(order_id=order_id)
            logger.info(f"Cancelled order {order_id}")
            return {"status": "ok", "order_id": order_id}
        except Exception as e:
            logger.error(f"Cancel order error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict:
        """
        Cancel all open orders, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol to filter by
        
        Returns:
            Result dict
        """
        if not self.client.trading_client:
            return {"status": "error", "error": "Client not initialized"}
        
        try:
            markets = [symbol] if symbol else None
            await self.client.trading_client.orders.mass_cancel(
                markets=markets,
                cancel_all=True if not symbol else False,
            )
            logger.info(f"Cancelled all orders" + (f" for {symbol}" if symbol else ""))
            return {"status": "ok"}
        except Exception as e:
            logger.error(f"Cancel all orders error: {e}")
            return {"status": "error", "error": str(e)}
    
    # ==================== POSITION TPSL ====================
    async def set_position_tpsl(
        self,
        symbol: str,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
    ) -> Dict:
        """
        Set or update TP/SL for an existing position.
        
        Args:
            symbol: Trading pair
            tp_price: Take profit price
            sl_price: Stop loss price
        
        Returns:
            Result dict
        """
        # This would need to use the position TPSL endpoint
        # For now, log a warning as this requires additional SDK methods
        logger.warning(f"Position TPSL update not yet implemented for {symbol}")
        return {"status": "warning", "message": "Not implemented"}
    
    # ==================== TRAILING STOP ====================
    async def update_trailing_stop(
        self,
        symbol: str,
        new_sl_price: float,
    ) -> Dict:
        """
        Update trailing stop loss for a position.
        
        Args:
            symbol: Trading pair
            new_sl_price: New stop loss price
        
        Returns:
            Result dict
        """
        # First cancel existing SL, then place new one
        return await self.set_position_tpsl(symbol, sl_price=new_sl_price)
