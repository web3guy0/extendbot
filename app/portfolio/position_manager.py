"""
Position Manager - Smart Position Monitoring & Management

Features:
1. Detect & manage manual positions (add TP/SL)
2. Validate entry quality (is price good?)
3. Monitor setup health (early exit on deterioration)
4. Trailing stop management
5. Break-even protection

This runs alongside the main trading loop to manage ALL positions,
whether opened by the bot or manually by the user.
"""

import asyncio
import json
import logging
from decimal import Decimal
from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# File to persist bot position tracking across restarts
BOT_POSITIONS_FILE = Path("data/bot_positions.json")


class PositionHealth(Enum):
    """Position health status"""
    EXCELLENT = "excellent"  # Strong in profit, setup intact
    GOOD = "good"           # In profit or at entry
    WARNING = "warning"     # Setup weakening
    CRITICAL = "critical"   # Setup failed, consider exit
    

class ExitReason(Enum):
    """Reasons for early exit"""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    SETUP_FAILED = "setup_failed"
    REGIME_CHANGED = "regime_changed"
    MOMENTUM_LOST = "momentum_lost"
    BREAK_EVEN = "break_even"
    MANUAL = "manual"


@dataclass
class ManagedPosition:
    """Tracked position with management state"""
    symbol: str
    size: float
    side: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    
    # TP/SL
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_order_id: Optional[int] = None
    sl_order_id: Optional[int] = None
    
    # Trailing stop
    trailing_active: bool = False
    trailing_distance_pct: float = 1.5  # 1.5% trail
    highest_price: float = 0.0  # For longs
    lowest_price: float = float('inf')  # For shorts
    
    # Break-even
    break_even_activated: bool = False
    break_even_trigger_pct: float = 1.0  # Move SL to entry after 1% profit
    
    # Health tracking
    health: PositionHealth = PositionHealth.GOOD
    health_checks_failed: int = 0
    last_health_check: Optional[datetime] = None
    
    # Source tracking
    is_manual: bool = False  # True if opened manually (not by bot)
    managed_by_bot: bool = False  # True once bot has set TP/SL
    
    # Performance
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    max_profit_pct: float = 0.0
    max_drawdown_pct: float = 0.0


class PositionManager:
    """
    Smart Position Manager
    
    Responsibilities:
    1. Detect new positions (manual or bot-created)
    2. Set TP/SL on unprotected positions
    3. Monitor position health using strategy signals
    4. Execute early exits when setup fails
    5. Manage trailing stops
    6. Move stops to break-even
    """
    
    def __init__(self, client, order_manager, strategy=None, config: Dict = None):
        """
        Initialize Position Manager
        
        Args:
            client: HyperLiquid client for position queries
            order_manager: Order manager for placing TP/SL
            strategy: Strategy instance for health checks
            config: Configuration options
        """
        self.client = client
        self.order_manager = order_manager
        self.strategy = strategy
        self.config = config or {}
        
        # Managed positions
        self.positions: Dict[str, ManagedPosition] = {}
        self.known_position_ids: Set[str] = set()
        self._positions_lock = asyncio.Lock()  # Thread safety for position modifications
        
        # Load persisted bot positions from file
        self._load_bot_positions()
        
        # Configuration
        self.auto_tpsl_enabled = self.config.get('auto_tpsl', True)
        self.health_check_enabled = self.config.get('health_check', True)
        # NOTE: Trailing stop disabled here - handled by bot.py with proper throttling
        # Having it in both places causes duplicate orders!
        self.trailing_stop_enabled = False  # self.config.get('trailing_stop', True)
        self.break_even_enabled = self.config.get('break_even', True)
        self.early_exit_enabled = self.config.get('early_exit', True)
        
        # Default TP/SL percentages for manual positions
        self.default_tp_pct = Decimal(str(self.config.get('default_tp_pct', 3.0)))
        self.default_sl_pct = Decimal(str(self.config.get('default_sl_pct', 1.5)))
        
        # Health check thresholds
        self.health_check_interval = timedelta(seconds=30)
        self.max_health_failures = 3  # Exit after 3 consecutive failures
        
        # Trailing stop settings
        self.trailing_activation_pct = Decimal(str(self.config.get('trailing_activation_pct', 1.5)))
        self.trailing_distance_pct = Decimal(str(self.config.get('trailing_distance_pct', 1.0)))
        
        # Break-even settings
        self.break_even_trigger_pct = Decimal(str(self.config.get('break_even_trigger_pct', 1.0)))
        
        logger.info("📊 Position Manager initialized")
        logger.info(f"   Auto TP/SL: {self.auto_tpsl_enabled}")
        logger.info(f"   Health Checks: {self.health_check_enabled}")
        logger.info(f"   Trailing Stop: {self.trailing_stop_enabled}")
        logger.info(f"   Break-Even: {self.break_even_enabled}")
        logger.info(f"   Early Exit: {self.early_exit_enabled}")
    
    @property
    def managed_positions(self) -> Dict[str, ManagedPosition]:
        """Get all currently managed positions"""
        return self.positions
    
    def remove_position(self, symbol: str):
        """Remove a position from tracking (after it's closed)"""
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"🗑️ Removed {symbol} from position tracking")
    
    async def scan_positions(self) -> List[ManagedPosition]:
        """
        Scan for all open positions and detect new/manual ones.
        
        Returns:
            List of newly detected positions
        """
        try:
            # Get current positions from exchange (sync call)
            current_positions = self.client.get_open_positions()
            if current_positions is None:
                current_positions = []
            new_positions = []
            
            # Thread-safe position modification
            async with self._positions_lock:
                # Track which symbols have positions
                current_symbols = set()
                
                for pos in current_positions:
                    symbol = pos['symbol']
                    current_symbols.add(symbol)
                    
                    # Create position key
                    pos_key = f"{symbol}_{pos['side']}"
                    
                    if pos_key not in self.positions:
                        # New position detected!
                        is_bot_position = pos_key in self.known_position_ids
                        
                        managed = ManagedPosition(
                            symbol=symbol,
                            size=abs(pos['size']),
                            side=pos['side'],
                            entry_price=pos['entry_price'],
                            entry_time=datetime.now(timezone.utc),
                            is_manual=not is_bot_position,  # True if NOT in known_position_ids
                            unrealized_pnl=pos.get('unrealized_pnl', 0),
                            # CRITICAL: If bot created this position, it already set TP/SL
                            # Don't set TP/SL again or we get duplicates!
                            managed_by_bot=is_bot_position,
                        )
                        
                        # Calculate PnL percentage
                        if managed.entry_price > 0:
                            if managed.side == 'long':
                                current = pos.get('mark_price', managed.entry_price)
                                managed.unrealized_pnl_pct = (current - managed.entry_price) / managed.entry_price * 100
                            else:
                                current = pos.get('mark_price', managed.entry_price)
                                managed.unrealized_pnl_pct = (managed.entry_price - current) / managed.entry_price * 100
                        
                        self.positions[pos_key] = managed
                        new_positions.append(managed)
                        
                        source = "MANUAL" if managed.is_manual else "BOT"
                        logger.info(f"📍 New {source} position detected: {symbol} {pos['side'].upper()} @ ${pos['entry_price']:.4f}")
                    else:
                        # Update existing position
                        managed = self.positions[pos_key]
                        managed.size = abs(pos['size'])
                        managed.unrealized_pnl = pos.get('unrealized_pnl', 0)
                        
                        # Update PnL tracking
                        if managed.entry_price > 0:
                            current = pos.get('mark_price', managed.entry_price)
                            if managed.side == 'long':
                                managed.unrealized_pnl_pct = (current - managed.entry_price) / managed.entry_price * 100
                                if current > managed.highest_price:
                                    managed.highest_price = current
                            else:
                                managed.unrealized_pnl_pct = (managed.entry_price - current) / managed.entry_price * 100
                                if current < managed.lowest_price:
                                    managed.lowest_price = current
                            
                            # Track max profit
                            if managed.unrealized_pnl_pct > managed.max_profit_pct:
                                managed.max_profit_pct = managed.unrealized_pnl_pct
                            
                            # Track max drawdown from peak
                            drawdown = managed.max_profit_pct - managed.unrealized_pnl_pct
                            if drawdown > managed.max_drawdown_pct:
                                managed.max_drawdown_pct = drawdown
                
                # Remove closed positions - simplified comprehension
                closed_symbols = set(self.positions.keys()) - {f"{p['symbol']}_{p['side']}" for p in current_positions}
                for pos_key in closed_symbols:
                    logger.info(f"📤 Position closed: {pos_key}")
                    del self.positions[pos_key]
                    # Also remove from known bot positions
                    symbol, side = pos_key.rsplit('_', 1)
                    self.unmark_position(symbol, side)
            
            return new_positions
            
        except Exception as e:
            logger.error(f"Error scanning positions: {e}")
            return []
    
    async def manage_position(self, position: ManagedPosition, candles: List[Dict] = None) -> Optional[ExitReason]:
        """
        Manage a single position - set TP/SL, check health, etc.
        
        Args:
            position: The position to manage
            candles: Current candle data for health checks
            
        Returns:
            ExitReason if position should be closed, None otherwise
        """
        try:
            # 1. Set TP/SL if not protected
            # CRITICAL: Always check for missing TP/SL, even for bot positions
            # This handles: restart after cancelling orders, orphaned positions, etc.
            if self.auto_tpsl_enabled:
                await self._ensure_position_protection(position)
            
            # 2. Check break-even
            if self.break_even_enabled and not position.break_even_activated:
                await self._check_break_even(position)
            
            # 3. Update trailing stop
            if self.trailing_stop_enabled and position.trailing_active:
                await self._update_trailing_stop(position)
            
            # 4. Health check (can we stay in this trade?)
            if self.health_check_enabled and candles:
                exit_reason = await self._check_position_health(position, candles)
                if exit_reason:
                    return exit_reason
            
            return None
            
        except Exception as e:
            logger.error(f"Error managing position {position.symbol}: {e}")
            return None
    
    async def _ensure_position_protection(self, position: ManagedPosition):
        """
        Ensure position has TP/SL protection.
        This checks ACTUAL orders, not just managed_by_bot flag.
        Handles: restart after cancelling orders, orphaned positions, etc.
        
        THREAD SAFETY: Uses _positions_lock to prevent race conditions
        """
        try:
            # Thread-safe check and update
            async with self._positions_lock:
                # Throttle: Only check protection every 60 seconds per position
                now = datetime.now(timezone.utc)
                last_check = getattr(position, '_last_protection_check', None)
                if last_check and (now - last_check).total_seconds() < 60:
                    return  # Already checked recently
                position._last_protection_check = now
                
                # Check if we're already processing this position
                if getattr(position, '_protection_in_progress', False):
                    logger.debug(f"🔒 {position.symbol}: Protection already in progress, skipping")
                    return
                position._protection_in_progress = True
            
            try:
                # Get actual open orders for this symbol
                if hasattr(self.client, 'get_frontend_open_orders'):
                    open_orders = self.client.get_frontend_open_orders(position.symbol)
                else:
                    open_orders = self.client.get_open_orders(position.symbol)
                
                # Log for debugging
                logger.debug(f"🔍 {position.symbol}: Found {len(open_orders)} orders")
                
                # Check for existing TP/SL orders
                has_tp, has_sl = self._detect_tpsl_orders(open_orders, position.side)
                
                logger.debug(f"🔍 {position.symbol}: has_tp={has_tp}, has_sl={has_sl}")
                
                if has_tp and has_sl:
                    # Position is protected, mark as managed
                    async with self._positions_lock:
                        if not position.managed_by_bot:
                            position.managed_by_bot = True
                            logger.info(f"✅ {position.symbol} already has TP/SL protection")
                    return
                
                # Only set protection if BOTH are missing (prevent partial spam)
                if has_tp or has_sl:
                    # Has one but not both - log but don't spam
                    logger.info(f"ℹ️ {position.symbol} has partial protection: TP={has_tp}, SL={has_sl}")
                    if has_sl:
                        # Has SL, that's the important one - mark as protected
                        async with self._positions_lock:
                            position.managed_by_bot = True
                    return
                
                # Missing ALL protection! Set TP/SL
                logger.warning(f"⚠️ {position.symbol} missing protection! Setting TP/SL...")
                await self._set_position_protection(position, has_tp=has_tp, has_sl=has_sl)
                
            finally:
                # Always release the in-progress flag
                async with self._positions_lock:
                    position._protection_in_progress = False
            
        except Exception as e:
            logger.error(f"Error ensuring protection for {position.symbol}: {e}")
            async with self._positions_lock:
                position._protection_in_progress = False
    
    def _detect_tpsl_orders(self, orders: List[Dict], position_side: str) -> Tuple[bool, bool]:
        """Detect if TP and SL orders exist for a position"""
        has_tp = False
        has_sl = False
        
        for o in orders:
            # Skip non-reduce-only orders (not TP/SL)
            if not o.get('reduceOnly', False):
                continue
            
            order_type = o.get('orderType', '')
            order_type_str = order_type if isinstance(order_type, str) else ''
            order_type_lower = order_type_str.lower()
            
            # Method 1: Check orderType string directly
            # "Stop Market" = SL, "Take Profit Market" = TP
            if 'take profit' in order_type_lower:
                has_tp = True
                continue
            if 'stop' in order_type_lower:
                has_sl = True
                continue
            
            # Method 2: Check trigger condition for trigger orders
            if o.get('isTrigger', False):
                trigger_cond = o.get('triggerCondition', '')
                order_side = o.get('side', '')  # 'A' = sell, 'B' = buy
                
                # Parse trigger condition: "Price above X" or "Price below X"
                if 'above' in trigger_cond.lower():
                    # Price above = SL for SHORT, TP for LONG
                    if position_side == 'short':
                        has_sl = True
                    else:
                        has_tp = True
                elif 'below' in trigger_cond.lower():
                    # Price below = TP for SHORT, SL for LONG
                    if position_side == 'short':
                        has_tp = True
                    else:
                        has_sl = True
            
            # Method 3: Check orderType dict (older format)
            if isinstance(order_type, dict):
                trigger = order_type.get('trigger', {})
                tpsl = trigger.get('tpsl', '')
                if tpsl == 'tp':
                    has_tp = True
                elif tpsl == 'sl':
                    has_sl = True
        
        return has_tp, has_sl

    async def _set_position_protection(self, position: ManagedPosition, has_tp: bool = False, has_sl: bool = False):
        """Set TP/SL on unprotected position - with verification"""
        try:
            # If we already have both, nothing to do
            if has_tp and has_sl:
                async with self._positions_lock:
                    position.managed_by_bot = True
                logger.info(f"✅ {position.symbol} already has TP/SL protection")
                return
            
            # Calculate TP/SL prices
            entry = Decimal(str(position.entry_price))
            
            if position.side == 'long':
                tp_price = float(entry * (1 + self.default_tp_pct / 100))
                sl_price = float(entry * (1 - self.default_sl_pct / 100))
            else:
                tp_price = float(entry * (1 - self.default_tp_pct / 100))
                sl_price = float(entry * (1 + self.default_sl_pct / 100))
            
            # Set TP/SL using order manager
            logger.info(f"🛡️ Setting protection for {position.symbol} {position.side.upper()}")
            logger.info(f"   Entry: ${position.entry_price:.4f}")
            logger.info(f"   TP: ${tp_price:.4f} (+{self.default_tp_pct}%)")
            logger.info(f"   SL: ${sl_price:.4f} (-{self.default_sl_pct}%)")
            
            result = self.order_manager.set_position_tpsl(
                symbol=position.symbol,
                tp_price=tp_price if not has_tp else None,
                sl_price=sl_price if not has_sl else None
            )
            
            if result.get('status') == 'ok':
                # Verify orders were actually placed by checking exchange again
                await asyncio.sleep(0.5)  # Small delay for order propagation
                
                if hasattr(self.client, 'get_frontend_open_orders'):
                    verify_orders = self.client.get_frontend_open_orders(position.symbol)
                else:
                    verify_orders = self.client.get_open_orders(position.symbol)
                
                verify_tp, verify_sl = self._detect_tpsl_orders(verify_orders, position.side)
                
                if (not has_tp and not verify_tp) or (not has_sl and not verify_sl):
                    logger.warning(f"⚠️ TP/SL verification failed! Expected TP={not has_tp}, SL={not has_sl}, Got TP={verify_tp}, SL={verify_sl}")
                    # Don't mark as managed - will retry next scan
                    return
                
                async with self._positions_lock:
                    position.tp_price = tp_price
                    position.sl_price = sl_price
                    position.managed_by_bot = True
                logger.info(f"✅ Protection set and verified for {position.symbol}")
            else:
                logger.warning(f"⚠️ Failed to set protection: {result}")
                
        except Exception as e:
            logger.error(f"Error setting position protection: {e}")
    
    async def _check_break_even(self, position: ManagedPosition):
        """Move stop to break-even when profit threshold reached"""
        try:
            if position.unrealized_pnl_pct >= float(self.break_even_trigger_pct):
                # Profit threshold reached - move SL to entry
                entry = position.entry_price
                
                # Add small buffer (0.1%) to lock in a tiny profit
                # For LONG: SL slightly above entry (locks in 0.1% profit when price drops)
                # For SHORT: SL slightly below entry (locks in 0.1% profit when price rises)
                if position.side == 'long':
                    new_sl = entry * 1.001  # Slightly above entry for longs
                else:
                    new_sl = entry * 0.999  # Slightly below entry for shorts (locks in profit)
                
                logger.info(f"🔒 Break-even activated for {position.symbol}")
                logger.info(f"   Profit: {position.unrealized_pnl_pct:.2f}%")
                logger.info(f"   Moving SL from ${position.sl_price:.4f} to ${new_sl:.4f}")
                
                # Cancel old SL and set new one
                # Note: This is a simplified version - production should be more careful
                result = self.order_manager.set_position_tpsl(
                    symbol=position.symbol,
                    sl_price=new_sl
                )
                
                if result.get('status') == 'ok':
                    position.sl_price = new_sl
                    position.break_even_activated = True
                    position.trailing_active = True  # Enable trailing from break-even
                    logger.info(f"✅ Break-even set, trailing enabled")
                    
        except Exception as e:
            logger.error(f"Error in break-even check: {e}")
    
    async def _update_trailing_stop(self, position: ManagedPosition):
        """Update trailing stop based on price movement"""
        try:
            # Get current price
            current_price = self.client.get_mid_price(position.symbol)
            if current_price <= 0:
                return
            
            # Calculate new trailing SL
            trail_distance = current_price * float(self.trailing_distance_pct) / 100
            
            if position.side == 'long':
                # Update highest price
                if current_price > position.highest_price:
                    position.highest_price = current_price
                
                # New SL trails below highest
                new_sl = position.highest_price - trail_distance
                
                # Only move SL up, never down
                if position.sl_price and new_sl > position.sl_price:
                    logger.info(f"📈 Trailing SL up: ${position.sl_price:.4f} → ${new_sl:.4f}")
                    self.order_manager.set_position_tpsl(position.symbol, sl_price=new_sl)
                    position.sl_price = new_sl
            else:
                # For shorts, update lowest price (our best profit point)
                if current_price < position.lowest_price:
                    position.lowest_price = current_price
                
                # For SHORT: SL trails ABOVE the lowest price (locks in profit as price drops)
                # The SL is a BUY order that triggers if price rises
                new_sl = position.lowest_price + trail_distance
                
                # For shorts, we move SL DOWN as price drops (tightening the stop)
                # But new_sl must still be ABOVE current price to be valid
                if new_sl > current_price and position.sl_price and new_sl < position.sl_price:
                    logger.info(f"📉 Trailing SL down: ${position.sl_price:.4f} → ${new_sl:.4f}")
                    self.order_manager.set_position_tpsl(position.symbol, sl_price=new_sl)
                    position.sl_price = new_sl
                    
        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
    
    async def _check_position_health(self, position: ManagedPosition, candles: List[Dict]) -> Optional[ExitReason]:
        """
        Check if setup is still valid using strategy analysis.
        
        Returns ExitReason if position should be closed early.
        """
        try:
            if not self.strategy:
                return None
            
            # Rate limit health checks
            now = datetime.now(timezone.utc)
            if position.last_health_check:
                if now - position.last_health_check < self.health_check_interval:
                    return None
            
            position.last_health_check = now
            
            # Use strategy to check if setup is still valid
            health_result = await self._evaluate_setup_health(position, candles)
            position.health = health_result
            
            if health_result == PositionHealth.CRITICAL:
                position.health_checks_failed += 1
                logger.warning(f"⚠️ {position.symbol} health check failed ({position.health_checks_failed}/{self.max_health_failures})")
                
                if position.health_checks_failed >= self.max_health_failures:
                    logger.warning(f"🚨 {position.symbol} setup failed - early exit triggered")
                    return ExitReason.SETUP_FAILED
            else:
                # Reset failure counter on good health
                position.health_checks_failed = 0
            
            return None
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return None
    
    async def _evaluate_setup_health(self, position: ManagedPosition, candles: List[Dict]) -> PositionHealth:
        """
        Evaluate if the trading setup is still valid.
        
        Checks:
        1. Is the trend still in our favor?
        2. Has momentum reversed?
        3. Did market regime change?
        4. Are we seeing reversal patterns?
        """
        try:
            if not candles or len(candles) < 20:
                return PositionHealth.GOOD  # Not enough data
            
            # Get current price
            current_price = candles[-1].get('close', 0)
            entry = position.entry_price
            
            # Calculate current profit
            if position.side == 'long':
                profit_pct = (current_price - entry) / entry * 100
            else:
                profit_pct = (entry - current_price) / entry * 100
            
            # Check 1: Major loss (more than half of SL)
            if profit_pct < -float(self.default_sl_pct) / 2:
                # Losing significantly - check if setup still valid
                
                # Simple momentum check using recent candles
                recent = candles[-5:]
                if position.side == 'long':
                    # For long: if all recent candles are bearish, setup failed
                    bearish_count = sum(1 for c in recent if c['close'] < c['open'])
                    if bearish_count >= 4:
                        return PositionHealth.CRITICAL
                else:
                    # For short: if all recent candles are bullish, setup failed
                    bullish_count = sum(1 for c in recent if c['close'] > c['open'])
                    if bullish_count >= 4:
                        return PositionHealth.CRITICAL
            
            # Check 2: Gave back too much profit
            if position.max_profit_pct > 1.5 and profit_pct < position.max_profit_pct * 0.3:
                # Had nice profit but gave back 70%+
                logger.warning(f"⚠️ {position.symbol} gave back {position.max_profit_pct - profit_pct:.1f}% profit")
                return PositionHealth.WARNING
            
            # Check 3: Use strategy's regime detector if available
            if hasattr(self.strategy, 'regime_detector') and hasattr(self.strategy, 'strategies'):
                for name, strat in self.strategy.strategies.items():
                    if hasattr(strat, 'regime_detector'):
                        regime = strat.regime_detector.detect(candles)
                        
                        # Check if regime is against our position
                        if position.side == 'long' and regime.regime_type == 'trending_down':
                            if regime.confidence > 0.7:
                                logger.warning(f"⚠️ Regime turned bearish against long position")
                                return PositionHealth.CRITICAL
                        elif position.side == 'short' and regime.regime_type == 'trending_up':
                            if regime.confidence > 0.7:
                                logger.warning(f"⚠️ Regime turned bullish against short position")
                                return PositionHealth.CRITICAL
                        break
            
            # Check 4: Profit evaluation
            if profit_pct > 2.0:
                return PositionHealth.EXCELLENT
            elif profit_pct > 0:
                return PositionHealth.GOOD
            elif profit_pct > -0.5:
                return PositionHealth.GOOD
            else:
                return PositionHealth.WARNING
                
        except Exception as e:
            logger.error(f"Error evaluating setup health: {e}")
            return PositionHealth.GOOD  # Default to good on error
    
    async def execute_early_exit(self, position: ManagedPosition, reason: ExitReason):
        """
        Execute early exit for a position.
        
        Args:
            position: Position to close
            reason: Reason for early exit
        """
        try:
            logger.warning(f"🚨 EARLY EXIT: {position.symbol} {position.side.upper()}")
            logger.warning(f"   Reason: {reason.value}")
            logger.warning(f"   Entry: ${position.entry_price:.4f}")
            logger.warning(f"   Current PnL: {position.unrealized_pnl_pct:.2f}%")
            
            # Close position using market order
            result = self.order_manager.market_close(position.symbol)
            
            if result.get('status') == 'ok' or 'response' in result:
                logger.info(f"✅ Early exit executed for {position.symbol}")
                
                # Remove from tracked positions (thread-safe)
                pos_key = f"{position.symbol}_{position.side}"
                async with self._positions_lock:
                    if pos_key in self.positions:
                        del self.positions[pos_key]
            else:
                logger.error(f"❌ Early exit failed: {result}")
                
        except Exception as e:
            logger.error(f"Error executing early exit: {e}")
    
    def validate_entry(self, symbol: str, side: str, entry_price: float, candles: List[Dict] = None) -> Dict[str, Any]:
        """
        Validate if an entry price is good.
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            entry_price: The entry price to validate
            candles: Current candle data
            
        Returns:
            Dict with 'valid', 'score', 'feedback'
        """
        try:
            feedback = []
            score = 50  # Start neutral
            
            if not candles or len(candles) < 20:
                return {'valid': True, 'score': 50, 'feedback': ['Not enough data to validate']}
            
            current = candles[-1]['close']
            
            # Check 1: Entry vs current price
            diff_pct = abs(entry_price - current) / current * 100
            if diff_pct > 1:
                feedback.append(f"⚠️ Entry is {diff_pct:.1f}% from current price")
                score -= 10
            
            # Check 2: Entry vs recent range
            recent_high = max(c['high'] for c in candles[-20:])
            recent_low = min(c['low'] for c in candles[-20:])
            
            if side == 'long':
                # For longs, better to enter near support
                if entry_price > recent_high * 0.98:
                    feedback.append("⚠️ Buying near resistance - risky")
                    score -= 20
                elif entry_price < recent_low * 1.02:
                    feedback.append("✅ Good entry near support")
                    score += 20
            else:
                # For shorts, better to enter near resistance
                if entry_price < recent_low * 1.02:
                    feedback.append("⚠️ Shorting near support - risky")
                    score -= 20
                elif entry_price > recent_high * 0.98:
                    feedback.append("✅ Good entry near resistance")
                    score += 20
            
            # Check 3: Trend alignment
            sma20 = sum(c['close'] for c in candles[-20:]) / 20
            if side == 'long' and current > sma20:
                feedback.append("✅ Price above SMA20 - trend aligned")
                score += 10
            elif side == 'short' and current < sma20:
                feedback.append("✅ Price below SMA20 - trend aligned")
                score += 10
            else:
                feedback.append("⚠️ Trading against short-term trend")
                score -= 10
            
            # Determine validity
            valid = score >= 40
            
            return {
                'valid': valid,
                'score': max(0, min(100, score)),
                'feedback': feedback
            }
            
        except Exception as e:
            logger.error(f"Error validating entry: {e}")
            return {'valid': True, 'score': 50, 'feedback': [f'Error: {e}']}
    
    def _load_bot_positions(self):
        """Load bot position tracking from persistent storage"""
        try:
            if BOT_POSITIONS_FILE.exists():
                with open(BOT_POSITIONS_FILE, 'r') as f:
                    data = json.load(f)
                    self.known_position_ids = set(data.get('known_position_ids', []))
                    logger.info(f"📂 Loaded {len(self.known_position_ids)} known bot positions from file")
        except Exception as e:
            logger.warning(f"Could not load bot positions file: {e}")
            self.known_position_ids = set()
    
    def _save_bot_positions(self):
        """Save bot position tracking to persistent storage"""
        try:
            BOT_POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(BOT_POSITIONS_FILE, 'w') as f:
                json.dump({
                    'known_position_ids': list(self.known_position_ids),
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save bot positions file: {e}")
    
    def mark_position_as_bot_created(self, symbol: str, side: str):
        """Mark a position as created by the bot (not manual) and persist"""
        pos_key = f"{symbol}_{side}"
        self.known_position_ids.add(pos_key)
        self._save_bot_positions()
        logger.info(f"🤖 Marked {pos_key} as bot-created position")
    
    def unmark_position(self, symbol: str, side: str):
        """Remove position from known bot positions (when closed)"""
        pos_key = f"{symbol}_{side}"
        if pos_key in self.known_position_ids:
            self.known_position_ids.discard(pos_key)
            self._save_bot_positions()
            logger.info(f"📤 Removed {pos_key} from bot positions tracking")
    
    def get_position_status(self, symbol: str = None) -> List[Dict]:
        """Get status of all managed positions"""
        positions = []
        for key, pos in self.positions.items():
            if symbol and pos.symbol != symbol:
                continue
            positions.append({
                'symbol': pos.symbol,
                'side': pos.side,
                'size': pos.size,
                'entry_price': pos.entry_price,
                'tp_price': pos.tp_price,
                'sl_price': pos.sl_price,
                'pnl_pct': pos.unrealized_pnl_pct,
                'health': pos.health.value,
                'is_manual': pos.is_manual,
                'managed': pos.managed_by_bot,
                'trailing': pos.trailing_active,
                'break_even': pos.break_even_activated,
            })
        return positions
