#!/usr/bin/env python3
"""
ExtendBot - Master Bot Controller for Extended Exchange
Orchestrates trading with complete risk management on Extended Exchange.

Based on HyperAI Bot architecture but using Extended Exchange SDK.

SDK Reference: https://github.com/x10xchange/python_sdk
API Docs: https://api.docs.extended.exchange
"""

import asyncio
import signal
import sys
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any, Optional, List
from types import FrameType
import json
import re

# Add to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


class SensitiveDataFilter(logging.Filter):
    """Filter to mask sensitive data in logs (tokens, API keys, URLs with tokens)"""
    
    def __init__(self):
        super().__init__()
        # Get sensitive tokens from environment
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.api_key = os.getenv('API_KEY', '')
        self.private_key = os.getenv('PRIVATE_KEY', '')
        
    def filter(self, record):
        """Mask sensitive data in log records"""
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            
            # Mask Telegram bot token in URLs
            if self.telegram_token and self.telegram_token in msg:
                if ':' in self.telegram_token:
                    bot_id = self.telegram_token.split(':')[0]
                    msg = msg.replace(self.telegram_token, f"{bot_id}:***MASKED***")
                else:
                    msg = msg.replace(self.telegram_token, "***MASKED***")
            
            # Mask API URLs with tokens using regex
            msg = re.sub(
                r'(https?://[^/]+/bot)(\d+:[A-Za-z0-9_-]+)',
                r'\1***MASKED***',
                msg
            )
            
            # Mask API key
            if self.api_key and len(self.api_key) > 10:
                msg = msg.replace(self.api_key, '***MASKED***')
            
            # Mask private key
            if self.private_key and len(self.private_key) > 10:
                msg = msg.replace(self.private_key, '***MASKED***')
            
            record.msg = msg
            
        return True


# Import Extended Exchange integration
from app.ex.ex_client import ExtendedClient
from app.ex.ex_websocket import ExtendedWebSocket
from app.ex.ex_order_manager import ExtendedOrderManager

# Import strategies
from app.strategies.strategy_manager import StrategyManager

# Import Position Manager
from app.portfolio.position_manager import PositionManager

# Import Multi-Asset Manager
from app.portfolio.multi_asset_manager import MultiAssetManager, get_multi_asset_manager

# Import risk management
from app.risk.risk_engine import RiskEngine
from app.risk.kill_switch import KillSwitch
from app.risk.drawdown_monitor import DrawdownMonitor
from app.risk.kelly_criterion import KellyCriterion, get_kelly_calculator
from app.risk.small_account_mode import SmallAccountMode, get_small_account_mode

# Import paper trading
from app.execution.paper_trading import PaperTradingEngine, is_paper_trading_enabled, get_paper_trading_balance

# Import Telegram bot
from app.tg_bot.bot import TelegramBot

# Import error handler
from app.utils.error_handler import ErrorHandler

# Import database
from app.database.db_manager import DatabaseManager

# Import indicator calculator
from app.utils.indicator_calculator import IndicatorCalculator

# Create logs directory
Path('logs').mkdir(exist_ok=True)

# Setup logging with sensitive data filter
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/bot_{datetime.now(timezone.utc).strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

# Add sensitive data filter to all handlers
sensitive_filter = SensitiveDataFilter()
for handler in logging.root.handlers:
    handler.addFilter(sensitive_filter)

logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('telegram').setLevel(logging.WARNING)
logging.getLogger('telegram.ext').setLevel(logging.WARNING)
logging.getLogger('websockets').setLevel(logging.WARNING)

# Global shutdown event
shutdown_event = asyncio.Event()


def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
    """Handle shutdown signals"""
    logger.info("\n🛑 Shutdown signal received, cleaning up...")
    shutdown_event.set()


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class AccountManagerProxy:
    """
    Dynamic proxy that always returns fresh values from the bot.
    """
    def __init__(self, bot: 'ExtendBot'):
        self._bot = bot
    
    @property
    def current_equity(self):
        return self._bot.account_value
    
    @property
    def current_balance(self):
        return self._bot.account_value
    
    @property
    def peak_equity(self):
        return self._bot.peak_equity
    
    @property
    def session_start_equity(self):
        return self._bot.session_start_equity
    
    @property
    def session_pnl(self):
        return self._bot.session_pnl
    
    @property
    def margin_used(self):
        return self._bot.margin_used


class PositionManagerProxy:
    """Dynamic proxy for position manager."""
    def __init__(self, bot: 'ExtendBot'):
        self._bot = bot
        self.open_positions = {}
    
    def get_position(self, symbol: str):
        return None


class ExtendBot:
    """
    Master trading bot controller for Extended Exchange.
    
    Features:
    - Full SDK integration with Stark signatures
    - Real-time WebSocket subscriptions
    - Multi-asset trading support
    - Complete risk management
    - Telegram notifications
    """
    
    def __init__(self):
        # Mode configuration
        self.mode = os.getenv('BOT_MODE', 'rule_based')
        self.symbol = os.getenv('SYMBOL', 'SOL-USD')  # Extended uses BTC-USD format
        
        # Timeframe configuration
        self.timeframe = os.getenv('TIMEFRAME', '1m')
        valid_timeframes = ['1m', '5m', '15m', '1h', '4h']
        if self.timeframe not in valid_timeframes:
            logger.warning(f"Invalid TIMEFRAME={self.timeframe}, using 1m")
            self.timeframe = '1m'
        
        # Multi-asset trading
        self.multi_asset_mode = os.getenv('MULTI_ASSET_MODE', 'false').lower() == 'true'
        multi_assets_env = os.getenv('MULTI_ASSETS', 'BTC-USD,ETH-USD,SOL-USD')
        self.multi_assets = [s.strip() for s in multi_assets_env.split(',') if s.strip()]
        self.max_positions = int(os.getenv('MAX_POSITIONS', '3'))
        
        # Multi-asset manager
        self.asset_manager: Optional[MultiAssetManager] = None
        
        # Strategies per symbol
        self.strategies: Dict[str, StrategyManager] = {}
        
        # Exchange components
        self.client: Optional[ExtendedClient] = None
        self.websocket: Optional[ExtendedWebSocket] = None
        self.order_manager: Optional[ExtendedOrderManager] = None
        
        # Strategy Manager
        self.strategy: Optional[StrategyManager] = None
        
        # Position Manager
        self.position_manager: Optional[PositionManager] = None
        
        # Risk management
        self.risk_engine: Optional[RiskEngine] = None
        self.kill_switch: Optional[KillSwitch] = None
        self.drawdown_monitor: Optional[DrawdownMonitor] = None
        self.kelly: Optional[KellyCriterion] = None
        self.small_account_mode: Optional[SmallAccountMode] = None
        
        # Paper Trading
        self.paper_trading: Optional[PaperTradingEngine] = None
        self.is_paper_trading = is_paper_trading_enabled()
        
        # Telegram bot
        self.telegram_bot: Optional[TelegramBot] = None
        
        # Position tracking
        self._position_details: Dict[str, Dict] = {}
        self._position_lock = asyncio.Lock()
        
        # Candle cache
        self._candles_cache: List[Dict[str, Any]] = []
        self._last_candle_fetch: Optional[datetime] = None
        self._candle_update_pending = False
        
        # BTC candles for correlation
        self._btc_candles_cache: List[Dict[str, Any]] = []
        self._last_btc_fetch: Optional[datetime] = None
        
        # HTF candles for multi-timeframe
        self._htf_candles_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._last_htf_fetch: Optional[datetime] = None
        self._htf_intervals = ['15m', '1h', '4h']
        
        # Indicator calculator
        self.indicator_calc: Optional[IndicatorCalculator] = None
        
        # Position monitoring
        self._last_position_check: Optional[datetime] = None
        self._position_check_interval = 3.0
        self._atr_value: Optional[Decimal] = None
        
        # Trailing stop throttle
        self._last_trail_update: Dict[str, datetime] = {}
        self._trail_update_interval = 30
        
        # Trade tracking
        self._active_trade_ids: Dict[str, Dict] = {}
        
        # State persistence
        self._state_file = Path('data/active_trades.json')
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Error handler
        self.error_handler: Optional[ErrorHandler] = None
        
        # Database
        self.db: Optional[DatabaseManager] = None
        
        # Account tracking
        self.account_value = Decimal('0')
        self.peak_equity = Decimal('0')
        self.session_start_equity = Decimal('0')
        self.session_pnl = Decimal('0')
        self.margin_used = Decimal('0')
        self.account_state: Dict[str, Any] = {}
        
        # State
        self.is_running = False
        self.is_paused = False
        self.trades_executed = 0
        self.start_time: Optional[datetime] = None
        
        # Data collection
        self.trade_log_path = Path('data/trades')
        self.trade_log_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("🤖 ExtendBot initialized")
        logger.info(f"   Mode: {self.mode}")
        logger.info(f"   Timeframe: {self.timeframe}")
        if self.multi_asset_mode:
            logger.info(f"   🌐 Multi-Asset Mode: ENABLED")
            logger.info(f"   Assets: {', '.join(self.multi_assets)}")
            logger.info(f"   Max Positions: {self.max_positions}")
        else:
            logger.info(f"   Symbol: {self.symbol}")
    
    # ==================== STATE PERSISTENCE ====================
    def _save_active_trades(self):
        """Persist active trades to file for crash recovery."""
        try:
            data = {}
            for symbol, info in self._active_trade_ids.items():
                data[symbol] = {
                    'trade_id': info.get('trade_id'),
                    'entry_price': str(info.get('entry_price', 0)),
                    'quantity': str(info.get('quantity', 0)),
                    'side': info.get('side'),
                    'entry_time': info.get('entry_time').isoformat() if info.get('entry_time') else None,
                    'tp_price': str(info.get('tp_price', 0)) if info.get('tp_price') else None,
                    'sl_price': str(info.get('sl_price', 0)) if info.get('sl_price') else None,
                }
            self._state_file.write_text(json.dumps(data, indent=2))
            logger.debug(f"💾 Saved {len(data)} active trades to state file")
        except Exception as e:
            logger.error(f"Failed to save active trades state: {e}")
    
    def _load_active_trades(self):
        """Load persisted active trades on startup."""
        try:
            if not self._state_file.exists():
                return False
            
            data = json.loads(self._state_file.read_text())
            if not data:
                return False
            
            recovered = 0
            for symbol, info in data.items():
                entry_time = None
                if info.get('entry_time'):
                    try:
                        entry_time = datetime.fromisoformat(info['entry_time'])
                    except ValueError:
                        entry_time = datetime.now(timezone.utc)
                
                self._active_trade_ids[symbol] = {
                    'trade_id': info.get('trade_id'),
                    'entry_price': Decimal(str(info.get('entry_price', 0))),
                    'quantity': Decimal(str(info.get('quantity', 0))),
                    'side': info.get('side'),
                    'entry_time': entry_time,
                    'tp_price': Decimal(str(info.get('tp_price'))) if info.get('tp_price') else None,
                    'sl_price': Decimal(str(info.get('sl_price'))) if info.get('sl_price') else None,
                }
                recovered += 1
            
            if recovered > 0:
                logger.info(f"🔄 Recovered {recovered} active trades from previous session")
            return recovered > 0
            
        except Exception as e:
            logger.warning(f"Failed to load active trades state: {e}")
            return False
    
    def _clear_trade_state(self, symbol: str):
        """Remove a trade from active tracking."""
        if symbol in self._active_trade_ids:
            del self._active_trade_ids[symbol]
            self._save_active_trades()
    
    async def initialize(self) -> bool:
        """Initialize all components."""
        try:
            logger.info("🔧 Initializing ExtendBot components...")
            
            # Check for Paper Trading Mode
            if self.is_paper_trading:
                paper_balance = get_paper_trading_balance()
                self.paper_trading = PaperTradingEngine(paper_balance)
                logger.info("📝 ═══════════════════════════════════════")
                logger.info("📝 PAPER TRADING MODE ENABLED")
                logger.info(f"📝 Virtual Balance: ${paper_balance}")
                logger.info("📝 No real trades will be executed!")
                logger.info("📝 ═══════════════════════════════════════")
            
            # Load Extended Exchange credentials
            api_key = os.getenv('API_KEY')
            private_key = os.getenv('PRIVATE_KEY')
            public_key = os.getenv('PUBLIC_KEY')
            vault_id = os.getenv('VAULT_ID')
            testnet = os.getenv('TESTNET', 'false').lower() == 'true'
            
            if not all([api_key, private_key, public_key, vault_id]):
                raise ValueError("Missing Extended Exchange credentials in .env file")
            
            # Initialize Extended Exchange client
            self.client = ExtendedClient(
                api_key=api_key,
                private_key=private_key,
                public_key=public_key,
                vault_id=int(vault_id),
                testnet=testnet
            )
            await self.client.initialize()
            
            # Initialize WebSocket
            self.websocket = ExtendedWebSocket(
                api_key=api_key,
                testnet=testnet
            )
            
            # Subscribe to market data
            if self.multi_asset_mode:
                for asset in self.multi_assets:
                    await self.websocket.subscribe_orderbook(asset, self._on_orderbook)
                    await self.websocket.subscribe_candles(asset, self.timeframe, self._on_new_candle)
                logger.info(f"📊 Subscribed to data for: {', '.join(self.multi_assets)}")
            else:
                await self.websocket.subscribe_orderbook(self.symbol, self._on_orderbook)
                await self.websocket.subscribe_candles(self.symbol, self.timeframe, self._on_new_candle)
                logger.info(f"📊 Subscribed to data for: {self.symbol}")
            
            # Subscribe to account updates (authenticated)
            await self.websocket.subscribe_account_updates(self._on_account_update)
            
            # Start WebSocket connection
            await self.websocket.start()
            logger.info("📊 WebSocket connection active")
            
            # Initialize Order Manager
            self.order_manager = ExtendedOrderManager(self.client, on_fill=self._on_fill)
            
            # Set leverage
            leverage = int(os.getenv('MAX_LEVERAGE', '5'))
            
            # Initialize strategies
            if self.multi_asset_mode:
                logger.info(f"🌐 Multi-Asset Mode: Initializing {len(self.multi_assets)} assets...")
                
                self.asset_manager = get_multi_asset_manager(
                    enabled_assets=self.multi_assets,
                    max_positions=self.max_positions
                )
                
                for asset in self.multi_assets:
                    await self.order_manager.set_leverage(asset, leverage)
                    self.strategies[asset] = StrategyManager(asset)
                    logger.info(f"   ✅ {asset}: Strategy + {leverage}x leverage")
                
                self.strategy = self.strategies.get(self.multi_assets[0])
                logger.info(f"🌐 Multi-Asset Mode: {len(self.strategies)} strategies initialized")
            else:
                await self.order_manager.set_leverage(self.symbol, leverage)
                logger.info(f"⚙️  Leverage set to {leverage}x for {self.symbol}")
                
                logger.info(f"🎯 Initializing Strategy Manager for {self.symbol}")
                self.strategy = StrategyManager(self.symbol)
            
            # Initialize indicator calculator
            self.indicator_calc = IndicatorCalculator()
            logger.info("📊 Shared indicator calculator initialized")
            
            # Initialize Position Manager
            position_manager_config = {
                'check_interval_seconds': 30,
                'auto_tpsl': True,
                'early_exit': True,
                'health_check': True,
                'trailing_stop': True,
                'break_even': True,
                'default_tp_pct': float(os.getenv('DEFAULT_TP_PCT', '3.0')),
                'default_sl_pct': float(os.getenv('DEFAULT_SL_PCT', '1.5')),
            }
            self.position_manager = PositionManager(
                client=self.client,
                order_manager=self.order_manager,
                strategy=self.strategy,
                config=position_manager_config
            )
            logger.info("🎯 Position Manager initialized")
            
            # Get initial account state
            await self.update_account_state()
            
            # Check for small account mode
            self.small_account_mode = get_small_account_mode(self.account_value)
            if self.small_account_mode and self.small_account_mode.is_small_account:
                logger.info("💰 SMALL ACCOUNT MODE ACTIVATED")
                logger.info(f"   Account value: ${self.account_value:.2f}")
                self.small_account_mode.apply_config()
            
            # Initialize risk management
            account_manager_proxy = AccountManagerProxy(self)
            position_manager_proxy = PositionManagerProxy(self)
            
            risk_config = {
                'max_position_size_pct': float(os.getenv('MAX_POSITION_SIZE_PCT', '55')),
                'max_positions': int(os.getenv('MAX_POSITIONS', '1')),
                'max_leverage': int(os.getenv('MAX_LEVERAGE', '5')),
                'max_daily_loss_pct': float(os.getenv('MAX_DAILY_LOSS_PCT', '5')),
                'max_drawdown_pct': float(os.getenv('MAX_DRAWDOWN_PCT', '10'))
            }
            self.risk_engine = RiskEngine(account_manager_proxy, position_manager_proxy, risk_config)
            
            kill_switch_config = {
                'daily_loss_trigger_pct': float(os.getenv('KILL_SWITCH_DAILY_LOSS_PCT', os.getenv('MAX_DAILY_LOSS_PCT', '5'))),
                'drawdown_trigger_pct': float(os.getenv('KILL_SWITCH_DRAWDOWN_PCT', os.getenv('MAX_DRAWDOWN_PCT', '10')))
            }
            self.kill_switch = KillSwitch(account_manager_proxy, position_manager_proxy, kill_switch_config)
            
            drawdown_config = {
                'warning_threshold_pct': 5,
                'critical_threshold_pct': float(os.getenv('MAX_DRAWDOWN_PCT', '10')),
                'auto_pause_enabled': True,
                'auto_pause_threshold_pct': 12
            }
            self.drawdown_monitor = DrawdownMonitor(account_manager_proxy, drawdown_config)
            
            # Initialize Kelly Criterion
            kelly_enabled = os.getenv('KELLY_ENABLED', 'true').lower() == 'true'
            if kelly_enabled:
                kelly_fraction = float(os.getenv('KELLY_FRACTION', '0.5'))
                kelly_min_trades = int(os.getenv('KELLY_MIN_TRADES', '20'))
                kelly_max_pct = float(os.getenv('KELLY_MAX_POSITION_PCT', '25'))
                self.kelly = KellyCriterion(
                    kelly_fraction=kelly_fraction,
                    min_trades=kelly_min_trades,
                    max_position_pct=kelly_max_pct
                )
                logger.info(f"📊 Kelly Criterion enabled: {kelly_fraction:.0%} Kelly")
            
            # Initialize Telegram bot
            if os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'):
                try:
                    logger.info("📱 Initializing Telegram bot...")
                    config = {
                        'max_leverage': int(os.getenv('MAX_LEVERAGE', '5')),
                        'max_daily_loss_pct': float(os.getenv('MAX_DAILY_LOSS_PCT', '5'))
                    }
                    self.telegram_bot = TelegramBot(self, config)
                    await self.telegram_bot.start()
                    
                    self.error_handler = ErrorHandler(self.telegram_bot)
                    logger.info("🛡️ Error handler initialized")
                    
                    # Initialize database
                    database_url = os.getenv('DATABASE_URL')
                    if database_url:
                        try:
                            logger.info("📊 Connecting to database...")
                            self.db = DatabaseManager(database_url)
                            await self.db.connect()
                            logger.info("✅ Database connected")
                        except Exception as db_error:
                            logger.error(f"❌ Database connection failed: {db_error}")
                            self.db = None
                    
                except Exception as e:
                    logger.warning(f"⚠️ Telegram bot initialization failed: {e}")
                    self.telegram_bot = None
            
            logger.info("✅ All components initialized")
            logger.info(f"💰 Starting Balance: ${self.account_value:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}", exc_info=True)
            return False
    
    async def update_account_state(self):
        """Update account state from exchange."""
        try:
            account_state = await self.client.get_account_state()
            
            old_value = self.account_value
            self.account_value = Decimal(str(account_state.get('account_value', 0)))
            self.margin_used = Decimal(str(account_state.get('margin_used', 0)))
            
            value_change = abs(float(self.account_value - old_value)) if old_value > 0 else 999
            if value_change > 0.50 or old_value == 0:
                logger.info(f"📊 Account updated: value=${self.account_value:.2f}, margin=${self.margin_used:.2f}")
            
            if self.account_value > self.peak_equity:
                self.peak_equity = self.account_value
            
            if self.session_start_equity == 0:
                self.session_start_equity = self.account_value
            
            self.session_pnl = self.account_value - self.session_start_equity
            self.account_state = account_state
            
        except Exception as e:
            logger.error(f"Error updating account state: {e}")
    
    def _on_orderbook(self, symbol: str, data: Dict[str, Any]):
        """Callback for orderbook updates."""
        pass  # Used for price data
    
    def _on_new_candle(self, symbol: str, candle: Dict[str, Any]):
        """Callback for real-time candle updates."""
        if self.multi_asset_mode and self.asset_manager:
            if symbol in self.multi_assets:
                self.asset_manager.mark_candle_update_pending(symbol)
                if symbol in self.strategies:
                    strategy = self.strategies[symbol]
                    if hasattr(strategy, 'invalidate_indicator_cache'):
                        strategy.invalidate_indicator_cache()
        else:
            if symbol == self.symbol:
                self._candle_update_pending = True
                if hasattr(self.strategy, 'invalidate_indicator_cache'):
                    self.strategy.invalidate_indicator_cache()
                if self.indicator_calc:
                    self.indicator_calc.invalidate_cache()
    
    def _on_account_update(self, data: Dict[str, Any]):
        """Callback for account updates."""
        try:
            asyncio.create_task(self.update_account_state())
        except Exception as e:
            logger.error(f"Error in account update callback: {e}")
    
    def _on_fill(self, fill: Dict[str, Any]):
        """Callback for fill updates."""
        try:
            symbol = fill.get('market', '')
            side = fill.get('side', '')
            size = fill.get('amount', 0)
            price = fill.get('price', 0)
            realized_pnl = fill.get('realizedPnl', 0)
            
            pnl_str = f" | P&L: ${realized_pnl:+.2f}" if realized_pnl else ""
            logger.info(f"📥 FILL: {symbol} {side.upper()} {size} @ ${price}{pnl_str}")
            
            if self.telegram_bot and realized_pnl:
                emoji = "🟢" if side.lower() == "buy" else "🔴"
                pnl_emoji = "✅" if realized_pnl > 0 else "❌"
                message = f"{emoji} **FILL**\n\n{symbol} {side.upper()} {size} @ ${price}\n{pnl_emoji} P&L: ${realized_pnl:+.2f}"
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._send_notification(message))
                except RuntimeError:
                    pass
                    
        except Exception as e:
            logger.error(f"Error in fill callback: {e}")
    
    async def _send_notification(self, message: str):
        """Send Telegram notification."""
        try:
            if self.telegram_bot:
                await self.telegram_bot.send_message(message)
        except Exception as e:
            logger.debug(f"Notification failed: {e}")
    
    async def run_trading_loop(self):
        """Main trading loop."""
        logger.info("🚀 Starting trading loop...")
        self.is_running = True
        self.start_time = datetime.now(timezone.utc)
        
        # Load any previously active trades
        self._load_active_trades()
        
        # Send startup notification
        if self.telegram_bot:
            await self._send_notification(
                f"🤖 **ExtendBot Started**\n\n"
                f"Mode: {self.mode}\n"
                f"Symbol: {self.symbol if not self.multi_asset_mode else ', '.join(self.multi_assets)}\n"
                f"Balance: ${self.account_value:.2f}"
            )
        
        loop_count = 0
        
        while self.is_running and not shutdown_event.is_set():
            try:
                loop_count += 1
                
                # Check if paused
                if self.is_paused:
                    await asyncio.sleep(1)
                    continue
                
                # Update account state periodically
                if loop_count % 10 == 0:
                    await self.update_account_state()
                
                # Check risk limits
                if self.kill_switch and self.kill_switch.check():
                    logger.warning("🚨 Kill switch triggered!")
                    self.is_paused = True
                    if self.telegram_bot:
                        await self._send_notification("🚨 **KILL SWITCH TRIGGERED**\nTrading paused due to risk limits.")
                    continue
                
                # Get market data
                market_data = await self._get_market_data()
                if not market_data:
                    await asyncio.sleep(1)
                    continue
                
                # Generate signal
                signal = None
                if self.multi_asset_mode:
                    signal = await self._scan_multi_asset_signals(self.account_state)
                else:
                    if self.strategy:
                        signal = await self.strategy.generate_signal(market_data, self.account_state)
                
                # Execute signal
                if signal:
                    await self._execute_signal(signal)
                
                # Monitor positions
                await self._monitor_positions(self.account_state)
                
                # Sleep between iterations
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                if self.error_handler:
                    await self.error_handler.handle_error(e, "trading_loop")
                await asyncio.sleep(5)
        
        logger.info("📴 Trading loop stopped")
    
    async def _get_market_data(self) -> Optional[Dict[str, Any]]:
        """Get current market data."""
        try:
            price = await self.client.get_mid_price(self.symbol)
            if price <= 0:
                return None
            
            candles = await self.client.get_candles(self.symbol, self.timeframe, 150)
            
            return {
                'symbol': self.symbol,
                'price': price,
                'candles': candles,
                'timestamp': datetime.now(timezone.utc)
            }
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    async def _scan_multi_asset_signals(self, account_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Scan all enabled assets for trading signals."""
        if not self.asset_manager:
            return None
        
        self.asset_manager.update_from_account_state(account_state)
        
        if not self.asset_manager.can_open_new_position():
            return None
        
        available = self.asset_manager.get_assets_without_positions()
        if not available:
            return None
        
        for symbol in available:
            can_trade, reason = self.asset_manager.can_trade_asset(symbol)
            if not can_trade:
                continue
            
            strategy = self.strategies.get(symbol)
            if not strategy:
                continue
            
            price = await self.client.get_mid_price(symbol)
            if price <= 0:
                continue
            
            candles = await self.client.get_candles(symbol, self.timeframe, 150)
            if not candles:
                continue
            
            market_data = {
                'symbol': symbol,
                'price': price,
                'candles': candles,
                'timestamp': datetime.now(timezone.utc)
            }
            
            try:
                signal = await strategy.generate_signal(market_data, account_state)
                if signal:
                    self.asset_manager.record_signal(symbol)
                    logger.info(f"🌐 Multi-Asset Signal: {symbol}")
                    return signal
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return None
    
    async def _execute_signal(self, signal: Dict[str, Any]):
        """Execute a trading signal."""
        try:
            symbol = signal.get('symbol', self.symbol)
            direction = signal.get('direction')
            signal_type = signal.get('signal_type', 'UNKNOWN')
            
            if not direction:
                return
            
            is_buy = direction.lower() in ('long', 'buy')
            
            # Calculate position size
            position_size_pct = float(signal.get('position_size_pct', 25))
            
            if self.kelly:
                kelly_size = self.kelly.calculate_position_size(float(self.account_value))
                if kelly_size > 0:
                    position_size_pct = min(position_size_pct, kelly_size)
            
            # Apply risk engine limits
            if self.risk_engine:
                max_size = self.risk_engine.get_max_position_size(symbol)
                position_size_pct = min(position_size_pct, max_size)
            
            position_value = float(self.account_value) * (position_size_pct / 100)
            price = await self.client.get_mid_price(symbol)
            size = position_value / price
            
            # Get TP/SL from signal
            tp_price = signal.get('tp_price')
            sl_price = signal.get('sl_price')
            
            # Execute trade
            logger.info(f"📈 Executing {signal_type}: {symbol} {'LONG' if is_buy else 'SHORT'}")
            logger.info(f"   Size: {size:.4f} ({position_size_pct:.1f}% of ${self.account_value:.2f})")
            
            if self.is_paper_trading and self.paper_trading:
                # Paper trade
                result = self.paper_trading.execute_trade(
                    symbol=symbol,
                    side='buy' if is_buy else 'sell',
                    size=size,
                    price=price,
                    tp_price=tp_price,
                    sl_price=sl_price
                )
            else:
                # Real trade
                result = await self.order_manager.market_entry_with_tpsl(
                    symbol=symbol,
                    is_buy=is_buy,
                    size=size,
                    tp_price=tp_price,
                    sl_price=sl_price,
                )
            
            if result.get('status') == 'ok':
                self.trades_executed += 1
                
                # Track trade
                self._active_trade_ids[symbol] = {
                    'trade_id': result.get('order_id'),
                    'entry_price': Decimal(str(price)),
                    'quantity': Decimal(str(size)),
                    'side': 'long' if is_buy else 'short',
                    'entry_time': datetime.now(timezone.utc),
                    'tp_price': Decimal(str(tp_price)) if tp_price else None,
                    'sl_price': Decimal(str(sl_price)) if sl_price else None,
                }
                self._save_active_trades()
                
                # Log trade
                await self._log_trade(signal, result, price, size)
                
                # Send notification
                if self.telegram_bot:
                    emoji = "🟢" if is_buy else "🔴"
                    await self._send_notification(
                        f"{emoji} **{signal_type}**\n\n"
                        f"{symbol} {'LONG' if is_buy else 'SHORT'}\n"
                        f"Size: {size:.4f}\n"
                        f"Entry: ${price:.2f}\n"
                        f"TP: ${tp_price:.2f}\n"
                        f"SL: ${sl_price:.2f}"
                    )
                
                logger.info(f"✅ Trade executed successfully")
            else:
                logger.error(f"❌ Trade failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Error executing signal: {e}", exc_info=True)
    
    async def _log_trade(self, signal: Dict, result: Dict, price: float, size: float):
        """Log trade to file and database."""
        trade_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': signal.get('symbol', self.symbol),
            'direction': signal.get('direction'),
            'signal_type': signal.get('signal_type'),
            'strategy': signal.get('strategy'),
            'price': price,
            'size': size,
            'tp_price': signal.get('tp_price'),
            'sl_price': signal.get('sl_price'),
            'order_id': result.get('order_id'),
        }
        
        # Log to JSONL file
        log_file = self.trade_log_path / f"trades_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(trade_data) + '\n')
        
        # Log to database
        if self.db:
            try:
                await self.db.record_trade(trade_data)
            except Exception as e:
                logger.error(f"Failed to record trade to DB: {e}")
    
    async def _monitor_positions(self, account_state: Dict[str, Any]):
        """Monitor active positions."""
        try:
            now = datetime.now(timezone.utc)
            
            if self._last_position_check is not None:
                time_since_check = (now - self._last_position_check).total_seconds()
                if time_since_check < self._position_check_interval:
                    return
            
            self._last_position_check = now
            
            positions = await self.client.get_all_positions()
            current_symbols = {pos['symbol'] for pos in positions if float(pos.get('size', 0)) != 0}
            
            async with self._position_lock:
                for pos in positions:
                    size = float(pos.get('size', 0))
                    if size != 0:
                        symbol = pos['symbol']
                        self._position_details[symbol] = {
                            'entry_price': float(pos.get('entry_price', 0)),
                            'size': abs(size),
                            'side': 'long' if size > 0 else 'short',
                            'unrealized_pnl': float(pos.get('unrealized_pnl', 0))
                        }
                
                if hasattr(self, '_last_positions'):
                    closed_positions = self._last_positions - current_symbols
                    for symbol in closed_positions:
                        logger.info(f"🔄 Position closed: {symbol}")
                        self._clear_trade_state(symbol)
            
            self._last_positions = current_symbols
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("📴 Shutting down ExtendBot...")
        self.is_running = False
        
        try:
            if self.websocket:
                await self.websocket.stop()
            
            if self.telegram_bot:
                await self._send_notification("📴 **ExtendBot Stopped**")
                await self.telegram_bot.stop()
            
            if self.db:
                await self.db.disconnect()
            
            logger.info("✅ Shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


async def main():
    """Main entry point."""
    bot = ExtendBot()
    
    if not await bot.initialize():
        logger.error("Failed to initialize bot")
        return 1
    
    try:
        await bot.run_trading_loop()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await bot.shutdown()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
