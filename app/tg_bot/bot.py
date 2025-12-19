"""
Modern Telegram Bot for HyperBot
Clean architecture with handlers, formatters, and keyboards.

Features:
- Interactive inline keyboards
- Rich message formatting
- Paginated lists
- Conversation flows
- Rate limiting
- Scheduled updates
- Clean error handling
"""

import logging
import os
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta

from telegram import Update, InlineKeyboardMarkup
from telegram.ext import (
    Application, 
    CommandHandler, 
    CallbackQueryHandler, 
    ContextTypes,
)
from telegram.error import TelegramError

from .formatters import MessageFormatter
from .keyboards import KeyboardFactory

logger = logging.getLogger(__name__)


def mask_token(token: str) -> str:
    """Mask sensitive token for logging."""
    if not token or len(token) < 20:
        return "***"
    if ':' in token:
        parts = token.split(':')
        return f"{parts[0]}:{parts[1][:3]}...{parts[1][-4:]}"
    return f"{token[:10]}...{token[-4:]}"


class TelegramBot:
    """
    Modern Telegram Bot with clean architecture.
    
    Features:
    - Modular handlers
    - Rich formatting
    - Interactive keyboards
    - Conversation flows
    - Rate limiting
    - Graceful error handling
    """
    
    # Conversation states
    SET_TP_PRICE = 1
    SET_SL_PRICE = 2
    
    def __init__(self, bot_instance, config: Dict[str, Any] = None):
        """
        Initialize Telegram bot.
        
        Args:
            bot_instance: Main trading bot instance
            config: Optional configuration
        """
        self.bot = bot_instance
        self.config = config or {}
        
        # Credentials
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.token or not self.chat_id:
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID required in .env")
        
        self.application: Optional[Application] = None
        self.is_running = False
        
        # Formatters
        self.fmt = MessageFormatter
        self.kb = KeyboardFactory
        
        # Session tracking
        self.session_start = datetime.now(timezone.utc)
        
        # Rate limiting
        self._last_message_time: Dict[str, datetime] = {}
        self._message_cooldown = timedelta(seconds=2)
        
        # Notification settings
        self.notify_signals = os.getenv('TG_NOTIFY_SIGNALS', 'true').lower() == 'true'
        self.notify_fills = os.getenv('TG_NOTIFY_FILLS', 'true').lower() == 'true'
        self.notify_pnl_warnings = os.getenv('TG_NOTIFY_PNL_WARNINGS', 'true').lower() == 'true'
        self.status_interval = int(os.getenv('TG_STATUS_INTERVAL', '3600'))
        
        # Background tasks
        self._status_task: Optional[asyncio.Task] = None
        
        logger.info(f"📱 Telegram Bot initialized")
        logger.info(f"   Token: {mask_token(self.token)}")
        logger.info(f"   Chat ID: {self.chat_id}")
    
    # ==================== LIFECYCLE ====================
    
    async def start(self):
        """Start the Telegram bot."""
        if self.is_running:
            return
        
        try:
            self.application = (
                Application.builder()
                .token(self.token)
                .read_timeout(30)
                .write_timeout(30)
                .connect_timeout(30)
                .build()
            )
            
            # Register command handlers
            self._register_handlers()
            
            # Start bot
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling(drop_pending_updates=True)
            
            self.is_running = True
            
            # Start scheduled updates
            if self.status_interval > 0:
                self._status_task = asyncio.create_task(self._scheduled_status())
            
            # Send startup message
            await self._send_startup_message()
            
            logger.info("✅ Telegram Bot started")
            
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
            raise
    
    async def stop(self):
        """Stop the Telegram bot gracefully."""
        if not self.is_running:
            return
        
        try:
            # Cancel background tasks
            if self._status_task and not self._status_task.done():
                self._status_task.cancel()
                try:
                    await self._status_task
                except asyncio.CancelledError:
                    pass
            
            # Send shutdown message
            try:
                await self.send_message("🛑 *BOT SHUTTING DOWN*\n\nGoodbye!")
            except Exception:
                pass
            
            # Stop application
            if self.application:
                if self.application.updater:
                    await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
            
            self.is_running = False
            logger.info("✅ Telegram Bot stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")
            self.is_running = False
    
    def _register_handlers(self):
        """Register all command and callback handlers."""
        app = self.application
        
        # Command handlers
        commands = [
            ("start", self._cmd_start),
            ("menu", self._cmd_start),
            ("help", self._cmd_help),
            ("status", self._cmd_dashboard),
            ("dashboard", self._cmd_dashboard),
            ("pos", self._cmd_positions),
            ("positions", self._cmd_positions),
            ("trades", self._cmd_trades),
            ("pnl", self._cmd_pnl),
            ("balance", self._cmd_balance),
            ("market", self._cmd_market),
            ("stats", self._cmd_stats),
            ("close", self._cmd_close),
            ("closeall", self._cmd_closeall),
            ("sl", self._cmd_set_sl),
            ("tp", self._cmd_set_tp),
            ("managed", self._cmd_managed),
            ("logs", self._cmd_logs),
            ("db", self._cmd_db_stats),
            ("kelly", self._cmd_kelly),
            ("regime", self._cmd_regime),
            ("assets", self._cmd_assets),
            ("alerts", self._cmd_alerts),
            ("config", self._cmd_config),
            ("signal", self._cmd_signal),
            ("analyze", self._cmd_signal),  # Alias
            # NEW: Quick trading commands
            ("buy", self._cmd_buy),
            ("sell", self._cmd_sell),
            ("long", self._cmd_buy),  # Alias
            ("short", self._cmd_sell),  # Alias
            # NEW: Account & reports
            ("tier", self._cmd_tier),
            ("account", self._cmd_tier),  # Alias
            ("report", self._cmd_report),
            ("performance", self._cmd_report),  # Alias
            # NEW: Risk commands
            ("risk", self._cmd_risk),
            ("killswitch", self._cmd_killswitch),
            ("ks", self._cmd_killswitch),  # Alias
        ]
        
        for cmd, handler in commands:
            app.add_handler(CommandHandler(cmd, handler))
        
        # Callback query handler
        app.add_handler(CallbackQueryHandler(self._handle_callback))
    
    # ==================== MESSAGE HELPERS ====================
    
    async def send_message(
        self, 
        text: str, 
        reply_markup: InlineKeyboardMarkup = None,
        parse_mode: str = 'Markdown',
        disable_preview: bool = True,
    ) -> bool:
        """Send message to configured chat."""
        if not self.application:
            return False
        
        try:
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_preview,
            )
            return True
        except TelegramError as e:
            if "parse entities" in str(e).lower() or "can't find end" in str(e).lower():
                # Markdown parsing failed - retry without parse_mode
                logger.warning(f"Markdown parse failed in send_message, retrying as plain text")
                try:
                    plain_text = text.replace('*', '').replace('_', '').replace('`', '')
                    await self.application.bot.send_message(
                        chat_id=self.chat_id,
                        text=plain_text,
                        reply_markup=reply_markup,
                        parse_mode=None,
                        disable_web_page_preview=disable_preview,
                    )
                    return True
                except Exception as retry_error:
                    logger.error(f"Plain text retry also failed: {retry_error}")
                    return False
            else:
                logger.error(f"Telegram send error: {e}")
                return False
    
    async def _edit_or_reply(
        self,
        update: Update,
        text: str,
        reply_markup: InlineKeyboardMarkup = None,
        parse_mode: str = 'Markdown',
    ):
        """Edit message if callback, otherwise reply."""
        try:
            if update.callback_query:
                await update.callback_query.edit_message_text(
                    text=text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode,
                )
            elif update.message:
                await update.message.reply_text(
                    text=text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode,
                )
        except TelegramError as e:
            # Ignore "message not modified" error - it's harmless
            if "not modified" in str(e).lower():
                pass  # Silent ignore
            elif "parse entities" in str(e).lower() or "can't find end" in str(e).lower():
                # Markdown parsing failed - retry without parse_mode
                logger.warning(f"Markdown parse failed, retrying as plain text: {e}")
                try:
                    # Remove markdown formatting chars for plain text
                    plain_text = text.replace('*', '').replace('_', '').replace('`', '')
                    if update.callback_query:
                        await update.callback_query.edit_message_text(
                            text=plain_text,
                            reply_markup=reply_markup,
                            parse_mode=None,
                        )
                    elif update.message:
                        await update.message.reply_text(
                            text=plain_text,
                            reply_markup=reply_markup,
                            parse_mode=None,
                        )
                except Exception as retry_error:
                    logger.error(f"Plain text retry also failed: {retry_error}")
            else:
                logger.error(f"Message error: {e}")
    
    def _can_send(self, msg_type: str) -> bool:
        """Rate limiting check."""
        now = datetime.now(timezone.utc)
        last = self._last_message_time.get(msg_type)
        
        if last and (now - last) < self._message_cooldown:
            return False
        
        self._last_message_time[msg_type] = now
        return True
    
    # ==================== COMMAND HANDLERS ====================
    
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command - show main menu."""
        message = (
            "🤖 *HYPERBOT CONTROL PANEL*\n\n"
            "Welcome to your trading bot dashboard!\n\n"
            "Use the buttons below or type commands.\n"
            "Type /help for full command list."
        )
        await self._edit_or_reply(update, message, self.kb.main_menu())
    
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        await self._edit_or_reply(update, self.fmt.format_help(), self.kb.back_to_menu())
    
    async def _cmd_dashboard(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status or /dashboard command."""
        try:
            # Gather dashboard data
            data = await self._get_dashboard_data()
            message = self.fmt.format_dashboard(data)
            await self._edit_or_reply(update, message, self.kb.dashboard_actions())
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            await self._edit_or_reply(
                update, 
                self.fmt.format_error("Dashboard Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command."""
        try:
            positions = await self._get_positions()
            message = self.fmt.format_positions_list(positions)
            
            if positions:
                keyboard = self.kb.positions_list(positions)
            else:
                keyboard = self.kb.empty_positions()
            
            await self._edit_or_reply(update, message, keyboard)
        except Exception as e:
            logger.error(f"Positions error: {e}")
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Positions Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command."""
        try:
            trades = await self._get_recent_trades()
            message = self.fmt.format_trades_list(trades)
            
            if trades:
                keyboard = self.kb.trades_list()
            else:
                keyboard = self.kb.empty_trades()
            
            await self._edit_or_reply(update, message, keyboard)
        except Exception as e:
            logger.error(f"Trades error: {e}")
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Trades Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pnl command."""
        try:
            data = await self._get_pnl_data()
            message = self.fmt.format_pnl_breakdown(data)
            await self._edit_or_reply(update, message, self.kb.quick_actions())
        except Exception as e:
            logger.error(f"PnL error: {e}")
            await self._edit_or_reply(
                update,
                self.fmt.format_error("P&L Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command - quick balance check."""
        try:
            account = float(self.bot.account_value)
            margin = float(self.bot.margin_used)
            available = account - margin
            
            message = (
                "💰 *QUICK BALANCE*\n\n"
                f"Balance:   {self.fmt.format_money(account)}\n"
                f"Margin:    {self.fmt.format_money(margin)}\n"
                f"Available: {self.fmt.format_money(available)}"
            )
            await self._edit_or_reply(update, message, self.kb.quick_actions())
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Balance Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_market(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /market command."""
        try:
            lines = [
                "📊 *MARKET OVERVIEW*",
                "",
            ]
            
            # Get symbols to show
            if hasattr(self.bot, 'trading_symbols'):
                symbols = self.bot.trading_symbols
            else:
                symbols = [getattr(self.bot, 'symbol', 'SOL')]
            
            for symbol in symbols:
                try:
                    # Get price from websocket or client
                    price = None
                    if hasattr(self.bot, 'websocket') and self.bot.websocket:
                        market_data = self.bot.websocket.get_market_data(symbol)
                        if market_data:
                            price = market_data.get('price')
                    
                    if not price and hasattr(self.bot, 'client'):
                        price = await self.bot.client.get_market_price(symbol)
                    
                    if price:
                        lines.append(f"💰 *{symbol}*: ${float(price):,.2f}")
                    else:
                        lines.append(f"💰 *{symbol}*: Price unavailable")
                except Exception as e:
                    lines.append(f"💰 *{symbol}*: Error - {str(e)[:20]}")
            
            # Get regime
            regime = "Unknown"
            if hasattr(self.bot, 'strategy') and hasattr(self.bot.strategy, 'strategies'):
                for name, strat in self.bot.strategy.strategies.items():
                    if hasattr(strat, 'regime_detector'):
                        r = strat.regime_detector.current_regime
                        regime = r.value if hasattr(r, 'value') else str(r)
                        break
            
            lines.extend([
                "",
                f"🌡️ Market Regime: {regime}",
                f"🕐 Time: {datetime.now(timezone.utc).strftime('%H:%M UTC')}",
            ])
            
            await self._edit_or_reply(update, "\n".join(lines), self.kb.quick_actions())
        except Exception as e:
            logger.error(f"Market error: {e}")
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Market Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command."""
        try:
            # Get strategy stats
            stats = {}
            if hasattr(self.bot, 'strategy') and self.bot.strategy:
                stats = self.bot.strategy.get_statistics() if hasattr(self.bot.strategy, 'get_statistics') else {}
            
            lines = [
                "📊 *PERFORMANCE STATISTICS*",
                "",
                "─────── Session ────────",
                f"Trades Executed: {getattr(self.bot, 'trades_executed', 0)}",
                f"Session P&L: {self.fmt.format_money(getattr(self.bot, 'session_pnl', 0), sign=True)}",
                f"Uptime: {self.fmt.format_uptime(self.session_start)}",
                "",
                "─────── Strategy ───────",
            ]
            
            # Add strategy breakdown
            breakdown = stats.get('strategy_breakdown', {})
            if breakdown:
                for name, data in breakdown.items():
                    sig_count = data.get('signals', 0)
                    trade_count = data.get('trades', 0)
                    lines.append(f"{name}: {sig_count} signals, {trade_count} trades")
            else:
                lines.append("No strategy stats available")
            
            lines.extend([
                "",
                "─────── Totals ────────",
                f"Total Signals: {stats.get('total_signals', 0)}",
                f"Executed: {stats.get('total_trades', 0)}",
                f"Rate: {self.fmt.format_percent(stats.get('execution_rate', 0) * 100)}",
            ])
            
            # Add kill switch stats if available
            if hasattr(self.bot, 'kill_switch') and self.bot.kill_switch:
                ks = self.bot.kill_switch
                lines.extend([
                    "",
                    "─────── Kill Switch ─────",
                    f"Consecutive Losses: {getattr(ks, 'consecutive_losses', 0)}",
                    f"Trading Allowed: {'✅' if getattr(ks, 'trading_allowed', True) else '❌'}",
                ])
            
            await self._edit_or_reply(update, "\n".join(lines), self.kb.quick_actions())
        except Exception as e:
            logger.error(f"Stats error: {e}", exc_info=True)
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Stats Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_close(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /close [symbol] command."""
        try:
            if not context.args:
                await self._edit_or_reply(
                    update,
                    "Usage: /close SYMBOL\nExample: /close SOL",
                    self.kb.back_to_menu()
                )
                return
            
            symbol = context.args[0].upper()
            message = f"⚠️ *CLOSE POSITION*\n\nAre you sure you want to close {symbol}?"
            await self._edit_or_reply(update, message, self.kb.close_confirm(symbol))
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Close Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_closeall(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /closeall command."""
        try:
            positions = await self._get_positions()
            
            if not positions:
                await self._edit_or_reply(update, "📭 No positions to close", self.kb.back_to_menu())
                return
            
            total_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)
            
            message = (
                f"⚠️ *CLOSE ALL POSITIONS*\n\n"
                f"This will close {len(positions)} position(s)\n"
                f"Current P&L: {self.fmt.format_money(total_pnl, sign=True)}\n\n"
                f"Are you absolutely sure?"
            )
            await self._edit_or_reply(update, message, self.kb.closeall_confirm())
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Close All Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_set_sl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /sl [symbol] [price] command."""
        try:
            if len(context.args) < 2:
                await self._edit_or_reply(
                    update,
                    "Usage: /sl SYMBOL PRICE\nExample: /sl SOL 145.50",
                    self.kb.back_to_menu()
                )
                return
            
            symbol = context.args[0].upper()
            price = float(context.args[1])
            
            result = self.bot.order_manager.set_position_tpsl(symbol=symbol, sl_price=price)
            
            if result.get('status') == 'ok':
                await self._edit_or_reply(
                    update,
                    f"✅ *STOP LOSS SET*\n\n{symbol}: ${price:.4f}",
                    self.kb.back_to_menu()
                )
            else:
                await self._edit_or_reply(
                    update,
                    self.fmt.format_error("Set SL Failed", str(result)),
                    self.kb.back_to_menu()
                )
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Set SL Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_set_tp(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /tp [symbol] [price] command."""
        try:
            if len(context.args) < 2:
                await self._edit_or_reply(
                    update,
                    "Usage: /tp SYMBOL PRICE\nExample: /tp SOL 165.00",
                    self.kb.back_to_menu()
                )
                return
            
            symbol = context.args[0].upper()
            price = float(context.args[1])
            
            result = self.bot.order_manager.set_position_tpsl(symbol=symbol, tp_price=price)
            
            if result.get('status') == 'ok':
                await self._edit_or_reply(
                    update,
                    f"✅ *TAKE PROFIT SET*\n\n{symbol}: ${price:.4f}",
                    self.kb.back_to_menu()
                )
            else:
                await self._edit_or_reply(
                    update,
                    self.fmt.format_error("Set TP Failed", str(result)),
                    self.kb.back_to_menu()
                )
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Set TP Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_managed(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /managed command."""
        try:
            if not hasattr(self.bot, 'position_manager') or not self.bot.position_manager:
                await self._edit_or_reply(update, "⚠️ Position Manager not initialized", self.kb.back_to_menu())
                return
            
            pm = self.bot.position_manager
            managed = pm.managed_positions
            
            if not managed:
                await self._edit_or_reply(
                    update,
                    "📊 *MANAGED POSITIONS*\n\nNo positions currently managed.",
                    self.kb.back_to_menu()
                )
                return
            
            lines = ["📊 *MANAGED POSITIONS*\n"]
            
            for symbol, pos in managed.items():
                source = "👤" if pos.is_manual else "🤖"
                side_emoji = "🟢" if pos.side.lower() == 'long' else "🔴"
                
                lines.append(f"{side_emoji} *{symbol}* {source}")
                lines.append(f"   Entry: ${pos.entry_price:.4f}")
                lines.append(f"   Size:  {pos.size:.4f}")
                
                # Show SL/TP prices if set
                sl_text = f"${pos.sl_price:.2f}" if pos.sl_price else "Not Set"
                tp_text = f"${pos.tp_price:.2f}" if pos.tp_price else "Not Set"
                sl_icon = "✅" if pos.sl_price else "⚠️"
                tp_icon = "✅" if pos.tp_price else "⚠️"
                
                lines.append(f"   SL: {sl_icon} {sl_text}")
                lines.append(f"   TP: {tp_icon} {tp_text}")
                
                # Show P&L if available
                if hasattr(pos, 'unrealized_pnl_pct') and pos.unrealized_pnl_pct:
                    pnl_emoji = "🟢" if pos.unrealized_pnl_pct > 0 else "🔴"
                    lines.append(f"   P&L: {pnl_emoji} {pos.unrealized_pnl_pct:+.2f}%")
                lines.append("")
            
            await self._edit_or_reply(update, "\n".join(lines), self.kb.back_to_menu())
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Managed Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /logs command."""
        try:
            from pathlib import Path
            import subprocess
            
            log_date = datetime.now().strftime('%Y%m%d')
            
            # Check multiple possible log locations
            log_paths = [
                Path(f"/root/hyperbot/logs/bot_{log_date}.log"),
                Path("/root/hyperbot/logs/bot.log"),
                Path(f"/home/hyperbot/logs/bot_{log_date}.log"),
                Path(__file__).parent.parent.parent / f"logs/bot_{log_date}.log",
                Path(__file__).parent.parent.parent / "logs/bot.log",
                Path("/workspaces/hyperbot/logs/bot.log"),
            ]
            
            log_lines = []
            log_source = "unknown"
            
            # Try to find log file
            for path in log_paths:
                try:
                    if path.exists():
                        with open(path, 'r', errors='ignore') as f:
                            log_lines = f.readlines()[-100:]
                        if log_lines:
                            log_source = str(path)
                            break
                except Exception:
                    continue
            
            # If no log file, try pm2 logs
            if not log_lines:
                try:
                    result = subprocess.run(
                        ['pm2', 'logs', 'hyperbot', '--lines', '30', '--nostream'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.stdout:
                        log_lines = result.stdout.strip().split('\n')
                        log_source = "pm2"
                except Exception:
                    pass
            
            if not log_lines:
                message = (
                    "📝 *RECENT LOGS*\n\n"
                    "📭 No logs available.\n\n"
                    "Use pm2 logs hyperbot on VPS."
                )
                await self._edit_or_reply(update, message, self.kb.logs_actions())
                return
            
            # Format logs - simplified approach
            formatted = []
            for line in log_lines[-30:]:
                line = line.strip()
                if not line:
                    continue
                
                # Simple emoji based on content
                if 'ERROR' in line or 'error' in line.lower():
                    emoji = "❌"
                elif 'WARNING' in line or 'warning' in line.lower():
                    emoji = "⚠️"
                elif 'Signal' in line or 'SIGNAL' in line:
                    emoji = "📡"
                elif 'Trade' in line or 'Order' in line:
                    emoji = "💹"
                else:
                    emoji = "📝"
                
                # Truncate line for display
                display_line = line[:80] if len(line) > 80 else line
                # Remove characters that break markdown
                display_line = display_line.replace('*', '').replace('_', '').replace('`', '')
                formatted.append(f"{emoji} {display_line}")
            
            if formatted:
                message = "📝 *RECENT LOGS*\n\n" + "\n".join(formatted[-15:])
            else:
                message = "📝 *RECENT LOGS*\n\n📭 No parseable logs."
            
            await self._edit_or_reply(update, message, self.kb.logs_actions())
            
        except Exception as e:
            logger.error(f"Logs error: {e}", exc_info=True)
            await self._edit_or_reply(
                update,
                f"❌ Logs Error: {str(e)[:100]}",
                self.kb.back_to_menu()
            )
    
    async def _cmd_db_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /db command."""
        try:
            if not self.bot.db:
                await self._edit_or_reply(update, "❌ Database not connected", self.kb.back_to_menu())
                return
            
            stats = await self.bot.db.get_total_stats()
            
            # Handle None/empty stats safely
            total_trades = stats.get('total_trades') or 0
            winning = stats.get('winning_trades') or 0
            losing = stats.get('losing_trades') or 0
            win_rate = stats.get('win_rate') or 0
            total_pnl = stats.get('total_pnl') or 0
            best = stats.get('best_trade') or 0
            worst = stats.get('worst_trade') or 0
            
            if total_trades == 0:
                lines = [
                    "📊 *DATABASE STATISTICS*",
                    "",
                    "📭 No closed trades recorded yet.",
                    "",
                    "_Trades will appear here once positions are closed._",
                ]
            else:
                lines = [
                    "📊 *DATABASE STATISTICS*",
                    "",
                    f"📈 Total Trades: {int(total_trades)}",
                    f"✅ Wins: {int(winning)}",
                    f"❌ Losses: {int(losing)}",
                    f"📊 Win Rate: {self.fmt.format_percent(float(win_rate))}",
                    "",
                    f"💰 Total P&L: {self.fmt.format_money(float(total_pnl), sign=True)}",
                    f"🏆 Best Trade: {self.fmt.format_money(float(best), sign=True)}",
                    f"💔 Worst Trade: {self.fmt.format_money(float(worst), sign=True)}",
                ]
            
            await self._edit_or_reply(update, "\n".join(lines), self.kb.back_to_menu())
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("DB Stats Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_kelly(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /kelly command."""
        try:
            kc = getattr(self.bot, 'kelly_calculator', None)
            
            if not kc:
                lines = [
                    "🎲 *KELLY CRITERION*",
                    "",
                    "⚠️ Kelly Calculator not initialized.",
                    "",
                    "_Requires 20+ closed trades in database._",
                ]
                await self._edit_or_reply(update, "\n".join(lines), self.kb.back_to_menu())
                return
            
            # Get Kelly stats
            recommended = getattr(kc, 'recommended_size', 0) or 0
            win_rate = getattr(kc, 'win_rate', 0) or 0
            avg_win = getattr(kc, 'avg_win', 0) or 0
            avg_loss = getattr(kc, 'avg_loss', 0) or 0
            trade_count = getattr(kc, 'trade_count', 0) or 0
            
            lines = [
                "🎲 *KELLY CRITERION*",
                "",
                f"🎯 Recommended Size: {self.fmt.format_percent(recommended * 100)}",
                f"📈 Win Rate: {self.fmt.format_percent(win_rate * 100)}",
                f"✅ Avg Win: {self.fmt.format_money(avg_win, sign=True)}",
                f"❌ Avg Loss: {self.fmt.format_money(abs(avg_loss))}",
                f"📊 Trades Analyzed: {trade_count}",
                "",
                f"_{('Active' if trade_count >= 20 else 'Need ' + str(20 - trade_count) + ' more trades')}_",
            ]
            
            await self._edit_or_reply(update, "\n".join(lines), self.kb.back_to_menu())
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Kelly Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_regime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /regime command."""
        try:
            regime = "Unknown"
            confidence = 0
            regimes_by_symbol = {}
            
            # Multi-asset mode: get regime for each symbol
            if hasattr(self.bot, 'multi_asset_strategies'):
                for symbol, strat in self.bot.multi_asset_strategies.items():
                    if hasattr(strat, 'regime_detector'):
                        r = strat.regime_detector.current_regime
                        regimes_by_symbol[symbol] = r.value if hasattr(r, 'value') else str(r)
            
            # Single symbol mode
            if hasattr(self.bot, 'strategy') and hasattr(self.bot.strategy, 'strategies'):
                for name, strat in self.bot.strategy.strategies.items():
                    if hasattr(strat, 'regime_detector'):
                        r = strat.regime_detector.current_regime
                        regime = r.value if hasattr(r, 'value') else str(r)
                        confidence = getattr(strat.regime_detector, 'regime_confidence', 0) or 0
                        break
            
            # Format regime with emoji
            regime_emoji = {
                'TRENDING_UP': '🟢 TRENDING UP',
                'TRENDING_DOWN': '🔴 TRENDING DOWN',
                'RANGING': '▫️ RANGING',
                'VOLATILE': '🌊 VOLATILE',
                'BREAKOUT': '🚀 BREAKOUT',
                'LOW_VOL': '💤 LOW VOLATILITY',
                'UNKNOWN': '❓ UNKNOWN',
            }.get(regime, f'❓ {regime}')
            
            lines = [
                "🌡️ *MARKET REGIME*",
                "",
                f"Current: {regime_emoji}",
                f"Confidence: {self.fmt.format_percent(confidence * 100)}" if confidence else "",
            ]
            
            # Show multi-asset regimes if available
            if regimes_by_symbol:
                lines.append("")
                lines.append("🌐 *By Symbol:*")
                for sym, reg in regimes_by_symbol.items():
                    lines.append(f"  {sym}: {reg}")
            
            await self._edit_or_reply(update, "\n".join(filter(None, lines)), self.kb.back_to_menu())
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Regime Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_assets(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /assets command for multi-asset mode."""
        try:
            if not hasattr(self.bot, 'trading_symbols'):
                symbols = [self.bot.symbol]
            else:
                symbols = self.bot.trading_symbols
            
            lines = [
                "🌐 *TRADING ASSETS*",
                "",
            ]
            
            for symbol in symbols:
                lines.append(f"• {symbol}")
            
            await self._edit_or_reply(update, "\n".join(lines), self.kb.back_to_menu())
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Assets Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_alerts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /alerts command - notification settings."""
        message = (
            "🔔 *NOTIFICATION SETTINGS*\n\n"
            f"Signal Alerts: {'✅' if self.notify_signals else '❌'}\n"
            f"Trade Fills: {'✅' if self.notify_fills else '❌'}\n"
            f"P&L Warnings: {'✅' if self.notify_pnl_warnings else '❌'}\n"
            f"Status Interval: {self.status_interval}s"
        )
        await self._edit_or_reply(
            update, 
            message, 
            self.kb.notification_settings(
                self.notify_signals,
                self.notify_fills,
                self.notify_pnl_warnings
            )
        )
    
    async def _cmd_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /config command - show current configuration."""
        try:
            lines = [
                "⚙️ *BOT CONFIGURATION*",
                "",
                "─────── Trading ───────",
                f"Mode: {'Paper' if getattr(self.bot, 'is_paper_trading', False) else 'Live'}",
                f"Network: {os.getenv('HYPERLIQUID_NETWORK', 'mainnet').upper()}",
                f"Leverage: {os.getenv('MAX_LEVERAGE', '5')}x",
                "",
                "─────── Position Sizing ───────",
                f"Position Size: {os.getenv('POSITION_SIZE_PCT', '25')}%",
                f"Max Positions: {os.getenv('MAX_POSITIONS', '3')}",
                "",
                "─────── Risk Management ───────",
                f"ATR SL Mult: {os.getenv('ATR_SL_MULTIPLIER', '2.0')}x",
                f"ATR TP Mult: {os.getenv('ATR_TP_MULTIPLIER', '3.5')}x",
                f"Max Drawdown: {os.getenv('MAX_DRAWDOWN_PCT', '15')}%",
                "",
                "─────── Assets ───────",
            ]
            
            # Get trading symbols
            if hasattr(self.bot, 'trading_symbols'):
                symbols = self.bot.trading_symbols
            else:
                symbols = [getattr(self.bot, 'symbol', 'SOL')]
            lines.append(f"Symbols: {', '.join(symbols)}")
            
            # Get strategy info
            lines.extend([
                "",
                "─────── Strategy ───────",
                f"Signal Threshold: {os.getenv('MIN_SIGNAL_SCORE', '15')}/25",
                f"Cooldown: {os.getenv('SIGNAL_COOLDOWN', '180')}s",
            ])
            
            await self._edit_or_reply(update, "\n".join(lines), self.kb.back_to_menu())
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Config Error", str(e)),
                self.kb.back_to_menu()
            )
    
    # ==================== NEW: QUICK TRADING COMMANDS ====================
    
    async def _cmd_buy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /buy [symbol] [size%] - Quick long position."""
        try:
            if not context.args:
                await self._edit_or_reply(
                    update,
                    "📈 *QUICK BUY (LONG)*\n\n"
                    "Usage: `/buy SYMBOL [size%]`\n\n"
                    "Examples:\n"
                    "• `/buy SOL` - Default size\n"
                    "• `/buy SOL 25` - 25% of balance\n\n"
                    "_This opens a LONG position_",
                    self.kb.back_to_menu()
                )
                return
            
            symbol = context.args[0].upper()
            size_pct = float(context.args[1]) if len(context.args) > 1 else None
            
            # Get current price
            price = await self.bot.client.get_market_price(symbol)
            
            # Confirm trade
            size_text = f"{size_pct}%" if size_pct else "Default"
            message = (
                f"📈 *CONFIRM LONG TRADE*\n\n"
                f"Symbol: {symbol}\n"
                f"Price: ${float(price):,.4f}\n"
                f"Size: {size_text}\n\n"
                f"⚠️ Are you sure?"
            )
            await self._edit_or_reply(update, message, self.kb.trade_confirm(symbol, 'buy', size_pct))
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Buy Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_sell(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /sell [symbol] [size%] - Quick short position."""
        try:
            if not context.args:
                await self._edit_or_reply(
                    update,
                    "📉 *QUICK SELL (SHORT)*\n\n"
                    "Usage: `/sell SYMBOL [size%]`\n\n"
                    "Examples:\n"
                    "• `/sell SOL` - Default size\n"
                    "• `/sell SOL 25` - 25% of balance\n\n"
                    "_This opens a SHORT position_",
                    self.kb.back_to_menu()
                )
                return
            
            symbol = context.args[0].upper()
            size_pct = float(context.args[1]) if len(context.args) > 1 else None
            
            # Get current price
            price = await self.bot.client.get_market_price(symbol)
            
            # Confirm trade
            size_text = f"{size_pct}%" if size_pct else "Default"
            message = (
                f"📉 *CONFIRM SHORT TRADE*\n\n"
                f"Symbol: {symbol}\n"
                f"Price: ${float(price):,.4f}\n"
                f"Size: {size_text}\n\n"
                f"⚠️ Are you sure?"
            )
            await self._edit_or_reply(update, message, self.kb.trade_confirm(symbol, 'sell', size_pct))
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Sell Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_tier(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /tier command - Show account tier and settings."""
        try:
            balance = float(self.bot.account_value)
            
            # Determine tier
            if balance < 50:
                tier = "🔴 MICRO"
                tier_emoji = "🐜"
                leverage = "10x"
                max_pos = 1
                signal_req = "18/25"
                advice = "Focus on quality over quantity. Wait for A+ setups only."
            elif balance < 100:
                tier = "🟠 SMALL"
                tier_emoji = "🐰"
                leverage = "7x"
                max_pos = 1
                signal_req = "16/25"
                advice = "Build consistency. One trade at a time."
            elif balance < 500:
                tier = "🟡 STARTER"
                tier_emoji = "🦊"
                leverage = "5x"
                max_pos = 2
                signal_req = "15/25"
                advice = "Diversify slightly. Good risk management."
            else:
                tier = "🟢 NORMAL"
                tier_emoji = "🦁"
                leverage = "5x"
                max_pos = 3
                signal_req = "12/25"
                advice = "Full strategy enabled. Stay disciplined."
            
            lines = [
                f"{tier_emoji} *ACCOUNT TIER: {tier}*",
                "",
                "═══════ STATUS ═══════",
                f"💰 Balance: ${balance:,.2f}",
                f"📊 Leverage: {leverage}",
                f"📍 Max Positions: {max_pos}",
                f"🎯 Signal Threshold: {signal_req}",
                "",
                "═══════ TIER BRACKETS ═══════",
                f"{'→' if balance < 50 else '•'} Micro: < $50",
                f"{'→' if 50 <= balance < 100 else '•'} Small: $50 - $100",
                f"{'→' if 100 <= balance < 500 else '•'} Starter: $100 - $500",
                f"{'→' if balance >= 500 else '•'} Normal: > $500",
                "",
                "═══════ ADVICE ═══════",
                f"💡 _{advice}_",
            ]
            
            await self._edit_or_reply(update, "\n".join(lines), self.kb.back_to_menu())
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Tier Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /report command - Performance report."""
        try:
            if not self.bot.db:
                await self._edit_or_reply(update, "❌ Database not connected", self.kb.back_to_menu())
                return
            
            # Get stats from database
            stats = await self.bot.db.get_total_stats()
            pnl_data = await self._get_pnl_data()
            
            total_trades = stats.get('total_trades') or 0
            win_rate = (stats.get('win_rate') or 0)
            total_pnl = stats.get('total_pnl') or 0
            
            # Calculate averages
            avg_win = stats.get('avg_win') or 0
            avg_loss = stats.get('avg_loss') or 0
            
            # Profit factor
            gross_profit = stats.get('gross_profit') or 0
            gross_loss = abs(stats.get('gross_loss') or 1)
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            lines = [
                "📊 *PERFORMANCE REPORT*",
                "",
                "═══════ OVERALL ═══════",
                f"📈 Total Trades: {int(total_trades)}",
                f"🎯 Win Rate: {self.fmt.format_percent(float(win_rate))}",
                f"💰 Total P&L: {self.fmt.format_money(float(total_pnl), sign=True)}",
                f"📊 Profit Factor: {profit_factor:.2f}",
                "",
                "═══════ AVERAGES ═══════",
                f"✅ Avg Win: {self.fmt.format_money(float(avg_win), sign=True)}",
                f"❌ Avg Loss: {self.fmt.format_money(float(avg_loss), sign=True)}",
                f"📊 Expectancy: {self.fmt.format_money((float(win_rate)/100 * float(avg_win)) + ((100-float(win_rate))/100 * float(avg_loss)), sign=True)}/trade",
                "",
                "═══════ PERIODS ═══════",
                f"📅 Today: {self.fmt.format_money(pnl_data.get('today_pnl', 0), sign=True)}",
                f"📆 This Week: {self.fmt.format_money(pnl_data.get('weekly_pnl', 0), sign=True)}",
                f"📆 This Month: {self.fmt.format_money(pnl_data.get('monthly_pnl', 0), sign=True)}",
                "",
                "═══════ SESSION ═══════",
                f"⏰ Uptime: {self.fmt.format_uptime(self.session_start)}",
                f"💹 Session P&L: {self.fmt.format_money(pnl_data.get('session_pnl', 0), sign=True)}",
            ]
            
            # Rating
            if total_trades >= 20:
                if win_rate >= 65 and profit_factor >= 1.5:
                    rating = "⭐⭐⭐⭐⭐ EXCELLENT"
                elif win_rate >= 55 and profit_factor >= 1.2:
                    rating = "⭐⭐⭐⭐ GOOD"
                elif win_rate >= 45 and profit_factor >= 1.0:
                    rating = "⭐⭐⭐ AVERAGE"
                else:
                    rating = "⭐⭐ NEEDS WORK"
                lines.extend(["", f"📊 Rating: {rating}"])
            else:
                lines.extend(["", f"_Need {20 - int(total_trades)} more trades for rating_"])
            
            await self._edit_or_reply(update, "\n".join(lines), self.kb.back_to_menu())
        except Exception as e:
            logger.error(f"Report error: {e}", exc_info=True)
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Report Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /risk command - Show risk status."""
        try:
            balance = float(self.bot.account_value)
            margin = float(self.bot.margin_used)
            
            # Get drawdown info
            peak = float(getattr(self.bot, 'peak_equity', balance))
            drawdown = ((peak - balance) / peak * 100) if peak > 0 else 0
            
            # Get kill switch status
            ks = getattr(self.bot, 'kill_switch', None)
            ks_active = not getattr(ks, 'trading_allowed', True) if ks else False
            consec_losses = getattr(ks, 'consecutive_losses', 0) if ks else 0
            
            # Margin usage
            margin_pct = (margin / balance * 100) if balance > 0 else 0
            
            # Risk level
            if drawdown > 10 or margin_pct > 70 or ks_active:
                risk_level = "🔴 HIGH RISK"
            elif drawdown > 5 or margin_pct > 50:
                risk_level = "🟠 ELEVATED"
            else:
                risk_level = "🟢 NORMAL"
            
            lines = [
                f"🛡️ *RISK STATUS: {risk_level}*",
                "",
                "═══════ DRAWDOWN ═══════",
                f"💰 Current: ${balance:,.2f}",
                f"📈 Peak: ${peak:,.2f}",
                f"📉 Drawdown: {self.fmt.format_percent(drawdown)}",
                f"⚠️ Max Allowed: {os.getenv('MAX_DRAWDOWN_PCT', '10')}%",
                "",
                "═══════ MARGIN ═══════",
                f"📊 Used: {self.fmt.format_money(margin)}",
                f"📊 Usage: {self.fmt.format_percent(margin_pct)}",
                "",
                "═══════ KILL SWITCH ═══════",
                f"🔴 Status: {'TRIGGERED' if ks_active else '✅ Normal'}",
                f"📉 Consecutive Losses: {consec_losses}",
                f"⚠️ Trigger at: 3 losses",
            ]
            
            await self._edit_or_reply(update, "\n".join(lines), self.kb.back_to_menu())
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Risk Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_killswitch(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /killswitch command - Emergency stop/resume."""
        try:
            ks = getattr(self.bot, 'kill_switch', None)
            
            if not ks:
                await self._edit_or_reply(
                    update,
                    "⚠️ Kill Switch not initialized",
                    self.kb.back_to_menu()
                )
                return
            
            current_status = getattr(ks, 'trading_allowed', True)
            
            message = (
                f"🔴 *KILL SWITCH*\n\n"
                f"Current Status: {'✅ Trading Allowed' if current_status else '🛑 STOPPED'}\n\n"
                f"What would you like to do?"
            )
            await self._edit_or_reply(update, message, self.kb.killswitch_actions(current_status))
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Kill Switch Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _cmd_signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /signal [symbol] command - Full market analysis on demand.
        
        Usage: /signal SOL or /signal BTC
        Returns comprehensive analysis including:
        - Current price and 24h change
        - Technical indicators (RSI, MACD, EMAs, ADX, ATR)
        - Market regime detection
        - Signal score (long/short)
        - Recommended entry, SL, TP levels
        - Smart Money Concepts analysis
        - Order flow bias
        """
        try:
            # Get symbol from args or use default
            if context.args:
                symbol = context.args[0].upper()
            else:
                # Show usage
                symbols = getattr(self.bot, 'trading_symbols', [self.bot.symbol])
                await self._edit_or_reply(
                    update,
                    f"📊 *SIGNAL ANALYSIS*\n\n"
                    f"Usage: `/signal SYMBOL`\n\n"
                    f"Configured: {', '.join(symbols)}\n"
                    f"(Any symbol can be analyzed)\n\n"
                    f"Example: `/signal SOL`",
                    self.kb.back_to_menu()
                )
                return
            
            # Send "analyzing" message
            msg = await update.effective_message.reply_text(
                f"🔍 Analyzing *{symbol}*...\n\n_Fetching candles from API..._",
                parse_mode='Markdown'
            )
            
            # STEP 1: Fetch candles directly from API (works for any symbol)
            candles = []
            current_price = 0
            fetch_error = None
            
            try:
                if hasattr(self.bot, 'client') and self.bot.client:
                    candles = self.bot.client.get_candles(symbol, interval='1m', limit=200)
                    logger.info(f"Fetched {len(candles) if candles else 0} candles for {symbol}")
            except Exception as e:
                fetch_error = str(e)
                logger.warning(f"Failed to fetch candles for {symbol}: {e}")
            
            if not candles or len(candles) < 50:
                error_detail = f"\nError: {fetch_error}" if fetch_error else ""
                await msg.edit_text(
                    f"❌ Could not fetch market data for {symbol}\n\n"
                    f"Symbol may not exist on HyperLiquid.{error_detail}\n"
                    f"Try: SOL, BTC, ETH, etc.",
                    parse_mode='Markdown'
                )
                return
            
            # Get current price from candles
            current_price = float(candles[-1].get('close', candles[-1].get('c', 0)))
            
            if len(candles) < 100:
                await msg.edit_text(
                    f"⏳ Need more data for {symbol}\n\n"
                    f"Have {len(candles)} candles, need 100+\n"
                    f"_Try again in a few minutes._",
                    parse_mode='Markdown'
                )
                return
            
            # STEP 2: Get strategy (prefer symbol-specific, fallback to any swing strategy)
            strategy = None
            if hasattr(self.bot, 'multi_asset_strategies') and symbol in self.bot.multi_asset_strategies:
                strategy = self.bot.multi_asset_strategies[symbol]
            elif hasattr(self.bot, 'strategy') and hasattr(self.bot.strategy, 'strategies'):
                # Get first swing strategy
                for name, strat in self.bot.strategy.strategies.items():
                    if 'swing' in name.lower():
                        strategy = strat
                        break
            
            if not strategy:
                # No strategy available - show basic price info
                await msg.edit_text(
                    f"📊 *{symbol} QUICK VIEW*\n\n"
                    f"💰 Price: ${current_price:,.4f}\n"
                    f"📈 Candles: {len(candles)}\n\n"
                    f"_No strategy available for full analysis._\n"
                    f"_Bot only has strategy for configured symbols._",
                    parse_mode='Markdown'
                )
                return
            
            # Calculate indicators
            indicators = strategy._calculate_indicators(candles) if hasattr(strategy, '_calculate_indicators') else {}
            
            # Get regime
            regime = "Unknown"
            regime_confidence = 0
            if hasattr(strategy, 'regime_detector'):
                r, conf, params = strategy.regime_detector.detect_regime(
                    candles,
                    adx=indicators.get('adx'),
                    atr=indicators.get('atr'),
                    bb_bandwidth=indicators.get('bb_bandwidth'),
                )
                regime = r.value if hasattr(r, 'value') else str(r)
                regime_confidence = conf
            
            # Calculate signal scores (using enhanced score with penalties)
            long_score = 0
            short_score = 0
            long_details = {}
            short_details = {}
            
            if hasattr(strategy, '_calculate_signal_score') and hasattr(strategy, '_calculate_enhanced_score'):
                from decimal import Decimal
                price = Decimal(str(current_price))
                
                # Get SMC analysis
                smc = {}
                if hasattr(strategy, 'smc_analyzer'):
                    smc = strategy.smc_analyzer.analyze(candles) or {}
                
                # Get order flow
                of = {}
                if hasattr(strategy, 'order_flow'):
                    of = strategy.order_flow.analyze_from_candles(candles) or {}
                
                # Calculate base scores first
                long_base = strategy._calculate_signal_score(
                    direction='long',
                    indicators=indicators,
                    regime=strategy.regime_detector.current_regime if hasattr(strategy, 'regime_detector') else None,
                    regime_params={},
                    smc_analysis=smc,
                    of_analysis=of,
                    current_price=price,
                )
                short_base = strategy._calculate_signal_score(
                    direction='short',
                    indicators=indicators,
                    regime=strategy.regime_detector.current_regime if hasattr(strategy, 'regime_detector') else None,
                    regime_params={},
                    smc_analysis=smc,
                    of_analysis=of,
                    current_price=price,
                )
                
                # Now calculate enhanced scores with penalties
                long_score, long_details = strategy._calculate_enhanced_score('long', candles, indicators, long_base)
                short_score, short_details = strategy._calculate_enhanced_score('short', candles, indicators, short_base)
            elif hasattr(strategy, '_calculate_signal_score'):
                # Fallback to base score only
                from decimal import Decimal
                price = Decimal(str(current_price))
                smc = strategy.smc_analyzer.analyze(candles) if hasattr(strategy, 'smc_analyzer') else {}
                of = strategy.order_flow.analyze_from_candles(candles) if hasattr(strategy, 'order_flow') else {}
                
                long_score = strategy._calculate_signal_score(
                    direction='long', indicators=indicators,
                    regime=strategy.regime_detector.current_regime if hasattr(strategy, 'regime_detector') else None,
                    regime_params={}, smc_analysis=smc or {}, of_analysis=of or {}, current_price=price,
                )
                short_score = strategy._calculate_signal_score(
                    direction='short', indicators=indicators,
                    regime=strategy.regime_detector.current_regime if hasattr(strategy, 'regime_detector') else None,
                    regime_params={}, smc_analysis=smc or {}, of_analysis=of or {}, current_price=price,
                )
            
            # Calculate TP/SL levels
            atr = indicators.get('atr', 0)
            if atr and current_price:
                from decimal import Decimal
                atr_val = Decimal(str(atr)) if not isinstance(atr, Decimal) else atr
                price_val = Decimal(str(current_price))
                
                sl_mult = Decimal(os.getenv('ATR_SL_MULTIPLIER', '2.0'))
                tp_mult = Decimal(os.getenv('ATR_TP_MULTIPLIER', '3.5'))
                
                long_sl = float(price_val - (atr_val * sl_mult))
                long_tp = float(price_val + (atr_val * tp_mult))
                short_sl = float(price_val + (atr_val * sl_mult))
                short_tp = float(price_val - (atr_val * tp_mult))
            else:
                long_sl = long_tp = short_sl = short_tp = 0
            
            # Format regime emoji
            regime_emoji = {
                'TRENDING_UP': '🟢 TRENDING UP',
                'TRENDING_DOWN': '🔴 TRENDING DOWN',
                'RANGING': '◻️ RANGING',
                'VOLATILE': '🌊 VOLATILE',
                'BREAKOUT': '🚀 BREAKOUT',
                'LOW_VOL': '💤 LOW VOL',
                'UNKNOWN': '❓ UNKNOWN',
            }.get(regime, f'❓ {regime}')
            
            # Determine signal recommendation
            threshold = int(os.getenv('MIN_SIGNAL_SCORE', '15'))
            max_score = 25  # Full theoretical max with all indicators
            if long_score >= threshold and long_score > short_score:
                signal_rec = f"🟢 *LONG* (Score: {long_score}/{max_score})"
                rec_entry = current_price
                rec_sl = long_sl
                rec_tp = long_tp
            elif short_score >= threshold and short_score > long_score:
                signal_rec = f"🔴 *SHORT* (Score: {short_score}/{max_score})"
                rec_entry = current_price
                rec_sl = short_sl
                rec_tp = short_tp
            else:
                signal_rec = f"⏸️ *NO SIGNAL* (L:{long_score} S:{short_score} need {threshold}+)"
                rec_entry = rec_sl = rec_tp = 0
            
            # Build comprehensive message
            lines = [
                f"📊 *{symbol} ANALYSIS*",
                "",
                "═══════ PRICE ═══════",
                f"💰 Current: ${float(current_price):,.4f}",
                f"📈 ATR: ${float(atr):,.4f}" if atr else "",
                "",
                "═══════ REGIME ═══════",
                f"{regime_emoji}",
                f"Confidence: {regime_confidence*100:.0f}%" if regime_confidence else "",
                "",
                "═══════ INDICATORS ═══════",
                f"📊 RSI: {float(indicators.get('rsi', 0)):.1f}" if indicators.get('rsi') else "",
                f"📈 ADX: {float(indicators.get('adx', 0)):.1f}" if indicators.get('adx') else "",
                f"📉 EMA9/21: ${float(indicators.get('ema_fast', 0)):,.2f} / ${float(indicators.get('ema_slow', 0)):,.2f}" if indicators.get('ema_fast') else "",
            ]
            
            # Add MACD
            macd = indicators.get('macd', {})
            if macd:
                macd_val = macd.get('macd', 0)
                signal_val = macd.get('signal', 0)
                hist = macd.get('histogram', 0)
                if macd_val:
                    lines.append(f"📊 MACD: {float(macd_val):.4f} (Sig: {float(signal_val):.4f})")
            
            lines.extend([
                "",
                "═══════ SIGNAL SCORES ═══════",
                f"🟢 Long:  {long_score}/{max_score} {'✅' if long_score >= threshold else ''}",
                f"🔴 Short: {short_score}/{max_score} {'✅' if short_score >= threshold else ''}",
            ])
            
            # Add penalty details if any
            long_penalties = long_details.get('penalties', []) if long_details else []
            short_penalties = short_details.get('penalties', []) if short_details else []
            if long_penalties or short_penalties:
                lines.append("")
                lines.append("═══════ PENALTIES ═══════")
                if long_penalties:
                    for p in long_penalties:
                        # Escape underscores in penalty types to avoid markdown issues
                        ptype = str(p.get('type', 'unknown')).replace('_', ' ')
                        lines.append(f"🟢 ⛔ {ptype}: {p.get('score', 0)}")
                if short_penalties:
                    for p in short_penalties:
                        ptype = str(p.get('type', 'unknown')).replace('_', ' ')
                        lines.append(f"🔴 ⛔ {ptype}: {p.get('score', 0)}")
            
            lines.extend([
                "",
                "═══════ RECOMMENDATION ═══════",
                signal_rec,
            ])
            
            if rec_entry:
                lines.extend([
                    "",
                    "═══════ LEVELS ═══════",
                    f"📍 Entry: ${rec_entry:,.4f}",
                    f"🛑 SL: ${rec_sl:,.4f} ({abs((rec_sl-rec_entry)/rec_entry*100):.2f}%)",
                    f"🎯 TP: ${rec_tp:,.4f} ({abs((rec_tp-rec_entry)/rec_entry*100):.2f}%)",
                    f"📊 R:R: 1:{abs((rec_tp-rec_entry)/(rec_entry-rec_sl)):.1f}" if rec_sl != rec_entry else "",
                ])
            
            lines.extend([
                "",
                f"Analysis at {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}",
            ])
            
            message_text = "\n".join(filter(None, lines))
            
            try:
                await msg.edit_text(message_text, parse_mode='Markdown')
            except Exception as md_error:
                # Fallback to plain text if markdown fails
                logger.warning(f"Markdown parse failed, using plain text: {md_error}")
                # Remove markdown formatting
                plain_text = message_text.replace('*', '').replace('_', '')
                await msg.edit_text(plain_text)
            
        except Exception as e:
            logger.error(f"Signal analysis error: {e}", exc_info=True)
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Signal Analysis Error", str(e)),
                self.kb.back_to_menu()
            )
    
    # ==================== CALLBACK HANDLER ====================
    
    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard button presses."""
        query = update.callback_query
        await query.answer()
        
        action = query.data
        
        # Route to appropriate handler
        handlers = {
            "main_menu": self._cmd_start,
            "dashboard": self._cmd_dashboard,
            "refresh_dashboard": self._cmd_dashboard,
            "refresh": self._cmd_dashboard,  # Generic refresh goes to dashboard
            "positions": self._cmd_positions,
            "refresh_positions": self._cmd_positions,
            "trades": self._cmd_trades,
            "refresh_trades": self._cmd_trades,
            "pnl": self._cmd_pnl,
            "refresh_pnl": self._cmd_pnl,
            "market": self._cmd_market,
            "refresh_market": self._cmd_market,
            "stats": self._cmd_stats,
            "refresh_stats": self._cmd_stats,
            "logs": self._cmd_logs,
            "refresh_logs": self._cmd_logs,
            "help": self._cmd_help,
            "settings": self._cmd_alerts,
            "settings_notifications": self._cmd_alerts,  # Settings sub-menu
            "settings_display": self._cmd_config,  # Display settings -> show config
            "noop": lambda u, c: None,
        }
        
        if action in handlers:
            await handlers[action](update, context)
        elif action == "bot_start":
            await self._handle_bot_start(update)
        elif action == "bot_pause":
            await self._handle_bot_pause(update)
        elif action.startswith("close_confirm_"):
            symbol = action.replace("close_confirm_", "")
            await self._handle_close_confirm(update, symbol)
        elif action.startswith("close_execute_"):
            symbol = action.replace("close_execute_", "")
            await self._handle_close_execute(update, symbol)
        elif action == "closeall_confirm":
            await self._cmd_closeall(update, context)
        elif action == "closeall_execute":
            await self._handle_closeall_execute(update)
        elif action.startswith("pos_detail_"):
            symbol = action.replace("pos_detail_", "")
            await self._handle_position_detail(update, symbol)
        elif action.startswith("set_tp_"):
            symbol = action.replace("set_tp_", "")
            await self._handle_set_tp_prompt(update, symbol)
        elif action.startswith("set_sl_"):
            symbol = action.replace("set_sl_", "")
            await self._handle_set_sl_prompt(update, symbol)
        elif action.startswith("toggle_"):
            await self._handle_toggle_setting(update, action)
        elif action.startswith("positions_page_"):
            page = int(action.replace("positions_page_", ""))
            await self._handle_positions_page(update, page)
        elif action.startswith("trades_page_"):
            # Trades pagination - just refresh trades for now
            await self._cmd_trades(update, context)
        elif action.startswith("auto_tpsl_"):
            symbol = action.replace("auto_tpsl_", "")
            await self._handle_auto_tpsl(update, symbol)
        elif action.startswith("set_tp_price_") or action.startswith("set_sl_price_"):
            # Handle price selection from keyboard (format: set_tp_price_SYMBOL_PRICE)
            await self._handle_price_selection(update, action)
        else:
            # Unknown callback - log it
            logger.warning(f"Unknown callback action: {action}")
            await self._edit_or_reply(
                update,
                f"⚠️ Unknown action: {action}",
                self.kb.back_to_menu()
            )
    
    async def _handle_bot_start(self, update: Update):
        """Handle bot start button."""
        if hasattr(self.bot, 'resume'):
            self.bot.resume()
        self.bot.is_paused = False
        
        await self._edit_or_reply(
            update,
            "🚀 *BOT STARTED*\n\nTrading resumed. Monitoring markets...",
            self.kb.bot_control(True, False)
        )
    
    async def _handle_bot_pause(self, update: Update):
        """Handle bot pause button."""
        if hasattr(self.bot, 'pause'):
            self.bot.pause()
        self.bot.is_paused = True
        
        await self._edit_or_reply(
            update,
            "⏸️ *BOT PAUSED*\n\nNo new trades will be opened.",
            self.kb.bot_control(True, True)
        )
    
    async def _handle_close_confirm(self, update: Update, symbol: str):
        """Show close confirmation."""
        message = f"⚠️ *CLOSE {symbol}?*\n\nThis action cannot be undone."
        await self._edit_or_reply(update, message, self.kb.close_confirm(symbol))
    
    async def _handle_close_execute(self, update: Update, symbol: str):
        """Execute position close."""
        try:
            await self._edit_or_reply(update, f"⏳ Closing {symbol}...", None)
            
            result = self.bot.order_manager.market_close(symbol)
            
            if result.get('status') == 'ok':
                await self._edit_or_reply(
                    update,
                    f"✅ *POSITION CLOSED*\n\n{symbol} has been closed.",
                    self.kb.back_to_menu()
                )
            else:
                await self._edit_or_reply(
                    update,
                    self.fmt.format_error("Close Failed", str(result)),
                    self.kb.back_to_menu()
                )
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Close Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _handle_closeall_execute(self, update: Update):
        """Execute close all positions."""
        try:
            positions = await self._get_positions()
            
            if not positions:
                await self._edit_or_reply(update, "📭 No positions to close", self.kb.back_to_menu())
                return
            
            await self._edit_or_reply(update, f"⏳ Closing {len(positions)} positions...", None)
            
            closed = 0
            for pos in positions:
                try:
                    result = self.bot.order_manager.market_close(pos['symbol'])
                    if result.get('status') == 'ok':
                        closed += 1
                except Exception as e:
                    logger.error(f"Error closing {pos['symbol']}: {e}")
            
            await self._edit_or_reply(
                update,
                f"✅ *ALL POSITIONS CLOSED*\n\nClosed {closed}/{len(positions)} positions.",
                self.kb.back_to_menu()
            )
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Close All Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _handle_position_detail(self, update: Update, symbol: str):
        """Show position detail with actions."""
        try:
            positions = await self._get_positions()
            pos = next((p for p in positions if p.get('symbol') == symbol), None)
            
            if not pos:
                await self._edit_or_reply(update, f"❌ Position {symbol} not found", self.kb.back_to_menu())
                return
            
            side_emoji = "🟢" if pos.get('side', '').lower() == 'long' else "🔴"
            pnl = pos.get('unrealized_pnl', 0)
            
            lines = [
                f"{side_emoji} *{symbol} POSITION*",
                "",
                f"Side:    {pos.get('side', 'N/A').upper()}",
                f"Size:    {self.fmt.format_number(abs(pos.get('size', 0)))}",
                f"Entry:   {self.fmt.format_money(pos.get('entry_price', 0), 4)}",
                f"Current: {self.fmt.format_money(pos.get('current_price', 0), 4)}",
                "",
                f"P&L: {self.fmt.pnl_emoji(pnl)} {self.fmt.format_money(pnl, sign=True)}",
                f"     ({self.fmt.format_percent(pos.get('unrealized_pnl_pct', 0), sign=True)})",
            ]
            
            await self._edit_or_reply(update, "\n".join(lines), self.kb.position_detail(symbol))
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Position Detail Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _handle_set_tp_prompt(self, update: Update, symbol: str):
        """Show TP price selection."""
        try:
            price = await self.bot.client.get_market_price(symbol)
            message = f"🎯 *SET TAKE PROFIT*\n\n{symbol}\nCurrent: {self.fmt.format_money(float(price), 4)}"
            await self._edit_or_reply(update, message, self.kb.price_input(symbol, 'tp', float(price)))
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Set TP Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _handle_set_sl_prompt(self, update: Update, symbol: str):
        """Show SL price selection."""
        try:
            price = await self.bot.client.get_market_price(symbol)
            message = f"🛑 *SET STOP LOSS*\n\n{symbol}\nCurrent: {self.fmt.format_money(float(price), 4)}"
            await self._edit_or_reply(update, message, self.kb.price_input(symbol, 'sl', float(price)))
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Set SL Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _handle_toggle_setting(self, update: Update, action: str):
        """Toggle notification setting."""
        if action == "toggle_signals":
            self.notify_signals = not self.notify_signals
        elif action == "toggle_fills":
            self.notify_fills = not self.notify_fills
        elif action == "toggle_pnl_warnings":
            self.notify_pnl_warnings = not self.notify_pnl_warnings
        
        await self._cmd_alerts(update, None)
    
    async def _handle_positions_page(self, update: Update, page: int):
        """Handle positions pagination."""
        try:
            positions = await self._get_positions()
            message = self.fmt.format_positions_list(positions, page=page)
            await self._edit_or_reply(update, message, self.kb.positions_list(positions, page=page))
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Positions Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _handle_auto_tpsl(self, update: Update, symbol: str):
        """Handle auto TP/SL setting based on ATR."""
        try:
            import os
            from decimal import Decimal
            
            # Get current position
            positions = await self._get_positions()
            pos = next((p for p in positions if p.get('symbol') == symbol), None)
            
            if not pos:
                await self._edit_or_reply(update, f"❌ No position found for {symbol}", self.kb.back_to_menu())
                return
            
            # Get ATR from strategy if available
            atr = None
            if hasattr(self.bot, 'multi_asset_strategies') and symbol in self.bot.multi_asset_strategies:
                strategy = self.bot.multi_asset_strategies[symbol]
                if hasattr(strategy, '_calculate_indicators'):
                    candles = []
                    if hasattr(self.bot, 'asset_manager'):
                        candles = self.bot.asset_manager.get_candles(symbol)
                    if candles:
                        indicators = strategy._calculate_indicators(candles)
                        atr = indicators.get('atr')
            
            if not atr:
                await self._edit_or_reply(
                    update, 
                    f"⚠️ Cannot calculate ATR for {symbol}. Use manual TP/SL.",
                    self.kb.position_detail(symbol)
                )
                return
            
            # Calculate TP/SL based on ATR
            entry = Decimal(str(pos.get('entry_price', 0)))
            atr_val = Decimal(str(atr))
            sl_mult = Decimal(os.getenv('ATR_SL_MULTIPLIER', '2.0'))
            tp_mult = Decimal(os.getenv('ATR_TP_MULTIPLIER', '3.5'))
            
            is_long = pos.get('side', '').lower() == 'long'
            if is_long:
                sl_price = float(entry - (atr_val * sl_mult))
                tp_price = float(entry + (atr_val * tp_mult))
            else:
                sl_price = float(entry + (atr_val * sl_mult))
                tp_price = float(entry - (atr_val * tp_mult))
            
            # Set TP/SL
            result = self.bot.order_manager.set_position_tpsl(symbol=symbol, sl_price=sl_price, tp_price=tp_price)
            
            if result.get('status') == 'ok':
                await self._edit_or_reply(
                    update,
                    f"✅ *AUTO TP/SL SET*\n\n{symbol}\n🛑 SL: ${sl_price:.4f}\n🎯 TP: ${tp_price:.4f}",
                    self.kb.back_to_menu()
                )
            else:
                await self._edit_or_reply(
                    update,
                    self.fmt.format_error("Auto TP/SL Failed", str(result)),
                    self.kb.back_to_menu()
                )
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Auto TP/SL Error", str(e)),
                self.kb.back_to_menu()
            )
    
    async def _handle_price_selection(self, update: Update, action: str):
        """Handle price selection from inline keyboard."""
        try:
            # Parse action: set_tp_price_SYMBOL_PRICE or set_sl_price_SYMBOL_PRICE
            parts = action.split('_')
            if len(parts) < 5:
                await self._edit_or_reply(update, "❌ Invalid price action", self.kb.back_to_menu())
                return
            
            action_type = parts[1]  # tp or sl
            symbol = parts[3]
            price = float(parts[4])
            
            if action_type == 'tp':
                result = self.bot.order_manager.set_position_tpsl(symbol=symbol, tp_price=price)
                label = "TAKE PROFIT"
            else:
                result = self.bot.order_manager.set_position_tpsl(symbol=symbol, sl_price=price)
                label = "STOP LOSS"
            
            if result.get('status') == 'ok':
                await self._edit_or_reply(
                    update,
                    f"✅ *{label} SET*\n\n{symbol}: ${price:.4f}",
                    self.kb.back_to_menu()
                )
            else:
                await self._edit_or_reply(
                    update,
                    self.fmt.format_error(f"Set {label} Failed", str(result)),
                    self.kb.back_to_menu()
                )
        except Exception as e:
            await self._edit_or_reply(
                update,
                self.fmt.format_error("Price Selection Error", str(e)),
                self.kb.back_to_menu()
            )
    
    # ==================== DATA FETCHERS ====================
    
    async def _get_dashboard_data(self) -> Dict[str, Any]:
        """Gather dashboard data from bot."""
        positions = await self._get_positions()
        
        return {
            'account_value': float(self.bot.account_value),
            'margin_used': float(self.bot.margin_used),
            'daily_pnl': float(getattr(self.bot, 'session_pnl', 0)),
            'daily_pnl_pct': 0,  # TODO: Calculate
            'open_positions': len(positions),
            'is_running': self.bot.is_running,
            'is_paused': getattr(self.bot, 'is_paused', False),
            'uptime': self.session_start,
            'trades_today': getattr(self.bot, 'trades_executed', 0),
        }
    
    async def _get_positions(self) -> List[Dict[str, Any]]:
        """Get current open positions."""
        try:
            account = await self.bot.client.get_account_state()
            positions = account.get('positions', [])
            
            # Enrich with current prices
            for pos in positions:
                symbol = pos.get('symbol')
                try:
                    current = await self.bot.client.get_market_price(symbol)
                    pos['current_price'] = float(current)
                except Exception:
                    pos['current_price'] = pos.get('entry_price', 0)
            
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    async def _get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent completed trades."""
        try:
            fills = self.bot.client.info.user_fills(self.bot.client.address)
            
            trades = []
            for fill in fills[:50]:
                pnl = float(fill.get('closedPnl', '0'))
                if pnl != 0:
                    trades.append({
                        'symbol': fill.get('coin'),
                        'side': 'long' if fill.get('side') == 'B' else 'short',
                        'pnl': pnl,
                        'time': datetime.fromtimestamp(fill['time'] / 1000, tz=timezone.utc),
                        'price': float(fill.get('px', 0)),
                        'size': float(fill.get('sz', 0)),
                    })
            
            return trades[:limit]
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []
    
    async def _get_pnl_data(self) -> Dict[str, Any]:
        """Get P&L breakdown data."""
        try:
            fills = self.bot.client.info.user_fills(self.bot.client.address)
            
            now = datetime.now(timezone.utc)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)
            
            today_pnl = 0
            today_fees = 0
            today_trades = 0
            weekly_pnl = 0
            monthly_pnl = 0
            
            for fill in fills:
                fill_time = datetime.fromtimestamp(fill['time'] / 1000, tz=timezone.utc)
                pnl = float(fill.get('closedPnl', '0'))
                fee = float(fill.get('fee', '0'))
                
                if fill_time >= today_start:
                    if pnl != 0:
                        today_pnl += pnl
                        today_trades += 1
                    today_fees += fee
                
                if fill_time >= week_ago and pnl != 0:
                    weekly_pnl += pnl
                
                if fill_time >= month_ago and pnl != 0:
                    monthly_pnl += pnl
            
            return {
                'today_pnl': today_pnl,
                'today_fees': today_fees,
                'today_trades': today_trades,
                'weekly_pnl': weekly_pnl,
                'monthly_pnl': monthly_pnl,
                'session_pnl': float(getattr(self.bot, 'session_pnl', 0)),
                'session_start': self.session_start,
            }
        except Exception as e:
            logger.error(f"Error getting PnL data: {e}")
            return {}
    
    async def _get_market_data(self) -> Dict[str, Any]:
        """Get market overview data."""
        try:
            symbol = self.bot.symbol
            price = await self.bot.client.get_market_price(symbol)
            
            candles = getattr(self.bot, 'current_candles', [])
            
            if candles and len(candles) > 0:
                high_24h = max(c.get('high', c.get('h', 0)) for c in candles[-1440:])
                low_24h = min(c.get('low', c.get('l', 0)) for c in candles[-1440:])
                open_24h = candles[-min(1440, len(candles))].get('open', candles[-1].get('o', float(price)))
                change_24h = ((float(price) - open_24h) / open_24h * 100) if open_24h else 0
            else:
                high_24h = low_24h = float(price)
                change_24h = 0
            
            # Get regime
            regime = "Unknown"
            if hasattr(self.bot, 'strategy') and hasattr(self.bot.strategy, 'strategies'):
                for name, strat in self.bot.strategy.strategies.items():
                    if hasattr(strat, 'regime_detector'):
                        regime = strat.regime_detector.current_regime.value
                        break
            
            return {
                'symbol': symbol,
                'price': float(price),
                'change_24h': change_24h,
                'high_24h': high_24h,
                'low_24h': low_24h,
                'regime': regime,
                'session': datetime.now(timezone.utc).strftime('%H:%M UTC'),
            }
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {'symbol': 'N/A', 'price': 0}
    
    # ==================== NOTIFICATIONS ====================
    
    async def notify_signal(self, signal: Dict[str, Any]):
        """Send signal notification."""
        if not self.notify_signals or not self._can_send('signal'):
            return
        
        # Defensive: ensure signal is a dict
        if not isinstance(signal, dict):
            logger.warning(f"notify_signal received non-dict: {type(signal)}")
            return
        
        message = self.fmt.format_signal_notification(signal)
        await self.send_message(message)
    
    async def notify_fill(self, fill: Dict[str, Any]):
        """Send fill notification."""
        if not self.notify_fills or not self._can_send('fill'):
            return
        
        # Defensive: ensure fill is a dict
        if not isinstance(fill, dict):
            logger.warning(f"notify_fill received non-dict: {type(fill)}")
            return
        
        message = self.fmt.format_fill_notification(fill)
        await self.send_message(message)
    
    async def notify_pnl_warning(self, pnl: float, threshold: float):
        """Send P&L warning notification."""
        if not self.notify_pnl_warnings or not self._can_send('pnl_warning'):
            return
        
        emoji = "⚠️" if pnl < 0 else "🎉"
        message = (
            f"{emoji} *P&L ALERT*\n\n"
            f"Session P&L: {self.fmt.format_money(pnl, sign=True)}\n"
            f"Threshold: {self.fmt.format_percent(threshold)}"
        )
        await self.send_message(message)
    
    async def notify_error(self, error: str, context: str = "System"):
        """Send error notification."""
        if not self._can_send('error'):
            return
        
        message = self.fmt.format_error(f"{context} Error", error[:200])
        await self.send_message(message)
    
    # ==================== SCHEDULED TASKS ====================
    
    async def _scheduled_status(self):
        """Send periodic status updates."""
        while self.is_running:
            try:
                await asyncio.sleep(self.status_interval)
                
                if not self.is_running:
                    break
                
                data = await self._get_dashboard_data()
                message = (
                    "📊 *SCHEDULED STATUS*\n\n"
                    f"Balance: {self.fmt.format_money(data['account_value'])}\n"
                    f"P&L: {self.fmt.pnl_emoji(data['daily_pnl'])} {self.fmt.format_money(data['daily_pnl'], sign=True)}\n"
                    f"Positions: {data['open_positions']}\n"
                    f"Status: {'🟢 Active' if data['is_running'] and not data['is_paused'] else '⏸️ Paused'}"
                )
                await self.send_message(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduled status error: {e}")
    
    async def _send_startup_message(self):
        """Send startup notification."""
        message = (
            "🚀 *HYPERBOT STARTED*\n\n"
            f"💰 Balance: {self.fmt.format_money(float(self.bot.account_value))}\n"
            f"🎯 Mode: Multi-Asset Trading\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
            "Tap /menu for control panel"
        )
        await self.send_message(message, self.kb.main_menu())
