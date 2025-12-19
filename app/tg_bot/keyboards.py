"""
Keyboard Factory for Telegram Bot
Creates consistent, reusable inline keyboards.
"""

from typing import List, Dict, Any
from telegram import InlineKeyboardButton, InlineKeyboardMarkup


class KeyboardFactory:
    """
    Factory for creating Telegram inline keyboards.
    Provides consistent styling and reusable patterns.
    """
    
    # ==================== MAIN MENUS ====================
    
    @staticmethod
    def main_menu() -> InlineKeyboardMarkup:
        """Create main menu keyboard."""
        keyboard = [
            [
                InlineKeyboardButton("📊 Dashboard", callback_data="dashboard"),
                InlineKeyboardButton("💼 Positions", callback_data="positions"),
            ],
            [
                InlineKeyboardButton("📈 Trades", callback_data="trades"),
                InlineKeyboardButton("💰 P&L", callback_data="pnl"),
            ],
            [
                InlineKeyboardButton("🚀 START", callback_data="bot_start"),
                InlineKeyboardButton("⏸️ PAUSE", callback_data="bot_pause"),
            ],
            [
                InlineKeyboardButton("📊 Market", callback_data="market"),
                InlineKeyboardButton("📈 Stats", callback_data="stats"),
            ],
            [
                InlineKeyboardButton("❓ Help", callback_data="help"),
                InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def quick_actions() -> InlineKeyboardMarkup:
        """Create quick actions keyboard."""
        keyboard = [
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="refresh"),
                InlineKeyboardButton("🏠 Menu", callback_data="main_menu"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def logs_actions() -> InlineKeyboardMarkup:
        """Create logs keyboard with proper refresh."""
        keyboard = [
            [
                InlineKeyboardButton("🔄 Refresh Logs", callback_data="refresh_logs"),
                InlineKeyboardButton("🏠 Menu", callback_data="main_menu"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def back_to_menu() -> InlineKeyboardMarkup:
        """Create back to menu keyboard."""
        keyboard = [
            [InlineKeyboardButton("🏠 Back to Menu", callback_data="main_menu")],
        ]
        return InlineKeyboardMarkup(keyboard)
    
    # ==================== DASHBOARD ====================
    
    @staticmethod
    def dashboard_actions() -> InlineKeyboardMarkup:
        """Create dashboard action buttons."""
        keyboard = [
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="refresh_dashboard"),
                InlineKeyboardButton("💼 Positions", callback_data="positions"),
            ],
            [
                InlineKeyboardButton("📈 Trades", callback_data="trades"),
                InlineKeyboardButton("💰 P&L", callback_data="pnl"),
            ],
            [
                InlineKeyboardButton("🏠 Menu", callback_data="main_menu"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)
    
    # ==================== POSITIONS ====================
    
    @staticmethod
    def positions_list(positions: List[Dict[str, Any]], page: int = 1, per_page: int = 5) -> InlineKeyboardMarkup:
        """
        Create positions list with action buttons.
        
        Args:
            positions: List of position dicts
            page: Current page
            per_page: Items per page
        """
        keyboard = []
        
        # Position action buttons
        total = len(positions)
        start = (page - 1) * per_page
        end = min(start + per_page, total)
        
        for pos in positions[start:end]:
            symbol = pos.get('symbol', 'N/A')
            side = "🟢" if pos.get('side', '').lower() == 'long' else "🔴"
            pnl = pos.get('unrealized_pnl', 0)
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            
            keyboard.append([
                InlineKeyboardButton(
                    f"{side} {symbol} ({pnl_str})",
                    callback_data=f"pos_detail_{symbol}"
                ),
                InlineKeyboardButton("❌", callback_data=f"close_confirm_{symbol}"),
            ])
        
        # Pagination
        total_pages = (total + per_page - 1) // per_page
        if total_pages > 1:
            nav_row = []
            if page > 1:
                nav_row.append(InlineKeyboardButton("◀️", callback_data=f"positions_page_{page-1}"))
            nav_row.append(InlineKeyboardButton(f"{page}/{total_pages}", callback_data="noop"))
            if page < total_pages:
                nav_row.append(InlineKeyboardButton("▶️", callback_data=f"positions_page_{page+1}"))
            keyboard.append(nav_row)
        
        # Actions
        keyboard.append([
            InlineKeyboardButton("🔄 Refresh", callback_data="positions"),
            InlineKeyboardButton("⚠️ Close All", callback_data="closeall_confirm"),
        ])
        
        keyboard.append([InlineKeyboardButton("🏠 Menu", callback_data="main_menu")])
        
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def position_detail(symbol: str) -> InlineKeyboardMarkup:
        """Create position detail keyboard with actions."""
        keyboard = [
            [
                InlineKeyboardButton("🎯 Set TP", callback_data=f"set_tp_{symbol}"),
                InlineKeyboardButton("🛑 Set SL", callback_data=f"set_sl_{symbol}"),
            ],
            [
                InlineKeyboardButton("📊 Auto TP/SL", callback_data=f"auto_tpsl_{symbol}"),
            ],
            [
                InlineKeyboardButton("❌ Close Position", callback_data=f"close_confirm_{symbol}"),
            ],
            [
                InlineKeyboardButton("◀️ Back", callback_data="positions"),
                InlineKeyboardButton("🔄 Refresh", callback_data=f"pos_detail_{symbol}"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)
    
    # ==================== CONFIRMATIONS ====================
    
    @staticmethod
    def close_confirm(symbol: str) -> InlineKeyboardMarkup:
        """Create close position confirmation keyboard."""
        keyboard = [
            [
                InlineKeyboardButton("✅ Yes, Close", callback_data=f"close_execute_{symbol}"),
                InlineKeyboardButton("❌ Cancel", callback_data="positions"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def closeall_confirm() -> InlineKeyboardMarkup:
        """Create close all confirmation keyboard."""
        keyboard = [
            [
                InlineKeyboardButton("⚠️ YES, CLOSE ALL", callback_data="closeall_execute"),
            ],
            [
                InlineKeyboardButton("❌ Cancel", callback_data="positions"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)
    
    # ==================== TRADES ====================
    
    @staticmethod
    def trades_list(page: int = 1, total_pages: int = 1) -> InlineKeyboardMarkup:
        """Create trades list keyboard with pagination."""
        keyboard = []
        
        # Pagination
        if total_pages > 1:
            nav_row = []
            if page > 1:
                nav_row.append(InlineKeyboardButton("◀️", callback_data=f"trades_page_{page-1}"))
            nav_row.append(InlineKeyboardButton(f"{page}/{total_pages}", callback_data="noop"))
            if page < total_pages:
                nav_row.append(InlineKeyboardButton("▶️", callback_data=f"trades_page_{page+1}"))
            keyboard.append(nav_row)
        
        keyboard.append([
            InlineKeyboardButton("🔄 Refresh", callback_data="trades"),
            InlineKeyboardButton("🏠 Menu", callback_data="main_menu"),
        ])
        
        return InlineKeyboardMarkup(keyboard)
    
    # ==================== BOT CONTROL ====================
    
    @staticmethod
    def bot_control(is_running: bool, is_paused: bool) -> InlineKeyboardMarkup:
        """Create bot control keyboard based on current state."""
        keyboard = []
        
        if is_paused:
            keyboard.append([
                InlineKeyboardButton("🚀 RESUME TRADING", callback_data="bot_start"),
            ])
        elif is_running:
            keyboard.append([
                InlineKeyboardButton("⏸️ PAUSE TRADING", callback_data="bot_pause"),
            ])
        else:
            keyboard.append([
                InlineKeyboardButton("🚀 START TRADING", callback_data="bot_start"),
            ])
        
        keyboard.append([InlineKeyboardButton("🏠 Menu", callback_data="main_menu")])
        
        return InlineKeyboardMarkup(keyboard)
    
    # ==================== SETTINGS ====================
    
    @staticmethod
    def settings_menu() -> InlineKeyboardMarkup:
        """Create settings menu keyboard."""
        keyboard = [
            [
                InlineKeyboardButton("🔔 Notifications", callback_data="settings_notifications"),
            ],
            [
                InlineKeyboardButton("📊 Display", callback_data="settings_display"),
            ],
            [
                InlineKeyboardButton("🏠 Menu", callback_data="main_menu"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def notification_settings(
        signals: bool = True,
        fills: bool = True,
        pnl_warnings: bool = True,
    ) -> InlineKeyboardMarkup:
        """Create notification settings toggles."""
        keyboard = [
            [
                InlineKeyboardButton(
                    f"{'✅' if signals else '❌'} Signal Alerts",
                    callback_data="toggle_signals"
                ),
            ],
            [
                InlineKeyboardButton(
                    f"{'✅' if fills else '❌'} Trade Fills",
                    callback_data="toggle_fills"
                ),
            ],
            [
                InlineKeyboardButton(
                    f"{'✅' if pnl_warnings else '❌'} P&L Warnings",
                    callback_data="toggle_pnl_warnings"
                ),
            ],
            [
                InlineKeyboardButton("◀️ Back", callback_data="settings"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)
    
    # ==================== NUMERIC INPUT ====================
    
    @staticmethod
    def price_input(symbol: str, action: str, current_price: float) -> InlineKeyboardMarkup:
        """
        Create price input keyboard with quick adjustments.
        
        Args:
            symbol: Trading symbol
            action: 'tp' or 'sl'
            current_price: Current market price
        """
        # Calculate quick adjustment amounts (percentage based)
        adjustments = [0.5, 1.0, 2.0, 5.0]
        
        keyboard = []
        
        # Price adjustment buttons
        for adj in adjustments:
            if action == 'tp':
                new_price = current_price * (1 + adj / 100)
                label = f"+{adj}%"
            else:
                new_price = current_price * (1 - adj / 100)
                label = f"-{adj}%"
            
            keyboard.append([
                InlineKeyboardButton(
                    f"{label} (${new_price:.2f})",
                    callback_data=f"set_{action}_price_{symbol}_{new_price:.4f}"
                ),
            ])
        
        keyboard.append([
            InlineKeyboardButton("❌ Cancel", callback_data=f"pos_detail_{symbol}"),
        ])
        
        return InlineKeyboardMarkup(keyboard)
    
    # ==================== EMPTY STATE ====================
    
    @staticmethod
    def empty_positions() -> InlineKeyboardMarkup:
        """Create keyboard for empty positions state."""
        keyboard = [
            [
                InlineKeyboardButton("📊 Market", callback_data="market"),
                InlineKeyboardButton("📈 View Trades", callback_data="trades"),
            ],
            [InlineKeyboardButton("🏠 Menu", callback_data="main_menu")],
        ]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def empty_trades() -> InlineKeyboardMarkup:
        """Create keyboard for empty trades state."""
        keyboard = [
            [
                InlineKeyboardButton("💼 Positions", callback_data="positions"),
                InlineKeyboardButton("📊 Market", callback_data="market"),
            ],
            [InlineKeyboardButton("🏠 Menu", callback_data="main_menu")],
        ]
        return InlineKeyboardMarkup(keyboard)
    
    # ==================== NEW: TRADING KEYBOARDS ====================
    
    @staticmethod
    def trade_confirm(symbol: str, action: str, size_pct: float = None) -> InlineKeyboardMarkup:
        """Create trade confirmation keyboard."""
        size_str = f"_{size_pct}" if size_pct else ""
        keyboard = [
            [
                InlineKeyboardButton(
                    "✅ CONFIRM TRADE", 
                    callback_data=f"execute_{action}_{symbol}{size_str}"
                ),
            ],
            [
                InlineKeyboardButton("❌ Cancel", callback_data="main_menu"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def killswitch_actions(trading_allowed: bool) -> InlineKeyboardMarkup:
        """Create kill switch action buttons."""
        if trading_allowed:
            keyboard = [
                [
                    InlineKeyboardButton("🛑 STOP ALL TRADING", callback_data="ks_stop"),
                ],
                [
                    InlineKeyboardButton("🏠 Back", callback_data="main_menu"),
                ],
            ]
        else:
            keyboard = [
                [
                    InlineKeyboardButton("✅ RESUME TRADING", callback_data="ks_resume"),
                ],
                [
                    InlineKeyboardButton("🏠 Back", callback_data="main_menu"),
                ],
            ]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def quick_trade_menu() -> InlineKeyboardMarkup:
        """Create quick trade menu."""
        keyboard = [
            [
                InlineKeyboardButton("📈 LONG SOL", callback_data="quick_buy_SOL"),
                InlineKeyboardButton("📉 SHORT SOL", callback_data="quick_sell_SOL"),
            ],
            [
                InlineKeyboardButton("📈 LONG ETH", callback_data="quick_buy_ETH"),
                InlineKeyboardButton("📉 SHORT ETH", callback_data="quick_sell_ETH"),
            ],
            [
                InlineKeyboardButton("📈 LONG BTC", callback_data="quick_buy_BTC"),
                InlineKeyboardButton("📉 SHORT BTC", callback_data="quick_sell_BTC"),
            ],
            [InlineKeyboardButton("🏠 Menu", callback_data="main_menu")],
        ]
        return InlineKeyboardMarkup(keyboard)
