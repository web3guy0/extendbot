"""
Rich Message Formatters for Telegram Bot
Creates beautiful, consistent message layouts with Unicode art and clear formatting.
"""

from typing import Dict, Any, List
from datetime import datetime, timezone


class MessageFormatter:
    """
    Modern message formatter with consistent styling.
    Uses Unicode art, emojis, and clean layouts.
    """
    
    # Unicode box drawing characters
    BOX_TOP = "╭" + "─" * 30 + "╮"
    BOX_BOTTOM = "╰" + "─" * 30 + "╯"
    BOX_DIVIDER = "├" + "─" * 30 + "┤"
    
    # Progress bar characters
    PROGRESS_FULL = "█"
    PROGRESS_EMPTY = "░"
    PROGRESS_HALF = "▓"
    
    # Trend indicators
    TREND_UP = "▲"
    TREND_DOWN = "▼"
    TREND_FLAT = "━"
    
    @staticmethod
    def escape_markdown(text: str) -> str:
        """Escape Markdown special characters."""
        chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in chars:
            text = text.replace(char, f'\\{char}')
        return text
    
    @staticmethod
    def format_money(value: float, decimals: int = 2, sign: bool = False) -> str:
        """Format money with optional sign."""
        if sign:
            return f"${value:+,.{decimals}f}"
        return f"${value:,.{decimals}f}"
    
    @staticmethod
    def format_percent(value: float, decimals: int = 2, sign: bool = False) -> str:
        """Format percentage with optional sign."""
        if sign:
            return f"{value:+.{decimals}f}%"
        return f"{value:.{decimals}f}%"
    
    @staticmethod
    def format_number(value: float, decimals: int = 4) -> str:
        """Format number with specified decimals."""
        return f"{value:,.{decimals}f}"
    
    @staticmethod
    def progress_bar(percent: float, width: int = 10) -> str:
        """Create a Unicode progress bar."""
        percent = max(0, min(100, percent))
        filled = int(percent / 100 * width)
        return MessageFormatter.PROGRESS_FULL * filled + MessageFormatter.PROGRESS_EMPTY * (width - filled)
    
    @staticmethod
    def pnl_emoji(value: float) -> str:
        """Get emoji for PnL value."""
        if value > 0:
            return "🟢"
        elif value < 0:
            return "🔴"
        return "⚪"
    
    @staticmethod
    def trend_indicator(value: float, threshold: float = 0.1) -> str:
        """Get trend indicator."""
        if value > threshold:
            return MessageFormatter.TREND_UP
        elif value < -threshold:
            return MessageFormatter.TREND_DOWN
        return MessageFormatter.TREND_FLAT
    
    @staticmethod
    def time_ago(dt: datetime) -> str:
        """Format time as relative string."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        diff = now - dt
        
        if diff.total_seconds() < 60:
            return "just now"
        elif diff.total_seconds() < 3600:
            mins = int(diff.total_seconds() / 60)
            return f"{mins}m ago"
        elif diff.total_seconds() < 86400:
            hours = int(diff.total_seconds() / 3600)
            return f"{hours}h ago"
        else:
            days = diff.days
            return f"{days}d ago"
    
    @staticmethod
    def format_uptime(start_time: datetime) -> str:
        """Format uptime as human readable string."""
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        diff = now - start_time
        
        days = diff.days
        hours, remainder = divmod(diff.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m {seconds}s"
    
    # ==================== DASHBOARD MESSAGES ====================
    
    @classmethod
    def format_dashboard(cls, data: Dict[str, Any]) -> str:
        """
        Format main dashboard message.
        
        Args:
            data: {
                'account_value': float,
                'margin_used': float,
                'daily_pnl': float,
                'daily_pnl_pct': float,
                'open_positions': int,
                'is_running': bool,
                'is_paused': bool,
                'uptime': datetime,
                'trades_today': int,
            }
        """
        account = data.get('account_value', 0)
        margin = data.get('margin_used', 0)
        available = account - margin
        margin_pct = (margin / account * 100) if account > 0 else 0
        
        daily_pnl = data.get('daily_pnl', 0)
        daily_pct = data.get('daily_pnl_pct', 0)
        
        status = "🟢 ACTIVE" if data.get('is_running') and not data.get('is_paused') else \
                 "⏸️ PAUSED" if data.get('is_paused') else "🔴 STOPPED"
        
        uptime = cls.format_uptime(data.get('uptime', datetime.now(timezone.utc)))
        
        lines = [
            "📊 *HYPERBOT DASHBOARD*",
            "",
            "─────── 💰 Account ───────",
            f"Balance:    {cls.format_money(account)}",
            f"Margin:     {cls.format_money(margin)} ({cls.format_percent(margin_pct)})",
            f"Available:  {cls.format_money(available)}",
            "",
            "─────── 📈 Today ─────────",
            f"P&L: {cls.pnl_emoji(daily_pnl)} {cls.format_money(daily_pnl, sign=True)} ({cls.format_percent(daily_pct, sign=True)})",
            f"Trades:     {data.get('trades_today', 0)}",
            "",
            "─────── ⚡ Status ────────",
            f"Bot:        {status}",
            f"Positions:  {data.get('open_positions', 0)}",
            f"Uptime:     {uptime}",
        ]
        
        return "\n".join(lines)
    
    @classmethod
    def format_positions_list(cls, positions: List[Dict[str, Any]], page: int = 1, per_page: int = 5) -> str:
        """
        Format positions list with pagination.
        
        Args:
            positions: List of position dicts
            page: Current page (1-indexed)
            per_page: Positions per page
        """
        if not positions:
            return (
                "📭 *NO OPEN POSITIONS*\n\n"
                "All clear! No positions currently open.\n"
                "Use strategy signals or manual trading to open positions."
            )
        
        total = len(positions)
        total_pages = (total + per_page - 1) // per_page
        start = (page - 1) * per_page
        end = min(start + per_page, total)
        
        lines = [
            f"💼 *OPEN POSITIONS* ({total} total)",
            f"Page {page}/{total_pages}",
            "",
        ]
        
        total_pnl = 0
        for i, pos in enumerate(positions[start:end], start + 1):
            side = (pos.get('side') or 'unknown').upper()
            side_emoji = "🟢" if side == 'LONG' else "🔴"
            
            pnl = pos.get('unrealized_pnl') or 0
            pnl_pct = pos.get('unrealized_pnl_pct') or 0
            total_pnl += pnl
            
            entry = pos.get('entry_price') or 0
            current = pos.get('current_price') or entry
            size = pos.get('size') or 0
            leverage = pos.get('leverage') or 1
            
            lines.extend([
                f"━━━ {i}. {side_emoji} {side} {pos.get('symbol', 'N/A')} ━━━",
                f"📍 Entry: {cls.format_money(entry, 4)} → {cls.format_money(current, 4)}",
                f"📦 Size: {cls.format_number(abs(size))} ({leverage}x)",
                f"{cls.pnl_emoji(pnl)} PnL: {cls.format_money(pnl, sign=True)} ({cls.format_percent(pnl_pct, sign=True)})",
                "",
            ])
        
        lines.append(f"─────────────────────────")
        lines.append(f"📊 Total Unrealized: {cls.pnl_emoji(total_pnl)} {cls.format_money(total_pnl, sign=True)}")
        
        return "\n".join(lines)
    
    @classmethod
    def format_trades_list(cls, trades: List[Dict[str, Any]], limit: int = 10) -> str:
        """
        Format recent trades list.
        
        Args:
            trades: List of trade dicts with 'symbol', 'side', 'pnl', 'time', 'entry', 'exit'
            limit: Max trades to show
        """
        if not trades:
            return (
                "📭 *NO RECENT TRADES*\n\n"
                "No completed trades yet.\n"
                "Trades will appear here after positions are closed."
            )
        
        lines = [
            f"📈 *RECENT TRADES* (Last {min(len(trades), limit)})",
            "",
        ]
        
        total_pnl = 0
        wins = 0
        losses = 0
        
        for trade in trades[:limit]:
            pnl = trade.get('pnl', 0)
            total_pnl += pnl
            
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
            
            side_emoji = "🟢" if trade.get('side', '').lower() == 'long' else "🔴"
            time_str = cls.time_ago(trade.get('time', datetime.now(timezone.utc)))
            
            lines.append(
                f"{cls.pnl_emoji(pnl)} {side_emoji} {trade.get('symbol', 'N/A')} "
                f"{cls.format_money(pnl, sign=True)} • {time_str}"
            )
        
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        
        lines.extend([
            "",
            "─────────────────────────",
            f"📊 Total: {cls.pnl_emoji(total_pnl)} {cls.format_money(total_pnl, sign=True)}",
            f"✅ Wins: {wins}  ❌ Losses: {losses}  📈 Win Rate: {cls.format_percent(win_rate)}",
        ])
        
        return "\n".join(lines)
    
    @classmethod
    def format_pnl_breakdown(cls, data: Dict[str, Any]) -> str:
        """
        Format PnL breakdown.
        
        Args:
            data: {
                'today_pnl': float,
                'today_fees': float,
                'today_trades': int,
                'weekly_pnl': float,
                'monthly_pnl': float,
                'session_pnl': float,
                'session_start': datetime,
            }
        """
        today_pnl = data.get('today_pnl', 0)
        today_fees = data.get('today_fees', 0)
        today_net = today_pnl - today_fees
        
        lines = [
            "💰 *P&L BREAKDOWN*",
            "",
            "─────── 📅 Today ────────",
            f"Gross P&L:  {cls.pnl_emoji(today_pnl)} {cls.format_money(today_pnl, sign=True)}",
            f"Fees:       💸 {cls.format_money(today_fees)}",
            f"Net P&L:    {cls.pnl_emoji(today_net)} {cls.format_money(today_net, sign=True)}",
            f"Trades:     🔄 {data.get('today_trades', 0)}",
            "",
            "─────── 📊 Period ───────",
            f"7 Days:     {cls.pnl_emoji(data.get('weekly_pnl', 0))} {cls.format_money(data.get('weekly_pnl', 0), sign=True)}",
            f"30 Days:    {cls.pnl_emoji(data.get('monthly_pnl', 0))} {cls.format_money(data.get('monthly_pnl', 0), sign=True)}",
            "",
            "─────── ⏱️ Session ──────",
            f"Session:    {cls.pnl_emoji(data.get('session_pnl', 0))} {cls.format_money(data.get('session_pnl', 0), sign=True)}",
        ]
        
        if data.get('session_start'):
            lines.append(f"Started:    {data['session_start'].strftime('%m/%d %H:%M UTC')}")
        
        return "\n".join(lines)
    
    @classmethod
    def format_market_overview(cls, data: Dict[str, Any]) -> str:
        """
        Format market overview.
        
        Args:
            data: {
                'symbol': str,
                'price': float,
                'change_24h': float,
                'high_24h': float,
                'low_24h': float,
                'volume_24h': float,
                'regime': str,
                'session': str,
            }
        """
        price = data.get('price', 0)
        change = data.get('change_24h', 0)
        
        lines = [
            f"📊 *MARKET: {data.get('symbol', 'N/A')}*",
            "",
            "─────── 💵 Price ────────",
            f"Current:    {cls.format_money(price, 4)}",
            f"24h Change: {cls.trend_indicator(change)} {cls.format_percent(change, sign=True)}",
            "",
            "─────── 📈 Range ────────",
            f"24h High:   {cls.format_money(data.get('high_24h', 0), 4)}",
            f"24h Low:    {cls.format_money(data.get('low_24h', 0), 4)}",
        ]
        
        if data.get('volume_24h'):
            lines.append(f"Volume:     {cls.format_money(data.get('volume_24h', 0))}")
        
        lines.extend([
            "",
            "─────── 🎯 Analysis ─────",
            f"Regime:     {data.get('regime', 'Unknown')}",
            f"Session:    {data.get('session', 'Unknown')}",
        ])
        
        return "\n".join(lines)
    
    @classmethod
    def format_signal_notification(cls, signal: Dict[str, Any]) -> str:
        """
        Format new trading signal notification.
        
        Args:
            signal: Signal dict with entry, tp, sl, score, reason, etc.
        """
        # Defensive: ensure signal is a dict
        if not isinstance(signal, dict):
            return f"⚠️ Invalid signal format: {type(signal)}"
        
        side = signal.get('side', 'buy').upper()
        side_emoji = "🟢 LONG" if side == 'BUY' else "🔴 SHORT"
        
        entry = signal.get('entry_price', 0)
        tp = signal.get('take_profit', 0)
        sl = signal.get('stop_loss', 0)
        
        # Calculate percentages
        tp_pct = ((tp - entry) / entry * 100) if entry and side == 'BUY' else ((entry - tp) / entry * 100) if entry else 0
        sl_pct = ((entry - sl) / entry * 100) if entry and side == 'BUY' else ((sl - entry) / entry * 100) if entry else 0
        
        score = signal.get('signal_score', 0)
        max_score = signal.get('max_score', 12)
        score_bar = cls.progress_bar(score / max_score * 100, 8)
        
        lines = [
            f"🎯 *NEW SIGNAL*",
            "",
            f"{side_emoji} {signal.get('symbol', 'N/A')}",
            "",
            f"📍 Entry:   {cls.format_money(entry, 4)}",
            f"🎯 TP:      {cls.format_money(tp, 4)} (+{cls.format_percent(abs(tp_pct))})",
            f"🛑 SL:      {cls.format_money(sl, 4)} (-{cls.format_percent(abs(sl_pct))})",
            "",
            f"📦 Size:    {cls.format_number(signal.get('size', 0))} ({signal.get('leverage', 1)}x)",
            f"⭐ Score:   {score_bar} {score}/{max_score}",
        ]
        
        if signal.get('regime'):
            lines.append(f"📊 Regime:  {signal['regime']}")
        
        if signal.get('reason'):
            lines.append(f"📝 Reason:  {signal['reason'][:50]}")
        
        return "\n".join(lines)
    
    @classmethod
    def format_fill_notification(cls, fill: Dict[str, Any]) -> str:
        """Format trade fill notification."""
        # Defensive: ensure fill is a dict
        if not isinstance(fill, dict):
            return f"⚠️ Invalid fill format: {type(fill)}"
        
        side = fill.get('side', 'buy').upper()
        side_emoji = "🟢" if side == 'BUY' or side == 'B' else "🔴"
        
        pnl = fill.get('closed_pnl', 0)
        
        if pnl != 0:
            # Closing trade
            lines = [
                f"💰 *TRADE CLOSED*",
                "",
                f"{side_emoji} {fill.get('symbol', 'N/A')}",
                f"Price: {cls.format_money(fill.get('price', 0), 4)}",
                f"Size:  {cls.format_number(fill.get('size', 0))}",
                "",
                f"{cls.pnl_emoji(pnl)} Realized P&L: {cls.format_money(pnl, sign=True)}",
            ]
        else:
            # Opening trade
            lines = [
                f"📦 *POSITION OPENED*",
                "",
                f"{side_emoji} {fill.get('symbol', 'N/A')}",
                f"Price: {cls.format_money(fill.get('price', 0), 4)}",
                f"Size:  {cls.format_number(fill.get('size', 0))}",
            ]
        
        return "\n".join(lines)
    
    @classmethod
    def format_error(cls, title: str, error: str, suggestion: str = None) -> str:
        """Format error message."""
        lines = [
            f"❌ *{title}*",
            "",
            f"Error: {error[:200]}",
        ]
        
        if suggestion:
            lines.extend(["", f"💡 {suggestion}"])
        
        return "\n".join(lines)
    
    @classmethod
    def format_success(cls, title: str, message: str) -> str:
        """Format success message."""
        return f"✅ *{title}*\n\n{message}"
    
    @classmethod
    def format_help(cls) -> str:
        """Format comprehensive help message."""
        return """📚 *HYPERBOT COMMANDS*

─────── 📊 Monitoring ───────
/status  • Dashboard overview
/pos     • Open positions
/trades  • Recent trade history
/pnl     • P&L breakdown
/balance • Quick balance check

─────── 📈 Market ───────────
/market  • Current prices & data
/regime  • Market regime analysis
/signal [symbol] • Full analysis
/assets  • Configured assets

─────── 🎛️ Trading ──────────
/buy [symbol] [%] • Open LONG
/sell [symbol] [%] • Open SHORT
/close [symbol] • Close position
/closeall • Close all positions

─────── 🎯 Risk Management ──
/sl [symbol] [price] • Set stop loss
/tp [symbol] [price] • Set take profit
/managed • View managed positions
/risk    • Risk status
/killswitch • Emergency stop

─────── 📊 Analytics ────────
/stats   • Performance stats
/report  • Full performance report
/kelly   • Kelly sizing info
/db      • Database stats
/tier    • Account tier info

─────── ⚙️ Settings ─────────
/alerts  • Configure notifications
/config  • View configuration
/logs    • Recent logs

─────── ℹ️ Help ─────────────
/help    • This help menu
/menu    • Main menu

💡 Tips:
• Use /signal SOL for live analysis
• Use /tier to see your account settings
• Use /report for performance review"""
