"""
Analytics Dashboard - Generate trading performance insights from database
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class AnalyticsDashboard:
    """Generate trading analytics and performance reports"""
    
    def __init__(self, db_manager):
        """
        Initialize analytics dashboard
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
    
    async def get_overview(self) -> Dict[str, Any]:
        """Get comprehensive trading overview"""
        stats = await self.db.get_total_stats()
        
        if not stats or stats.get('total_trades', 0) == 0:
            return {
                'status': 'NO_DATA',
                'message': 'No trades found in database'
            }
        
        return {
            'status': 'SUCCESS',
            'total_trades': stats.get('total_trades', 0),
            'winning_trades': stats.get('winning_trades', 0),
            'losing_trades': stats.get('losing_trades', 0),
            'win_rate': stats.get('win_rate', 0),
            'total_pnl': stats.get('total_pnl', 0),
            'avg_win': stats.get('avg_win', 0),
            'avg_loss': stats.get('avg_loss', 0),
            'best_trade': stats.get('best_trade', 0),
            'worst_trade': stats.get('worst_trade', 0),
            'profit_factor': self._calculate_profit_factor(
                stats.get('avg_win', 0),
                stats.get('avg_loss', 0),
                stats.get('winning_trades', 0),
                stats.get('losing_trades', 0)
            )
        }
    
    async def get_daily_performance(self, days: int = 30) -> Dict[str, Any]:
        """
        Get daily performance breakdown
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Daily stats with trends
        """
        daily_data = await self.db.get_daily_performance(days=days)
        
        if not daily_data:
            return {
                'status': 'NO_DATA',
                'message': f'No daily data available for last {days} days'
            }
        
        # Calculate trends
        profitable_days = sum(1 for day in daily_data if float(day.get('total_pnl', 0)) > 0)
        total_days = len(daily_data)
        
        return {
            'status': 'SUCCESS',
            'period': f'Last {days} days',
            'trading_days': total_days,
            'profitable_days': profitable_days,
            'losing_days': total_days - profitable_days,
            'daily_win_rate': round(profitable_days / total_days * 100, 2) if total_days > 0 else 0,
            'best_day': max(daily_data, key=lambda d: float(d.get('total_pnl', 0))) if daily_data else None,
            'worst_day': min(daily_data, key=lambda d: float(d.get('total_pnl', 0))) if daily_data else None,
            'avg_daily_pnl': sum(float(d.get('total_pnl', 0)) for d in daily_data) / total_days if total_days > 0 else 0,
            'daily_data': daily_data[:10]  # Latest 10 days for display
        }
    
    async def get_symbol_performance(self) -> Dict[str, Any]:
        """Get performance breakdown by trading symbol"""
        symbol_data = await self.db.get_symbol_performance()
        
        if not symbol_data:
            return {
                'status': 'NO_DATA',
                'message': 'No symbol data available'
            }
        
        return {
            'status': 'SUCCESS',
            'total_symbols': len(symbol_data),
            'best_symbol': symbol_data[0] if symbol_data else None,
            'worst_symbol': symbol_data[-1] if symbol_data else None,
            'symbols': symbol_data
        }
    
    async def get_hourly_analysis(self) -> Dict[str, Any]:
        """Get best and worst trading hours"""
        hourly_data = await self.db.get_hourly_activity()
        
        if not hourly_data:
            return {
                'status': 'NO_DATA',
                'message': 'No hourly data available'
            }
        
        # Find best and worst hours
        best_hours = sorted(hourly_data, key=lambda h: float(h.get('total_pnl', 0)), reverse=True)[:5]
        worst_hours = sorted(hourly_data, key=lambda h: float(h.get('total_pnl', 0)))[:5]
        
        # Find most active hours
        most_active = sorted(hourly_data, key=lambda h: int(h.get('total_trades', 0)), reverse=True)[:5]
        
        return {
            'status': 'SUCCESS',
            'best_hours': best_hours,
            'worst_hours': worst_hours,
            'most_active_hours': most_active,
            'recommendation': self._generate_hourly_recommendation(best_hours, worst_hours)
        }
    
    async def get_ml_performance(self) -> Dict[str, Any]:
        """Get ML model performance analysis"""
        ml_data = await self.db.get_ml_model_performance()
        
        if not ml_data:
            return {
                'status': 'NO_DATA',
                'message': 'No ML predictions recorded yet'
            }
        
        best_model = ml_data[0] if ml_data else None
        
        return {
            'status': 'SUCCESS',
            'total_models': len(ml_data),
            'best_model': best_model,
            'models': ml_data,
            'recommendation': self._generate_ml_recommendation(ml_data)
        }
    
    async def generate_full_report(self) -> str:
        """
        Generate comprehensive analytics report
        
        Returns:
            Formatted text report for Telegram
        """
        overview = await self.get_overview()
        
        if overview.get('status') == 'NO_DATA':
            return "📊 *Analytics Dashboard*\n\n❌ No trading data available yet.\n\nStart trading to see analytics!"
        
        daily = await self.get_daily_performance(days=30)
        symbols = await self.get_symbol_performance()
        hourly = await self.get_hourly_analysis()
        ml = await self.get_ml_performance()
        
        report = []
        
        # Header
        report.append("📊 *HYPERBOT ANALYTICS DASHBOARD*")
        report.append("=" * 45)
        report.append("")
        
        # Overview
        report.append("📈 *OVERALL PERFORMANCE*")
        report.append(f"Total Trades: {overview['total_trades']}")
        report.append(f"Win Rate: {overview['win_rate']}%")
        report.append(f"Total P&L: ${overview['total_pnl']:+.2f}")
        report.append(f"Profit Factor: {overview['profit_factor']:.2f}")
        report.append(f"Best Trade: ${overview['best_trade']:+.2f}")
        report.append(f"Worst Trade: ${overview['worst_trade']:+.2f}")
        report.append("")
        
        # Daily performance
        if daily.get('status') == 'SUCCESS':
            report.append("📅 *DAILY PERFORMANCE (Last 30 Days)*")
            report.append(f"Trading Days: {daily['trading_days']}")
            report.append(f"Profitable Days: {daily['profitable_days']} ({daily['daily_win_rate']}%)")
            report.append(f"Avg Daily P&L: ${daily['avg_daily_pnl']:+.2f}")
            
            if daily.get('best_day'):
                best = daily['best_day']
                report.append(f"Best Day: {best.get('trade_date')} (${float(best.get('total_pnl', 0)):+.2f})")
            report.append("")
        
        # Symbol performance
        if symbols.get('status') == 'SUCCESS':
            report.append("🎯 *TOP SYMBOLS*")
            for symbol in symbols['symbols'][:5]:
                report.append(
                    f"{symbol['symbol']}: "
                    f"{symbol['total_trades']} trades, "
                    f"{symbol['win_rate']}% WR, "
                    f"${float(symbol['total_pnl']):+.2f}"
                )
            report.append("")
        
        # Hourly analysis
        if hourly.get('status') == 'SUCCESS':
            report.append("⏰ *BEST TRADING HOURS (UTC)*")
            for hour in hourly['best_hours'][:3]:
                report.append(
                    f"{int(hour['hour_utc']):02d}:00 - "
                    f"{hour['total_trades']} trades, "
                    f"{hour['win_rate']}% WR, "
                    f"${float(hour['total_pnl']):+.2f}"
                )
            
            if hourly.get('recommendation'):
                report.append(f"\n💡 {hourly['recommendation']}")
            report.append("")
        
        # ML performance
        if ml.get('status') == 'SUCCESS':
            report.append("🤖 *ML MODEL PERFORMANCE*")
            for model in ml['models']:
                report.append(
                    f"{model['model_name']}: "
                    f"{model['accuracy']}% accurate, "
                    f"{model['total_predictions']} predictions"
                )
            
            if ml.get('recommendation'):
                report.append(f"\n💡 {ml['recommendation']}")
            report.append("")
        
        # Footer
        report.append("=" * 45)
        report.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        
        return "\n".join(report)
    
    # Helper methods
    
    def _calculate_profit_factor(
        self,
        avg_win: float,
        avg_loss: float,
        winning_trades: int,
        losing_trades: int
    ) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if losing_trades == 0 or avg_loss == 0:
            return 0.0
        
        gross_profit = avg_win * winning_trades
        gross_loss = abs(avg_loss) * losing_trades
        
        return round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0
    
    def _generate_hourly_recommendation(
        self,
        best_hours: List[Dict],
        worst_hours: List[Dict]
    ) -> str:
        """Generate trading hour recommendation"""
        if not best_hours:
            return "Not enough data for recommendation"
        
        best_hour = int(best_hours[0].get('hour_utc', 0))
        worst_hour = int(worst_hours[0].get('hour_utc', 0))
        
        return (
            f"Focus on trading around {best_hour:02d}:00 UTC. "
            f"Avoid {worst_hour:02d}:00 UTC."
        )
    
    def _generate_ml_recommendation(self, ml_data: List[Dict]) -> str:
        """Generate ML model recommendation"""
        if not ml_data:
            return "No ML data available"
        
        best = ml_data[0]
        accuracy = float(best.get('accuracy', 0))
        
        if accuracy >= 70:
            return f"✅ {best['model_name']} is performing excellently ({accuracy}% accurate)"
        elif accuracy >= 60:
            return f"⚠️  {best['model_name']} needs improvement ({accuracy}% accurate)"
        else:
            return f"❌ Consider retraining models (best: {accuracy}% accurate)"


async def format_analytics_message(dashboard: AnalyticsDashboard, query_type: str = "full") -> str:
    """
    Format analytics for Telegram display
    
    Args:
        dashboard: AnalyticsDashboard instance
        query_type: Type of report ('full', 'daily', 'symbols', 'hours', 'ml')
    
    Returns:
        Formatted message string
    """
    if query_type == "full":
        return await dashboard.generate_full_report()
    
    elif query_type == "daily":
        data = await dashboard.get_daily_performance(days=30)
        if data.get('status') == 'NO_DATA':
            return "❌ No daily performance data available"
        
        msg = [
            "📅 *Daily Performance (Last 30 Days)*",
            "",
            f"Trading Days: {data['trading_days']}",
            f"Profitable: {data['profitable_days']} ({data['daily_win_rate']}%)",
            f"Avg Daily P&L: ${data['avg_daily_pnl']:+.2f}",
            ""
        ]
        
        if data.get('best_day'):
            best = data['best_day']
            msg.append(f"Best Day: {best.get('trade_date')}")
            msg.append(f"  P&L: ${float(best.get('total_pnl', 0)):+.2f}")
            msg.append(f"  Trades: {best.get('total_trades')}")
            msg.append(f"  Win Rate: {best.get('win_rate')}%")
        
        return "\n".join(msg)
    
    elif query_type == "symbols":
        data = await dashboard.get_symbol_performance()
        if data.get('status') == 'NO_DATA':
            return "❌ No symbol performance data available"
        
        msg = [
            "🎯 *Symbol Performance*",
            ""
        ]
        
        for symbol in data['symbols'][:10]:
            msg.append(
                f"{symbol['symbol']}: "
                f"{symbol['win_rate']}% WR, "
                f"${float(symbol['total_pnl']):+.2f} "
                f"({symbol['total_trades']} trades)"
            )
        
        return "\n".join(msg)
    
    elif query_type == "hours":
        data = await dashboard.get_hourly_analysis()
        if data.get('status') == 'NO_DATA':
            return "❌ No hourly data available"
        
        msg = [
            "⏰ *Trading Hours Analysis (UTC)*",
            "",
            "*Top 5 Hours:*"
        ]
        
        for hour in data['best_hours']:
            msg.append(
                f"{int(hour['hour_utc']):02d}:00 - "
                f"{hour['win_rate']}% WR, "
                f"${float(hour['total_pnl']):+.2f} "
                f"({hour['total_trades']} trades)"
            )
        
        msg.append(f"\n💡 {data['recommendation']}")
        
        return "\n".join(msg)
    
    elif query_type == "ml":
        data = await dashboard.get_ml_performance()
        if data.get('status') == 'NO_DATA':
            return "❌ No ML performance data available"
        
        msg = [
            "🤖 *ML Model Performance*",
            ""
        ]
        
        for model in data['models']:
            msg.append(
                f"{model['model_name']}: "
                f"{model['accuracy']}% accurate "
                f"({model['total_predictions']} predictions)"
            )
        
        msg.append(f"\n💡 {data['recommendation']}")
        
        return "\n".join(msg)
    
    else:
        return "❌ Unknown analytics query type"
