"""
PostgreSQL Database Manager for HyperBot
Handles all database operations with connection pooling and async support
"""
import asyncio
import asyncpg
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Async PostgreSQL database manager with connection pooling"""
    
    def __init__(self, database_url: str, min_pool_size: int = 2, max_pool_size: int = 10):
        """
        Initialize database manager
        
        Args:
            database_url: PostgreSQL connection string (postgres://user:pass@host:port/db)
            min_pool_size: Minimum connections in pool
            max_pool_size: Maximum connections in pool
        """
        self.database_url = database_url
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.pool: Optional[asyncpg.Pool] = None
    
    @asynccontextmanager
    async def _get_conn(self):
        """Get a connection from pool with automatic pool check"""
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")
        async with self.pool.acquire() as conn:
            yield conn
        
    async def connect(self):
        """Create connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                command_timeout=60,
                timeout=30
            )
            logger.info(f"✅ Database pool created (min={self.min_pool_size}, max={self.max_pool_size})")
            
            # Run migrations
            await self.run_migrations()
            
        except Exception as e:
            # Clean up pool if it was created before failure
            if self.pool:
                await self.pool.close()
                self.pool = None
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("🔌 Database pool closed")
    
    async def run_migrations(self):
        """Run database schema migrations"""
        try:
            schema_file = Path(__file__).parent / "schema.sql"
            
            if schema_file.exists():
                with open(schema_file, 'r') as f:
                    schema_sql = f.read()
                
                async with self._get_conn() as conn:
                    await conn.execute(schema_sql)
                    logger.info("✅ Database schema migrations completed")
            else:
                logger.warning("⚠️ Schema file not found, skipping migrations")
                
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            raise
    
    # ==================== TRADES ====================
    
    async def insert_trade(
        self,
        symbol: str,
        signal_type: str,
        entry_price: float,
        quantity: float,
        confidence_score: Optional[float] = None,
        strategy_name: Optional[str] = None,
        account_equity: Optional[float] = None,
        session_pnl: Optional[float] = None,
        order_id: Optional[str] = None
    ) -> int:
        """
        Insert a new trade
        
        Returns:
            Trade ID
        """
        async with self._get_conn() as conn:
            trade_id = await conn.fetchval(
                """
                INSERT INTO trades (
                    symbol, signal_type, entry_price, quantity,
                    confidence_score, strategy_name, account_equity,
                    session_pnl, order_id, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'OPEN')
                RETURNING id
                """,
                symbol, signal_type, entry_price, quantity,
                confidence_score, strategy_name, account_equity,
                session_pnl, order_id
            )
            logger.info(f"📝 Trade #{trade_id} inserted: {signal_type} {quantity} {symbol} @ {entry_price}")
            return trade_id
    
    async def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        commission: float = 0.0,
        duration_seconds: Optional[int] = None
    ):
        """Close an existing trade with exit data"""
        async with self._get_conn() as conn:
            await conn.execute(
                """
                UPDATE trades SET
                    exit_price = $2,
                    pnl = $3,
                    pnl_percent = $4,
                    commission = $5,
                    duration_seconds = $6,
                    status = 'CLOSED',
                    closed_at = NOW()
                WHERE id = $1
                """,
                trade_id, exit_price, pnl, pnl_percent, commission, duration_seconds
            )
            logger.info(f"✅ Trade #{trade_id} closed: PnL=${pnl:+.4f} ({pnl_percent:+.2f}%)")
    
    async def get_open_trades(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open trades, optionally filtered by symbol"""
        async with self._get_conn() as conn:
            if symbol:
                rows = await conn.fetch(
                    "SELECT * FROM trades WHERE status = 'OPEN' AND symbol = $1 ORDER BY timestamp DESC",
                    symbol
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY timestamp DESC"
                )
            return [dict(row) for row in rows]
    
    async def get_recent_trades(self, limit: int = 10, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent closed trades"""
        async with self._get_conn() as conn:
            if symbol:
                rows = await conn.fetch(
                    "SELECT * FROM trades WHERE status = 'CLOSED' AND symbol = $1 ORDER BY closed_at DESC LIMIT $2",
                    symbol, limit
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY closed_at DESC LIMIT $1",
                    limit
                )
            return [dict(row) for row in rows]
    
    # ==================== SIGNALS ====================
    
    async def insert_signal(
        self,
        symbol: str,
        signal_type: str,
        price: float,
        confidence_score: float,
        indicators: Dict[str, float],
        volatility: Optional[float] = None,
        liquidity_score: Optional[float] = None
    ) -> int:
        """
        Insert a trading signal
        
        Args:
            indicators: Dict with keys: rsi, macd, macd_signal, macd_histogram, 
                       ema_9, ema_21, ema_50, adx, atr, volume
        
        Returns:
            Signal ID
        """
        async with self._get_conn() as conn:
            signal_id = await conn.fetchval(
                """
                INSERT INTO signals (
                    symbol, signal_type, price, confidence_score,
                    rsi, macd, macd_signal, macd_histogram,
                    ema_9, ema_21, ema_50, adx, atr, volume,
                    volatility, liquidity_score
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
                )
                RETURNING id
                """,
                symbol, signal_type, price, confidence_score,
                indicators.get('rsi'), indicators.get('macd'), 
                indicators.get('macd_signal'), indicators.get('macd_histogram'),
                indicators.get('ema_9'), indicators.get('ema_21'), indicators.get('ema_50'),
                indicators.get('adx'), indicators.get('atr'), indicators.get('volume'),
                volatility, liquidity_score
            )
            return signal_id
    
    async def mark_signal_executed(self, signal_id: int, trade_id: int):
        """Mark a signal as executed with corresponding trade ID"""
        async with self._get_conn() as conn:
            await conn.execute(
                "UPDATE signals SET executed = TRUE, trade_id = $2 WHERE id = $1",
                signal_id, trade_id
            )
    
    async def mark_signal_rejected(self, signal_id: int, reason: str):
        """Mark a signal as rejected with reason"""
        async with self._get_conn() as conn:
            await conn.execute(
                "UPDATE signals SET executed = FALSE, rejection_reason = $2 WHERE id = $1",
                signal_id, reason
            )
    
    async def update_trade_exit(
        self,
        symbol: str,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        exit_reason: str = 'CLOSED',
    ) -> bool:
        """
        Update the most recent open trade for a symbol with exit data.
        
        Called when a position is closed (TP hit, SL hit, manual close).
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            pnl: Profit/loss in USD
            pnl_percent: Profit/loss percentage
            exit_reason: 'TP', 'SL', 'MANUAL', 'LIQUIDATION', etc.
            
        Returns:
            True if trade was updated, False if no open trade found
        """
        async with self._get_conn() as conn:
            # Find most recent open trade for this symbol
            trade = await conn.fetchrow(
                """
                SELECT id, timestamp FROM trades 
                WHERE symbol = $1 AND status = 'OPEN'
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                symbol
            )
            
            if not trade:
                logger.warning(f"⚠️ No open trade found for {symbol} to close")
                return False
            
            trade_id = trade['id']
            entry_time = trade['timestamp']
            
            # Calculate duration
            duration = (datetime.now(timezone.utc) - entry_time).total_seconds() if entry_time else None
            
            # Update trade
            await conn.execute(
                """
                UPDATE trades SET
                    exit_price = $2,
                    pnl = $3,
                    pnl_percent = $4,
                    status = 'CLOSED',
                    closed_at = NOW(),
                    duration_seconds = $5,
                    notes = $6
                WHERE id = $1
                """,
                trade_id, exit_price, pnl, pnl_percent, int(duration) if duration else None, exit_reason
            )
            
            logger.info(f"✅ Trade #{trade_id} closed: {symbol} @ ${exit_price} | PnL: ${pnl:+.2f} ({pnl_percent:+.2f}%) | Reason: {exit_reason}")
            return True
    
    async def get_trade_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get trading statistics for Kelly Criterion and ML training.
        
        Returns:
            Dict with win_rate, avg_win, avg_loss, total_trades, etc.
        """
        async with self._get_conn() as conn:
            # Use parameterized query to prevent SQL injection
            stats = await conn.fetchrow(
                """
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN pnl > 0 THEN 1 END) as wins,
                    COUNT(CASE WHEN pnl < 0 THEN 1 END) as losses,
                    AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                    AVG(CASE WHEN pnl < 0 THEN ABS(pnl) END) as avg_loss,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    STDDEV(pnl) as pnl_stddev
                FROM trades
                WHERE status = 'CLOSED'
                    AND closed_at >= NOW() - INTERVAL '1 day' * $1
                    AND pnl IS NOT NULL
                """,
                days  # Safe parameterized query
            )
            
            if not stats or stats['total_trades'] == 0:
                return {
                    'win_rate': 0.5,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'total_trades': 0,
                    'total_pnl': 0,
                }
            
            total = stats['total_trades']
            wins = stats['wins'] or 0
            
            return {
                'win_rate': wins / total if total > 0 else 0.5,
                'wins': wins,
                'losses': stats['losses'] or 0,
                'avg_win': float(stats['avg_win'] or 0),
                'avg_loss': float(stats['avg_loss'] or 0),
                'total_trades': total,
                'total_pnl': float(stats['total_pnl'] or 0),
                'avg_pnl': float(stats['avg_pnl'] or 0),
                'pnl_stddev': float(stats['pnl_stddev'] or 0),
            }
    
    # ==================== ML PREDICTIONS ====================
    
    async def insert_ml_prediction(
        self,
        signal_id: int,
        model_name: str,
        predicted_signal: str,
        confidence: float,
        top_features: Optional[List[tuple]] = None
    ) -> int:
        """
        Insert ML model prediction
        
        Args:
            top_features: List of (feature_name, importance) tuples
        
        Returns:
            Prediction ID
        """
        feature_names = [None, None, None]
        feature_importances = [None, None, None]
        
        if top_features:
            for i, (name, importance) in enumerate(top_features[:3]):
                feature_names[i] = name
                feature_importances[i] = importance
        
        async with self._get_conn() as conn:
            pred_id = await conn.fetchval(
                """
                INSERT INTO ml_predictions (
                    signal_id, model_name, predicted_signal, confidence,
                    feature_1_name, feature_1_importance,
                    feature_2_name, feature_2_importance,
                    feature_3_name, feature_3_importance
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
                """,
                signal_id, model_name, predicted_signal, confidence,
                feature_names[0], feature_importances[0],
                feature_names[1], feature_importances[1],
                feature_names[2], feature_importances[2]
            )
            return pred_id
    
    async def update_prediction_outcome(self, prediction_id: int, actual_signal: str, was_correct: bool):
        """Update ML prediction with actual outcome"""
        async with self._get_conn() as conn:
            await conn.execute(
                "UPDATE ml_predictions SET actual_signal = $2, was_correct = $3 WHERE id = $1",
                prediction_id, actual_signal, was_correct
            )
    
    # ==================== ACCOUNT SNAPSHOTS ====================
    
    async def insert_account_snapshot(
        self,
        equity: float,
        available_balance: float,
        margin_used: float = 0.0,
        unrealized_pnl: float = 0.0,
        session_pnl: float = 0.0,
        daily_pnl: float = 0.0,
        weekly_pnl: float = 0.0,
        total_trades: int = 0,
        winning_trades: int = 0,
        losing_trades: int = 0,
        win_rate: Optional[float] = None,
        max_drawdown_pct: Optional[float] = None,
        sharpe_ratio: Optional[float] = None
    ):
        """Insert account snapshot"""
        async with self._get_conn() as conn:
            await conn.execute(
                """
                INSERT INTO account_snapshots (
                    equity, available_balance, margin_used, unrealized_pnl,
                    session_pnl, daily_pnl, weekly_pnl,
                    total_trades, winning_trades, losing_trades, win_rate,
                    max_drawdown_pct, sharpe_ratio
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                equity, available_balance, margin_used, unrealized_pnl,
                session_pnl, daily_pnl, weekly_pnl,
                total_trades, winning_trades, losing_trades, win_rate,
                max_drawdown_pct, sharpe_ratio
            )
    
    # ==================== ANALYTICS ====================
    
    async def get_daily_performance(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily performance stats"""
        async with self._get_conn() as conn:
            rows = await conn.fetch(
                "SELECT * FROM daily_performance ORDER BY trade_date DESC LIMIT $1",
                days
            )
            return [dict(row) for row in rows]
    
    async def get_symbol_performance(self) -> List[Dict[str, Any]]:
        """Get performance by symbol"""
        async with self._get_conn() as conn:
            rows = await conn.fetch("SELECT * FROM symbol_performance ORDER BY total_pnl DESC")
            return [dict(row) for row in rows]
    
    async def get_hourly_activity(self) -> List[Dict[str, Any]]:
        """Get trading activity by hour"""
        async with self._get_conn() as conn:
            rows = await conn.fetch("SELECT * FROM hourly_activity ORDER BY hour_utc")
            return [dict(row) for row in rows]
    
    async def get_ml_model_performance(self) -> List[Dict[str, Any]]:
        """Get ML model accuracy stats"""
        async with self._get_conn() as conn:
            rows = await conn.fetch("SELECT * FROM ml_model_performance ORDER BY accuracy DESC")
            return [dict(row) for row in rows]
    
    async def get_total_stats(self) -> Dict[str, Any]:
        """Get overall trading statistics"""
        async with self._get_conn() as conn:
            row = await conn.fetchrow(
                """
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    ROUND(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::NUMERIC / NULLIF(COUNT(*), 0) * 100, 2) as win_rate,
                    ROUND(SUM(pnl)::NUMERIC, 4) as total_pnl,
                    ROUND(AVG(CASE WHEN pnl > 0 THEN pnl END)::NUMERIC, 4) as avg_win,
                    ROUND(AVG(CASE WHEN pnl < 0 THEN pnl END)::NUMERIC, 4) as avg_loss,
                    ROUND(MAX(pnl)::NUMERIC, 4) as best_trade,
                    ROUND(MIN(pnl)::NUMERIC, 4) as worst_trade
                FROM trades
                WHERE status = 'CLOSED' AND closed_at IS NOT NULL
                """
            )
            return dict(row) if row else {}
    
    async def get_trades_for_ml(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get trade data with signals for ML training
        
        Returns list of dicts with trade outcomes and signal features
        """
        async with self._get_conn() as conn:
            query = """
                SELECT 
                    t.id as trade_id,
                    t.symbol,
                    t.signal_type,
                    t.entry_price,
                    t.exit_price,
                    t.pnl,
                    t.pnl_percent,
                    t.duration_seconds,
                    s.rsi,
                    s.macd,
                    s.macd_signal,
                    s.macd_histogram,
                    s.ema_9,
                    s.ema_21,
                    s.ema_50,
                    s.adx,
                    s.atr,
                    s.volume,
                    s.volatility,
                    s.liquidity_score,
                    s.confidence_score
                FROM trades t
                JOIN signals s ON s.trade_id = t.id
                WHERE t.status = 'CLOSED' AND t.closed_at IS NOT NULL
                ORDER BY t.timestamp DESC
            """
            
            if limit:
                # Use parameterized query to prevent SQL injection
                query += " LIMIT $1"
                rows = await conn.fetch(query, limit)
            else:
                rows = await conn.fetch(query)
            return [dict(row) for row in rows]
    
    # ==================== SYSTEM EVENTS ====================
    
    async def log_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        details: Optional[Dict] = None,
        error_count: int = 1,
        consecutive_errors: int = 0
    ):
        """Log system event"""
        async with self._get_conn() as conn:
            await conn.execute(
                """
                INSERT INTO system_events (
                    event_type, severity, message, details,
                    error_count, consecutive_errors
                ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                event_type, severity, message, details,
                error_count, consecutive_errors
            )
