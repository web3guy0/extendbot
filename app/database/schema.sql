-- HyperBot PostgreSQL Database Schema
-- Optimized for analytics and ML training

-- Trades table - Main trading history
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(10) NOT NULL CHECK (signal_type IN ('BUY', 'SELL')),
    entry_price DECIMAL(18, 8) NOT NULL,
    exit_price DECIMAL(18, 8),
    quantity DECIMAL(18, 8) NOT NULL,
    pnl DECIMAL(18, 4),
    pnl_percent DECIMAL(10, 4),
    commission DECIMAL(18, 8) DEFAULT 0,
    duration_seconds INTEGER,
    status VARCHAR(20) NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED', 'CANCELLED')),
    
    -- Strategy info
    strategy_name VARCHAR(50),
    confidence_score DECIMAL(10, 4),
    
    -- Account state at trade time
    account_equity DECIMAL(18, 4),
    session_pnl DECIMAL(18, 4),
    
    -- Execution info
    order_id VARCHAR(100),
    execution_time_ms INTEGER,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);

-- Indexes for trades table
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status);
CREATE INDEX IF NOT EXISTS idx_trades_pnl ON trades (pnl DESC);

-- Signals table - All trading signals (executed or not)
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(10) NOT NULL CHECK (signal_type IN ('BUY', 'SELL')),
    price DECIMAL(18, 8) NOT NULL,
    confidence_score DECIMAL(10, 4),
    
    -- Technical indicators at signal time
    rsi DECIMAL(10, 4),
    macd DECIMAL(18, 8),
    macd_signal DECIMAL(18, 8),
    macd_histogram DECIMAL(18, 8),
    ema_9 DECIMAL(18, 8),
    ema_21 DECIMAL(18, 8),
    ema_50 DECIMAL(18, 8),
    adx DECIMAL(10, 4),
    atr DECIMAL(18, 8),
    volume DECIMAL(18, 4),
    
    -- Signal quality
    volatility DECIMAL(10, 6),
    liquidity_score DECIMAL(5, 4),
    
    -- Execution status
    executed BOOLEAN DEFAULT FALSE,
    trade_id INTEGER REFERENCES trades(id),
    rejection_reason VARCHAR(200),
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for signals table
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals (symbol);
CREATE INDEX IF NOT EXISTS idx_signals_executed ON signals (executed);
CREATE INDEX IF NOT EXISTS idx_signals_confidence ON signals (confidence_score DESC);

-- ML Predictions table - Model predictions for analysis
CREATE TABLE IF NOT EXISTS ml_predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    signal_id INTEGER REFERENCES signals(id),
    
    -- Model predictions
    model_name VARCHAR(50) NOT NULL,
    predicted_signal VARCHAR(10) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL,
    
    -- Feature importance (top 5)
    feature_1_name VARCHAR(50),
    feature_1_importance DECIMAL(10, 6),
    feature_2_name VARCHAR(50),
    feature_2_importance DECIMAL(10, 6),
    feature_3_name VARCHAR(50),
    feature_3_importance DECIMAL(10, 6),
    
    -- Actual outcome (filled after trade closes)
    actual_signal VARCHAR(10),
    was_correct BOOLEAN,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for ml_predictions table
CREATE INDEX IF NOT EXISTS idx_ml_predictions_timestamp ON ml_predictions (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_model ON ml_predictions (model_name);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_confidence ON ml_predictions (confidence DESC);

-- Account Snapshots table - Regular account state snapshots
CREATE TABLE IF NOT EXISTS account_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Account metrics
    equity DECIMAL(18, 4) NOT NULL,
    available_balance DECIMAL(18, 4) NOT NULL,
    margin_used DECIMAL(18, 4) DEFAULT 0,
    unrealized_pnl DECIMAL(18, 4) DEFAULT 0,
    
    -- Session metrics
    session_pnl DECIMAL(18, 4) DEFAULT 0,
    daily_pnl DECIMAL(18, 4) DEFAULT 0,
    weekly_pnl DECIMAL(18, 4) DEFAULT 0,
    
    -- Trading stats
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 2),
    
    -- Risk metrics
    max_drawdown_pct DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for account_snapshots table
CREATE INDEX IF NOT EXISTS idx_account_snapshots_timestamp ON account_snapshots (timestamp DESC);

-- Performance Metrics table - Aggregated daily/hourly stats
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    period_type VARCHAR(10) NOT NULL CHECK (period_type IN ('HOURLY', 'DAILY', 'WEEKLY', 'MONTHLY')),
    
    -- Trading volume
    total_trades INTEGER DEFAULT 0,
    total_volume DECIMAL(18, 4) DEFAULT 0,
    
    -- PnL metrics
    total_pnl DECIMAL(18, 4) DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 2),
    avg_win DECIMAL(18, 4),
    avg_loss DECIMAL(18, 4),
    profit_factor DECIMAL(10, 4),
    
    -- Performance metrics
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown_pct DECIMAL(10, 4),
    recovery_factor DECIMAL(10, 4),
    
    -- Best performers
    best_symbol VARCHAR(20),
    best_symbol_pnl DECIMAL(18, 4),
    worst_symbol VARCHAR(20),
    worst_symbol_pnl DECIMAL(18, 4),
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Unique constraint - one record per period
    UNIQUE(period_start, period_type)
);

-- Indexes for performance_metrics table
CREATE INDEX IF NOT EXISTS idx_performance_metrics_period ON performance_metrics (period_start DESC, period_type);

-- Market Data table - For backtesting and analysis
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    
    -- OHLCV data
    open DECIMAL(18, 8) NOT NULL,
    high DECIMAL(18, 8) NOT NULL,
    low DECIMAL(18, 8) NOT NULL,
    close DECIMAL(18, 8) NOT NULL,
    volume DECIMAL(18, 4) NOT NULL,
    
    -- Derived metrics
    volatility DECIMAL(10, 6),
    avg_spread DECIMAL(18, 8),
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Unique constraint - one record per timestamp/symbol
    UNIQUE(timestamp, symbol)
);

-- Indexes for market_data table
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp_symbol ON market_data (timestamp DESC, symbol);

-- System Events table - Bot events, errors, restarts
CREATE TABLE IF NOT EXISTS system_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    message TEXT NOT NULL,
    details JSONB,
    
    -- Error tracking
    error_count INTEGER DEFAULT 1,
    consecutive_errors INTEGER DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for system_events table
CREATE INDEX IF NOT EXISTS idx_system_events_timestamp ON system_events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_events_severity ON system_events (severity);
CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events (event_type);

-- Create useful views for analytics

-- Daily Performance View
CREATE OR REPLACE VIEW daily_performance AS
SELECT 
    DATE(timestamp) as trade_date,
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
GROUP BY DATE(timestamp)
ORDER BY trade_date DESC;

-- Symbol Performance View
CREATE OR REPLACE VIEW symbol_performance AS
SELECT 
    symbol,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    ROUND(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::NUMERIC / NULLIF(COUNT(*), 0) * 100, 2) as win_rate,
    ROUND(SUM(pnl)::NUMERIC, 4) as total_pnl,
    ROUND(AVG(pnl)::NUMERIC, 4) as avg_pnl,
    ROUND(MAX(pnl)::NUMERIC, 4) as best_trade,
    ROUND(MIN(pnl)::NUMERIC, 4) as worst_trade,
    ROUND(AVG(duration_seconds)::NUMERIC / 60, 2) as avg_duration_minutes
FROM trades
WHERE status = 'CLOSED' AND closed_at IS NOT NULL
GROUP BY symbol
ORDER BY total_pnl DESC;

-- Hourly Trading Activity View
CREATE OR REPLACE VIEW hourly_activity AS
SELECT 
    EXTRACT(HOUR FROM t.timestamp) as hour_utc,
    COUNT(*) as total_trades,
    SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    ROUND(SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END)::NUMERIC / NULLIF(COUNT(*), 0) * 100, 2) as win_rate,
    ROUND(SUM(t.pnl)::NUMERIC, 4) as total_pnl,
    ROUND(AVG(s.volatility)::NUMERIC, 6) as avg_volatility
FROM trades t
LEFT JOIN signals s ON t.id = s.trade_id
WHERE t.status = 'CLOSED' AND t.closed_at IS NOT NULL
GROUP BY EXTRACT(HOUR FROM t.timestamp)
ORDER BY hour_utc;

-- ML Model Performance View
CREATE OR REPLACE VIEW ml_model_performance AS
SELECT 
    model_name,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct_predictions,
    ROUND(SUM(CASE WHEN was_correct THEN 1 ELSE 0 END)::NUMERIC / NULLIF(COUNT(*), 0) * 100, 2) as accuracy,
    ROUND(AVG(confidence)::NUMERIC, 4) as avg_confidence,
    MAX(timestamp) as last_prediction
FROM ml_predictions
WHERE was_correct IS NOT NULL
GROUP BY model_name
ORDER BY accuracy DESC;

-- Comments for documentation
COMMENT ON TABLE trades IS 'Main trading history with entry/exit and PnL data';
COMMENT ON TABLE signals IS 'All trading signals with technical indicators';
COMMENT ON TABLE ml_predictions IS 'Machine learning model predictions and outcomes';
COMMENT ON TABLE account_snapshots IS 'Regular snapshots of account state and metrics';
COMMENT ON TABLE performance_metrics IS 'Aggregated performance statistics by time period';
COMMENT ON TABLE market_data IS 'Historical market OHLCV data for backtesting';
COMMENT ON TABLE system_events IS 'Bot events, errors, and system logs';

-- Migration: Alter confidence_score columns to handle larger values
-- This is safe to run multiple times (idempotent)
DO $$
BEGIN
    -- Alter trades.confidence_score if it exists and is too small
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'trades' AND column_name = 'confidence_score'
    ) THEN
        ALTER TABLE trades ALTER COLUMN confidence_score TYPE DECIMAL(10, 4);
    END IF;
    
    -- Alter signals.confidence_score if it exists and is too small
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'signals' AND column_name = 'confidence_score'
    ) THEN
        ALTER TABLE signals ALTER COLUMN confidence_score TYPE DECIMAL(10, 4);
    END IF;
END $$;
