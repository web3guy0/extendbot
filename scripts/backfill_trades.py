#!/usr/bin/env python3
"""
Backfill historical trades from JSONL files to PostgreSQL database.
This enables ML/AI training on historical trade data.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from app.database.db_manager import DatabaseManager


async def backfill_trades():
    """Parse JSONL files and insert trades into PostgreSQL."""
    
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not set in environment")
        return
    
    # Initialize database
    db = DatabaseManager(database_url)
    await db.connect()
    
    trades_dir = Path(__file__).parent.parent / "data" / "trades"
    jsonl_files = sorted(trades_dir.glob("trades_*.jsonl"))
    
    if not jsonl_files:
        print("❌ No JSONL files found in data/trades/")
        return
    
    print(f"📂 Found {len(jsonl_files)} JSONL files")
    
    total_trades = 0
    total_signals = 0
    failed = 0
    
    for jsonl_file in jsonl_files:
        if jsonl_file.stat().st_size == 0:
            print(f"⏭️  Skipping empty file: {jsonl_file.name}")
            continue
            
        print(f"\n📄 Processing: {jsonl_file.name}")
        
        with open(jsonl_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  ⚠️  Line {line_num}: Invalid JSON - {e}")
                    failed += 1
                    continue
                
                signal = record.get('signal', {})
                result = record.get('result', {})
                market_data = record.get('market_data', {})
                account_state = record.get('account_state', {})
                
                # Skip if not a successful trade
                if not result.get('success'):
                    continue
                
                # Extract data
                symbol = signal.get('symbol', 'UNKNOWN')
                signal_type = signal.get('signal_type') or signal.get('side', 'unknown')
                
                # Normalize signal_type to BUY/SELL
                if signal_type.lower() in ['long', 'buy']:
                    signal_type = 'BUY'
                elif signal_type.lower() in ['short', 'sell']:
                    signal_type = 'SELL'
                else:
                    signal_type = signal_type.upper()
                
                # Build indicators from signal data
                indicators = {
                    'momentum_pct': signal.get('momentum_pct'),
                    'volume_ratio': signal.get('volume_ratio'),
                    'breakout_level': signal.get('breakout_level'),
                    'breakout_pct': signal.get('breakout_pct'),
                }
                
                # Get actual fill price from result
                entry_price = signal.get('entry_price', 0)
                
                # Try to get actual fill price from raw response
                raw = result.get('raw', {})
                response = raw.get('response', {})
                data = response.get('data', {})
                statuses = data.get('statuses', [])
                if statuses and isinstance(statuses[0], dict):
                    filled = statuses[0].get('filled', {})
                    if filled.get('avgPx'):
                        entry_price = float(filled['avgPx'])
                
                try:
                    # Insert signal first
                    signal_id = await db.insert_signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        price=float(entry_price),
                        confidence_score=float(signal.get('confidence', 0.5)),
                        indicators=indicators,
                        volatility=market_data.get('volatility'),
                        liquidity_score=market_data.get('liquidity_score')
                    )
                    total_signals += 1
                    
                    # Extract order_id from raw response
                    order_id = None
                    if statuses and isinstance(statuses[0], dict):
                        filled = statuses[0].get('filled', {})
                        order_id = str(filled.get('oid', 'unknown'))
                    
                    # Insert trade
                    trade_id = await db.insert_trade(
                        symbol=symbol,
                        signal_type=signal_type,
                        entry_price=float(entry_price),
                        quantity=float(signal.get('size', 0)),
                        confidence_score=float(signal.get('confidence', 0.5)),
                        strategy_name=signal.get('strategy', 'Unknown'),
                        account_equity=float(account_state.get('equity', 0)),
                        session_pnl=float(account_state.get('session_pnl', 0)),
                        order_id=order_id
                    )
                    
                    # Link signal to trade
                    await db.mark_signal_executed(signal_id, trade_id)
                    
                    total_trades += 1
                    
                    if total_trades % 10 == 0:
                        print(f"  ✅ Imported {total_trades} trades...")
                        
                except Exception as e:
                    print(f"  ❌ Line {line_num}: Database error - {e}")
                    failed += 1
    
    # Close database connection
    await db.disconnect()
    
    print(f"\n{'='*50}")
    print(f"✅ Backfill Complete!")
    print(f"   📊 Signals imported: {total_signals}")
    print(f"   📈 Trades imported: {total_trades}")
    print(f"   ❌ Failed: {failed}")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(backfill_trades())
