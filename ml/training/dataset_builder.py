"""
Dataset Builder - Convert executed trades into labeled ML training data
Reads from PostgreSQL database (with JSONL fallback)
"""

import json
import logging
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Builds ML training dataset from executed trade logs
    Supports both PostgreSQL (preferred) and JSONL (fallback)
    """
    
    def __init__(self, trades_dir: str = 'data/trades', 
                 output_dir: str = 'data/model_dataset',
                 database_url: Optional[str] = None):
        """
        Initialize dataset builder
        
        Args:
            trades_dir: Directory containing trade logs (fallback)
            output_dir: Output directory for ML dataset
            database_url: PostgreSQL connection string (optional)
        """
        self.trades_dir = Path(trades_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.db = None
        
        logger.info("📊 Dataset Builder initialized")
        logger.info(f"   Trades dir: {self.trades_dir}")
        logger.info(f"   Output dir: {self.output_dir}")
        logger.info(f"   Database: {'Connected' if self.database_url else 'JSONL fallback'}")
    
    async def connect_db(self):
        """Connect to database if available"""
        if self.database_url:
            try:
                from app.database.db_manager import DatabaseManager
                self.db = DatabaseManager(self.database_url)
                await self.db.connect()
                logger.info("✅ Database connected for dataset building")
            except Exception as e:
                logger.warning(f"⚠️  Database connection failed: {e}, falling back to JSONL")
                self.db = None
    
    async def disconnect_db(self):
        """Disconnect from database"""
        if self.db:
            await self.db.disconnect()
    
    async def load_trades_from_db(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load trades from PostgreSQL database
        
        Args:
            limit: Max number of trades to load (None = all)
        
        Returns:
            List of trade records with features
        """
        if not self.db:
            return []
        
        try:
            trades = await self.db.get_trades_for_ml(limit=limit)
            logger.info(f"📥 Loaded {len(trades)} trades from database")
            return trades
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
            return []
    
    def load_trade_logs(self) -> List[Dict[str, Any]]:
        """
        Load all trade logs from JSONL files (fallback method)
        
        Returns:
            List of trade records
        """
        trades = []
        
        for log_file in self.trades_dir.glob('trades_*.jsonl'):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        trade = json.loads(line.strip())
                        trades.append(trade)
            except Exception as e:
                logger.error(f"Error loading {log_file}: {e}")
        
        logger.info(f"📥 Loaded {len(trades)} trade records from JSONL")
        return trades
    
    async def build_dataset_async(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Build complete training dataset (async version)
        Tries database first, falls back to JSONL
        
        Args:
            limit: Maximum number of trades to load
        
        Returns:
            DataFrame with features and labels
        """
        # Try database first
        await self.connect_db()
        trades = await self.load_trades_from_db(limit=limit) if self.db else []
        await self.disconnect_db()
        
        # Fall back to JSONL if database empty
        if not trades:
            logger.info("📁 Loading from JSONL files (fallback)")
            trades = self.load_trade_logs()
        
        return self._process_trades(trades)
    
    def build_dataset(self) -> pd.DataFrame:
        """
        Build complete training dataset (sync version for compatibility)
        
        Returns:
            DataFrame with features and labels
        """
        trades = self.load_trade_logs()
        return self._process_trades(trades)
    
    def _process_trades(self, trades: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process trades into DataFrame
        
        Args:
            trades: List of trade records (from DB or JSONL)
        
        Returns:
            DataFrame with features and labels
        """
        if len(trades) < 100:
            logger.warning(f"⚠️  Only {len(trades)} trades - need at least 100 for meaningful training")
        
        # Convert to DataFrame
        records = []
        
        for trade in trades:
            try:
                # Database format (direct fields)
                if 'trade_id' in trade:
                    record = {
                        'signal_type': 1 if trade.get('signal_type') == 'BUY' else -1,
                        'entry_price': trade.get('entry_price', 0),
                        'exit_price': trade.get('exit_price', 0),
                        'pnl': trade.get('pnl', 0),
                        'pnl_percent': trade.get('pnl_percent', 0),
                        'rsi': trade.get('rsi'),
                        'macd': trade.get('macd'),
                        'macd_signal': trade.get('macd_signal'),
                        'macd_histogram': trade.get('macd_histogram'),
                        'ema_9': trade.get('ema_9'),
                        'ema_21': trade.get('ema_21'),
                        'ema_50': trade.get('ema_50'),
                        'adx': trade.get('adx'),
                        'atr': trade.get('atr'),
                        'volume': trade.get('volume'),
                        'volatility': trade.get('volatility'),
                        'liquidity_score': trade.get('liquidity_score'),
                        'confidence_score': trade.get('confidence_score'),
                        'success': 1 if trade.get('pnl', 0) > 0 else 0
                    }
                else:
                    # JSONL format (nested structure)
                    signal = trade.get('signal', {})
                    market = trade.get('market_data', {})
                    account = trade.get('account_state', {})
                    result = trade.get('result', {})
                    
                    record = {
                        'timestamp': trade.get('timestamp'),
                        'signal_type': 1 if signal.get('signal_type') == 'long' else -1,
                        'entry_price': signal.get('entry_price', 0),
                        'size': signal.get('size', 0),
                        'leverage': signal.get('leverage', 1),
                        'stop_loss': signal.get('stop_loss', 0),
                        'take_profit': signal.get('take_profit', 0),
                        'momentum_pct': signal.get('momentum_pct', 0),
                        'market_price': market.get('price', 0),
                        'account_equity': account.get('equity', 0),
                        'session_pnl': account.get('session_pnl', 0),
                        'success': 1 if result.get('success') else 0
                    }
                    
                    # Calculate risk/reward ratio
                    if record['entry_price'] > 0 and record.get('stop_loss', 0) > 0:
                        record['risk_reward_ratio'] = abs(
                            (record['take_profit'] - record['entry_price']) / 
                            (record['entry_price'] - record['stop_loss'])
                        )
                    else:
                        record['risk_reward_ratio'] = 0
                
                records.append(record)
                
            except Exception as e:
                logger.error(f"Error processing trade: {e}")
                continue
        
        df = pd.DataFrame(records)
        
        logger.info(f"✅ Dataset built: {len(df)} samples")
        logger.info(f"   Features: {len(df.columns)}")
        if len(df) > 0:
            logger.info(f"   Success rate: {df['success'].mean()*100:.1f}%")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = 'training_dataset.csv'):
        """
        Save dataset to file
        
        Args:
            df: Dataset DataFrame
            filename: Output filename
        """
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"💾 Dataset saved to {output_path}")
    
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get dataset statistics"""
        return {
            'total_samples': len(df),
            'features': list(df.columns),
            'success_rate': float(df['success'].mean()),
            'long_signals': int((df['signal_type'] == 1).sum()),
            'short_signals': int((df['signal_type'] == -1).sum()),
            'avg_momentum': float(df['momentum_pct'].mean()),
            'avg_size': float(df['size'].mean()),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            }
        }


def main():
    """Main execution"""
    builder = DatasetBuilder()
    
    # Build dataset
    df = builder.build_dataset()
    
    if len(df) > 0:
        # Save dataset
        builder.save_dataset(df)
        
        # Print statistics
        stats = builder.get_statistics(df)
        print("\n" + "="*60)
        print("📊 DATASET STATISTICS")
        print("="*60)
        print(f"Total Samples: {stats['total_samples']}")
        print(f"Success Rate: {stats['success_rate']*100:.1f}%")
        print(f"Long Signals: {stats['long_signals']}")
        print(f"Short Signals: {stats['short_signals']}")
        print(f"Avg Momentum: {stats['avg_momentum']:.2f}%")
        print(f"Avg Size: {stats['avg_size']:.4f}")
        print("="*60 + "\n")
        
        if stats['total_samples'] >= 1000:
            print("✅ Ready for ML training (1,000+ samples)")
        else:
            print(f"⏳ Need {1000 - stats['total_samples']} more trades for training")
    else:
        print("❌ No trades found - run bot first to collect data")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
