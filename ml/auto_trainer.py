"""
Auto-Trainer - Automatic ML model retraining system
Monitors trade logs and retrains models when sufficient new data is available
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import asyncio

logger = logging.getLogger(__name__)


class AutoTrainer:
    """
    Automatic ML training system
    - Monitors trade logs for new data
    - Triggers retraining when >= 100 new trades
    - Updates models automatically
    - Sends notifications via Telegram
    """
    
    def __init__(self, min_trades_for_retrain: int = None):
        """
        Initialize auto-trainer
        
        Args:
            min_trades_for_retrain: Minimum new trades before retraining
        """
        # Get from env or use default
        if min_trades_for_retrain is None:
            min_trades_for_retrain = int(os.getenv('MIN_TRADES_FOR_RETRAIN', '100'))
        
        self.min_trades = min_trades_for_retrain
        self.trades_dir = Path(os.getenv('TRADE_LOG_PATH', 'data/trades'))
        self.model_save_path = Path(os.getenv('MODEL_SAVE_PATH', 'ml/models/saved'))
        self.last_train_time: Optional[datetime] = None
        self.last_trade_count = 0
        self.is_training = False
        
        # Ensure directories exist
        self.trades_dir.mkdir(parents=True, exist_ok=True)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Count existing trades
        if self.trades_dir.exists():
            self.last_trade_count = self._count_trades()
        
        logger.info(f"🤖 Auto-Trainer initialized")
        logger.info(f"   Min trades for retrain: {self.min_trades}")
        logger.info(f"   Trade log path: {self.trades_dir}")
        logger.info(f"   Model save path: {self.model_save_path}")
        logger.info(f"   Current trade count: {self.last_trade_count}")
    
    def _count_trades(self) -> int:
        """Count total trades in log files"""
        total = 0
        for log_file in self.trades_dir.glob('trades_*.jsonl'):
            try:
                with open(log_file, 'r') as f:
                    total += sum(1 for _ in f)
            except Exception as e:
                logger.error(f"Error counting trades in {log_file}: {e}")
        return total
    
    async def check_and_train(self, telegram_bot=None) -> bool:
        """
        Check if retraining is needed and execute
        
        Args:
            telegram_bot: Optional Telegram bot for notifications
            
        Returns:
            True if training was triggered, False otherwise
        """
        if self.is_training:
            logger.info("⏳ Training already in progress, skipping")
            return False
        
        # Count current trades
        current_count = self._count_trades()
        new_trades = current_count - self.last_trade_count
        
        if new_trades < self.min_trades:
            logger.info(f"📊 {new_trades} new trades (need {self.min_trades} for retrain)")
            return False
        
        # Trigger training
        logger.info(f"🚀 Triggering auto-retraining ({new_trades} new trades)")
        
        if telegram_bot:
            await telegram_bot.send_message(
                f"🤖 *AUTO-TRAINING STARTED*\n\n"
                f"New trades: {new_trades}\n"
                f"Total trades: {current_count}\n"
                f"Training in progress..."
            )
        
        self.is_training = True
        success = await self._run_training(telegram_bot)
        self.is_training = False
        
        if success:
            self.last_trade_count = current_count
            self.last_train_time = datetime.now(timezone.utc)
        
        return success
    
    async def _run_training(self, telegram_bot=None) -> bool:
        """
        Execute the training pipeline
        
        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = datetime.now(timezone.utc)
            
            # Step 1: Build dataset
            logger.info("📊 Step 1/3: Building dataset...")
            from ml.training.dataset_builder import DatasetBuilder
            builder = DatasetBuilder()
            df = builder.build_dataset()
            
            if len(df) < 100:
                logger.warning(f"⚠️  Only {len(df)} samples - need at least 100")
                return False
            
            logger.info(f"✅ Dataset built: {len(df)} samples")
            
            # Step 2: Engineer features
            logger.info("🔧 Step 2/3: Engineering features...")
            from ml.training.feature_engineering import FeatureEngineer
            engineer = FeatureEngineer()
            X_train, y_train = engineer.prepare_features(df)
            
            logger.info(f"✅ Features prepared: {X_train.shape}")
            
            # Step 3: Train models
            logger.info("🎯 Step 3/3: Training models...")
            from ml.training.model_trainer import ModelTrainer
            trainer = ModelTrainer()
            metrics = trainer.train_all_models(X_train, y_train)
            
            # Calculate duration
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Log results
            logger.info(f"✅ Training complete in {duration:.1f}s")
            for model_name, model_metrics in metrics.items():
                accuracy = model_metrics.get('accuracy', 0)
                logger.info(f"   {model_name}: {accuracy:.2%} accuracy")
            
            # Send Telegram notification
            if telegram_bot:
                best_model = max(metrics.items(), key=lambda x: x[1].get('accuracy', 0))
                await telegram_bot.send_message(
                    f"✅ *AUTO-TRAINING COMPLETE*\n\n"
                    f"⏱ Duration: {duration:.1f}s\n"
                    f"📊 Samples: {len(df)}\n"
                    f"🎯 Best Model: {best_model[0]}\n"
                    f"📈 Accuracy: {best_model[1].get('accuracy', 0):.2%}\n\n"
                    f"Models updated and ready to use!"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
            
            if telegram_bot:
                await telegram_bot.send_message(
                    f"❌ *AUTO-TRAINING FAILED*\n\n"
                    f"Error: {str(e)[:200]}\n\n"
                    f"Check logs for details."
                )
            
            return False
    
    async def schedule_daily_check(self, telegram_bot=None):
        """
        Run daily training checks (background task)
        
        Args:
            telegram_bot: Optional Telegram bot for notifications
        """
        logger.info("⏰ Starting daily training scheduler")
        
        while True:
            try:
                # Wait 24 hours
                await asyncio.sleep(24 * 60 * 60)
                
                logger.info("⏰ Daily training check triggered")
                await self.check_and_train(telegram_bot)
                
            except Exception as e:
                logger.error(f"Error in training scheduler: {e}", exc_info=True)
                await asyncio.sleep(60 * 60)  # Wait 1 hour on error
