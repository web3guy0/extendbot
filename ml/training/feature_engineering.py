"""
Feature Engineering - Create advanced features for ML training
Extracts momentum, orderflow, volatility, and trend features
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Advanced feature engineering for trading ML models
    """
    
    def __init__(self):
        logger.info("🔧 Feature Engineer initialized")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features
        
        Args:
            df: Raw dataset
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("🔬 Engineering features...")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').copy()
        
        # Price-based features
        df = self._add_momentum_features(df)
        df = self._add_volatility_features(df)
        df = self._add_trend_features(df)
        
        # Risk features
        df = self._add_risk_features(df)
        
        # Time features
        df = self._add_time_features(df)
        
        # Fill any NaN values
        df = df.fillna(0)
        
        logger.info(f"✅ Features engineered: {len(df.columns)} total features")
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features"""
        # Price momentum over different windows
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df['entry_price'].pct_change(window) * 100
            df[f'momentum_{window}_abs'] = df[f'momentum_{window}'].abs()
        
        # Momentum acceleration
        df['momentum_acceleration'] = df['momentum_10'].diff()
        
        # Momentum strength
        df['momentum_strength'] = df['momentum_pct'].abs()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        # Rolling volatility (standard deviation of returns)
        for window in [5, 10, 20]:
            returns = df['entry_price'].pct_change()
            df[f'volatility_{window}'] = returns.rolling(window).std() * 100
        
        # Volatility regime (high/low)
        df['volatility_regime'] = (df['volatility_10'] > df['volatility_10'].median()).astype(int)
        
        # Price range
        df['price_range_10'] = df['entry_price'].rolling(10).max() - df['entry_price'].rolling(10).min()
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based features"""
        # Simple moving averages
        for window in [5, 10, 20]:
            df[f'sma_{window}'] = df['entry_price'].rolling(window).mean()
        
        # Price vs SMA (trend strength)
        df['price_vs_sma_10'] = ((df['entry_price'] - df['sma_10']) / df['sma_10']) * 100
        df['price_vs_sma_20'] = ((df['entry_price'] - df['sma_20']) / df['sma_20']) * 100
        
        # SMA crossover signals
        df['sma_5_10_cross'] = (df['sma_5'] > df['sma_10']).astype(int)
        df['sma_10_20_cross'] = (df['sma_10'] > df['sma_20']).astype(int)
        
        # Trend consistency (how many consecutive up/down moves)
        price_change = df['entry_price'].diff()
        df['trend_consistency'] = (price_change > 0).rolling(10).sum()
        
        return df
    
    def _add_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk-related features"""
        # Position size as % of equity
        df['position_pct_equity'] = (df['size'] * df['entry_price']) / df['account_equity'] * 100
        
        # Risk per trade
        df['risk_per_trade'] = abs(df['entry_price'] - df['stop_loss']) * df['size']
        df['risk_pct'] = (df['risk_per_trade'] / df['account_equity']) * 100
        
        # Reward per trade
        df['reward_per_trade'] = abs(df['take_profit'] - df['entry_price']) * df['size']
        df['reward_pct'] = (df['reward_per_trade'] / df['account_equity']) * 100
        
        # Risk-reward ratio
        df['risk_reward_ratio'] = df['reward_per_trade'] / (df['risk_per_trade'] + 0.001)
        
        # Leverage risk
        df['leverage_risk'] = df['leverage'] * df['risk_pct']
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'])
        
        # Hour of day
        df['hour'] = df['datetime'].dt.hour
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Session (Asian/European/US)
        df['session'] = df['hour'].apply(self._get_trading_session)
        
        return df
    
    @staticmethod
    def _get_trading_session(hour: int) -> int:
        """
        Get trading session from hour
        0 = Asian (00:00-08:00 UTC)
        1 = European (08:00-16:00 UTC)
        2 = US (16:00-24:00 UTC)
        """
        if 0 <= hour < 8:
            return 0  # Asian
        elif 8 <= hour < 16:
            return 1  # European
        else:
            return 2  # US
    
    def get_feature_importance_names(self) -> list[str]:
        """Get list of engineered feature names"""
        return [
            # Momentum
            'momentum_5', 'momentum_10', 'momentum_20',
            'momentum_5_abs', 'momentum_10_abs', 'momentum_20_abs',
            'momentum_acceleration', 'momentum_strength',
            
            # Volatility
            'volatility_5', 'volatility_10', 'volatility_20',
            'volatility_regime', 'price_range_10',
            
            # Trend
            'sma_5', 'sma_10', 'sma_20',
            'price_vs_sma_10', 'price_vs_sma_20',
            'sma_5_10_cross', 'sma_10_20_cross',
            'trend_consistency',
            
            # Risk
            'position_pct_equity', 'risk_per_trade', 'risk_pct',
            'reward_per_trade', 'reward_pct', 'risk_reward_ratio',
            'leverage_risk',
            
            # Time
            'hour', 'day_of_week', 'is_weekend', 'session'
        ]


def main():
    """Test feature engineering"""
    import sys
    sys.path.append('../..')
    from ml.training.dataset_builder import DatasetBuilder
    
    # Load dataset
    builder = DatasetBuilder()
    df = builder.build_dataset()
    
    if len(df) == 0:
        print("❌ No data found")
        return
    
    # Engineer features
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df)
    
    # Save engineered dataset
    output_path = builder.output_dir / 'training_dataset_engineered.csv'
    df_engineered.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("🔬 FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Original features: {len(builder.build_dataset().columns)}")
    print(f"Engineered features: {len(df_engineered.columns)}")
    print(f"Samples: {len(df_engineered)}")
    print(f"Output: {output_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
