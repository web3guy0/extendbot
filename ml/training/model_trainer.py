"""
Model Trainer - Train ML models with engineered features
"""

import logging
import joblib
from pathlib import Path
from typing import Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Train and save multiple ML models
    """
    
    def __init__(self, models_dir: str = 'ml/models'):
        """
        Initialize model trainer
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🎯 Model Trainer initialized")
        logger.info(f"   Models dir: {self.models_dir}")
    
    def train_all_models(self, X_train, y_train) -> Dict[str, Dict[str, Any]]:
        """
        Train all model types
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of model metrics
        """
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        metrics = {}
        
        for name, model in models.items():
            logger.info(f"🎯 Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                # Save model
                model_path = self.models_dir / f"{name}.joblib"
                joblib.dump(model, model_path)
                
                metrics[name] = {
                    'accuracy': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'path': str(model_path)
                }
                
                logger.info(f"✅ {name}: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
                
            except Exception as e:
                logger.error(f"❌ Failed to train {name}: {e}")
                metrics[name] = {'error': str(e)}
        
        return metrics
