"""
Modeling Module for MLB Pitch Sequencing
Handles model training, evaluation, and performance analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MLBModel:
    """Handles model training and evaluation for pitch sequencing"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_importance = {}
    
    def train_and_evaluate(self, df_model):
        """Train multiple models and evaluate performance"""
        logger.info("Starting model training and evaluation...")
        
        # Prepare features and target
        feature_cols = [col for col in df_model.columns if col.endswith('_encoded') or 
                        col in ['balls', 'strikes', 'outs_when_up', 'on_1b', 'on_2b', 'on_3b', 
                               'inning', 'prev_velocity_1', 'velocity_delta_1', 'prev_reward_1',
                               'repeat_pitch_1', 'repeat_zone_1']]
        
        X = df_model[feature_cols].fillna(0)
        logger.info(f"Feature dtypes:\n{X.dtypes}")
        y = df_model['pitch_reward']
        
        # Split data (time-based split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        
        # Store for later use
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        # Train models
        self._train_xgboost(X_train, y_train)
        self._train_random_forest(X_train, y_train)
        
        # Evaluate all models
        results = {}
        for name, model in self.models.items():
            results[name] = self._evaluate_model(model, X_test, y_test, name)
        
        self.results = results
        return results
    
    def _train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        
        xgb_model = xgb.XGBRegressor(
            enable_categorical=True,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_model
        
        # Store feature importance
        self.feature_importance['XGBoost'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("XGBoost training completed")
    
    def _train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        self.models['Random Forest'] = rf_model
        
        # Store feature importance
        self.feature_importance['Random Forest'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Random Forest training completed")
    
    def _evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        logger.info(f"Evaluating {model_name}...")
        
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'predictions': y_pred
        }
        
        logger.info(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return results
    
    def get_feature_importance(self, model_name):
        """Get feature importance for specified model"""
        if model_name in self.feature_importance:
            return self.feature_importance[model_name]
        else:
            logger.warning(f"Feature importance not available for {model_name}")
            return pd.DataFrame()
    
    def save_models(self, save_dir):
        """Save trained models to disk"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            model_file = save_path / f"{name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, model_file)
            logger.info(f"Saved {name} model to {model_file}")
        
        # Save feature importance
        for name, importance_df in self.feature_importance.items():
            importance_file = save_path / f"{name.lower().replace(' ', '_')}_importance.csv"
            importance_df.to_csv(importance_file, index=False)
            logger.info(f"Saved {name} feature importance to {importance_file}")
    
    def load_model(self, model_path):
        """Load a saved model"""
        try:
            model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return None
    
    def cross_validate_model(self, model, X, y, cv_folds=5):
        """Perform cross-validation on a model"""
        from sklearn.model_selection import cross_val_score
        
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
        
        results = {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        logger.info(f"CV R² Score: {results['mean_cv_score']:.4f} (+/- {results['std_cv_score']*2:.4f})")
        
        return results