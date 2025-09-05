"""
Model Monitoring and Performance Tracking for MLB Pitch Sequencing
Handles drift detection, performance monitoring, and model validation
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

class ModelMonitor:
    """Monitor model performance and detect drift over time"""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.current_metrics = {}
        self.alerts = []
        self.performance_history = []
    
    def set_baseline(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    feature_distributions: pd.DataFrame) -> None:
        """
        Set baseline performance metrics for comparison
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            feature_distributions: DataFrame with feature values for distribution analysis
        """
        logger.info("Setting baseline performance metrics...")
        
        self.baseline_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'feature_means': {col: feature_distributions[col].mean() 
                            for col in feature_distributions.columns},
            'feature_stds': {col: feature_distributions[col].std() 
                           for col in feature_distributions.columns},
            'prediction_mean': y_pred.mean(),
            'prediction_std': y_pred.std(),
            'timestamp': pd.Timestamp.now(),
            'sample_size': len(y_true)
        }
        
        logger.info(f"Baseline set - RMSE: {self.baseline_metrics['rmse']:.4f}, "
                   f"R²: {self.baseline_metrics['r2']:.4f}")
    
    def evaluate_current(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        feature_distributions: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate current model performance
        
        Args:
            y_true: True target values  
            y_pred: Predicted values
            feature_distributions: DataFrame with current feature values
            
        Returns:
            Dictionary with current performance metrics
        """
        logger.info("Evaluating current performance...")
        
        self.current_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'feature_means': {col: feature_distributions[col].mean() 
                            for col in feature_distributions.columns if col in feature_distributions.columns},
            'feature_stds': {col: feature_distributions[col].std() 
                           for col in feature_distributions.columns if col in feature_distributions.columns},
            'prediction_mean': y_pred.mean(),
            'prediction_std': y_pred.std(),
            'timestamp': pd.Timestamp.now(),
            'sample_size': len(y_true)
        }
        
        # Store in history
        self.performance_history.append(self.current_metrics.copy())
        
        logger.info(f"Current performance - RMSE: {self.current_metrics['rmse']:.4f}, "
                   f"R²: {self.current_metrics['r2']:.4f}")
        
        return self.current_metrics
    
    def detect_drift(self, threshold: float = 0.1) -> List[str]:
        """
        Detect performance and data drift
        
        Args:
            threshold: Percentage threshold for drift detection
            
        Returns:
            List of drift alerts
        """
        if not self.baseline_metrics or not self.current_metrics:
            logger.warning("Cannot detect drift - missing baseline or current metrics")
            return []
        
        alerts = []
        
        # Performance drift detection
        performance_alerts = self._detect_performance_drift(threshold)
        alerts.extend(performance_alerts)
        
        # Data drift detection
        data_alerts = self._detect_data_drift(threshold * 2)  # Higher threshold for data drift
        alerts.extend(data_alerts)
        
        # Prediction distribution drift
        prediction_alerts = self._detect_prediction_drift(threshold)
        alerts.extend(prediction_alerts)
        
        self.alerts = alerts
        
        if alerts:
            logger.warning(f"Detected {len(alerts)} drift alerts")
        else:
            logger.info("No drift detected")
        
        return alerts
    
    def _detect_performance_drift(self, threshold: float) -> List[str]:
        """Detect performance metric drift"""
        alerts = []
        
        # RMSE drift
        rmse_change = abs(self.current_metrics['rmse'] - self.baseline_metrics['rmse'])
        rmse_pct_change = rmse_change / self.baseline_metrics['rmse']
        if rmse_pct_change > threshold:
            alerts.append(f"RMSE degraded by {rmse_pct_change:.1%} "
                         f"({self.baseline_metrics['rmse']:.4f} → {self.current_metrics['rmse']:.4f})")
        
        # R² drift
        if self.baseline_metrics['r2'] != 0:
            r2_change = abs(self.current_metrics['r2'] - self.baseline_metrics['r2'])
            r2_pct_change = r2_change / abs(self.baseline_metrics['r2'])
            if r2_pct_change > threshold:
                direction = "improved" if self.current_metrics['r2'] > self.baseline_metrics['r2'] else "degraded"
                alerts.append(f"R² {direction} by {r2_pct_change:.1%} "
                             f"({self.baseline_metrics['r2']:.4f} → {self.current_metrics['r2']:.4f})")
        
        # MAE drift
        mae_change = abs(self.current_metrics['mae'] - self.baseline_metrics['mae'])
        mae_pct_change = mae_change / self.baseline_metrics['mae']
        if mae_pct_change > threshold:
            alerts.append(f"MAE changed by {mae_pct_change:.1%}")
        
        return alerts
    
    def _detect_data_drift(self, threshold: float) -> List[str]:
        """Detect feature distribution drift"""
        alerts = []
        
        # Check feature mean drift
        for feature in self.baseline_metrics['feature_means']:
            if feature in self.current_metrics['feature_means']:
                baseline_mean = self.baseline_metrics['feature_means'][feature]
                current_mean = self.current_metrics['feature_means'][feature]
                
                if baseline_mean != 0:
                    drift = abs(current_mean - baseline_mean) / abs(baseline_mean)
                    if drift > threshold:
                        alerts.append(f"Feature drift in {feature}: mean changed by {drift:.1%}")
        
        # Check feature standard deviation drift
        for feature in self.baseline_metrics['feature_stds']:
            if feature in self.current_metrics['feature_stds']:
                baseline_std = self.baseline_metrics['feature_stds'][feature]
                current_std = self.current_metrics['feature_stds'][feature]
                
                if baseline_std != 0:
                    drift = abs(current_std - baseline_std) / baseline_std
                    if drift > threshold:
                        alerts.append(f"Feature variability drift in {feature}: std changed by {drift:.1%}")
        
        return alerts
    
    def _detect_prediction_drift(self, threshold: float) -> List[str]:
        """Detect prediction distribution drift"""
        alerts = []
        
        # Prediction mean drift
        pred_mean_change = abs(self.current_metrics['prediction_mean'] - 
                              self.baseline_metrics['prediction_mean'])
        if self.baseline_metrics['prediction_mean'] != 0:
            pred_mean_pct_change = pred_mean_change / abs(self.baseline_metrics['prediction_mean'])
            if pred_mean_pct_change > threshold:
                alerts.append(f"Prediction distribution shift: mean changed by {pred_mean_pct_change:.1%}")
        
        # Prediction standard deviation drift
        pred_std_change = abs(self.current_metrics['prediction_std'] - 
                             self.baseline_metrics['prediction_std'])
        if self.baseline_metrics['prediction_std'] != 0:
            pred_std_pct_change = pred_std_change / self.baseline_metrics['prediction_std']
            if pred_std_pct_change > threshold:
                alerts.append(f"Prediction variability changed by {pred_std_pct_change:.1%}")
        
        return alerts
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.baseline_metrics or not self.current_metrics:
            return {'error': 'Missing baseline or current metrics'}
        
        summary = {
            'baseline': {
                'rmse': self.baseline_metrics['rmse'],
                'mae': self.baseline_metrics['mae'],
                'r2': self.baseline_metrics['r2'],
                'timestamp': self.baseline_metrics['timestamp'],
                'sample_size': self.baseline_metrics['sample_size']
            },
            'current': {
                'rmse': self.current_metrics['rmse'],
                'mae': self.current_metrics['mae'],
                'r2': self.current_metrics['r2'],
                'timestamp': self.current_metrics['timestamp'],
                'sample_size': self.current_metrics['sample_size']
            },
            'changes': {
                'rmse_change': self.current_metrics['rmse'] - self.baseline_metrics['rmse'],
                'mae_change': self.current_metrics['mae'] - self.baseline_metrics['mae'],
                'r2_change': self.current_metrics['r2'] - self.baseline_metrics['r2']
            },
            'alerts': self.alerts,
            'total_evaluations': len(self.performance_history)
        }
        
        return summary
    
    def generate_monitoring_report(self) -> str:
        """Generate a text report of monitoring results"""
        if not self.baseline_metrics or not self.current_metrics:
            return "Monitoring report unavailable - missing metrics"
        
        report = [
            "=== MODEL MONITORING REPORT ===",
            f"Report generated: {pd.Timestamp.now()}",
            "",
            "PERFORMANCE COMPARISON:",
            f"  Baseline RMSE: {self.baseline_metrics['rmse']:.4f}",
            f"  Current RMSE:  {self.current_metrics['rmse']:.4f}",
            f"  Change:        {self.current_metrics['rmse'] - self.baseline_metrics['rmse']:+.4f}",
            "",
            f"  Baseline R²:   {self.baseline_metrics['r2']:.4f}",
            f"  Current R²:    {self.current_metrics['r2']:.4f}",
            f"  Change:        {self.current_metrics['r2'] - self.baseline_metrics['r2']:+.4f}",
            "",
            "DRIFT ALERTS:"
        ]
        
        if self.alerts:
            for alert in self.alerts:
                report.append(f"  ⚠️  {alert}")
        else:
            report.append("  ✅ No drift detected")
        
        report.extend([
            "",
            f"EVALUATION HISTORY: {len(self.performance_history)} evaluations",
            "=== END REPORT ==="
        ])
        
        return "\n".join(report)
    
    def recommend_actions(self) -> List[str]:
        """Recommend actions based on monitoring results"""
        recommendations = []
        
        if not self.alerts:
            recommendations.append("✅ Model performing within expected parameters")
            return recommendations
        
        # Performance-based recommendations
        for alert in self.alerts:
            if "RMSE degraded" in alert:
                recommendations.append("🔄 Consider model retraining with recent data")
                recommendations.append("📊 Investigate data quality issues")
            elif "Feature drift" in alert:
                recommendations.append("🔍 Analyze feature distributions for changes")
                recommendations.append("⚙️ Consider feature preprocessing updates")
            elif "Prediction distribution" in alert:
                recommendations.append("🎯 Review prediction calibration")
                recommendations.append("📈 Validate against ground truth data")
        
        # General recommendations
        if len(self.alerts) > 3:
            recommendations.append("🚨 Multiple alerts detected - prioritize model refresh")
        
        return recommendations
    
    def export_metrics_history(self) -> pd.DataFrame:
        """Export performance history as DataFrame"""
        if not self.performance_history:
            return pd.DataFrame()
        
        # Flatten metrics for DataFrame
        flattened_history = []
        for i, metrics in enumerate(self.performance_history):
            record = {
                'evaluation_id': i,
                'timestamp': metrics['timestamp'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'sample_size': metrics['sample_size'],
                'prediction_mean': metrics['prediction_mean'],
                'prediction_std': metrics['prediction_std']
            }
            flattened_history.append(record)
        
        return pd.DataFrame(flattened_history)