#!/usr/bin/env python3
"""
MLB Pitch Sequencing Optimization - Main Execution Script
Advanced Baseball Analytics & Predictive Modeling

This script orchestrates the complete analysis pipeline for MLB pitch sequencing optimization.
Run this file to execute the full project workflow.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.modeling import MLBModel
from src.recommendation_engine import PitchRecommendationEngine
from src.simulation import GameSimulator
from src.monitoring import ModelMonitor

def main():
    """Main execution function for the MLB pitch sequencing project"""
    
    print("=" * 60)
    print("MLB PITCH SEQUENCING OPTIMIZATION")
    print("Advanced Baseball Analytics & Predictive Modeling")
    print("=" * 60)
    
    # Create necessary directories
    for dir_name in ['data', 'outputs', 'models']:
        Path(dir_name).mkdir(exist_ok=True)
    
    try:
        # Step 1: Data Acquisition and Processing
        print("\n STEP 1: Data Acquisition & Processing")
        print("-" * 50)
        
        processor = DataProcessor()
        df_raw = processor.get_statcast_data(year=2023)
        df_clean = processor.clean_and_engineer_basic_features(df_raw)
        
        logger.info(f"Processed {len(df_clean):,} pitches")
        
        # Step 2: Advanced Feature Engineering
        print("\n STEP 2: Feature Engineering")
        print("-" * 50)
        
        engineer = FeatureEngineer()
        df_features = engineer.create_sequence_features(df_clean)
        df_model_ready, encoders = engineer.prepare_modeling_data(df_features)
        
        logger.info(f"Created {len(df_model_ready.columns)} features")
        
        # Step 3: Model Training and Evaluation
        print("\n STEP 3: Model Training & Evaluation")
        print("-" * 50)
        
        model_trainer = MLBModel()
        results = model_trainer.train_and_evaluate(df_model_ready)
        
        best_model_name = max(results, key=lambda x: results[x]['R²'])
        best_model = model_trainer.models[best_model_name]
        
        print(f"Best model: {best_model_name} (R² = {results[best_model_name]['R²']:.3f})")
        
        # Step 4: Recommendation System
        print("\n STEP 4: Pitch Recommendation System")
        print("-" * 50)
        
        available_pitches = df_clean['pitch_type_clean'].unique()
        available_zones = df_clean['zone_9'].unique()
        
        engine = PitchRecommendationEngine(
            model=best_model,
            encoders=encoders,
            pitch_types=available_pitches,
            zones=available_zones
        )
        
        # Example recommendation
        example_state = {
            'balls': 1, 'strikes': 2, 'outs_when_up': 1,
            'on_1b': 1, 'on_2b': 0, 'on_3b': 0, 'inning': 7,
            'prev_pitch_type_1': 'FF', 'prev_zone_1': 'Z11',
            'prev_velocity_1': 94.5
        }
        
        recommendations = engine.recommend_next_pitch(example_state)
        
        print("Example Recommendation (1-2 count, runner on first):")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"{i}. {rec['pitch_type']} in zone {rec['zone']} "
                  f"(reward: {rec['expected_reward']:.3f})")
        
        # Step 5: Game Simulation
        print("\n STEP 5: Game Simulation")
        print("-" * 50)
        
        simulator = GameSimulator(engine)
        
        # Simulate a few at-bats
        simulation_states = [
            {'balls': 0, 'strikes': 0, 'outs_when_up': 0, 
             'on_1b': 0, 'on_2b': 0, 'on_3b': 0, 'inning': 1},
            {'balls': 2, 'strikes': 2, 'outs_when_up': 2, 
             'on_1b': 0, 'on_2b': 1, 'on_3b': 0, 'inning': 8}
        ]
        
        for i, state in enumerate(simulation_states, 1):
            print(f"\nAt-Bat Simulation #{i}:")
            result = simulator.simulate_at_bat(state)
            print(f"Result: {result['result']} ({result['pitch_count']} pitches)")
        
        # Step 6: Model Monitoring Setup
        print("\n STEP 6: Model Monitoring")
        print("-" * 50)
        
        monitor = ModelMonitor()
        
        # Get features and targets for monitoring
        """
        feature_cols = [col for col in df_model_ready.columns 
                       if col not in ['pitch_reward']]
        """
        feature_cols = [
            col for col in df_model_ready.columns
            if col not in ['pitch_reward'] and pd.api.types.is_numeric_dtype(df_model_ready[col])
        ]

        X = df_model_ready[feature_cols].fillna(0)
        y = df_model_ready['pitch_reward']
        
        # Split for monitoring demo
        split_idx = int(len(X) * 0.8)
        X_baseline, X_current = X[:split_idx], X[split_idx:]
        y_baseline, y_current = y[:split_idx], y[split_idx:]
        
        y_pred_baseline = best_model.predict(X_baseline)
        y_pred_current = best_model.predict(X_current)
        
        monitor.set_baseline(y_baseline, y_pred_baseline, X_baseline)
        monitor.evaluate_current(y_current, y_pred_current, X_current)
        
        drift_alerts = monitor.detect_drift()
        if drift_alerts:
            print(" Model drift alerts:")
            for alert in drift_alerts:
                print(f"  - {alert}")
        else:
            print(" No model drift detected")
        
        # Step 7: Results Export
        print("\n STEP 7: Exporting Results")
        print("-" * 50)
        
        # Save models and results
        model_trainer.save_models('models/')
        
        # Export analysis results
        results_df = pd.DataFrame({
            'model': list(results.keys()),
            'rmse': [results[k]['RMSE'] for k in results.keys()],
            'r2': [results[k]['R²'] for k in results.keys()]
        })
        
        results_df.to_csv('outputs/model_performance.csv', index=False)
        
        # Feature importance
        feature_importance = model_trainer.get_feature_importance(best_model_name)
        feature_importance.to_csv('outputs/feature_importance.csv', index=False)
        
        # Sample predictions
        sample_data = df_features[['pitch_type_clean', 'zone_9', 'count_state', 
                                  'pitch_reward']].head(1000)
        sample_data.to_csv('outputs/sample_analysis_data.csv', index=False)
        
        print("Results exported to outputs/ directory")
        
        # Final Summary
        print("\n🎯 PROJECT SUMMARY")
        print("=" * 50)
        
        total_pitches = len(df_features)
        best_r2 = results[best_model_name]['R²']
        
        print(f"""
DATA ANALYSIS:
   • Processed {total_pitches:,} pitches
   • Engineered {len(feature_cols)} features
   • Built comprehensive recommendation system

MODEL PERFORMANCE:
   • Best model: {best_model_name}
   • R² Score: {best_r2:.3f}
   • Production-ready deployment framework

BUSINESS IMPACT:
   • Real-time pitch recommendations
   • Competitive advantage through data science
   • Research-grade statistical analysis
        """)
        
        print("=" * 50)
        print(" MLB PITCH SEQUENCING OPTIMIZATION COMPLETE!")
        print("   Portfolio-ready baseball analytics solution")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    main()