"""
Feature Engineering Module for MLB Pitch Sequencing
Handles sequence analysis and advanced feature creation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Handles advanced feature engineering for pitch sequences"""
    
    def __init__(self):
        self.encoders = {}
    
    def create_sequence_features(self, df):
        """Create features based on pitch sequences"""
        logger.info("Creating sequence features...")
        
        df_seq = df.copy()
        
        # Sort by game, at-bat, and pitch number
        df_seq = df_seq.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
        
        # Create sequence features within each at-bat
        groupby_ab = df_seq.groupby(['game_pk', 'at_bat_number'])
        
        # Previous pitch features (within at-bat)
        logger.info("Creating previous pitch features...")
        for i in range(1, 4):  # Last 3 pitches
            df_seq[f'prev_pitch_type_{i}'] = groupby_ab['pitch_type_clean'].shift(i)
            df_seq[f'prev_zone_{i}'] = groupby_ab['zone_9'].shift(i)
            df_seq[f'prev_velocity_{i}'] = groupby_ab['release_speed'].shift(i)
            df_seq[f'prev_reward_{i}'] = groupby_ab['pitch_reward'].shift(i)
        
        # Velocity deltas
        logger.info("Computing velocity deltas...")
        df_seq['velocity_delta_1'] = df_seq['release_speed'] - df_seq['prev_velocity_1']
        df_seq['velocity_delta_2'] = df_seq['prev_velocity_1'] - df_seq['prev_velocity_2']
        
        # Sequence patterns
        logger.info("Creating sequence patterns...")
        df_seq['sequence_pattern'] = df_seq.apply(self._get_sequence_pattern, axis=1)
        
        # Same pitch repetition
        df_seq['repeat_pitch_1'] = (
            df_seq['pitch_type_clean'] == df_seq['prev_pitch_type_1']
        ).astype(int)
        df_seq['repeat_zone_1'] = (
            df_seq['zone_9'] == df_seq['prev_zone_1']
        ).astype(int)
        
        # Alternating patterns
        df_seq['alternating_type'] = (
            (df_seq['pitch_type_clean'] == df_seq['prev_pitch_type_2']) & 
            (df_seq['prev_pitch_type_1'] != df_seq['prev_pitch_type_2'])
        ).astype(int)
        
        logger.info(f"Sequence features created. Shape: {df_seq.shape}")
        return df_seq
    
    def prepare_modeling_data(self, df):
        """Prepare data for machine learning models"""
        logger.info("Preparing features for modeling...")
        
        # Select relevant features
        feature_columns = [
            # Count state
            'balls', 'strikes', 'outs_when_up',
            
            # Base runners
            'on_1b', 'on_2b', 'on_3b',
            
            # Game context
            'inning',
            
            # Sequence features
            'prev_pitch_type_1', 'prev_pitch_type_2',
            'prev_zone_1', 'prev_zone_2',
            'prev_velocity_1', 'velocity_delta_1',
            'prev_reward_1',
            
            # Patterns
            'repeat_pitch_1', 'repeat_zone_1',
            
            # Current pitch (target features for recommendation)
            'pitch_type_clean', 'zone_9'
        ]
        
        # Create modeling dataset
        df_model = df[feature_columns + ['pitch_reward']].copy()
        
        logger.info(f"Initial dataset size: {len(df_model):,} rows")
        
        # Handle base runner columns
        logger.info("Processing base runner features...")
        for col in ['on_1b', 'on_2b', 'on_3b']:
            if col in df_model.columns:
                df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0)
                df_model[col] = (df_model[col] != 0).astype(int)
        
        # Handle numeric columns
        logger.info("Processing numeric features...")
        numeric_cols = ['balls', 'strikes', 'outs_when_up', 'inning', 
                       'prev_velocity_1', 'velocity_delta_1', 'prev_reward_1',
                       'repeat_pitch_1', 'repeat_zone_1']
        
        for col in numeric_cols:
            if col in df_model.columns:
                if col in ['prev_velocity_1']:
                    df_model[col] = df_model[col].fillna(92.0)  # Average velocity
                elif col in ['velocity_delta_1', 'prev_reward_1']:
                    df_model[col] = df_model[col].fillna(0.0)
                elif col in ['repeat_pitch_1', 'repeat_zone_1']:
                    df_model[col] = df_model[col].fillna(0)
                else:
                    df_model[col] = df_model[col].fillna(0)
        
        # Handle categorical columns
        logger.info("Processing categorical features...")
        categorical_cols = ['prev_pitch_type_1', 'prev_pitch_type_2', 'prev_zone_1', 'prev_zone_2', 
                           'pitch_type_clean', 'zone_9']
        
        for col in categorical_cols:
            if col in df_model.columns:
                df_model[col] = df_model[col].fillna('MISSING').astype(str)
        
        # Remove rows with missing target
        initial_size = len(df_model)
        df_model = df_model.dropna(subset=['pitch_reward'])
        logger.info(f"Removed {initial_size - len(df_model):,} rows with missing targets")
        
        # Encode categorical variables
        logger.info("Encoding categorical variables...")
        encoders = {}
        for col in categorical_cols:
            if col in df_model.columns:
                encoder = LabelEncoder()
                df_model[col + '_encoded'] = encoder.fit_transform(df_model[col])
                encoders[col] = encoder
                logger.info(f"  {col}: {len(encoder.classes_)} unique values")
        
        logger.info(f"Final dataset size: {len(df_model):,} rows")
        logger.info(f"Features created: {len(df_model.columns)} columns")
        
        return df_model, encoders
    
    def _get_sequence_pattern(self, row):
        """Create sequence pattern string from previous pitches"""
        seq = []
        for i in range(1, 4):
            pitch = row.get(f'prev_pitch_type_{i}')
            if pd.notna(pitch):
                seq.append(pitch)
        return '_'.join(reversed(seq)) if seq else 'START'