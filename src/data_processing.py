"""
Data Processing Module for MLB Pitch Sequencing
Handles data acquisition, cleaning, and basic preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles all data processing operations"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def get_statcast_data(self, year=2023, cache=True):
        """Download and cache Statcast data"""
        cache_file = self.data_dir / f"statcast_{year}.parquet"
        
        if cache and cache_file.exists():
            logger.info(f"Loading cached data for {year}...")
            return pd.read_parquet(cache_file)
        
        logger.info(f"Attempting to download Statcast data for {year}...")
        
        try:
            # Try to import pybaseball
            from pybaseball import statcast
            
            start_date = f"{year}-03-30"
            end_date = f"{year}-10-31"
            
            df = statcast(start_dt=start_date, end_dt=end_date)
            
            if cache:
                df.to_parquet(cache_file, index=False)
                logger.info(f"Data cached to {cache_file}")
            
            return df
            
        except Exception as e:
            logger.warning(f"Could not download real data: {e}")
            logger.info("Generating sample data for demonstration...")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create realistic sample data for demonstration"""
        np.random.seed(42)
        n_pitches = 10000
        
        logger.info(f"Creating {n_pitches:,} sample pitches...")
        
        # Base attributes
        games = np.random.randint(1, 100, n_pitches)
        at_bats = np.random.randint(1, 50, n_pitches)
        pitch_numbers = np.random.randint(1, 12, n_pitches)
        
        # Count states
        balls = np.random.choice([0, 1, 2, 3], n_pitches, p=[0.3, 0.3, 0.25, 0.15])
        strikes = np.random.choice([0, 1, 2], n_pitches, p=[0.4, 0.35, 0.25])
        
        # Pitch types with realistic distribution
        pitch_types = np.random.choice(
            ['FF', 'SL', 'CH', 'CU', 'SI', 'FC'], 
            n_pitches, 
            p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
        )
        
        # Locations (plate coordinates)
        plate_x = np.random.normal(0, 1.2, n_pitches)
        plate_z = np.random.normal(2.5, 0.8, n_pitches)
        
        # Strike zone bounds
        sz_top = np.random.normal(3.4, 0.2, n_pitches)
        sz_bot = np.random.normal(1.6, 0.2, n_pitches)
        
        # Outcomes
        outcomes = np.random.choice(
            ['ball', 'called_strike', 'swinging_strike', 'foul', 'hit_into_play'],
            n_pitches,
            p=[0.35, 0.15, 0.1, 0.25, 0.15]
        )
        
        # Velocities
        velocities = np.random.normal(92, 6, n_pitches)
        
        # Player IDs
        pitcher_ids = np.random.randint(100000, 700000, n_pitches)
        batter_ids = np.random.randint(100000, 700000, n_pitches)
        
        sample_df = pd.DataFrame({
            'game_pk': games,
            'at_bat_number': at_bats,
            'pitch_number': pitch_numbers,
            'balls': balls,
            'strikes': strikes,
            'pitch_type': pitch_types,
            'plate_x': plate_x,
            'plate_z': plate_z,
            'sz_top': sz_top,
            'sz_bot': sz_bot,
            'description': outcomes,
            'release_speed': velocities,
            'pitcher': pitcher_ids,
            'batter': batter_ids,
            'outs_when_up': np.random.choice([0, 1, 2], n_pitches),
            'inning': np.random.randint(1, 10, n_pitches),
            'on_1b': np.random.choice([0, 1], n_pitches, p=[0.7, 0.3]),
            'on_2b': np.random.choice([0, 1], n_pitches, p=[0.8, 0.2]),
            'on_3b': np.random.choice([0, 1], n_pitches, p=[0.9, 0.1])
        })
        
        return sample_df
    
    def clean_and_engineer_basic_features(self, df):
        """Clean data and engineer basic features"""
        logger.info("Starting data cleaning and basic feature engineering...")
        
        df_clean = df.copy()
        
        # 1. Standardize pitch types
        logger.info("Standardizing pitch types...")
        df_clean['pitch_type_clean'] = df_clean['pitch_type'].apply(self._standardize_pitch_type)
        
        # 2. Create location zones
        logger.info("Creating location zones...")
        df_clean['zone_9'] = df_clean.apply(self._create_zone_9, axis=1)
        
        # 3. Count states and situations
        logger.info("Creating count states...")
        df_clean['count_state'] = df_clean['balls'].astype(str) + '-' + df_clean['strikes'].astype(str)
        df_clean['leverage_situation'] = df_clean.apply(self._get_leverage_situation, axis=1)
        
        # 4. Base-out states
        logger.info("Creating base-out states...")
        df_clean['base_out_state'] = df_clean.apply(self._get_base_out_state, axis=1)
        
        # 5. Pitch rewards
        logger.info("Calculating pitch rewards...")
        df_clean['pitch_reward'] = df_clean.apply(self._calculate_pitch_reward, axis=1)
        
        logger.info(f"Data cleaning completed. Shape: {df_clean.shape}")
        return df_clean
    
    def _standardize_pitch_type(self, pitch_type):
        """Standardize pitch type codes"""
        if pd.isna(pitch_type):
            return 'UNK'
        
        pitch_map = {
            'FF': 'FF', 'FT': 'FF', 'FA': 'FF', 'SI': 'SI',
            'SL': 'SL', 'ST': 'SL', 'SLV': 'SL',
            'CH': 'CH', 'FS': 'CH',
            'CU': 'CU', 'KC': 'CU', 'KN': 'CU',
            'FC': 'FC'
        }
        return pitch_map.get(str(pitch_type).upper(), str(pitch_type).upper())
    
    def _create_zone_9(self, row):
        """Create 9-zone grid location"""
        px, pz = row['plate_x'], row['plate_z']
        if pd.isna(px) or pd.isna(pz):
            return 'Z_UNK'
        
        # Horizontal zones
        if px < -0.8:
            h_zone = 0  # Left
        elif px < 0.8:
            h_zone = 1  # Center
        else:
            h_zone = 2  # Right
        
        # Vertical zones using strike zone
        sz_top = row.get('sz_top', 3.4)
        sz_bot = row.get('sz_bot', 1.6)
        
        if pd.isna(sz_top): sz_top = 3.4
        if pd.isna(sz_bot): sz_bot = 1.6
        
        sz_mid = (sz_top + sz_bot) / 2
        
        if pz < sz_bot:
            v_zone = 0  # Low
        elif pz < sz_mid:
            v_zone = 1  # Middle-low
        elif pz < sz_top:
            v_zone = 2  # Middle-high
        else:
            v_zone = 3  # High
        
        return f'Z{v_zone}{h_zone}'
    
    def _get_leverage_situation(self, row):
        """Determine leverage situation"""
        balls, strikes = row['balls'], row['strikes']
        if strikes == 2:
            return 'two_strike'
        elif balls == 3:
            return 'full_count' if strikes == 2 else 'three_ball'
        elif balls == 0 and strikes == 0:
            return 'first_pitch'
        else:
            return 'neutral'
    
    def _get_base_out_state(self, row):
        """Create base-out state description"""
        bases = ''
        
        # Handle base runners
        for base_col, base_num in [('on_1b', '1'), ('on_2b', '2'), ('on_3b', '3')]:
            base_value = row.get(base_col)
            
            has_runner = False
            if base_value is not None and not pd.isna(base_value):
                try:
                    numeric_value = float(base_value)
                    if numeric_value > 0:
                        has_runner = True
                except (ValueError, TypeError):
                    if str(base_value).strip() not in ['', 'nan', 'NaN', '0', '0.0']:
                        has_runner = True
            
            if has_runner:
                bases += base_num
        
        if not bases:
            bases = 'empty'
        
        outs = row.get('outs_when_up', 0)
        return f"{bases}_{outs}out"
    
    def _calculate_pitch_reward(self, row):
        """Calculate immediate pitch-level reward"""
        description = str(row.get('description', '')).lower()
        
        if any(x in description for x in ['called_strike', 'swinging_strike']):
            return 1.0
        elif 'foul' in description and row.get('strikes', 0) == 2:
            return 0.5  # Foul with 2 strikes
        elif 'foul' in description:
            return 0.3
        elif 'ball' in description:
            return -0.5
        elif 'hit_into_play' in description:
            return -0.2  # Neutral for contact
        else:
            return 0.0