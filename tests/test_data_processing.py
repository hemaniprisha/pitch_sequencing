"""
Unit tests for data processing module
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import DataProcessor

class TestDataProcessor:
    """Test cases for DataProcessor class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = DataProcessor()
    
    def test_create_sample_data(self):
        """Test sample data generation"""
        df = self.processor._create_sample_data()
        
        # Check basic structure
        assert len(df) == 10000
        assert 'pitch_type' in df.columns
        assert 'balls' in df.columns
        assert 'strikes' in df.columns
        
        # Check data ranges
        assert df['balls'].min() >= 0
        assert df['balls'].max() <= 3
        assert df['strikes'].min() >= 0
        assert df['strikes'].max() <= 2
    
    def test_standardize_pitch_type(self):
        """Test pitch type standardization"""
        # Test known mappings
        assert self.processor._standardize_pitch_type('FF') == 'FF'
        assert self.processor._standardize_pitch_type('FT') == 'FF'
        assert self.processor._standardize_pitch_type('SL') == 'SL'
        assert self.processor._standardize_pitch_type('ST') == 'SL'
        
        # Test unknown/null values
        assert self.processor._standardize_pitch_type(None) == 'UNK'
        assert self.processor._standardize_pitch_type(pd.NA) == 'UNK'
    
    def test_create_zone_9(self):
        """Test zone creation logic"""
        # Test center strike zone
        row = {'plate_x': 0, 'plate_z': 2.5, 'sz_top': 3.4, 'sz_bot': 1.6}
        zone = self.processor._create_zone_9(row)
        assert zone.startswith('Z')
        
        # Test missing values
        row_missing = {'plate_x': None, 'plate_z': None}
        zone_missing = self.processor._create_zone_9(row_missing)
        assert zone_missing == 'Z_UNK'
    
    def test_calculate_pitch_reward(self):
        """Test pitch reward calculation"""
        # Test strike outcomes
        strike_row = {'description': 'called_strike', 'strikes': 1}
        assert self.processor._calculate_pitch_reward(strike_row) == 1.0
        
        # Test ball outcomes
        ball_row = {'description': 'ball', 'strikes': 1}
        assert self.processor._calculate_pitch_reward(ball_row) == -0.5
        
        # Test foul with 2 strikes
        foul_2strikes = {'description': 'foul', 'strikes': 2}
        assert self.processor._calculate_pitch_reward(foul_2strikes) == 0.5
        
        # Test regular foul
        foul_regular = {'description': 'foul', 'strikes': 1}
        assert self.processor._calculate_pitch_reward(foul_regular) == 0.3
    
    def test_get_leverage_situation(self):
        """Test leverage situation identification"""
        # Two strike count
        two_strike = {'balls': 1, 'strikes': 2}
        assert self.processor._get_leverage_situation(two_strike) == 'two_strike'
        
        # Full count
        full_count = {'balls': 3, 'strikes': 2}
        assert self.processor._get_leverage_situation(full_count) == 'full_count'
        
        # Three ball count
        three_ball = {'balls': 3, 'strikes': 1}
        assert self.processor._get_leverage_situation(three_ball) == 'three_ball'
        
        # First pitch
        first_pitch = {'balls': 0, 'strikes': 0}
        assert self.processor._get_leverage_situation(first_pitch) == 'first_pitch'
        
        # Neutral
        neutral = {'balls': 1, 'strikes': 1}
        assert self.processor._get_leverage_situation(neutral) == 'neutral'
    
    def test_get_base_out_state(self):
        """Test base-out state creation"""
        # Bases empty
        empty_bases = {'on_1b': 0, 'on_2b': 0, 'on_3b': 0, 'outs_when_up': 1}
        assert self.processor._get_base_out_state(empty_bases) == 'empty_1out'
        
        # Runner on first
        runner_1b = {'on_1b': 1, 'on_2b': 0, 'on_3b': 0, 'outs_when_up': 0}
        assert self.processor._get_base_out_state(runner_1b) == '1_0out'
        
        # Bases loaded
        loaded = {'on_1b': 1, 'on_2b': 1, 'on_3b': 1, 'outs_when_up': 2}
        assert self.processor._get_base_out_state(loaded) == '123_2out'
    
    def test_clean_and_engineer_basic_features(self):
        """Test full cleaning pipeline"""
        # Create test data
        test_data = pd.DataFrame({
            'pitch_type': ['FF', 'SL', 'CH'],
            'plate_x': [0, -1, 1],
            'plate_z': [2.5, 2.0, 3.0],
            'sz_top': [3.4, 3.4, 3.4],
            'sz_bot': [1.6, 1.6, 1.6],
            'balls': [1, 2, 0],
            'strikes': [1, 1, 2],
            'description': ['ball', 'called_strike', 'foul'],
            'outs_when_up': [0, 1, 2],
            'on_1b': [0, 1, 0],
            'on_2b': [0, 0, 1],
            'on_3b': [1, 0, 0]
        })
        
        result = self.processor.clean_and_engineer_basic_features(test_data)
        
        # Check new columns were created
        assert 'pitch_type_clean' in result.columns
        assert 'zone_9' in result.columns
        assert 'count_state' in result.columns
        assert 'leverage_situation' in result.columns
        assert 'base_out_state' in result.columns
        assert 'pitch_reward' in result.columns
        
        # Check data integrity
        assert len(result) == 3
        assert result['pitch_type_clean'].tolist() == ['FF', 'SL', 'CH']
        assert result['count_state'].tolist() == ['1-1', '2-1', '0-2']

@pytest.fixture
def sample_dataframe():
    """Fixture providing sample DataFrame for testing"""
    return pd.DataFrame({
        'pitch_type': ['FF', 'SL', 'CH', 'CU'],
        'plate_x': [0, -0.5, 0.5, 0],
        'plate_z': [2.5, 2.0, 3.0, 1.5],
        'balls': [1, 2, 0, 3],
        'strikes': [1, 1, 2, 1],
        'description': ['ball', 'called_strike', 'foul', 'ball']
    })

def test_integration_full_pipeline(sample_dataframe):
    """Integration test for full data processing pipeline"""
    processor = DataProcessor()
    result = processor.clean_and_engineer_basic_features(sample_dataframe)
    
    # Should not crash and should return DataFrame
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    
    # Check all expected columns exist
    expected_cols = ['pitch_type_clean', 'zone_9', 'count_state', 
                    'leverage_situation', 'base_out_state', 'pitch_reward']
    for col in expected_cols:
        assert col in result.columns