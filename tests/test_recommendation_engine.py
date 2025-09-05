"""
Unit tests for recommendation engine
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.recommendation_engine import PitchRecommendationEngine

class TestPitchRecommendationEngine:
    """Test cases for PitchRecommendationEngine"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.predict.return_value = [0.5]
        
        # Create mock encoders
        self.mock_encoders = {
            'pitch_type_clean': Mock(),
            'prev_pitch_type_1': Mock(),
            'zone_9': Mock()
        }
        self.mock_encoders['pitch_type_clean'].transform.return_value = [0]
        self.mock_encoders['prev_pitch_type_1'].transform.return_value = [1]
        self.mock_encoders['zone_9'].transform.return_value = [2]
        
        # Initialize engine
        self.engine = PitchRecommendationEngine(
            model=self.mock_model,
            encoders=self.mock_encoders,
            pitch_types=['FF', 'SL', 'CH'],
            zones=['Z00', 'Z11', 'Z22']
        )
    
    def test_initialization(self):
        """Test engine initialization"""
        assert self.engine.model == self.mock_model
        assert len(self.engine.pitch_types) == 3
        assert len(self.engine.zones) == 3
        assert self.engine.model_version == "1.0.0"
    
    def test_validate_game_state(self):
        """Test game state validation"""
        # Valid state
        valid_state = {
            'balls': 2,
            'strikes': 1,
            'outs_when_up': 1,
            'inning': 7,
            'on_1b': 0,
            'on_2b': 1,
            'on_3b': 0
        }
        is_valid, errors = self.engine.validate_game_state(valid_state)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid state - too many balls
        invalid_state = valid_state.copy()
        invalid_state['balls'] = 4
        is_valid, errors = self.engine.validate_game_state(invalid_state)
        assert not is_valid
        assert len(errors) > 0
        assert "Balls cannot exceed 3" in errors
        
        # Invalid state - too many strikes
        invalid_state = valid_state.copy()
        invalid_state['strikes'] = 3
        is_valid, errors = self.engine.validate_game_state(invalid_state)
        assert not is_valid
        assert "Strikes cannot exceed 2" in errors
    
    def test_recommend_next_pitch(self):
        """Test pitch recommendation"""
        game_state = {
            'balls': 1,
            'strikes': 1,
            'outs_when_up': 0,
            'on_1b': 0,
            'on_2b': 0,
            'on_3b': 0,
            'inning': 1
        }
        
        recommendations = self.engine.recommend_next_pitch(game_state, top_k=2)
        
        # Should return recommendations
        assert len(recommendations) <= 2
        assert isinstance(recommendations, list)
        
        # Each recommendation should have required fields
        if recommendations:
            rec = recommendations[0]
            assert 'pitch_type' in rec
            assert 'zone' in rec
            assert 'expected_reward' in rec
            assert 'confidence' in rec
    
    def test_create_feature_vector(self):
        """Test feature vector creation"""
        game_state = {
            'balls': 2,
            'strikes': 1,
            'outs_when_up': 1,
            'on_1b': 1,
            'on_2b': 0,
            'on_3b': 0,
            'inning': 5,
            'prev_pitch_type_1': 'FF',
            'prev_velocity_1': 94.0
        }
        
        features = self.engine._create_feature_vector(game_state, 'SL', 'Z11')
        
        # Should return a list of features
        assert isinstance(features, list)
        assert len(features) > 0
        
        # First few features should match game state
        assert features[0] == 2  # balls
        assert features[1] == 1  # strikes
        assert features[2] == 1  # outs
    
    def test_simulate_state_change(self):
        """Test state simulation for sequence recommendations"""
        initial_state = {'balls': 1, 'strikes': 1}
        
        # Strike outcome
        strike_state = self.engine._simulate_state_change(initial_state, 'strike')
        assert strike_state['strikes'] == 2
        assert strike_state['balls'] == 1
        
        # Ball outcome
        ball_state = self.engine._simulate_state_change(initial_state, 'ball')
        assert ball_state['balls'] == 2
        assert strike_state['strikes'] == 1
        
        # Foul with less than 2 strikes
        foul_state = self.engine._simulate_state_change(initial_state, 'foul')
        assert foul_state['strikes'] == 2
        
        # Foul with 2 strikes (should not increase)
        two_strike_state = {'balls': 1, 'strikes': 2}
        foul_2strike = self.engine._simulate_state_change(two_strike_state, 'foul')
        assert foul_2strike['strikes'] == 2
    
    def test_get_situational_recommendations(self):
        """Test situational recommendations with context"""
        valid_state = {
            'balls': 2,
            'strikes': 2,
            'outs_when_up': 2,
            'on_3b': 1,
            'inning': 9
        }
        
        result = self.engine.get_situational_recommendations(valid_state)
        
        # Should return structured response
        assert isinstance(result, dict)
        assert 'recommendations' in result
        assert 'game_context' in result
        assert 'leverage_situation' in result
        
        # Check game context analysis
        context = result['game_context']
        assert context['scoring_threat'] == 'HIGH'  # Runner on 3rd
        assert context['count'] == '2-2'
    
    def test_assess_risk_level(self):
        """Test risk level assessment"""
        # High risk: 3 balls, pitch outside zone
        high_risk_state = {'balls': 3, 'strikes': 1}
        high_risk_rec = {'zone': 'Z00'}  # Outside zone
        risk = self.engine._assess_risk_level(high_risk_state, high_risk_rec)
        assert risk == 'HIGH'
        
        # High risk: 2 strikes, pitch in middle of zone
        high_risk_state2 = {'balls': 1, 'strikes': 2}
        high_risk_rec2 = {'zone': 'Z11'}  # Middle of zone
        risk2 = self.engine._assess_risk_level(high_risk_state2, high_risk_rec2)
        assert risk2 == 'HIGH'
        
        # Low risk: neutral count
        low_risk_state = {'balls': 1, 'strikes': 1}
        low_risk_rec = {'zone': 'Z10'}
        risk3 = self.engine._assess_risk_level(low_risk_state, low_risk_rec)
        assert risk3 == 'LOW'

@pytest.fixture
def mock_trained_model():
    """Fixture providing a mock trained model"""
    model = Mock()
    model.predict.return_value = np.array([0.3, 0.7, -0.2])
    return model

@pytest.fixture 
def mock_encoders():
    """Fixture providing mock encoders"""
    encoders = {}
    for feature in ['pitch_type_clean', 'prev_pitch_type_1', 'zone_9']:
        encoder = Mock()
        encoder.transform.return_value = [0]
        encoders[feature] = encoder
    return encoders

def test_integration_full_recommendation_flow(mock_trained_model, mock_encoders):
    """Integration test for full recommendation flow"""
    engine = PitchRecommendationEngine(
        model=mock_trained_model,
        encoders=mock_encoders,
        pitch_types=['FF', 'SL'],
        zones=['Z11', 'Z00']
    )
    
    game_state = {
        'balls': 1,
        'strikes': 2,
        'outs_when_up': 1,
        'on_2b': 1,
        'inning': 8
    }
    
    # Should complete without errors
    recommendations = engine.recommend_next_pitch(game_state)
    assert isinstance(recommendations, list)
    
    situational = engine.get_situational_recommendations(game_state)
    assert isinstance(situational, dict)