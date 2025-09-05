"""
Pitch Recommendation Engine for MLB Sequencing
Provides real-time pitch recommendations based on trained models
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PitchRecommendationEngine:
    """Advanced pitch recommendation system for real-time use"""
    
    def __init__(self, model, encoders, pitch_types, zones):
        """
        Initialize recommendation engine
        
        Args:
            model: Trained ML model for prediction
            encoders: Dictionary of label encoders for categorical features
            pitch_types: List of available pitch types
            zones: List of available pitch zones
        """
        self.model = model
        self.encoders = encoders
        self.pitch_types = pitch_types
        self.zones = zones
        self.model_version = "1.0.0"
        logger.info(f"Recommendation engine initialized with {len(pitch_types)} pitch types and {len(zones)} zones")
    
    def recommend_next_pitch(self, game_state, top_k=3, available_pitches=None):
        """
        Recommend top-k pitches for given game state
        
        Args:
            game_state: Dictionary containing current game situation
            top_k: Number of recommendations to return
            available_pitches: Optional list to constrain available pitches
        
        Returns:
            List of pitch recommendations with expected rewards
        """
        recommendations = []
        
        # Use provided pitches or all available
        pitches_to_try = available_pitches if available_pitches else self.pitch_types
        
        # Try all combinations of pitch types and zones
        for pitch_type in pitches_to_try:
            for zone in self.zones:
                try:
                    # Create feature vector
                    features = self._create_feature_vector(game_state, pitch_type, zone)
                    
                    # Predict expected reward
                    # expected_reward = self.model.predict([features])[0]
                    expected_reward = self.model.predict(features)[0]
                    recommendations.append({
                        'pitch_type': pitch_type,
                        'zone': zone,
                        'expected_reward': expected_reward,
                        'confidence': abs(expected_reward),  # Simple confidence measure
                        'game_state': f"{game_state.get('balls', 0)}-{game_state.get('strikes', 0)}"
                    })
                    
                except Exception as e:
                    logger.debug(f"Could not predict for {pitch_type} in {zone}: {e}")
                    continue
        
        # Sort by expected reward and return top-k
        recommendations.sort(key=lambda x: x['expected_reward'], reverse=True)
        return recommendations[:top_k]
    
    def recommend_pitch_sequence(self, initial_state, sequence_length=3):
        """
        Recommend a sequence of pitches for multiple scenarios
        
        Args:
            initial_state: Starting game state
            sequence_length: Number of pitches to recommend in sequence
        
        Returns:
            Dictionary with sequence recommendations for different outcomes
        """
        sequences = {}
        current_state = initial_state.copy()
        
        for outcome_type in ['strike', 'ball', 'foul']:
            sequence = []
            state = current_state.copy()
            
            for pitch_num in range(sequence_length):
                rec = self.recommend_next_pitch(state, top_k=1)
                if rec:
                    sequence.append(rec[0])
                    # Simulate state change based on outcome type
                    state = self._simulate_state_change(state, outcome_type)
                
            sequences[outcome_type] = sequence
        
        return sequences
    
    """
    def _create_feature_vector(self, game_state, pitch_type, zone):
        Create feature vector for prediction
        features = []
        
        # Basic game state
        features.extend([
            game_state.get('balls', 0),
            game_state.get('strikes', 0),
            game_state.get('outs_when_up', 0),
            game_state.get('on_1b', 0),
            game_state.get('on_2b', 0),
            game_state.get('on_3b', 0),
            game_state.get('inning', 1)
        ])
        
        # Previous pitch features (encoded)
        prev_pitch_1 = game_state.get('prev_pitch_type_1', 'MISSING')
        prev_pitch_2 = game_state.get('prev_pitch_type_2', 'MISSING')
        prev_zone_1 = game_state.get('prev_zone_1', 'MISSING')
        prev_zone_2 = game_state.get('prev_zone_2', 'MISSING')
        
        # Encode using fitted encoders (with error handling)
        for encoder_name, value in [
            ('prev_pitch_type_1', prev_pitch_1),
            ('prev_pitch_type_2', prev_pitch_2),
            ('prev_zone_1', prev_zone_1),
            ('prev_zone_2', prev_zone_2),
            ('pitch_type_clean', pitch_type),
            ('zone_9', zone)
        ]:
            try:
                if encoder_name in self.encoders:
                    encoded_value = self.encoders[encoder_name].transform([value])[0]
                else:
                    encoded_value = 0
            except (ValueError, KeyError):
                # Handle unseen categories
                encoded_value = 0
            features.append(encoded_value)
        
        # Additional features
        features.extend([
            game_state.get('prev_velocity_1', 90),
            game_state.get('velocity_delta_1', 0),
            game_state.get('prev_reward_1', 0),
            1 if pitch_type == prev_pitch_1 else 0,  # repeat_pitch_1
            1 if zone == prev_zone_1 else 0  # repeat_zone_1
        ])
        
        return features
    """
    def _create_feature_vector(self, game_state, pitch_type, zone):
        """Create feature vector for prediction with correct schema"""
        # Start with base dict of features
        feature_dict = {
            'balls': game_state.get('balls', 0),
            'strikes': game_state.get('strikes', 0),
            'outs_when_up': game_state.get('outs_when_up', 0),
            'on_1b': game_state.get('on_1b', 0),
            'on_2b': game_state.get('on_2b', 0),
            'on_3b': game_state.get('on_3b', 0),
            'inning': game_state.get('inning', 1),
            'prev_velocity_1': game_state.get('prev_velocity_1', 90),
            'velocity_delta_1': game_state.get('velocity_delta_1', 0),
            'prev_reward_1': game_state.get('prev_reward_1', 0),
        }

        # Encode categorical features with the same encoders used in training
        for col, raw_value in [
            ('prev_pitch_type_1', game_state.get('prev_pitch_type_1', 'MISSING')),
            ('prev_pitch_type_2', game_state.get('prev_pitch_type_2', 'MISSING')),
            ('prev_zone_1', game_state.get('prev_zone_1', 'MISSING')),
            ('prev_zone_2', game_state.get('prev_zone_2', 'MISSING')),
            ('pitch_type_clean', pitch_type),
            ('zone_9', zone),
        ]:
            try:
                if col in self.encoders:
                    feature_dict[f"{col}_encoded"] = self.encoders[col].transform([raw_value])[0]
                else:
                    feature_dict[f"{col}_encoded"] = 0
            except Exception:
                feature_dict[f"{col}_encoded"] = 0

        # Derived features
        feature_dict['repeat_pitch_1'] = 1 if pitch_type == game_state.get('prev_pitch_type_1') else 0
        feature_dict['repeat_zone_1'] = 1 if zone == game_state.get('prev_zone_1') else 0

        # Convert to DataFrame so columns align with training
        return pd.DataFrame([feature_dict])

    def _simulate_state_change(self, state, outcome):
        """Simulate state change based on pitch outcome"""
        new_state = state.copy()
        
        if outcome == 'strike':
            new_state['strikes'] = min(2, new_state.get('strikes', 0) + 1)
        elif outcome == 'ball':
            new_state['balls'] = min(3, new_state.get('balls', 0) + 1)
        elif outcome == 'foul' and new_state.get('strikes', 0) < 2:
            new_state['strikes'] = new_state.get('strikes', 0) + 1
        
        return new_state
    
    def validate_game_state(self, game_state):
        """Validate that game state is reasonable"""
        errors = []
        
        # Check count bounds
        if game_state.get('balls', 0) > 3:
            errors.append("Balls cannot exceed 3")
        if game_state.get('strikes', 0) > 2:
            errors.append("Strikes cannot exceed 2")
        if game_state.get('outs_when_up', 0) > 2:
            errors.append("Outs cannot exceed 2")
        if game_state.get('inning', 1) < 1 or game_state.get('inning', 1) > 20:
            errors.append("Inning must be between 1 and 20")
        
        # Check base runners
        for base in ['on_1b', 'on_2b', 'on_3b']:
            if game_state.get(base, 0) not in [0, 1]:
                errors.append(f"Base runner indicator {base} must be 0 or 1")
        
        return len(errors) == 0, errors
    
    def get_situational_recommendations(self, game_state):
        """Get recommendations with situational context"""
        is_valid, errors = self.validate_game_state(game_state)
        if not is_valid:
            return {'error': 'Invalid game state', 'details': errors}
        
        # Get base recommendations
        recommendations = self.recommend_next_pitch(game_state, top_k=5)
        
        # Add situational context
        for rec in recommendations:
            rec['situation_notes'] = self._get_situation_notes(game_state, rec)
            rec['risk_level'] = self._assess_risk_level(game_state, rec)
        
        return {
            'recommendations': recommendations,
            'game_context': self._analyze_game_context(game_state),
            'leverage_situation': self._get_leverage_situation(game_state)
        }
    
    def _get_situation_notes(self, game_state, recommendation):
        """Generate situational notes for recommendation"""
        notes = []
        
        balls = game_state.get('balls', 0)
        strikes = game_state.get('strikes', 0)
        
        if strikes == 2:
            notes.append("Two-strike advantage - batter defensive")
        if balls == 3:
            notes.append("Must throw strike - avoid risky locations")
        if balls == 0 and strikes == 0:
            notes.append("First pitch - establish strike zone")
        
        # Zone-specific notes
        zone = recommendation['zone']
        if zone in ['Z11', 'Z12', 'Z21']:
            notes.append("In strike zone - higher contact risk")
        elif zone in ['Z00', 'Z02', 'Z20', 'Z22']:
            notes.append("Outside zone - chase pitch")
        
        return '; '.join(notes) if notes else "Standard situation"
    
    def _assess_risk_level(self, game_state, recommendation):
        """Assess risk level of recommendation"""
        balls = game_state.get('balls', 0)
        strikes = game_state.get('strikes', 0)
        zone = recommendation['zone']
        
        # High risk scenarios
        if balls == 3 and zone not in ['Z10', 'Z11', 'Z12', 'Z21']:
            return 'HIGH'
        elif strikes == 2 and zone in ['Z11', 'Z12']:
            return 'HIGH'
        elif balls >= 2 and strikes == 0:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _analyze_game_context(self, game_state):
        """Analyze overall game context"""
        context = {
            'count': f"{game_state.get('balls', 0)}-{game_state.get('strikes', 0)}",
            'baserunners': sum([game_state.get(f'on_{b}b', 0) for b in [1, 2, 3]]),
            'inning': game_state.get('inning', 1),
            'outs': game_state.get('outs_when_up', 0)
        }
        
        # Scoring threat
        if game_state.get('on_3b', 0) == 1:
            context['scoring_threat'] = 'HIGH'
        elif game_state.get('on_2b', 0) == 1:
            context['scoring_threat'] = 'MEDIUM'
        else:
            context['scoring_threat'] = 'LOW'
        
        return context
    
    def _get_leverage_situation(self, game_state):
        """Determine leverage situation"""
        balls = game_state.get('balls', 0)
        strikes = game_state.get('strikes', 0)
        
        if strikes == 2:
            return 'two_strike_advantage'
        elif balls == 3:
            return 'must_throw_strike'
        elif balls == 0 and strikes == 0:
            return 'first_pitch'
        else:
            return 'neutral_count'