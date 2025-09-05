"""
Pitch Recommendation Engine for MLB Sequencing
Provides real-time pitch recommendations based on trained models
"""

import pandas as pd
import logging
import copy

logger = logging.getLogger(__name__)

class PitchRecommendationEngine:
    """Advanced pitch recommendation system for real-time use"""
    
    def __init__(self, model, encoders, pitch_types, zones):
        self.model = model
        self.encoders = encoders
        self.pitch_types = pitch_types
        self.zones = zones
        self.model_version = "1.0.0"
        logger.info(f"Recommendation engine initialized with {len(pitch_types)} pitch types and {len(zones)} zones")
    
    def recommend_next_pitch(self, game_state, top_k=3, available_pitches=None):
        """Recommend top-k pitches for a given game state"""
        recommendations = []
        pitches_to_try = available_pitches if available_pitches else self.pitch_types
        
        for pitch_type in pitches_to_try:
            for zone in self.zones:
                try:
                    features = self._create_feature_vector(game_state, pitch_type, zone)
                    expected_reward = self.model.predict(features)[0]
                    recommendations.append({
                        'pitch_type': pitch_type,
                        'zone': zone,
                        'expected_reward': expected_reward,
                        'confidence': abs(expected_reward),
                        'game_state': f"{game_state.get('balls',0)}-{game_state.get('strikes',0)}"
                    })
                except Exception as e:
                    logger.debug(f"Prediction failed for {pitch_type} in {zone}: {e}")
        
        # Always return top_k even if rewards are negative
        recommendations.sort(key=lambda x: x['expected_reward'], reverse=True)
        return recommendations[:top_k] if recommendations else [{'pitch_type': None, 'zone': None, 'expected_reward': 0}]
    
    def recommend_pitch_sequence(self, initial_state, sequence_length=3, top_k=1):
        """
        Recommend sequences for different outcomes (strike, ball, foul)
        Returns: dict of sequences keyed by outcome
        """
        sequences = {}
        for outcome_type in ['strike', 'ball', 'foul']:
            seq = []
            state = copy.deepcopy(initial_state)
            for _ in range(sequence_length):
                recs = self.recommend_next_pitch(state, top_k=top_k)
                if recs:
                    seq.append(recs)
                    # Advance state using realistic simulation
                    state = self._simulate_state_change(state, outcome_type)
            sequences[outcome_type] = seq
        return sequences
    
    def _create_feature_vector(self, game_state, pitch_type, zone):
        """Convert game state + pitch to feature vector for model"""
        features = {
            'balls': game_state.get('balls',0),
            'strikes': game_state.get('strikes',0),
            'outs_when_up': game_state.get('outs_when_up',0),
            'on_1b': game_state.get('on_1b',0),
            'on_2b': game_state.get('on_2b',0),
            'on_3b': game_state.get('on_3b',0),
            'inning': game_state.get('inning',1),
            'prev_velocity_1': game_state.get('prev_velocity_1',90),
            'velocity_delta_1': game_state.get('velocity_delta_1',0),
            'prev_reward_1': game_state.get('prev_reward_1',0),
            'repeat_pitch_1': int(pitch_type == game_state.get('prev_pitch_type_1')),
            'repeat_zone_1': int(zone == game_state.get('prev_zone_1'))
        }
        for col, val in [('prev_pitch_type_1', game_state.get('prev_pitch_type_1','MISSING')),
                         ('prev_pitch_type_2', game_state.get('prev_pitch_type_2','MISSING')),
                         ('prev_zone_1', game_state.get('prev_zone_1','MISSING')),
                         ('prev_zone_2', game_state.get('prev_zone_2','MISSING')),
                         ('pitch_type_clean', pitch_type),
                         ('zone_9', zone)]:
            try:
                features[f"{col}_encoded"] = self.encoders[col].transform([val])[0] if col in self.encoders else 0
            except:
                features[f"{col}_encoded"] = 0
        return pd.DataFrame([features])
    
    def _simulate_state_change(self, state, outcome):
        """Advance balls/strikes realistically for a pitch outcome"""
        new_state = state.copy()
        if outcome == 'strike':
            new_state['strikes'] = min(2, new_state.get('strikes',0)+1)
        elif outcome == 'ball':
            new_state['balls'] = min(3, new_state.get('balls',0)+1)
        elif outcome == 'foul':
            if new_state.get('strikes',0) < 2:
                new_state['strikes'] += 1
        return new_state
    
    def validate_game_state(self, game_state):
        """Check that game state values are within baseball rules"""
        errors = []
        if not 0 <= game_state.get('balls',0) <= 3: errors.append("Balls 0-3")
        if not 0 <= game_state.get('strikes',0) <= 2: errors.append("Strikes 0-2")
        if not 0 <= game_state.get('outs_when_up',0) <= 2: errors.append("Outs 0-2")
        if not 1 <= game_state.get('inning',1) <= 20: errors.append("Inning 1-20")
        for b in ['on_1b','on_2b','on_3b']:
            if game_state.get(b,0) not in [0,1]: errors.append(f"{b} must be 0 or 1")
        return len(errors) == 0, errors
    
    def get_situational_recommendations(self, game_state, top_k=5):
        """Return recommendations with notes, risk, and context"""
        valid, errors = self.validate_game_state(game_state)
        if not valid: return {'error':'Invalid state', 'details': errors}
        recs = self.recommend_next_pitch(game_state, top_k=top_k)
        for r in recs:
            r['situation_notes'] = self._get_situation_notes(game_state, r)
            r['risk_level'] = self._assess_risk_level(game_state, r)
        return {
            'recommendations': recs,
            'game_context': self._analyze_game_context(game_state),
            'leverage_situation': self._get_leverage_situation(game_state)
        }
    
    def _get_situation_notes(self, gs, rec):
        notes = []
        balls, strikes = gs.get('balls',0), gs.get('strikes',0)
        if strikes == 2: notes.append("Two-strike advantage - batter defensive")
        if balls == 3: notes.append("Must throw strike - avoid risky locations")
        if balls == 0 and strikes == 0: notes.append("First pitch - establish strike zone")
        zone = rec['zone']
        if zone in ['Z11','Z12','Z21']: notes.append("In strike zone - higher contact risk")
        elif zone in ['Z00','Z02','Z20','Z22']: notes.append("Outside zone - chase pitch")
        return '; '.join(notes) if notes else "Standard situation"
    
    def _assess_risk_level(self, gs, rec):
        balls, strikes, zone = gs.get('balls',0), gs.get('strikes',0), rec['zone']
        if balls==3 and zone not in ['Z10','Z11','Z12','Z21']: return 'HIGH'
        elif strikes==2 and zone in ['Z11','Z12']: return 'HIGH'
        elif balls>=2 and strikes==0: return 'MEDIUM'
        else: return 'LOW'
    
    def _analyze_game_context(self, gs):
        return {
            'count': f"{gs.get('balls',0)}-{gs.get('strikes',0)}",
            'baserunners': sum(gs.get(f'on_{b}b',0) for b in [1,2,3]),
            'inning': gs.get('inning',1),
            'outs': gs.get('outs_when_up',0),
            'scoring_threat': 'HIGH' if gs.get('on_3b',0) else 'MEDIUM' if gs.get('on_2b',0) else 'LOW'
        }
    
    def _get_leverage_situation(self, gs):
        balls, strikes = gs.get('balls',0), gs.get('strikes',0)
        if strikes==2: return 'two_strike_advantage'
        elif balls==3: return 'must_throw_strike'
        elif balls==0 and strikes==0: return 'first_pitch'
        else: return 'neutral_count'
