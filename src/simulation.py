"""
Game Simulation Engine for MLB Pitch Sequencing
Simulates at-bats and games using trained recommendation models
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class GameSimulator:
    """Simulate at-bats and full games using the trained model"""
    
    def __init__(self, recommendation_engine):
        """
        Initialize game simulator
        
        Args:
            recommendation_engine: PitchRecommendationEngine instance
        """
        self.engine = recommendation_engine
        
        # Pitch outcome probabilities by pitch type
        self.pitch_outcomes = {
            'FF': {'strike': 0.35, 'ball': 0.40, 'contact': 0.25},
            'SL': {'strike': 0.45, 'ball': 0.35, 'contact': 0.20},
            'CH': {'strike': 0.40, 'ball': 0.35, 'contact': 0.25},
            'CU': {'strike': 0.50, 'ball': 0.30, 'contact': 0.20},
            'SI': {'strike': 0.38, 'ball': 0.37, 'contact': 0.25},
            'FC': {'strike': 0.42, 'ball': 0.33, 'contact': 0.25}
        }
        
        logger.info("Game simulator initialized")
    
    def simulate_at_bat(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate a complete at-bat using AI recommendations
        
        Args:
            initial_state: Dictionary containing initial game state
            
        Returns:
            Dictionary with at-bat results and pitch sequence
        """
        state = initial_state.copy()
        pitch_sequence = []
        
        logger.debug(f"Starting at-bat simulation with state: {state}")
        
        while state['balls'] < 4 and state['strikes'] < 3:
            # Get AI recommendation
            recommendations = self.engine.recommend_next_pitch(state, top_k=1)
            if not recommendations:
                logger.warning("No recommendations available, ending at-bat")
                break
                
            recommended_pitch = recommendations[0]
            pitch_type = recommended_pitch['pitch_type']
            zone = recommended_pitch['zone']
            
            # Simulate pitch outcome
            outcome = self._simulate_pitch_outcome(pitch_type, zone, state)
            
            # Update state based on outcome
            if outcome == 'strike':
                state['strikes'] += 1
            elif outcome == 'ball':
                state['balls'] += 1
            elif outcome == 'contact':
                # Contact ends the at-bat
                result = self._determine_contact_result(zone, state)
                break
            
            # Record pitch in sequence
            pitch_sequence.append({
                'pitch_number': len(pitch_sequence) + 1,
                'pitch_type': pitch_type,
                'zone': zone,
                'outcome': outcome,
                'count_before': f"{state['balls']}-{state['strikes']}",
                'expected_reward': recommended_pitch['expected_reward']
            })
            
            # Update sequence memory for next pitch
            state['prev_pitch_type_1'] = pitch_type
            state['prev_zone_1'] = zone
            state['prev_reward_1'] = 1.0 if outcome == 'strike' else -0.5 if outcome == 'ball' else 0.3
        
        # Determine final at-bat result
        if state['balls'] == 4:
            result = 'walk'
        elif state['strikes'] == 3:
            result = 'strikeout'
        else:
            result = getattr(self, '_last_contact_result', 'single')  # Default if contact occurred
        
        return {
            'result': result,
            'pitch_count': len(pitch_sequence),
            'sequence': pitch_sequence,
            'final_count': f"{state['balls']}-{state['strikes']}",
            'initial_state': initial_state,
            'final_state': state
        }
    
    def simulate_multiple_at_bats(self, states: List[Dict[str, Any]], n_simulations: int = 1) -> pd.DataFrame:
        """
        Simulate multiple at-bats for analysis
        
        Args:
            states: List of initial game states
            n_simulations: Number of times to simulate each state
            
        Returns:
            DataFrame with simulation results
        """
        results = []
        
        for i, state in enumerate(states):
            for sim_num in range(n_simulations):
                # Add some randomness to avoid identical simulations
                state_copy = state.copy()
                if sim_num > 0:
                    # Slightly vary velocity or other features
                    state_copy['prev_velocity_1'] = state_copy.get('prev_velocity_1', 92) + np.random.normal(0, 0.5)
                
                at_bat_result = self.simulate_at_bat(state_copy)
                
                results.append({
                    'state_id': i,
                    'simulation': sim_num,
                    'result': at_bat_result['result'],
                    'pitch_count': at_bat_result['pitch_count'],
                    'final_count': at_bat_result['final_count'],
                    'initial_balls': state['balls'],
                    'initial_strikes': state['strikes'],
                    'outs': state.get('outs_when_up', 0),
                    'baserunners': state.get('on_1b', 0) + state.get('on_2b', 0) + state.get('on_3b', 0)
                })
        
        return pd.DataFrame(results)
    
    def _simulate_pitch_outcome(self, pitch_type: str, zone: str, state: Dict[str, Any]) -> str:
        """
        Simulate the outcome of a single pitch
        
        Args:
            pitch_type: Type of pitch thrown
            zone: Location zone of the pitch
            state: Current game state
            
        Returns:
            Outcome string ('strike', 'ball', 'contact')
        """
        # Get base probabilities for this pitch type
        probs = self.pitch_outcomes.get(pitch_type, self.pitch_outcomes['FF'])
        
        # Adjust probabilities based on count
        if state['strikes'] == 2:
            # Batter more defensive with 2 strikes
            probs = {'strike': 0.5, 'ball': 0.3, 'contact': 0.2}
        elif state['balls'] == 3:
            # Batter more selective with 3 balls
            probs = {'strike': 0.3, 'ball': 0.6, 'contact': 0.1}
        
        # Adjust based on zone (in-zone vs out-of-zone)
        if zone in ['Z11', 'Z12', 'Z21']:  # Strike zone
            probs['strike'] = min(0.6, probs['strike'] + 0.1)
            probs['contact'] = min(0.4, probs['contact'] + 0.1)
            probs['ball'] = max(0.1, probs['ball'] - 0.2)
        elif zone in ['Z00', 'Z02', 'Z20', 'Z22']:  # Outside zone
            probs['ball'] = min(0.7, probs['ball'] + 0.2)
            probs['strike'] = max(0.1, probs['strike'] - 0.1)
            probs['contact'] = max(0.05, probs['contact'] - 0.1)
        
        # Random outcome based on adjusted probabilities
        rand = np.random.random()
        if rand < probs['strike']:
            return 'strike'
        elif rand < probs['strike'] + probs['ball']:
            return 'ball'
        else:
            return 'contact'
    
    def _determine_contact_result(self, zone: str, state: Dict[str, Any]) -> str:
        """
        Determine the result of contact (hit outcome)
        
        Args:
            zone: Zone where pitch was located
            state: Game state when contact occurred
            
        Returns:
            Contact result string
        """
        # Simplified contact outcomes based on zone
        if zone in ['Z11', 'Z12', 'Z21']:  # Middle of zone - better contact
            outcomes = ['single', 'double', 'triple', 'home_run', 'out']
            probabilities = [0.25, 0.08, 0.02, 0.05, 0.60]
        else:  # Edge zones - weaker contact
            outcomes = ['single', 'double', 'out']
            probabilities = [0.15, 0.03, 0.82]
        
        result = np.random.choice(outcomes, p=probabilities)
        self._last_contact_result = result  # Store for at-bat result
        return result
    
    def analyze_simulation_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze simulation results for insights
        
        Args:
            results_df: DataFrame from simulate_multiple_at_bats
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Overall outcome distribution
        outcome_dist = results_df['result'].value_counts(normalize=True)
        analysis['outcome_distribution'] = outcome_dist.to_dict()
        
        # Average pitch count by initial count
        pitch_count_by_count = results_df.groupby(['initial_balls', 'initial_strikes'])['pitch_count'].agg([
            'mean', 'std', 'count'
        ]).round(2)
        analysis['pitch_count_by_initial_count'] = pitch_count_by_count.to_dict()
        
        # Success rates (strikeout + weak contact as pitcher success)
        pitcher_success = results_df['result'].isin(['strikeout', 'out']).mean()
        analysis['pitcher_success_rate'] = pitcher_success
        
        # Efficiency metrics
        analysis['average_pitches_per_at_bat'] = results_df['pitch_count'].mean()
        analysis['strikeout_rate'] = (results_df['result'] == 'strikeout').mean()
        analysis['walk_rate'] = (results_df['result'] == 'walk').mean()
        
        return analysis
    
    def simulate_inning(self, initial_bases: Dict[str, int] = None) -> Dict[str, Any]:
        """
        Simulate a full inning with multiple at-bats
        
        Args:
            initial_bases: Initial base state (default: bases empty)
            
        Returns:
            Dictionary with inning results
        """
        if initial_bases is None:
            initial_bases = {'on_1b': 0, 'on_2b': 0, 'on_3b': 0}
        
        outs = 0
        runs_scored = 0
        at_bats = []
        current_bases = initial_bases.copy()
        
        while outs < 3:
            # Create at-bat state
            at_bat_state = {
                'balls': 0,
                'strikes': 0,
                'outs_when_up': outs,
                'inning': 1,
                **current_bases
            }
            
            # Simulate at-bat
            at_bat_result = self.simulate_at_bat(at_bat_state)
            at_bats.append(at_bat_result)
            
            # Update game state based on result
            result = at_bat_result['result']
            
            if result == 'strikeout':
                outs += 1
            elif result == 'walk':
                runs_scored += self._advance_runners_walk(current_bases)
            elif result in ['single', 'double', 'triple', 'home_run']:
                runs_scored += self._advance_runners_hit(current_bases, result)
            elif result == 'out':
                outs += 1
                # Potential advancement on out (simplified)
                if current_bases['on_3b'] == 1 and np.random.random() < 0.3:
                    runs_scored += 1
                    current_bases['on_3b'] = 0
        
        return {
            'runs_scored': runs_scored,
            'at_bats': len(at_bats),
            'total_pitches': sum(ab['pitch_count'] for ab in at_bats),
            'strikeouts': sum(1 for ab in at_bats if ab['result'] == 'strikeout'),
            'walks': sum(1 for ab in at_bats if ab['result'] == 'walk'),
            'hits': sum(1 for ab in at_bats if ab['result'] in ['single', 'double', 'triple', 'home_run']),
            'at_bat_details': at_bats
        }
    
    def _advance_runners_walk(self, bases: Dict[str, int]) -> int:
        """Advance runners on a walk, return runs scored"""
        runs = 0
        
        if bases['on_3b'] == 1 and bases['on_2b'] == 1 and bases['on_1b'] == 1:
            # Bases loaded walk
            runs = 1
        elif bases['on_2b'] == 1 and bases['on_1b'] == 1:
            # Runners on 1st and 2nd
            bases['on_3b'] = 1
        elif bases['on_1b'] == 1:
            # Runner on 1st moves to 2nd
            bases['on_2b'] = 1
        
        # Batter takes 1st
        bases['on_1b'] = 1
        return runs
    
    def _advance_runners_hit(self, bases: Dict[str, int], hit_type: str) -> int:
        """Advance runners on a hit, return runs scored"""
        runs = 0
        
        if hit_type == 'single':
            if bases['on_3b'] == 1:
                runs += 1
                bases['on_3b'] = 0
            if bases['on_2b'] == 1:
                bases['on_3b'] = 1
                bases['on_2b'] = 0
            if bases['on_1b'] == 1:
                bases['on_2b'] = 1
            bases['on_1b'] = 1
            
        elif hit_type == 'double':
            runs += bases['on_3b'] + bases['on_2b']
            bases['on_3b'] = bases['on_1b']
            bases['on_2b'] = 1
            bases['on_1b'] = 0
            
        elif hit_type == 'triple':
            runs += bases['on_3b'] + bases['on_2b'] + bases['on_1b']
            bases['on_3b'] = 1
            bases['on_2b'] = 0
            bases['on_1b'] = 0
            
        elif hit_type == 'home_run':
            runs += 1 + bases['on_3b'] + bases['on_2b'] + bases['on_1b']
            bases['on_3b'] = 0
            bases['on_2b'] = 0
            bases['on_1b'] = 0
        
        return runs