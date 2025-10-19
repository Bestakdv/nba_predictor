import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams
import time

class NBADataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'points_per_game', 'rebounds_per_game', 'assists_per_game',
            'field_goal_percentage', 'three_point_percentage',
            'free_throw_percentage', 'steals_per_game', 'blocks_per_game'
        ]
        self.api_feature_mapping = {
            'points_per_game': 'PTS',
            'rebounds_per_game': 'REB',
            'assists_per_game': 'AST',
            'field_goal_percentage': 'FG_PCT',
            'three_point_percentage': 'FG3_PCT',
            'free_throw_percentage': 'FT_PCT',
            'steals_per_game': 'STL',
            'blocks_per_game': 'BLK'
        }
        
    def fetch_current_season_stats(self) -> pd.DataFrame:
        try:
            print("Attempting to fetch NBA data...")
            # Get team stats for the current season
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                per_mode_detailed='PerGame',
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Base',
                plus_minus='N',
                pace_adjust='N',
                rank='N',
                outcome='',
                location='',
                month=0,
                season_segment='',
                date_from='',
                date_to='',
                opponent_team_id=0,
                vs_conference='',
                vs_division='',
                game_segment='',
                period=0,
                shot_clock_range='',
                last_n_games=0
            ).get_data_frames()[0]
            
            print("Successfully fetched team stats")
            
            # Get NBA teams
            print("Fetching team names...")
            nba_teams = teams.get_teams()
            team_id_to_name = {team['id']: team['full_name'] for team in nba_teams}
            print(f"Found {len(nba_teams)} teams")
            
            stats_data = {
                'team_name': [team_id_to_name[tid] for tid in team_stats['TEAM_ID']],
            }
            
            for our_name, api_name in self.api_feature_mapping.items():
                stats_data[our_name] = team_stats[api_name]
                
                # Convert percentages from API format to decimal
                if 'percentage' in our_name:
                    stats_data[our_name] = stats_data[our_name] / 100
            
            print("Successfully processed NBA data")
            return pd.DataFrame(stats_data)
            
        except Exception as e:
            print(f"Error fetching NBA data: {str(e)}")
            print("Falling back to sample data...")
            return self.load_sample_data()

    def load_sample_data(self) -> pd.DataFrame:
        print("Loading sample data...")
        sample_data = {
            'team_name': [
                'Lakers', 'Celtics', 'Warriors', 'Nets', 'Bucks',
                'Timberwolves', 'Thunder', 'Pacers', 'Knicks'
            ],
            'points_per_game': [
                112.5, 117.9, 118.9, 113.4, 116.9,
                114.3, 120.5, 117.4, 115.8
            ],
            'rebounds_per_game': [
                45.7, 44.8, 44.6, 45.1, 48.8,
                46.2, 44.8, 41.8, 42.6
            ],
            'assists_per_game': [
                25.3, 26.7, 29.8, 26.4, 25.8,
                24.9, 25.5, 29.7, 20.0
            ],
            'field_goal_percentage': [
                0.487, 0.479, 0.479, 0.488, 0.473,
                0.465, 0.454, 0.501, 0.444
            ],
            'three_point_percentage': [
                0.346, 0.378, 0.385, 0.374, 0.368,
                0.350, 0.319, 0.406, 0.358
            ],
            'free_throw_percentage': [
                0.788, 0.812, 0.821, 0.794, 0.770,
                0.774, 0.777, 0.795, 0.751
            ],
            'steals_per_game': [
                7.2, 7.1, 7.2, 7.0, 6.4,
                8.1, 10.6, 7.1, 7.8
            ],
            'blocks_per_game': [
                5.4, 5.6, 4.8, 5.3, 4.9,
                4.5, 6.1, 6.0, 4.6
            ]
        }
        print("Sample data loaded successfully")
        return pd.DataFrame(sample_data)

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        features = data[self.feature_columns].values
        return self.scaler.fit_transform(features)

    def prepare_game_features(self, team1_stats: Dict, team2_stats: Dict) -> np.ndarray:
        # Combine stats into feature differences
        feature_diff = []
        for feature in self.feature_columns:
            diff = team1_stats.get(feature, 0) - team2_stats.get(feature, 0)
            feature_diff.append(diff)
            
        return np.array(feature_diff).reshape(1, -1)

    def get_team_stats(self, team_name: str, data: pd.DataFrame) -> Dict:
        team_name = team_name.strip()
        team_data = data[data['team_name'].str.lower() == team_name.lower()].iloc[0]
        return {col: team_data[col] for col in self.feature_columns} 
