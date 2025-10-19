from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from typing import Tuple, Dict, List
import pandas as pd

class NBAPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_importances = {}
        
    def generate_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = []
        y = []
        teams = data['team_name'].unique()
        
        for i, team1 in enumerate(teams):
            for j, team2 in enumerate(teams):
                if i != j:
                    team1_stats = data[data['team_name'] == team1].iloc[0]
                    team2_stats = data[data['team_name'] == team2].iloc[0]
                    
                    features = []
                    for col in data.columns:
                        if col != 'team_name':
                            diff = team1_stats[col] - team2_stats[col]
                            features.append(diff)
                    
                    X.append(features)
                    y.append(1 if team1_stats['points_per_game'] > team2_stats['points_per_game'] else 0)
        
        return np.array(X), np.array(y)

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Get feature importances
        feature_names = ['points_per_game', 'field_goal_percentage', 'three_point_percentage',
                        'free_throw_percentage', 'rebounds_per_game', 'assists_per_game',
                        'steals_per_game', 'blocks_per_game']
        
        importances = self.model.feature_importances_
        self.feature_importances = dict(zip(feature_names, importances))
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': confusion_matrix(y_test, test_pred),
            'classification_report': classification_report(y_test, test_pred, output_dict=True),
            'feature_importances': dict(sorted(self.feature_importances.items(), 
                                             key=lambda x: x[1], reverse=True))
        }
        
        return metrics
        
    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        prediction = self.model.predict(features)
        confidence = self.model.predict_proba(features)[0]
        return prediction[0], max(confidence)

    def get_feature_importance_analysis(self) -> str:
        if not self.feature_importances:
            return "Model has not been trained yet."
            
        sorted_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)
        
        analysis = "Feature Importance Analysis:\n"
        analysis += "=" * 50 + "\n\n"
        
        for feature, importance in sorted_features:
            feature_name = feature.replace('_', ' ').title()
            importance_pct = importance * 100
            analysis += f"{feature_name}: {importance_pct:.1f}%\n"
            
            # Add interpretation
            if importance_pct > 20:
                analysis += "Very strong predictor of game outcomes\n"
            elif importance_pct > 10:
                analysis += "Significant influence on predictions\n"
            elif importance_pct > 5:
                analysis += "Moderate impact on game results\n"
            else:
                analysis += "Minor factor in predictions\n"
            analysis += "\n"
            
        return analysis

    def simulate_games(self, features: np.ndarray, num_simulations: int = 10) -> List[Tuple[int, float]]:
        results = []
        base_prediction = self.model.predict_proba(features)[0]
        
        # Calculate base win probability
        team1_base_prob = base_prediction[1]
        
        for i in range(num_simulations):
            # Add home court advantage alternating between teams
            home_advantage = 0.05
            is_team1_home = i % 2 == 0
            
            # Adjust probability based on home court
            if is_team1_home:
                team1_prob = team1_base_prob + home_advantage
            else:
                team1_prob = team1_base_prob - home_advantage
        
            team1_prob = max(0.1, min(0.9, team1_prob))
            
            # Add random variance based on team strength
            variance = 0.15
            random_factor = np.random.normal(0, variance/2)
            team1_prob += random_factor
            team1_prob = max(0.1, min(0.9, team1_prob))
            
            # Determine winner based on adjusted probability
            winner = 1 if np.random.random() < team1_prob else 0
            confidence = team1_prob if winner == 1 else (1 - team1_prob)
            
            results.append((winner, confidence))
            
        return results 

    def simulate_playoff_series(self, features: np.ndarray) -> Tuple[List[Tuple[int, float]], int, int]:
        results = []
        team1_wins = 0
        team2_wins = 0
        base_prediction = self.model.predict_proba(features)[0]
        team1_base_prob = base_prediction[1]
        
        home_court_schedule = [True, True, False, False, True, False, True]
        game_number = 0
        
        while team1_wins < 4 and team2_wins < 4:
            is_team1_home = home_court_schedule[game_number]
            home_advantage = 0.05
            
            if is_team1_home:
                team1_prob = team1_base_prob + home_advantage
            else:
                team1_prob = team1_base_prob - home_advantage
            
            # Playoff intensity increases variance slightly
            variance = 0.18
            random_factor = np.random.normal(0, variance/2)
            team1_prob += random_factor
            
            team1_prob = max(0.1, min(0.9, team1_prob))
            
            # Determine winner
            winner = 1 if np.random.random() < team1_prob else 0
            confidence = team1_prob if winner == 1 else (1 - team1_prob)
            
            # Update wins and store result
            if winner == 1:
                team1_wins += 1
            else:
                team2_wins += 1
                
            results.append((winner, confidence))
            game_number += 1
        
        return results, team1_wins, team2_wins 
