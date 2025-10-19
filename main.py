from data_processor import NBADataProcessor
from model import NBAPredictor

def main():
    # Initialize components
    data_processor = NBADataProcessor()
    predictor = NBAPredictor()
    
    # Load and prepare data
    data = data_processor.load_sample_data()
    
    # Generate and prepare training data
    X, y = predictor.generate_training_data(data)
    
    # Train the model
    predictor.train(X, y)
    
    def predict_game(team1: str, team2: str) -> None:
        try:
            # Get team statistics
            team1_stats = data_processor.get_team_stats(team1, data)
            team2_stats = data_processor.get_team_stats(team2, data)
            
            # Prepare features for prediction
            features = data_processor.prepare_game_features(team1_stats, team2_stats)
            
            # Make prediction
            prediction, confidence = predictor.predict(features)
            
            # Display results
            winner = team1 if prediction == 1 else team2
            confidence_pct = confidence * 100
            
            print(f"\nPrediction Results:")
            print(f"{'='*50}")
            print(f"Game: {team1} vs {team2}")
            print(f"Predicted Winner: {winner}")
            print(f"Confidence: {confidence_pct:.2f}%")
            print(f"{'='*50}")
            
        except IndexError:
            print(f"Error: One or both teams not found. Available teams: {', '.join(data['team_name'].unique())}")
    
    while True:
        print("\nAvailable teams:", ", ".join(data['team_name'].unique()))
        print("\nEnter two teams to predict (or 'quit' to exit)")
        
        team1 = input("Enter first team: ").strip()
        if team1.lower() == 'quit':
            break
            
        team2 = input("Enter second team: ").strip()
        if team2.lower() == 'quit':
            break
            
        predict_game(team1, team2)

if __name__ == "__main__":
    main() 
