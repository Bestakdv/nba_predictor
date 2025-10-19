import tkinter as tk
from tkinter import ttk, messagebox
from data_processor import NBADataProcessor
from model import NBAPredictor
import threading
import time

class NBAPredictor_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NBA Game Predictor")
        self.root.geometry("600x800")
        self.root.configure(bg='#f0f0f0')

        # Initialize the model and data processor
        self.data_processor = NBADataProcessor()
        self.predictor = NBAPredictor()
        
        # Setup UI before loading data
        self.setup_ui()
        
        # Load data and train model in background
        self.load_data_and_train()

    def load_data_and_train(self):
        self.status_label.config(text="Fetching NBA data...")
        threading.Thread(target=self._load_data_and_train, daemon=True).start()

    def _load_data_and_train(self):
        """Background thread for data loading and model training"""
        try:
            # Try to fetch current NBA data
            self.data = self.data_processor.fetch_current_season_stats()
            
            # Update team selection dropdowns
            self.root.after(0, self._update_team_dropdowns)
            
            # Train the model
            self.status_label.config(text="Training model...")
            X, y = self.predictor.generate_training_data(self.data)
            self.model_metrics = self.predictor.train(X, y)
            
            self.status_label.config(text="Ready!")
            self.predict_button.config(state='normal')
            self.refresh_button.config(state='normal')
            self.simulate_button.config(state='normal')
            self.playoff_button.config(state='normal')
            self.evaluate_button.config(state='normal')
            
        except Exception as e:
            self.status_label.config(text="Error loading data. Using sample data...")
            messagebox.showerror("Error", f"Failed to fetch NBA data: {str(e)}\nFalling back to sample data.")
            
            # Fall back to sample data
            self.data = self.data_processor.load_sample_data()
            self.root.after(0, self._update_team_dropdowns)
            
            # Train model with sample data
            X, y = self.predictor.generate_training_data(self.data)
            self.model_metrics = self.predictor.train(X, y)
            
            self.predict_button.config(state='normal')
            self.refresh_button.config(state='normal')
            self.simulate_button.config(state='normal')
            self.playoff_button.config(state='normal')
            self.evaluate_button.config(state='normal')

    def _update_team_dropdowns(self):
        teams = sorted(self.data['team_name'].tolist())
        self.team1_combo['values'] = teams
        self.team2_combo['values'] = teams

    def setup_ui(self):
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(pady=20)
        
        title_label = tk.Label(
            title_frame,
            text="NBA Game Predictor",
            font=('Helvetica', 24, 'bold'),
            bg='#f0f0f0'
        )
        title_label.pack()

        # Status and refresh frame
        status_frame = tk.Frame(self.root, bg='#f0f0f0')
        status_frame.pack(pady=5, fill='x', padx=20)
        
        # Status label
        self.status_label = tk.Label(
            status_frame,
            text="Initializing...",
            font=('Helvetica', 10, 'italic'),
            bg='#f0f0f0'
        )
        self.status_label.pack(side='left', pady=5)
        
        # Refresh button
        self.refresh_button = tk.Button(
            status_frame,
            text="↻ Refresh Data",
            command=self.refresh_data,
            font=('Helvetica', 10),
            state='disabled'
        )
        self.refresh_button.pack(side='right', pady=5)

        # Team selection frame
        selection_frame = tk.Frame(self.root, bg='#f0f0f0')
        selection_frame.pack(pady=20)

        # Team 1 selection
        team1_label = tk.Label(
            selection_frame,
            text="Team 1:",
            font=('Helvetica', 12),
            bg='#f0f0f0'
        )
        team1_label.grid(row=0, column=0, padx=10, pady=5)

        self.team1_var = tk.StringVar()
        self.team1_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.team1_var,
            state='readonly',
            width=30
        )
        self.team1_combo.grid(row=0, column=1, padx=10, pady=5)

        vs_label = tk.Label(
            selection_frame,
            text="VS",
            font=('Helvetica', 16, 'bold'),
            bg='#f0f0f0'
        )
        vs_label.grid(row=1, column=0, columnspan=2, pady=10)

        # Team 2 selection
        team2_label = tk.Label(
            selection_frame,
            text="Team 2:",
            font=('Helvetica', 12),
            bg='#f0f0f0'
        )
        team2_label.grid(row=2, column=0, padx=10, pady=5)

        self.team2_var = tk.StringVar()
        self.team2_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.team2_var,
            state='readonly',
            width=30
        )
        self.team2_combo.grid(row=2, column=1, padx=10, pady=5)

        # Predict button
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=20)

        self.predict_button = tk.Button(
            button_frame,
            text="Predict Winner",
            command=self.predict_game,
            font=('Helvetica', 12, 'bold'),
            bg='#4CAF50',
            fg='white',
            state='disabled'
        )
        self.predict_button.pack(side='left', padx=10)

        # Add Simulate button
        self.simulate_button = tk.Button(
            button_frame,
            text="Simulate 100 Series",
            command=self.simulate_games,
            font=('Helvetica', 12, 'bold'),
            bg='#2196F3',
            fg='white',
            state='disabled'
        )
        self.simulate_button.pack(side='left', padx=10)

        # Add Playoff Series button
        self.playoff_button = tk.Button(
            button_frame,
            text="Simulate Playoff Series",
            command=self.simulate_playoff_series,
            font=('Helvetica', 12, 'bold'),
            bg='#9C27B0',
            fg='white',
            state='disabled'
        )
        self.playoff_button.pack(side='left', padx=10)

        # Add Evaluate Model button
        self.evaluate_button = tk.Button(
            button_frame,
            text="Evaluate Model",
            command=self.evaluate_model,
            font=('Helvetica', 12, 'bold'),
            bg='#FF5722',
            fg='white',
            state='disabled'
        )
        self.evaluate_button.pack(side='left', padx=10)

        # Results frame
        self.results_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.results_frame.pack(pady=20, fill='x', padx=20)

        self.stats_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.stats_frame.pack(pady=20, fill='x', padx=20)

        style = ttk.Style()
        style.configure("Custom.Horizontal.TProgressbar", thickness=20)

    def refresh_data(self):
        """Refresh NBA data and retrain the model"""
        self.predict_button.config(state='disabled')
        self.refresh_button.config(state='disabled')
        self.simulate_button.config(state='disabled')
        self.playoff_button.config(state='disabled')
        self.evaluate_button.config(state='disabled')
        self.load_data_and_train()

    def update_stats_display(self, team1_stats, team2_stats):
        for widget in self.stats_frame.winfo_children():
            widget.destroy()

        # Headers
        tk.Label(
            self.stats_frame,
            text="Statistics Comparison",
            font=('Helvetica', 14, 'bold'),
            bg='#f0f0f0'
        ).pack(pady=10)

        # Display each statistic
        for stat in self.data_processor.feature_columns:
            stat_frame = tk.Frame(self.stats_frame, bg='#f0f0f0')
            stat_frame.pack(fill='x', pady=5)

            # Format the stat name
            stat_name = stat.replace('_', ' ').title()
            tk.Label(
                stat_frame,
                text=stat_name,
                font=('Helvetica', 10),
                bg='#f0f0f0'
            ).pack()

            # Create comparison bars
            comparison_frame = tk.Frame(stat_frame, bg='#f0f0f0')
            comparison_frame.pack(fill='x')

            val1 = team1_stats[stat]
            val2 = team2_stats[stat]
            if 'percentage' in stat:
                val1_str = f"{val1*100:.1f}%"
                val2_str = f"{val2*100:.1f}%"
            else:
                val1_str = f"{val1:.1f}"
                val2_str = f"{val2:.1f}"

            tk.Label(
                comparison_frame,
                text=val1_str,
                font=('Helvetica', 10),
                bg='#f0f0f0'
            ).pack(side='left', padx=5)

            # Progress bars
            max_val = max(val1, val2)
            min_val = min(val1, val2)
            range_val = max_val - min_val if max_val != min_val else max_val

            # Team 1 bar
            team1_progress = ttk.Progressbar(
                comparison_frame,
                style="Custom.Horizontal.TProgressbar",
                length=200,
                mode='determinate'
            )
            team1_progress.pack(side='left', padx=5)
            team1_progress['value'] = (val1 - min_val) / range_val * 100 if range_val else 50

            # Team 2 bar
            team2_progress = ttk.Progressbar(
                comparison_frame,
                style="Custom.Horizontal.TProgressbar",
                length=200,
                mode='determinate'
            )
            team2_progress.pack(side='left', padx=5)
            team2_progress['value'] = (val2 - min_val) / range_val * 100 if range_val else 50

            # Team 2 value
            tk.Label(
                comparison_frame,
                text=val2_str,
                font=('Helvetica', 10),
                bg='#f0f0f0'
            ).pack(side='left', padx=5)

    def predict_game(self):
        """Make a prediction for the selected teams"""
        team1 = self.team1_var.get()
        team2 = self.team2_var.get()

        if not team1 or not team2:
            messagebox.showerror("Error", "Please select both teams!")
            return

        if team1 == team2:
            messagebox.showerror("Error", "Please select different teams!")
            return

        try:
            # Get team statistics
            team1_stats = self.data_processor.get_team_stats(team1, self.data)
            team2_stats = self.data_processor.get_team_stats(team2, self.data)
            
            # Update stats display
            self.update_stats_display(team1_stats, team2_stats)
            
            # Prepare features for prediction
            features = self.data_processor.prepare_game_features(team1_stats, team2_stats)
            
            # Make prediction
            prediction, confidence = self.predictor.predict(features)
            
            # Clear previous results
            for widget in self.results_frame.winfo_children():
                widget.destroy()

            # Display results
            winner = team1 if prediction == 1 else team2
            confidence_pct = confidence * 100

            result_label = tk.Label(
                self.results_frame,
                text="Prediction Results",
                font=('Helvetica', 16, 'bold'),
                bg='#f0f0f0'
            )
            result_label.pack(pady=10)

            winner_label = tk.Label(
                self.results_frame,
                text=f"Predicted Winner: {winner}",
                font=('Helvetica', 14),
                bg='#f0f0f0'
            )
            winner_label.pack(pady=5)

            confidence_label = tk.Label(
                self.results_frame,
                text=f"Confidence: {confidence_pct:.1f}%",
                font=('Helvetica', 12),
                bg='#f0f0f0'
            )
            confidence_label.pack(pady=5)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def simulate_games(self):
        """Run 100 playoff series simulations between the selected teams"""
        if not self.team1_var.get() or not self.team2_var.get():
            messagebox.showerror("Error", "Please select both teams!")
            return

        team1 = self.team1_var.get()
        team2 = self.team2_var.get()

        if team1 == team2:
            messagebox.showerror("Error", "Please select different teams!")
            return

        try:
            # Get team statistics
            team1_stats = self.data_processor.get_team_stats(team1, self.data)
            team2_stats = self.data_processor.get_team_stats(team2, self.data)
            
            # Prepare features for prediction
            features = self.data_processor.prepare_game_features(team1_stats, team2_stats)
            
            # Run 100 series simulations
            team1_series_wins = 0
            total_games = 0
            series_lengths = {4: 0, 5: 0, 6: 0, 7: 0}  # Track series lengths
            team1_win_margins = {4: 0, 4.1: 0, 4.2: 0, 4.3: 0}  # Track win margins (4-0, 4-1, 4-2, 4-3)
            team2_win_margins = {4: 0, 4.1: 0, 4.2: 0, 4.3: 0}
            
            for _ in range(100):
                game_results, team1_wins, team2_wins = self.predictor.simulate_playoff_series(features)
                
                if team1_wins > team2_wins:
                    team1_series_wins += 1
                    margin = team1_wins + (team2_wins / 10)  # 4-0 = 4.0, 4-1 = 4.1, etc.
                    team1_win_margins[margin] += 1
                else:
                    margin = team2_wins + (team1_wins / 10)
                    team2_win_margins[margin] += 1
                
                series_length = len(game_results)
                series_lengths[series_length] += 1
                total_games += series_length
            
            team2_series_wins = 100 - team1_series_wins
            avg_games_per_series = total_games / 100
            
            # Display results
            result_text = f"\n100 Playoff Series Simulation Results:\n{'='*60}\n"
            result_text += f"Series Win Rate:\n"
            result_text += f"{team1}: {team1_series_wins}%\n"
            result_text += f"{team2}: {team2_series_wins}%\n\n"
            
            result_text += f"Average Series Length: {avg_games_per_series:.1f} games\n\n"
            
            result_text += f"Series Length Distribution:\n"
            result_text += f"4 games: {series_lengths[4]}%\n"
            result_text += f"5 games: {series_lengths[5]}%\n"
            result_text += f"6 games: {series_lengths[6]}%\n"
            result_text += f"7 games: {series_lengths[7]}%\n\n"
            
            result_text += f"Series Outcomes Breakdown:\n"
            if team1_series_wins > 0:
                result_text += f"{team1} wins:\n"
                result_text += f"  4-0: {team1_win_margins[4]}%\n"
                result_text += f"  4-1: {team1_win_margins[4.1]}%\n"
                result_text += f"  4-2: {team1_win_margins[4.2]}%\n"
                result_text += f"  4-3: {team1_win_margins[4.3]}%\n"
            
            if team2_series_wins > 0:
                result_text += f"{team2} wins:\n"
                result_text += f"  4-0: {team2_win_margins[4]}%\n"
                result_text += f"  4-1: {team2_win_margins[4.1]}%\n"
                result_text += f"  4-2: {team2_win_margins[4.2]}%\n"
                result_text += f"  4-3: {team2_win_margins[4.3]}%\n"
            
            result_text += f"\n{'='*60}"
            
            # Update results display
            for widget in self.results_frame.winfo_children():
                widget.destroy()
                
            result_label = tk.Label(
                self.results_frame,
                text=result_text,
                font=('Courier', 12),
                justify='left',
                bg='#f0f0f0'
            )
            result_label.pack(pady=10)
            
            # Update stats display
            self.update_stats_display(team1_stats, team2_stats)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def simulate_playoff_series(self):
        if not self.team1_var.get() or not self.team2_var.get():
            messagebox.showerror("Error", "Please select both teams!")
            return

        team1 = self.team1_var.get()
        team2 = self.team2_var.get()

        if team1 == team2:
            messagebox.showerror("Error", "Please select different teams!")
            return

        try:
            # Get team statistics
            team1_stats = self.data_processor.get_team_stats(team1, self.data)
            team2_stats = self.data_processor.get_team_stats(team2, self.data)
            
            features = self.data_processor.prepare_game_features(team1_stats, team2_stats)
            
            game_results, team1_wins, team2_wins = self.predictor.simulate_playoff_series(features)
            
            # Calculate series winner and number of games
            series_winner = team1 if team1_wins > team2_wins else team2
            num_games = len(game_results)
            
            # Create game-by-game results
            game_by_game = []
            for i, (winner, conf) in enumerate(game_results):
                game_winner = team1 if winner == 1 else team2
                game_by_game.append(f"Game {i+1}: {game_winner} wins (conf: {conf*100:.1f}%)")
            
            # Display results
            result_text = f"\nPlayoff Series Results:\n{'='*50}\n"
            result_text += f"Series Winner: {series_winner} ({team1_wins}-{team2_wins})\n"
            result_text += f"Series Length: {num_games} games\n\n"
            result_text += "Game by Game:\n"
            result_text += "\n".join(game_by_game)
            result_text += f"\n{'='*50}"
            
            # Update results display
            for widget in self.results_frame.winfo_children():
                widget.destroy()
                
            result_label = tk.Label(
                self.results_frame,
                text=result_text,
                font=('Courier', 12),
                justify='left',
                bg='#f0f0f0'
            )
            result_label.pack(pady=10)
            
            # Update stats display
            self.update_stats_display(team1_stats, team2_stats)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def evaluate_model(self):
        """Display model evaluation metrics and analysis"""
        try:
            # Clear previous results
            for widget in self.results_frame.winfo_children():
                widget.destroy()

            # Create evaluation text
            metrics = self.model_metrics
            eval_text = "Model Evaluation Results:\n"
            eval_text += "=" * 50 + "\n\n"
            
            # Accuracy metrics
            eval_text += "Accuracy Metrics:\n"
            eval_text += f"Training Accuracy: {metrics['train_accuracy']*100:.1f}%\n"
            eval_text += f"Testing Accuracy: {metrics['test_accuracy']*100:.1f}%\n"
            eval_text += f"Cross-validation Score: {metrics['cv_mean']*100:.1f}% (±{metrics['cv_std']*100:.1f}%)\n\n"
            
            # Classification report
            eval_text += "Detailed Performance Metrics:\n"
            report = metrics['classification_report']
            eval_text += f"Precision: {report['weighted avg']['precision']:.3f}\n"
            eval_text += f"Recall: {report['weighted avg']['recall']:.3f}\n"
            eval_text += f"F1-Score: {report['weighted avg']['f1-score']:.3f}\n\n"
            conf_matrix = metrics['confusion_matrix']
            eval_text += "Confusion Matrix:\n"
            eval_text += f"True Negatives: {conf_matrix[0][0]}\n"
            eval_text += f"False Positives: {conf_matrix[0][1]}\n"
            eval_text += f"False Negatives: {conf_matrix[1][0]}\n"
            eval_text += f"True Positives: {conf_matrix[1][1]}\n\n"
            eval_text += self.predictor.get_feature_importance_analysis()
            
            # Display results
            result_label = tk.Label(
                self.results_frame,
                text=eval_text,
                font=('Courier', 12),
                justify='left',
                bg='#f0f0f0'
            )
            result_label.pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during evaluation: {str(e)}")

def main():
    root = tk.Tk()
    app = NBAPredictor_GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
