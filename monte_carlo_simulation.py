import random

def simulate_series(prob=0.75):
    """
    Simulate a single World Series.
    
    Each game is simulated with team A winning with probability 'prob'.
    The series stops once either team wins 4 games.
    Returns True if team A wins the series, False otherwise.
    """
    wins_A = 0  # Count of wins for team A
    wins_B = 0  # Count of wins for team B
    while wins_A < 4 and wins_B < 4:
        # Generate a random number in [0, 1) to simulate a game
        if random.random() < prob:
            wins_A += 1  # Team A wins this game
        else:
            wins_B += 1  # Team B wins this game
    return wins_A == 4

# Number of simulated series
num_series = 1000

# Count the number of series that team A wins
wins = sum(simulate_series() for _ in range(num_series))

# Calculate and print the estimated probability
estimated_probability = wins / num_series
print("Estimated probability that team A wins the World Series:", estimated_probability)
