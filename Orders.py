import random
from diplomacy import Game
import pandas as pd
import os

# Define a function to simulate games and extract possible valid orders
def generate_order_dataset(num_games, output_file):
    all_orders = []

    for game_number in range(1, num_games + 1):
        # Create a new game
        game = Game()

        # Play the game until it's done, extracting valid orders for each phase
        while not game.is_game_done:
            possible_orders = game.get_all_possible_orders()

            # Store orders for each power in the current phase
            phase_orders = {}
            for power_name in game.powers:
                power_orders = [
                    random.choice(possible_orders[loc])
                    for loc in game.get_orderable_locations(power_name)
                    if possible_orders[loc]
                ]
                phase_orders[power_name] = power_orders

            # Add orders to the list for each phase
            all_orders.append(phase_orders)

            # Process the game to advance to the next phase
            game.process()

    # Save the orders to the output file (as a CSV for simplicity)
    df = pd.DataFrame(all_orders)  # Create a DataFrame from the list of orders
    df.to_csv(output_file, index=False)  # Save to CSV

# Create a folder for the dataset if it doesn't exist
os.makedirs('dataset', exist_ok=True)

# Generate the dataset with possible orders for a set of games
generate_order_dataset(num_games=100, output_file='dataset/orders.csv')
