import random
from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format
import os
import json  # Import json to convert the dictionary to a JSON string

# Create a folder named 'synt_games' if it doesn't already exist
os.makedirs('synt_games', exist_ok=True)

# Generate and save 100 games
num_games = 100

for game_number in range(1, num_games + 1):
    # Create a new game
    game = Game()
    
    # Play the game until it's done
    while not game.is_game_done:
        # Get all possible orders for the current phase
        possible_orders = game.get_all_possible_orders()
        
        # Assign random orders for each power
        for power_name in game.powers:
            power_orders = [
                random.choice(possible_orders[loc])
                for loc in game.get_orderable_locations(power_name)
                if possible_orders[loc]
            ]
            game.set_orders(power_name, power_orders)
        
        # Process the game to advance to the next phase
        game.process()

    # Save the game in the specified folder with a numbered filename
    filename = f'synt_games/game_{game_number}.json'
    with open(filename, 'w') as f:
        # Convert the game data to a JSON string before writing
        json_data = json.dumps(to_saved_game_format(game))
        f.write(json_data)  # Now writing a string, not a dictionary
