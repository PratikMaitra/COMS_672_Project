import pandas as pd
from datasets import Dataset

# Function to format the assistant's response without repeating the user prompt
def formatted_train(user_prompt, assistant_response) -> str:
    # Format with distinct roles, avoiding redundant user prompt
    return f"user: {user_prompt}\nassistant: {assistant_response}\n"

# Load the original CSV file
orders_df = pd.read_csv('dataset/orders.csv')

# Separate the assistant's expected response from the user prompt
orders_df["formatted_text"] = orders_df.apply(
    lambda row: "\n".join([f"{country}: {orders}" for country, orders in row.items() if country != 'formatted_text']),
    axis=1,
)

# Create the training data without repeating the user prompt
orders_df["training_data"] = orders_df["formatted_text"].apply(
    lambda orders: f"{orders}\n"
)

# Save the cleaned data to a new CSV file
output_path = "dataset/formatted_diplomacy_dataset.csv"
orders_df[["training_data"]].to_csv(output_path, index=False)  # Save only the cleaned training_data

print(f"Formatted dataset saved to {output_path}")

# Prepare training data for your model
def prepare_train_data(file_path):
    # Load the formatted CSV
    data_df = pd.read_csv(file_path)
    
    # Define the user prompt to initiate the interaction
    user_prompt = "You are an AI agent playing Diplomacy. Please give the orders for your turn."
    
    # Create the final training text with proper formatting
    data_df["text"] = data_df["training_data"].apply(
        lambda response: formatted_train(user_prompt, response)
    )
    
    # Convert to a Hugging Face dataset
    dataset = Dataset.from_pandas(data_df[["text"]])
    return dataset

# Path to your formatted CSV
csv_file_path = "dataset/formatted_diplomacy_dataset.csv"

# Prepare the dataset for training
training_data = prepare_train_data(csv_file_path)
