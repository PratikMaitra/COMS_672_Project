from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import pandas as pd
from transformers import GenerationConfig
from time import perf_counter
import os

# Initialize the tokenizer and model
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, load_in_8bit=False,
                                             device_map="auto",
                                             trust_remote_code=True)



model_path = "models/tinyllama-diplo-v1/checkpoint-250"

peft_model = PeftModel.from_pretrained(model, model_path, from_transformers=True, device_map="auto")

model = peft_model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Define a function to format the prompt
def formatted_prompt(question):
    return f"user\n{question}\nassistant:"

# Define a function to generate a response based on the input and temperature setting
def generate_response(user_input, temperature):
    prompt = formatted_prompt(user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    generation_config = GenerationConfig(
        do_sample=True, top_k=5, temperature=temperature,
        repetition_penalty=1.2, max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id
    )
    start_time = perf_counter()
    outputs = model.generate(**inputs, **vars(generation_config))
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_time = perf_counter() - start_time
    return response, round(output_time, 2)

# Generate responses and store them in a dataframe
def store_responses(temperature):
    responses = []
    for _ in range(10):
        response, time_taken = generate_response('You are an AI agent playing Diplomacy. Please give the orders for your turn.', temperature)
        responses.append({'Response': response, 'Time Taken': time_taken})
    return pd.DataFrame(responses)

# Store responses for two different temperatures
df_temperature_0_1 = store_responses(0.1)
df_temperature_1 = store_responses(1)

# Save the dataframes to Excel files
df_temperature_0_1.to_excel('Result_temperature_0_1.xlsx', index=False)
df_temperature_1.to_excel('Result_temperature_1.xlsx', index=False)
