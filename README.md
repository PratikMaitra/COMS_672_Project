# COMS_672_Project

# Tiny Llama Diplomacy Bot

A lightweight AI model trained to play Diplomacy, a strategic board game that involves negotiation, cooperation, and competition. This project leverages parameter-efficient fine-tuning (PEFT) techniques to adapt the model for variant maps and issue effective orders.

## Table of Contents

- [Motivation](#motivation)
- [Project Goals](#project-goals)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Challenges](#challenges)
- [Contributing](#contributing)
- [References](#references)

## Motivation

Diplomacy requires AI to handle complex interactions and strategies involving multiple players. This project aims to build an AI model capable of managing various Diplomacy variants using existing data and state-of-the-art AI techniques.

## Project Goals

- Train a lightweight AI model to play Diplomacy.
- Adapt the model to different variant maps and develop efficient communication.
- Issue valid orders in a game of Diplomacy.

## Features

- Pre-trained model for issuing orders in Diplomacy.
- Synthetic dataset derived from existing Diplomacy games.
- Efficient map parser that handles arbitrary map variants.
- Parameter-efficient fine-tuning (PEFT) techniques used for training.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PratikMaitra/COMS_672_Project.git
   cd COMS_672_Project
2. Install the required Python packages:
bash
Copy code
pip install -r requirements.txt

Usage
Prepare your dataset (see Dataset section).
Train the model:
bash
Copy code
python train.py --dataset <path_to_dataset>
Use the trained model:
bash
Copy code
python play.py --model <path_to_trained_model>

## Dataset
We generated a synthetic dataset by incorporating instructions from CICERO's data and commands from 100 completed Diplomacy games. This dataset contains approximately 20,000 rows of Diplomacy commands.

## Model
The model is based on TinyLlama-1.1B-Chat-v1.0, pre-trained on 3 trillion tokens. Fine-tuning uses PEFT techniques like LoRA to reduce trainable parameters, enabling training on consumer-grade hardware.

## Results
The trained Tiny Llama model can issue valid orders in an entire Diplomacy game. While hallucination and gibberish can occur, the model shows promise for strategic reasoning.

## Challenges
Limited availability of high-quality open-source datasets.
Diplomacy’s complex rules and vast search space.
Computational resources required to train models effectively.
Contributing
Contributions are welcome! Please submit a pull request with your proposed changes.

## References
Anton, et al. Mastering the Game of No-Press Diplomacy via Human-Regularized Reinforcement Learning and Planning.
FAIR, et al. Human-level play in the game of Diplomacy by combining language models with strategic reasoning.

