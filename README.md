# COMS_672_Project

# Tiny Llama Diplomacy Bot

A lightweight AI model trained to play Diplomacy, a strategic board game that involves negotiation, cooperation, and competition. This project leverages parameter-efficient fine-tuning (PEFT) techniques like Low Rank Adaption of LLMs(LoRA) to adapt the model for variant maps and issue effective orders.

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
2. Install the required Python packages as required.


## Usage
Prepare your dataset for the particular variant you want to see the LLM roleplay (see Dataset section) or use our provided default dataset. Then run the training scripts to train the model.

If you simply want to do inference on the already fine-tuned model use the Jupyter Notebook [TinyllamaDiplo.ipynb] we have provided in the repository. 

## Dataset
We generated a synthetic dataset by incorporating instructions from CICERO's data and commands from 100 completed Diplomacy games. This dataset contains approximately 20,000 rows of Diplomacy commands. The training dataset for our model can be found in the dataset folder of the repository. The default dataset is based on the World War I (European theatre) diplomacy map.

## Model
The model is based on TinyLlama-1.1B-Chat-v1.0, pre-trained on 3 trillion tokens. Fine-tuning uses PEFT techniques like LoRA to reduce trainable parameters, enabling training on consumer-grade hardware.

## Results
The trained Tiny Llama model can issue valid orders in an entire Diplomacy game. While hallucination and gibberish can occur, the model shows promise for strategic reasoning.

We have provided the results of our model on 10 prompts using the temperature setting of 0.1 and 1. The results are stored in the xlsx format within the results folder.

## Challenges
Limited availability of high-quality open-source datasets.
Diplomacy’s complex rules and vast search space.
Computational resources required to train models effectively.
Contributing
Contributions are welcome! Please submit a pull request with your proposed changes.

## References

- **Anton et al.** *Mastering the Game of No-Press Diplomacy via Human-Regularized Reinforcement Learning and Planning.* NeurIPS 2021.

- **FAIR et al.** *Human-level Play in the Game of Diplomacy by Combining Language Models with Strategic Reasoning.* Science 2022.
- **Ding, N.,et al. Parameter-efficient fine-tuning of large-scale pre-trained language models.
- **Bakhtin, Anton, et al. "No-press diplomacy from scratch." Advances in Neural Information Processing Systems 34 (2021)


