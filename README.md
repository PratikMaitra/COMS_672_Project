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
   git clone <repo_url>
   cd <repo_folder>
