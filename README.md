# Project 4 — Reinforcement Learning for IoT Resource Management (Simulated)

## Overview
Custom Gym environment simulates multiple IoT devices with queues and energy budgets. A DQN agent (Stable-Baselines3) can be trained to schedule transmissions.

## Run
```bash
pip install -r requirements.txt
python train_dqn.py   # trains a DQN agent (requires torch & stable-baselines3)
python evaluate.py    # evaluate the trained agent
```
