# AGENTS.md - OpenCode Guidance for snake_game Repository

## Repository Overview
This repository implements a two-player snake game with reinforcement learning where:
- Player 1: Snake agent tries to eat fruits while avoiding self-collision
- Player 2: Fruit agent tries to position itself to make the snake collide with itself
Both agents use LLM reasoning coupled with RL, storing decision weights and context in backend storage.

## Getting Started
1. Install Python 3.8+ and required dependencies: `pip install -r requirements.txt`
2. Set up environment variables for LLM API keys in `.env` file
3. Run the game: `python main.py`
4. For frontend development: Open `frontend/index.html` in a browser

## Project Structure
- `agents/` - Contains SnakeAgent and FruitAgent implementations
- `environment/` - Game environment and state management
- `storage/` - Persistent storage for agent experiences and decision weights
- `utils/` - Helper functions for game logic and RL algorithms
- `config/` - Configuration files for game parameters and agent hyperparameters
- `frontend/` - Web-based frontend for game visualization
  - `index.html` - Main game interface with controls and display

## Key Implementation Details
- Agents use LLM for reasoning combined with Q-learning for decision making
- Experience replay stores successful/unsuccessful fruit placements and snake movements
- Reward functions:
  - Snake: +10 for eating fruit, -10 for self-collision, +0.1 for survival
  - Fruit: +10 for causing snake collision, -1 for being eaten, +0.1 for evasion
- Context storage maintains history of effective strategies for both agents
- Frontend communicates with backend via REST API endpoints (to be implemented)

## Development Commands
- Run tests: `python -m pytest tests/`
- Lint code: `flake8 .`
- Format code: `black .`
- Train agents: `python main.py --train --episodes 1000`
- Play game: `python main.py --play`
- Visualize training: `tensorboard --logdir logs/`
- Start frontend server: `python -m http.server 8000` (from frontend directory)

## Architecture Notes
- Game state is represented as a grid with coordinates for snake body and fruit position
- Agents observe state through partial visibility (local neighborhood around head/fruit)
- Decision weights are stored in JSON format with timestamps for tracking improvement
- Backend storage uses SQLite for persistence of experiences and model parameters
- Frontend uses vanilla HTML/CSS/JS for simplicity, updating via DOM manipulation
- In production, frontend would connect to backend API for real game state updates
- Main entry point is `main.py` with training and playing modes
- Agents store experiences in SQLite databases in the `storage/` directory
- Configuration is managed through files in the `config/` directory

## Important Files
- `main.py`: Entry point for training and playing the game
- `agents/snake_agent.py`: Snake agent implementation with Q-learning
- `agents/fruit_agent.py`: Fruit agent implementation with Q-learning
- `environment/game_env.py`: Game environment managing state and rules
- `utils/helpers.py`: Utility functions for game logic
- `config/game_config.py`: Game parameters and hyperparameters