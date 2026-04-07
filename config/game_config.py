"""
Game configuration parameters
"""

# Grid settings
GRID_SIZE = 30
CELL_SIZE = 20  # For rendering

# Game settings
MAX_TURNS = 1000
WIN_SCORE_SNAKE = 50  # Score needed for snake to win
WIN_SCORE_FRUIT = 50  # Score needed for fruit to win

# Reward settings
SNAKE_EAT_FRUIT_REWARD = 10.0
SNAKE_SELF_COLLISION_PENALTY = -10.0
SNAKE_SURVIVAL_REWARD = 0.1

FRUIT_CAUSE_COLLISION_REWARD = 10.0
FRUIT_EATEN_PENALTY = -1.0
FRUIT_EVASION_REWARD = 0.1

# Agent hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON_START = 0.1
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Experience replay settings
EXPERIENCE_REPLAY_SIZE = 10000
BATCH_SIZE = 32

# Context storage settings
CONTEXT_STORAGE_SIZE = 1000

# LLM settings (would be used with actual LLM API)
LLM_MODEL = "gpt-4"  # Placeholder
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 150

# Database settings
DB_PATH = "storage/game_data.sqlite"

# File paths
MODEL_SAVE_PATH = "storage/models/"
LOG_PATH = "logs/"
FRUIT_HISTORY_PATH = "storage/fruit_history.json"
SNAKE_HISTORY_PATH = "storage/snake_history.json"
