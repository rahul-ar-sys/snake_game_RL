"""
Snake Agent Implementation
Uses LLM reasoning combined with reinforcement learning for decision making.
"""

import json
import random
import sqlite3
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


@dataclass
class GameState:
    snake_body: List[Tuple[int, int]]
    fruit_position: Tuple[int, int]
    grid_size: int
    snake_score: int
    fruit_score: int
    turn: int


class SnakeAgent:
    def __init__(self, grid_size: int = 30):
        self.grid_size = grid_size
        self.q_table = {}  # State-action values
        self.experience_replay = []  # Store experiences for replay
        self.context_storage = []  # Store effective strategies
        self.epsilon = 0.1  # Exploration rate
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.db_path = "storage/snake_agent.db"
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for persistence"""
        import os

        os.makedirs("storage", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY,
                state TEXT,
                action INTEGER,
                reward REAL,
                next_state TEXT,
                done BOOLEAN,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decision_weights (
                id INTEGER PRIMARY KEY,
                state_key TEXT,
                weights TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def get_state_key(self, state: GameState) -> str:
        """Convert game state to a string key for Q-table"""
        # Simplified state representation - in practice, this would be more sophisticated
        head_x, head_y = state.snake_body[0]
        fruit_x, fruit_y = state.fruit_position

        # Relative position of fruit to snake head
        rel_fruit_x = fruit_x - head_x
        rel_fruit_y = fruit_y - head_y

        # Snake direction (based on last two segments)
        if len(state.snake_body) >= 2:
            neck_x, neck_y = state.snake_body[1]
            dir_x = head_x - neck_x
            dir_y = head_y - neck_y
        else:
            dir_x, dir_y = 0, 1  # Default direction

        # Body collision danger in each direction (simplified)
        dangers = self._check_dangers(state)

        return (
            f"{rel_fruit_x},{rel_fruit_y},{dir_x},{dir_y},{','.join(map(str, dangers))}"
        )

    def _check_dangers(self, state: GameState) -> List[int]:
        """Check for dangers (walls, self) in each direction"""
        head_x, head_y = state.snake_body[0]
        dangers = [0, 0, 0, 0]  # up, down, left, right

        # Check each direction
        # Up
        if head_y == 0 or (head_x, head_y - 1) in state.snake_body:
            dangers[0] = 1
        # Down
        if head_y == state.grid_size - 1 or (head_x, head_y + 1) in state.snake_body:
            dangers[1] = 1
        # Left
        if head_x == 0 or (head_x - 1, head_y) in state.snake_body:
            dangers[2] = 1
        # Right
        if head_x == state.grid_size - 1 or (head_x + 1, head_y) in state.snake_body:
            dangers[3] = 1

        return dangers

    def choose_action(self, state: GameState) -> Action:
        """Choose action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(list(Action))

        # Exploitation
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in Action}

        # Find action with highest Q-value
        best_action = max(self.q_table[state_key], key=self.q_table[state_key].get)
        return best_action

    def update_q_value(
        self,
        state: GameState,
        action: Action,
        reward: float,
        next_state: GameState,
        done: bool,
    ):
        """Update Q-value using Q-learning update rule"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Initialize Q-table entries if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in Action}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {action: 0.0 for action in Action}

        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key].values())

        if done:
            target = reward
        else:
            target = reward + self.discount_factor * max_next_q

        new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table[state_key][action] = new_q

        # Store experience
        self._store_experience(state, action, reward, next_state, done)

    def _store_experience(
        self,
        state: GameState,
        action: Action,
        reward: float,
        next_state: GameState,
        done: bool,
    ):
        """Store experience in database for replay"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO experiences (state, action, reward, next_state, done)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                json.dumps(
                    {
                        "snake_body": state.snake_body,
                        "fruit_position": state.fruit_position,
                        "grid_size": state.grid_size,
                        "snake_score": state.snake_score,
                        "fruit_score": state.fruit_score,
                        "turn": state.turn,
                    }
                ),
                action.value,
                reward,
                json.dumps(
                    {
                        "snake_body": next_state.snake_body,
                        "fruit_position": next_state.fruit_position,
                        "grid_size": next_state.grid_size,
                        "snake_score": next_state.snake_score,
                        "fruit_score": next_state.fruit_score,
                        "turn": next_state.turn,
                    }
                ),
                done,
            ),
        )
        conn.commit()
        conn.close()

        # Also keep in memory for quick replay
        self.experience_replay.append((state, action, reward, next_state, done))
        # Limit experience replay size
        if len(self.experience_replay) > 10000:
            self.experience_replay.pop(0)

    def replay_experiences(self, batch_size: int = 32):
        """Learn from past experiences"""
        if len(self.experience_replay) < batch_size:
            return

        batch = random.sample(self.experience_replay, batch_size)
        for state, action, reward, next_state, done in batch:
            self.update_q_value(state, action, reward, next_state, done)

    def store_context(self, state: GameState, action: Action, outcome: str):
        """Store successful/unsuccessful strategies for LLM reasoning"""
        context_entry = {
            "state": {
                "snake_body": state.snake_body,
                "fruit_position": state.fruit_position,
                "snake_score": state.snake_score,
                "turn": state.turn,
            },
            "action": action.name,
            "outcome": outcome,  # 'success', 'failure', 'neutral'
            "timestamp": str(pd.Timestamp.now())
            if "pd" in globals()
            else str(random.random()),
        }
        self.context_storage.append(context_entry)

        # Keep only recent contexts
        if len(self.context_storage) > 1000:
            self.context_storage = self.context_storage[-1000:]

    def get_action_explanation(self, state: GameState, action: Action) -> str:
        """Generate explanation for action choice using LLM-like reasoning"""
        # This would interface with an actual LLM in a full implementation
        # For now, return a rule-based explanation

        state_key = self.get_state_key(state)
        q_values = self.q_table.get(state_key, {a: 0.0 for a in Action})

        explanation = f"Chose {action.name} because:\n"
        explanation += f"- Q-value: {q_values[action]:.3f}\n"

        # Add context-based reasoning
        head_x, head_y = state.snake_body[0]
        fruit_x, fruit_y = state.fruit_position

        if fruit_y < head_y:
            explanation += "- Fruit is above snake\n"
        elif fruit_y > head_y:
            explanation += "- Fruit is below snake\n"
        if fruit_x < head_x:
            explanation += "- Fruit is left of snake\n"
        elif fruit_x > head_x:
            explanation += "- Fruit is right of snake\n"

        # Check immediate dangers
        dangers = self._check_dangers(state)
        danger_dirs = []
        if dangers[0]:
            danger_dirs.append("up")
        if dangers[1]:
            danger_dirs.append("down")
        if dangers[2]:
            danger_dirs.append("left")
        if dangers[3]:
            danger_dirs.append("right")

        if danger_dirs:
            explanation += f"- Danger in direction(s): {', '.join(danger_dirs)}\n"
        else:
            explanation += "- No immediate dangers detected\n"

        return explanation
