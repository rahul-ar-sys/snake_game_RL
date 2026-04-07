"""
Game Environment for Two-Player Snake Game
Manages game state, rules, and interactions between snake and fruit agents.
"""

import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class GameStatus(Enum):
    RUNNING = 0
    SNAKE_WON = 1  # Snake ate fruit
    FRUIT_WON = 2  # Fruit caused snake to collide
    DRAW = 3  # Max turns reached or other termination


@dataclass
class GameState:
    snake_body: List[Tuple[int, int]]
    fruit_position: Tuple[int, int]
    grid_size: int
    snake_score: int
    fruit_score: int
    turn: int
    status: GameStatus


class SnakeGameEnv:
    def __init__(self, grid_size: int = 30, max_turns: int = 1000):
        self.grid_size = grid_size
        self.max_turns = max_turns
        self.reset()

    def reset(self) -> GameState:
        """Reset the game to initial state"""
        # Initialize snake in center
        center_x, center_y = self.grid_size // 2, self.grid_size // 2
        self.snake_body = [
            (center_x, center_y),
            (center_x, center_y - 1),
            (center_x, center_y - 2),
        ]

        # Place fruit randomly (not on snake)
        self.fruit_position = self._place_fruit()

        self.snake_score = 0
        self.fruit_score = 0
        self.turn = 0
        self.status = GameStatus.RUNNING

        return self._get_state()

    def _place_fruit(self) -> Tuple[int, int]:
        """Place fruit at random position not occupied by snake"""
        while True:
            pos = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            if pos not in self.snake_body:
                return pos

    def _get_state(self) -> GameState:
        """Get current game state"""
        return GameState(
            snake_body=self.snake_body.copy(),
            fruit_position=self.fruit_position,
            grid_size=self.grid_size,
            snake_score=self.snake_score,
            fruit_score=self.fruit_score,
            turn=self.turn,
            status=self.status,
        )

    def step(
        self, snake_action: int, fruit_action: int
    ) -> Tuple[GameState, float, float, bool, Dict[str, Any]]:
        """
        Execute one step of the game

        Args:
            snake_action: Action for snake agent (0=up, 1=down, 2=left, 3=right)
            fruit_action: Action for fruit agent (0=up, 1=down, 2=left, 3=right, 4=stay)

        Returns:
            Tuple of (next_state, snake_reward, fruit_reward, done, info)
        """
        if self.status != GameStatus.RUNNING:
            # Game already ended
            return self._get_state(), 0.0, 0.0, True, {"reason": "game_already_ended"}

        self.turn += 1

        # Process fruit move first (fruit moves to try to trap snake)
        old_fruit_pos = self.fruit_position
        self.fruit_position = self._apply_fruit_action(
            self.fruit_position, fruit_action
        )

        # Process snake move
        old_snake_head = self.snake_body[0]
        new_snake_head = self._apply_snake_action(old_snake_head, snake_action)

        # Check for collisions
        snake_reward = 0.0
        fruit_reward = 0.0
        done = False
        info = {}

        # Check if snake hit wall
        if (
            new_snake_head[0] < 0
            or new_snake_head[0] >= self.grid_size
            or new_snake_head[1] < 0
            or new_snake_head[1] >= self.grid_size
        ):
            # Snake hit wall - fruit wins
            self.status = GameStatus.FRUIT_WON
            snake_reward = -10.0
            fruit_reward = 10.0
            done = True
            info["reason"] = "snake_hit_wall"

        # Check if snake hit itself
        elif new_snake_head in self.snake_body:
            # Snake hit itself - fruit wins
            self.status = GameStatus.FRUIT_WON
            snake_reward = -10.0
            fruit_reward = 10.0
            done = True
            info["reason"] = "snake_hit_self"

        # Check if snake ate fruit
        elif new_snake_head == self.fruit_position:
            # Snake ate fruit - snake wins
            self.snake_body.insert(0, new_snake_head)  # Grow snake
            self.snake_score += 1
            snake_reward = 10.0
            fruit_reward = -1.0

            # Place new fruit
            self.fruit_position = self._place_fruit()
            self.fruit_score += 1
            fruit_reward += 0.1  # Small reward for evasion

            # Check if max length reached (optional win condition)
            if len(self.snake_body) >= self.grid_size * self.grid_size // 2:
                self.status = GameStatus.SNAKE_WON
                done = True
                info["reason"] = "snake_max_length"
            else:
                info["reason"] = "snake_ate_fruit"

        else:
            # Normal move - snake moves forward
            self.snake_body.insert(0, new_snake_head)
            self.snake_body.pop()  # Remove tail (no growth)

            # Small survival reward for snake
            snake_reward = 0.1
            fruit_reward = 0.1  # Small evasion reward for fruit

            info["reason"] = "normal_move"

        # Check if max turns reached
        if self.turn >= self.max_turns and self.status == GameStatus.RUNNING:
            # Draw - neither agent won
            self.status = GameStatus.DRAW
            done = True
            info["reason"] = "max_turns_reached"

        return self._get_state(), snake_reward, fruit_reward, done, info

    def _apply_snake_action(
        self, head_pos: Tuple[int, int], action: int
    ) -> Tuple[int, int]:
        """Apply snake action to get new head position"""
        x, y = head_pos
        if action == 0:  # Up
            return (x, y - 1)
        elif action == 1:  # Down
            return (x, y + 1)
        elif action == 2:  # Left
            return (x - 1, y)
        elif action == 3:  # Right
            return (x + 1, y)
        else:
            raise ValueError(f"Invalid snake action: {action}")

    def _apply_fruit_action(
        self, fruit_pos: Tuple[int, int], action: int
    ) -> Tuple[int, int]:
        """Apply fruit action to get new fruit position"""
        x, y = fruit_pos
        if action == 0:  # Up
            return (x, max(0, y - 1))
        elif action == 1:  # Down
            return (x, min(self.grid_size - 1, y + 1))
        elif action == 2:  # Left
            return (max(0, x - 1), y)
        elif action == 3:  # Right
            return (min(self.grid_size - 1, x + 1), y)
        elif action == 4:  # Stay
            return (x, y)
        else:
            raise ValueError(f"Invalid fruit action: {action}")

    def render(self) -> str:
        """Render the game state as a string for debugging"""
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Place snake body
        for i, (x, y) in enumerate(self.snake_body):
            if i == 0:  # Head
                grid[y][x] = "H"
            else:  # Body
                grid[y][x] = "S"

        # Place fruit
        fx, fy = self.fruit_position
        grid[fy][fx] = "F"

        # Convert to string
        lines = []
        for row in grid:
            lines.append(" ".join(row))
        return "\n".join(lines)
