"""
Basic tests for the snake game
"""

import unittest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.game_env import SnakeGameEnv
from agents.snake_agent import SnakeAgent
from agents.fruit_agent import FruitAgent


class TestSnakeGame(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.env = SnakeGameEnv(grid_size=10)
        self.snake_agent = SnakeAgent(grid_size=10)
        self.fruit_agent = FruitAgent(grid_size=10)

    def test_game_initialization(self):
        """Test that the game initializes correctly"""
        state = self.env.reset()
        self.assertIsInstance(state, self.env.GameState)
        self.assertEqual(len(state.snake_body), 3)  # Initial snake length
        self.assertEqual(state.turn, 0)
        self.assertEqual(state.status, self.env.GameStatus.RUNNING)

    def test_snake_agent_creation(self):
        """Test that snake agent is created correctly"""
        self.assertIsInstance(self.snake_agent, SnakeAgent)
        self.assertEqual(self.snake_agent.grid_size, 10)

    def test_fruit_agent_creation(self):
        """Test that fruit agent is created correctly"""
        self.assertIsInstance(self.fruit_agent, FruitAgent)
        self.assertEqual(self.fruit_agent.grid_size, 10)

    def test_step_function(self):
        """Test that the step function works"""
        state = self.env.reset()
        # Take a step with arbitrary actions
        next_state, snake_reward, fruit_reward, done, info = self.env.step(
            0, 0
        )  # Up, Up

        self.assertIsInstance(next_state, self.env.GameState)
        self.assertIsInstance(snake_reward, float)
        self.assertIsInstance(fruit_reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_agent_action_selection(self):
        """Test that agents can select actions"""
        state = self.env.reset()

        # Test snake agent
        snake_action = self.snake_agent.choose_action(state)
        self.assertIn(snake_action.value, [0, 1, 2, 3])  # Valid snake actions

        # Test fruit agent
        fruit_action = self.fruit_agent.choose_action(state)
        self.assertIn(fruit_action.value, [0, 1, 2, 3, 4])  # Valid fruit actions


if __name__ == "__main__":
    unittest.main()
