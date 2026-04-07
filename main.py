"""
Main entry point for the two-player snake game with RL agents
"""

import argparse
import time
import os
from agents.snake_agent import SnakeAgent
from agents.fruit_agent import FruitAgent
from environment.game_env import SnakeGameEnv
from utils.helpers import manhattan_distance


def train_agents(episodes: int = 1000, render: bool = False, delay: float = 0.1):
    """Train the snake and fruit agents through self-play"""

    # Initialize agents and environment
    snake_agent = SnakeAgent()
    fruit_agent = FruitAgent()
    env = SnakeGameEnv()

    # Create directories if they don't exist
    os.makedirs("storage", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("storage/models", exist_ok=True)

    print(f"Starting training for {episodes} episodes...")

    for episode in range(episodes):
        state = env.reset()
        total_snake_reward = 0
        total_fruit_reward = 0
        steps = 0

        while (
            state.status == env.GameStatus.RUNNING and steps < 1000
        ):  # Prevent infinite loops
            # Get actions from both agents
            snake_action = snake_agent.choose_action(state).value
            fruit_action = fruit_agent.choose_action(state).value

            # Execute step
            next_state, snake_reward, fruit_reward, done, info = env.step(
                snake_action, fruit_action
            )

            # Update agents
            snake_agent.update_q_value(
                state, snake_action, snake_reward, next_state, done
            )
            fruit_agent.update_q_value(
                state, fruit_action, fruit_reward, next_state, done
            )

            # Store context for LLM-like reasoning
            if done:
                snake_outcome = (
                    "success"
                    if snake_reward > 0
                    else "failure"
                    if snake_reward < 0
                    else "neutral"
                )
                fruit_outcome = (
                    "success"
                    if fruit_reward > 0
                    else "failure"
                    if fruit_reward < 0
                    else "neutral"
                )
                snake_agent.store_context(state, snake_action, snake_outcome)
                fruit_agent.store_context(state, fruit_action, fruit_outcome)

            # Accumulate rewards
            total_snake_reward += snake_reward
            total_fruit_reward += fruit_reward
            steps += 1

            # Render if requested
            if render:
                os.system("cls" if os.name == "nt" else "clear")
                print(f"Episode: {episode + 1}/{episodes}")
                print(f"Step: {steps}")
                print(
                    f"Snake Score: {state.snake_score} | Fruit Score: {state.fruit_score}"
                )
                print(env.render())
                print(f"Info: {info}")
                time.sleep(delay)

            state = next_state

        # Experience replay
        if episode % 10 == 0:  # Replay every 10 episodes
            snake_agent.replay_experiences()
            fruit_agent.replay_experiences()

        # Print progress
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{episodes} completed")
            print(f"  Snake Reward: {total_snake_reward:.2f}")
            print(f"  Fruit Reward: {total_fruit_reward:.2f}")
            print(f"  Steps: {steps}")
            print(f"  Snake Epsilon: {snake_agent.epsilon:.3f}")
            print(f"  Fruit Epsilon: {fruit_agent.epsilon:.3f}")
            print()

        # Decay exploration rate
        snake_agent.epsilon = max(0.01, snake_agent.epsilon * 0.995)
        fruit_agent.epsilon = max(0.01, fruit_agent.epsilon * 0.995)

    print("Training completed!")

    # Save final agent states (in a full implementation, this would save Q-tables, etc.)
    print("Saving agent models...")
    # This would be implemented with actual model saving


def play_game(render: bool = True, delay: float = 0.2):
    """Play a single game with trained agents"""

    # Initialize agents and environment
    snake_agent = SnakeAgent()
    fruit_agent = FruitAgent()
    env = SnakeGameEnv()

    state = env.reset()
    steps = 0

    print("Starting game...")
    print("Snake Goal: Eat fruits while avoiding self-collision")
    print("Fruit Goal: Position itself to make snake collide")
    print()

    if render:
        os.system("cls" if os.name == "nt" else "clear")
        print(f"Initial State:")
        print(f"Snake Score: {state.snake_score} | Fruit Score: {state.fruit_score}")
        print(env.render())
        time.sleep(delay)

    while state.status == env.GameStatus.RUNNING and steps < 1000:
        # Get actions from both agents
        snake_action = snake_agent.choose_action(state).value
        fruit_action = fruit_agent.choose_action(state).value

        # Get explanations (for debugging/learning)
        if steps == 0:  # Only show explanations for first move to avoid spam
            print("=== AGENT REASONING ===")
            print("SNAKE AGENT:")
            print(
                snake_agent.get_action_explanation(
                    state, snake_agent.choose_action(state)
                )
            )
            print("\nFRUIT AGENT:")
            print(
                fruit_agent.get_action_explanation(
                    state, fruit_agent.choose_action(state)
                )
            )
            print("======================\n")

        # Execute step
        next_state, snake_reward, fruit_reward, done, info = env.step(
            snake_action, fruit_action
        )

        if render:
            os.system("cls" if os.name == "nt" else "clear")
            print(f"Step: {steps + 1}")
            print(
                f"Snake Score: {state.snake_score} | Fruit Score: {state.fruit_score}"
            )
            print(f"Last Action - Snake: {snake_action}, Fruit: {fruit_action}")
            print(f"Reward - Snake: {snake_reward:.2f}, Fruit: {fruit_reward:.2f}")
            print(env.render())
            print(f"Info: {info}")
            if done:
                print(f"GAME OVER - Reason: {info.get('reason', 'unknown')}")
            time.sleep(delay)

        state = next_state
        steps += 1

        if done:
            break

    print(f"\nGame finished after {steps} steps")
    print(f"Final Score - Snake: {state.snake_score}, Fruit: {state.fruit_score}")
    print(f"Result: {state.status.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-Player Snake Game with RL Agents")
    parser.add_argument("--train", action="store_true", help="Train agents")
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--play", action="store_true", help="Play a game with trained agents"
    )
    parser.add_argument("--render", action="store_true", help="Render the game")
    parser.add_argument(
        "--delay", type=float, default=0.1, help="Delay between frames (seconds)"
    )

    args = parser.parse_args()

    if args.train:
        train_agents(episodes=args.episodes, render=args.render, delay=args.delay)
    elif args.play:
        play_game(render=args.render, delay=args.delay)
    else:
        # Default: play a game
        play_game(render=True, delay=args.delay)
