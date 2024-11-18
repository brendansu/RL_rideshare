import tensorflow as tf
import numpy as np
from environment import RideshareEnv  # Your custom environment
from agent import RLAgent  # Your RL agent class
from utils import save_model, load_model, evaluate_agent  # Utility functions


def main():
    # -------------------------------
    # 1. Hyperparameters
    # -------------------------------
    config = {
        'episodes': 1000,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'batch_size': 64,
        'eval_frequency': 50,
        'save_model_path': 'models/rideshare_rl',
    }

    # -------------------------------
    # 2. Initialize Environment
    # -------------------------------
    env = RideshareEnv()  # Replace with your actual environment initialization
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # -------------------------------
    # 3. Initialize Agent
    # -------------------------------
    agent = RLAgent(state_dim, action_dim, config['learning_rate'], config['gamma'])

    # -------------------------------
    # 4. Training Loop
    # -------------------------------
    for episode in range(config['episodes']):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Agent selects action
            action = agent.select_action(state)

            # Environment returns next state and reward
            next_state, reward, done, info = env.step(action)

            # Store experience and train
            agent.store_experience(state, action, reward, next_state, done)
            agent.train(batch_size=config['batch_size'])

            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}/{config['episodes']}, Total Reward: {total_reward}")

        # -------------------------------
        # 5. Evaluation and Checkpoints
        # -------------------------------
        if (episode + 1) % config['eval_frequency'] == 0:
            evaluate_agent(env, agent)
            save_model(agent, config['save_model_path'])

    # -------------------------------
    # 6. Final Save
    # -------------------------------
    save_model(agent, config['save_model_path'])
    print("Training Complete! Model saved.")


if __name__ == "__main__":
    main()
