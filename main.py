import tensorflow as tf
import numpy as np
from environment import RideshareEnv  # Your custom environment
from agent import RLAgent  # Your RL agent class
from utils import save_model, load_model, evaluate_agent, preprocess_observation  # Utility functions


def main():
    # -------------------------------
    # 1. Hyperparameters
    # -------------------------------
    config = {
        'episodes': 10,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'batch_size': 64,
        'eval_frequency': 50,
        'save_model_path': 'models/rideshare_rl',
        'tflitemodel_path': 'models/rideshare_rl.tflite'
    }

    # -------------------------------
    # 2. Initialize Environment
    # -------------------------------
    env = RideshareEnv()  # Replace with your actual environment initialization
    demand_dim = env.grid_size[0] * env.grid_size[1]
    state_dim = env.num_vehicles * 2 + demand_dim
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
        observation = preprocess_observation(state)
        total_reward = 0
        done = False

        while not done:
            # Agent selects action
            action = agent.select_action(observation)

            # Environment returns next state and reward
            next_obs, reward, done, info = env.step(action)

            next_obs = preprocess_observation(next_obs)

            # Store experience and train
            agent.store_experience(observation, action, reward, next_obs, done)
            agent.train()

            observation = next_obs
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
    save_model(agent, config['save_model_path'], config['tflitemodel_path'])
    print("Training Complete! Model saved.")


if __name__ == "__main__":
    main()
