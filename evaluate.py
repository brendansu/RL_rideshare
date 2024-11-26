import tensorflow as tf
import argparse
from environment import RideshareEnv  # Ensure this matches your environment setup
import matplotlib.pyplot as plt

def evaluate_model(model_path, env, num_episodes=10):
    """
    Evaluate the model by running it in the environment.

    Args:
        model_path (str): Path to the saved model directory.
        env: Environment to evaluate the model in.
        num_episodes (int): Number of episodes to evaluate.

    Returns:
        float: Average cumulative reward across episodes.
        list: Rewards per episode.
    """
    # Load the saved model
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")

    cumulative_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Predict the action using the model
            state_input = tf.expand_dims(state, axis=0)  # Add batch dimension
            q_values = model.predict(state_input, verbose=0)
            action = tf.argmax(q_values[0]).numpy()  # Select greedy action

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        cumulative_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}: Total Reward = {episode_reward}")

    avg_reward = sum(cumulative_rewards) / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")

    return avg_reward, cumulative_rewards

if __name__ == "__main__":
    # Parse arguments from the command line
    parser = argparse.ArgumentParser(description="Evaluate a trained RL model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model directory.")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate.")
    args = parser.parse_args()

    # Initialize the environment
    env = RideshareEnv()  # Replace with your environment setup

    # Evaluate the model
    avg_reward, rewards = evaluate_model(args.model_path, env, args.num_episodes)

    # Plot results
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Evaluation Performance")
    plt.show()