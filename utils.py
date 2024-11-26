import numpy as np
import tensorflow as tf

def save_model(model, save_path, tflite_path):
    # Save the TensorFlow model in its standard format
    model.save(save_path)
    print(f"Model saved to {save_path}")

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
    tflite_model = converter.convert()

    # Save the .tflite model
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model converted and saved to {tflite_path}")

def load_model(model_path, tflite=False):
    """
    Load a TensorFlow or TensorFlow Lite model.

    Args:
        model_path (str): Path to the model file or directory.
        tflite (bool): If True, loads a TensorFlow Lite model. Otherwise, loads a standard TensorFlow model.

    Returns:
        model (tf.keras.Model or tf.lite.Interpreter): Loaded model object.
    """
    if tflite:
        # Load TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print(f"TensorFlow Lite model loaded from {model_path}")
        return interpreter
    else:
        # Load TensorFlow model
        model = tf.keras.models.load_model(model_path)
        print(f"TensorFlow model loaded from {model_path}")
        return model

def evaluate_agent(env, model, episodes=10, tflite=False):
    """
    Evaluate a trained agent in the given environment.

    Args:
        env: The environment to evaluate in.
        model: The trained model (TensorFlow or TensorFlow Lite interpreter).
        episodes (int): Number of episodes to evaluate.
        tflite (bool): If True, the model is a TensorFlow Lite interpreter.

    Returns:
        float: Average reward over the evaluation episodes.
    """
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            if tflite:
                # For TensorFlow Lite model
                input_details = model.get_input_details()
                output_details = model.get_output_details()

                # Prepare input data
                input_data = np.array([state], dtype=np.float32)
                model.set_tensor(input_details[0]['index'], input_data)

                # Run inference
                model.invoke()
                action_probs = model.get_tensor(output_details[0]['index'])

                # Select the action with the highest probability
                action = np.argmax(action_probs[0])
            else:
                # For TensorFlow model
                state_input = np.expand_dims(state, axis=0)  # Add batch dimension
                action_probs = model.predict(state_input)
                action = np.argmax(action_probs[0])

            # Take action in the environment
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{episodes}: Reward = {episode_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {episodes} episodes: {avg_reward}")
    return avg_reward

def preprocess_observation(observation):
    """
    Preprocess the observation dictionary into a single flattened array.
    """
    vehicle_positions = observation["vehicle_positions"].flatten()  # Flatten vehicle positions
    demand_grid = observation["demand_grid"].flatten()  # Flatten demand grid
    return np.concatenate([vehicle_positions, demand_grid])  # Combine into a single array