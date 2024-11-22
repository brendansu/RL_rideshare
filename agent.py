import numpy as np
import tensorflow as tf
from collections import deque
import random

class RLAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, buffer_size=10000, batch_size=64):
        """
        Initialize the RL agent.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            buffer_size (int): Size of the experience replay buffer.
            batch_size (int): Number of samples for each training batch.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        # Build models
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def build_model(self):
        """
        Build the neural network model.

        Returns:
            tf.keras.Model: The compiled model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='linear')  # Linear activation for Q-values
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        """
        Update the target model weights to match the main model weights.
        """
        self.target_model.set_weights(self.model.get_weights())

    def select_action(self, state, epsilon=0.1):
        """
        Select an action using epsilon-greedy policy.

        Args:
            state (np.array): Current state.
            epsilon (float): Probability of selecting a random action.

        Returns:
            int: Selected action.
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)  # Random action
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])  # Greedy action

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.

        Args:
            state (np.array): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.array): Next state.
            done (bool): Whether the episode is done.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        """
        Train the agent using a batch of experiences from the replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to train

        # Sample a batch of experiences
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        # Predict Q-values
        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Update Q-values with the Bellman equation
        for i in range(self.batch_size):
            q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i]) * (1 - dones[i])

        # Train the model on the updated Q-values
        self.model.fit(states, q_values, verbose=0, batch_size=self.batch_size)

    def save(self, path):
        """
        Save the model to the specified path.

        Args:
            path (str): Path to save the model.
        """
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path):
        """
        Load a model from the specified path.

        Args:
            path (str): Path to load the model from.
        """
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")