import numpy as np
import gymnasium
from gymnasium import spaces


class RideshareEnv(gymnasium.Env):
    """
    Custom Rideshare Environment for reinforcement learning.
    """

    def __init__(self, num_vehicles=10, grid_size=(10, 10), max_steps=100):
        """
        Initialize the Rideshare environment.

        Args:
            num_vehicles (int): Number of vehicles in the simulation.
            grid_size (tuple): Dimensions of the environment grid (rows, cols).
            max_steps (int): Maximum steps per episode.
        """
        super(RideshareEnv, self).__init__()

        self.num_vehicles = num_vehicles
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0

        # Define action and observation spaces
        self.action_space = spaces.Discrete(
            self.num_vehicles * 5)  # 5 actions per vehicle (stay, up, down, left, right)
        self.observation_space = spaces.Box(
            low=0, high=max(self.grid_size), shape=(self.num_vehicles, 2), dtype=np.int32
        )

        # Initialize state
        self.state = None
        self.demand = None
        self.reset()

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            np.array: Initial state.
        """
        self.current_step = 0
        self.state = np.random.randint(0, self.grid_size[0], size=(self.num_vehicles, 2))
        self.demand = self.generate_demand()
        return self.state

    def step(self, action):
        """
        Perform a step in the environment.

        Args:
            action (int): The action index.

        Returns:
            tuple: (next_state, reward, done, info)
        """
        self.current_step += 1

        # Decode action
        vehicle_idx, movement = divmod(action, 5)
        self.move_vehicle(vehicle_idx, movement)

        # Calculate reward
        reward = self.calculate_reward()

        # Check if the episode is done
        done = self.current_step >= self.max_steps

        # Generate new demand periodically
        if self.current_step % 10 == 0:
            self.demand = self.generate_demand()

        return self.state, reward, done, {}

    def render(self, mode='human'):
        """
        Render the environment (text-based).
        """
        print(f"Step: {self.current_step}")
        print(f"Vehicle positions: {self.state}")
        print(f"Demand locations: {self.demand}")

    def generate_demand(self):
        """
        Generate random demand locations.

        Returns:
            np.array: Array of demand locations.
        """
        num_demands = np.random.randint(5, 15)
        return np.random.randint(0, self.grid_size[0], size=(num_demands, 2))

    def move_vehicle(self, vehicle_idx, movement):
        """
        Move a specific vehicle based on the movement action.

        Args:
            vehicle_idx (int): Index of the vehicle to move.
            movement (int): Movement action (0: stay, 1: up, 2: down, 3: left, 4: right).
        """
        if movement == 1:  # Up
            self.state[vehicle_idx][0] = max(0, self.state[vehicle_idx][0] - 1)
        elif movement == 2:  # Down
            self.state[vehicle_idx][0] = min(self.grid_size[0] - 1, self.state[vehicle_idx][0] + 1)
        elif movement == 3:  # Left
            self.state[vehicle_idx][1] = max(0, self.state[vehicle_idx][1] - 1)
        elif movement == 4:  # Right
            self.state[vehicle_idx][1] = min(self.grid_size[1] - 1, self.state[vehicle_idx][1] + 1)

    def calculate_reward(self):
        """
        Calculate the reward based on vehicle positions and demand fulfillment.

        Returns:
            float: Reward value.
        """
        reward = 0
        for demand_location in self.demand:
            distances = np.linalg.norm(self.state - demand_location, axis=1)
            closest_vehicle_idx = np.argmin(distances)
            if distances[closest_vehicle_idx] < 1.0:  # Demand fulfilled
                reward += 10  # Positive reward for fulfilling demand
        return reward