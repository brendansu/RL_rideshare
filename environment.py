import numpy as np
import gymnasium
from gymnasium import spaces
from tensorflow.python.keras.backend import dtype
import json
import geopandas as gpd
from shapely.geometry import shape
import pickle
from math import atan2, degrees
import os

class RideshareEnv(gymnasium.Env):
    """
    Custom Rideshare Environment for reinforcement learning.
    """

    def __init__(self, num_vehicles=10, grid_size=(10, 10), max_steps=100, map_name = 'grid'):
        """
        Initialize the Rideshare environment.

        Args:
            num_vehicles (int): Number of vehicles in the simulation.
            grid_size (tuple): Dimensions of the environment grid (rows, cols).
            max_steps (int): Maximum steps per episode.
        """
        super(RideshareEnv, self).__init__()

        self.map = self.parse_map(map_name)
        self.num_vehicles = num_vehicles
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0

        # Define action and observation spaces
        self.action_space = spaces.Discrete(
            self.num_vehicles * 5)  # 5 actions per vehicle (stay, up, down, left, right)
        self.observation_space = spaces.Dict({
            "vehicle_positions": spaces.Box(
                low=0, high=max(self.grid_size), shape=(self.num_vehicles, 2), dtype=np.int32
            ),
            "demand_grid": spaces.Box(
                low=0, high=15, shape=(self.grid_size[0],self.grid_size[1]), dtype=np.int32
            )
        })

        # Initialize state
        self.demand = None
        self.demand_grid = None
        self.state = None
        self.reset()

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            np.array: Initial state.
        """
        self.current_step = 0
        self.demand = self.generate_demand()
        self.convert_demand()
        self.state = {
            "vehicle_positions":np.random.randint(0, self.grid_size[0], size=(self.num_vehicles, 2)),
            "demand_grid":self.demand_grid
        }
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
        if self.current_step % 5 == 0:
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
        # num_demands = np.random.randint(5, 15)
        num_demands = 15
        return np.random.randint(0, self.grid_size[0], size=(num_demands, 2))

    def convert_demand(self):
        """
        Generate random demand locations.

        Returns:
            np.array: Array of demand locations.
        """
        self.demand_grid = np.zeros((self.grid_size[0], self.grid_size[1]), dtype=int)
        for x, y in self.demand:
            self.demand_grid[x,y] += 1

    def move_vehicle(self, vehicle_idx, movement):
        """
        Move a specific vehicle based on the movement action.

        Args:
            vehicle_idx (int): Index of the vehicle to move.
            movement (int): Movement action (0: stay, 1: up, 2: down, 3: left, 4: right).
        """
        if movement == 1:  # Up
            self.state["vehicle_positions"][vehicle_idx][0] = max(0, self.state["vehicle_positions"][vehicle_idx][0] - 1)
        elif movement == 2:  # Down
            self.state["vehicle_positions"][vehicle_idx][0] = min(self.grid_size[0] - 1, self.state["vehicle_positions"][vehicle_idx][0] + 1)
        elif movement == 3:  # Left
            self.state["vehicle_positions"][vehicle_idx][1] = max(0, self.state["vehicle_positions"][vehicle_idx][1] - 1)
        elif movement == 4:  # Right
            self.state["vehicle_positions"][vehicle_idx][1] = min(self.grid_size[1] - 1, self.state["vehicle_positions"][vehicle_idx][1] + 1)

    def calculate_reward(self):
        """
        Calculate the reward based on vehicle positions and demand fulfillment.

        Returns:
            float: Reward value.
        """
        reward = 0
        for demand_location in self.demand:
            distances = np.linalg.norm(self.state["vehicle_positions"] - demand_location, axis=1)
            closest_vehicle_idx = np.argmin(distances)
            if distances[closest_vehicle_idx] < 1.0:  # Demand fulfilled
                reward += 10  # Positive reward for fulfilling demand
        return reward

    def parse_map(self, map_name):
        """
        Parses the GeoJSON file containing the hexagonal grid map and identifies neighbors for each cell.

        Args:
            map_name (str): The name of the GeoJSON file to parse.

        Returns:
            dict: A dictionary with cell IDs as keys and neighbor information as values.
        """
        if os.path.exists(map_name + '.pickle'):
            print(f"Loading grid map from {map_name + '.pickle'}...")
            with open(map_name + '.pickle', 'rb') as file:
                grid_map = pickle.load(file)
            return grid_map
        else:
            # Load the GeoJSON file
            with open(map_name + '.geojson', 'r') as file:
                grid_data = json.load(file)

            grid_map = {}

            # Extract grid cells
            grid_cells = grid_data['features']

            id = 0

            # Populate the grid map with cell coordinates and initialize neighbors
            for cell in grid_cells:
                cell_id = str(id)  # Adjust key if different
                coordinates = cell['geometry']['coordinates']
                grid_map[cell_id] = {
                    'coordinates': coordinates,
                    'neighbors': {}
                }
                id += 1

            # Compute centroids of the grid cells
            centroids = {
                cell_id: shape({'type': 'Polygon', 'coordinates': grid_map[cell_id]['coordinates']}).centroid
                for cell_id in grid_map
            }

            # Helper function to calculate direction
            def get_direction(reference, neighbor):
                dx = neighbor.x - reference.x
                dy = neighbor.y - reference.y
                angle = (degrees(atan2(dy, dx)) + 360) % 360

                if 30 < angle <= 90:
                    return "north-east"
                elif 90 < angle <= 150:
                    return "north"
                elif 150 < angle <= 210:
                    return "north-west"
                elif 210 < angle <= 270:
                    return "south-west"
                elif 270 < angle <= 330:
                    return "south"
                else:
                    return "south-east"

            # Determine neighbors and assign directions
            threshold = 0.035  # Adjust based on expected distance between hexagons: 0.0336 -> 1.73 miles
            for cell_id, centroid in centroids.items():
                for other_id, other_centroid in centroids.items():
                    if cell_id != other_id and centroid.distance(other_centroid) < threshold:
                        direction = get_direction(centroid, other_centroid)
                        grid_map[cell_id]['neighbors'][direction] = other_id

            with open(map_name + '.pickle', 'wb') as file:
                pickle.dump(grid_map, file)
            print(f"Grid map saved to {map_name + '.pickle'}")

            return grid_map