import tensorflow as tf
import numpy as np
from environment_ray import MyMultiAgentEnv  # Your custom environment
from agent import RLAgent  # Your RL agent class
from utils import save_model, load_model, evaluate_agent, preprocess_observation  # Utility functions


def main():
    # Create the environment instance
    env = MyMultiAgentEnv(config=None, num_vehicles=10000, map_name='data/grid_projected', demand_name='data/req_filtered')

    # Reset the environment
    observations, _ = env.reset()

    # **Run the simulation for 288 time steps (1 full day)**
    for _ in range(288):  # 5 minutes per step → 288 steps = 24 hours
        observations, _, terminateds, _, _ = env.step({})  # Call step with no actions
        if terminateds["__all__"]:
            break  # Stop if simulation reaches its end

    env.plot_grid_state(step=150)
    env.plot_grid_state(step=200)

    env.save_all_demands()
    env.save_fleet_status()
    print("Simulation completed successfully!")


if __name__ == "__main__":
    main()