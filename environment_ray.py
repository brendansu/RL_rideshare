import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class MyMultiAgentEnv(MultiAgentEnv):

    def __init__(self, config=None):
        super().__init__()

        # map related definitions
        self.map = # load map
        self.map_graph = # graph format map
        self.grid_cnt = # number of grids on the map

        # fleet related definitions
        self.fleet_size = # fleet size
        self.portion_pool_veh = # portion of fleet vehicle accepting pooling orders
        self.fleet_status = {} # data structure to store all fleet vehicle status

        # demand related definitions
        self.all_demands = {} # store all demands, data structure subject to further updates
        self.left_over_demands = {} # store left over demands from previous step, data structure subject to further updates

        # each grid is controlled by an agent
        self.agents = self.possible_agents = [
            f"zone_{i}" for i in range(self.grid_cnt)
        ]

        # each agent has its observation and action spaces
        self.observation_spaces = {
            f"zone_{i}": gym.spaces.Discrete('size of observation space here') for i in range(self.grid_cnt)
        }

        self.action_spaces = {
            f"zone_{i}": gym.spaces.Discrete('size of action space here') for i in range(self.grid_cnt)
        }

        self.time_step = 0 # clock of the simulation, each step forward 3 minutes, each 5 steps take new demand, 480 steps per day
        self.last_move = None

    def reset(self, *, seed=None, options=None):
        # return observation dict and infos dict.
        self.time_step = 0 # initialize clock
        self.left_over_demands = {} # initialize unfulfilled orders
        self.initialize_fleet(seed) # initialize operation fleet
        init_observation = {f"zone_{i}": [] for i in range(self.grid_cnt)} # representation of empty observation subject to change if error is raised
        return init_observation, {}

    def step(self, action_dict):
        # return observation dict, rewards dict, termination/truncation dicts, and infos
        # advance clock
        self.time_step += 1

        # update demand in the simulation
        if self.time_step % 3 == 1: # add new orders every 15 minutes
            new_orders = self.get_demands((self.time_step-1)/3)# bootstrap new orders
            curr_demands = self.left_over_demands + new_orders #

        for req in curr_demands:
            id_req = req['id']
            self.all_demands['id']['wait_time'] += 1 # add one time step to the wait time of all trips in curr_demands

        # dispatch order


        # get repositioning moves at each step
        moves = {}

        for i in range(self.grid_cnt):
            moves[] = action_dict[f"zone_{i}"] # get move of each zone

        # update fleet vehicle distribution


        rewards = {}
        for i in range(self.grid_cnt):
            reward

        terminateds = {"__all__": self.time_step >= 288 # 288 stand for 288 time steps in a day, 5 min a step, 20 mph
        return observations, rewards, terminateds, {}, {}

    def initialize_fleet(self, seed):
        # input: fleet size, portion of fleet vehicle for pooling, random seed
        # output: data structure containing the status of all fleet vehicles
        self.fleet_status = {} # emptying the old data

        # add logic for initializing new data after this