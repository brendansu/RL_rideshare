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
        for veh in self.fleet_status: # get idle vehicle queue
            if veh['time_2_empty'] == 0:
                curr_supply.




        # get repositioning moves at each step
        moves = {}

        for i in range(self.grid_cnt):
            moves[] = action_dict[f"zone_{i}"] # get move of each zone

        # if vehicle is going to stay

        # update fleet vehicle distribution
        for veh in self.fleet_status:
            # update vehicle status
            # reposition vehicle: current grid update to reposition target grid, mileage plus one, time to empty set to zero
            # in task vehicle: if time to empty is 1, set to zero then update current grid, if time to empty more than 1, time to empty minus one, mileage plus one
            if veh['repo_flag'] == 1:
                veh['repo_flag'] = 0 # reset reposition flag
                veh['tot_mile'] += 1 # add mileage
                dest = veh['dest_grid'] # get destination
                veh['curr_grid'] = dest # update position
                veh['dest_grid'] = None # update destination
            elif veh['time_2_empty'] != 0:
                veh['tot_mile'] += 1 # add mileage
                if veh['time_2_empty'] == 1: # if the vehicle is one step to the destination
                    dest = veh['dest_grid'] # get destination
                    veh['curr_grid'] = dest # update position
                    veh['dest_grid'] = None # update destination
                veh['time_2_empty'] -= 1 # update time to empty

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

    def find_adjacent_groups(self, requests):
        # generate a list of grouped requests
        # each group is a list of requests going from zones A and B (adjacent) to zones X and Y (adjacent)

        groups = []  # list to store different groups
        visited = set()  # mark visited requests in the list
        locked_pickup_zones = set()
        locked_dropoff_zones = set()

        for i, req1 in enumerate(requests):  # curr_demands, deal with data structure later
            if req1['id'] in visited:
                continue

            group = [req1]  # start to group requests
            visited.add(req1['id']) # mark a request as 'visited' if a request is added to a group

            # find the first matching request of req1 and establish locked-up zones for matching
            if not locked_pickup_zones and not locked_dropoff_zones:
                for req2 in requests[i + 1:]:
                    if req2['id'] not in visited:
                        # check adjacency for both pickup and dropoff areas
                        if (self.are_adjacent(req1['pick_up'], req2['pick_up'])) and (
                                self.are_adjacent(req1['drop_off'], req2['drop_off'])):
                            locked_pickup_zones = {req1['pick_up'], req2['pick_up']}
                            group.append(req2)

            pickup_zones = {req1['pick_up']} # pickup zone to look at
            dropoff_zones = {req1['drop_off']} # dropoff zone to look at

            for req2 in requests[i + 1:]:  # start to find pooling requests that meet pooling requirements
                if req2['id'] not in visited:
