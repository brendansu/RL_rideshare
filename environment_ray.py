import queue
import random
import json
import geopandas as gpd
from fiona.crs import defaultdict
from shapely.geometry import shape
import pickle
from math import atan2, degrees, floor
import os
import networkx as nx
import pandas as pd
import time
import matplotlib.pyplot as plt

import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class MyMultiAgentEnv(MultiAgentEnv):

    def __init__(self, config=None, num_vehicles=1000, map_name = 'data/grid_projected', demand_name = 'data/req_filtered'):
        super().__init__()
        config = config or {}

        # map related definitions
        self.map = self.parse_map(map_name)# load map
        self.grid_map = gpd.read_file(map_name + '.geojson')
        self.map_graph = self.build_hex_grid_graph()# graph format map
        self.grid_cnt = len(self.map)# number of grids on the map

        # fleet related definitions
        self.fleet_size = num_vehicles# fleet size
        self.portion_pool_veh = config.get("portion_pool_veh", 0.3)# portion of fleet vehicle accepting pooling orders
        self.fleet_status = {} # data structure to store all fleet vehicle status

        # demand related definitions
        self.demand_path = demand_name + '.csv'
        self.all_demands = {} # store all demands, data structure subject to further updates
        self.all_demands_time = {} # trade space for time, time-keyed all demands
        self.all_demands_time = defaultdict(list)
        self.left_over_demands = [] # store left over demands from previous step, data structure subject to further updates

        # each grid is controlled by an agent
        self.agents = self.possible_agents = [
            f"zone_{i}" for i in range(self.grid_cnt)
        ]

        # each agent has its observation and action spaces
        self.observation_spaces = {
            f"zone_{i}": gym.spaces.Discrete(28) for i in range(self.grid_cnt) # supply/demand/pool/solo * 1 + 6 neighbors
        }

        self.action_spaces = {
            f"zone_{i}": gym.spaces.Discrete(14) for i in range(self.grid_cnt) # solo/pool vehicles to 1 + 6 neighbors
        }

        self.time_step = 0 # clock of the simulation, each step forward 3 minutes, each 5 steps take new demand, 480 steps per day
        self.last_move = None
        self.neighbor_query_seq = ["north", "north-east", "north-west", "south", "south-east", "south-west"]

        self.visualization_data = []

    def reset(self, *, seed=None, options=None):
        print('reset environment')
        # return observation dict and infos dict.
        self.time_step = 0 # initialize clock
        self.left_over_demands = [] # initialize unfulfilled orders
        self.initialize_fleet(seed) # initialize operation fleet
        print(f'fleet initialized with {self.fleet_size} vehicles')
        self.initialize_demand()  # initialize operation fleet
        print('all demand loaded')
        init_observation = {f"zone_{i}": [] for i in range(self.grid_cnt)} # representation of empty observation subject to change if error is raised
        return init_observation, {}

    def step(self, action_dict):
        # return observation dict, rewards dict, termination/truncation dicts, and infos
        # advance clock
        tic_step = time.time()
        self.time_step += 1
        print(f'taking step no.{self.time_step}')

        # update demand in the simulation
        if self.time_step % 3 == 1: # add new orders every 15 minutes
            new_orders = self.all_demands_time[((self.time_step-1)/3)]# bootstrap new orders
            curr_demands = self.left_over_demands + new_orders
        else:
            curr_demands = self.left_over_demands

        # Initialize tracking data
        solo_veh_distribution = {zone: 0 for zone in self.map}  # Vehicles per grid
        solo_req_distribution = {zone: 0 for zone in self.map}  # Requests per grid
        pool_veh_distribution = {zone: 0 for zone in self.map}  # Vehicles per grid
        pool_req_distribution = {zone: 0 for zone in self.map}  # Requests per grid

        # Count idle vehicles in each grid
        for veh in self.fleet_status.keys():
            zone = self.fleet_status[veh]['curr_grid']
            if self.fleet_status[veh]['pool_auth']:
                pool_veh_distribution[zone] += 1
            else:
                solo_veh_distribution[zone] += 1

        # Count requests in each grid
        for req_id in curr_demands:
            zone = self.all_demands[req_id]['pick_up']
            if self.all_demands[req_id]['pool_auth']:
                pool_req_distribution[zone] += 1
            else:
                solo_req_distribution[zone] += 1

        # Store data for visualization
        self.visualization_data.append({
            "step": self.time_step,
            "solo_veh_distribution": solo_veh_distribution,
            "solo_req_distribution": solo_req_distribution,
            "pool_veh_distribution": pool_veh_distribution,
            "pool_req_distribution": pool_req_distribution
        })

        for req in curr_demands:
            self.all_demands[req]['wait_time'] += 1 # add one time step to the wait time of all trips in curr_demands

        # dispatch order
        tic = time.time()
        obs, self.left_over_demands, unassigned_solo, unassigned_pool = self.assign_req_to_veh(curr_demands)
        toc = time.time()
        print(f"Order dispatching: Elapsed time: {toc - tic:.4f} seconds")

        # the observation of each zone should include obs_solo and obs_pool information of the zone itself and its neighbors
        observations = {}
        for zone in self.map:
            neighbors = [
                self.map[zone]['neighbors'].get(direction, 'none')
                for direction in self.neighbor_query_seq
            ]

            zones = [zone] +  neighbors
            observations[zone] = sum([obs[grid] if grid != 'none' else [0, 0, 0, 0] for grid in zones], [])

        # get repositioning moves at each step
        # based on the observation in each zone get the action of each zone
        # each zone generates 14 numbers
        moves = {}

        tic = time.time()
        for zone in self.map:
            solo_idle = len(unassigned_solo[zone])
            pool_idle = len(unassigned_pool[zone])
            if solo_idle == 0 and pool_idle == 0:
                moves[zone] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                continue
            else:
                solo_req_within = observations[zone][1]
                pool_req_within = observations[zone][3]
                solo_req_n = observations[zone][5]
                pool_req_n = observations[zone][7]
                solo_req_ne = observations[zone][9]
                pool_req_ne = observations[zone][11]
                solo_req_nw = observations[zone][13]
                pool_req_nw = observations[zone][15]
                solo_req_s = observations[zone][17]
                pool_req_s = observations[zone][19]
                solo_req_se = observations[zone][21]
                pool_req_se = observations[zone][23]
                solo_req_sw = observations[zone][25]
                pool_req_sw = observations[zone][27]
                total_req_solo = solo_req_within + solo_req_nw + solo_req_n + solo_req_ne + solo_req_se + solo_req_s + solo_req_sw + 0.01
                total_req_pool = pool_req_within + pool_req_nw + pool_req_n + pool_req_ne + pool_req_se + pool_req_s + pool_req_sw + 0.01
                solo_repo_within = floor(solo_idle * solo_req_within / total_req_solo)
                pool_repo_within = floor(pool_idle * pool_req_within / total_req_pool)
                solo_repo_nw = floor(solo_idle * solo_req_nw / total_req_solo)
                pool_repo_nw = floor(pool_idle * pool_req_nw / total_req_pool)
                solo_repo_n = floor(solo_idle * solo_req_n / total_req_solo)
                pool_repo_n = floor(pool_idle * pool_req_n / total_req_pool)
                solo_repo_ne = floor(solo_idle * solo_req_ne / total_req_solo)
                pool_repo_ne = floor(pool_idle * pool_req_ne / total_req_pool)
                solo_repo_se = floor(solo_idle * solo_req_se / total_req_solo)
                pool_repo_se = floor(pool_idle * pool_req_se / total_req_pool)
                solo_repo_s = floor(solo_idle * solo_req_s / total_req_solo)
                pool_repo_s = floor(pool_idle * pool_req_s / total_req_pool)
                solo_repo_sw = floor(solo_idle * solo_req_sw / total_req_solo)
                pool_repo_sw = floor(pool_idle * pool_req_sw / total_req_pool)
                moves[zone] = [solo_repo_within, pool_repo_within, solo_repo_nw, pool_repo_nw, solo_repo_n,
                               pool_repo_n, solo_repo_ne, pool_repo_ne, solo_repo_se, pool_repo_se, solo_repo_s,
                               pool_repo_s, solo_repo_sw, pool_repo_sw] # get move of each zone
                for _ in range(solo_repo_nw):
                    veh = unassigned_solo[zone][0]
                    self.fleet_status[veh]['repo_flag'] = 1
                    self.fleet_status[veh]['dest_grid'] = self.map[zone]['neighbors']['north-west']
                    unassigned_solo[zone].remove(veh)
                for _ in range(solo_repo_n):
                    veh = unassigned_solo[zone][0]
                    self.fleet_status[veh]['repo_flag'] = 1
                    self.fleet_status[veh]['dest_grid'] = self.map[zone]['neighbors']['north']
                    unassigned_solo[zone].remove(veh)
                for _ in range(solo_repo_ne):
                    veh = unassigned_solo[zone][0]
                    self.fleet_status[veh]['repo_flag'] = 1
                    self.fleet_status[veh]['dest_grid'] = self.map[zone]['neighbors']['north-east']
                    unassigned_solo[zone].remove(veh)
                for _ in range(solo_repo_se):
                    veh = unassigned_solo[zone][0]
                    self.fleet_status[veh]['repo_flag'] = 1
                    self.fleet_status[veh]['dest_grid'] = self.map[zone]['neighbors']['south-east']
                    unassigned_solo[zone].remove(veh)
                for _ in range(solo_repo_s):
                    veh = unassigned_solo[zone][0]
                    self.fleet_status[veh]['repo_flag'] = 1
                    self.fleet_status[veh]['dest_grid'] = self.map[zone]['neighbors']['south']
                    unassigned_solo[zone].remove(veh)
                for _ in range(solo_repo_sw):
                    veh = unassigned_solo[zone][0]
                    self.fleet_status[veh]['repo_flag'] = 1
                    self.fleet_status[veh]['dest_grid'] = self.map[zone]['neighbors']['south-west']
                    unassigned_solo[zone].remove(veh)
                for _ in range(pool_repo_nw):
                    veh = unassigned_pool[zone][0]
                    self.fleet_status[veh]['repo_flag'] = 1
                    self.fleet_status[veh]['dest_grid'] = self.map[zone]['neighbors']['north-west']
                    unassigned_pool[zone].remove(veh)
                for _ in range(pool_repo_n):
                    veh = unassigned_pool[zone][0]
                    self.fleet_status[veh]['repo_flag'] = 1
                    self.fleet_status[veh]['dest_grid'] = self.map[zone]['neighbors']['north']
                    unassigned_pool[zone].remove(veh)
                for _ in range(pool_repo_ne):
                    veh = unassigned_pool[zone][0]
                    self.fleet_status[veh]['repo_flag'] = 1
                    self.fleet_status[veh]['dest_grid'] = self.map[zone]['neighbors']['north-east']
                    unassigned_pool[zone].remove(veh)
                for _ in range(pool_repo_se):
                    veh = unassigned_pool[zone][0]
                    self.fleet_status[veh]['repo_flag'] = 1
                    self.fleet_status[veh]['dest_grid'] = self.map[zone]['neighbors']['south-east']
                    unassigned_pool[zone].remove(veh)
                for _ in range(pool_repo_s):
                    veh = unassigned_pool[zone][0]
                    self.fleet_status[veh]['repo_flag'] = 1
                    self.fleet_status[veh]['dest_grid'] = self.map[zone]['neighbors']['south']
                    unassigned_pool[zone].remove(veh)
                for _ in range(pool_repo_sw):
                    veh = unassigned_pool[zone][0]
                    self.fleet_status[veh]['repo_flag'] = 1
                    self.fleet_status[veh]['dest_grid'] = self.map[zone]['neighbors']['south-west']
                    unassigned_pool[zone].remove(veh)
        toc = time.time()
        print(f"Reposition decision making: Elapsed time: {toc - tic:.4f} seconds")

        # set reposition vehicle

        # update fleet vehicle distribution
        tic = time.time()
        for veh in self.fleet_status.keys():
            # update vehicle status
            # reposition vehicle: current grid update to reposition target grid, mileage plus one, time to empty set to zero
            # in task vehicle: if time to empty is 1, set to zero then update current grid, if time to empty more than 1, time to empty minus one, mileage plus one
            status = self.fleet_status[veh]

            if status['repo_flag']:
                status['repo_flag'] = 0 # reset reposition flag
                status['tot_mile'] += 1 # add mileage
                status['curr_grid'] = status['dest_grid'] # update position
                status['dest_grid'] = None # update destination
                status['occupancy'] = 0  # update destination
            elif status['time_2_empty'] > 0:
                status['tot_mile'] += 1 # add mileage
                status['time_2_empty'] -= 1  # update time to empty
                if status['time_2_empty'] == 0: # if the vehicle is one step to the destination
                    status['curr_grid'] = status['dest_grid'] # update position
                    status['dest_grid'] = None # update destination
                    status['occupancy'] = 0  # update destination
        toc = time.time()
        print(f"Update fleet status: Elapsed time: {toc - tic:.4f} seconds")

        # rewards = {}
        # for i in range(self.grid_cnt):
        #     reward to be added back

        terminateds = {"__all__": self.time_step >= 288} # 288 stand for 288 time steps in a day, 5 min a step, 20 mph
        toc_step = time.time()
        print(f"Current step: Elapsed time: {toc_step - tic_step:.4f} seconds")
        print("___________________________________________________________")
        return observations, {}, terminateds, {}, {}

    def initialize_fleet(self, seed):
        # input: fleet size, portion of fleet vehicle for pooling, random seed
        # output: data structure containing the status of all fleet vehicles
        self.fleet_status = {} # emptying the old data
        rng = random.Random(seed)

        # add logic for initializing new data after this
        for veh_id in range(self.fleet_size):
            self.fleet_status[f'veh_{veh_id}'] = {
                'curr_grid': str(rng.randint(0, self.grid_cnt - 1)),
                'dest_grid': None,
                'time_2_empty': 0,
                'tot_mile': 0,
                'pool_auth': rng.random() < self.portion_pool_veh,
                'repo_flag': 0,
                'occupancy':0
            }

    def initialize_demand(self):
        df = pd.read_csv(self.demand_path)
        df["share"] = df["share"].astype(bool)

        for req_id, row in df.iterrows():
            time = int(row['time'])
            self.all_demands_time[time].append(f'req_{req_id}')
            self.all_demands[f'req_{req_id}'] = {
                'time': time,
                'pick_up': str(row['pickup']),
                'drop_off': str(row['dropoff']),
                'fare': float(row['fare']),
                'pool_auth': row['share'],
                'status': 'unassigned',
                'wait_time': 0
            }

    def find_adjacent_groups(self, requests):
        # generate a list of grouped requests
        # each group is a list of requests going from zones A and B (adjacent) to zones X and Y (adjacent)

        groups = []  # list to store different groups
        visited = set()  # mark visited requests in the list
        transfer_solo = [] # requests that could not be matched with any other requests within 15 minutes need to be transferred to solo requests

        for i, req1 in enumerate(requests):  # curr_demands, deal with data structure later
            # 1st layer of the loop: req1 is the 'leader' of the group of requests
            locked_pickup_zones = set()
            locked_dropoff_zones = set()
            pickup_locked = False
            dropoff_locked = False

            if req1 in visited: # if a trip request has been assigned to a group
                continue # skip the request

            group = [req1]  # start to group requests if req1 is an unsettled request
            visited.add(req1) # mark a request as 'visited' if a request is added to a group
            locked_pickup_zones.add(self.all_demands[req1]['pick_up']) # update pickup zone for the group
            locked_dropoff_zones.add(self.all_demands[req1]['drop_off']) # update dropoff zone for the group

            j = i+1

            while not ((pickup_locked and dropoff_locked) or j == len(requests)): # try to lock up pickup and dropoff zones first
                req2 = requests[j]
                j += 1
                if req2 not in visited:
                    # check adjacency for pickup and dropoff
                    if (self.are_adjacent(self.all_demands[req1]['pick_up'], self.all_demands[req2]['pick_up']) and
                            self.are_adjacent(self.all_demands[req1]['drop_off'], self.all_demands[req2]['drop_off'])):
                        if pickup_locked and not dropoff_locked: # if pickup zone is locked up
                            if self.all_demands[req2]['pick_up'] not in locked_pickup_zones: # if the req is out of pickup zone
                                continue # move on to the next request
                            else: # if the req is within the pickup zone
                                locked_dropoff_zones.add(self.all_demands[req2]['drop_off']) # add the dropoff zone of the request to the dropoff zone set
                                visited.add(req2) # confirm adding to the pooling group
                                group.append(req2) # confirm adding to the pooling group
                                if len(locked_dropoff_zones) == 2: # if the dropoff zone is a new zone
                                    dropoff_locked = True # lock up the dropoff zone set
                        elif dropoff_locked and not pickup_locked: # if dropoff zone is locked up
                            if self.all_demands[req2]['drop_off'] not in locked_dropoff_zones: # if the req is out of dropoff zone
                                continue # move on to the next request
                            else: # if the req is within the dropoff zone
                                locked_pickup_zones.add(self.all_demands[req2]['pick_up']) # add the pickup zone of the request to the pickup zone set
                                visited.add(req2) # confirm adding to the pooling group
                                group.append(req2) # confirm adding to the pooling group
                                if len(locked_pickup_zones) == 2: # if the pickup zone is a new zone
                                    pickup_locked = True # lock up the pickup zone set
                        else: # if neither pickup or dropoff has been locked up, this request will be the first to be pooled
                            visited.add(req2) # confirm adding to the pooling group
                            group.append(req2) # confirm adding to the pooling group
                            locked_dropoff_zones.add(self.all_demands[req2]['drop_off']) # update dropoff zones and lockup status
                            if len(locked_dropoff_zones) == 2:
                                dropoff_locked = True
                            locked_pickup_zones.add(self.all_demands[req2]['pick_up'])  # update pickup zones and lockup status
                            if len(locked_pickup_zones) == 2:
                                pickup_locked = True

            if j < len(requests): # if there are requests to group after locking up pickup and dropoff zones
                for req3 in requests[j:]: # search from this trip on (all trips before requests[i+1] have been tried)
                    if req3 not in visited:
                        # after locking up pickup and dropoff zones, just need to check whether the zones are in range
                        if ((self.all_demands[req3]['pick_up'] in locked_pickup_zones) and
                                (self.all_demands[req3]['drop_off'] in locked_dropoff_zones)):
                            visited.add(req3)
                            group.append(req3)

            if len(group) == 1:
                transfer_solo.append(req1) # if this trip cannot be pooled, then transfer to solo request
            else:
                groups.append(group) # the group of pooling requests formulated around req1 is recorded

        return groups, transfer_solo

    def assign_req_to_veh(self, curr_demands):
        curr_pool_supply = {zone: queue.Queue() for zone in self.map}
        curr_solo_supply = {zone: queue.Queue() for zone in self.map}
        curr_pool_dem_group = {zone: queue.Queue() for zone in self.map}
        curr_solo_demand = {zone: queue.Queue() for zone in self.map}

        observe = {zone: [0, 0, 0, 0] for zone in self.map}

        unfulfilled = curr_demands
        unassigned_solo = {}
        unassigned_solo = defaultdict(list)
        unassigned_pool = {}
        unassigned_pool = defaultdict(list)

        for veh in self.fleet_status.keys(): # get idle vehicle queue
            if self.fleet_status[veh]['time_2_empty'] == 0: # if the vehicle is idle
                location = self.fleet_status[veh]['curr_grid']  # get the vehicle's current location
                if self.fleet_status[veh]['pool_auth'] == 1: # if the vehicle is authorized to serve pooling request
                    curr_pool_supply[location].put(veh)
                    unassigned_pool[location].append(veh)
                else:
                    curr_solo_supply[location].put(veh)
                    unassigned_solo[location].append(veh)

        curr_pool_demand_grid = defaultdict(list)
        curr_pool_demand = []
        for req in curr_demands:
            location = self.all_demands[req]['pick_up']
            if self.all_demands[req]['pool_auth']:
                curr_pool_demand.append(req)
                curr_pool_demand_grid[location].append(req)
            else:
                curr_solo_demand[location].put(req)

        print('    start to generate pooling groups')
        tic = time.time()
        pooling_groups, extra_solo_req = self.find_adjacent_groups(curr_pool_demand) # generate pooling groups and find requests that cannot be pooled
        print('    pooling groups generated')
        toc = time.time()
        print(f"   Pooling group generation: Elapsed time: {toc - tic:.4f} seconds")

        for group in pooling_groups:
            location = self.all_demands[group[0]]['pick_up']
            curr_pool_dem_group[location].put(group)

        for req in extra_solo_req:
            location = self.all_demands[req]['pick_up']
            curr_solo_demand[location].put(req)

        for zone in self.map:
            while not curr_solo_supply[zone].empty() and not curr_solo_demand[zone].empty():
                veh = curr_solo_supply[zone].get()
                req = curr_solo_demand[zone].get()

                dest = self.all_demands[req]['drop_off']
                self.fleet_status[veh].update({
                    'dest_grid': dest,
                    'time_2_empty': self.route_solo(zone, dest),
                    'repo_flag': 0,
                    'occupancy': 1
                })
                unassigned_solo[zone].remove(veh)

                self.all_demands[req]['status'] = 'assigned'
                unfulfilled.remove(req)

            while not curr_pool_supply[zone].empty() and not curr_pool_dem_group[zone].empty():
                veh = curr_pool_supply[zone].get()
                group = curr_pool_dem_group[zone].get()
                remain_capacity = 4

                pickup_points = set()
                dropoff_points = set()

                pickup_leader = self.all_demands[group[0]]['pick_up']
                dropoff_leader = self.all_demands[group[0]]['drop_off']

                while group and remain_capacity > 0:
                    remain_capacity -= 1
                    pickup_points.add(self.all_demands[group[0]]['pick_up'])
                    dropoff_points.add(self.all_demands[group[0]]['drop_off'])
                    self.all_demands[group[0]]['status'] = 'pooled'
                    unfulfilled.remove(group[0])
                    curr_pool_demand_grid[self.all_demands[group[0]]['pick_up']].remove(group[0])
                    group.pop(0)

                if group:
                    curr_pool_dem_group[zone].put(group)

                self.fleet_status[veh].update({
                    'dest_grid': dropoff_leader,
                    'time_2_empty': self.route_solo(zone, dropoff_leader)+1,
                    'repo_flag': 0,
                    'occupancy': 4 - remain_capacity
                })
                unassigned_pool[zone].remove(veh)

            unfulfilled_pool_req = sum(1 for req in curr_pool_demand_grid[zone] if self.all_demands[req]['status'] != 'pooled')

            observe[zone] = [curr_solo_supply[zone].qsize(), curr_solo_demand[zone].qsize(),
                             curr_pool_supply[zone].qsize(), unfulfilled_pool_req]

        return observe, unfulfilled, unassigned_solo, unassigned_pool

    def are_adjacent(self, zone1, zone2):
        if zone2 in self.map[zone1]['neighbors'].values():
            return True
        return False

    def get_neighbors(self, zone):
        neighbor_list = []
        for zone in self.map[zone]['neighbors'].values():
            neighbor_list.append(zone)
        return neighbor_list

    def route_solo(self, pickup, dropoff):
        try:
            path = nx.shortest_path_length(self.map_graph, source=pickup, target=dropoff)
            return path
        except nx.NetworkXNoPath:
            return [pickup, dropoff]
        except KeyError:
            return []

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

            grid_id = 0

            # Populate the grid map with cell coordinates and initialize neighbors
            for cell in grid_cells:
                cell_id = str(grid_id)  # Adjust key if different
                coordinates = cell['geometry']['coordinates']
                grid_map[cell_id] = {
                    'coordinates': coordinates,
                    'neighbors': {}
                }
                grid_id += 1

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
            threshold = 3000  # Adjust based on expected distance between hexagons: 2784 meters -> 1.73 miles
            for cell_id, centroid in centroids.items():
                for other_id, other_centroid in centroids.items():
                    if cell_id != other_id and centroid.distance(other_centroid) < threshold:
                        direction = get_direction(centroid, other_centroid)
                        grid_map[cell_id]['neighbors'][direction] = other_id

            with open(map_name + '.pickle', 'wb') as file:
                pickle.dump(grid_map, file)
            print(f"Grid map saved to {map_name + '.pickle'}")

            return grid_map

    def build_hex_grid_graph(self):
        """Build graph from parsed neighbor relationships"""
        G = nx.Graph()

        # Add nodes with positional data
        for zone_id, data in self.map.items():
            centroid = shape({
                'type': 'Polygon',
                'coordinates': data['coordinates']
            }).centroid
            G.add_node(zone_id, pos=(centroid.x, centroid.y))

        # Add edges from pre-computed neighbors
        for zone_id, data in self.map.items():
            for direction, neighbor_id in data['neighbors'].items():
                G.add_edge(zone_id, neighbor_id)

        return G

    def save_all_demands(self, filename=f"all_demands_{time.time() - round(time.time())}.csv"):
        # Convert dictionary to DataFrame
        df = pd.DataFrame.from_dict(self.all_demands, orient="index")

        # Save to CSV
        df.to_csv(filename, index_label="request_id")
        print(f"✅ Saved all demands to {filename}")

    def save_fleet_status(self, filename=f"fleet_status_{time.time() - round(time.time())}.csv"):
        # Convert dictionary to DataFrame
        df = pd.DataFrame.from_dict(self.fleet_status, orient="index")

        # Save to CSV
        df.to_csv(filename, index_label="vehicle_id")
        print(f"✅ Saved fleet status to {filename}")

    def plot_grid_state(self, step):
        """
        Visualizes the hexagonal grid with the number of vehicles and requests per grid.

        Args:
            step (int): The simulation step to visualize.
        """
        # Load grid map
        grid_df = gpd.read_file("data/grid_projected.geojson")

        # Get data for the given step
        step_data = next((data for data in self.visualization_data if data["step"] == step), None)
        if step_data is None:
            print(f"Step {step} not found in data.")
            return

        # Convert distributions to DataFrames
        solo_req_df = pd.DataFrame(list(step_data["solo_req_distribution"].items()),
                                   columns=["grid_id", "solo_req_distribution"])
        pool_req_df = pd.DataFrame(list(step_data["pool_req_distribution"].items()),
                                   columns=["grid_id", "pool_req_distribution"])
        solo_veh_df = pd.DataFrame(list(step_data["solo_veh_distribution"].items()),
                                   columns=["grid_id", "solo_veh_distribution"])
        pool_veh_df = pd.DataFrame(list(step_data["pool_veh_distribution"].items()),
                                   columns=["grid_id", "pool_veh_distribution"])

        # Merge with grid map
        grid_df["grid_id"] = grid_df.index.astype(str)
        grid_df = (grid_df
                   .merge(solo_req_df, on="grid_id", how="left")
                   .merge(pool_req_df, on="grid_id", how="left")
                   .merge(solo_veh_df, on="grid_id", how="left")
                   .merge(pool_veh_df, on="grid_id", how="left"))

        # Fill NaN values (grids without requests or vehicles)
        for col in ["solo_veh_distribution", "pool_req_distribution", "solo_veh_distribution", "pool_veh_distribution"]:
            grid_df[col].fillna(0, inplace=True)

        # **Plot in a 2×2 layout**
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Define subplot titles and data
        plot_data = [
            ("Solo Requests", "solo_req_distribution", "Reds", axes[0, 0]),
            ("Pooled Requests", "pool_req_distribution", "Purples", axes[0, 1]),
            ("Solo Vehicles", "solo_veh_distribution", "Blues", axes[1, 0]),
            ("Pooled Vehicles", "pool_veh_distribution", "Greens", axes[1, 1])
        ]

        # Plot each category
        for title, column, cmap, ax in plot_data:
            grid_df.plot(ax=ax, column=column, cmap=cmap, legend=True, alpha=0.7, edgecolor="black")
            ax.set_title(f"{title} at Step {step}")

            # Add labels
            for _, row in grid_df.iterrows():
                centroid = row.geometry.centroid
                ax.text(centroid.x, centroid.y, f"{int(row[column])}", fontsize=8, ha='center', va='center')

        plt.tight_layout()
        plt.show()