import math
import numpy as np

from tasks.Task import Task


class Inventory(Task):
    
    PRICE = 0.5
    PRODUCTION_COST = 0.1
    STORAGE_COST = 0.02
    PENALTY_COST = 0.1
    TRUCK_COST = 0.1
    TRUCK_CAPACITY = 5
    MEAN_DEMAND_POISSON = 2.5
    PRODUCTION_LIMIT = 10
    SHIPMENT_LIMIT = 10
    STORAGE_LIMIT = 50
    
    def initial_state(self, training=True):
        return (10, 0)
    
    def valid_actions(self):
        return Inventory.PRODUCTION_LIMIT * Inventory.SHIPMENT_LIMIT
        
    def transition(self, state, action):
        
        # what is the demand d_k?
        demand = np.random.poisson(lam=Inventory.MEAN_DEMAND_POISSON, size=(1))
        if demand[0] > state[1]:
            penalty_cost = 0.0  # (demand[0] - state[1]) * Inventory.PENALTY_COST
            demand[0] = state[1]
        else:
            penalty_cost = 0.0
            
        # how much to ship and how much to produce
        to_ship = action // Inventory.PRODUCTION_LIMIT
        to_produce = action - to_ship * Inventory.PRODUCTION_LIMIT
        if to_ship > state[0]: 
            to_ship = state[0]
        if state[1] - demand[0] + to_ship > Inventory.STORAGE_LIMIT:
            to_ship = Inventory.STORAGE_LIMIT - state[1] + demand[0]
        if state[0] - to_ship + to_produce > Inventory.STORAGE_LIMIT:
            to_produce = Inventory.STORAGE_LIMIT - state[0] + to_ship
            
        # rewards and costs
        state_array = np.asarray(state)
        revenue = Inventory.PRICE * np.sum(demand)
        production_cost = Inventory.PRODUCTION_COST * to_produce
        storage_cost = Inventory.STORAGE_COST * np.sum(state_array)
        # penalty_cost = -Inventory.PENALTY_COST * np.sum(((state_array < 0) * state_array)[1:])
        transport_cost = Inventory.TRUCK_COST * math.ceil(to_ship / Inventory.TRUCK_CAPACITY)
        net_reward = revenue - production_cost - storage_cost - transport_cost - penalty_cost
            
        # state update
        new_factory_inventory = state[0] - to_ship + to_produce
        new_store_inventory = state[1] - demand[0] + to_ship
        new_state = (new_factory_inventory, new_store_inventory)
        
        return new_state, net_reward, False

    def number_states(self):
        return (Inventory.STORAGE_LIMIT + 1) * 2

    def default_encoding(self, state):
        arr = np.zeros(self.number_states(), dtype=np.float32)
        arr[state[0]] = 1.0
        arr[Inventory.STORAGE_LIMIT + 1 + state[1]] = 1.0
        arr = arr.reshape((1, -1))
        return arr
