
# coding: utf-8

# In[17]:

import warnings
import numpy as np
from operator import itemgetter
import scipy.stats
import time
import itertools

import sys
sys.path.append('.')
import RM_helper
import RM_demand_model
import RM_evaluator


# In[13]:

##############################
###### Single_RM DP ##########
##############################

class Single_RM_static():
    """Solve a single resource revenue management problem (static model) using Dynamic Programming model, 
        with the following attributes:
    
        Given:
        ----------
        products: 2D np array
            contains products, each represented in the form of [product_name, expected_revenue], 
            ordered in descending order of revenue
            size n_products * 2
        demands: 2D np array
            contains the mean and std of the demand distribution for each product
            size total_time * n_products
        capacity: integer
            the total capacity C, remaining capacity x ranges from 0 to C
        
        To be calculated:
        ----------
        value_functions: 2D np array
            contains value function, ranged over products(from highest fare, to lowest fare), and remaining capacity
            size n_products * (capacity + 1)
        protection_levels: 2D np array
            contains the time-dependent optimal protection level for each class, from the highest fare to lowest
            size n_products
        bid_prices: 2D np array
            contains the bid-price for each product with different remaining capacity of the resource,
            from the highest fare to lowest
            size (n_products - 1) * (capacity + 1)
    """
    
    
    def __init__(self, products, demands, capacity):
        """Return a framework for a single-resource RM problem."""
        self.products = products
        self.demands = demands
        self.capacity = capacity
        self.n_products = len(products)
        
        self.value_functions = []
        self.protection_levels = []
        self.bid_prices = []
        # Check that the data of demands is specified for each product
        if len(demands) != self.n_products:
            raise ValueError('RM_exact: Single_RM_static init(), Size of demands is not as expected.')
        
        # Make sure the products are sorted in descending order based on their revenues
        for j in range(self.n_products-1):
            if products[j][1] < products[j+1][1]:
                raise ValueError('RM_exact: Single_RM_static init(), The products are not in the descending order of                 their revenues.')
        
    def calc_value_func(self):
        """Calculate the value functions of this problem and the protection levels for the products."""
        
        self.value_functions = [[0] * (self.capacity + 1) for _ in range(self.n_products)]
        self.protection_levels = [0] * self.n_products
        
        for j in range(self.n_products):
            
            price = self.products[j][1]
            normal_distr = scipy.stats.norm(self.demands[j][0], self.demands[j][1])
            
            for x in range(self.capacity + 1):    
                val = 0
                dj = 0
                while (normal_distr.pdf(dj) > 1e-5) or (dj < self.demands[j][0]):
                    prob_dj = normal_distr.pdf(dj)
                    if j > 0:
                        u = min(dj, max(x-self.protection_levels[j-1], 0))
                        max_val = price * u + self.value_functions[j-1][x-u]
                    else:
                        u = min(dj, x)
                        max_val = price * u
                        
                    val += prob_dj * max_val
                    dj += 1
                
                self.value_functions[j][x] = val
                
            # calculates protection levels for the current fare class    
            if j < (self.n_products - 1):
                for x in range(self.capacity, 0, -1 ):
                    if self.products[j+1][1] < (self.value_functions[j][x] - self.value_functions[j][x -1]):
                        self.protection_levels[j] = x
                        break
            self.protection_levels[-1] = self.capacity
            
        return self.value_functions

    def get_bid_prices(self):
        if not self.value_functions:
            self.calc_value_func()
        
        self.bid_prices = [[0] * (self.capacity + 1) for _ in range(self.n_products - 1)]
        for j in range(1, self.n_products):
            value_func_prev = self.value_functions[j - 1]
            for x in range(1, self.capacity + 1):
                bid_price = value_func_prev[x] - value_func_prev[x-1]
                self.bid_prices[j - 1][x] = round(bid_price, 3)
                
        return self.bid_prices
    
    def get_protection_levels(self):
        if not self.value_functions:
            self.calc_value_func()
            
        return self.protection_levels
    
    def get_booking_limits(self):
        if not self.protection_levels:
            self.calc_value_func()
            
        booking_limits = [self.capacity] + [self.capacity - self.protection_levels[j-1] for j in range(1, self.n_products)]
        
        return booking_limits

start_time = time.time()
# Examples, ref: example 2.3, 2.4 in "The Theory and Practice of Revenue Management"
products = [[1, 1050], [2,567], [3, 534], [4,520]]
# products = [[1, 1050], [2,950], [3, 699], [4,520]]
demands = [(17.3, 5.8), (45.1, 15.0), (39.6, 13.2), (34.0, 11.3)]
cap = 80
# problem = Single_RM_static(products, demands, cap)
# problem.calc_value_func()
# print(problem.get_protection_levels())
# print(problem.get_bid_prices())
# print(problem.get_booking_limits())
# print("--- %s seconds ---" % (time.time() - start_time))


# In[18]:

##############################
###### Single_RM DP ##########
##############################

class Single_RM_dynamic():
    """Solve a single resource revenue management problem (dynamic model) using Dynamic Programming model, 
        with the following attributes:
    
        Given:
        ----------
        products: 2D np array
            contains products, each represented in the form of [product_name, expected_revenue], 
            ordered in descending order of revenue
            size n_products * 2
        arrival_rates: 2D np array
            contains the probability of a demand for each product
            allows two formats, 1: probabilities in each time period, 2: constant probabilities, time independent
            size total_time * n_products
        capacity: integer
            the total capacity C, remaining capacity x ranges from 0 to C
        total_time: integer
            the max time period T, time period t ranges from 1 to T
        
        To be calculated:
        ----------
        value_functions: 2D np array
            contains value function, ranged over time periods(from t=1, to t = T), and remaining capacity
            size total_time * (capacity + 1), i.e. T*(C+1)
        protection_levels: 2D np array
            contains the time-dependent optimal protection level for each class
            size total_time * n_products
            (although it's always zero for all the products in the last time period, 
                and for the last product in each time period)
        bid_prices: 2D np array
            contains the bid-price at each time period with different remaining capacity of the resource,
            size total_time * (capacity + 1)
    """
    
    def __init__(self, products, arrival_rates, capacity, total_time):
        """Return a framework for a single-resource RM problem."""
        self.products = products
        self.arrival_rates = arrival_rates
        self.capacity = capacity
        self.total_time = total_time
        self.n_products = len(products)
        self.n_arrival_rates_periods = len(arrival_rates)
        
        self.value_functions = []
        self.bid_prices = []
        self.protection_levels = []
        
        # Check that the sequence of arrival_rates is specified for each time period
        if self.n_arrival_rates_periods > 1 and (len(arrival_rates) != total_time or                                                  len(arrival_rates[0]) != self.n_products):
            raise ValueError('RM_exact: Single_RM_dynamic init(), Size of arrival_rates is not as expected.')
        
        # Important assumption: at most one demand will occur in each time period
        if ((self.n_arrival_rates_periods == 1) and (sum(arrival_rates[0]) > 1))             or ((self.n_arrival_rates_periods > 1) and any(sum(arrival_rates[t]) > 1 for t in range(total_time))):
                raise ValueError('RM_exact: Single_RM_dynamic init(), There may be more than 1 demand arriving.')
        
        # Make sure the products are sorted in descending order based on their revenues
        for j in range(self.n_products-1):
            if products[j][1] < products[j+1][1]:
                raise ValueError('RM_exact: Single_RM_dynamic init(), The products are not in the descending order of                 their revenues.')
        
    def calc_value_func(self):
        """Calculate the value functions of this problem backwards from the last time period to the beginning."""
        
        self.value_functions = [[0]*(self.capacity+1) for _ in range(self.total_time)] 
        self.bid_prices = [[0] * (self.capacity + 1) for _ in range(self.total_time)]
        
        for t in range(self.total_time - 1, -1, -1):
            if self.n_arrival_rates_periods > 1:
                arrival_rates_t = self.arrival_rates[t]
            else:
                arrival_rates_t = self.arrival_rates[0]
            for x in range(1, self.capacity + 1):
                value = 0
                delta_next_V = 0
                if t < self.total_time - 1:
                    value += self.value_functions[t+1][x]
                    delta_next_V = self.value_functions[t+1][x] - self.value_functions[t+1][x-1]

                for j in range(self.n_products):
                    rev = self.products[j][1]

                    value += arrival_rates_t[j] * max(rev - delta_next_V, 0)
                
                self.value_functions[t][x] = round(value, 3)
                self.bid_prices[t][x] = self.value_functions[t][x] - self.value_functions[t][x-1]
        return self.value_functions
    
    def get_bid_prices(self):
        if not self.value_functions:
            self.calc_value_func()
        
        return self.bid_prices
    
    def get_protection_levels(self):
        """Calculate and return the time-dependent optimal protection levels of this problem. """
        if not self.value_functions:
            self.calc_value_func()
            
        self.protection_levels = [[0]* self.n_products for _ in range(self.total_time)]
        
        for t in range(self.total_time - 1):
            for j in range(self.n_products - 1):
                price = self.products[j+1][1]
                for x in range(self.capacity, 0, -1):
                    delta_V = self.value_functions[t+1][x] - self.value_functions[t+1][x-1]
                    if price < delta_V:
                        self.protection_levels[t][j] = x
                        break
        return self.protection_levels

# p = [[1, 1050], [2,567], [3, 534], [4,520]]
# T = 10
# ar = [RM_evaluator.sample_random_probs(4, 0.8) for i in range(T)]
# problem = Single_RM_dynamic(p, ar, 10, 10)
# print(problem.calc_value_func())
# print(problem.get_protection_levels())
# print(problem.get_bid_prices())


# In[19]:

##############################
###### Network_RM DP ######### 
##############################

class Network_RM():
    """Solve a multi-resource(network) revenue management problem using Dynamic Programming model,
        with the following attributes:
    
        Given:
        ----------
        products: 2D np array
            contains products, each represented in the form of [product_name, expected_revenue], 
            ordered in descending order of revenue
            size n_products * 2
        resources: np array
            contains names of resources, size n_resources
        capacities: np array
            contains the capacity for each resource
            size n_resources
        total_time: integer
            the max time period T, time period t ranges from 1 to T
        demand_model: RM_demand_model.model
            a model object that specifys the arrival rates of the products at each time period
            
        To be calculated:
        ----------
        n_states: integer
            the total number of states, based on the given capacities for resources
        incidence_matrix: 2D np array
            indicates which product uses which resources, e.g. incidence_matrix[i][j] = 1 if product j uses resource i
            size n_resources * n_products
        value_functions: 2D np array
            contains value function, ranged over time periods(from t=1, to t = T), and remaining capacity
            size total_time * n_states
    """
    
    def __init__(self, products, resources, capacities, total_time, demand_model):
        """Return a framework for a single-resource RM problem."""
        
        self.products = products
        self.resources = resources
        self.capacities = capacities
        self.total_time = total_time
        self.n_products = len(products)
        self.n_resources = len(resources)
        self.demand_model = demand_model
        
        self.value_functions = []
        self.protection_levels = []
        self.incidence_matrix = []
        
        # Check that the capacity for each resource is given
        if len(capacities) != self.n_resources:
            raise ValueError('RM_exact: Network_RM init(), Number of capacities for resources is not correct.')
        
        # Make sure the products are sorted in descending order based on their revenues
        for j in range(self.n_products-1):
            if products[j][1] < products[j+1][1]:
                raise ValueError('RM_exact: Network_RM init(), The products are not in the descending order of their                 revenues.')
            
        self.incidence_matrix = RM_helper.calc_incidence_matrix(products, resources)
        self.calc_number_of_state()
        
    def calc_number_of_state(self):
        """Calculates the number of states in this problem"""
        
        self.n_states = 1
        
        for c in self.capacities:
            self.n_states *= (c+1)
        
    def optimal_control(self, state_num, t):
        """
        helper func: return the optimal control, in time period t, given the state number for the remaining capacities.
        for each product, the optimal control is to accept its demand iff we have sufficient remaining capacity, 
        and its price exceeds the opportunity cost of the reduction in resource capacities 
        required to satisify the request
        """
        cap_vector = RM_helper.remain_cap(self.n_states, self.capacities, state_num)
        
        u = [0] * self.n_products
        
        for j in range(self.n_products):
            incidence_vector = [row[j] for row in self.incidence_matrix]
            reduced_cap = [x - a_j for x, a_j in zip(cap_vector, incidence_vector)]
            if all(c >= 0 for c in reduced_cap):
                delta = 0 # opportunity cost
                if t < self.total_time - 1:
                    reduced_state = RM_helper.state_index(self.n_states, self.capacities, reduced_cap)
                    delta = self.value_functions[t+1][state_num] - self.value_functions[t+1][reduced_state]
                
                if self.products[j][1] >= delta:
                    u[j] = 1
        return u
                
    def eval_value(self, t, state_num, product_num, control_product):
        """helper func: evaluate the value for period t and state x, ref: equation 3.1 in the book"""
        
        value = self.products[product_num][1] * control_product
        incidence_vector = [row[product_num] for row in self.incidence_matrix]
        Au = [x * control_product for x in incidence_vector]
        
        if t < self.total_time - 1:
            curr_x = RM_helper.remain_cap(self.n_states, self.capacities, state_num)
            x_Au = [x_i - Au_i for x_i, Au_i in zip(curr_x, Au)]
            state_x_Au = RM_helper.state_index(self.n_states, self.capacities, x_Au)
            value += self.value_functions[t+1][state_x_Au]
        return value
   
    def calc_value_func(self):
        """Return the value functions of this problem, calculate it if necessary. """
        self.value_functions = [[0] * self.n_states for _ in range(self.total_time)] 
        for t in range(self.total_time - 1, -1, -1):
            arrival_rates_t = self.demand_model.current_arrival_rates(t)
            
            for x in range(self.n_states): 
                value = 0
                opt_control = self.optimal_control(x, t)
                for j in range(self.n_products):
                    arrival_rate = arrival_rates_t[j]
                    j_value = 0
                    if arrival_rate > 0:
                        u_j = opt_control[j]
                        j_value = self.eval_value(t, x, j, u_j)
                        
                        value += j_value * arrival_rate
                
                arrival_rate = 1- sum(arrival_rates_t)
                if t < (self.total_time-1):
                    no_request_val = self.value_functions[t+1][x]
                    value += no_request_val * arrival_rate
                self.value_functions[t][x] = round(value, 3)
                
        return self.value_functions
    
    def get_bid_prices(self):
        """return the bid prices for resources over all time periods and all remaining capacities situations."""
        if not self.value_functions:
            self.calc_value_func()
        return RM_helper.network_bid_prices(self.value_functions, self.products, self.resources, self.capacities,                                             self.incidence_matrix, self.n_states)
        
    def total_expected_revenue(self):
        """returns the expected revenues """
        if not self.value_functions:
            self.calc_value_func()
        
        return self.value_functions[0][-1]

# start_time = time.time()
# p = [['1a', 1050], ['2a',590], ['1b', 801], ['2b', 752], ['1ab', 760,], ['2ab', 1400]]
# r = ['a', 'b']
# c = [3,5]
# ar = [[0.1, 0.2, 0.05, 0.28, 0.14, 0.21]]
# ps = RM_helper.sort_product_revenues(p)
# T = 10
# dm = RM_demand_model.model(ar, T, 1)
# problem = Network_RM(ps, r, c, T, dm)
# print(problem.calc_value_func())
# print(problem.get_bid_prices())
# print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:



