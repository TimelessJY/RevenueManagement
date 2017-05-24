
# coding: utf-8

# In[1]:

import warnings
import numpy as np
from operator import itemgetter
import scipy.stats
import time
from operator import itemgetter
import itertools

import sys
sys.path.append('.')
import RM_helper


# In[45]:

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
            contains value function, ranged over products and remaining capacity
            size n_products * (capacity + 1)
        protection_levels: 2D np array
            contains the time-dependent optimal protection level for each class
            size n_products
        bid_prices: 2D np array
            contains the bid-price for each product with different remaining capacity of the resource
            size (n_products - 1) * (capacity + 1)
    """
    
    value_functions = []
    protection_levels = []
    bid_prices = []
    
    def __init__(self, products, demands, capacity):
        """Return a framework for a single-resource RM problem."""
        self.products = products
        self.demands = demands
        self.capacity = capacity
        self.n_products = len(products)
        
        # Check that the data of demands is specified for each product
        if len(demands) != self.n_products:
            raise ValueError('Size of demands is not as expected.')
        
        # Make sure the products are sorted in descending order based on their revenues
        for j in range(self.n_products-1):
            if products[j][1] < products[j+1][1]:
                raise ValueError('The products are not in the descending order of their revenues.')
        
    def value_func(self):
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
            
#         print("Expected revenue=", self.value_functions[self.n_products-1][self.capacity])
#         print(self.value_functions)
#         print(self.protection_levels)
        
    def get_bid_prices(self):
        if not self.value_functions:
            self.value_func()
        
        self.bid_prices = [[0] * (self.capacity + 1) for _ in range(self.n_products - 1)]
        for j in range(1, self.n_products):
            value_func_prev = self.value_functions[j - 1]
            for x in range(1, self.capacity + 1):
                bid_price = value_func_prev[x] - value_func_prev[x-1]
                self.bid_prices[j - 1][x] = round(bid_price, 3)
                
        return self.bid_prices
    
    def get_protection_levels(self):
        if not self.value_functions:
            self.value_func()
            
        return self.protection_levels

start_time = time.time()
# Examples, ref: example 2.3, 2.4 in "The Theory and Practice of Revenue Management"
products = [[1, 1050], [2,567], [3, 534], [4,520]]
# products = [[1, 1050], [2,950], [3, 699], [4,520]]
demands = [(17.3, 5.8), (45.1, 15.0), (39.6, 13.2), (34.0, 11.3)]
cap = 80
problem = Single_RM_static(products, demands, cap)
# problem.value_func()
# print(problem.get_protection_levels())
# print(problem.get_bid_prices())
# print("--- %s seconds ---" % (time.time() - start_time))


# In[4]:

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
        demands: 2D np array
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
            contains value function, ranged over time periods and remaining capacity
            size total_time * (capacity + 1), i.e. T*(C+1)
        protection_levels: 2D np array
            contains the time-dependent optimal protection level for each class
            size total_time * n_products
            (although it's always zero for all the products in the last time period, 
                and for the last product in each time period)
    """
    
    value_functions = []
    protection_levels = []
    
    def __init__(self, products, demands, capacity, total_time):
        """Return a framework for a single-resource RM problem."""
        self.products = products
        self.demands = demands
        self.capacity = capacity
        self.total_time = total_time
        self.n_products = len(products)
        self.n_demand_periods = len(demands)
        
        # Check that the sequence of demands is specified for each time period
        if self.n_demand_periods > 1 and (len(demands) != total_time or len(demands[0]) != self.n_products):
            raise ValueError('Size of demands is not as expected.')
        
        # Important assumption: at most one demand will occur in each time period
        if ((self.n_demand_periods == 1) and (sum(demands[0]) > 1))             or ((self.n_demand_periods > 1) and any(sum(demands[t]) > 1 for t in range(total_time))):
                raise ValueError('There may be more than 1 demand arriving.')
        
        # Make sure the products are sorted in descending order based on their revenues
        for j in range(self.n_products-1):
            if products[j][1] < products[j+1][1]:
                raise ValueError('The products are not in the descending order of their revenues.')
        
    def value_func(self):
        """Calculate and return  the value functions of this problem."""
        
        self.value_functions = [[0]*(self.capacity+1) for _ in range(self.total_time)] 
        for t in range(self.total_time - 1, -1, -1):
            for x in range(1, self.capacity + 1):
                value = 0
                delta_next_V = 0
                if t < self.total_time - 1:
                    value += self.value_functions[t+1][x]
                    delta_next_V = self.value_functions[t+1][x] - self.value_functions[t+1][x-1]

                for j in range(self.n_products):
                    if self.n_demand_periods > 1:
                        demand_prob = self.demands[t][j]
                    else:
                        demand_prob = self.demands[0][j]
                    rev = self.products[j][1]

                    value += demand_prob * max(rev - delta_next_V, 0)
                
                self.value_functions[t][x] = round(value, 3)
        return self.value_functions
    
    def optimal_protection_levels(self):
        """Calculate and return the optimal protection levels of this problem. """
        
        self.protection_levels = [[0]* self.n_products for _ in range(self.total_time)]
        
        for t in range(self.total_time - 1):
            for j in range(self.n_products - 1):
                for x in range(self.capacity, 0, -1):
                    delta_V = self.value_functions[t+1][x] - self.value_functions[t+1][x-1]
                    if self.products[j+1][1] < delta_V:
                        self.protection_levels[t][j] = x
                        break
        return self.protection_levels

products = [1, 30], [2, 25], [3, 12], [4, 4]
demands = [[0, 0.2, 0, 0.7], [0.2, 0.1, 0, 0.5], [0.1, 0.3, 0.1,0.1]]
# problem = Single_RM_dynamic(products, demands, 3, 3)
# print(problem.value_func())


# In[4]:

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
        demands: 2D np array
            contains requests for products in each period,
            request for product j arrives in period t, if demands[t][j] = revenue_j > 0,
            size total_time * n_products
        capacities: np array
            contains the capacity for each resource
            size n_resources
        total_time: integer
            the max time period T, time period t ranges from 1 to T
        
        To be calculated:
        ----------
        n_states: integer
            the total number of states, based on the given capacities for resources
        incidence_matrix: 2D np array
            indicates which product uses which resources, e.g. incidence_matrix[i][j] = 1 if product j uses resource i
            size n_resources * n_products
        value_functions: 2D np array
            contains value function, ranged over time periods and remaining capacity
            size total_time * n_states
    """
    
    value_functions = []
    protection_levels = []
    incidence_matrix = []
    
    def __init__(self, products, resources, demands, capacities, total_time):
        """Return a framework for a single-resource RM problem."""
        
        self.products = products
        self.resources = resources
        self.demands = demands
        self.capacities = capacities
        self.total_time = total_time
        self.n_products = len(products)
        self.n_resources = len(resources)
        self.n_demand_periods = len(demands)
        
        # Check that the sequence of demands is specified for each time period
        if self.n_demand_periods > 1 and (len(demands) != total_time or len(demands[0]) != self.n_products):
            raise ValueError('Size of demands is not as expected.')
            
        # Check that the capacity for each resource is given
        if len(capacities) != self.n_resources:
            raise ValueError('Number of capacities for resources is not correct.')
        
        # Important assumption: at most one demand will occur in each time period
        if ((self.n_demand_periods == 1) and (sum(demands[0]) > 1))             or ((self.n_demand_periods > 1) and any(sum(demands[t]) > 1 for t in range(total_time))):
                raise ValueError('There may be more than 1 demand arriving.')
        
        # Make sure the products are sorted in descending order based on their revenues
        for j in range(self.n_products-1):
            if products[j][1] < products[j+1][1]:
                raise ValueError('The products are not in the descending order of their revenues.')
            
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
            reduced_cap = [x_i - a_j_i for x_i, a_j_i in zip(cap_vector, incidence_vector)]
            if all(c >= 0 for c in reduced_cap):
                delta = 0
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
            state_x_Au = RM_helper.state_index(self.n_states, self.capacities,x_Au)
            value += self.value_functions[t+1][state_x_Au]
        return value
   
    def value_func(self):
        """Return the value functions of this problem, calculate it if necessary. """
        self.value_functions = [[0] * self.n_states for _ in range(self.total_time)] 
        for t in range(self.total_time - 1, -1, -1):
            if self.n_demand_periods > 1:
                demands = self.demands[t]
            else:
                demands = self.demands[0]
            for x in range(self.n_states): 
                value = 0
                opt_control = self.optimal_control(x, t)
                for j in range(self.n_products):
                    demand = demands[j]
                    j_value = 0
                    if demand > 0:
                        u_j = opt_control[j]
                        j_value = self.eval_value(t, x, j, u_j)
                        
                        value += j_value * demand
                
                demand = 1- sum(demands)
                if t < (self.total_time-1):
                    no_request_val = self.value_functions[t+1][x]
                    value += no_request_val * demand
                self.value_functions[t][x] = round(value, 3)
                
        return self.value_functions
    
    def bid_prices(self):
        """return the bid prices for resources over all time periods and all remaining capacities situations."""
        if not self.value_functions:
            self.value_func()
        return RM_helper.network_bid_prices(self.value_functions, self.products, self.resources, self.capacities,                                             self.incidence_matrix, self.n_states)
        
    def total_expected_revenue(self):
        """returns the expected revenues """
        if not self.value_functions:
            self.value_func()
        
        return self.value_functions[0][-1]

start_time = time.time()

# ps = [['a1', 0.02, 200], ['a2', 0.06, 503], ['ab1', 0.08, 400],['ab2', 0.01, 704], ['b1', 0.05, 601], ['b2', 0.12, 106],\
#             ['bc', 0.03, 920],['c1', 0.07, 832],['d1', 0.14, 397], ['d2',  0.18, 533], ['ad', 0.09, 935], \
#       ['ae', 0.013, 205],['f3', 0.004, 589], ['fb', 0.009, 422]]
# products,demands, _ = RM_helper.sort_product_demands(ps)
# resources = ['a', 'b', 'c', 'd', 'e', 'f']

ps = [['a1', 0.22, 200], ['a2', 0.06, 503], ['ab1', 0.18, 400],['ab2', 0.1, 704], ['b1', 0.05, 601],       ['b2', 0.12, 106], ['bc', 0.13, 920],['c1', 0.07, 832]]
resources = ['a', 'b', 'c']
cap = [16] * 3
T = 6

products,demands, _ = RM_helper.sort_product_demands(ps)
problem = Network_RM(products, resources, [demands], cap, T)

# vf = problem.value_func()
# print(vf)
# print(problem.total_expected_revenue())
# print(problem.bid_prices())

# print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:



