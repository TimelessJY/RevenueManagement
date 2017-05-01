
# coding: utf-8

# In[ ]:

##############################
###### Single_RM DP ##########
##############################
import warnings
import numpy as np
from operator import itemgetter
import scipy.stats
import time

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
            contains value function, ranged over time periods and remaining capacity
            size n_products * (capacity + 1)
        protection_levels: 2D np array
            contains the time-dependent optimal protection level for each class
            size total_time * n_products
    """
    
    value_functions = []
    protection_levels = []
    
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
            for x in range(self.capacity + 1):
                normal_distr = scipy.stats.norm(self.demands[j][0], self.demands[j][1])
                
                val = 0
                for dj in range(self.capacity + 1):
                    prob_dj = normal_distr.pdf(dj)
                    if prob_dj > 0:
                        if j > 0:
                            u = min(dj, max(x-self.protection_levels[j-1], 0))
                            max_val = self.products[j][1] * u + self.value_functions[j-1][x-u]
                        else:
                            u = min(dj, x)
                            max_val = self.products[j][1] * u
                        
                        val += prob_dj * max_val
                
                self.value_functions[j][x] = val
                
                
            # calculates protection levels for the current fare class    
            if j < (self.n_products - 1):
                for x in range(self.capacity, 0, -1 ):
                    if self.products[j+1][1] < (self.value_functions[j][x] - self.value_functions[j][x -1]):
                        self.protection_levels[j] = x
                        break
#         print("Expected revenue=", self.value_functions[self.n_products-1][self.capacity], \
#               ", with protection levels=", self.protection_levels) 
        return (self.value_functions, self.protection_levels)
    
    def marginal_value_check(self):
        """checks whether the marginal values in computed value functions satisfy the proposition 2.21"""
        for j in range(self.n_products):
            delta_Vj= [x-y for x, y in zip(self.value_functions[j][1:], self.value_functions[j])]
            if any(delta_Vj[i] < delta_Vj[i+1] for i in range(self.capacity - 1)):
                print("error type 1")
            if j < (self.n_products -1):
                delta_Vj_next = [x-y for x, y in zip(self.value_functions[j+1][1:], self.value_functions[j+1])]
                if any(delta_Vj[i] > delta_Vj_next[i] for i in range(self.capacity)):
                    print("error type 2")
                    print("at j = ", j)
                    print("delta_vj= ", delta_Vj)
                    print("delta_vj_next= ", delta_Vj_next)
                    print()

# Examples, ref: example 2.3, 2.4 in "The Theory and Practice of Revenue Management"
# products = [[1, 1050], [2,567], [3, 534], [4,520]]
products = [[1, 1050], [2,950], [3, 699], [4,520]]
demands = [(17.3, 5.8), (45.1, 15.0), (39.6, 13.2), (34.0, 11.3)]
problem = Single_RM_static(products, demands, 80)

start_time = time.time()

problem.value_func()
print("--- %s seconds ---" % (time.time() - start_time))


# In[9]:

##############################
###### Single_RM DP ##########
##############################
from operator import itemgetter

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
problem = Single_RM_dynamic(products, demands, 3, 3)
print(problem.value_func())


# In[14]:

##############################
###### Network_RM DP #########
##############################
import itertools

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
        
        self.calc_incidence_matrix()
        self.calc_number_of_state()
        
        
    def calc_incidence_matrix(self):
        """Constructs the incidence matrix, such that A[i][j] = 1 if product j uses resource i, 0 otherwise"""
        
        self.incidence_matrix = [[0] * self.n_products for _ in range(self.n_resources)] 
    
        for i in range(self.n_resources):
            for j in range(self.n_products):
                if self.resources[i] in self.products[j][0]: # test if product j uses resource i
                    self.incidence_matrix[i][j] = 1
        
    def calc_number_of_state(self):
        """Calculates the number of states in this problem"""
        
        self.n_states = 1
        
        for c in self.capacities:
            self.n_states *= (c+1)
        
    def state_number(self, remain_cap):
        """helper func: converts the given array of remaining capacities into the state number"""
        """e.g. given total capacities [1,2,1], and the remained capacities [0, 2, 1], should return 5"""
        
        state_num = 0
        capacity_for_others = self.n_states
        
        for i in range(self.n_resources):
            capacity_for_others /= self.capacities[i] + 1
            state_num += remain_cap[i] * capacity_for_others
        return int(state_num)
        
    def remain_cap(self, state_number):
        """helper func: reverse of function state_number(), to convert the given state number into remained capacities"""
        """e.g. given total capacities [1,2,3] and state_number 5, should return [0, 2, 1]"""
        
        if state_number >= self.n_states:
            raise RuntimeError(
                'Error when converting state number to remained capacities; given state number is too large.')
        
        remain_cap = []
        capacity_for_others = self.n_states
        
        for i in range(self.n_resources):
            capacity_for_others /= self.capacities[i] + 1
            remain_cap.append(int(state_number // capacity_for_others))
            state_number %= capacity_for_others
        return remain_cap
        
    def optimal_control(self, state_num, t):
        """
        helper func: return the optimal control, in time period t, given the state number for the remaining capacities.
        for each product, the optimal control is to accept its demand iff we have sufficient remaining capacity, 
        and its price exceeds the opportunity cost of the reduction in resource capacities 
        required to satisify the request
        """
        cap_vector = self.remain_cap(state_num)
        
        u = [0] * self.n_products
        for j in range(self.n_products):
            incidence_vector = [row[j] for row in self.incidence_matrix]
            diff = [x_i - a_j_i for a_j_i, x_i in zip(incidence_vector, cap_vector)]
            if all(diff_i >= 0 for diff_i in diff):
                delta = 0
                if t < self.total_time - 1:
                    delta = self.value_functions[t+1][state_num] - self.value_functions[t+1][self.state_number(diff)]
                
                if products[j][1] >= delta:
                    u[j] = 1
#         print("optimal control", u)
        return u
                
    def eval_value(self, t, control, state_num, product_num):
        """helper func: evaluate the value for period t and state x, ref: equation 3.1 in the book"""
        price_vector = [0] * self.n_products
        price_vector[product_num] = self.products[product_num][1]
        value = np.dot(price_vector, control)
        Au = np.dot(self.incidence_matrix, control).tolist()
        x_Au = [x_i - Au_i for x_i, Au_i in zip(self.remain_cap(state_num), Au)]
        if t < self.total_time - 1:
            state_x_Au = self.state_number(x_Au)
            value += self.value_functions[t+1][self.state_number(x_Au)]
        return value
        
    def check_valid_control(self, state_num, control):
        """
        helper func: check if the given control is valid within the given state, 
        as we only accept decisions u (i.e. control here) if Au <= x
        """
        Au = np.dot(self.incidence_matrix, control).tolist()
        x_Au = [x_i - Au_i for x_i, Au_i in zip(self.remain_cap(state_num), Au)]
        return all(x_Au_i >= 0 for x_Au_i in x_Au)
    
    def value_func(self):
        """Return the value functions of this problem, calculate it if necessary. """
        
        self.value_functions = [[0] * self.n_states for _ in range(self.total_time)] 
        for t in range(self.total_time - 1, -1, -1):
            for x in range(self.n_states): 
                value = 0
                opt_control = self.optimal_control(x, t)
                for j in range(self.n_products):
                    if self.n_demand_periods > 1:
                        demand = self.demands[t][j]
                    else: 
                        demand = self.demands[0][j]
                    j_value = 0
                    if demand > 0:
                        count_1 = opt_control.count(1)
                        if count_1 == 0:
                            j_value = self.eval_value(t, opt_control, x, j)
                        else:
                            indicies_1 = [i for i, x in enumerate(opt_control) if x == 1]
                            index_to_change =  list(map(list, itertools.product([0, 1], repeat=count_1)))
                            for i in range(2**count_1):
                            # try the optimal control and all possible sub-optimal controls
                                sub_control = list(opt_control)
                                to_change = index_to_change[i] # the indicies to change to get sub-optimal control
                                change_indicies = [i for i, x in enumerate(to_change) if x == 1]
                                for index in change_indicies:
                                    sub_control[indicies_1[index]] = 0

                                if self.check_valid_control(x, sub_control):
                                    sub_val = self.eval_value(t, sub_control, x, j)
                                    j_value = max(j_value, sub_val)
                        value += j_value * demand
                self.value_functions[t][x] = round(value, 3)
                
        return self.value_functions
    
    def bid_price(self, curr_time):
        """Calculate the bid prices for resources at the given time."""
        bid_prices = [0 for _ in range(self.n_states)]
        
products = [ ['12', 500], ['1', 250], ['2', 250]]
resources = ['1', '2']
demands = [[0.4, 0.3, 0.3],[0.8, 0, 0]]
problem = Network_RM(products, resources, demands, [1,1], 2)
print(problem.value_func())


# In[ ]:



