
# coding: utf-8

# In[243]:

##############################
###### Single_RM DP ##########
##############################
import warnings
import numpy as np
from operator import itemgetter


class Single_RM():
    """A single resource revenue management problem, with the following attributes:
    
        Given:
        ----------
        products: 2D np array
            contains products, each represented in the form of [product_name, expected_revenue], 
            ordered in descending order of revenue
            size n_products * 2
        demands: 2D np array
            contains the probability of a demand for each product in each time period
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
            size (total_time - 1) * (n_products - 1)
    """
    
    value_functions = []
    protection_levels = []
    
    def __init__(self, products, demands, capacity, total_time = len(demands)):
        """Return a framework for a single-resource RM problem."""
        self.products = products
        self.demands = demands
        self.capacity = capacity
        self.total_time = total_time
        self.n_products = len(products)
        
        # Check that the sequence of demands is specified for each time period
        if len(demands) != total_time or len(demands[0]) != self.n_products:
            raise ValueError('Size of demands is not as expected.')
        
        # Important assumption: at most one demand will occur in each time period
        for t in range(total_time):
            if sum(demands[t]) > 1:
                raise ValueError('There may be more than 1 demand arriving in period '+ str(t+1) + '.')
        
        # Make sure the products are sorted in descending order based on their revenues
        for j in range(self.n_products-1):
            if products[j][1] < products[j+1][1]:
                raise ValueError('The products are not in the descending order of their revenues.')
        
    def value_func(self):
        """Return the value functions of this problem, calculate it if necessary. """
        
        if len(self.value_functions) > 0:
            return self.value_functions
        
        value_functions = [[0]*(self.capacity+1) for _ in range(self.total_time)] 
        for t in range(self.total_time - 1, -1, -1):
            for x in range(1, self.capacity + 1):
                value = 0
                if t + 1 == self.total_time:
                    delta_next_V = 0
                else:
                    delta_next_V = value_functions[t+1][x] - value_functions[t+1][x-1]

                for j in range(self.n_products):
                    demand_prob = self.demands[t][j]
                    rev = self.products[j][1]

                    value += demand_prob * max(rev - delta_next_V, 0)
                value_functions[t][x] = round(value, 3)
        self.value_functions = value_functions
        return value_functions
    
    def optimal_protection_levels(self):
        """Return the optimal protection levels of this problem, calculate it if necessary. """
        
        if len(self.protection_levels) > 0:
            return self.protection_levels
        
        protection_levels = [[0]*(self.n_products - 1) for _ in range(self.total_time - 1)]
        
        for t in range(self.total_time - 1):
            for j in range(self.n_products - 1):
                for x in range(self.capacity, 0, -1):
                    delta_V = self.value_functions[t+1][x] - self.value_functions[t+1][x-1]
#                     print("delta: ", delta_V)
                    if self.products[j+1][1] < delta_V:
                        protection_levels[t][j] = x
#                         print("for t,j=", t, j, ", x = ", x)
                        break
        self.protection_levels = protection_levels
        return protection_levels

products = [1, 30], [2, 25], [3, 12], [4, 4]
demands = [[0, 0.2, 0, 0.7], [0.2, 0.1, 0, 0.5], [0.1, 0.3, 0.1,0.1]]
problem = Single_RM(products, demands, 3, 3)
print(problem.value_func())

problem.optimal_protection_levels()


# In[244]:

##############################
###### Network_RM DP #########
##############################
import itertools
class Network_RM():
    """A multi-resource(network) revenue management problem, with the following attributes:
    
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
    incidence_matrix
    
    def __init__(self, products, resources, demands, capacities, total_time = len(demands)):
        """Return a framework for a single-resource RM problem."""
        
        self.products = products
        self.resources = resources
        self.demands = demands
        self.capacities = capacities
        self.total_time = total_time
        self.n_products = len(products)
        self.n_resources = len(resources)
        
        # Check that the sequence of demands is specified for each time period
        if len(demands) != total_time or len(demands[0]) != self.n_products:
            raise ValueError('Size of demands is not as expected.')
            
        # Check that the capacity for each resource is given
        if len(capacities) != self.n_resources:
            raise ValueError('Number of capacities for products is not correct.')
        
        # Important assumption: at most one demand will occur in each time period
        # Demands should be specified so that demands[t][j] = revenue_j if demand for product j occurs in time period t
        for t in range(total_time):
            if sum(1 for x in demands[t] if x > 0) > 1:
                raise ValueError('There are more than 1 demand arriving in period '+ str(t+1) + '.')
            if sum(1 for j in range(self.n_products) if (demands[t][j] > 0 and demands[t][j] != products[j][1])) > 0:
                raise ValueError('Demand not specified correctly, in period '+ str(t+1))
        
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
#         print("for state ", state_num, ", remain cap=", cap_vector)
        
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
                
    def eval_value(self, t, control, state_num):
        """helper func: evaluate the value for period t and state x, ref: equation 3.1 in the book"""
        demand_t = self.demands[t]
        value = np.dot(demand_t, control)
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
#         print("checking control, x-Au = ", x_Au)
        return all(x_Au_i >= 0 for x_Au_i in x_Au)
    
    def value_func(self):
        """Return the value functions of this problem, calculate it if necessary. """
        
        if len(self.value_functions) > 0:
            return self.value_functions
        
        self.value_functions = [[0] * self.n_states for _ in range(self.total_time)] 
        for t in range(self.total_time - 1, -1, -1):
            for x in range(self.n_states): 
                opt_control = self.optimal_control(x, t)
                value = 0
                count_1 = opt_control.count(1)
                if count_1 == 0:
                    value = self.eval_value(t, opt_control, x)
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
#                         print("new control: ", sub_control)

                        if self.check_valid_control(x, sub_control):
                            sub_val = self.eval_value(t, sub_control, x)
#                             print("sub value: ", sub_val)
                            value = max(value, sub_val)
#                 print("t, x = ", t, x, ", value = ", value)
                self.value_functions[t][x] = round(value, 2)
                
        return self.value_functions
    
products = ['12', 30], ['2', 25], ['13', 12], ['23', 4]
resources=['1', '2', '3']
demands = [[0, 0, 12,0],[30, 0,0,0],[0,0,0,0]]
problem = Network_RM(products, resources, demands, [1,2,1], 3)
print(problem.value_func())


# In[ ]:



