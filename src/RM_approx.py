
# coding: utf-8

# In[2]:

import warnings
import numpy as np
from operator import itemgetter
import scipy.stats
import time
import math
import sys
sys.path.append('.')
import RM_helper
import RM_exact


# In[3]:

##############################
###### Single_EMSR ###########
##############################
class Single_EMSR():
    """Solve a single resource revenue management problem (static model) using EMSR heuristic,
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
            (although it's always zero for all the products in the last time period)
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
        
    def aggregated_demands(self, j):
        """helper func: returns the aggregated future demand for classes j, j - 1, ... 1"""
        return sum(self.demands[k][0] for k in range(j + 1))
            
    def weighted_average_revenue(self, j):
        """helper func: returns the weighted average revenue from calsses 1, ... j"""
        sum_rev_demand = sum(self.products[k][1] * self.demands[k][0] for k in range(j + 1))
        sum_demand = sum(self.demands[k][0] for k in range(j + 1))
        if sum_demand > 0:
            return sum_rev_demand / sum_demand
        else:
            return 0
        
    def calc_protection_levels(self):
        """ 
        calculate and return the protection levels for fare classes
        ref: section 2.2.4.2 EMSR-b method
        """
        self.protection_levels = [0] * self.n_products
        for j in range(self.n_products - 1):
            weighted_average_rev = self.weighted_average_revenue(j)
            future_demands = self.aggregated_demands(j)
            mean = sum(self.demands[k][0] for k in range(j+1))
            std = math.sqrt(sum(self.demands[k][1] ** 2 for k in range(j+1)))
            prob = self.products[j + 1][1] / weighted_average_rev
            distribution = scipy.stats.norm(mean, std)
            yj = round(distribution.isf(prob), 2)
            self.protection_levels[j] = min(yj, self.capacity)
        return self.protection_levels
    
    def calc_val(self, x, j, dj):
        """helper func: calculate value function for remaining capacity x, fare class j, and demand for that class dj"""
        if j > 0:
            u = min(dj, max(x - math.floor(self.protection_levels[j-1]), 0))
            val = self.products[j][1] * u + self.value_functions[j-1][x-u]
        else:
            u = min(dj, max(x, 0))
            val = self.products[j][1] * u
        return val
    
    def value_func(self):
        """Calculate the value functions of this problem, using the protection levels for the products."""
        
        self.value_functions = [[0] * (self.capacity + 1) for _ in range(self.n_products)]
        if not self.protection_levels:
            self.calc_protection_levels()
        
        for j in range(self.n_products):
            for x in range(self.capacity + 1):
                val = 0
                normal_distr = scipy.stats.norm(self.demands[j][0],self.demands[j][1])
                dj = 0
                while (normal_distr.pdf(dj) > 1e-5) or (dj < self.demands[j][0]):
                    prob_dj = normal_distr.pdf(dj)
                    val += prob_dj * self.calc_val(x,j,dj)
                    dj += 1
                
                self.value_functions[j][x] = val

#         print("Expected revenue=", self.value_functions[self.n_products-1][self.capacity], \
#               ", with protection levels=", self.protection_levels) 
        return (self.value_functions, self.protection_levels)
                
    
start_time = time.time()
products = [[1, 1050], [2,567], [3, 534], [4,520]]
# products = [[1, 1050], [2,950], [3, 699], [4,520]]
demands = [(17.3, 5.8), (45.1, 15.0), (39.6, 13.2), (34.0, 11.3)]
# problem = Single_EMSR(products, demands, 80)
# result = problem.value_func()
# print(result)

print("--- %s seconds ---" % (time.time() - start_time))


# In[2]:

##############################
###### Single_DCM ############
##############################

# Solves single-resource capacity control, using Discrete Choice Models.
# ref: The Theory and Practice of Revenue Management, section 2.6.2

# Identifies a list of efficient sets, given information about all possible sets of products(classes)
# ref: section 2.6.2.4
def efficient_sets(products, sets):
    """
    Parameter
    ----------
    products: 2D np array
        contains products, each represented in the form of [product_name, expected_revenue], 
        size n_products * 2
    sets: 2D np array
        contains sets of products, each consists of probabilities of every product
        size n_sets * n_products
        
    Returns
    -------
    effi_sets: 2D np array
        contains efficient sets, each in the form of [products_name, total_probability, total_expected_revenue]
    """
    
    n_products = len(products)  # number of all sets
    n_sets = len(sets) # number of sets
    
    candidate_sets = []
    
    for s in sets:
        total_prob = 0
        total_rev = 0
        set_name = ''
        for i in range(len(s)):
            if s[i] > 0: # i.e. the sets contains the i_th product
                total_prob += s[i]
                total_rev += products[i][1] * s[i]
                set_name += products[i][0]
        candidate_sets.append([set_name, total_prob, total_rev])
    
    effi_sets = []   # stores output
    prev_effi_set = -1    # store the previous efficient set, start with empty set
    prev_prob = 0   # store the choice probability of the previous efficient set
    prev_revenue = 0   # store the revenue of the previous efficient set

    while True:
        next_effi_set = -1     # store the next efficient set
        max_marginal_revenue_ratio = 0
        has_potential_set = False
        for i in range(n_sets):
            if i == prev_effi_set:
                continue
            prob = candidate_sets[i][1]
            revenue = candidate_sets[i][2]
            if prob >= prev_prob and revenue >= prev_revenue:
                has_potential_set = True
                marginal_revenue_ratio = (revenue - prev_revenue) / (prob - prev_prob)
                if marginal_revenue_ratio > max_marginal_revenue_ratio:
                    next_effi_set = i
                    max_marginal_revenue_ratio = marginal_revenue_ratio
        if not has_potential_set:
            # stop if there isn't any potential efficient sets
            break
        elif next_effi_set >= 0:    # if find a new efficient set
            prev_effi_set = next_effi_set
            prev_prob = candidate_sets[next_effi_set][1]
            prev_revenue = candidate_sets[next_effi_set][2]
            effi_sets.append(candidate_sets[next_effi_set])


    return effi_sets

# In nested policy, once identified the efficient sets
# can compute the objective function value to find out which efficient set is optimal for that capacity x.
# ref: section 2.6.2.5
def optimal_set_for_capacity(product_sets, marginal_values):
    """
    Parameter
    ----------
    product_sets: np array
        contains product sets, each in the form of [product_name, prob, revenue], size n_product_sets
    marginal_values: np array
        contains expected marginal value of every capacity at time t+1, size n_capacity
   
    Returns
    -------
    optimal_set: np array
        contains the set_index of the optimal set for capacity x, size n_capacity
    """
    
    n_capacity = len(marginal_values)
    optimal_set = []
    n_product_sets = len(product_sets)
    for i in range(n_capacity):
        max_diff = 0
        curr_opt_set = -1
        for j in range(n_product_sets):
            diff = product_sets[j][2] - product_sets[j][1] * marginal_values[i]
            if diff > max_diff:
                max_diff = diff
                curr_opt_set = product_sets[j][0]
        optimal_set.append(curr_opt_set)
    return optimal_set

                                
# In nested policy, calculate the optimal protection levels for each (efficient) class, at the given time, 
# given the result from value-function
def SINGLE_optimal_protection_levels(product_sets, values, time):
    """
    Parameter
    ----------
    product_sets: np array
        contains product sets, each in the form of [product_name, prob, revenue], size n_product_sets
    value: 2D np array
        contains the value functions, size (max_time + 1) * (total_capacity + 1)
    time: integer
        the discrete time point at which the optimal protection levels are requested
    Returns
    -------
    protection_levels: np array
        contains the optimal protection level for the given product sets at the given time, size n_product_sets
    """

    if not values: 
        return 0
    
    total_time = len(values)
    total_capacity = len(values[0]) - 1
    
    if time > total_time or time < 0:
        return 0
    
    value_delta = []
    for i in range(1, total_capacity + 1):
        value_delta.append(values[time][i] - values[time][i-1])
    
    n_product_sets = len(product_sets)
    protection_levels = []
    for i in range(n_product_sets - 1):
        for capacity in reversed(range(total_capacity)):
            diff = product_sets[i][2] - product_sets[i][1] * value_delta[capacity]
            nextDiff = product_sets[i+1][2] - product_sets[i+1][1] * value_delta[capacity]
            if diff > nextDiff:
                protection_levels.append(capacity + 1)
                break
    protection_levels.append(total_capacity)
    return protection_levels

# Calculates value functions V_t(x) for different remaining capacity, x = 0 ... C
# using backward computations, starting from V_T(x) back to V_0(x)
# ref: function 2.26
def SINGLE_value_function(product_sets, total_capacity, max_time, arrival_rate):
    """
    Parameter
    ----------
    product_sets: np array
        contains sets of products on offer, each in the form of [product_name, prob, revenue], size n_product_sets
        where the prob(probability) and revenue are aggregated values
    total_capacity(C): integer
        the total capacity
    max_time(T): integer
        the number of time periods
    arrival_rate: number
        the probability of arrival of a request, assumed to be constant for all time periods
    Returns
    -------
    value: 2D np array
        contains the value functions, size (max_time + 1) * (total_capacity + 1)
    """
    
    prev_V = [0] * (total_capacity + 1)
    V = []
    for t in range(max_time + 1):
        curr_V = [0] * (total_capacity + 1)
        for x in range(1, total_capacity + 1):
            max_obj_val = np.nan 
            delta = prev_V[x] - prev_V[x-1] # the marginal cost of capacity in the next period
            for s in product_sets:
                if len(s) == 0:
                    continue
                obj_val = s[2] - s[1] * delta # the difference between the expected revenue from offering set S, 
                # and the revenue if a request in set S is accepted
                if np.isnan(max_obj_val) or obj_val > max_obj_val:
                    max_obj_val = obj_val
            max_obj_val *= arrival_rate
            max_obj_val += prev_V[x]
            if np.isnan(max_obj_val):
                max_obj_val = 0
            curr_V[x] = round(max_obj_val, 3)
            
        V.insert(0, curr_V)
        prev_V = curr_V
    return V


# In[15]:

##############################
###### network_DAVN ##########
##############################

class Network_DAVN():
    """Solve a multi-resource(network) revenue management problem using DAVN method,
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
            contains the mean demands and std for products, each in the form of (product_name, [mean_demand, std])
            size n_products * 3 (assume the requests are time-independent), 
        capacities: np array
            contains the capacity for each resource
            size n_resources
        n_class: integer
            the number of virtual classes to partition the products into
            
        To be calculated:
        ----------
        incidence_matrix: 2D np array
            indicates which product uses which resources, e.g. incidence_matrix[i][j] = 1 if product j uses resource i
            size n_resources * n_products
        disp_adjusted_revs: 2D np array
            contains tuples for displacement-adjusted revenues, in the form of (name of product, value),
            these are sorted from the highest value to the lowest, for each resource, size n_resources * n_products
        virtual_classes: np array
            consists virtual classes for every resource, 
            each contains the names of products in that class, and the aggregated revenues of them
            size n_resources    
        aggregated_demands: np array
            consists of aggregated demands for products in each virtual class, for each resource
            size n_resources
        value_functions: 3D np array
            contains the value functions, size n_resources * n_virtual_classes[for each resource] * (capacity_i + 1)
        
    """
    incidence_matrix = []
    disp_adjusted_revs = []
    virtual_classes = []
    aggregated_demands = []
    value_functions = []
    
    def __init__(self, products, resources, demands, capacities, n_class):
        """Return a framework for a single-resource RM problem."""
        
        self.products = products
        self.resources = resources
        self.demands = demands
        self.capacities = capacities
        self.n_class = n_class
        self.n_products = len(products)
        self.n_resources = len(resources)
            
        # Check that the capacity for each resource is given
        if len(capacities) != self.n_resources:
            raise ValueError('Number of capacities for resources is not correct.')
            
        # Check that the demand for each product is given
        if len(demands) != self.n_products:
            raise ValueError('Number of demands for products is not correct.')            
        
        # Make sure the products are sorted in descending order based on their revenues
        for j in range(self.n_products-1):
            if products[j][1] < products[j+1][1]:
                self.products.sort(key = lambda tup: tup[1], reverse=True)
                break
            
        self.incidence_matrix = RM_helper.calc_incidence_matrix(products, resources)
    
    def calc_displacement_adjusted_revenue(self, static_bid_prices):
        """
        helper func: calculate the displacement-adjusted revenues, to approximate the net benefit of accepting 
        the demand for product j using resource i.
        ref: function 3.15
        
        Parameter
        ----------
        static_bid_prices: np array
            contains static bid prices or marginal value estimates, size n_resources
        """
        
        ## Calculates the sum of static bid prices for each product, over all resources it uses
        sum_static_bid_prices = [0] * self.n_products

        for j in range(self.n_products):
            for i in range(self.n_resources):
                if self.incidence_matrix[i][j] == 1:
                    sum_static_bid_prices[j] += static_bid_prices[i]

        ## Calculates the displacement-adjusted revenues, in sorted order
        self.disp_adjusted_revs = [[('', 0)] * self.n_products for _ in range(self.n_resources)] 

        for i in range(self.n_resources):
            for j in range(self.n_products):
                product_name = self.products[j][0]
                disp_value = 0
                if self.incidence_matrix[i][j] == 1: # only calculates for products that uses resource i
                    disp_value = float(self.products[j][1]) - sum_static_bid_prices[j] + static_bid_prices[i]
                self.disp_adjusted_revs[i][j] = (product_name, disp_value)
                
            self.disp_adjusted_revs[i].sort(key = lambda tup: tup[1], reverse=True)
    
    def calc_squared_deviation_of_revenue(self, i, l, k):
        """
        helper func: Calculates the squared deviation of revenue within partition (l, k), for resource i
        ref: example 3.5
        
        Parameter
        ----------
        i: integer
            the index of the resource
        l: integer
            the starting index of the partition, i.e. the index of the first product in this partition
            product index starts from 0
        k: integer
            the ending index of the partition, i.e. the index of the last product in this partition
        """

        if k < l:
            warnings.warn("Wrong indices for the partition")

        if i >= self.n_resources:
            warnings.warn("Resource index out of boundary")

        # calculated the weighted-average displacement adjusted revenue for the given partition
        sum_demands = 0
        demands_times_disp_adjusted_rev = 0
        for j in range(l, k + 1):
            product_name = self.disp_adjusted_revs[i][j][0]
            product_mean_demand = float(next((v[1][0] for v, v in enumerate(self.demands) if v[0] == product_name), 0))
            sum_demands += product_mean_demand
            demands_times_disp_adjusted_rev += product_mean_demand * self.disp_adjusted_revs[i][j][1]
        if sum_demands == 0:
            m = 0
        else:
            m = demands_times_disp_adjusted_rev / sum_demands  

        sqrd_deriv_revenue = 0
        for j in range(l, k + 1):
            product_name = self.disp_adjusted_revs[i][j][0]
            product_mean_demand = float(next((v[1][0] for v, v in enumerate(self.demands) if v[0] == product_name), 0))
            sqrd_deriv_revenue += product_mean_demand * (self.disp_adjusted_revs[i][j][1] - m)**2

        return sqrd_deriv_revenue

    def clustering(self):
        """
        Partition products using each resource into a group of virtual classes.
        This is done by dynamic programming, looking for the partitions that can give the minimum squared deriviation
        of revenue (i.e. total within-group variation)
        ref: section 3.4.3, example 3.5
        """
        self.virtual_classes = [[] for _ in range(self.n_resources)]
        self.aggregated_demands = [[] for _ in range(self.n_resources)]
        for i in range(self.n_resources):
            # only partition products that uses this resource
            n_available_products = self.n_products
            available_products = [j for j, k in enumerate(self.disp_adjusted_revs[i]) if k[1] == 0]
            if available_products:
                n_available_products = available_products[0]

            virtual_classes_for_resource = []
            if n_available_products > 0:
                v = self.calc_squared_deviation_matrix(i, n_available_products)
                virtual_classes_for_resource, demands_for_resource = self.partition_by(v, i, n_available_products)
            self.virtual_classes[i] = virtual_classes_for_resource
            self.aggregated_demands[i] = demands_for_resource
        
#         print("after clustering, classes=",self.virtual_classes, "demand = ", self.aggregated_demands)

    def calc_squared_deviation_matrix(self, resource_index, n_available_products):
        """
        helper func: calculate the minimum squared deviation for the current resource, while trying partition products
        into virtual classes. This is done by dynamic programming based indexing.
        ref: section 3.4.3, example 3.5
        
        Parameter
        ----------
        resource_index: integer
            the index of the resource whose minimum squared deviation is being calculated
        n_available_products: integer
            the number of products that uses the current resource        
        """
        
        # V_c(k) = min(over 1<= l <= k) {c_{lk} + V_{c-1}(l-1)}, note that k, l indexed from 1 onwards,
        # c indexed from 1 (as V_0(k) is not used).
        # indexes l, k used in calc_squared_deviation_of_revenue should start from 0
        
        n_class = min(self.n_class, n_available_products)
        # holds the minimum total squared deviation
        V = [[()]*(n_available_products +1) for _ in range(n_class)] 

        # initialize V_1(k) = c_1k, for k = 1..n_class
        V[0][0] = (0, 0)
        for k in range(1, n_available_products+ 1):
            V[0][k] = (self.calc_squared_deviation_of_revenue(resource_index, 0, k-1), 0)

#         print(V)
        # calculate V_2(k) onwards
        for c in range(1, n_class):
            for k in range(min(c + 1, n_available_products+1)):
                V[c][k] = (0, 0)
            for k in range(c + 1, n_available_products + 1):
                v = np.nan # record the minimum squared deviation
                opt_l = -1 # record the starting index of the partition which gives the minimum squard deviation
                for l in range(1, k + 1):
                    v_new = self.calc_squared_deviation_of_revenue(resource_index, l-1, k-1) + V[c-1][l-1][0]
                    if np.isnan(v) or v_new < v:
                        v = v_new
                        opt_l = l
                V[c][k] = (v, opt_l - 1)
#         print("for resource ", self.resources[resource_index], " V=", V)
        return V

    def partition_by(self, squared_devi_matrix, resource_index, n_available_products):
        """
        helper func: given the minimum squared deviation, return the corresponding virtual classes for the
        current resource, along with its aggregated revenue and demands.
        ref: section 3.4.3, example 3.5
        
        Parameter
        ----------
        resource_index: integer
            the index of the resource whose minimum squared deviation is being calculated
        n_available_products: integer
            the number of products that uses the current resource        
        """
        
        # find the indexes of products corresponding to the minimum squared deviation
        partition_indicies = []
        c = min(self.n_class, n_available_products) - 1
        l = n_available_products
        while True:
            start_index = squared_devi_matrix[c][l][1]
            if start_index == 0 or c == 0:
                break
            if not partition_indicies or start_index != partition_indicies[0]:
                partition_indicies.insert(0, start_index)
            c -= 1
            l -= 1
#         print("indicies for partition of source ", resources[resource_index], " is: ", partition_indicies)
        partition_indicies.append(n_available_products)
    
        # form the virtual classes, and aggregate the demands
        virtual_classes = [] # store all the virtual classes for this resource
        start_index = 0
        demands = [] # store aggregated demands for each virtual class of this resource
        for p in range(len(partition_indicies)):
            names = '' # concatenate the names of all products in this virtual class
            rev = 0
            mean_demand = 0
            variance = 0
            for j in range(start_index, partition_indicies[p]):
                if names:
                    names+=','
                product_name = self.disp_adjusted_revs[resource_index][j][0]
                names+= product_name
                demand = next((v[1] for v, v in enumerate(self.demands) if v[0] == product_name), 0)
                mean_demand += demand[0]
                variance += demand[1] ** 2
                rev += demand[0] * float(next((v[1] for v,v in enumerate(self.disp_adjusted_revs[resource_index]) 
                                             if v[0]==product_name),0))
                
            if mean_demand > 0:
                rev /= mean_demand
            mean_demand /= partition_indicies[p] - start_index
            
            start_index = partition_indicies[p]
            virtual_classes.append([names, round(rev, 3)])
            demands.append((round(mean_demand, 3), round(math.sqrt(variance), 3)))
            
        # sort virtual classes and demands based on descending order of revenues
        demands = [d for (v, d) in sorted(zip(virtual_classes, demands), key=lambda x:x[0][1], reverse = True)]
        virtual_classes = sorted(virtual_classes, key=lambda x:x[1], reverse = True)
        
        return (virtual_classes, demands)
    
    
    def calc_value_function(self, static_price):
        """
        Main Function:
        Calculates the value-function estimate for this DAVN problem, by clustering products into virtual classes 
        and then solving a single-resource problem
        
        Parameter
        ----------
        static_price: np array
            contains static bid prices or marginal value estimates, size n_resources
        """
        if len(static_price) != self.n_resources:
            raise ValueError('Static bid prices size not as expected')
        
#         print("received products, r, d = ", self.products, self.resources, self.demands)
        self.value_functions = []
        self.calc_displacement_adjusted_revenue(static_price)
        self.clustering()
        
        total_exp_rev = 0
        for i in range(self.n_resources):
#             print("vc=",self.virtual_classes[i],", demand=",self.aggregated_demands[i], ", cap=", self.capacities[i])
            single_res_prob = RM_exact.Single_RM_static(self.virtual_classes[i], self.aggregated_demands[i], 
                                                 self.capacities[i])
            value_func = single_res_prob.value_func()[0]
            self.value_functions.append(value_func)
            total_exp_rev += value_func[len(self.virtual_classes[i]) - 1][self.capacities[i]]
#         print("total_exp_rev = ", total_exp_rev)
        return (self.value_functions, total_exp_rev)
        
products = [['1a', 1050], ['2a',950], ['3a', 699], ['4a',520],['1b', 501], ['2b', 352], ['3b', 722], ['1ab', 760],            ['2ab', 1400]]
demands = [['1a', (17.3, 5.8)], ['2a', (45.1, 15.0)],['3a', (39.6, 13.2)],['4a', (34.0, 11.3)], ['1b', (20, 3.5)],            ['2b', (63.1, 2.5)], ['3b', (22.5, 6.1)], ['1ab', (11.5, 2.1)], ['2ab', (24.3, 6.4)]]
resources=['a', 'b']
n_virtual_class = 2
static_price = [0, 0]
# capacities = [130,130]
capacities = [60, 60]

# davn_prob = Network_DAVN(products, resources, demands, capacities,n_virtual_class)
# vf = davn_prob.calc_value_function(static_price)
# print(vf)


# In[5]:

##############################
###### iterative_DAVN ########
##############################

# Implement the iterative displacement-adjusted virtual nesting(DAVN) method for network RM problem
# The result is static bid prices estimated, either converged, or after a large number of computation rounds.
# ref: section 3.4.5.1
def iterative_DAVN(products, resources, demands, n_virtual_class, capacities, remain_cap):
    """
    Parameter
    ----------
    products: 2D np array
            contains products, each represented in the form of [product_name, expected_revenue], 
            ordered in descending order of revenue
            size n_products * 2
    resources: np array
        contains names of resources, size n_resources
    demands: 2D np array
        contains the mean demands and std for products, each in the form of (product_name, [mean_demand, std])
        size n_products * 3 (assume the requests are time-independent), 
    capacities: np array
        contains the capacity for each resource
        size n_resources
   
    Returns
    -------
    static_bid_prices: np array
        contains static bid prices, size n_resources
    """
    
    THRESHOLD = 0.001
    
    n_resources = len(resources) # number of resources
    n_products = len(products) # number of products
    
    # Step 0: initialize
    # initialize the static prices, one for each resource
    static_bid_prices = []
    static_bid_prices.append([0 for x in range(n_resources)])
    
    k = 0
    
    davn_prob = Network_DAVN(products, resources, demands, capacities,n_virtual_class)
    
    while k < 100:
        # Step 1: compute new displacement-adjusted revenues, compute value-function estimated using DAVN method
#         print("calculating value function using: ", static_bid_prices[k])
        value_funcs = davn_prob.calc_value_function(static_bid_prices[k])
        
#         print('value func: ', value_funcs)
        deltas = []
        for i in range(n_resources):
            value_func_i = value_funcs[0][i]
            capacity_i = remain_cap[i]
#             print(" i = ", i, ", value_func = ", value_func_i)
            delta = round(value_func_i[-1][capacity_i] - value_func_i[-1][capacity_i - 1], 4)
            deltas.append(delta)

        # Step 2: check for convergence
        if all(abs(deltas[i]-static_bid_prices[k][i]) < THRESHOLD for i in range(n_resources)):
            static_bid = [round(elem, 3) for elem in static_bid_prices[k]]
            print("stop at k = ", k, ", with static_bid_prices = ", static_bid, ", with total expected revenue=", value_funcs[1])
            return (static_bid, value_funcs[1])
        else:
            static_bid_prices.append(deltas)
            k+= 1
            
    static_bid = [round(elem, 3) for elem in static_bid_prices[k]]
    print("after 100 rounds, haven't converged")
    return (static_bid_prices, value_funcs[1])
    
products = [['1a', 1050], ['2a',950], ['3a', 699], ['4a',520],['1b', 501], ['2b', 352], ['3b', 722], ['1ab', 760],            ['2ab', 1400]]

demands = [['1a', (17.3, 5.8)], ['2a', (45.1, 15.0)],['3a', (39.6, 13.2)],['4a', (34.0, 11.3)], ['1b', (20, 3.5)],            ['2b', (63.1, 2.5)], ['3b', (22.5, 6.1)], ['1ab', (11.5, 2.1)], ['2ab', (24.3, 6.4)]]

resources=['a', 'b']
capacities = [130,130]

# iterative_DAVN(products, resources, demands, 1, capacities, capacities)


# In[4]:

##############################
###### network_DLP ###########
##############################
import cvxopt
from cvxopt import matrix, solvers, glpk

def network_DLP(products, resources, demands, capacities):
    """Solve a multiple-resource revenue management problem using Deterministic Linear Programming.
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
        total_time: integer
            the max time period T, time period t ranges from 1 to T
    """
    n_products = len(products)
    n_resources = len(resources)
        
    if len(demands) > 1:
        D = [[sum(d[j][0]) for d in demands] for j in range(n_products)]
        mu = [d/len(demands) for d in D]
    else:
        D = [d[0] for d in demands[0]]
        mu = D
    
    A = RM_helper.calc_incidence_matrix(products, resources)
    c = matrix([-float(product[1]) for product in products])
    h = np.array(capacities + [0] * n_products + mu).astype(float)
    h = matrix(h)
    G = [0] * n_products
    for i in range(n_products):
        row = helper_row_G(A, i, n_products)
        G[i] = row
    G = np.asarray(G).astype(float).tolist()
    G = matrix(G)
    cvxopt.solvers.options['show_progress'] = False
    sol = solvers.lp(c, G, h)
    print(sol['x'])
    print("sol1:val=",sol['primal objective'])
    print("sol1:dual_sol=", sol['dual objective'])
    print("sol1:dual_var=", sol['z'])
    
    print("int solutions:")
    integer_index = [i for i in range(n_products)]
    (status, x) = glpk.ilp(c,G,h, I=set(integer_index))
    int_sol=np.array(x)
    print("value=", round((c.T*x)[0], 2))
    
def helper_row_G(A, i, n_products):
    row1 = [a[i] for a in A]
    row2 = [0] * n_products
    row2[i] = -1
    row3 = [0] * n_products
    row3[i] = 1
    return row1+row2+row3
    
products = [['1a', (17.3, 5.8), 1050], ['2a', (45.1, 15.0),950], ['3a', (39.6, 13.2), 699], ['4a', (34.0, 11.3),520],            ['1b', (20, 3.5), 501], ['2b', (63.1, 2.5), 352], ['3b', (22.5, 6.1), 722], ['1ab', (11.5, 2.1), 760],            ['2ab', (24.3, 6.4), 1400]]

resources = ['a', 'b']
capacities = [30,30]
np.set_printoptions(threshold='nan')
(pros, demands, demands_with_names) = RM_helper.sort_product_demands(products)
# print(pros)
# print(demands)
# network_DLP(pros, resources, [demands], capacities)


# In[ ]:



