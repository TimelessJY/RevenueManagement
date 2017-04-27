
# coding: utf-8

# In[6]:

##############################
###### Single_DCM ############
##############################

import warnings
import numpy as np

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


# effi_sets = [['Y', 0.3, 240], ['YK', 0.8, 465], ['YMK', 1, 505]]
# values = calc_value_function(effi_sets, 10, 10, 0.2)
# # print(len(values), len(values[0]))
# print(values)
# # print(values[9][2])

# products = [['Y', 800], ['M',500], ['K',450]]
# marginal_values = [780, 624, 520, 445.71, 390,346.67, 312.00, 283.64, 260.00,\
#                                          240,222.86,208,195,183.53,173.33,164.21,156,148.57,141.82,135.65]
    
# sets= [[0.3, 0, 0], [0, 0.4, 0], [0, 0, 0.5], [0.1, 0.6, 0], [0.3,0,0.5], [0,0.4,0.5], [0.1, 0.4,0.5]]
# efficient_sets = efficient_sets(products, sets)
# print("efficient-sets", efficient_sets, "\n")
# values = SINGLE_value_function(efficient_sets, 20, 10, 0.5)



# value = calc_value_function([['AB', 0.25, 100.0]], 1, 1, 0.5)
# print(values)

# print(SINGLE_optimal_protection_levels(efficient_sets, values, 6))

sets = [['AB', 0.5, 50.0]]
print(SINGLE_value_function(sets, 1, 2, 2.5) )


# In[113]:

##############################
###### network_DAVN ##########
##############################

import sys
sys.path.append('/Users/jshan/Desktop/RevenueManagement')
from src import RM_exact

def convert_to_weights(nums):
    """Scale a list of numbers so that their sum becomes 1."""
    total = sum(nums)
    if total > 1:
        weighted = [round(x / total, 4) for x in nums]
    return weighted

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
        #demands: 2D np array
            contains requests for products in each period,
            request for product j arrives in period t, if demands[t][j] = revenue_j > 0,
            size total_time * n_products
        capacities: np array
            contains the capacity for each resource
            size n_resources
        total_time: integer
            the max time period T, time period t ranges from 1 to T
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
            contains the value functions, size n_resources * total_time * capacity_i
        
    """
    incidence_matrix = []
    disp_adjusted_revs = []
    virtual_classes = []
    aggregated_demands = []
    value_functions = []
    
    def __init__(self, products, resources, demands, capacities, total_time, n_class):
        """Return a framework for a single-resource RM problem."""
        
        self.products = products
        self.resources = resources
        self.demands = demands
        self.capacities = capacities
        self.total_time = total_time
        self.n_class = n_class
        self.n_products = len(products)
        self.n_resources = len(resources)
            
        # Check that the capacity for each resource is given
        if len(capacities) != self.n_resources:
            raise ValueError('Number of capacities for resources is not correct.')
        
        # Make sure the products are sorted in descending order based on their revenues
        for j in range(self.n_products-1):
            if products[j][1] < products[j+1][1]:
#                 self.products.sort(key = lambda tup: tup[1], reverse=True)
#                 break
                raise ValueError('The products are not in the descending order of their revenues.')
        

        if n_class > self.n_products:
            warnings.warn("More virtual classes than number of products")
            
        self.calc_incidence_matrix()
        
    def calc_incidence_matrix(self):
        """helper func: constructs the incidence matrix"""
        
        self.incidence_matrix = [[0] * self.n_products for _ in range(self.n_resources)] 
    
        for i in range(self.n_resources):
            for j in range(self.n_products):
                if self.resources[i] in self.products[j][0]: # test if product j uses resource i
                    self.incidence_matrix[i][j] = 1
    
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
            
        return self.disp_adjusted_revs
    
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
        
        Returns
        -------
        sqrd_deriv_revenue: number
            the squared deviation of revenue within the given partition
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
            product_mean_demand = float(next((v[1] for v, v in enumerate(self.demands) if v[0] == product_name), 0))
            sum_demands += product_mean_demand
            demands_times_disp_adjusted_rev += product_mean_demand * self.disp_adjusted_revs[i][j][1]
        if sum_demands == 0:
            m = 0
        else:
            m = demands_times_disp_adjusted_rev / sum_demands  

        sqrd_deriv_revenue = 0
        for j in range(l, k + 1):
            product_name = self.disp_adjusted_revs[i][j][0]
            product_mean_demand = float(next((v[1] for v, v in enumerate(self.demands) if v[0] == product_name), 0))
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
        for i in range(self.n_resources):
            # only partition products that uses this resource
            n_available_products = self.n_products
            available_products = [j for j, k in enumerate(self.disp_adjusted_revs[i]) if k[1] == 0]
            if available_products:
                n_available_products = available_products[0]

            virtual_classes_for_resource = []
            if n_available_products > 0:
                v = self.calc_squared_deviation_matrix(i, n_available_products)
                virtual_classes_for_resource = self.partition_by(v, i, n_available_products)
            self.virtual_classes[i] = virtual_classes_for_resource
        
#         print("after clustering, classes=",self.virtual_classes, "demand = ", self.aggregated_demands)

    def calc_squared_deviation_matrix(self, resource_index, n_available_products):
        """
        helper func: calculate the minimum squared deviation for the current resource, while trying partition products
        into virtual classes. This is done by dynamic programming.
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
        
        # holds the minimum total squared deviation
        V = [[()]*(n_available_products +1) for _ in range(self.n_class)] 

        # initialize V_1(k) = c_1k, for k = 1..n_class
        V[0][0] = (0, 0)
        for k in range(1, n_available_products+ 1):
            V[0][k] = (self.calc_squared_deviation_of_revenue(resource_index, 0, k-1), 0)

#         print(V)
        # calculate V_2(k) onwards
        for c in range(1, self.n_class):
            for k in range(min(c + 2, n_available_products+1)):
                V[c][k] = (0, 0)
            for k in range(c + 2, n_available_products + 1):
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
        helper func: given the minimum squared deviation, return the corresponding virtual classes for the current resource.
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
        c = self.n_class - 1
        l = n_available_products
        while True:
            start_index = squared_devi_matrix[c][l][1]
            if start_index == 0 or c == 0:
                break
            if not partition_indicies or start_index != partition_indicies[0]:
                partition_indicies.insert(0, start_index)
            c -= 1
            l -= 1
#         print("indicies for partition of source ", resources[i], " is: ", partition_indicies)
        partition_indicies.append(n_available_products)
    
        # form the virtual classes, and aggregate the demands
        virtual_classes = [] # store all the virtual classes for this resource
        start_index = 0
        demands = [] # store aggregated demands for each virtual class of this resource
        for p in range(len(partition_indicies)):
            names = '' # concatenate the names of all products in this virtual class
            revs = 0
            demand_prob = 0
            for j in range(start_index, partition_indicies[p]):
                if names:
                    names+=','
                product_name = self.disp_adjusted_revs[resource_index][j][0]
                names+= product_name
                demand = float(next((v[1] for v, v in enumerate(self.demands) if v[0] == product_name), 0))
                demand_prob += demand
                revs += demand * float(next((v[1] for v,v in enumerate(self.disp_adjusted_revs[resource_index]) 
                                             if v[0]==product_name),0))
                
            start_index = partition_indicies[p]
            virtual_classes.append([names, round(revs, 3)])
            demands.append(round(demand_prob, 3))
            
        demands = convert_to_weights(demands)
        # sort virtual classes and demands based on descending order of revenues
        demands = [d for (v, d) in sorted(zip(virtual_classes, demands), key=lambda x:x[0][1], reverse = True)]
        virtual_classes = sorted(virtual_classes, key=lambda x:x[1], reverse = True)
        
        self.aggregated_demands.append([demands])
#         print("virtual classes for ", resources[resource_index], " is: ", virtual_classes, " demand is: ", demands)
        return virtual_classes
    
    # Main function, calculates the value-function estimate for DAVN problem,
    # by clustering products into virtual classes and then solving a single-resource problem
    def network_DAVN_value_function(self, static_price, arrival_rate):
        """
        Main Function:
        Calculates the value-function estimate for this DAVN problem, by clustering products into virtual classes 
        and then solving a single-resource problem
        
        Parameter
        ----------
        static_price: np array
            contains static bid prices or marginal value estimates, size n_resources
        arrival_rate: number
            the probability of arrival of a request, assumed to be constant for all time periods
        """
        
        self.calc_displacement_adjusted_revenue(static_price)
        self.clustering()
        
        for i in range(self.n_resources):
            single_res_prob = RM_exact.Single_RM(self.virtual_classes[i], self.aggregated_demands[i], 
                                                 self.capacities[i], self.total_time)
            value_func = single_res_prob.value_func()
            self.value_functions.append(value_func)
        return self.value_functions
            
products =[['AEB1', 420], ['CAE1', 330], ['EB1', 320], ['AEB2', 290], ['CA1', 280], ['CAE2', 260], ['AE1', 220], 
           ['EB2', 220], ['CA2', 190], ['AEB3', 190], ['AE2', 150], ['CAE3', 150], ['EB3', 140], 
           ['CA3', 110], ['AE3', 80]]
resources = ['AE', 'EB', 'CA']
demands = [['AE1', 0 ], ['AE2', 0], ['AE3', 0.5 ], ['EB1',0.2 ],['EB2', 0.3],['EB3', 0.4 ], ['CA1', 0.14 ],
           ['CA2', 0.29],['CA3',0.43], ['CAE1',0 ],['CAE2',0.33 ], ['CAE3',0.33],['AEB1', 0.2 ],
           ['AEB2', 0.3],['AEB3', 0.4]]

n_virtual_class = 3
static_price = [0, 0, 0]
capacities = [1,1,1]

davn_prob = Network_DAVN(products, resources, demands, capacities, 3, n_virtual_class)
# print(davn_prob.calc_displacement_adjusted_revenue(static_price))
# davn_prob.clustering()
print(davn_prob.network_DAVN_value_function(static_price, 0.3))


# In[112]:

##############################
###### iterative_DAVN ########
##############################

# Implement the iterative displacement-adjusted virtual nesting(DAVN) method for network RM problem
# The result is static bid prices estimated, either converged, or after a large number of computation rounds.
# ref: section 3.4.5.1
def iterative_DAVN(products, resources, n_virtual_class, total_capacity, max_time, arrival_rate,                    current_time):
    """
    Parameter
    ----------
    products: np array
        contains products, each in the form of [name, probabilities, revenue], size n_products
    resources: np array
        contains names of resources, size n_resources
    n_virtual_class: integer
        the number of virtual classes to partition the products into
    mean_demands: np array
        contains mean demands of products, in the form of [product_name, mean_demand], size n_products
    total_capacity(C): integer
        the total capacity
    max_time(T): integer
        the number of time periods
    arrival_rate: number
        the probability of arrival of a request, assumed to be constant for all time periods
    current_time: integer
        the current time period
   
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
    
    while k < 100:
    
        # Step 1: compute new displacement-adjusted revenues, compute value-function estimated using DAVN method
        print("calculating value function using: ", static_bid_prices[k])
        value_funcs = network_DAVN_value_function(products, resources, static_bid_prices[k],                                                n_virtual_class, total_capacity, max_time, arrival_rate)
        print('value func: ', value_funcs)
        deltas = []
        for i in range(n_resources):
            value_func_i = value_funcs[i][current_time]
#             value_func_i = value_funcs[i][max_time]
            print(" i = ", i, ", value_func = ", value_func_i)
#             delta = value_func_i[total_capacity - 1] - value_func_i[total_capacity - 2]
            delta = round(value_func_i[total_capacity] - value_func_i[total_capacity - 1], 4)
            deltas.append(delta)

        # Step 2: check for convergence
        convergent = True
        for i in range(n_resources):
            if abs(deltas[i]-static_bid_prices[k][i]) >= THRESHOLD:
                convergent = False
                break

        if not convergent:
            static_bid_prices.append(deltas)
            k += 1
        else:
            print("stop at k = ", k, ", with static_bid_prices = ", static_bid_prices[k])
            return static_bid_prices[k]
        
    print("after 100 rounds, haven't converged")
    return static_bid_prices[k]
    
# products = [['AB', 0.5, 100], ['CD', 0.5, 156], ['ABC', 0.5, 800], ['BCD',0.5, 1000]]

# products = [['AB', 0.5, 100], ['CD', 0.5, 100]]
# products = [['AB', 0.05, 2000], ['CD', 0.25, 500], ['ABC', 0.5, 700], ['BCD', 0.2, 200]]
products = [['AB', 0.2, 2000], ['CD', 0.1, 500], ['ABC', 0.5, 700], ['BCD', 0.2, 200]]
resources =  ['AB', 'BC', 'CD']
n_virtual_class = 2
# static_price = [0.5, 0.5, 0.5]
# disp_adjusted_revenue = calc_displacement_adjusted_revenue(products, resources, static_price)
# calculate_value_function(products, resources, static_price, n_virtual_class, mean_demands, 10, 2, 0.5)

iterative_DAVN(products, resources, n_virtual_class, 2, 10, 0.3, 8)

# network_DAVN_value_function(products, resources, static_price, n_virtual_class, mean_demands, 1, 2, 5)


# 

# In[ ]:



