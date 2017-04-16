
# coding: utf-8

# In[140]:

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


# In[141]:

##############################
###### network_DAVN ##########
##############################

# Calculate the displacement-adjusted revenues, 
# which is to approximate the net benefit of accepting product j on resource i
# ref: function 3.15
def calc_displacement_adjusted_revenue(products, resources, static_bid_prices):
    """
    Parameter
    ----------
    products: np array
        contains products, each in the form of [name, probabilities, revenue], size n_products
    resources: np array
        contains names of resources, size n_resources
    static_bid_prices: np array
        contains static bid prices or marginal value estimates, size n_resources
   
    Returns
    -------
    disp_adjusted_revs: 2D np array
        contains tuples for displacement-adjusted revenues, in the form of (value, name of product),
        these are sorted from the highest value to the lowest, for each resource, size n_resources * n_products
    """
    
    
    n_resources = len(resources) # number of resources
    n_products = len(products) # number of products
    
    ## Constructs the incidence matrix, such that A[i][j] = 1 if product j uses resource i, 0 otherwise
    A = [[0]*n_products for _ in range(n_resources)] 
    
    for i in range(n_resources):
        for j in range(n_products):
            if resources[i] in products[j][0]: # test if product j uses resource i
                A[i][j] = 1
    
    ## Calculates the sum of static bid prices for each product, over all resources it uses
    sum_static_bid_prices = [0]*n_products
    
    for j in range(n_products):
        for i in range(n_resources):
            if A[i][j] == 1:
                sum_static_bid_prices[j] += static_bid_prices[i]

    ## Calculates the displacement-adjusted revenues, in sorted order
    disp_adjusted_revs = [[(0, '')]*n_products for _ in range(n_resources)] 
    
    for i in range(n_resources):
        for j in range(n_products):
            product_name = products[j][0]
            if A[i][j] == 1: # only calculates for products that uses resource i
                disp_adjusted_revs[i][j] = (int(products[j][2]) - sum_static_bid_prices[j] + static_bid_prices[i],                                             product_name)
            else:
                disp_adjusted_revs[i][j] = (0, product_name)
        disp_adjusted_revs[i].sort(key = lambda tup: tup[0], reverse=True)
    

    return disp_adjusted_revs


# Calculates the squared deviation of revenue within partition (l, k), for resource i
# ref: example 3.5
def calc_squared_deviation_of_revenue(i, l, k, products, disp_adjusted_revs):
    """
    Parameter
    ----------
    i: integer
        the index of the resource
    l: integer
        the starting index of the partition, i.e. the index of the first product in this partition
        product index starts from 0
    k: integer
        the ending index of the partition, i.e. the index of the last product in this partition
    products: np array
        contains products, each in the form of [name, probabilities, revenue], size n_products
    disp_adjusted_revs: 2D np array
        contains tuples for displacement-adjusted revenues, in the form of (value, name of product),
        size n_resources * n_products
   
    Returns
    -------
    sqrd_deriv_revenue: number
        the squared deviation of revenue within the given partition
    """
    
    if k < l:
        warnings.warn("Wrong indices for the partition")
    
    if i >= len(disp_adjusted_revs):
        warnings.warn("Resource index out of boundary")
        
    # calculated the weighted-average displacement adjusted revenue for the given partition
    sum_demands = 0
    demands_times_disp_adjusted_rev = 0
    for j in range(l, k + 1):
        product_name = disp_adjusted_revs[i][j][1]
        product_mean_demand = float(next((v[1] for v, v in enumerate(products) if v[0] == product_name), 0))
        sum_demands += product_mean_demand
        demands_times_disp_adjusted_rev += product_mean_demand * disp_adjusted_revs[i][j][0]
    if sum_demands == 0:
        m = 0
    else:
        m = demands_times_disp_adjusted_rev / sum_demands  
    
    sqrd_deriv_revenue = 0
    for j in range(l, k + 1):
        product_name = disp_adjusted_revs[i][j][1]
        product_mean_demand = float(next((v[1] for v, v in enumerate(products) if v[0] == product_name), 0))
        sqrd_deriv_revenue += product_mean_demand * (disp_adjusted_revs[i][j][0] - m)**2
    return sqrd_deriv_revenue


# Implement the clustering process, partition products using each resource into a group of virtual classes.
# This is done by dynamic programming, looking for the partitions that can give the minimum squared deriviation
# of revenue (i.e. total within-group variation)
# ref: section 3.4.3, example 3.5
def clustering(products, resources, disp_adjusted_revs, n_virtual_class):
    """
    Parameter
    ----------
    products: np array
        contains products, each in the form of [name, probabilities, revenue], size n_products
    resources: np array
        contains names of resources, size n_resources
    disp_adjusted_revs: 2D np array
        contains tuples for displacement-adjusted revenues, in the form of (value, name of product),
        size n_resources * n_products
    n_virtual_class: integer
        the number of virtual classes to partition the products into
   
    Returns
    -------
    paritions: np array
        consists virtual classes for every resource, each contains the names of products in that class
        size n_partition
    """
    
    if n_virtual_class > len(disp_adjusted_revs[0]):
        warnings.warn("More virtual classes than number of products")
        
    n_resources = len(resources) # number of resources
    n_products = len(products) # number of products
    
    # calculate the minimum total squared deviation using dynamic programming, with the formula
    # V_c(k) = min(over 1<= l <= k) {c_{lk} + V_{c-1}(l-1)}, note that k, l indexed from 1 onwards,
    # c indexed from 1 (as V_0(k) is not used).
    # indexes l, k used in calc_squared_deviation_of_revenue should start from 0

    partitions_for_resources = []
    for i in range(n_resources):
        # only partition products that uses this resource
        available_products = [j for j, k in enumerate(disp_adjusted_revs[i]) if k[0] == 0]
        if available_products:
            n_available_products = available_products[0]
        else:
            n_available_products = n_products
            
        if n_available_products > 0:
            # holds the minimum total squared deviation
            V = [[()]*(n_available_products +1) for _ in range(n_virtual_class)] 

            # initialize V_1(k) = c_1k, for k = 1..n_virtual_class
            V[0][0] = (0, 0)
            for k in range(2, n_available_products+ 1):
                V[0][k] = (calc_squared_deviation_of_revenue(i, 0, k-1, products, disp_adjusted_revs), 0)

            # calculate V_2(k) onwards
            for c in range(1, n_virtual_class):
                for k in range(min(c + 2, n_available_products+1)):
                    V[c][k] = (0, 0)
                for k in range(c + 2, n_available_products + 1):
                    v = np.nan # record the minimum squared deviation
                    opt_l = -1 # record the starting index of the partition which gives the minimum squard deviation
                    for l in range(1, k + 1):
                        v_new = calc_squared_deviation_of_revenue(i, l-1, k-1, products, disp_adjusted_revs)                                + V[c-1][l-1][0]
                        if np.isnan(v) or v_new < v:
                            v = v_new
                            opt_l = l
                    V[c][k] = (v, opt_l - 1)

#             print(V)
            partition_indicies = []
            c = n_virtual_class - 1
            l = n_available_products
            while True:
                start_index = V[c][l][1]
                if start_index == 0 or c == 0:
                    break
                if not partition_indicies or start_index != partition_indicies[0]:
                    partition_indicies.insert(0, start_index)
                c -= 1
                l -= 1
    #         print("indicies for partition of source ", resources[i], " is: ", partition_indicies)

            partition_indicies.append(n_available_products)
            partitions = []
            start_index = 0
            for p in range(len(partition_indicies)):
                partition = []
                names = ''
                for j in range(start_index, partition_indicies[p]):
                    if names:
                        names+=','
                    names+= disp_adjusted_revs[i][j][1]
                partition = [names]
                start_index = partition_indicies[p]
                partitions.append(partition)
        else:
            partitions = []
        print("virtual classes of products for resource ", resources[i], " is: ", partitions)
        
        partitions_for_resources.append(partitions)
    return partitions_for_resources

# Computes and append a probability of demand for each virtual class of products, 
# which is the mean demand weighted average displacement-adjusted revenue.
# ref: section 3.4.3
def probability_of_demands(virtual_classes, products):
    """
    Parameter
    ----------
    virtual_classes: 2D np array
        contains virtual classes of products for each resource, size n_resources * n_products
    products: np array
        contains products, each in the form of [name, probabilities, revenue], size n_products
   
    Returns
    -------
    virtual_classes: 2D np array
        consists virtual classes of products for each resource, with probability of a demand added
        size n_resources * n_products
    """
    for i in range(len(virtual_classes)):
        for j in range(len(virtual_classes[i])):
            products_in_class = [x.strip() for x in virtual_classes[i][j][0].split(',')]
            total_demand = 0
            for product in products_in_class:
                demand = float(next((v[1] for v, v in enumerate(products) if v[0] == product), 0))
                total_demand += demand
            virtual_classes[i][j].append(total_demand)
    return virtual_classes


# Computes and append a representative revenue value for each virtual class of products, 
# which is the mean demand weighted average displacement-adjusted revenue.
# ref: section 3.4.3
def representative_revenue(virtual_classes, products, disp_adjusted_revs):
    """
    Parameter
    ----------
    virtual_classes: 2D np array
        contains virtual classes of products for each resource, size n_resources * n_products
    products: np array
        contains products, each in the form of [name, probabilities, revenue], size n_products
    disp_adjusted_revs: 2D np array
        contains tuples for displacement-adjusted revenues, in the form of (value, name of product),
        size n_resources * n_products
   
    Returns
    -------
    virtual_classes: 2D np array
        consists virtual classes of products for each resource, with representative revenue added
        size n_resources * n_products
    """
    for i in range(len(virtual_classes)):
        for j in range(len(virtual_classes[i])):
            if not virtual_classes[i][j]:
                continue
            products_in_class = [x.strip() for x in virtual_classes[i][j][0].split(',')]
            representative_rev = 0
            weighted_disp_adjusted_rev = 0
            for product in products_in_class:
                demand = float(next((v[1] for v, v in enumerate(products) if v[0]==product), 0))
                disp_adjusted_rev = float(next((v[0] for v,v in enumerate(disp_adjusted_revs[i]) if v[1]==product),0))
                weighted_disp_adjusted_rev += disp_adjusted_rev * demand
            virtual_classes[i][j].append(weighted_disp_adjusted_rev)

    return virtual_classes

# Main function, calculates the value-function estimate for DAVN problem,
# by clustering products into virtual classes and then solving a single-resource problem
def network_DAVN_value_function(products, resources, static_bid_prices, n_virtual_class, total_capacity,                              max_time, arrival_rate):
    """
    Parameter
    ----------
    products: np array
        contains products, each in the form of [name, probabilities, revenue], size n_products
    resources: np array
        contains names of resources, size n_resources
    static_bid_prices: np array
        contains static bid prices or marginal value estimates, size n_resources
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
    
    Returns
    -------
    value: 3D np array
        contains the value functions, size n_resources * (max_time + 1) * (total_capacity + 1)
    """
        
    # calculates the displacement-adjusted revenues
    disp_adjusted_revenue = calc_displacement_adjusted_revenue(products, resources, static_bid_prices)
    # clusters products into virtual classes
    virtual_classes = clustering(products, resources, disp_adjusted_revenue, n_virtual_class)
    print(virtual_classes)
    
    # appends the probability of a demand and a representative revenue onto each virtual class
    probab_appended = probability_of_demands(virtual_classes, products)
    complete_classes = representative_revenue(virtual_classes, products, disp_adjusted_revenue)
    
    print(complete_classes)
    value_functions = []
    for i in range(len(resources)):
        # for each resource, solve a single-resource problem
        value_func = SINGLE_value_function(complete_classes[i], total_capacity, max_time, arrival_rate)
        value_functions.append(value_func)
    return value_functions



# products = [['AB', 0.5, 100], ['CD', 0.5, 100], ['ABC', 0.5, 1000], ['BCD',0.5, 1000]]
# resources =  ['AB', 'BC', 'CD']
# mean_demands = [['AB', 10.1], ['CD', 5.3], ['ABC',8], ['BCD', 9.2]]
# n_virtual_class = 3
# # static_price = [0.5, 0.5, 0.5]

# static_price = [0, 0, 0]
# # disp_adjusted_revenue = calc_displacement_adjusted_revenue(products, resources, static_price)
# calculate_value_function(products, resources, static_price, n_virtual_class, mean_demands, 4, 2, 0.5)

# products = [['AE1', 0, 220], ['AE2', 0, 150], ['AE3', 0.5, 80], ['EB1',0.2, 320],['EB2', 0.3, 220],['EB3', 0.4, 140], \
#             ['CA1', 0.14, 280],['CA2', 0.29,190],['CA3',0.43, 110], ['CAE1',0, 330],['CAE2',0.33, 260],\
#             ['CAE3',0.33, 150],['AEB1', 0.2, 420],['AEB2', 0.3, 290],['AEB3', 0.4, 190]]
# resources = ['AE', 'EB', 'CA']
# mean_demands = [['AE1', 1], ['AE2', 1], ['AE3', 1], ['EB1', 1],['EB2', 1],['EB3', 1], ['CA1', 1],['CA2', 1],\
#                 ['CA3', 1], ['CAE1', 1],['CAE2', 1],['CAE3', 1],['AEB1', 1],['AEB2', 1],['AEB3', 1]] # not given
# n_virtual_class = 3
# static_price = [0, 0, 0]# not given
# disp_adjusted = calc_displacement_adjusted_revenue(products, resources, static_price)
# print(disp_adjusted, '\n')
# print(clustering(products, resources, disp_adjusted, n_virtual_class, mean_demands))


# products = [['AB', 0.5, 100], ['CD', 0.5,100], ['ABC', 0.5, 1000], ['BCD',0.5, 1000]]

# products = [['AB', 0.5, 100], ['CD', 0.5, 100]]
# resources =  ['AB', 'BC', 'CD']
# demand = 500
# mean_demands = [['AB', demand], ['CD', demand], ['ABC',demand], ['BCD', demand]]
# n_virtual_class = 2
# static_price = [0, 0, 0]
# # disp_adjusted = calc_displacement_adjusted_revenue(products, resources, static_price)
# # print(disp_adjusted, '\n')
# # print(clustering(products, resources, disp_adjusted, n_virtual_class, mean_demands))
# value = calculate_value_function(products, resources, static_price, n_virtual_class, mean_demands, 1,1, 0.9)
# print(value)

    
    
# products = [['AB', 0.5, 100], ['CD', 0.5, 100]]
# resources =  ['AB', 'BC', 'CD']
# demand = 1
# # mean_demands = [['AB', 10.1], ['CD', 5.3], ['ABC',8], ['BCD', 9.2], ['CDA', 3]]
# # mean_demands = [['AB', demand], ['CD',demand], ['ABC',demand], ['BCD', demand]]
# n_virtual_class = 2
# static_price = [0, 0, 0]
# disp_adjusted = calc_displacement_adjusted_revenue(products, resources, static_price)
# print(disp_adjusted, '\n')
# network_DAVN_value_function(products, resources, static_price, n_virtual_class, 1, 2, 2.5)


# products = [['AB', 0.05, 2000], ['CD', 0.25, 500], ['ABC', 0.5, 700], ['BCD', 0.2, 200]]
# # virtual_classes = [[['AB,ABC']], [['ABC,BCD']], [['CD,BCD']]]
# resources = ['AB', 'BC', 'CD']
# disp_adjusted = calc_displacement_adjusted_revenue(products, resources, [0, 0, 0])
# # representative_rev = representative_revenue(virtual_classes, products, disp_adjusted)
# # print(disp_adjusted, "\n")
# # print(representative_rev)


# products = [['AB', 0.05, 2000], ['ABC', 0.2, 4],  ['BC', 0.5, 30], ['CD', 0.2, 4555]]
# resources= ['AB', 'CD', 'BC']
# disp_adjusted = calc_displacement_adjusted_revenue(products, resources, [0, 0, 0])
# print(disp_adjusted)

# clustering(products, resources, disp_adjusted, 2)


# In[142]:

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
            delta = value_func_i[total_capacity] - value_func_i[total_capacity - 1]
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

# products = [['AB', 0.5, 100], ['CD', 0.5, 156], ['ABC', 0.5, 800], ['BCD',0.5, 1000],['CDA', 0.5, 401]]
# products = [['AB', 0.5, 100], ['CD', 0.5, 156], ['ABC', 0.5, 800], ['BCD',0.5, 1000]]

# products = [['AB', 0.5, 100], ['CD', 0.5, 100]]
products = [['AB', 0.05, 2000], ['CD', 0.25, 500], ['ABC', 0.5, 700], ['BCD', 0.2, 200]]
resources =  ['AB', 'BC', 'CD']
demand = 1
# mean_demands = [['AB', 10.1], ['CD', 5.3], ['ABC',8], ['BCD', 9.2], ['CDA', 3]]
# mean_demands = [['AB', demand], ['CD',demand], ['ABC',demand], ['BCD', demand]]
n_virtual_class = 2
static_price = [0, 0, 0]
##### iterate between [0, 0, 0] and [450, 450, 450] (pi)
# static_price = [0.5, 0.5, 0.5]
# disp_adjusted_revenue = calc_displacement_adjusted_revenue(products, resources, static_price)
# calculate_value_function(products, resources, static_price, n_virtual_class, mean_demands, 10, 2, 0.5)

iterative_DAVN(products, resources, n_virtual_class, 3, 10, 0.3, 8)

# network_DAVN_value_function(products, resources, static_price, n_virtual_class, mean_demands, 1, 2, 5)


# 

# In[ ]:



