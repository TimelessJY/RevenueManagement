
# coding: utf-8

# In[15]:

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
def optimal_set_for_capacity(effi_sets, marginal_values):
    """
    Parameter
    ----------
    effi_sets: np array
        contains efficient sets, each in the form of [product_name, prob, revenue], size n_effi_sets
    marginal_values: np array
        contains expected marginal value of every capacity at time t+1, size n_capacity
   
    Returns
    -------
    optimal_set: np array
        contains the set_index of the optimal set for capacity x, size n_capacity
    """
    
    n_capacity = len(marginal_values)
    optimal_set = []
    n_effi_sets = len(effi_sets)
    for i in range(n_capacity):
        max_diff = 0
        curr_opt_set = -1
        for j in range(n_effi_sets):
            diff = effi_sets[j][2] - effi_sets[j][1] * marginal_values[i]
            if diff > max_diff:
                max_diff = diff
                curr_opt_set = effi_sets[j][0]
        optimal_set.append(curr_opt_set)
    return optimal_set

                                
# In nested policy, calculate the optimal protection levels for each (efficient) class
def optimal_protection_levels(effi_sets, marginal_values):
    """
    Parameter
    ----------
    effi_sets: np array
        contains efficient sets, each in the form of [product_name, prob, revenue], size n_effi_sets
    marginal_values: np array
        contains expected marginal value of every capacity at time t+1, size n_marginal_values
   
    Returns
    -------
    protection_levels: np array
        contains the optimal protection level for the given efficient sets, size n_efficient_sets
    """

    n_effi_sets = len(effi_sets)
    n_marginal_values = len(marginal_values)
    protection_levels = []
    for i in range(n_effi_sets - 1):
        for capacity in reversed(range(n_marginal_values)):
            diff = effi_sets[i][2] - effi_sets[i][1] * marginal_values[capacity]
            nextDiff = effi_sets[i+1][2] - effi_sets[i+1][1] * marginal_values[capacity]
            if diff > nextDiff:
                protection_levels.append(capacity + 1)
                break
    protection_levels.append(n_marginal_values)
    return protection_levels

# Calculates value functions V_t(x) for different remaining capacity, x = 0 ... C
# using backward computations, starting from V_T(x) back to V_0(x)
# ref: function 2.26
def calc_value_function(effi_sets, total_capacity, max_time, arrival_rate):
    """
    Parameter
    ----------
    effi_sets: np array
        contains efficient sets, each in the form of [product_name, prob, revenue], size n_effi_sets
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
            max_obj_val = 0 
            delta = prev_V[x] - prev_V[x-1] # the marginal cost of capacity in the next period
            for s in effi_sets:
                obj_val = s[2] - s[1] * delta # the difference between the expected revenue from offering set S, 
                # and the revenue if a request in set S is accepted
                if obj_val > max_obj_val:
                    max_obj_val = obj_val
            max_obj_val *= arrival_rate
            max_obj_val += prev_V[x]
            curr_V[x] = max_obj_val
        V.insert(0, curr_V)
        prev_V = curr_V
    return V


# In[ ]:



