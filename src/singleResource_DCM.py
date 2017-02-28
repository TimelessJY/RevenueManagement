
# coding: utf-8

# In[27]:

import warnings
import numpy as np

# Solves single-resource capacity control, using Discrete Choice Models.
# ref: The Theory and Practice of Revenue Management, section 2.6.2

# Identifies a list of efficient sets, given information about all possible sets of products(classes)
# ref: section 2.6.2.4
def efficient_sets(products):
    """
    Parameter
    ----------
    products: np array
        contains tuples for products, in the form of (product_name, probability of purchase, expected revenue), 
        size n_products
   
    Returns
    -------
    effi_sets: np array
        contains tuples for efficient sets, in the form of (name, prob, revenue), size n_products
    """
    
    n_products = len(products)  # number of all sets

    effi_sets = []   # stores output
    prev_effi_set = -1    # store the previous efficient set, start with empty set
    prev_prob = 0   # store the choice probability of the previous efficient set
    prev_revenue = 0   # store the revenue of the previous efficient set

    while True:
        next_effi_set = -1     # store the next efficient set
        max_marginal_revenue_ratio = 0
        has_potential_set = False
        for i in range(n_products):
            if i == prev_effi_set:
                continue
            prob = products[i][1]
            revenue = products[i][2]
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
            prev_prob = products[next_effi_set][1]
            prev_revenue = products[next_effi_set][2]
            effi_sets.append(products[next_effi_set])


    return effi_sets

# In nested policy, once identified the efficient sets
# can compute the objective function value to find out which efficient set is optimal for that capacity x.
# ref: section 2.6.2.5
def optimal_set_for_capacity(effi_sets, marginal_values):
    """
    Parameter
    ----------
    effi_sets: np array
        contains tuples for efficient sets, in the form of (product_name, prob, revenue), size n_effi_sets
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
        contains tuples for efficient sets, in the form of (product_name, prob, revenue), size n_effi_sets
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


# In[ ]:



