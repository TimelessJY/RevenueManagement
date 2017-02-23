
# coding: utf-8

# In[14]:

import warnings
import numpy as np

# Solves single-resource capacity control, using Discrete Choice Models.
# ref: The Theory and Practice of Revenue Management, section 2.6.2


# Identifies a list of efficient sets, given information about all possible sets of products(classes)
# ref: section 2.6.2.4
def efficient_sets(probs, revenues):
    """
    Parameter
    ----------
    probs: np array
        contains probability of purchase for each set of products, size n_sets
    revenues: np array
        contains expected revenue for each set, size n_sets
   
    Returns
    -------
    effi_sets: np array
        contains tuples for efficient sets, in the form of (set_index, prob, revenue), size n_sets
    """
    
    if len(revenues) != len(probs):
        warnings.warn("Wrong size of input in efficientSets()")

    n_sets = min(len(probs), len(revenues))    # number of all sets

    effi_sets = list()   # stores output
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
            prob = float(probs[i])
            revenue = revenues[i]
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
            prev_prob = probs[next_effi_set]
            prev_revenue = revenues[next_effi_set]
            effi_sets.append((next_effi_set + 1, prev_prob, prev_revenue))


    return effi_sets

# In nested policy, once identified the efficient sets
# can compute the objective function value to find out which efficient set is optimal for that capacity x.
# ref: section 2.6.2.5
def optimal_set_for_capacity(effi_sets, marginal_values):
    """
    Parameter
    ----------
    effi_sets: np array
        contains tuples for efficient sets, in the form of (set_index, prob, revenue), size n_sets
    marginal_values: np array
        contains expected marginal value of every capacity at time t+1, size n_capacity
   
    Returns
    -------
    optimal_set: np array
        contains the set_index of the optimal set for capacity x, size n_capacity
    """
    n_capacity = len(marginal_values)
    optimal_set = list()
    for i in range(n_capacity):
        max_diff = 0
        curr_opt_set = -1
        for j in range(len(effi_sets)):
            diff = effi_sets[j][2] - effi_sets[j][1] * marginal_values[i]
            if diff > max_diff:
                max_diff = diff
                curr_opt_set = effi_sets[j][0]
        optimal_set.append(curr_opt_set)
    return optimal_set


# In[ ]:



