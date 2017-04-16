
# coding: utf-8

# In[7]:

import warnings
import numpy as np

import sys
sys.path.append('/Users/jshan/Desktop/RevenueManagement')
from src import network_DAVN

# Implement the iterative displacement-adjusted virtual nesting(DAVN) method for network RM problem
# The result is static bid prices estimated, either converged, or after a large number of computation rounds.
# ref: section 3.4.5.1
def iterative_DAVN(products, resources, n_virtual_class, mean_demands, total_capacity, max_time, arrival_rate,                    current_time):
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
        value_funcs = network_DAVN.calculate_value_function(products, resources, static_bid_prices[k],                                                n_virtual_class, mean_demands, total_capacity, max_time, arrival_rate)
        print('value func: ', value_funcs)
        deltas = []
        for i in range(n_resources):
            value_func_i = value_funcs[i][max_time]
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
    


# In[ ]:




# In[ ]:



