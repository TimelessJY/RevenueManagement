
# coding: utf-8

# In[48]:

import numpy as np

# Calculate the displacement-adjusted revenues, 
# which is to approximate the net benefit of accepting product j on resource i
# ref: function 3.15
def calc_displacement_adjusted_revenue(products, resources, static_bid_prices):
    """
    Parameter
    ----------
    products: np array
        contains tuples for products, in the form of (name, revenue), size n_products
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
            if not set(products[j][0]).isdisjoint(resources[i]):
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
                disp_adjusted_revs[i][j] = (int(products[j][1]) - sum_static_bid_prices[j] + static_bid_prices[i],                                             product_name)
            else:
                disp_adjusted_revs[i][j] = (0, product_name)
        disp_adjusted_revs[i].sort(key = lambda tup: tup[0], reverse=True)
    

    return disp_adjusted_revs

# Calculates the squared deviation of revenue within partition (l, k), for resource i
# ref: example 3.5
def calc_squared_deviation_of_revenue(i, l, k, mean_demands, disp_adjusted_revs):
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
    mean_demands: np array
        contains tuples for mean demands of products, in the form of (product_name, mean_demand), size n_products
    disp_adjusted_revs: 2D np array
        contains tuples for displacement-adjusted revenues, in the form of (value, name of product),
        size n_resources * n_products
   
    Returns
    -------
    sqrd_deriv_revenue: number
        the squared deviation of revenue within the given partition
    """
    
    # calculated the weighted-average displacement adjusted revenue for the given partition
    sum_demands = 0
    demands_times_disp_adjusted_rev = 0
    for j in range(l, k + 1):
        product_name = disp_adjusted_revs[i][j][1]
        product_mean_demand = float(next((v[1] for v, v in enumerate(mean_demands) if v[0] == product_name), 0))
        sum_demands += product_mean_demand
        demands_times_disp_adjusted_rev += product_mean_demand * disp_adjusted_revs[i][j][0]
    if sum_demands == 0:
        m = 0
    else:
        m = demands_times_disp_adjusted_rev / sum_demands  
    
    sqrd_deriv_revenue = 0
    for j in range(l, k + 1):
        product_name = disp_adjusted_revs[i][j][1]
        product_mean_demand = float(next((v[1] for v, v in enumerate(mean_demands) if v[0] == product_name), 0))
        sqrd_deriv_revenue += product_mean_demand * (disp_adjusted_revs[i][j][0] - m)**2
    return sqrd_deriv_revenue

