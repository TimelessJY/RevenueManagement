
# coding: utf-8

# In[18]:

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


# In[ ]:



