
# coding: utf-8

# In[4]:

import warnings
import numpy as np

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
                disp_adjusted_revs[i][j] = (int(products[j][2]) - sum_static_bid_prices[j] + static_bid_prices[i],                                             product_name)
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
    
    if k < l:
        warnings.warn("Wrong indices for the partition")
    
    if i >= len(disp_adjusted_revs):
        warnings.warn("Resource index out of boundary")
        
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


# Implement the clustering process, partition products using each resource into a group of virtual classes.
# This is done by dynamic programming, looking for the partitions that can give the minimum squared deriviation
# of revenue (i.e. total within-group variation)
# ref: section 3.4.3, example 3.5
def clustering(products, resources, disp_adjusted_revs, n_virtual_class, mean_demands):
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
    mean_demands: np array
        contains mean demands for each product, size n_products
   
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
        V = [[()]*(n_products +1) for _ in range(n_virtual_class)] # holds the minimum total squared deviation

        # initialize V_1(k) = c_1k, for k = 1..n_virtual_class
        V[0][0] = (0, 0)
        for k in range(1, n_virtual_class + 1):
            V[0][k] = (calc_squared_deviation_of_revenue(i, 0, k-1, mean_demands, disp_adjusted_revs), 0)
        for k in range(n_virtual_class + 1, n_products+1):
            V[0][k] = (0, 0)

        # calculate V_2(k) onwards
        for c in range(1, n_virtual_class):
            for k in range(c + 2):
                V[c][k] = (0, 0)
            for k in range(c + 2, n_products + 1):
                v = np.nan # record the minimum squared deviation
                opt_l = -1 # record the starting index of the partition which gives the minimum squard deviation
                for l in range(1, k + 1):
                    v_new = calc_squared_deviation_of_revenue(i, l-1, k-1, mean_demands, disp_adjusted_revs)                            + V[c-1][l-1][0]
                    if np.isnan(v) or v_new < v:
                        v = v_new
                        opt_l = l

                V[c][k] = (v, opt_l - 1)

        partition_indicies = []
        c = n_virtual_class - 1
        l = n_products
        while True:
            l = V[c][l][1]
            partition_indicies.insert(0, l)
            c -= 1
            if l == 0 or c == 0:
                break
        print("indicies for partition of source ", resources[i], " is: ", partition_indicies)

        partition_indicies.append(n_products)
        partitions = []
        start_index = 0
        for p in range(len(partition_indicies)):
            partition = []
            for j in range(start_index, partition_indicies[p]):
                partition.append(disp_adjusted_revs[i][j][1])
            start_index = partition_indicies[p]
            partitions.append(partition)

        print("virtual classes of products for resource ", resources[i], " is: ", partitions)
        
        partitions_for_resources.append(partitions)
    return partitions_for_resources


# In[ ]:




# In[ ]:



