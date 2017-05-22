
# coding: utf-8

# In[3]:

import numpy as np
import time
import random
import bisect


# In[27]:

def sort_product_demands(products):
    """
    sorts the given products of form:[product_name, demand, revenue], into a list of [product_name, revenue]
    and a list of demands, according to the descending order of the revenue of each product
    """
    n_products = len(products)
    demands = []
    demands_with_name = []
    products.sort(key = lambda tup: tup[2], reverse=True)
    demands = [p[1] for p in products]
    demands_with_name = [[p[0], p[1]] for p in products]
    products = [[p[0], p[2]] for p in products]
    return (products, demands, demands_with_name)

def marginal_value_check(value_func):
    """checks whether the marginal values in computed value functions satisfy the proposition 2.21"""
    dim = len(value_func)
    for j in range(dim):
        delta_V= [x-y for x, y in zip(value_func[j][1:], value_func[j])]
        print("delta = ", delta_V)
        if any(delta_V[i] < delta_V[i+1] for i in range(len(delta_V) - 1)):
            print("error type 1")
        if j < (dim -1):
            delta_V_next = [x-y for x, y in zip(value_func[j+1][1:], value_func[j+1])]
            print("delta_next = ", delta_V_next)
            if any(delta_V[i] > delta_V_next[i] for i in range(len(delta_V))):
                print("error type 2")

def calc_incidence_matrix(products, resources):
    """constructs the incidence matrix, indicating which product uses which resources, 
        e.g. incidence_matrix[i][j] = 1 if product j uses resource i
        size n_resources * n_products"""
    
    n_products = len(products)
    n_resources = len(resources)

    incidence_matrix = [[0] * n_products for _ in range(n_resources)] 

    for i in range(n_resources):
        for j in range(n_products):
            if resources[i] in products[j][0]: # test if product j uses resource i
                incidence_matrix[i][j] = 1
    return incidence_matrix


# In[62]:

def state_index(n_states, capacities, remain_cap):
    """converts the given array of remaining capacities into the state number"""
    """e.g. given total capacities [1,2,1], and the remained capacities [0, 2, 1], should return 5"""

    if n_states == 0:
        n_states = 
        for c in capacities:
            n_states *= (c + 1)
    
    state_num = 0
    capacity_for_others = n_states

    for i in range(len(capacities)):
        capacity_for_others /= capacities[i] + 1
        state_num += remain_cap[i] * capacity_for_others
    return int(state_num)
        
def remain_cap(n_states, capacities, state_number):
    """reverse of function state_number(), to convert the given state number into remained capacities"""
    """e.g. given total capacities [1,2,3] and state_number 5, should return [0, 2, 1]"""

    if state_number >= n_states:
        raise RuntimeError(
            'Error when converting state number to remained capacities; given state number is too large.')

    remain_cap = []
    capacity_for_others = n_states

    for i in range(len(capacities)):
        capacity_for_others /= capacities[i] + 1
        remain_cap.append(int(state_number // capacity_for_others))
        state_number %= capacity_for_others
    return remain_cap


# In[66]:

def sample_demands(demands, total_time):
    """samples a series of index of products, whose request arrives at each period in the given total time """
    cumu_prob = [0] * len(demands)
    up_to = 0
    for i in range(len(demands)):
        up_to += demands[i]
        cumu_prob[i] = up_to

    cumu_prob.append(1.0)
    
    sample_index = [0] * total_time
    for t in range(total_time):
        rand = random.random()
        fall_into = bisect.bisect(cumu_prob, rand)
        sample_index[t] = fall_into
    return sample_index


# In[65]:

def network_bid_prices(value_func, products, resources, capacities, incidence_matrix, n_states):
    """Calculate the bid prices for resources at every state in every time period."""
    """Time index convention: starts from t=1, terminates at t=T, where len(value_func) = T """
    bid_prices = []

    n_resources = len(resources)
    if not incidence_matrix:
        incidence_matrix = calc_incidence_matrix(products, resources)
        
    for t in range(len(value_func)):
        bid_price_t = []
        for s in range(n_states):
            A = []
            b = []
            bp_t_s = [None] * n_resources
            for j in range(len(products)):
                incidence_vector = [row[j] for row in incidence_matrix]
                if incidence_vector not in A:
                    V_diff = value_func[t][s]
                    remained_cap = remain_cap(n_states, capacities, s)
                    reduced_cap = [a_i - b_i for a_i, b_i in zip(remained_cap, incidence_vector)]
                    if all(c >= 0 for c in reduced_cap):
                        V_diff -= value_func[t][state_index(n_states, capacities, reduced_cap)]
                        if sum(incidence_vector) == 1:
                            bp_t_s[incidence_vector.index(1)] = V_diff
                    A.append(incidence_vector)
                    b.append(V_diff)
#             print("bp_t_s:", bp_t_s)
            if any(bp is None for bp in bp_t_s):
                if not A:
                    bp_t_s = [0] * n_resources
                elif len(A) < n_resources:
                    bp_t_s = [0 if x is None else x for x in bp_t_s]
                else:
                    A_solve = A[:n_resources]
                    b_solve = b[:n_resources]
#                     print("A,b=", A_solve, b_solve)
                    bp = np.linalg.solve(A_solve, b_solve)
#                     print("result=", bp)
                    bp_t_s = [0 if x < 0 else x for x in bp]

            bid_price_t.append([round(bp_r, 3) for bp_r in bp_t_s])
        bid_prices.append(bid_price_t)
    return bid_prices

