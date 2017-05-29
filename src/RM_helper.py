
# coding: utf-8

# In[104]:

import numpy as np
import time
import random
import bisect
import networkx as nx
import matplotlib.pyplot as plt


# In[106]:

def sort_product_demands(products):
    """
    sorts the given products of form:[product_name, demand, revenue], into a list of [product_name, revenue]
    and a list of demands, according to the descending order of the revenue of each product
    """
    n_products = len(products)
    demands = []
    demands_with_name = []
    products.sort(key = lambda tup: tup[1], reverse=True)
    demands = [p[2] for p in products]
    demands_with_name = [[p[0], p[2]] for p in products]
    products = [[p[0], p[1]] for p in products]
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


# In[107]:

def state_index(n_states, capacities, remain_cap):
    """converts the given array of remaining capacities into the state number"""
    """e.g. given total capacities [1,2,1], and the remained capacities [0, 2, 1], should return 5"""

    if n_states == 0:
        n_states = 1
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


# In[108]:

def sample_network_demands(demands, total_time):
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

def sample_single_static_demands(demands):
    """given demands for products in descending order of their revenue, samples a list of demands for each product 
    in ascending order of their revenue."""
    sampled_demands = []
    for i in range(len(demands)):
        sample = np.random.normal(demands[i][0], demands[i][1])
        sampled_demands.append(int(sample))
        
    return sampled_demands


# In[109]:

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


# In[110]:

def extract_legs_info(products, resources):
    """plots a graph of flights, produces the incidence matrix, and returns a complete list of flight itineraries."""
    """input:
       products: list of itineraries, in the form of [name, [(revenue, arrival_rate) for fare classes]]."""
    graph = nx.DiGraph()
    
    # produces the full resources, by adding the opposite direction of each flight leg.
    full_resources = resources[:]
    for r in resources:
        oppo_r = r.split('-')
        full_resources.append(oppo_r[1] + '-' + oppo_r[0])
    
    n_products = len(products)
    itinerary_fares = []
    incidence_matrix = [[0] * n_products for _ in range(len(full_resources))] 
    
    for p in range(n_products):
        itinerary = products[p]
        nodes = itinerary[0].split('-')
        for n in range(len(nodes) - 1):
            leg_name = nodes[n] + '-' + nodes[n+1]
            leg_index = full_resources.index(leg_name)
            incidence_matrix[leg_index][p] = 1
        
        for f in range(len(itinerary[1])):
            fare = itinerary[1][f]
            fare_name = itinerary[0] + ',' + str(f + 1)
            itinerary_fares.append([fare_name, fare[0], fare[1]])
    
    for leg in resources:
        nodes = leg.split('-')
        start = nodes[0]
        end = nodes[1]
        graph.add_node(start)
        graph.add_node(end)
        graph.add_edge(start, end)
        graph.add_edge(end, start)
        
    plt.clf()
    nx.draw_networkx(graph)
    plt.savefig('flights-network.png')
    graph_info = "airline network: n_nodes=" + str(len(graph.nodes())) + ", n_edges=" + str(len(graph.edges()))
#     print(graph_info)
    return incidence_matrix, itinerary_fares

products = [['A-B', [(430, 0.21), (300, 0.29), (200, 0.43)]], ['A-E', [(220, 0), (150,0),(80,0.5)]],
            ['A-E-B',[(420, 0.2), (290,0.3), (190, 0.4)]], ['B-A', [(430, 0.21), (300, 0.29), (200, 0.43)]], 
           ['A-B-A', [(500, 0.2), (300, 0.19),(200, 0.5)]]]
resources = ['A-E', 'A-B', 'E-B']
extract_legs_info(products, resources)


# In[ ]:



