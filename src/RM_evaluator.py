
# coding: utf-8

# In[1]:

import itertools
import random
import pandas
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import networkx as nx

import sys
sys.path.append('.')
import RM_helper
import RM_compare
import RM_exact
import RM_approx
import RM_ADP
import RM_demand_model


# In[2]:

PRICE_LIMITS = [150, 250] # maximum prices for flight legs in a 3-spoke or 4-spoke network
sum_arrival_rates = [0.3, 0.45, 0.9] # sum of arrival rates for low,med,hi demand levels
connect_symbol = '_'

def generate_network(n_spokes, demand_type, fare_class = 1):
    """Generates a network using the given number of spokes, and the demand type, with random prices, and arrival rates
    of itineraries. Currently only supports 1 fare class per itinerary. """
    resources = [] # records flight legs names
    itineraries = [] # records names and (revenue, arrival rate) pairs of fare classes of itineraries
    hub_name = 'hub'
    spoke_names = []
    
    # produce flight legs (single-direction)
    for i in range(n_spokes):
        spoke_name = chr(65 + i)
        spoke_names.append(spoke_name)
        resources.append(spoke_name + connect_symbol + hub_name)
    
    # produce single-leg itineraries
    single_legs = resources[:]
    single_legs += reverse_itinerary(resources)
    
    # produce double-leg itineraries
    double_legs = []
    two_spoke_pairs = list(itertools.combinations(''.join(spoke_names), 2))
    for pair in two_spoke_pairs:
        iti = connect_symbol.join([pair[0], hub_name, pair[1]])
        double_legs.append(iti)
    
    double_legs += reverse_itinerary(double_legs)
    
    # produce double-leg itineraries, between the hub and the same spoke, i.e. round-trips between spoke and hub
    round_legs = []
    for spoke in spoke_names:
        round_legs.append(connect_symbol.join([spoke, hub_name, spoke]))
    
    # aggregate all itineraries, and randomly generate the price and arrival rate
    itineraries += single_legs + double_legs + round_legs
    f = len(itineraries) * fare_class
    arrival_rates = generate_random_arrival_rate(f, demand_type)
    
    for i in range(f):
        full_iti = [itineraries[i]]
        price = generate_random_price(itineraries[i])
        full_iti.append([price])
        itineraries[i] = full_iti
    return resources, itineraries, arrival_rates
    
def reverse_itinerary(itinerary_names):
    """helper func: given a list of itinerary names, generate a list of reversed itineraries for them. """
    reversed_itineraries = []
    for itinerary in itinerary_names:
        nodes = itinerary.split(connect_symbol)
        nodes.reverse()
        reversed_name = connect_symbol.join(nodes)
        reversed_itineraries.append(reversed_name)
    return reversed_itineraries

def generate_random_arrival_rate(n, demand_type):
    """helper func: depending on the demand type, returns a list of arrival rates for different demand levels. """
    """only low demand level is returned if the demand type is 1."""
    arrival_rates = [sample_random_probs(n, sum_arrival_rates[0])] # sampled arrival rates of low demand level
    
    if demand_type == 2:
        med_level = sample_random_probs(n, sum_arrival_rates[1])
        hi_level = sample_random_probs(n, sum_arrival_rates[2])
        arrival_rates += [med_level, hi_level]
    return arrival_rates
        
def sample_random_probs(n, total_sum):
    """helper func: generate n random values in [0,1] and normalize them so that their sum is equal to total_sum."""
    M = sys.maxsize
    x = random.sample(range(M), n - 1)
    x.insert(0, 0)
    x.append(M)
    x.sort()
    y = [x[i + 1] - x[i] for i in range(n)]
    unit_simplex = [y_i / (1/total_sum * M) for y_i in y]
    return unit_simplex

def generate_random_price(itinerary_name):
    """helper func: generate a random price for the given itinerary, limit depends on how many flight legs it uses."""
    leg_num = itinerary_name.count(connect_symbol)
    price = random.randint(50, PRICE_LIMITS[leg_num-1])
    return price

def extract_legs_info(products, resources):
    """plots a graph of flights, produces the incidence matrix, and returns a complete list of flight itineraries."""
    """input:
       products: list of itineraries, in the form of [name, [(revenue, arrival_rate) for fare classes]]."""
    graph = nx.DiGraph()
    
    # produces the full resources, by adding the opposite direction of each flight leg.
    full_resources = resources[:]
    for r in resources:
        oppo_r = r.split(connect_symbol)
        full_resources.append(oppo_r[1] + connect_symbol + oppo_r[0])
    
    n_products = len(products)
    itinerary_fares = []
    
    for p in range(n_products):
        itinerary = products[p]
        nodes = itinerary[0].split(connect_symbol)
        for n in range(len(nodes) - 1):
            leg_name = nodes[n] + connect_symbol + nodes[n+1]
            leg_index = full_resources.index(leg_name)
        
        for f in range(len(itinerary[1])):
            fare = itinerary[1][f]
            fare_name = itinerary[0] + ',' + str(f + 1)
            itinerary_fares.append([fare_name, fare])
    
    for leg in resources:
        nodes = leg.split(connect_symbol)
        start = nodes[0]
        end = nodes[1]
        graph.add_node(start)
        graph.add_node(end)
        graph.add_edge(start, end)
        graph.add_edge(end, start)
        
#     plt.clf()
#     nx.draw_networkx(graph)
#     plt.savefig('flights-network.png')
    products = RM_helper.sort_product_revenues(itinerary_fares)
    return products

# resources, itineraries, arrival_rates = generate_network(3, 1)
# extract_legs_info(itineraries, resources)


# In[3]:

def compare_EMSR_b_with_exact_single_static(pros, cap, iterations):
    """Compare the EMSR-b method, with single-static DP model."""
    products, demands,_ = RM_helper.sort_product_demands(pros)
    
    diff_percents = []
    
    exact = RM_exact.Single_RM_static(products, demands, cap)
    exact_bid_prices = exact.get_bid_prices()
    exact_protection_levels = exact.get_protection_levels()
    
    heuri = RM_approx.Single_EMSR(products, demands, cap)
    heuri_protection_levels = heuri.get_protection_levels()

    bid_prices = [exact_bid_prices]
    protection_levels = [exact_protection_levels, heuri_protection_levels]
    
    exact_revs_diff = []
    exact_LF_diff = []
    exact_heuri_revs_diff = []
    exact_heuri_LF_diff = []

    results = [exact_protection_levels]
    for i in range(iterations):
        requests = RM_helper.sample_single_static_demands(demands)
        bp_result = RM_compare.simulate_single_static_bidprices_control(bid_prices, products, demands, cap, requests)
        pl_result = RM_compare.simulate_single_static_protectionlevel_control(protection_levels, products, demands,                                                                                cap, requests)
        
        exact_pl_rev = pl_result[0][0]
        exact_pl_LF = pl_result[0][1]
    
        exact_revs_diff.append(round((exact_pl_rev - bp_result[0][0])/ exact_pl_rev, 5))
        exact_LF_diff.append(round((exact_pl_LF - bp_result[0][1])/ exact_pl_LF, 5))
        exact_heuri_revs_diff.append(round((exact_pl_rev - pl_result[1][0])/exact_pl_rev, 5))
        exact_heuri_LF_diff.append(round((exact_pl_LF - pl_result[1][1]) / exact_pl_LF, 5))

    results+= [np.mean(exact_revs_diff) * 100, np.mean(exact_LF_diff) * 100, heuri_protection_levels, 
               np.mean(exact_heuri_revs_diff) * 100, np.mean(exact_heuri_LF_diff) * 100]
    return results

def visualize_perf_EMSR_b(products, cap_lb, cap_ub, cap_interval, iterations):
    """Visualize the performance of EMSR-b method, against single-static DP model."""
    capacities = [c for c in range(cap_lb, cap_ub + 1, cap_interval)]
    col_titles = ["exact-protection_levels", "mean-diff_exact %", "mean_diff_exact_LF %", "EMSR-b-protection_levels",                   "mean-diff_pl %", "mean-diff_pl_LF %"]

    table_data = []
    
    for cap in capacities:
        result= compare_EMSR_b_with_exact_single_static(products, cap, iterations)
        
        table_data.append(result)
    
    print(pandas.DataFrame(table_data, capacities, col_titles))
    return table_data

pros = [[1, 1050,(17.3, 5.8)], [2, 567, (45.1, 15.0)], [3, 534, (39.6, 13.2)], [4,520,(34.0, 11.3)]]
# pros = [[1, 1050,(17.3, 5.8)], [2, 950, (45.1, 15.0)], [3, 699, (39.6, 13.2)], [4,520,(34.0, 11.3)]]
cap_lb = 80
cap_ub = 160
cap_interval = 10
iteration = 100

# data = visualize_perf_EMSR_b(pros, cap_lb, cap_ub,cap_interval,iteration)
# exact_revs = [d[1] for d in data]
# exact_LF = [d[2] for d in data]

# plt.clf()
# x= np.linspace(cap_lb, cap_ub, (cap_ub - cap_lb) / cap_interval + 1)
# plt.plot(x, exact_revs, linestyle='dashed', marker='s', label='Revenue Difference')
# plt.plot(x, exact_LF, linestyle='dashed', marker = 'o', label='Load Factor Difference')
    
# plt.legend()
# plt.ylabel('Bid-price vs Protection-level Control')
# plt.xlabel('Resource Capacity')
# # plt.show()
# plt.savefig('single_static_exact_diff')


# exact_heuri_revs = [d[4] for d in data]
# exact_heuri_LF = [d[5] for d in data]
# plt.clf()
# x= np.linspace(cap_lb, cap_ub, (cap_ub - cap_lb) / cap_interval + 1)
# plt.plot(x, exact_heuri_revs, linestyle='dashed', marker='s', label='Revenue Difference')
# plt.plot(x, exact_heuri_LF, linestyle='dashed', marker = 'o', label='Load Factor Difference')
    
# plt.legend()
# plt.ylabel('Exact vs EMSR-b Protection-levels Control')
# plt.xlabel('Resource Capacity')
# # plt.show()
# plt.savefig('single_static_diff')



# In[6]:

p = 0.5
def generate_samples(total_num, n_spoke, cap, demand_type, n_fare_class):
    """ generate a collection of random problems to be used in evaluation,
    each specifying products, resources, capacities of resources, total time, demand model"""
    problem_sets = []
    for i in range(total_num):
        resources, itineraries, arrival_rates = generate_network(n_spoke, demand_type, n_fare_class)
        products = extract_legs_info(itineraries, resources)
        capacities = [cap] * len(resources)
        total_time = cap * 3
        dm = None
        dm = RM_demand_model.model(arrival_rates, total_time, demand_type, p)
        
        problem = [products, resources, capacities, total_time, dm]
        problem_sets.append(problem)
        
    return problem_sets
    
def compare_with_DP(n_spoke, cap, iterations, n_virtual_class, K):
    """ small network problems, solved by DP, DLPDAVN, and ADP respectively """
    col_titles = ["rev_DLPDAVN_mean %",  "rev_LPADP_mean %", ]
    table_data = []
    problems = generate_samples(10, n_spoke, cap, 1, 1)
    for prob in problems:
        diff_DLPDAVN = []
        diff_LPADP = []
        
        products = prob[0]
        resources = prob[1]
        capacities = prob[2]
        total_time = prob[3]
        demand_model = prob[4]
        a = demand_model.arrival_rates['low']
        
        exactDP_model = RM_exact.Network_RM(products, resources, capacities, total_time, demand_model)
        DLPDAVN_model = RM_approx.DLP_DAVN(products, resources, capacities, total_time, n_virtual_class, demand_model)
        LPADP_model = RM_ADP.ALP(products, resources, capacities, total_time, demand_model)
        
        exactDP_bid_prices = exactDP_model.get_bid_prices()
        LPADP_bid_prices = LPADP_model.get_bid_prices(K)
        
        bid_prices = [exactDP_bid_prices, LPADP_bid_prices]
        
        for i in range(iterations):
            requests = demand_model.sample_network_arrival_rates()
            
            eval_results = RM_compare.simulate_network_bidprices_control(bid_prices, products, resources, capacities,                                                                         total_time, requests)
#             print("eval_results: ", eval_results)
            exactDP_rev = eval_results[0][0]
            LPADP_rev = eval_results[0][0]
            DLPDAVN_rev = DLPDAVN_model.performance(requests)[0]
            
            diff_LPADP.append((exactDP_rev - LPADP_rev)/exactDP_rev)
            diff_DLPDAVN.append((exactDP_rev - DLPDAVN_rev)/exactDP_rev)
            
        table_data.append([np.mean(diff_LPADP) * 100, np.mean(diff_DLPDAVN) * 100])
            
    print(pandas.DataFrame(table_data, range(len(problems)), col_titles))
    return table_data
    
# compare_with_DP(3, 5, 3, 3, 10)


# In[ ]:



