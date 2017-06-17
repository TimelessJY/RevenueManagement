
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
import math

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
    reversed_resource = reverse_itinerary(resources)
    single_legs += reversed_resource
    resources += reversed_resource
    
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
# print("resources = ", resources)
# extract_legs_info(itineraries, resources)


# In[3]:

def compare_EMSR_b_with_exact_single_static(pros, cap, iterations):
    """Compare the EMSR-b method, with single-static DP model."""
    products, demands,_ = RM_helper.sort_product_demands(pros)
    
    diff_percents = []
    
    exact_time = time.time()
    exact = RM_exact.Single_RM_static(products, demands, cap)
    exact_bid_prices = exact.get_bid_prices()
    exact_protection_levels = exact.get_protection_levels()
    exact_time = time.time() - exact_time
    
    heuri_time = time.time()
    heuri = RM_approx.Single_EMSR(products, demands, cap)
    heuri_protection_levels = heuri.get_protection_levels()
    heuri_time = time.time() - heuri_time

    bid_prices = [exact_bid_prices]
    protection_levels = [exact_protection_levels, heuri_protection_levels]
    
    # comparison result of exact method using bid-price control and protection-level control
    exact_revs_diff = []
    exact_LF_diff = []
    # comparison result of exact method vs EMSR-b method, both using protection-level control
    exact_heuri_revs_diff = []
    exact_heuri_LF_diff = []
    exact_revs = []

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
        exact_revs.append(exact_pl_rev)

    results+= [np.mean(exact_revs_diff) * 100, np.mean(exact_LF_diff) * 100, heuri_protection_levels, 
               np.mean(exact_heuri_revs_diff) * 100, np.std(exact_heuri_revs_diff),
               np.mean(exact_heuri_LF_diff) * 100, np.std(exact_heuri_revs_diff), exact_time, heuri_time,
               np.mean(exact_revs)]
    return results

def visualize_perf_EMSR_b(products, cap_lb, cap_ub, cap_interval, iterations):
    """Visualize the performance of EMSR-b method, against single-static DP model."""
    capacities = [c for c in range(cap_lb, cap_ub + 1, cap_interval)]
    col_titles = ["exact-protection_levels", "mean-diff_exact %", "mean_diff_exact_LF %", "EMSR-b-protection_levels",                   "mean-diff_pl %", "std-diff_pl", "mean-diff_pl_LF %", "std-diff_pl_LF", "time_dp", "time_emsrb",                   "total_rev_exact"]

    table_data = []
    
    for cap in capacities:
        result= compare_EMSR_b_with_exact_single_static(products, cap, iterations)
        
        table_data.append(result)
    
    print(pandas.DataFrame(table_data, capacities, col_titles))
    return table_data

# pros = [[1, 1050,(17.3, 5.8)], [2, 567, (45.1, 15.0)], [3, 534, (39.6, 13.2)], [4,520,(34.0, 11.3)]]
# # pros = [[1, 1050,(17.3, 5.8)], [2, 950, (45.1, 15.0)], [3, 699, (39.6, 13.2)], [4,520,(34.0, 11.3)]]
# cap_lb = 50
# cap_ub = 150
# cap_interval = 10
# iteration = 1000

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
# exact_heuri_revs_std = [d[5] for d in data]
# exact_heuri_LF = [d[6] for d in data]
# exact_heuri_LF_std = [d[7] for d in data]
# exact_time = [d[8] for d in data]
# heuri_time = [d[9] for d in data]

# plt.clf()
# x= np.linspace(cap_lb, cap_ub, (cap_ub - cap_lb) / cap_interval + 1)
# plt.plot(x, exact_heuri_revs, linestyle='dashed', marker='s', label='Revenue Difference %')
# plt.plot(x, exact_heuri_LF, linestyle='dashed', marker = 'o', label='Load Factor Difference %')
    
# plt.legend()
# plt.ylabel('Exact DP vs EMSR-b')
# plt.xlabel('Resource Capacity')
# # plt.show()
# plt.savefig('single_static_diff')

# plt.clf()
# plt.plot(x, exact_time, linestyle='dashed', marker='s', label='Exact DP')
# plt.plot(x, heuri_time, linestyle='dashed', marker = 'o', label='EMSR-b')
    
# plt.legend()
# plt.ylabel('Exact DP vs EMSR-b: Planning Time (s)')
# plt.xlabel('Resource Capacity')
# # plt.show()
# plt.savefig('single_static_time_diff')


# In[4]:

p = 0.5
def generate_samples(total_num, n_spoke, cap, demand_type, n_fare_class):
    """ generate a collection of random problems to be used in evaluation,
    each specifying products, resources, capacities of resources, total time, demand model"""
    problem_sets = []
    for i in range(total_num):
        resources, itineraries, arrival_rates = generate_network(n_spoke, demand_type, n_fare_class)
        products = extract_legs_info(itineraries, resources)
        capacities = [cap] * len(resources)
        total_time = cap * len(resources) * 5
        dm = None
        dm = RM_demand_model.model(arrival_rates, total_time, demand_type, p)
        
        problem = [products, resources, capacities, total_time, dm]
        problem_sets.append(problem)
        
    return problem_sets
    
def compare_with_DP(total_num, n_spoke, cap, iterations, demand_type, n_virtual_class, K):
    """ small network problems, solved by DP, DLPDAVN, and ADP respectively """
    col_titles = ["rev_DLPDAVN_mean %", "loadF_DLPDAVN_mean %", "rev_LPADP_mean %", "loadF_LPADP_mean %", 
                  "rev_DLPVD_mean %", "loadF_DLPVD_mean","exact_rev", "exact_LF"]
    table_data = []
    problems = generate_samples(total_num, n_spoke, cap, demand_type, 1)
    for prob in problems:
        compare_results = [[] for _ in range(len(col_titles))]
        
        products = prob[0]
        resources = prob[1]
        capacities = prob[2]
        total_time = prob[3]
        demand_model = prob[4]
        
        exactDP_model = RM_exact.Network_RM(products, resources, capacities, total_time, demand_model)
        DLPDAVN_model = RM_approx.DLP_DAVN(products, resources, capacities, total_time, n_virtual_class, demand_model)
        LPADP_model = RM_ADP.ALP(products, resources, capacities, total_time, demand_model)
        DLPVD_model = RM_approx.DLPVD(products, resources, capacities, total_time, demand_model)
        
        exactDP_bid_prices = exactDP_model.get_bid_prices()
        LPADP_bid_prices = LPADP_model.get_bid_prices(K)
        
        bid_prices = [exactDP_bid_prices, LPADP_bid_prices]
        
        for i in range(iterations):
            requests = demand_model.sample_network_arrival_rates()
            
            eval_results = RM_compare.simulate_network_bidprices_control(bid_prices, products, resources, capacities,                                                                         total_time, requests)
            exactDP_rev = eval_results[0][0]
            exactDP_LF = eval_results[0][1]
            
            DLPDAVN_result = DLPDAVN_model.performance(requests)
            compare_results[0].append((exactDP_rev - DLPDAVN_result[0])/exactDP_rev * 100)
            compare_results[1].append((exactDP_LF - DLPDAVN_result[1]) / exactDP_LF * 100)
            
            LPADP_results = eval_results[1]
            compare_results[2].append((exactDP_rev - LPADP_results[0])/exactDP_rev * 100)
            compare_results[3].append((exactDP_LF - LPADP_results[1]) / exactDP_LF * 100)
            
            DLPVD_result = DLPVD_model.performance(requests)
            compare_results[4].append((exactDP_rev - DLPVD_result[0])/exactDP_rev * 100)
            compare_results[5].append((exactDP_LF - DLPVD_result[1]) / exactDP_LF * 100)
            
            compare_results[6].append(exactDP_rev)
            compare_results[7].append(exactDP_LF)
            
        table_data.append([np.mean(result) for result in compare_results])
            
    print(pandas.DataFrame(table_data,  columns = col_titles))
    return table_data
    
# result = compare_with_DP(3, 3, 3, 5, 2, 6, 40)

# x = [data[-1] for data in result]
# DLPDAVN_perf = [data[0] for data in result]
# LPADP_perf = [data[2] for data in result]
# DLPVD_perf = [data[4] for data in result]

# plt.clf()
# plt.plot(x, DLPDAVN_perf, "o")
# plt.ylabel('Revenue Difference against exactDP %')
# plt.xlabel('Load Factor by exact DP % ')
# # plt.show()
# plt.savefig('rev_perf_vs_exactDP_DLPDAVN')

# plt.clf()
# plt.plot(x, LPADP_perf, "o")
# plt.ylabel('Revenue Difference against exactDP %')
# plt.xlabel('Load Factor by exact DP % ')
# # plt.show()
# plt.savefig('rev_perf_vs_exactDP_LPADP')

# plt.clf()
# plt.plot(x, DLPVD_perf, "o")
# plt.ylabel('Revenue Difference against exactDP %')
# plt.xlabel('Load Factor by exact DP % ')
# # plt.show()
# plt.savefig('rev_perf_vs_exactDP_DLPVD')


# In[5]:

# Draw the graph of running time of the network_DP model
def eval_networkDP_runningTime(products, resources, cap_lb, cap_ub, total_time):
    """Evaluate the network DP method, trying with different capacities of resource, and different total time."""
    n_resources = len(resources)
    col_titles = ['Revenue', 'Bid Prices', 'Time']
    capacities = [c for c in range(cap_lb, cap_ub + 1)]
    
    table = []
    pros, arrival_rates, _ = RM_helper.sort_product_demands(products)
    print("arrival_rates", arrival_rates)
    demand_model = RM_demand_model.model([arrival_rates], total_time, 1)
    
    for cap in capacities:
        caps = [cap] * n_resources
        
        result= []
        
        DP_time = time.time()
        problem = RM_exact.Network_RM(pros, resources, caps, total_time, demand_model)
        DP_vf = problem.calc_value_func()
        bid_prices = problem.get_bid_prices()
        DP_time = time.time() - DP_time

        result.append(DP_vf[0][-1])
        result.append(bid_prices)
        result.append(DP_time)
        
        table.append(result)
        
    print(pandas.DataFrame(table, capacities, col_titles))
    return table
        
# ps1 = [['a1', 200,0.02], ['a2', 503, 0.06], ['ab1', 400, 0.08],['ab2', 704, 0.01], ['b1', 601, 0.05], \
#       ['b2', 106, 0.12], ['bc', 920, 0.03],['c1', 832, 0.07], ['d1', 397, 0.14], ['d2', 533, 0.18], ['ad', 935, 0.09],\
#       ['ae', 205, 0.013],['f3', 589, 0.004], ['fb', 422, 0.009]]
# rs1 = ['a', 'b', 'c', 'd', 'e', 'f']

# ps2 = [['a1', 200,0.02], ['a2', 503, 0.06], ['ab1', 400, 0.08],['ab2', 704, 0.01], ['b1', 601, 0.05], \
#       ['b2', 106, 0.12], ['bc', 920, 0.03],['c1', 832, 0.07]]
# rs2 = ['a', 'b', 'c']

# cap_ub = 8
# T = 10
# ps = ps1
# rs = rs1
# tables = []
# final_result = []
# for i in range(3):
#     performance = eval_networkDP_runningTime(ps, rs, 1, cap_ub, T * (i + 1))
#     tables.append(performance)
#     final_result.append(([d[0] for d in performance], [d[2] for d in performance]))

# x= np.linspace(1, cap_ub, cap_ub)

# plt.clf()
# line1, = plt.plot(x,final_result[0][0], marker='^', label='max_time='+str(T))
# line2, = plt.plot(x,final_result[1][0], marker='o', label='max_time='+str(T * 2))
# line3, = plt.plot(x,final_result[2][0], marker='x', label='max_time='+str(T * 3))

# plt.legend(handler_map={line1: HandlerLine2D(numpoints=1),line2: HandlerLine2D(numpoints=1),
#                         line3: HandlerLine2D(numpoints=1)})
# plt.ylabel('Expected Revenue')
# plt.xlabel('Resource Capacity')
# # plt.show()
# plt.savefig('pictures/network-DP-revs-3resource')

# plt.clf()
# line1, = plt.plot(x,final_result[0][1], marker='^', label='max_time='+str(T))
# line2, = plt.plot(x,final_result[1][1], marker='o', label='max_time='+str(T * 2))
# line3, = plt.plot(x,final_result[2][1], marker='x', label='max_time='+str(T * 3))

# plt.legend(handler_map={line1: HandlerLine2D(numpoints=1),line2: HandlerLine2D(numpoints=1),
#                         line3: HandlerLine2D(numpoints=1)})
# plt.ylabel('Running Time(s)')
# plt.xlabel('Resource Capacity')
# # plt.show()
# plt.savefig('pictures/network-DP-time-3resource')


# In[16]:

# compare different numbers of virtual classes that DAVN decomposes into, in terms of revenue performance
def DAVN_compare_n_vc(total_num, n_spoke, cap, iterations, demand_type, n_virtual_classes):
    col_titles = ["rev_DLPVD", "LF_DLPVD","rev_DLPDAVN_mean %", "loadF_DLPDAVN_mean %", "DLPDAVN_time"]
    table_data = []
    problems = generate_samples(total_num, n_spoke, cap, demand_type, 1)
    n_vc = len(n_virtual_classes)
    for prob in problems:
        compare_results = [[] for _ in range(len(col_titles))]
        for index in [2, 3, 4]:
            compare_results[index] = [[] for _ in range(n_vc)]
#         print(prob)
        products = prob[0]
        resources = prob[1]
        capacities = prob[2]
        total_time = prob[3]
        demand_model = prob[4]
#         print(demand_model.arrival_rates)
        DLPVD_model = RM_approx.DLPVD(products, resources, capacities, total_time, demand_model)
        DLPDAVN_models = []
        for index in range(n_vc):
            DLPDAVN_models.append(RM_approx.DLP_DAVN(products, resources, capacities, total_time,                                                     n_virtual_classes[index], demand_model))
            
        for i in range(iterations):
            requests = demand_model.sample_network_arrival_rates()
            DLPVD_result = DLPVD_model.performance(requests)
            
            DLPVD_rev = DLPVD_result[0]
            DLPVD_LF = DLPVD_result[1]
            compare_results[0].append(DLPVD_rev)
            compare_results[1].append(DLPVD_LF)
            
            for index in range(n_vc):
                DLPDAVN_time = time.time()
                DLPDAVN_result = DLPDAVN_models[index].performance(requests)
                DLPDAVN_time = time.time() - DLPDAVN_time
                compare_results[-1][index].append(DLPDAVN_time)
                compare_results[2][index].append((DLPDAVN_result[0] - DLPVD_rev)/DLPVD_rev * 100)
                compare_results[3][index].append((DLPDAVN_result[1] - DLPVD_LF)/DLPVD_LF * 100)
                
        results = [np.mean(compare_results[0]), np.mean(compare_results[1])]
        results += [np.mean(compare_results[2], 1)]
        results += [np.mean(compare_results[3], 1)]
        results.append(np.mean(compare_results[-1], 1))
        
        table_data.append(results)
        
    df = pandas.DataFrame(table_data,  columns = col_titles)
    print(df)
    return table_data, df
    
VCs = [1,2,3,4]
tim = time.time()
result, dataframe = DAVN_compare_n_vc(10, 4, 6, 20, 2, VCs)
print("time consumed: ", time.time() - tim)
markers = ['^', 'o', 'x', 's']
pic_name = ['pictures/DLPDAVN_VCs_', '_perf_vs_DLPVD, freq1, demand1,spoke3']

x = [data[1] for data in result]
DLPDAVN_perf = [data[2] for data in result]
DLPDAVN_LF = [data[3] for data in result]
DLPDAVN_time = [data[4] for data in result]

xint = range(math.floor(min(x)) - 3, math.ceil(max(x))+ 3)
plt.clf()
for i in range(len(VCs)):
    plt.scatter(x, [item[i] for item in DLPDAVN_perf], marker = markers[i], label = 'vc='+ str(VCs[i]))
plt.legend()
plt.ylabel('Revenue Difference against DLPVD %')
plt.xlabel('Load Factor by DLPVD % ')
# plt.xticks(xint)
plt.ylim(plt.ylim()[0] - 3, plt.ylim()[1] + 3)
# plt.show()
plt.savefig(''.join([pic_name[0], 'rev', pic_name[1]]))

plt.clf()
for i in range(len(VCs)):
    plt.scatter(x, [item[i] for item in DLPDAVN_LF], marker = markers[i], label = 'vc='+ str(VCs[i]))
plt.legend()
plt.ylabel('Load Factor Difference against DLPVD %')
plt.xlabel('Load Factor by DLPVD % ')
# plt.xticks(xint)
plt.ylim(plt.ylim()[0] - 3, plt.ylim()[1] + 3)
# plt.show()
plt.savefig(''.join([pic_name[0], 'LF', pic_name[1]]))

plt.clf()
x = VCs[:]
y = np.mean(DLPDAVN_time, 0)
plt.scatter(x, y, marker = 'o')
plt.ylabel('Running Time(s)')
plt.xlabel('Max number of virtual classes')
# plt.show()
plt.savefig(''.join([pic_name[0], 'time', pic_name[1]]))

file_name = 'csv-files/DLPDAVN_VCs_performance.csv'
dataframe.to_csv(file_name, sep='\t')


# In[52]:

# # compare performances of DLPDAVN and LPADP against DLPVD
# def compare_with_DLPVD(total_num, n_spoke, cap, iterations, demand_type, n_virtual_classes, Ks):
#     col_titles = ["rev_DLPVD", "LF_DLPVD","rev_DLPDAVN_mean %", "loadF_DLPDAVN_mean %", "rev_LPADP_mean %", 
#                   "loadF_LPADP_mean %"]
#     table_data = []
#     problems = generate_samples(total_num, n_spoke, cap, demand_type, 1)
#     for prob in problems:
#         compare_results = [[] for _ in range(len(col_titles))]
        
#         products = prob[0]
#         resources = prob[1]
#         capacities = prob[2]
#         total_time = prob[3]
#         demand_model = prob[4]
        
#         DLPVD_model = RM_approx.DLPVD(products, resources, capacities, total_time, demand_model)
#         DLPDAVN_models = [RM_approx.DLP_DAVN(products, resources, capacities, total_time, n_vc, demand_model)
#                           for n_vc in n_virtual_classes]
#         LPADP_model = RM_ADP.ALP(products, resources, capacities, total_time, demand_model)
#         compare_results[2] = [[] for _ in range(len(n_virtual_classes))]
#         compare_results[3] = [[] for _ in range(len(n_virtual_classes))]
#         compare_results[4] = [[] for _ in range(len(Ks))]
#         compare_results[5] = [[] for _ in range(len(Ks))]
        
#         for i in range(iterations):
#             requests = demand_model.sample_network_arrival_rates()
            
#             DLPVD_result = DLPVD_model.performance(requests)
#             DLPVD_rev = DLPVD_result[0]
#             DLPVD_LF = DLPVD_result[1]
            
#             compare_results[0].append(DLPVD_rev)
#             compare_results[1].append(DLPVD_LF)
            
            
#             for p in range(len(n_virtual_classes)):
#                 DLPDAVN_result = DLPDAVN_models[p].performance(requests)
#                 compare_results[2][p].append((DLPDAVN_result[0] - DLPVD_rev)/DLPVD_rev * 100)
#                 compare_results[3][p].append((DLPDAVN_result[1] - DLPVD_LF) / DLPVD_LF * 100)
            
#             for q in range(len(Ks)):
#                 LPADP_bid_prices = LPADP_model.get_bid_prices(Ks[q])
#                 eval_results = RM_compare.simulate_network_bidprices_control([LPADP_bid_prices], products, resources, \
#                                                                              capacities, total_time, requests)

#                 LPADP_results = eval_results[0]
#                 compare_results[4][q].append((LPADP_results[0] - DLPVD_rev)/DLPVD_rev * 100)
#                 compare_results[5][q].append((LPADP_results[1] - DLPVD_LF) / DLPVD_LF * 100)
            
#         problem_result = [np.mean(result) for result in compare_results[:2]]
#         problem_result += [np.mean(result, 1) for result in compare_results[2:4]]
#         problem_result += [np.mean(result, 1) for result in compare_results[4:6]]
#         table_data.append(problem_result)
            
#     df = pandas.DataFrame(table_data,  columns = col_titles)
#     print(df)
#     return table_data, df
    
# VCs = [2, 4]
# Ks = [50, 100, 200]
# markers = ['^', 'o', 'x']
# tim = time.time()
# result, dataframe = compare_with_DLPVD(1, 3, 3, 1, 1, VCs, Ks)

# pic_name = ['pictures/rev_perf_vs_DLPVD_', ', demand1, spoke3']
# x = [data[1] for data in result]
# DLPDAVN_perf = [data[2] for data in result]
# LPADP_perf = [data[4] for data in result]

# xint = range(math.floor(min(x)) - 3, math.ceil(max(x))+ 3)

# plt.clf()
# for i in range(len(VCs)):
#     plt.scatter(x, [item[i] for item in DLPDAVN_perf], marker = markers[i], label = 'vc='+ str(VCs[i]))
# plt.legend()
# plt.ylabel('Revenue Difference against exactDP %')
# plt.xlabel('Load Factor by DLPVD % ')
# # plt.xticks(xint)
# plt.ylim(plt.ylim()[0] - 3, plt.ylim()[1] + 3)
# # plt.show()
# plt.savefig(''.join([pic_name[0], 'DLPDAVN', pic_name[1]]))

# plt.clf()
# for i in range(len(Ks)):
#     plt.scatter(x, [item[i] for item in LPADP_perf], marker = markers[i], label = 'K='+ str(Ks[i]))
# plt.legend()
# plt.ylabel('Revenue Difference against exactDP %')
# plt.xlabel('Load Factor by DLPVD % ')
# # plt.xticks(xint)
# plt.ylim(plt.ylim()[0] - 3, plt.ylim()[1] + 3)
# # plt.show()
# plt.savefig(''.join([pic_name[0], 'LPADP', pic_name[1]]))

# file_name = 'csv-files/perf_against_DLPVD.csv'
# dataframe.to_csv(file_name, sep='\t')


# In[55]:

# compare different numbers of states that LPADP samples to obtain conditions, in terms of revenue performance
def LPADP_compare_K(total_num, n_spoke, cap, iterations, demand_type, Ks):
    col_titles = ["rev_DLPVD", "LF_DLPVD","rev_LPADP_mean %", "loadF_LPADP_mean %", "LPADP_time"]
    table_data = []
    problems = generate_samples(total_num, n_spoke, cap, demand_type, 1)
    n_Ks = len(Ks)
    for prob in problems:
        compare_results = [[] for _ in range(len(col_titles))]
        for index in [2, 3, 4]:
            compare_results[index] = [[] for _ in range(n_Ks)]
        
        products = prob[0]
        resources = prob[1]
        capacities = prob[2]
        total_time = prob[3]
        demand_model = prob[4]
        
        LPADP_model = RM_ADP.ALP(products, resources, capacities, total_time, demand_model)
        LPADP_bid_prices = []
        DLPVD_model = RM_approx.DLPVD(products, resources, capacities, total_time, demand_model)
        
        for index in range(n_Ks):
            LPADP_time = time.time()
            LPADP_bid_prices.append(LPADP_model.get_bid_prices(Ks[index]))
            LPADP_time = time.time() - LPADP_time
            
            compare_results[-1][index].append(LPADP_time)
            
        for i in range(iterations):
            requests = demand_model.sample_network_arrival_rates()
            DLPVD_result = DLPVD_model.performance(requests)
            DLPVD_rev = DLPVD_result[0]
            DLPVD_LF = DLPVD_result[1]
            compare_results[0].append(DLPVD_rev)
            compare_results[1].append(DLPVD_LF)
            
            eval_results = RM_compare.simulate_network_bidprices_control(LPADP_bid_prices, products, resources,                                                                          capacities, total_time, requests)
            for index in range(n_Ks):
                compare_results[2][index].append((eval_results[index][0] - DLPVD_rev)/ DLPVD_rev * 100)
                compare_results[3][index].append((eval_results[index][1] - DLPVD_LF)/ DLPVD_LF * 100)
                
        results = [np.mean(compare_results[0]), np.mean(compare_results[1])]
        results += [np.mean(compare_results[2], 1)]
        results += [np.mean(compare_results[3], 1)]
        results.append(np.mean(compare_results[-1], 1))
        
        table_data.append(results)
            
    df = pandas.DataFrame(table_data,  columns = col_titles)
    print(df)
    return table_data, df
    
Ks = [10, 50, 100, 150, 200]
tim = time.time()
result, dataframe = LPADP_compare_K(1, 3, 3, 2, 2, Ks)
print("consumed time: ", time.time() - tim)

markers = ['^', 'o', 'x', 's', 'd']
pic_name = ['pictures/LPADP_Ks_', '_perf_vs_DLPVD']
x = [data[1] for data in result]
LPADP_perf = [data[2] for data in result]
LPADP_LF = [data[3] for data in result]
LPADP_time = [data[4] for data in result]

xint = range(math.floor(min(x)) - 3, math.ceil(max(x))+ 3)
plt.clf()
for i in range(len(Ks)):
    plt.scatter(x, [item[i] for item in LPADP_perf], marker = markers[i], label = 'K='+ str(Ks[i]))
plt.legend()
plt.ylabel('Revenue Difference against DLPVD %')
plt.xlabel('Load Factor by DLPVD % ')
# plt.xticks(xint)
plt.ylim(plt.ylim()[0] - 3, plt.ylim()[1] + 3)
# plt.show()
plt.savefig(''.join([pic_name[0], 'rev', pic_name[1]]))

plt.clf()
for i in range(len(Ks)):
    plt.scatter(x, [item[i] for item in LPADP_LF], marker = markers[i], label = 'K='+ str(Ks[i]))
plt.legend()
plt.ylabel('Load Factor Difference against DLPVD %')
plt.xlabel('Load Factor by DLPVD % ')
# plt.xticks(xint)
plt.ylim(plt.ylim()[0] - 3, plt.ylim()[1] + 3)
# plt.show()
plt.savefig(''.join([pic_name[0], 'LF', pic_name[1]]))

x = Ks[:]
plt.clf()
y = np.mean(LPADP_time, 0)
plt.scatter(x, y, marker = 'o')
plt.ylabel('Running Time(s)')
plt.xlabel('K')
plt.ylim(plt.ylim()[0] - 3, plt.ylim()[1] + 3)
# plt.show()
plt.savefig(''.join([pic_name[0], 'time', pic_name[1]]))


file_name = 'csv-files/LPADP_Ks_performance.csv'
dataframe.to_csv(file_name, sep='\t')


# In[ ]:

p = 0.5
def generate_samples_vary_time(total_num, n_spoke, demand_type, n_fare_class):
    """ generate a collection of random problems to be used in evaluation,
    each specifying products, resources, capacities of resources, total time, demand model"""
    problem_sets = []
    for i in range(total_num):
        resources, itineraries, arrival_rates = generate_network(n_spoke, demand_type, n_fare_class)
        products = extract_legs_info(itineraries, resources)
        if n_spoke == 3:
            caps = [3,5]
                
        else:
            caps = [5,10]
            
        for c in caps:
            capacities = [c] * len(resources)
            
            times = [3,4]
            for t in times:
                total_time = c * len(resources)*t
                
                dm = None
                dm = RM_demand_model.model(arrival_rates, total_time, demand_type, p)
        
                problem = [products, resources, capacities, total_time, dm]
                problem_sets.append(problem)
    print(len(problem_sets))
    return problem_sets
    
# compare performances of DLPDAVN and LPADP against DLPVD
def compare_with_DLPVD(total_num, n_spoke, iterations, demand_type, n_vc, K):
    problems = generate_samples_vary_time(total_num, n_spoke, demand_type, 1)
    DLPVD_perf = [[] for _ in range(3)] # rev, load factor, time
    DLPDAVN_perf = [[] for _ in range(3)] # rev, load factor, time, compare with DLPVD
    LPADP_perf = [[] for _ in range(3)] # rev, load factor, time, compare with DLPVD
    
    for prob in problems:                  
        products = prob[0]
        resources = prob[1]
        caps = prob[2]
        total_t = prob[3]
        demand_model = prob[4]
        
        DLPVD_model = RM_approx.DLPVD(products, resources, caps, total_t, demand_model)
        DLPDAVN_model = RM_approx.DLP_DAVN(products, resources, caps, total_t, n_vc, demand_model)
        LPADP_model = RM_ADP.ALP(products, resources, caps, total_t, demand_model)

        for i in range(iterations):
            requests = demand_model.sample_network_arrival_rates()

            t = time.time()
            DLPVD_result = DLPVD_model.performance(requests)
            DLPVD_perf[2].append(time.time() - t)
            DLPVD_rev = DLPVD_result[0]
            DLPVD_perf[0].append(DLPVD_rev)
            DLPVD_LF = DLPVD_result[1]
            DLPVD_perf[1].append(DLPVD_LF)

            t = time.time()
            DLPDAVN_result = DLPDAVN_model.performance(requests)
            DLPDAVN_perf[2].append(time.time() - t)
            DLPDAVN_rev = (DLPDAVN_result[0] - DLPVD_rev)/DLPVD_rev * 100
            DLPDAVN_perf[0].append(DLPDAVN_rev)
            DLPDAVN_LF = (DLPDAVN_result[1] - DLPVD_LF) / DLPVD_LF * 100
            DLPDAVN_perf[1].append(DLPDAVN_LF)

            t = time.time()
            LPADP_bid_price = LPADP_model.get_bid_prices(K)
            eval_results = RM_compare.simulate_network_bidprices_control([LPADP_bid_price], products, resources,                                                                              caps, total_t, requests)

            LPADP_perf[2].append(time.time() - t)
            LPADP_result = eval_results[0]
            LPADP_rev = (LPADP_result[0] - DLPVD_rev)/DLPVD_rev * 100
            LPADP_perf[0].append(LPADP_rev)
            LPADP_LF = (LPADP_result[1] - DLPVD_LF) / DLPVD_LF * 100
            LPADP_perf[1].append(LPADP_LF)

    print("DLPVD average performance: rev, LF, time:", np.mean(DLPVD_perf, 1))
    print("DLPDAVN average performance: rev, LF, time:", np.mean(DLPDAVN_perf, 1))
    print("LPADP average performance: rev, LF, time:", np.mean(LPADP_perf, 1))
    
    pic_name = ['pictures/rev_perf_vs_DLPVD_', 'demand1, spoke3']
    x = DLPVD_perf[1][:]
    DLPDAVN_rev_diff = [DLPDAVN_perf[0]]
    DLPDAVN_LF_diff = [DLPDAVN_perf[1]]
    
    plot_graph(x, DLPDAVN_rev_diff, 'o', 'DLPDAVN, vc=' + str(n_vc), 'Revenue Difference against DLPVD %', 'DLPDAVN_rev')
    plot_graph(x, DLPDAVN_LF_diff, 's', 'DLPDAVN, vc=' + str(n_vc), 'LF Difference against DLPVD %', 'DLPDAVN_lf') 
    plot_graph(x, DLPDAVN_perf[2], '^', 'DLPDAVN, vc=' + str(n_vc),  'Running Time(s)', 'DLPDAVN_time')
#     plt.clf()
#     plt.scatter(x, DLPDAVN_rev_diff, marker = 'o', label =)
#     plt.legend()
#     plt.ylabel()
#     plt.xlabel('Load Factor induced by DLPVD % ')
#     plt.ylim(plt.ylim()[0] - 3, plt.ylim()[1] + 3)
#     plt.show()
#     plt.savefig(''.join([pic_name[0], 'DLPDAVN_rev', pic_name[1]]))
    
    
    LPADP_rev_diff = [LPADP_perf[0]]
    LPADP_LF_diff = [LPADP_perf[1]]
    plot_graph(x, LPADP_rev_diff, 'o', 'LPADP, K=' + str(K),  'Revenue Difference against DLPVD %', 'LPADP_rev')
    plot_graph(x, LPADP_LF_diff, 's', 'LPADP, K=' + str(K),  'LF Difference against DLPVD %', 'LPADP_lf')
    plot_graph(x, LPADP_perf[2], '^', 'LPADP, K=' + str(K),  'Running Time(s)', 'LPADP_time')

#     plt.clf()
#     plt.scatter(x, DLPDAVN_rev_diff, marker = 'o', label = 'DLPDAVN, vc = 3')
#     plt.legend()
#     plt.ylabel('Revenue Difference against DLPVD %')
#     plt.xlabel('Load Factor induced by DLPVD % ')
#     plt.ylim(plt.ylim()[0] - 3, plt.ylim()[1] + 3)
#     plt.show()
    
    
def plot_graph(x, y,m, l, x_name, g_name):
    DLPDAVN_rev_diff = [DLPDAVN_perf[0]]

    plt.clf()
    plt.scatter(x, y, marker = m, label = l)
    plt.legend()
    plt.ylabel(x_name)
    plt.xlabel('Load Factor induced by DLPVD % ')
    plt.ylim(plt.ylim()[0] - 3, plt.ylim()[1] + 3)
    plt.show()
#     plt.savefig(''.join([pic_name[0], g_name, pic_name[1]]))
    
VCs = 3
Ks = 100
markers = ['^', 'o', 'x']
pic_name = ['pictures/rev_perf_vs_DLPVD_', '_demand1, spoke3']
compare_with_DLPVD(10, 3, 5, 1, VCs, Ks)

# pic_name = ['pictures/rev_perf_vs_DLPVD_', ', demand1, spoke3']
# x = [data[1] for data in result]
# DLPDAVN_perf = [data[2] for data in result]
# LPADP_perf = [data[4] for data in result]

# xint = range(math.floor(min(x)) - 3, math.ceil(max(x))+ 3)

# plt.clf()
# for i in range(len(VCs)):
#     plt.scatter(x, [item[i] for item in DLPDAVN_perf], marker = markers[i], label = 'vc='+ str(VCs[i]))
# plt.legend()
# plt.ylabel('Revenue Difference against exactDP %')
# plt.xlabel('Load Factor by DLPVD % ')
# # plt.xticks(xint)
# plt.ylim(plt.ylim()[0] - 3, plt.ylim()[1] + 3)
# # plt.show()
# plt.savefig(''.join([pic_name[0], 'DLPDAVN', pic_name[1]]))

# plt.clf()
# for i in range(len(Ks)):
#     plt.scatter(x, [item[i] for item in LPADP_perf], marker = markers[i], label = 'K='+ str(Ks[i]))
# plt.legend()
# plt.ylabel('Revenue Difference against exactDP %')
# plt.xlabel('Load Factor by DLPVD % ')
# # plt.xticks(xint)
# plt.ylim(plt.ylim()[0] - 3, plt.ylim()[1] + 3)
# # plt.show()
# plt.savefig(''.join([pic_name[0], 'LPADP', pic_name[1]]))

# file_name = 'csv-files/perf_against_DLPVD.csv'
# dataframe.to_csv(file_name, sep='\t')



# In[17]:

print([[cap] * 3 for cap in [3,5]])


# In[18]:

a = [[1,2], [10, 20]]
print(np.mean(a, 0), np.mean(a, 1))


# In[ ]:



