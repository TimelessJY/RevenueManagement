
# coding: utf-8

# In[26]:

import pandas
import time
import sys
sys.path.append('.')
import RM_exact
import RM_approx
import RM_helper
import RM_ADP

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


# In[27]:

def compare_iDAVN_singleDPstatic(products, resources, n_class, cap_lb, cap_ub, cap_interval):
    """Compare the iterative DAVN method, with a collection of single-resource static DP model."""
    n_resources = len(resources)
    col_titles = ['DAVN:bid-p', 'DAVN:rev', 'DAVN:time']
    capacities = [c for c in range(cap_lb, cap_ub + 1, cap_interval)]
    for i in range(n_resources):
            resource_name = resources[i]
            col_titles.append('S-S: rev-' + resource_name)
    
    col_titles.append("S-S: sum")
    col_titles.append("S-S:total time")
    
    table_data = []
    (pros, demands, demands_with_names) = RM_helper.sort_product_demands(products)
    for cap in capacities:
        result= []
        caps = [cap] * n_resources
        
        DAVN_time = time.time()
        DAVN_bid_prices, DAVN_total_rev = RM_approx.iterative_DAVN(pros, resources, demands_with_names, n_class,                                                                   caps, caps)
        DAVN_time = time.time() - DAVN_time

        result.append(DAVN_bid_prices)
        result.append(DAVN_total_rev)
        result.append(DAVN_time)

        single_static_vf = []
        single_total_time =0
        for i in range(n_resources):
            resource_name = resources[i]
            products_i = [j for j in products if resource_name in j[0]]
            ps, ds, _ = RM_helper.sort_product_demands(products_i)
            
            single_time = time.time()
            problem = RM_exact.Single_RM_static(ps, ds, cap)
            vf_i = problem.calc_value_func()
            single_time = time.time() - single_time
            single_total_time += single_time
            
            single_static_vf.append(vf_i[0][-1][-1])
            result.append(vf_i[0][-1][-1])
        result.append(sum(single_static_vf))
        result.append(single_total_time)
        
        table_data.append(result)
    
    print(pandas.DataFrame(table_data, capacities, col_titles))
    return table_data


# Compare
products = [['1a', (17.3, 5.8), 1050], ['2a', (45.1, 15.0),950], ['3a', (39.6, 13.2), 699], ['4a', (34.0, 11.3),520],            ['1b', (20, 3.5), 501], ['2b', (63.1, 2.5), 352], ['3b', (22.5, 6.1), 722], ['1ab', (11.5, 2.1), 760],            ['2ab', (24.3, 6.4), 1400]]
resources = ['a', 'b']
# compare_iDAVN_singleDPstatic(products,resources, 6, 80, 120, 10)
# lb = 60
# # ub = 160
# ub = 70
# data = compare_iDAVN_singleDPstatic(products, resources, 4, lb, ub, 10)

# revs_DAVN = [d[1] for d in data]
# revs_singleDP = [d[5] for d in data]
# revs_diff_perc = [(d[1]-d[5]) / d[1] * 100 for d in data]
# time_DAVN = [d[2] for d in data]
# time_singleDP = [d[6] for d in data]

# plt.clf()
# x= np.linspace(lb, ub, (ub - lb) / 10 + 1)
# line1, = plt.plot(x,revs_DAVN, marker='^', label='DAVN')
# line2, = plt.plot(x,revs_singleDP, marker='o', label='Single_Static_DP')

# plt.legend(handler_map={line1: HandlerLine2D(numpoints=1),line2: HandlerLine2D(numpoints=1)})
# plt.ylabel('Expected Revenue')
# plt.xlabel('Resource Capacity')
# # plt.show()
# plt.savefig('DAVN-ssDP-revs')

# plt.clf()
# line3, = plt.plot(x,time_DAVN, marker='^', label='DAVN')
# line4, = plt.plot(x,time_singleDP, marker='o', label='Single_Static_DP')

# plt.legend(handler_map={line3: HandlerLine2D(numpoints=1),line4: HandlerLine2D(numpoints=1)})
# plt.ylabel('Running Time(s)')
# plt.xlabel('Resource Capacity')
# # plt.show()
# plt.savefig('DAVN-ssDP-time')
# plt.clf()
# plt.plot(x, revs_diff_perc, 'bo-')
# plt.ylabel('Revenue Difference(%)')
# plt.xlabel('Resource Capacity')
# plt.savefig('DAVN-ssDP-diff')


# In[28]:

def evaluate_network_control(products, resources, demands, capacities, approxed_bid_prices, total_time, iterations):
    """using the given bid-prices of a heuristic/approximation to evaluate the difference between revenues gained 
    between that heuristic/approximation with optimal method, i.e. network-DP model"""
    incidence_matrix = RM_helper.calc_incidence_matrix(products, resources)
    
    diff_percents = []
    
    exact_method = RM_exact.Network_RM(products, resources, [demands], capacities, total_time)
    exact_method.value_func()
    exact_bid_prices = exact_method.bid_prices()
    
    for round in range(iterations):
        requests = RM_helper.sample_network_demands(demands, total_time)

        rev_exact = 0 # records the total revenue using the optimal control
        curr_cap_exact = capacities[:]
        rev_heuri = 0 # records the total revenue using the control produced by heuristic/approximation
        curr_cap_heuri = capacities[:]

        total_states = 1
        for c in capacities:
                total_states *= (c+1)

        for t in range(total_time):
            prod_requested = requests[t]
            if prod_requested < len(products):
                # a request arrives
                incidence_vector = [row[prod_requested] for row in incidence_matrix]
                state_index_exact = RM_helper.state_index(total_states, capacities, curr_cap_exact)
                state_index_heuri = RM_helper.state_index(total_states, capacities, curr_cap_heuri)
                
                rev = products[prod_requested][1]
                if decide_to_sell(incidence_vector, curr_cap_exact, exact_bid_prices, rev, t, state_index_exact):
                    curr_cap_exact = [c_i - x_i for c_i, x_i in zip(curr_cap_exact, incidence_vector)]
                    rev_exact += rev
                if decide_to_sell(incidence_vector, curr_cap_heuri, approxed_bid_prices, rev, t, state_index_heuri):
                    curr_cap_heuri = [c_i - x_i for c_i, x_i in zip(curr_cap_heuri, incidence_vector)]
                    rev_heuri += rev
#         print("exact rev=", rev_exact, ", heuristic rev=", rev_heuri)
        if rev_exact >= rev_heuri:
            diff = rev_exact - rev_heuri
            diff_percent = (diff / rev_exact) * 100
            diff_percents.append(diff_percent)
    
    avrg_diff_percent = np.mean(diff_percents)
    return avrg_diff_percent
            
def decide_to_sell(incidence_vector, remained_cap, resource_bid_prices, profit, t, s):
    """deicide at time t, state s, whether to sell the product according to its profit"""
    if t < len(resource_bid_prices) - 1:
        bid_prices = resource_bid_prices[t+1][s]
        opportunity_cost = np.dot(incidence_vector, bid_prices)
    else:
        opportunity_cost = 0
    return all(x_i <= c_i for x_i, c_i in zip(incidence_vector, remained_cap)) and profit >= opportunity_cost


# In[29]:

def eval_ADP_DPf(pros, resources, capacities, total_time, iterations):
    """Compare the ADP algorithm using DP model with feature extraction, with exact DP model of network problems."""
    products, demands,_ = RM_helper.sort_product_demands(pros)
    problem = RM_ADP.DP_w_featureExtraction(products, resources, [demands], capacities, total_time)
    problem.value_func("")
    bid_prices = problem.bid_prices()
    diff_percent = evaluate_network_control(products, resources, demands, capacities, bid_prices, total_time,                                             iterations)
    return diff_percent

def visualize_perf_ADP_DPf(products, resources, T_lb, T_ub, T_interval, cap_lb, cap_ub, cap_interval, iterations):
    """Visualize the performance of ADP_DPf method, against network_DP model."""
    n_resources = len(resources)
    Ts = [T for T in range(T_lb, T_ub + 1, T_interval)]
    col_titles = [('diff-T='+str(T)) for T in Ts]
    col_titles.append('mean_diff')
    
    capacities = [c for c in range(cap_lb, cap_ub + 1, cap_interval)]
    table_data = []
    
    for cap in capacities:
        caps = [cap] * n_resources
        result= []
        for T in Ts:
            result.append(eval_ADP_DPf(products, resources, caps, T, iterations))
        
        result.append(np.mean(result))
        table_data.append(result)
    
    print(pandas.DataFrame(table_data, capacities, col_titles))
    return table_data

# ps = [['a1', 0.22, 200], ['a2', 0.06, 503], ['ab1', 0.18, 400],['ab2', 0.1, 704], ['b1', 0.05, 601], \
#       ['b2', 0.12, 106], ['bc', 0.13, 920],['c1', 0.07, 832]]
# resources = ['a', 'b', 'c']

# T_lb = 10
# T_ub = 20
# T_interval = 10
# cap_lb = 2
# cap_ub = 20
# cap_interval = 8 
# iterations = 20
# data = visualize_perf_ADP_DPf(ps, resources, T_lb, T_ub, T_interval, cap_lb, cap_ub, cap_interval, iterations)

# plt.clf()
# x= np.linspace(cap_lb, cap_ub, (cap_ub - cap_lb) / cap_interval + 1)
# for i in range(1):
#     T = T_lb + T_interval * i
#     rev_diff = [d[i] for d in data]
#     plt.plot(x, rev_diff, linestyle='dashed', marker='s', label="T="+str(T))
    
# plt.legend()
# plt.ylabel('Difference in Expected Revenue(%)')
# plt.xlabel('Resource Capacity')
# plt.show()
# plt.savefig('ADP_DPf_networkDP-rev-diff')


# In[30]:

def simulate_single_static_bidprices_control(bid_prices, products, demands, capacity, requests = []):
    """Simulates bid-price control, on a single-static problem, with initial capacity given. 
    ----------------------------
    Inputs:
        bid_prices: bid prices of methods to be simulated
        products: i.e. itineraries, assumed to be sorted in descending order of revenus, in the form of 
                (name, revenue)
        demands: mean and std of demand distribution for products, in the same order as the products are given
        capacity: initial capacity of the resource
        requests: demand for each product, might not be given
    Returns: total revenue and load factor of each method. """
    
    n_methods = len(bid_prices)
    if not requests:
        requests = RM_helper.sample_single_static_demands(demands)
    revs = [0] * n_methods # records the total revenue using bid prices produced by the two methods, i.e. bid_prices
    curr_cap = [capacity] * n_methods

    n_products = len(products)
    for m in range(n_methods):
        for fare_class in range(n_products - 1, 0, -1):
            price = products[fare_class][1]
            bid_price_j = bid_prices[m][fare_class - 1]
            remain_cap = curr_cap[m]
            if price >= bid_price_j[remain_cap]:
                # only sell product of current fare class if its profit exceeds the bid price of current class
                for z in range(min(requests[fare_class], remain_cap), -1, -1):
                    if price >= bid_price_j[remain_cap - z]:
                        curr_cap[m] -= z
                        revs[m] += price * z
                        break
        # for the highest fare class, accept all requests
        request = requests[0]
        z = min(request, curr_cap[m])

        curr_cap[m] -= z
        revs[m] += products[0][1] * z
    
    result = [(revs[m], round((capacity - curr_cap[m]) / capacity * 100,3)) for m in range(n_methods)]
    return result

def simulate_single_static_protectionlevel_control(protection_levels, products, demands, capacity, requests = []):
    """Simulates protection-level control, on a single-static problem, with initial capacity given. 
    ----------------------------
    Inputs:
        protection_levels: protection levels of methods to be simulated
        products: i.e. itineraries, assumed to be sorted in descending order of revenus, in the form of 
                (name, revenue)
        demands: mean and std of demand distribution for products, in the same order as the products are given
        capacity: initial capacity of the resource
        requests: demand for each product, might not be given
    Returns: total revenue and load factor of each method. """
    
    n_methods = len(protection_levels)
    if not requests:
        requests = RM_helper.sample_single_static_demands(demands)
    revs = [0] * n_methods # records the total revenue using bid prices produced by the two methods, i.e. bid_prices
    curr_cap = [capacity] * n_methods

    n_products = len(products)
    for m in range(n_methods):
        for fare_class in range(n_products - 1, 0, -1):
            price = products[fare_class][1]
            pl_j = protection_levels[m][fare_class - 1]
            remain_cap = curr_cap[m]

            decision = int(min(max(0, remain_cap - pl_j), requests[fare_class]))

            curr_cap[m] -= decision
            revs[m] += price * decision
        # for the highest fare class, accept all requests
        request = requests[0]
        decision = min(request, curr_cap[m])

        curr_cap[m] -= decision
        revs[m] += products[0][1] * decision
    
    result = [(revs[m], round((capacity - curr_cap[m]) / capacity * 100,3)) for m in range(n_methods)]
    return result

pros = [[1, 1050,(17.3, 5.8)], [2, 950, (45.1, 15.0)], [3, 699, (39.6, 13.2)], [4,520,(34.0, 11.3)]]
cap = 80
products, demands, _ = RM_helper.sort_product_demands(pros)
exact = RM_exact.Single_RM_static(products, demands, cap)
# exact_bid_prices = exact.get_bid_prices()
# print(simulate_single_static_bidprices_control([exact_bid_prices], products, demands, cap))

# exact_pl = exact.get_protection_levels()
# EMSR = RM_approx.Single_EMSR(products, demands, cap)
# EMSR_pl = EMSR.get_protection_levels()
# print(simulate_single_static_protectionlevel_control([exact_pl, EMSR_pl], products, demands, cap))


# In[31]:

def simulate_network_bidprices_control(bid_prices, products, resources,  capacities, T, requests):
    """Simulates bid-price control over the horizon T, on a network problems, with initial capacity given. 
    ----------------------------
    Inputs:
        bid_prices: bid prices of methods to be simulated
        products: i.e. itineraries, assumed to be sorted in descending order of revenus, in the form of (name, revenue)
        resources: i.e. flight legs
        capacities: initial capacities of resources
        T: total time, i.e. sales horizon
        requests: indicies of products that are requested during the actual sales
    Returns: total revenue and load factor of each method. """
    
    n_methods = len(bid_prices)
    incidence_matrix = RM_helper.calc_incidence_matrix(products, resources)
    revs = [0] * n_methods # records the total revenue using bid prices produced by the two methods, i.e. bid_prices
    curr_caps = [capacities[:]] * n_methods

    total_states = 1
    for c in capacities:
        total_states *= (c+1)

    for t in range(T):
        prod_requested = requests[t]
        if prod_requested < len(products):
            # a request arrives
            incidence_vector = [row[prod_requested] for row in incidence_matrix]
            state_index = [RM_helper.state_index(total_states, capacities, curr_caps[i]) for i in range(n_methods)]

            profit = products[prod_requested][1]
            for i in range(n_methods):
                if t < (T - 1): 
                    bp_t = bid_prices[i][t+1][state_index[i]]
                    opportunity_cost = np.dot(incidence_vector, bp_t)
                else:
                    opportunity_cost = 0
                    
                remain_cap = [c_i - x_i for c_i, x_i in zip(curr_caps[i], incidence_vector)]
                
                if profit >= opportunity_cost and all(c_i >= 0 for c_i in remain_cap):
                    # decides to sell the product
                    revs[i] += profit
                    curr_caps[i] = remain_cap[:]
                    
    result = [(revs[i], (1 - np.mean([r / c for r, c in zip(curr_caps[i], capacities)])) * 100)                for i in range(n_methods)]
    return result


# In[ ]:



