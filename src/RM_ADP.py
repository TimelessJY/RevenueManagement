
# coding: utf-8

# In[1]:

import numpy as np
import scipy.stats
import time
import random
import bisect

import sys
sys.path.append('.')
import RM_helper


# In[3]:

##############################################
###### ADP: using one-state_transition #######
##############################################

class One_state_transition():
    """ ADP algorithm using the one-step transition matix"""
    value_functions = []
    incidence_matrix = []
    default_iterations = 100
    
    def __init__(self, products, resources, demands, capacities, total_time):
        self.products = products
        self.resources = resources
        self.demands = demands
        self.capacities = capacities
        self.total_time = total_time
        self.n_products = len(products)
        self.n_resources = len(resources)
        self.n_demand_periods = len(demands)
        
        self.n_states = 1
        
        for c in self.capacities:
            self.n_states *= (c+1)
            
        self.incidence_matrix = RM_helper.calc_incidence_matrix(products, resources)
    
    def eval_value(t, control, state_num, product_num):
        """helper func: evaluate the value for period t and state x, ref: equation 3.1 in the book"""

        price_vector = [0] * self.n_products
        price_vector[product_num] = self.products[product_num][1]
        value = np.dot(price_vector, control)
        Au = np.dot(self.incidence_matrix, control).tolist()
        if t < self.total_time - 1:
            x_Au = [x_i - Au_i for x_i, Au_i in zip(RM_helper.remain_cap(self.n_states, self.capacities,state_num), Au)]
            state_x_Au = RM_helper.state_index(self.n_states, self.capacities,x_Au)
            value += self.value_functions[t+1][RM_helper.state_index(self.n_states, self.capacities,x_Au)]
        return value                

    def cumulative_probs(self,demands):
        cumu_prob = [0] * len(demands)
        up_to = 0
        for i in range(len(demands)):
            up_to += demands[i]
            cumu_prob[i] = up_to

        cumu_prob.append(1.0)
        return cumu_prob

    def sample_const_demands(self):
        """helper func: samples a series of index of products, whose request arrives at each period in the given total time """
        demands = self.demands[0]
        cumu_prob = self.cumulative_probs(demands)
        sample_index = [0] * self.total_time
        for t in range(self.total_time):
            rand = random.random()
            fall_into = bisect.bisect(cumu_prob, rand)
            sample_index[t] = fall_into
        return sample_index

    def sample_change_demands(self):
        """helper func: samples a series of index of products, whose request arrives at each period in the given total time """
        sample_index = [0] * self.total_time
        for t in range(self.total_time):
            demand_t = self.demands[t]
            cumu_prob = self.cumulative_probs(demand_t)

            rand = random.random()
            fall_into = bisect.bisect(cumu_prob, rand)
            sample_index[t] = fall_into
            print("cumu_prob = ", cumu_prob)
            print("random = ", rand, " fall into ", fall_into)
        return sample_index
    
    def value_func(self, n_iterations):
        prev_ests = [[0] * self.n_states for _ in range(self.total_time + 1)]
        n = 1 # iteration counter 

        while n <= n_iterations:
            S_t = self.n_states - 1 # initial state
            # Step 1, choose a sample path
            if self.n_demand_periods == 1:
                sampled_demands = self.sample_const_demands()
            else:
                sampled_demands = self.sample_change_demands()

#             print("at n=",n, "sampled requests are:", sampled_demands)
            # Step 2, iterate over all time points
            for t in range(self.total_time):
                current_estimation = prev_ests[:]
                request_product = sampled_demands[t]
                next_state = None

                if request_product == self.n_products: # sampled demand: no requests arrived
                    est_t = prev_ests[t+1][S_t]
                    next_state = S_t
                else: # sampled demand: a request for a product arrived
                    u = [0] * self.n_products # control variable

                    # consider between actions: don't sell (u)
                    value_not_sell = prev_ests[t+1][S_t]

                    # or sell(u_sell)
                    value_sell = 0
                    S_after_sell = S_t
                    cap_vector = RM_helper.remain_cap(self.n_states, self.capacities,S_t)
                    incidence_vector = [row[request_product] for row in self.incidence_matrix]
                    diff = [x_i - a_j_i for a_j_i, x_i in zip(incidence_vector, cap_vector)]
                    if all(diff_i >= 0 for diff_i in diff):
                        u_sell = [0] * self.n_products
                        u_sell[request_product] = 1

                        # find the state after selling the product
                        Au = np.dot(self.incidence_matrix, u_sell).tolist()
                        x_Au = [x_i - Au_i for x_i, Au_i in zip(cap_vector, Au)] # updated remaining capacity vector
                        S_after_sell = RM_helper.state_index(self.n_states, self.capacities,x_Au)
                        # estimated the value function if selling the product
                        value_sell = products[request_product][1] + prev_ests[t+1][S_after_sell]

                    if value_sell > value_not_sell:
                        est_t = value_sell
                        next_state = S_after_sell
                    else:
                        est_t = value_not_sell
                        next_state = S_t

                current_estimation[t][S_t] = est_t 
                S_t = next_state
                prev_ests = current_estimation

            n += 1
        self.value_functions = prev_ests
        return self.value_functions
    
    def bid_prices(self):
        """return the bid prices for resources over all time periods and all remaining capacities situations."""
        if not self.value_functions:
            self.value_func(self.default_iterations)

        return RM_helper.network_bid_prices(self.value_functions, self.products, self.resources, self.capacities,                                             self.incidence_matrix, self.n_states)

    def total_expected_revenue(self):
        if not self.value_functions:
            self.value_func(self.default_iterations)
            
        return self.approximations[0][-1]


ps = [['a1', 0.02, 200], ['a2', 0.06, 503], ['ab1', 0.08, 400],['ab2', 0.01, 704], ['b1', 0.05, 601],       ['b2', 0.12, 106], ['bc', 0.03, 920],['c1', 0.07, 832]]
products,demands, _ = RM_helper.sort_product_demands(ps)
demands = [demands]
resources = ['a', 'b', 'c']
capacities = [8] * 3

start_time = time.time()
problem = One_state_transition(products, resources, demands, capacities, 10)
vf = problem.value_func(1000)
# print(problem.bid_prices()
# print(vf)
print("--- %s seconds ---" % (time.time() - start_time))


# In[13]:

#############################################
###### ADP: DP with feature extraction ######
#############################################

class DP_w_featureExtraction():
    """ADP algorithm, using DP model with feature-extraction method."""
    incidence_matrix = []
    approximations = []
    default_method = "separable_affine"
    
    def __init__(self, products, resources, demands, capacities, total_time):
        
        self.products = products
        self.resources = resources
        self.demands = demands
        self.capacities = capacities
        self.total_time = total_time
        self.n_products = len(products)
        self.n_resources = len(resources)
        self.n_demand_periods = len(demands)
        
        self.n_states = 1
        
        for c in self.capacities:
            self.n_states *= (c+1)
            
        self.incidence_matrix = RM_helper.calc_incidence_matrix(products, resources)
    
    def value_func(self, feature_approx_method = "", m = 0):
        """Calculate the value functions, 
        using the given feature approximation method(default as separable affine, ref: An ADP approach to Network RM),
        with m states chosen in each time period to get observations. """
        
        if not feature_approx_method:
            feature_approx_method = self.default_method
            
        if m <= 0:
            m = int(self.n_states / 2)
            
        self.approximations = [[0] * self.n_states for _ in range(self.total_time)]
        t = self.total_time - 1

        for t in range(self.total_time -1, -1, -1):
            while(True):
                # choose m states, and evaluate the value in those states
                m_states = self.choose_m_states(m)
                m_vals = self.eval_values(m_states, t)

                # extract feature vectors from these m states, and solve for the optimal coefficients
                m_features = []
                for s in m_states:
                    feature_vector = self.extract_features(s, feature_approx_method)
                    m_features.append(feature_vector)

                find_coeff = self.solve_LLS_prob(m_features, m_vals)
                if find_coeff[0]:
                    break
                else:
                    # if the B matrix of the selected m states is singular, repeat the above process
                    continue

            r = find_coeff[1]
            
            # use the optimal coefficients computed to approximate value of other states
            for s in range(self.n_states):
                if s in m_states:
                    self.approximations[t][s] = round(m_vals[m_states.index(s)],4)
                else:
                    feature = self.extract_features(s, feature_approx_method)
                    approx_val = np.dot(feature, r)
                    self.approximations[t][s] = max(round(approx_val, 4), 0)
        
        return self.approximations
        
    def choose_m_states(self, m):
        """helper func: choose m states from all states, currently choosing randomly"""
        chosen_states = random.sample(range(0,self.n_states), m)
        return chosen_states
    
    def eval_values(self, m_states, t):
        """helper func: calculate the value of being at the given states, at time period t"""
        values = []
        demands_t = self.get_demands(t)
        for state in m_states:
            value = 0
            remain_cap = RM_helper.remain_cap(self.n_states, self.capacities, state)
            for f in range(self.n_products):
                incidence_vector = [row[f] for row in self.incidence_matrix]
                if all(A_f <= x for A_f,x in zip(incidence_vector,remain_cap)):
                    if t == (self.total_time - 1): 
                        # in the last time period, evaluate values of these m states exactly
                        value += demands_t[f] * self.products[f][1]
                    else:
                        # in other time periods, consider value in the next state as approximations produced
                        remain_cap_sell = [x - A_f for x, A_f in zip(remain_cap, incidence_vector)]
                        next_state_sell = RM_helper.state_index(self.n_states, self.capacities, remain_cap_sell)
                        value_sell = self.products[f][1] + self.approximations[t+1][next_state_sell]
                        
                        value_not_sell = self.approximations[t+1][state]
                        
                        value += demands_t[f] * max(value_sell, value_not_sell)
            
            if t < (self.total_time - 1):
                value += (1 - sum(demands_t)) * self.approximations[t+1][state]
            
            values.append(value)
        return values

    def extract_features(self, state, feature_approx_method):
        """helper func: use the given method to extract features of size (n_resource + 1) for the given states."""
        feature = []
        if feature_approx_method == self.default_method:
            remain_cap = RM_helper.remain_cap(self.n_states, self.capacities, state)
            feature = remain_cap[:]
            feature.append(1)
        else:
            """TODO: implement separable_concave"""
            
        return feature
        
    def solve_LLS_prob(self, feature_vectors, m_vals):
        """helper func: solve the linear least square problem, returns whether the B matrix produced by the selected
        m staes is singular, and if not, the optimal coefficients of basis functions. """
        m = len(m_vals)
        B = [[0 * (self.n_resources + 1)] for _ in range(self.n_resources + 1)]
        for feature_v in feature_vectors:
            B += np.outer(feature_v, feature_v)
            
        try:
            B_inverse = np.linalg.inv(B)
        except np.linalg.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                return(False, [])
            raise

        C = [0] * (self.n_resources + 1)
        for i in range(m):
            prod = [m_vals[i] * f_v for f_v in feature_vectors[i]]
            C = [C[r] + prod[r] for r in range(self.n_resources + 1)]
        coeff = B_inverse.dot(C)
        return (True, coeff)
              
    def get_demands(self, t):
        """helper func: return the demands of fare products in time period t. """
        if self.n_demand_periods > 1:
            return self.demands[t]
        else:
            return self.demands[0]
        
    def bid_prices(self):
        """return the bid prices for resources over all time periods and all remaining capacities situations."""
        if not self.approximations:
            self.value_func(self.default_method)
            
        return RM_helper.network_bid_prices(self.approximations, self.products, self.resources, self.capacities,                                             self.incidence_matrix, self.n_states)
    
    def total_expected_revenue(self):
        if not self.approximations:
            self.value_func(self.default_method)
            
        return self.approximations[0][-1]
    

# ps = [['a1', 0.02, 200], ['a2', 0.06, 503], ['ab1', 0.08, 400],['ab2', 0.01, 704], ['b1', 0.05, 601], \
#       ['b2', 0.12, 106], ['bc', 0.03, 920],['c1', 0.07, 832]]
# products,demands, _ = RM_helper.sort_product_demands(ps)
# demands = [demands]
# resources = ['a', 'b', 'c']
# capacities = [8] * 3

ps = [['a1', 0.02, 200], ['a2', 0.06, 503], ['ab1', 0.08, 400],['ab2', 0.01, 704], ['ab3', 0.05, 601],       ['ab4', 0.12, 106], ['bc', 0.03, 920],['c1', 0.07, 832]]
products,demands, _ = RM_helper.sort_product_demands(ps)
demands = [demands]
resources = ['a', 'b', 'c']
capacities = [5] * 3

start_time = time.time()
problem = DP_w_featureExtraction(products, resources, demands, capacities, 10)
# vf = problem.value_func()
# print(vf)
# print(problem.bid_prices())
# print(problem.total_expected_revenue())
# print("--- %s seconds ---" % (time.time() - start_time))


# In[43]:

import pulp

class ALP():
    """ADP algorithm, using Linear Programming approach, DP model with feature-extraction method.
    ref: An Approximate Dynamic Programming Approach to Network Revenue Mangement. """
    incidence_matrix = []
    approximations = []
    default_method = "separable_affine"
    
    def __init__(self, products, resources, demands, capacities, total_time, arrival_rates):
        
        self.products = products
        self.resources = resources
        self.demands = demands
        self.capacities = capacities
        self.total_time = total_time
        self.n_products = len(products)
        self.n_resources = len(resources)
        self.n_demand_periods = len(demands)
        self.product_names = [p[0] for p in products]
        self.prices = dict(products)
        self.arrival_rates = arrival_rates
        
        self.n_states = 1
        
        for c in self.capacities:
            self.n_states *= (c+1)
            
        self.incidence_matrix = RM_helper.calc_incidence_matrix(products, resources)

    def solve_DLP(self, remain_cap, curr_time):
        """in step 1: solves a DLP model, with the given remaining capacity, and the current time period; returns bid
        prices for resources. """
        DLP_model = pulp.LpProblem('DLP model', pulp.LpMaximize)
        y = pulp.LpVariable.dict('y_%s', self.product_names, lowBound= 0)
        
        # objective function
        DLP_model += sum([self.prices[j] * y[j] for j in self.product_names])
        
        # constraints 1, for each resource, the sum of products of consumption py each product and the booking limit of 
        # that product is less than the total capacity of that resource
        constraints = []
        for i in range(self.n_resources):
            incidence = self.incidence_matrix[i]
            incidence_vector = dict(zip(self.product_names, incidence))
            cap = remain_cap[i]
            c = sum([incidence_vector[i] * y[i] for i in self.product_names]) <= cap
            constraints.append(c)
            DLP_model += c, "c"+str(i)
            
        # constraints2, every booking limit should be less than the corresponding demand
        if self.n_demand_periods > 1:
            means = [d[0] for d in self.demands[curr_time]]
        else:
            means = [d[0] for d in self.demands[0]]
        means_vector = dict(zip(self.product_names, means))
        for j in self.product_names:
            DLP_model += y[j] <= means_vector[j]
                  
        DLP_model.solve()
#         print(DLP_model)
        
        bid_prices = [c.pi for c in constraints]
#         print("bid-prices= ", bid_prices)
        return bid_prices
    
    def simulate_bid_prices_control(self, initial_state, bid_prices):
        """helper func: sample a single request, and use the given bid-prices to simulate the optimal control."""
        sampled_request  = RM_helper.sample_network_demands(self.arrival_rates, 1)[0]
        # simulate a 1-time-period bid-price control
        new_state = initial_state[:]
        if sampled_request < self.n_products:
            # a request actually occurs
            incidence = [row[sampled_request] for row in self.incidence_matrix]
            if incidence <= initial_state and np.dot(incidence, bid_prices) <= self.products[sampled_request][1]:
                new_state = [new_i - inc_i for new_i, inc_i in zip(new_state, incidence)]
        return new_state

    def generate_basis_func(self, remain_cap):
        """helper func: generate basis function given the remaining capacity at the current time."""
        basis_funcs = []
        for i in range(self.n_resources):
            x_l = remain_cap[i] # remaining capacity for resource i
            b_f = [1] * (x_l + 1)
            zeros = [0] * (self.capacities[i] - x_l)
            b_f += zeros
            
            basis_funcs.append(b_f)
        return basis_funcs
    
    def find_available_products(self, remain_cap):
        """helper func: find all products whose consumptions of resources doesn't exceed current capacities. """
        avail_prod = []
        for j in range(self.n_products):
            incidence = [row[j] for row in self.incidence_matrix]
            if all(incidence[i] <= remain_cap[i] for i in range(self.n_resources)):
                avail_prod.append(j)
        return avail_prod
    
    def sample_visited_states(self, K):
        """Step 1: simulate K times, get a bid-price control policy, then use it to simulate a control, 
        and gather all visited states(in the form of remaining capacity of resources)"""
        visited_states = []
        curr_state = self.capacities[:]
        visited_states.append(curr_state)
        for i in range(K - 1):
            bid_prices = self.solve_DLP(curr_state, i)
            next_state = self.simulate_bid_prices_control(curr_state, bid_prices)
            visited_states.append(next_state)
            curr_state = next_state
        return visited_states
    
    def solve_RLP(self, sampled_states):
        """Step 2: use sampled states, to formulate and solve Relaxed LP"""
        """in step 1: solves a DLP model, with the given remaining capacity, and the current time period; returns bid
        prices for resources. """
        RLP_model = pulp.LpProblem('RLP model', pulp.LpMinimize)
        # names of variables r, total size: T * n_resources * (average initial capacity of resource)
        names = []
        flattened_names = []
        for t in range(self.total_time):
            name_t = []
            for i in range(self.n_resources):
                name_i = []
                for x in range(self.capacities[i] + 1):
                    name = self.resources[i] + '-' + str(t) + '-' + str(x)
                    name_i.append(name)
                    flattened_names.append(name)
                name_t.append(name_i)
            names.append(name_t)
        
        # names of variables y, total size: (number of sampled-states, i.e. K) - 1
        y_names = ['s' + str(i) for i in range(min(self.total_time - 1, len(sampled_states)))]
        # values that y variables help eliminating the max() operation on
        y_values = []
        
        # declare variables
        r = pulp.LpVariable.dict('r_%s', flattened_names, lowBound= 0)
        y = pulp.LpVariable.dict('y_%s', y_names)
        
        # objective function: minimize the value approximation of the initial state
        names_initial = [names[0][i] for i in range(self.n_resources)]
        flattened_names_initial = np.concatenate(names_initial, axis=0).tolist()
        
        r_initial = [r[name] for name in flattened_names_initial]
        RLP_model += sum(r_initial)
        
        # constraints 1, for each sampled state, TJ <= J
        constraints = []
        for t in range(len(sampled_states)):
            s = sampled_states[t]
            # calculate RHS, i.e. J(s), the state with current remaining capacity and at current time period
            basis_func_s = self.generate_basis_func(s) # basis function
            r_s_t = [[r[name] for name in n_i] for n_i in names[t]] # weights
            RHS = sum([np.dot(b_f_i, r_s_i) for b_f_i, r_s_i in zip(basis_func_s, r_s_t)])
            
            # calculate LHS, i.e. TJ(s)
            f_s = self.find_available_products(s) # products that can be sold, based on remaining capacities
            LHS = 0
    
            if t < (self.total_time - 1):
                r_s_t_next = [[r[name] for name in n_i] for n_i in names[t + 1]] # weights of next time periods
                # value approximation of the state with the same remaining capacity in next time period
                J_s = sum([np.dot(b_f_i, r_s_i) for b_f_i, r_s_i in zip(basis_func_s, r_s_t_next)])
                total_arrival_rate = 0
                
                y_values_t = []
                y_values_t.append(J_s)
                
                for f in f_s:
                    arrival_rate = self.arrival_rates[f]
                    total_arrival_rate += arrival_rate
                    
                    # approximates the value of the state in next time period, after selling product f
                    A_f = [row[f] for row in self.incidence_matrix]
                    s_f = [s_i - f_i for s_i, f_i in zip(s, A_f)]
                    basis_func_s_f = self.generate_basis_func(s_f)
                    J_s_f = sum([np.dot(b_f_i, r_s_i) for b_f_i, r_s_i in zip(basis_func_s_f, r_s_t_next)])
                    LHS += arrival_rate * y[y_names[t]]
                    y_values_t.append(self.products[f][1] + J_s_f)
                    
                LHS += (1- total_arrival_rate) * J_s
                y_values.append(y_values_t)
            else:
                LHS = sum([self.arrival_rates[f] * self.products[f][1] for f in f_s])

            constraint = LHS <= RHS
            RLP_model += constraint, "c" + str(t)
            
        # constraints 2, at each time period, for each resource, weights are in decreasing order
        for t in range(self.total_time):
            for i in range(self.n_resources):
                var_names = names[t][i]
                for x in range(1, self.capacities[i]):
                    constraint = r[var_names[x]] >= r[var_names[x+1]]
                    RLP_model += constraint
                
        # constraints 3, define the variables that helps eliminating the max() operations in constraints 1
        for t in range(len(sampled_states)):
            if t < self.total_time - 1:
                for y_v in y_values[t]:
                    RLP_model += y[y_names[t]] >= y_v
                
        RLP_model.solve()
#         print(RLP_model)
        varsdict = {}
        for v in RLP_model.variables():
            varsdict[v.name] = v.varValue
        return (varsdict, flattened_names)
    
    def collect_bid_prices(self, varsdict, varnames):
        """helper func: after step 2, collect bid prices for each time period and each state from the results of LP"""
        bid_prices = []
        for t in range(self.total_time):
            bid_prices_t = []
            for s in range(self.n_states):
                bid_prices_t_s = [0] * self.n_resources
                remain_cap = RM_helper.remain_cap(self.n_states, self.capacities, s)
                for i in range(self.n_resources):
                    var_name = 'r' + '_' + self.resources[i] + '_' + str(t) + '_' + str(remain_cap[i])
                    bid_prices_t_s[i] = max(varsdict[var_name], 0)
                bid_prices_t.append(bid_prices_t_s)
            bid_prices.append(bid_prices_t)
                    
        return bid_prices
    
    def get_bid_prices(self, K):
        """main func: given the number of states to be sampled, first simulate bid-price control policy to sample states, 
        then solve the relaxed LP problem to get the bid-price control for actual sale season. 
        returns the bid prices generated. """
        sampled_states = self.sample_visited_states(K)
        varsdict, varsnames = self.solve_RLP(sampled_states)
        bid_prices_collected = self.collect_bid_prices(varsdict, varsnames)
        return bid_prices_collected
        
p = [['1a', (17.3, 5.8), 1050], ['2a', (45.1, 15.0),950], ['3a', (39.6, 13.2), 699], ['4a', (34.0, 11.3),520],            ['1b', (20, 3.5), 501], ['2b', (63.1, 2.5), 352], ['3b', (22.5, 6.1), 722], ['1ab', (11.5, 2.1), 760],            ['2ab', (24.3, 6.4), 1400]]

resources = ['a', 'b']
capacities = [3,5]

arrival_rate = [0.1, 0.2, 0.04,0.06, 0.01, 0.08, 0.23, 0.11, 0.14]
pros, demands, _ = RM_helper.sort_product_demands(p)
problem = ALP(pros, resources, [demands], capacities, 3, arrival_rate)
problem.get_bid_prices(3)


# In[24]:

def solve_DLP():
    """in step 1: solves a DLP model, with the given remaining capacity, and the current time period; returns bid
    prices for resources. """
    DLP_model = pulp.LpProblem('DLP model', pulp.LpMaximize)
    names = [['a1', 'b1'], ['a2', 'b2']]
    flattern_names = np.array(names).ravel().tolist()
    prices = [[12, 23],[34,45]]
    y = pulp.LpVariable.dict('y_%s', flattern_names, lowBound= 0)
    # objective function
    DLP_model += sum([prices[i][j] * y[names[i][j]] for j in range(2) for i in range(2)])
    # constraints 1, for each resource, the sum of products of consumption py each product and the booking limit of 
    # that product is less than the total capacity of that resource
    constraints = []
    for i in range(2):
        p = [1,2]
        ps = dict(zip(names[i], p))
        print("ps = ", ps)
        c = sum([ps[j] * y[j] for j in names[i]]) <= 20
        constraints.append(c)
        DLP_model += c, "c"+str(i)
    # constraints2, every booking limit should be less than the corresponding demand
    DLP_model.solve()
    print(DLP_model)

    bid_prices = [c.pi for c in constraints]
#         print("bid-prices= ", bid_prices)
    return bid_prices

solve_DLP()


# In[73]:

a = [[1,2], [3,4]]
print(np.array(a).ravel().tolist())
print(a)


# In[2]:

str([1,2])


# In[ ]:



