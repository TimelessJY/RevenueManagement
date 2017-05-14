
# coding: utf-8

# In[75]:

##############################
###### ADP Methods ###########
##############################

import numpy as np
import scipy.stats
import time

# def ADP_naive(products, resources, demands, capacities, max_time, num_iterations):
class ADP():
    value_functions = []
    protection_levels = []
    incidence_matrix = []
    
    def __init__(self, products, resources, demands, capacities, total_time):
        """Return a framework for a single-resource RM problem."""
        
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
    
    def state_number(self, remain_cap):
        """helper func: converts the given array of remaining capacities into the state number"""
        """e.g. given total capacities [1,2,1], and the remained capacities [0, 2, 1], should return 5"""

        state_num = 0
        capacity_for_others = self.n_states

        for i in range(self.n_resources):
            capacity_for_others /= self.capacities[i] + 1
            state_num += remain_cap[i] * capacity_for_others
        return int(state_num)

    def remain_cap(self, state_number):
        """helper func: reverse of function state_number(), to convert the given state number into remained capacities"""
        """e.g. given total capacities [1,2,3] and state_number 5, should return [0, 2, 1]"""

        if state_number >= self.n_states:
            raise RuntimeError(
                'Error when converting state number to remained capacities; given state number is too large.')

        remain_cap = []
        capacity_for_others = self.n_states

        for i in range(self.n_resources):
            capacity_for_others /= self.capacities[i] + 1
            remain_cap.append(int(state_number // capacity_for_others))
            state_number %= capacity_for_others
        return remain_cap

    def eval_value(t, control, state_num, product_num):
        """helper func: evaluate the value for period t and state x, ref: equation 3.1 in the book"""

        price_vector = [0] * self.n_products
        price_vector[product_num] = self.products[product_num][1]
        value = np.dot(price_vector, control)
        Au = np.dot(self.incidence_matrix, control).tolist()
        if t < self.total_time - 1:
            x_Au = [x_i - Au_i for x_i, Au_i in zip(self.remain_cap(state_num), Au)]
            state_x_Au = self.state_number(x_Au)
            value += self.value_functions[t+1][self.state_number(x_Au)]
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
    
    def naive_method(self, n_iterations):
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
                    cap_vector = self.remain_cap(S_t)
                    incidence_vector = [row[request_product] for row in self.incidence_matrix]
                    diff = [x_i - a_j_i for a_j_i, x_i in zip(incidence_vector, cap_vector)]
                    if all(diff_i >= 0 for diff_i in diff):
                        u_sell = [0] * self.n_products
                        u_sell[request_product] = 1

                        # find the state after selling the product
                        Au = np.dot(self.incidence_matrix, u_sell).tolist()
                        x_Au = [x_i - Au_i for x_i, Au_i in zip(cap_vector, Au)] # updated remaining capacity vector
                        S_after_sell = self.state_number(x_Au)
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
#             print(prev_ests)
        print("Expected revenue at beginning: ", prev_ests[0][-1])
        return prev_ests

start_time = time.time()
# products = [ ['12', 500], ['1', 250], ['2', 250]]
# resources = ['1', '2']
# demands = [[0.4, 0.2, 0.3]]

# capacities=[4,4]



ps = [['a1', 0.02, 200], ['a2', 0.06, 503], ['ab1', 0.08, 400],['ab2', 0.01, 704], ['b1', 0.05, 601],       ['b2', 0.12, 106], ['bc', 0.03, 920],['c1', 0.07, 832]]
products,demands, _ = RM_helper.sort_product_demands(ps)
demands = [demands]
resources = ['a', 'b', 'c']
capacities = [8] * 3

problem = ADP(products, resources, demands, capacities, 10)
vf = problem.naive_method(1000)
# print(vf)
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:



