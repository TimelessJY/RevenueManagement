
# coding: utf-8

# In[62]:

import itertools
import sys
sys.path.append('.')
import RM_helper
import random


# In[63]:

PRICE_LIMITS = [150, 250]

def generate_network(n_spokes, demand_model):
    """Generates a network using the given number of spokes, and the demand model, with random prices, and arrival rates
    of itineraries. Currently only supports 1 fare class per itinerary. """
    resources = [] # records flight legs names
    itineraries = [] # records names and (revenue, arrival rate) pairs of fare classes of itineraries
    hub_name = 'HUB'
    spoke_names = []
    
    # produce flight legs (single-direction)
    for i in range(n_spokes):
        spoke_name = chr(65 + i)
        spoke_names.append(spoke_name)
        resources.append(spoke_name + '-' + hub_name)
    
    # produce single-leg itineraries
    single_legs = resources[:]
    single_legs += reverse_itinerary(resources)
    
    # produce double-leg itineraries
    double_legs = []
    two_spoke_pairs = list(itertools.combinations(''.join(spoke_names), 2))
    for pair in two_spoke_pairs:
        iti = '-'.join([pair[0], hub_name, pair[1]])
        double_legs.append(iti)
    
    double_legs += reverse_itinerary(double_legs)
    
    # produce double-leg itineraries, between the hub and the same spoke, i.e. round-trips between spoke and hub
    round_legs = []
    for spoke in spoke_names:
        round_legs.append('-'.join([spoke, hub_name, spoke]))
    
    # aggregate all itineraries, and randomly generate the price and arrival rate
    itineraries += single_legs + double_legs + round_legs
    f = len(itineraries)
    demands = generate_random_arrival_rate(f, 0.3, demand_model)
    for i in range(f):
        full_iti = [itineraries[i]]
        arrival_rate = demands[i]
        price = generate_random_price(itineraries[i])
        full_iti.append([(price, arrival_rate)])
        itineraries[i] = full_iti
    return resources, itineraries
    
def reverse_itinerary(itinerary_names):
    """helper func: given a list of itinerary names, generate a list of reversed itineraries for them. """
    reversed_itineraries = []
    for itinerary in itinerary_names:
        nodes = itinerary.split('-')
        nodes.reverse()
        reversed_name = '-'.join(nodes)
        reversed_itineraries.append(reversed_name)
    return reversed_itineraries

def generate_random_arrival_rate(n, total_sum, demand_model):
    """helper func: generate n random values in [0,1] and normalize them so that their sum is equal to total_sum."""
    if demand_model == 1:
        # constant arrival rates for classes over time
        M = sys.maxsize
        x = random.sample(range(M), n - 1)
        x.insert(0, 0)
        x.append(M)
        x.sort()
        y = [x[i + 1] - x[i] for i in range(n)]
        unit_simplex = [y_i / (1/total_sum * M) for y_i in y]
        return unit_simplex
    else:
        print("TODO:implement model 2")

def generate_random_price(itinerary_name):
    """helper func: generate a random price for the given itinerary, limit depends on how many flight legs it uses."""
    leg_num = itinerary_name.count('-')
    price = random.randint(50, PRICE_LIMITS[leg_num-1])
    return price

# resources, itineraries = generate_network(3, 1)
# RM_helper.extract_legs_info(itineraries, resources)


# In[ ]:



