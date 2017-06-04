
# coding: utf-8

# In[4]:

import random
import numpy as np


# In[79]:

class model():
    """Demand model for RM network problems, provides relative functionality.
            
    Given:
    ----------
    arrival_rates: 2D np array
        contains levels of arrival rates, low, medium, high, for products.
    model_type: integer
        1: time homogeneous arrival rates. 
        2: medium arrival rates for the first T/2 time periods; 
                with probability p, high arrival_rates for the last T/2 time periods; with probability 1-p, low. 
    total_time: integer
        the horizon over which the demand model is applied on. 
    p: float
        in model 2, the probability that the rates level after half time changes to high, (1-p) for changing to low.
    """
    
    arrival_rates = {}
    rates_levels = []
    
    def __init__(self, arrival_rates, total_time, model_type, p=0):
        if model_type > 2:
            raise ValueError('Unrecognized demand model.')
        if not arrival_rates:
            raise ValueError('No arrival rates data given.')
            
        self.model_type = model_type
        self.total_time = total_time
        self.change_time = int(total_time / 2)
        self.p = p
        
        self.extract_arrival_rates(arrival_rates)
        self.set_up_rates_levels()
        
    def extract_arrival_rates(self, arrival_rates):
        """helper func: extract out arrival rates into three different levels. """
        self.arrival_rates['low'] = arrival_rates[0]
        
        if self.model_type == 2:
            if len(arrival_rates) < 3:
                raise ValueError('Missing arrival rates data for the chosen model type.')
                
            self.arrival_rates['med'] = arrival_rates[1]
            self.arrival_rates['hi'] = arrival_rates[2]
            
    def set_up_rates_levels(self):
        """helper func: decides the demand level at each time period, to be used over the whole process. """
        if self.model_type == 1:
            self.rates_levels = ['low'] * self.total_time
        else:
            # model 2, medium for the first half of time periods
            self.rates_levels = ['med'] * self.change_time
            # with probability p, changes to high level afterwards
            new_level = 'low'
            rand = np.random.binomial(1, self.p)
            if rand == 1:
                new_level = 'hi'
            self.rates_levels += [new_level] * (self.total_time - self.change_time)      
        print("levels = ", self.rates_levels)
        
    def current_arrival_rates(self, t):
        """ returns a list of arrival rates for products at the given time period. """
        if t >= self.total_time:
            raise ValueError("Not valid time period.")
        return self.arrival_rates[self.rates_levels[t]]
    
    def current_mean_demands(self, curr_time):
        """ returns the mean demands of products at the current time. """
        sums = [0] * len(self.arrival_rates['low'])
        for t in range(curr_time, self.total_time):
            sums = [sum(x) for x in zip(sums, self.current_arrival_rates(t))]
            
        return sums
        
# rates = [[0.1, 0.2, 0.3],[0.4, 0.5, 0.6], [0.7, 0.8,0.9]]
# dm = model(rates, 10, 2, 0.5)
# dm.current_arrival_rates(1)
# dm.current_mean_demands(0)
# dm.set_up_rates_levels()


# In[ ]:



