
# coding: utf-8

# In[13]:

import unittest

import numpy as np

import sys
sys.path.append('/Users/jshan/Desktop/RevenueManagement')
from src import network_DAVN


class network_DAVN_tests(unittest.TestCase):

    # test data
    test_prob = np.array([0.3,0.4,0.5, 0.7,0.8,0.9,1])
    test_rev = np.array([240,200,225,380,465,425,505])
    
    def test_calc_displacement_adjusted_revenue(self):
        # ref: test data from section 3.4.5.3, table 3.3
        test_products = np.array([('AB', 100), ('CD', 100), ('ABC', 1000), ('BCD', 1000)])
        test_resources = np.array(['AB', 'BC', 'CD'])
        test_static_bid_prices = np.array([10, 30, 15])
        disp_adjusted_revenue = network_DAVN.calc_displacement_adjusted_revenue(test_products, test_resources,                                                                                 test_static_bid_prices)
        expected_disp_adjusted_revenue = [[(955, 'ABC'), (955, 'BCD'), (70, 'AB'), (0, 'CD')],                                          [(975, 'ABC'), (975, 'BCD'), (90, 'AB'), (85, 'CD')],                                           [(960, 'ABC'), (960, 'BCD'), (70, 'CD'), (0, 'AB')]]
        np.testing.assert_equal(disp_adjusted_revenue, expected_disp_adjusted_revenue)
        
a = network_DAVN_tests()
suite = unittest.TestLoader().loadTestsFromModule(a)
unittest.TextTestRunner().run(suite)


# In[ ]:



