
# coding: utf-8

# In[17]:

import unittest

import numpy as np

import sys
sys.path.append('/Users/jshan/Desktop/RevenueManagement')
from src import singleResource_DCM


class singleResource_DCM_tests(unittest.TestCase):

    # test data
    
    test_products = [['Y', 800], ['M',500], ['K',450]]
    test_sets = [[0.3, 0, 0], [0, 0.4, 0], [0, 0, 0.5], [0.1, 0.6, 0], [0.3,0,0.5], [0,0.4,0.5], [0.1, 0.4,0.5]]
    test_efficient_sets = [['Y', 0.3, 240.0], ['YK', 0.8, 465.0], ['YMK', 1.0, 505.0]]
#     test_products = [['Y', 0.3, 240], ['M', 0.4, 200], ['K', 0.5, 225], ['YM', 0.7, 380], \
#                               ['YK', 0.8, 465], ['MK', 0.9, 425], ['YMK', 1, 505]]
    test_marginal_values = [780, 624, 520, 445.71, 390,346.67, 312.00, 283.64, 260.00,                                         240,222.86,208,195,183.53,173.33,164.21,156,148.57,141.82,135.65]
    
    def test_efficient_sets(self):
        # ref: test data from section 2.6.2.1 and 2.6.2.5, table 2.9
        sets= [[0.3, 0, 0], [0, 0.4, 0], [0, 0, 0.5], [0.1, 0.6, 0], [0.3,0,0.5], [0,0.4,0.5], [0.1, 0.4,0.5]]
        efficient_sets = singleResource_DCM.efficient_sets(self.test_products, sets)
        expected_efficient_sets = [['Y', 0.3, 240], ['YK', 0.8, 465], ['YMK', 1, 505]]
        np.testing.assert_equal(efficient_sets, expected_efficient_sets)
        
    def test_optimal_set_for_capacity(self):
        # ref: test data from section 2.6.2.5, table 2.10
        effi_sets = [['Y', 0.3, 240], ['YK', 0.8, 465], ['YMK', 1, 505]]
        optimal_sets = singleResource_DCM.optimal_set_for_capacity(effi_sets, self.test_marginal_values)
        expected_optimal_sets = np.append(['Y' for _ in range(3)], ['YK' for _ in range(9)])
        expected_optimal_sets = np.append(expected_optimal_sets, ['YMK' for _ in range(8)])
        np.testing.assert_equal(optimal_sets, expected_optimal_sets)
        
    def test_optimal_protection_levels(self):
        # ref: test data from section 2.6.2.5, table 2.10
        effi_sets = [['Y', 0.3, 240], ['YK', 0.8, 465], ['YMK', 1, 505]]
        optimal_protection_levels = singleResource_DCM.optimal_protection_levels(effi_sets, self.test_marginal_values)
        expected_optimal_protection_levels = [3, 12, 20]
        np.testing.assert_equal(optimal_protection_levels, expected_optimal_protection_levels)
        
a = singleResource_DCM_tests()
suite = unittest.TestLoader().loadTestsFromModule(a)
unittest.TextTestRunner().run(suite)

