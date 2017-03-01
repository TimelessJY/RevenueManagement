
# coding: utf-8

# In[20]:

import unittest

import numpy as np

import sys
sys.path.append('/Users/jshan/Desktop/RevenueManagement')
from src import network_DAVN


class network_DAVN_tests(unittest.TestCase):

    # test data
    test_products = np.array([['AB', 0.5, 100], ['CD', 0.5, 100], ['ABC', 0.5, 1000], ['BCD',0.5, 1000]])
    test_resources = np.array(['AB', 'BC', 'CD'])
    test_mean_demands = np.array([('AB', 10.1), ('CD', 5.3), ('ABC',8), ('BCD', 9.2)])
    test_disp_adjusted_revenue = [[(955, 'ABC'), (955, 'BCD'), (70, 'AB'), (0, 'CD')],                                          [(975, 'ABC'), (975, 'BCD'), (90, 'AB'), (85, 'CD')],                                           [(960, 'ABC'), (960, 'BCD'), (70, 'CD'), (0, 'AB')]]
    
    def test_calc_displacement_adjusted_revenue(self):
        # ref: test data from section 3.4.5.3, table 3.3
        test_static_bid_prices = np.array([10, 30, 15])
        disp_adjusted_revenue = network_DAVN.calc_displacement_adjusted_revenue(self.test_products,                                                                                 self.test_resources,                                                                                 test_static_bid_prices)
        expected_disp_adjusted_revenue = self.test_disp_adjusted_revenue
        np.testing.assert_equal(disp_adjusted_revenue, expected_disp_adjusted_revenue)
        
    def test_calc_squared_derivation_of_revenue(self):
        resource_index = 0
        partition_start = 0
        partition_end = 2 # i.e. contains the first three products
        sqrd_deriv_rev = network_DAVN.calc_squared_deviation_of_revenue(resource_index, partition_start,                                                                             partition_end, self.test_mean_demands,                                                                            self.test_disp_adjusted_revenue)
        
        expected_sqrd_deriv_revenue = 4983950.43956
        np.testing.assert_almost_equal(sqrd_deriv_rev, expected_sqrd_deriv_revenue, decimal=5)
        
    def test_clustering(self):
        n_class = 3
        partitions = network_DAVN.clustering(self.test_products, self.test_resources, self.test_disp_adjusted_revenue,                                             n_class, self.test_mean_demands)
        expected_partitions = [[["ABC", "BCD"], ["AB"], ["CD"]] for i in range(2)]
        expected_partitions.append([['ABC', 'BCD'], ['CD'], ['AB']])
        
        np.testing.assert_equal(partitions, expected_partitions)
        
a = network_DAVN_tests()
suite = unittest.TestLoader().loadTestsFromModule(a)
unittest.TextTestRunner().run(suite)


# In[ ]:



