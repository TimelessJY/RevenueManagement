
# coding: utf-8

# In[49]:

import unittest

import numpy as np

import sys
sys.path.append('/Users/jshan/Desktop/RevenueManagement')
from src import singleResource_DCM


class singleResource_DCM_tests(unittest.TestCase):

    # test data
    test_prob = np.array([0.3,0.4,0.5, 0.7,0.8,0.9,1])
    test_rev = np.array([240,200,225,380,465,425,505])
    
    def test_efficientSets(self):
        # ref: test data from section 2.6.2.1 and 2.6.2.5, table 2.9
        efficient_sets = singleResource_DCM.efficient_sets(self.test_prob, self.test_rev)
        expected_efficient_sets = np.array([(1, 0.3, 240), (5, 0.8, 465), (7, 1, 505)])
        np.testing.assert_equal(efficient_sets, expected_efficient_sets)
        
    def test_optimalSetForCapacity(self):
        # ref: test data from section 2.6.2.5, table 2.10
        efficient_sets = singleResource_DCM.efficient_sets(self.test_prob, self.test_rev)
        test_marginal_values = np.array([780, 624, 520, 445.71, 390,346.67, 312.00, 283.64, 260.00,240,222.86,208,195,183.53,173.33,164.21,156,148.57,141.82,135.65])
        optimal_sets = singleResource_DCM.optimal_set_for_capacity(efficient_sets, test_marginal_values)
        expected_optimal_sets = np.append([1 for _ in range(3)], [5 for _ in range(9)])
        expected_optimal_sets = np.append(expected_optimal_sets, [7 for _ in range(8)])
        np.testing.assert_equal(optimal_sets, expected_optimal_sets)
        
a = singleResource_DCM_tests()
suite = unittest.TestLoader().loadTestsFromModule(a)
unittest.TextTestRunner().run(suite)


# In[ ]:




# In[ ]:



