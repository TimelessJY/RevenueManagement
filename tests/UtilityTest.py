
# coding: utf-8

# In[10]:

import unittest

import numpy as np

from src import utility


class UtilityTest(unittest.TestCase):

    def test_efficient_sets(self):
        test_prob = np.array([0.3,0.4,0.5, 0.7,0.8,0.9,1])
        test_rev = np.array([240,200,225,380,465,425,505])
        efficient_sets = utility.efficientSets(test_prob, test_rev)
        expected_efficient_sets = np.array([(1, 0.3, 240), (5, 0.8, 465), (7, 1, 505)])
        np.testing.assert_equal(efficient_sets, expected_efficient_sets)


# In[ ]:



