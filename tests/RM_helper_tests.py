
# coding: utf-8

# In[22]:

import unittest
import numpy as np

import sys
sys.path.append('../')
from src import RM_helper


# In[23]:


class RM_helper_tests(unittest.TestCase):

    def test_state_index(self):
        test_states = 12
        test_capacity = [1,2,1]
        test_remain_cap = [0, 2, 1]
        expected_state_number = 5
        state_number = RM_helper.state_index(test_states, test_capacity, test_remain_cap)
        np.testing.assert_equal(state_number, expected_state_number)
        
    def test_remain_cap(self):
        test_states = 12
        test_capacity = [1,2,1]
        test_state_number = 3
        expected_remain_cap = [0, 1, 1]
        remain_cap = RM_helper.remain_cap(test_states, test_capacity, test_state_number)
        np.testing.assert_equal(remain_cap, expected_remain_cap)
        
    
a = RM_helper_tests()
suite = unittest.TestLoader().loadTestsFromModule(a)
unittest.TextTestRunner().run(suite)


# In[ ]:



