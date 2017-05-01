
# coding: utf-8

# In[5]:

import pandas

import sys
sys.path.append('/Users/jshan/Desktop/RevenueManagement')
from src import RM_exact
from src import RM_approx

def compare_single_static(products, demands, cap_lb, cap_ub, cap_interval):
    """Compare the exact DP model with a heuristic, EMSR-b, for static models of single-resource RM problems."""
    col_titles = ['DP-rev', 'DP-protect', 'Appr-rev', 'Appr-protect']
    capacities = [c for c in range(cap_lb, cap_ub + 1, cap_interval)]
    n_products = len(products)
    
    result = []
    for cap in capacities:
        dp_model = RM_exact.Single_RM_static(products, demands, cap)
        dp_result = dp_model.value_func()
        
        approx_model = RM_approx.Single_EMSR(products, demands, cap)
        approx_result = approx_model.value_func()
        
        result.append([round(dp_result[0][n_products-1][cap], 2), dp_result[1],                        round(approx_result[0][n_products-1][cap], 2), approx_result[1]])
    
    print(pandas.DataFrame(result, capacities, col_titles))

# Examples, ref: example 2.3, 2.4 in "The Theory and Practice of Revenue Management"
# products = [[1, 1050], [2,567], [3, 534], [4,520]]
products=[[1, 1050], [2,950], [3, 699], [4,520]]
demands = [(17.3, 5.8), (45.1, 15.0), (39.6, 13.2), (34.0, 11.3)]
capacity = 100

compare_single_static(products, demands, 80, 150, 10)


# In[ ]:



