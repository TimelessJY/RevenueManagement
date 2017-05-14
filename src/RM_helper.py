
# coding: utf-8

# In[1]:

def sort_product_demands(products):
    """
    sorts the given products of form:[product_name, demand, revenue], into a list of [product_name, revenue]
    and a list of demands, according to the descending order of the revenue of each product
    """
    n_products = len(products)
    demands = []
    demands_with_name = []
    products.sort(key = lambda tup: tup[2], reverse=True)
    demands = [p[1] for p in products]
    demands_with_name = [[p[0], p[1]] for p in products]
    products = [[p[0], p[2]] for p in products]
    return (products, demands, demands_with_name)

def marginal_value_check(value_func):
    """checks whether the marginal values in computed value functions satisfy the proposition 2.21"""
    dim = len(value_func)
    for j in range(dim):
        delta_V= [x-y for x, y in zip(value_func[j][1:], value_func[j])]
        print("delta = ", delta_V)
        if any(delta_V[i] < delta_V[i+1] for i in range(len(delta_V) - 1)):
            print("error type 1")
        if j < (dim -1):
            delta_V_next = [x-y for x, y in zip(value_func[j+1][1:], value_func[j+1])]
            print("delta_next = ", delta_V_next)
            if any(delta_V[i] > delta_V_next[i] for i in range(len(delta_V))):
                print("error type 2")

def calc_incidence_matrix(products, resources):
    """constructs the incidence matrix, indicating which product uses which resources, 
        e.g. incidence_matrix[i][j] = 1 if product j uses resource i
        size n_resources * n_products"""
    
    n_products = len(products)
    n_resources = len(resources)

    incidence_matrix = [[0] * n_products for _ in range(n_resources)] 

    for i in range(n_resources):
        for j in range(n_products):
            if resources[i] in products[j][0]: # test if product j uses resource i
                incidence_matrix[i][j] = 1
    return incidence_matrix


# In[ ]:



