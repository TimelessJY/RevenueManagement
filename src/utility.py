
# coding: utf-8

# In[1]:

import warnings

# Identifies a list of efficient sets, given information about all possible sets
# Qs: a list of probability of purchase for each set
# Rs: a list of expected revenue for each set
# returns a list of efficient sets, in the form of (setIndex, Q, R)
def efficientSets(Qs, Rs):
    if len(Rs) != len(Qs):
        warnings.warn("Wrong size of input in efficientSets()")

    sets = min(len(Qs), len(Rs))    # number of all sets

    effiSets = list()   # stores output
    prevEffiSet = -1    # store the previous efficient set, start with empty set
    prevQ = 0   # store the choice probability of the previous efficient set
    prevR = 0   # store the revenue of the previous efficient set

    while True:
        nextEffiSet = -1     # store the next efficient set
        maxMarginalRevenueRatio = 0
        hasPotentialSet = False
        for i in range(sets):
            if i == prevEffiSet:
                continue
            q = float(Qs[i])
            r = Rs[i]
            if q >= prevQ and r >= prevR:
                hasPotentialSet = True
                marginalRevenueRatio = (r - prevR) / (q - prevQ)
                if marginalRevenueRatio > maxMarginalRevenueRatio:
                    nextEffiSet = i
                    maxMarginalRevenueRatio = marginalRevenueRatio
        if hasPotentialSet is False:
            # stop if there isn't any potential efficient sets
            break
        elif nextEffiSet >= 0:    # if find a new efficient set
            prevEffiSet = nextEffiSet
            prevQ = Qs[nextEffiSet]
            prevR = Rs[nextEffiSet]
            effiSets.append((nextEffiSet + 1, prevQ, prevR))


    return effiSets


# In[ ]:
