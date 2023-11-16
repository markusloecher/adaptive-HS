import numpy as np
from scipy.stats import logistic
import pandas as pd

def simulate_data(N: int, p: int):
    allXs = []
    #All features are discrete, with the jth feature containing j+1 distinct values
    for i in range(p):
        allXs.append([np.random.randint(0, i+2, N)])#i+2 
    #We randomly select a set S of 5 features from the first ten 
    rlvFtrs = np.array([0] * p)
    # rlvFtrs[np.random.randint(1, 11, 5)] = 1
    rlvFtrs[np.random.choice(range(0, 10), 5, replace=False)] = 1
    #position = np.where(rlvFtrs == 1)[0]

    Xrlv = np.array([0]*N) 

    for k in (np.where(rlvFtrs == 1)[0]):
        Xrlv = Xrlv + allXs[k]/(k+1)

    y = np.array([]) 

    for i in range(N):
        y = np.append(y, np.random.binomial(n = 1, p = logistic.cdf(2*Xrlv[0][i]/5 - 1), size = 1))

    x_train = pd.DataFrame(np.concatenate(allXs).T.reshape(N, p, order='F')) 

    return x_train, y, rlvFtrs
    
