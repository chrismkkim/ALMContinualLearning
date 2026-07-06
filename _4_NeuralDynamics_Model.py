#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import copy
import importlib
from utils import plot_stimulus 
from utils import plot_neuralstate_cd
from utils import plot_neuralstate_eucldist
from utils import functions 
from utils import plot_learned_activity

#%%
lam = -0.8
g = 2.0
dim = 4
W = np.array([[lam, g, 0,   0],
              [0, lam, g,   0],
              [0,   0, lam, g],
              [0,   0, 0,   lam]])

nstep = 200
x = np.zeros((nstep,dim))
x[0] = np.array([0,0,0.5,0.5])

dt = 0.1
tau = 4
for ti in range(nstep-1):
    x[ti+1] = (1-dt/tau)*x[ti] + dt/tau*W@x[ti]

plt.figure()
plt.plot(x[:,0])    
plt.plot(x[:,1])    
plt.plot(x[:,2])
plt.plot(x[:,3])
plt.tight_layout()
