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
from scipy.io import loadmat
import h5py

# f = h5py.File('data.jld', 'r')

#%%

dirpath = '/Users/kimchm/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/NIH/research/_published/balanced network/s1alm_data/processed/'
data_pyr_R = h5py.File(dirpath + 'rates_Pyr_lickright.jld', 'r')['Pyr'][:][:100,:].T
data_pyr_L = h5py.File(dirpath + 'rates_Pyr_lickleft.jld', 'r')['Pyr'][:][:100,:].T
data_fs_R = h5py.File(dirpath + 'rates_FS_lickright.jld', 'r')['FS'][:][:100,:].T
data_fs_L = h5py.File(dirpath + 'rates_FS_lickleft.jld', 'r')['FS'][:][:100,:].T


#%%

peak_pyr_R = np.max(data_pyr_R,axis=1)
peak_pyr_L = np.max(data_pyr_L,axis=1)
peak_fs_R = np.max(data_fs_R,axis=1)
peak_fs_L = np.max(data_fs_L,axis=1)

cells_sorted_pyr_R = np.argsort(peak_pyr_R)
cells_sorted_pyr_L = np.argsort(peak_pyr_L)
cells_sorted_fs_R  = np.argsort(peak_fs_R)
cells_sorted_fs_L  = np.argsort(peak_fs_L)

data_pyr_R_sorted = data_pyr_R[cells_sorted_pyr_R,:]
data_pyr_L_sorted = data_pyr_L[cells_sorted_pyr_L,:]
data_fs_R_sorted = data_fs_R[cells_sorted_fs_R,:]
data_fs_L_sorted = data_fs_L[cells_sorted_fs_L,:]

#%%
'''
Seuqnetial activity not observed in Finkelstein et al 2021 Nat. Neuro. 
'''

ncell_pyr_R, nstep = data_pyr_R.shape
ncell_pyr_L, nstep = data_pyr_L.shape
ncell_fs_R, nstep = data_fs_R.shape
ncell_fs_L, nstep = data_fs_L.shape
tvec  = 0.02*np.arange(nstep)

plt.figure(figsize=(6,6))
plt.subplot(221)
plt.imshow(data_pyr_R_sorted, cmap='jet', aspect='auto', vmin=0, vmax=20, extent=[tvec[0],tvec[-1],0,ncell_pyr_R-1])
plt.title('Pyr, Lick Right')
plt.subplot(222)
plt.imshow(data_pyr_L_sorted, cmap='jet', aspect='auto', vmin=0, vmax=20, extent=[tvec[0],tvec[-1],0,ncell_pyr_L-1])
plt.title('Pyr, Lick Left')
plt.subplot(223)
plt.imshow(data_fs_R_sorted, cmap='jet', aspect='auto', vmin=0, vmax=20, extent=[tvec[0],tvec[-1],0,ncell_fs_R-1])
plt.title('FS, Lick Right')
plt.xlabel('time (s)')
plt.ylabel('neuron index')
plt.subplot(224)
plt.imshow(data_fs_L_sorted, cmap='jet', aspect='auto', vmin=0, vmax=20, extent=[tvec[0],tvec[-1],0,ncell_fs_L-1])
plt.title('FS, Lick Left')
plt.tight_layout()
plt.savefig('figure/neural_dynamics/finkelstein/finkelstein_psth.pdf')
