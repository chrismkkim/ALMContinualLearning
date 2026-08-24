#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
# from sklearn.linear_model import LinearRegression
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
import mat73


#%%
dirpath = '/Users/kimchm/Documents/DaieNatNeuro2021/'
f = mat73.loadmat(dirpath + 'Daie_et_al_2020_targeted_photostim.mat', 'r')

#%%

choice = 'right'
# choice = 'left'

if choice == 'right':
    choice_neural = 'R'
    choice_behavior = 'CR'
if choice == 'left':
    choice_neural = 'L'
    choice_behavior = 'CL'
    
data = f['data']
data_R = data[choice_neural]
sess = 3 # experiment session
pg_nostim = 0 # 0 = non-photostim

#%%
data_R_sess = data_R[sess]
data_R_sess_nostim = data_R_sess[pg_nostim] # time x neuron x trial

CR = data[choice_behavior]
CR_sess = CR[sess]
CR_sess_nostim = CR_sess[pg_nostim]

data_CR_sess_nostim = data_R_sess_nostim[:,:,CR_sess_nostim]
data_CR_sess_avg = np.transpose(np.mean(data_CR_sess_nostim,axis=2))
ncells, ntime = data_CR_sess_avg.shape

#%%
epoch = data['epochs']
tsample_start, tsample_end = epoch[0]['sample']
tcue = epoch[0]['cue'].item()

dt = data['dt_si'][0].item()
t = dt * np.arange(ntime)
tgo = t - tcue

tix_sample_start = np.where(t > tsample_start)[0][0]
tix_sample_end   = np.where(t > tsample_end)[0][0]
tix_cue          = np.where(t > tcue)[0][0]

# activity of active cells
thr_var = 0.8
ncells, ntime = data_CR_sess_avg.shape
data_CR_sess_avg_var = (data_CR_sess_avg - np.mean(data_CR_sess_avg,axis=0))**2 # variance
data_CR_sess_norm = data_CR_sess_avg_var / np.sum(data_CR_sess_avg_var,axis=0)

cells_sorted = np.argsort(data_CR_sess_norm,axis=0)[::-1]
data_CR_sess_sorted = np.take_along_axis(data_CR_sess_norm, cells_sorted, axis=0)
data_CR_sess_cumsum = np.cumsum(data_CR_sess_sorted,axis=0)
idx_cross = np.argmax(data_CR_sess_cumsum >= thr_var, axis=0)
topcells_at_t = {i:[] for i in range(ntime)}
for i in range(ntime):
    # top cells at t
    topcells_at_t[i] = cells_sorted[:idx_cross[i],i]
    
frac_topcells_at_t = np.zeros(ntime)
for i in range(ntime):
    frac_topcells_at_t[i] = len(topcells_at_t[i]) / ncells
active_cells_intensity = np.zeros((ntime,ntime))
for i in range(ntime):
    active_cells_at_t = np.mean(data_CR_sess_avg[topcells_at_t[i]],axis=0)
    active_cells_intensity[i] = active_cells_at_t 
    # active_cells_intensity[i] = active_cells_at_t / np.abs(active_cells_at_t[i])
    
#%%


plt.figure(figsize=(2.4,1.8))
extent = [tgo[0],tgo[-1],tgo[-1],tgo[0]]
plt.imshow(active_cells_intensity,cmap='jet',aspect='auto', vmin=-0.2,vmax=0.4, extent=extent)
plt.axvline(tgo[tix_sample_start], c='gray', linestyle='--', lw=0.5)
plt.axvline(tgo[tix_sample_end], c='gray', linestyle='--', lw=0.5)
plt.axvline(tgo[tix_cue], c='gray', linestyle='--', lw=0.5)
plt.axhline(tgo[tix_sample_start], c='gray', linestyle='--', lw=0.5)
plt.axhline(tgo[tix_sample_end], c='gray', linestyle='--', lw=0.5)
plt.axhline(tgo[tix_cue], c='gray', linestyle='--', lw=0.5)
plt.xlabel('time to go cue')
plt.ylabel('ref time of\nactive neurons')
plt.colorbar()
plt.xticks([-3,0])
plt.yticks([-3,0])
plt.tight_layout()

if choice == 'right':
    plt.savefig('figure/neural_dynamics/finkelstein_daie/daie/daie_active_cells_heatmap_R.pdf', dpi=600)
if choice == 'left':
    plt.savefig('figure/neural_dynamics/finkelstein_daie/daie/daie_active_cells_heatmap_L.pdf', dpi=600)



#%%
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl

nref, ntime = active_cells_intensity.shape
_t = np.arange(ntime)

fig = plt.figure(figsize=(2.5, 2))
ax = fig.add_subplot(111, projection='3d')

zmin = -0.2 #np.nanmin(active_cells_intensity)
zmax = 0.4 #np.nanmax(active_cells_intensity)
# zmin = -1 #np.nanmin(active_cells_intensity)
# zmax = 3 #np.nanmax(active_cells_intensity)

cmap = plt.cm.jet
norm = mpl.colors.Normalize(vmin=zmin, vmax=zmax)

for ci in range(nref):

    z = active_cells_intensity[ci]
    x = np.full(ntime, ci)
    y = _t

    # Consecutive 3D points forming line segments
    points = np.column_stack((x, y, z))
    segments = np.stack((points[:-1], points[1:]), axis=1)

    # Value used to color each segment
    z_segment = (z[:-1] + z[1:]) / 2

    lc = Line3DCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=0.5,
        alpha=0.8
    )
    lc.set_array(z_segment)

    ax.add_collection3d(lc)

# Line3DCollection does not automatically set axis limits
ax.set_xlim(0, nref - 1)
ax.set_ylim(0, ntime - 1)
ax.set_zlim(zmin, zmax)
# ax.set_zlim(np.nanmin(active_cells_intensity), np.nanmax(active_cells_intensity))

# Colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    shrink=0.6,
    pad=0.02
)

ax.grid(False)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])


# Show only the xy pane
ax.zaxis.pane.fill = True
ax.zaxis.pane.set_facecolor((0.7, 0.7, 0.7, 0.15))
ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))

# Hide the xz and yz panes
for axis in [ax.xaxis, ax.yaxis]:
    axis.pane.fill = False
    axis.pane.set_edgecolor((1, 1, 1, 0))

# Hide all axis lines
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.line.set_color((1, 1, 1, 0))
    
# ax.view_init(elev=30, azim=15)
# ax.set_box_aspect((3, 1.5, 1))

ax.view_init(elev=20, azim=-15)
ax.set_box_aspect((3, 1.5, 0.5))

plt.tight_layout()

if choice == 'right':
    plt.savefig('figure/neural_dynamics/finkelstein_daie/daie/daie_active_cells_3d_R.pdf')
if choice == 'left':
    plt.savefig('figure/neural_dynamics/finkelstein_daie/daie/daie_active_cells_3d_L.pdf')


# %%
