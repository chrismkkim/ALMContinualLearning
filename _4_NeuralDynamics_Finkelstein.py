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

dirpath = '/Users/kimchm/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/NIH/research/ALM/finkelstein/neural data/'

time_sample = -3.0
time_delay  = -2.0
time_go     = 0.0
time_end    = 1.5
wid         = 0.02 #20ms
ntime       = int((time_end - time_sample) / wid)

tix_delay = int((time_delay - time_sample)/wid)
tix_go    = int((time_go    - time_sample)/wid)

data_pyr_R = h5py.File(dirpath + 'rates_Pyr_lickright.jld', 'r')['Pyr'][:][:ntime,:].T
data_pyr_L = h5py.File(dirpath + 'rates_Pyr_lickleft.jld', 'r')['Pyr'][:][:ntime,:].T
data_fs_R = h5py.File(dirpath + 'rates_FS_lickright.jld', 'r')['FS'][:][:ntime,:].T
data_fs_L = h5py.File(dirpath + 'rates_FS_lickleft.jld', 'r')['FS'][:][:ntime,:].T


#%%

peak_pyr_R = np.argmax(data_pyr_R,axis=1)
peak_pyr_L = np.argmax(data_pyr_L,axis=1)
peak_fs_R  = np.argmax(data_fs_R,axis=1)
peak_fs_L  = np.argmax(data_fs_L,axis=1)

cells_sorted_pyr_R = np.argsort(peak_pyr_R)
cells_sorted_pyr_L = np.argsort(peak_pyr_L)
cells_sorted_fs_R  = np.argsort(peak_fs_R)
cells_sorted_fs_L  = np.argsort(peak_fs_L)

data_pyr_R_sorted = data_pyr_R[cells_sorted_pyr_R,:]
data_pyr_L_sorted = data_pyr_L[cells_sorted_pyr_L,:]
data_fs_R_sorted  = data_fs_R[cells_sorted_fs_R,:]
data_fs_L_sorted  = data_fs_L[cells_sorted_fs_L,:]

#%%
'''
Seuqnetial activity not observed in Finkelstein et al 2021 Nat. Neuro. 
'''

ncell_pyr_R, nstep = data_pyr_R.shape
ncell_pyr_L, nstep = data_pyr_L.shape
ncell_fs_R, nstep = data_fs_R.shape
ncell_fs_L, nstep = data_fs_L.shape
tvec  = 0.02*np.arange(nstep)

plt.figure(figsize=(4,12))
plt.imshow(data_pyr_R_sorted, cmap='jet', aspect='auto', vmin=0, vmax=20, extent=[tvec[0],tvec[-1],0,ncell_pyr_R-1])
plt.tight_layout()
plt.savefig('figure/neural_dynamics/finkelstein_daie/finkelstein_pyr_R.pdf')

plt.figure(figsize=(4,12))
plt.imshow(data_pyr_L_sorted, cmap='jet', aspect='auto', vmin=0, vmax=20, extent=[tvec[0],tvec[-1],0,ncell_pyr_R-1])
plt.tight_layout()
plt.savefig('figure/neural_dynamics/finkelstein_daie/finkelstein_pyr_L.pdf')

plt.figure(figsize=(4,12))
plt.imshow(data_fs_R_sorted, cmap='jet', aspect='auto', vmin=0, vmax=20, extent=[tvec[0],tvec[-1],0,ncell_pyr_R-1])
plt.tight_layout()
plt.savefig('figure/neural_dynamics/finkelstein_daie/finkelstein_fs_R.pdf')

plt.figure(figsize=(4,12))
plt.imshow(data_fs_L_sorted, cmap='jet', aspect='auto', vmin=0, vmax=20, extent=[tvec[0],tvec[-1],0,ncell_pyr_R-1])
plt.tight_layout()
plt.savefig('figure/neural_dynamics/finkelstein_daie/finkelstein_fs_L.pdf')

#%%
plt.figure(figsize=(6,15))
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
plt.savefig('figure/neural_dynamics/finkelstein_daie/finkelstein_psth.pdf')


#%%
thr_var = 0.8
ncells, ntime = data_pyr_R.shape
data_pyr_R_sq = data_pyr_R**2
data_pyr_R_norm = data_pyr_R_sq / np.sum(data_pyr_R_sq,axis=0)
cells_sorted = np.argsort(data_pyr_R_norm,axis=0)[::-1]
data_pyr_R_sorted = np.take_along_axis(data_pyr_R_norm, cells_sorted, axis=0)
data_pyr_R_cumsum = np.cumsum(data_pyr_R_sorted,axis=0)
idx_cross = np.argmax(data_pyr_R_cumsum >= thr_var, axis=0)
topcells_at_t = {i:[] for i in range(ntime)}
for i in range(ntime):
    # top cells at t
    topcells_at_t[i] = cells_sorted[:idx_cross[i],i]
    
frac_topcells_at_t = np.zeros(ntime)
for i in range(ntime):
    frac_topcells_at_t[i] = len(topcells_at_t[i]) / ncells
active_cells_intensity = np.zeros((ntime,ntime))
for i in range(ntime):
    active_cells_at_t = np.mean(data_pyr_R[topcells_at_t[i]],axis=0)
    active_cells_intensity[i] = active_cells_at_t / active_cells_at_t[i]

dt = tvec[1]-tvec[0]
plt.figure(figsize=(2.4,1.8))
plt.imshow(active_cells_intensity,cmap='jet',aspect='auto',extent=[time_sample-dt/2,time_end+dt/2,time_end+dt/2,time_sample-dt/2])
plt.axvline(time_delay,color='gray',linestyle='--',lw=0.5)
plt.axvline(time_go,color='gray',linestyle='--',lw=0.5)
plt.axhline(time_delay,color='gray',linestyle='--',lw=0.5)
plt.axhline(time_go,color='gray',linestyle='--',lw=0.5)
plt.xlabel('time to go cue')
plt.ylabel('ref time of\nactive neurons')
plt.xticks([-2,-1,0,1])
plt.yticks([-2,-1,0,1])
plt.colorbar()
plt.tight_layout()
plt.savefig('figure/neural_dynamics/finkelstein_daie/finkelstein/finkelstein_active_cells_heatmap.pdf',dpi=600)
    
    
# plt.figure(figsize=(3,3))
# for ti in range(10):
#     plt.plot(data_pyr_R_cumsum[:,ti])
# plt.tight_layout()


# plt.figure(figsize=(1.5,1.3))
# plt.plot(frac_topcells_at_t)
# plt.tight_layout()




#%%
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl

nref, ntime = active_cells_intensity.shape
_t = np.arange(ntime)

fig = plt.figure(figsize=(2.5, 2))
ax = fig.add_subplot(111, projection='3d')

zmax = np.nanmax(active_cells_intensity)
zmin = np.nanmin(active_cells_intensity)

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
    

ax.view_init(elev=30, azim=15)
ax.set_box_aspect((3, 3, 1))


plt.tight_layout()

plt.savefig('figure/neural_dynamics/finkelstein_daie/finkelstein/finkelstein_active_cells_3d.pdf')

