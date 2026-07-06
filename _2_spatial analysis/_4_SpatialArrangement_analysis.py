import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import functions

dirpath = '/Users/kimchm/Documents/KimNiNature2024/MatConverted2Python/' 
metadata = pd.read_csv(dirpath + 'EDF10d_info_2026_01_16.csv')
CDdotprod_python = pd.read_csv('CDdotprod_python_deconvolved.txt')


# Compare CD dot products of CK and JH
metadata['Merged_ID'] = (
    metadata['Mouse ID'].str.strip("'") + '_' + 
    metadata['FOV'].str.strip("'") + '_' + 
    metadata['Session1'].str.strip("'").str.replace("-", "_") + '_' + 
    metadata['Session2'].str.strip("'").str.replace("-", "_")
)
cols = ['Merged_ID'] + [c for c in metadata.columns if c != 'Merged_ID']
metadata = metadata[cols]
metadata = metadata.sort_values(by='Merged_ID')
metadata = metadata.reset_index(drop=True)

filtered_CDdotprod_python = CDdotprod_python[CDdotprod_python['ID'].isin(metadata['Merged_ID'])].reset_index(drop=True)

is_aligned = metadata['Merged_ID'].equals(filtered_CDdotprod_python['ID'])
print('is aligned: ', is_aligned)

# filtered_CDdotprod_python.keys()
ck_CDdotprod = filtered_CDdotprod_python['CDdotproduct']
jh_CDdotprod = metadata['CDdotproduct']
# idline = np.arange(-0.4,0.8,0.1)
# plt.figure()
# plt.plot(idline, idline, color='gray', linestyle='--')
# plt.scatter(jh_CDdotprod, ck_CDdotprod)
# plt.xlabel('JH CD dot product')
# plt.ylabel('CK CD dot product')
# plt.tight_layout()
# plt.show()

def create_plot_sf(cmap_type, sf, vlim):
    sf_sum = np.sum(sf, axis=2)
    vmin, vmax = -vlim, vlim    
    norm = mcolors.Normalize(vmin=vmin,vmax=vmax)
    cmap = plt.get_cmap(cmap_type)
    plot_sf = cmap(norm(sf_sum))
    plot_sf[sf_sum==0, 3] = 0
    return plot_sf
    
# sort by relearns speed
colname_relearn_speed = 'Relative trials to reach\n75% performance'
relearn_speed = metadata[colname_relearn_speed]
sorted_indices_by_relearn_speed = np.argsort(relearn_speed)

# sort by CD dot product
colname_CDdotproduct = 'CDdotproduct'
CDdotproduct = filtered_CDdotprod_python[colname_CDdotproduct]
sorted_indices_by_CDdotproduct = np.argsort(CDdotproduct)
# filtered_CDdotprod_sorted = filtered_CDdotprod_python.sort_values(by='CDdotproduct').reset_index(drop=True)

nfile = len(filtered_CDdotprod_python)
frac_overlap_cells_L = np.zeros(nfile)
frac_overlap_cells_R = np.zeros(nfile)
CDdotprod_sorted    = np.zeros(nfile)
relearn_time_sorted = np.zeros(nfile)


for ix in range(nfile):
    
    print(ix)    
    
    #--- sorted by CD dot product ---#
    fidx = sorted_indices_by_CDdotproduct[ix]
    filename = filtered_CDdotprod_python['ID'][fidx]
    
    filepath = dirpath + filename + '.npy'
    data = np.load(filepath, allow_pickle=True)
    sf    = data['spatial_footprints']
    ncell = data['spatial_footprints'].shape[2]

    # compute CD1, CD2, and the dot product
    data_type = 'deconvolved'
    CD_dotproduct, CD_sess1_population, CD_sess2_population = functions.compute_CD_dotproduct(data, data_type)

    # spatial footprint weighted by CD
    sf_binary = np.zeros((512,512,ncell))
    for cell in range(ncell):
        cix = sf[:,:,cell] > 0
        sf_binary[cix,cell] += 1
        
    CDdotprod = np.round(CD_dotproduct, decimals=3)
    Relearn   = np.round(relearn_speed[fidx], decimals=3)    

    # select the top cells
    frac_cell = 1.0
    ncell_1R, ncell_1L = np.sum(CD_sess1_population>0), np.sum(CD_sess1_population<0)
    ncell_2R, ncell_2L = np.sum(CD_sess2_population>0), np.sum(CD_sess2_population<0)
    ncell_selected_1R, ncell_selected_1L = int(ncell_1R * frac_cell), int(ncell_1L * frac_cell)
    ncell_selected_2R, ncell_selected_2L = int(ncell_2R * frac_cell), int(ncell_2L * frac_cell)
    
    # all cells: 1R, 2R, 1L, 2L
    all_cells_1R, all_cells_2R, all_cells_1L, all_cells_2L = functions.find_all_cells(CD_sess1_population, CD_sess2_population)    
    # # top cells: 1R, 2R, 1L, 2L
    # top_cells_1L, top_cells_2L, top_cells_1R, top_cells_2R = functions.find_top_cells(CD_sess1_population, CD_sess2_population, ncell_selected_1L, ncell_selected_2L, ncell_selected_1R, ncell_selected_2R)
    
    # spatial footprint
    overlap_cells_1L_2L, exclusive_cells_1L, exclusive_cells_2L, \
    sf_overlap_1L_2L, sf_exclusive_1L, sf_exclusive_2L = functions.sf_lick_left(all_cells_1L, all_cells_2L, sf_binary)

    overlap_cells_1R_2R, exclusive_cells_1R, exclusive_cells_2R,\
    sf_overlap_1R_2R, sf_exclusive_1R, sf_exclusive_2R = functions.sf_lick_right(all_cells_1R, all_cells_2R, sf_binary)            
    
    # fraction of overlap cells
    frac_overlap_cells_L[ix] = overlap_cells_1L_2L.shape[0] / (overlap_cells_1L_2L.shape[0] + exclusive_cells_1L.shape[0] + exclusive_cells_2L.shape[0])
    frac_overlap_cells_R[ix] = overlap_cells_1R_2R.shape[0] / (overlap_cells_1R_2R.shape[0] + exclusive_cells_1R.shape[0] + exclusive_cells_2R.shape[0])
    CDdotprod_sorted[ix] = CD_dotproduct
    relearn_time_sorted[ix] = relearn_speed[fidx]
    
    
plt.figure(figsize=(4,3.5))
plt.plot(frac_overlap_cells_L, CDdotprod_sorted, label='Lick Left', marker='o', linestyle='', c='r')    
plt.plot(frac_overlap_cells_R, CDdotprod_sorted, label='Lick Right', marker='o', linestyle='', c='b')    
plt.xlabel('fraction of stable choice-selective cells')
plt.ylabel('CD dot product')
plt.legend()
plt.tight_layout()
plt.savefig('figure/spatial_footprint/frac_overlap/frac_stable_CDdotprod.pdf')


plt.figure(figsize=(4,3.5))
plt.plot(frac_overlap_cells_L, relearn_time_sorted, label='Lick Left', marker='o', linestyle='', c='r')    
plt.plot(frac_overlap_cells_R, relearn_time_sorted, label='Lick Right', marker='o', linestyle='', c='b')    
plt.xlabel('fraction of stable choice-selective cells')
plt.ylabel('relearn time')
plt.legend()
plt.tight_layout()
plt.savefig('figure/spatial_footprint/frac_overlap/frac_stable_Relearn.pdf')


plt.figure(figsize=(4,3.5))
plt.plot(CDdotprod_sorted, relearn_time_sorted, marker='o', linestyle='', c='k')    
plt.xlabel('CD dot product')
plt.ylabel('relearn time')
plt.tight_layout()
plt.savefig('figure/spatial_footprint/frac_overlap/CDdotprod_relearn.pdf')

x=1