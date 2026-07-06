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
CD_sess1 = np.empty(nfile,dtype=object)
CD_sess2 = np.empty(nfile,dtype=object)
CD_dotproduct_all = np.zeros(nfile)

for ix in range(nfile):
    
    print(ix)
    
    # #--- sorted by relearn speed ---#
    # fidx     = sorted_indices_by_relearn_speed[ix]    
    # filename = filtered_CDdotprod_python['ID'][fidx]
    
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

    CDdotprod = np.round(CD_dotproduct, decimals=2)
    Relearn   = np.round(relearn_speed[fidx], decimals=2)    

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
        
    # create plots for spatial footprint
    plot_sf_exclusive_1L = functions.create_plot_sf('Purples',sf_exclusive_1L)
    plot_sf_exclusive_2L = functions.create_plot_sf('Greens', sf_exclusive_2L)    
    plot_sf_overlap_1L_2L = functions.create_plot_sf('Set2', sf_overlap_1L_2L)

    plot_sf_exclusive_1R = functions.create_plot_sf('Purples',sf_exclusive_1R)
    plot_sf_exclusive_2R = functions.create_plot_sf('Greens', sf_exclusive_2R)    
    plot_sf_overlap_1R_2R = functions.create_plot_sf('Set2', sf_overlap_1R_2R)
    
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(plot_sf_exclusive_1L, aspect='auto')
    plt.imshow(plot_sf_exclusive_2L, aspect='auto')
    plt.imshow(plot_sf_overlap_1L_2L, aspect='auto')
    plt.title('Sess1 + Sess2 ' + '(CD dotprod ' + str(CDdotprod) + ')')    
    
    plt.subplot(132)
    plt.imshow(plot_sf_exclusive_1L, aspect='auto')
    plt.imshow(plot_sf_overlap_1L_2L, aspect='auto')
    plt.title('Sess 1 (Lick Left)')
    
    plt.subplot(133)
    plt.imshow(plot_sf_exclusive_2L, aspect='auto')
    plt.imshow(plot_sf_overlap_1L_2L, aspect='auto')
    plt.title('Sess 2 (Lick Left)')    
    plt.tight_layout()    
    plt.savefig('figure/spatial_footprint/lickleft/left_' + str(ix) + '_' + filename + '.pdf')
    plt.close()
            

    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(plot_sf_exclusive_1R, aspect='auto')
    plt.imshow(plot_sf_exclusive_2R, aspect='auto')
    plt.imshow(plot_sf_overlap_1R_2R, aspect='auto')
    plt.title('Sess1 + Sess2 ' + '(CD dotprod ' + str(CDdotprod) + ')')    
    
    plt.subplot(132)
    plt.imshow(plot_sf_overlap_1R_2R, aspect='auto')
    plt.imshow(plot_sf_exclusive_1R, aspect='auto')
    plt.title('Sess 1 (Lick Right)')
    
    plt.subplot(133)
    plt.imshow(plot_sf_overlap_1R_2R, aspect='auto')
    plt.imshow(plot_sf_exclusive_2R, aspect='auto')
    plt.title('Sess 2 (Lick Right)')
    plt.tight_layout()    
    plt.savefig('figure/spatial_footprint/lickright/right_' + str(ix) + '_' + filename + '.pdf')
    plt.close()
            
                        
    neuron_sorted_by_CD_sess1 = np.argsort(CD_sess1_population)    
    plt.figure(figsize=(6,5))
    plt.plot(CD_sess2_population[neuron_sorted_by_CD_sess1], c='darkorange', label='Sess 2')
    plt.plot(CD_sess1_population[neuron_sorted_by_CD_sess1], c='limegreen', label='Sess 1')
    plt.xlabel('neuron index (ordered by CD of Sess 1)')
    plt.ylabel('CD')
    plt.title('CDdotprod: ' + str(CDdotprod) + ', Relearn time: ' + str(Relearn))    
    plt.legend()
    plt.tight_layout()
    plt.savefig('figure/spatial_footprint/CD/CD_' + str(ix) + '_' + filename + '.pdf')
    plt.close()

    CD_sess1[ix] = CD_sess1_population[neuron_sorted_by_CD_sess1]
    CD_sess2[ix] = CD_sess2_population[neuron_sorted_by_CD_sess1]
    CD_dotproduct_all[ix] = CD_dotproduct
    
x=1


cell_idx_normalized = np.empty(nfile,dtype=object)
for ix in range(nfile):
    ncell                   = CD_sess1[ix].shape[0]
    cell_idx_normalized[ix] = np.arange(ncell) / ncell

nstep = 200
inc   = 1/nstep
CD_sess2_aligned = np.zeros((nfile,nstep))
CD_sess1_aligned = np.zeros((nfile,nstep))
for file in range(nfile):
    for ix in range(nstep):
        Lidx = np.where(cell_idx_normalized[file] < inc * ix)[0]
        Ridx = np.where(cell_idx_normalized[file] < inc * (ix+1))[0]
        if Lidx.shape[0] == 0:
            Lidx = 0
        else:
            Lidx = Lidx[-1]
        if Ridx.shape[0] == 0:
            Ridx = 0
        else:
            Ridx = Ridx[-1]
        CD_sess1_aligned[file,ix] = np.mean(CD_sess1[file][Lidx:Ridx])
        CD_sess2_aligned[file,ix] = np.mean(CD_sess2[file][Lidx:Ridx])

plt.figure()
vmax = 0.05
xmin, xmax = 0, 1
ymin, ymax = CD_dotproduct_all[0], CD_dotproduct_all[-1]
plt.imshow(np.flipud(CD_sess2_aligned), cmap='bwr_r', vmin=-vmax, vmax=vmax, extent=[xmin, xmax, ymin, ymax], aspect='auto')
plt.title('CD of Session 2')
plt.xlabel('normalized neuron index (ordered by CD of Session 1)')
plt.ylabel('CD dot product')
plt.colorbar()
plt.tight_layout()
plt.savefig('figure/spatial_footprint/frac_overlap/CD_sess2.pdf', dpi=300)



plt.figure()
vmax = 0.05
xmin, xmax = 0, 1
ymin, ymax = CD_dotproduct_all[0], CD_dotproduct_all[-1]
plt.imshow(np.flipud(CD_sess1_aligned), cmap='bwr_r', vmin=-vmax, vmax=vmax, extent=[xmin, xmax, ymin, ymax], aspect='auto')
plt.title('CD of Session 1')
plt.xlabel('normalized neuron index (ordered)')
plt.ylabel('CD dot product')
plt.colorbar()
plt.tight_layout()
plt.savefig('figure/spatial_footprint/frac_overlap/CD_sess1.pdf', dpi=300)




    
    