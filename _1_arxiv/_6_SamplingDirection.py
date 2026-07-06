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

tsample   = 1.57
tdelay    = 2.87
tresponse = 4.17
dt        = 1/6
ntimestep = 47
tvec      = dt * np.arange(ntimestep)
tix_sample   = np.where(tvec > tsample)[0][0]
tix_delay    = np.where(tvec > tdelay)[0][0]
tix_response = np.where(tvec > tresponse)[0][0]
lickright    = 0
lickleft     = 1


nfile = len(filtered_CDdotprod_python)
CD_delay_dotproduct_all  = np.zeros(nfile)
CD_sample_dotproduct_all = np.zeros(nfile)
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
    opto_sess1 = data['deconvolved'][0]
    opto_sess2 = data['deconvolved'][1]
    ncell      = opto_sess1.shape[1]

    # compute projected activity
    data_type = 'deconvolved'
    CD_sample_dotproduct, CD_sample_1_population, CD_sample_2_population,\
    _, _, _, _ = functions.compute_CD_sample(data, data_type)
    
    CD_context_dotproduct, CD_context_A_population, CD_context_P_population = functions.compute_CD_context(data, data_type)
    
    CD_delay_dotproduct, CD_sess1_population, CD_sess2_population, \
    _, _, _, _,\
    _, _, _, _ = functions.compute_CD_projection(data, data_type)
    
    cell_sorted_by_CD_sample_1 = np.argsort(CD_sample_1_population)
    
    plt.figure(figsize=(4,4))
    plt.plot(CD_sample_2_population[cell_sorted_by_CD_sample_1], color='darkorange')
    plt.plot(CD_sample_1_population[cell_sorted_by_CD_sample_1], color='limegreen')
    plt.xlabel('neuron')
    plt.ylabel('CD Sample')
    plt.title('CD delay ' + str(np.round(CD_delay_dotproduct,decimals=2)) + ', CD sample ' + str(np.round(CD_sample_dotproduct,decimals=2)))
    plt.tight_layout()
    plt.savefig('figure/sample/CD_sample_' + str(ix) + '_' + filename + '.pdf')
    plt.close()
    
    CD_delay_dotproduct_all[ix]  = CD_delay_dotproduct
    CD_sample_dotproduct_all[ix] = CD_sample_dotproduct
    
plt.figure(figsize=(3,2.5))
plt.plot(CD_delay_dotproduct_all, CD_sample_dotproduct_all, marker='o', c='k', linestyle='')
plt.xlabel('CD delay dot prod')
plt.ylabel('CD sample dot prod')
plt.xlim([-1.05,1])
plt.ylim([0.0,1])
plt.tight_layout()
plt.savefig('figure/sample/CDdotprod_delay_vs_sample.pdf')

x=1
