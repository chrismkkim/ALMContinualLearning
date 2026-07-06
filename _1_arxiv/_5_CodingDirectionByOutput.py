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
    CD_dotproduct, CD_sess1_population, CD_sess2_population, \
    proj_1R, proj_1L, proj_2R, proj_2L,\
    decision_1R, decision_1L, decision_2R, decision_2L = functions.compute_CD_delay(data, data_type)
    
    plt.figure(figsize=(6,3))
    plt.subplot(121)
    plt.axhline(0, color='gray', linestyle='--')
    plt.plot(decision_1R)
    plt.plot(decision_1L)
    plt.title('Session 1, CDdotprod ' + str(np.round(CD_dotproduct,decimals=3)))

    plt.subplot(122)
    plt.axhline(0, color='gray', linestyle='--')
    plt.plot(decision_2R)
    plt.plot(decision_2L)
    plt.title('Session 2')
    plt.tight_layout()
    plt.savefig('figure/decision/decision_' + str(ix) + '_' + filename + '.pdf')
    plt.close()
    
    
    plt.figure(figsize=(4,4))
    plt.plot(CD_sess1_population, CD_sess2_population, marker='.', linestyle='')
    plt.xlabel('CD Session1')
    plt.ylabel('CD Session2')
    plt.title('CDdotprod ' + str(np.round(CD_dotproduct,decimals=3)))
    plt.tight_layout()
    plt.savefig('figure/decision/CD_' + str(ix) + '_' + filename + '.pdf')
    plt.close()