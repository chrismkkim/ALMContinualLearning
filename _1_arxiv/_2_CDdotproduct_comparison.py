import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
dirpath = '/Users/kimchm/Documents/KimNiNature2024/MatConverted2Python/' 
metadata = pd.read_csv(dirpath + 'EDF10d_info_2026_01_16.csv')
CDdotprod_python = pd.read_csv('CDdotprod_python_deconvolved.txt')

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
idline = np.arange(-0.4,0.8,0.1)
my_CDdotprod = filtered_CDdotprod_python['CDdotproduct']
jh_CDdotprod = metadata['CDdotproduct']
plt.figure(figsize=(4,3.5))
plt.plot(idline, idline, color='gray', linestyle='--')
plt.scatter(jh_CDdotprod, my_CDdotprod, color='k')
plt.xlabel('CD dot product (JH)')
plt.ylabel('CD dot product (Chris)')
plt.tight_layout()
# plt.savefig('figure/summary/CDdotprod_chris_jh.pdf')

