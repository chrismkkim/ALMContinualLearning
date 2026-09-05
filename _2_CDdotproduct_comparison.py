#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import functions

#%%
dirpath = '/Users/kimchm/Documents/KimNiNature2024/MatConverted2Python/' 

# Load JH data and create new ID with the same format as the file names
metadata = pd.read_csv(dirpath + 'EDF10d_info_2026_01_16.csv')
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

# Compute CD dot product and save in a dataframe
allfiles = [f for f in os.listdir(dirpath) if f.startswith('BAYLOR')]
allfiles.sort()
ID = []
CDdotproduct = []
for fileName in allfiles:        
    filepath = dirpath + fileName
    data = np.load(filepath, allow_pickle=True)
    data_type = 'deconvolved'
    CD_dotproduct = functions.compute_CD_dotproduct_compare_to_JH(data, data_type)
    print(fileName[:-4], ', CD dot product: ', CD_dotproduct)
    ID.append(fileName[:-4])
    CDdotproduct.append(CD_dotproduct)
df_CDdotproduct = pd.DataFrame({'ID': ID,'CDdotproduct': CDdotproduct})

# Remove data not reported in the paper's CD dot product
#   - 58 data in total 
#   - 33 data reported in the paper's CD dot product plot.
filtered_CDdotprod_python = df_CDdotproduct[df_CDdotproduct['ID'].isin(metadata['Merged_ID'])].reset_index(drop=True)

is_aligned = metadata['Merged_ID'].equals(filtered_CDdotprod_python['ID'])
print('is aligned: ', is_aligned)

# Compare my and JH's CD dot product
idline = np.arange(-0.4,0.8,0.1)
my_CDdotprod = filtered_CDdotprod_python['CDdotproduct']
jh_CDdotprod = metadata['CDdotproduct']
plt.figure(figsize=(4,3.5))
plt.plot(idline, idline, color='gray', linestyle='--')
plt.scatter(jh_CDdotprod, my_CDdotprod, color='k')
plt.xlabel('CD dot product (JH)')
plt.ylabel('CD dot product (Chris)')
plt.tight_layout()
plt.savefig('figure/summary/CDdotprod_chris_jh.pdf')

