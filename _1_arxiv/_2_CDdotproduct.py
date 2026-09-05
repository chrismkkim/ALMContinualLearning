#%%
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import h5py 
import os
from utils import functions

dirpath = '/Users/kimchm/Documents/KimNiNature2024/MatConverted2Python/' 
# sessName = 'BAYLORJH035_FOV1_2021_12_09_2022_02_15.npy'
allfiles = [f for f in os.listdir(dirpath) if f.startswith('BAYLOR')]
allfiles.sort()

fileCDdotproduct = 'CDdotprod_python_deconvolved.txt'
f = open(fileCDdotproduct, 'w')
f.write('ID,CDdotproduct\n')

#%%
for fileName in allfiles:
        
    filepath = dirpath + fileName
    data = np.load(filepath, allow_pickle=True)

    data_type  = 'deconvolved' # dFF0 or deconvolved
    CD_dotproduct = functions.compute_CD_dotproduct_compare_to_JH(data, data_type)    
    
    print(fileName[:-4],', CD dot product: ', CD_dotproduct)
    f.write(f'{fileName[:-4]}, {CD_dotproduct}\n')
    
f.close()

