import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import os
import h5py, pickle 

savepath = '/Users/kimchm/Documents/KimNiNature2024/MatConverted2Python/' 
dirpath = '/Users/kimchm/Documents/KimNiNature2024/MatFilesFromJaeHyun/source/' 
allfiles = [f for f in os.listdir(dirpath) if f.startswith('BAYLOR')]

# fileName = 'BAYLORJH035_FOV1_2021_12_09_2022_02_15.mat'
for fileName in allfiles:      
    print(fileName)
    
    filepath = dirpath + fileName
    matdata = h5py.File(filepath, 'r')
    matdata.keys()
    keys = ['dFF0','deconvolved','spatial_footprints','total_roi_id']
    pydata = dict.fromkeys(keys)

    '''
    Convert dFF0
    '''
    nsess = 2
    dFF0  = np.empty(nsess,dtype=object)
    dff0  = matdata['dFF0']
    for sess in range(nsess):
        #-------------------------------
        #          SESSION
        #-------------------------------
        # reference for the session
        ref_session   = dff0[0,sess]
        # data for the session
        dff0_sess     = matdata[ref_session]
        #-------------------------------
        #    SESSION - (NFOV, NCELL)
        # nfov, ncell = dff0_sess.shape
        #-------------------------------    
        # each session has data of (nfov, ncell)
        nfov, ncell   = dff0_sess.shape
        dFF0_FOV_CELL = np.empty((nfov,ncell),dtype=object)

        for fov, cell in product(range(nfov), range(ncell)):
            #---------------------------------------------------
            #    SESSION - (NFOV, NCELL) - (NTRIAL, NTIME)
            # ntrial, ntime = matdata[ref_sess_fov_cell].shape
            #---------------------------------------------------
            # each (nfov, ncell) has data of (ntrial, ntime)
            ref_sess_fov_cell       = dff0_sess[fov,cell]
            dFF0_FOV_CELL[fov,cell] = matdata[ref_sess_fov_cell][:]
        dFF0[sess] = dFF0_FOV_CELL
        
    '''
    Convert deconvolved
    '''
    DECON = np.empty(nsess,dtype=object)
    decon = matdata['deconvolved']
    for sess in range(nsess):
        #-------------------------------
        #          SESSION
        #-------------------------------
        # reference for the session
        ref_session   = decon[0,sess]
        # data for the session
        decon_sess     = matdata[ref_session]
        #-------------------------------
        #    SESSION - (NFOV, NCELL)
        # nfov, ncell = dff0_sess.shape
        #-------------------------------    
        # each session has data of (nfov, ncell)
        nfov, ncell    = decon_sess.shape
        DECON_FOV_CELL = np.empty((nfov,ncell),dtype=object)

        for fov, cell in product(range(nfov), range(ncell)):
            #---------------------------------------------------
            #    SESSION - (NFOV, NCELL) - (NTRIAL, NTIME)
            # ntrial, ntime = matdata[ref_sess_fov_cell].shape
            #---------------------------------------------------
            # each (nfov, ncell) has data of (ntrial, ntime)
            ref_sess_fov_cell       = decon_sess[fov,cell]
            DECON_FOV_CELL[fov,cell] = matdata[ref_sess_fov_cell][:]
        DECON[sess] = DECON_FOV_CELL
        
    '''
    Save as .npy
    '''    
    pydata['dFF0'] = dFF0
    pydata['deconvolved'] = DECON
    pydata['spatial_footprints'] = matdata['spatial_footprints'][:]
    pydata['total_roi_id'] = matdata['total_roi_id'][:]
    with open(savepath + fileName[:-4] + '.npy', 'wb') as f:
        pickle.dump(pydata, f, protocol=4) # file size can be large > 4GB, so use protocol = 4

        