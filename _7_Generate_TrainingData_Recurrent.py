#%%
import os
import numpy as np
import copy
import importlib
from scipy.ndimage import gaussian_filter1d
from utils import functions, functions_xcontext, functions_training
from utils import experiment_param
import matplotlib.pyplot as plt

# %%
importlib.reload(functions)
importlib.reload(functions_xcontext)
importlib.reload(functions_training)

dirpath = '/Users/kimchm/Documents/KimNiNature2024/MatConverted2Python/' 
figpath = 'figure/neural_dynamics/cross_context/'
trainingdatapath = 'data/training/recurrent/training_data/'

par = experiment_param.ExperimentParam(dirpath)

# ndata = 11
# for idata in range(ndata):
idata = 0

# trial-averaged neural data
data_all, num_og_cells = functions.gen_trialavg_neural_data(dirpath, par)

# contstrain the data to nonoutlier neurons
_data_nonoutlier, nonoutlier = functions.gen_constrain_to_nonoutliers(par, data_all)

# normalize the mean rate
data_nonoutlier = functions.gen_normalize_by_meanrate(par, _data_nonoutlier)

# select top cells: 
#   - set the threshold for variance
thr_var = 0.99
dict_topcells = functions.gen_topcells(thr_var, par, data_nonoutlier)

# Combine all cross-context groups into unique neuron indices for each session.
tix_full = np.arange(4,20)
dict_topcells_1x2 = functions_xcontext.create_dict_topcells_1x2(par.nfile, dict_topcells, tix_full)
dict_topcells_1x2_combined = {}
for fx in range(par.nfile):
    dict_topcells_1x2_combined[fx] = np.unique(np.concatenate([dict_topcells_1x2[fx][cond1][cond2]
    for cond1 in ['P1+A1-','P1+A1+','P1-A1+'] for cond2 in ['P2+A2-','P2+A2+','P2-A2+']
    ])).astype(int)

# CD dot product
tix_latedelay = np.arange(12,16)
dict_topcells_1x2_delay = functions_xcontext.create_dict_topcells_1x2(par.nfile, dict_topcells, tix_latedelay)
dict_CDdp, CDdotproduct, CDdp_1x2 = functions_xcontext.create_dict_CDdotprod(par.nfile, dict_topcells, dict_topcells_1x2_delay, tix_latedelay, data_nonoutlier)

# dict_topcells_combined = {}
# for fx in range(par.nfile):
#     dict_topcells_combined[fx] = np.unique(np.concatenate([dict_topcells[cond]['topcells'][fx] for cond in par.acts]))

# for fx in range(par.nfile):
#     print('sess ', fx,'---')
#     print(dict_topcells_combined[fx].shape[0])
#     print(dict_topcells_1x2_combined[fx].shape[0]/dict_topcells_combined[fx].shape[0])
    
data_topcells = {fx:{} for fx in range(par.nfile)}
for fx in range(par.nfile):
    data_topcells[fx] = {cond : data_nonoutlier[fx][cond][dict_topcells_1x2_combined[fx]] for cond in par.acts}

# Smooth each neuron's time course independently before saving.
# Optional temporal smoothing; Gaussian width is sigma in time bins.
smooth_data = True
smoothing_sigma = 1.0
if smooth_data:
    data_topcells = {
        fx: {
            cond: gaussian_filter1d(
                np.asarray(activity, dtype=float),
                sigma=smoothing_sigma,
                axis=1,
                mode='reflect',
            )
            for cond, activity in session_activity.items()
        }
        for fx, session_activity in data_topcells.items()
    }

np.save(trainingdatapath + f'activity{idata}.npy', data_topcells)
np.save(trainingdatapath + f'CDdotproduct{idata}.npy', CDdotproduct)

# Fraction of nonoutlier neurons selected in each session.
neuron_fraction = np.full(par.nfile, np.nan)
for fx in range(par.nfile):
    n_selected = dict_topcells_1x2_combined[fx].size
    n_total = data_nonoutlier[fx]['P1'].shape[0]
    if n_total > 0:
        neuron_fraction[fx] = n_selected / n_total
np.save(trainingdatapath + f'neuron_fraction{idata}.npy', neuron_fraction)





#%%

# fx=26
# cond='A1'
# data_fx_cond = data_topcells[fx][cond]
# sorted = np.argsort(np.argmax(data_fx_cond,axis=1))

# plt.figure()
# plt.subplot(121)
# plt.imshow(data_topcells[fx]['P1'][sorted],cmap='jet',aspect='auto',vmin=0,vmax=0.2)
# plt.subplot(122)
# plt.imshow(data_topcells[fx]['A1'][sorted],cmap='jet',aspect='auto',vmin=0,vmax=0.2)
# plt.tight_layout()




# # find S, D, R neurons
# tix_sample = np.arange(4,8)
# tix_delay1 = np.arange(8,12)
# tix_delay2 = np.arange(12,16)
# tix_response = np.arange(16,20)

# dict_topcells_1x2 = functions_xcontext.create_dict_topcells_1x2(par.nfile, dict_topcells, tix_delay2)
# dict_CDdp, CDdotproduct, CDdp_1x2 = functions_xcontext.create_dict_CDdotprod(par.nfile, dict_topcells, dict_topcells_1x2, tix_delay2, data_nonoutlier)

# '''
# First train the network with this dataset. Only Sensory & Decision neurons, so it's simpler. Move onto the next if needed.
# '''
# #-----------------------------------
# # Keep D2 as one group
# #   * Sensory:  S+D1 \ D2
# #   * Decision: D2
# #-----------------------------------
# (dict_topcells_S,  dict_topcells_D1, 
# dict_topcells_D2, dict_topcells_R) = functions.gen_topcells_SDR(par, dict_topcells, tix_sample, tix_delay1, tix_delay2, tix_response, keep_D2=True)

# dict_topcells_1x2_S  = functions_xcontext.create_dict_topcells_1x2_SDR(par.nfile, dict_topcells_S)
# dict_topcells_1x2_D1 = functions_xcontext.create_dict_topcells_1x2_SDR(par.nfile, dict_topcells_D1)
# dict_topcells_1x2_D2 = functions_xcontext.create_dict_topcells_1x2_SDR(par.nfile, dict_topcells_D2)
# dict_topcells_1x2_R  = functions_xcontext.create_dict_topcells_1x2_SDR(par.nfile, dict_topcells_R)

# dict_topcells_S_4groups = functions_training.create_dict_topcells_4groups(par, dict_topcells_1x2_S)
# dict_topcells_D_4groups = functions_training.create_dict_topcells_4groups(par, dict_topcells_1x2_D2)
# dict_topcells_R_4groups = functions_training.create_dict_topcells_4groups(par, dict_topcells_1x2_R)

# dict_activity_S_4groups = functions_training.create_activity_4groups(par, dict_topcells_S_4groups, data_nonoutlier)
# dict_activity_D_4groups = functions_training.create_activity_4groups(par, dict_topcells_D_4groups, data_nonoutlier)
# dict_activity_R_4groups = functions_training.create_activity_4groups(par, dict_topcells_R_4groups, data_nonoutlier)

# #---------------------------------#
# #--- THIS IS THE TRAINING DATA ---#
# #---------------------------------#
# '''
# Use the 'nonoutlier' indices to get the indices of 'S','D','R' neurons in the original data

# dict_topcells_4groups:
#     * 'nonoutlier': indices of nonoutlier neurons in the original data
#     * 'S'         : indices of S neurons in the nonoutliers
#     * 'D'         : indices of D neurons in the nonoutliers
#     * 'R'         : indices of R neurons in the nonoutliers
# '''    
# dict_topcells_4groups = functions_training.combine_topcells_SDR(par, dict_topcells_S_4groups, dict_topcells_D_4groups, dict_topcells_R_4groups, nonoutlier, num_og_cells)
# dict_activity_4groups = functions_training.combine_activity_SDR(par, dict_activity_S_4groups, dict_activity_D_4groups, dict_activity_R_4groups)

# np.save(trainingdatapath + f'cells{idata}.npy', dict_topcells_4groups)
# np.save(trainingdatapath + f'activity{idata}.npy', dict_activity_4groups)



# cells = np.load(trainingdatapath + 'cells.npy', allow_pickle=True).item()
# activities = np.load(trainingdatapath + 'activity.npy', allow_pickle=True).item()








# #--------- below is optional -----------#
# # check if the number of S,D,R neurons matches
# #   - confirmed that the numbers match
# for fx in range(par.nfile):
#     cnt = 0
#     for neutype in ['S','D','R']:
#         for ig in range(4):
#            cnt += len(dict_topcells_4groups[fx][neutype][f'Group{ig}'])
#     assert cnt == dict_activity_4groups[fx]['P1'].shape[0]


# %%
