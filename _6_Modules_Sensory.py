import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.stats import norm
import copy
import importlib
from utils import functions, functions_xcontext, plot_modules_delay, plot_modules_sensory
from utils import experiment_param


# %%
importlib.reload(functions)
importlib.reload(functions_xcontext)

dirpath = '/Users/kimchm/Documents/KimNiNature2024/MatConverted2Python/' 
figpath = 'figure/neural_dynamics/cross_context/'
datapath = 'data/CDdotproduct/'

par = experiment_param.ExperimentParam(dirpath)

# trial-averaged neural data
data_all, num_og_cells = functions.gen_trialavg_neural_data(dirpath, par)

# contstrain the data to nonoutlier neurons
_data_nonoutlier, nonoutlier = functions.gen_constrain_to_nonoutliers(par, data_all)

# normalize the mean rate
data_nonoutlier = functions.gen_normalize_by_meanrate(par, _data_nonoutlier)

# select top cells: 
#   - set the threshold for variance
thr_var = 0.8
dict_topcells = functions.gen_topcells(thr_var, par, data_nonoutlier)

# find S, D, R neurons
tix_sample = np.arange(4,8)
tix_delay1 = np.arange(8,12)
tix_delay2 = np.arange(12,16)
tix_response = np.arange(16,20)

dict_topcells_1x2 = functions_xcontext.create_dict_topcells_1x2(par.nfile, dict_topcells, tix_delay2)
dict_CDdp, CDdotproduct, CDdp_1x2 = functions_xcontext.create_dict_CDdotprod(par.nfile, dict_topcells, dict_topcells_1x2, tix_delay2, data_nonoutlier)

keep_D2 = True
if keep_D2:    
    '''
    First train the network with this dataset. Only Sensory & Decision neurons, so it's simpler. Move onto the next if needed.
    '''
    #-----------------------------------
    # Keep D2 as one group
    #   * Sensory:  S+D1 \ D2
    #   * Decision: D2
    #-----------------------------------
    (dict_topcells_S,  dict_topcells_D1, 
    dict_topcells_D2, dict_topcells_R) = functions.gen_topcells_SDR(par, dict_topcells, tix_sample, tix_delay1, tix_delay2, tix_response, keep_D2)

    dict_topcells_1x2_S  = functions_xcontext.create_dict_topcells_1x2_SDR(par.nfile, dict_topcells_S)
    dict_topcells_1x2_D1 = functions_xcontext.create_dict_topcells_1x2_SDR(par.nfile, dict_topcells_D1)
    dict_topcells_1x2_D2 = functions_xcontext.create_dict_topcells_1x2_SDR(par.nfile, dict_topcells_D2)
    dict_topcells_1x2_R  = functions_xcontext.create_dict_topcells_1x2_SDR(par.nfile, dict_topcells_R)

    dict_module_activity_S = functions_xcontext.create_dict_module_activity(par.nfile, dict_topcells_1x2_S, data_nonoutlier, par.keys1, par.keys2, par.keys3)
    dict_module_activity_R = functions_xcontext.create_dict_module_activity(par.nfile, dict_topcells_1x2_R, data_nonoutlier, par.keys1, par.keys2, par.keys3)

    dict_within_selectivity_S = functions_xcontext.create_dict_within_selectivity(par.nfile, dict_module_activity_S, tix_sample, par.keys1, par.keys2, par.keys_within_context, par.keys_across_context)
    dict_within_selectivity_R = functions_xcontext.create_dict_within_selectivity(par.nfile, dict_module_activity_R, tix_sample, par.keys1, par.keys2, par.keys_within_context, par.keys_across_context)

    # CD dot product of sampling epoch
    dict_CDdp_S, CDdotproduct_S, CDdotproduct_1x2_S = functions_xcontext.create_dict_CDdotprod_in_epoch(par.nfile, dict_topcells_S, dict_topcells_1x2_S, tix_sample, data_nonoutlier)
    dict_CDdp_err_S = functions_xcontext.create_dict_CDdotprod_err(par.nfile, dict_CDdp_S, CDdotproduct_S)

    # CD dot product of sampling epoch
    dict_CDdp_D2, CDdotproduct_D2, CDdotproduct_1x2_D2 = functions_xcontext.create_dict_CDdotprod_in_epoch(par.nfile, dict_topcells_D2, dict_topcells_1x2_D2, tix_delay2, data_nonoutlier)
    dict_CDdp_err_D2 = functions_xcontext.create_dict_CDdotprod_err(par.nfile, dict_CDdp_D2, CDdotproduct_D2)

elif not keep_D2:
    '''
    Divide D2 into Sensory & Decision only if needed.
    '''    
    #-----------------------------------
    # Divide D2 into two groups
    #   * Sensory:            S+D1 \ D2
    #   * Sensory & Decision: S+D1 & D2 <-- Contributes significantly to the CD dot product of sampling epoch
    #   * Decision only:      D2 \ S+D1
    #-----------------------------------
    (dict_topcells_S,  dict_topcells_D1, 
    dict_topcells_D2S, dict_topcells_D2_S, dict_topcells_R) = functions.gen_topcells_SDR(par, dict_topcells, tix_sample, tix_delay1, tix_delay2, tix_response, keep_D2)
    
#%%


importlib.reload(plot_modules_sensory)

module_plots = plot_modules_sensory.ModulePlots(
        figpath = figpath,
        nfile = par.nfile,
        acts = par.acts,
        data_nonoutlier = data_nonoutlier,
        dict_topcells_S = dict_topcells_S,
        dict_topcells_D2 = dict_topcells_D2,
        dict_topcells_R = dict_topcells_D2,
        dict_topcells_1x2_S = dict_topcells_1x2_S,
        dict_topcells_1x2_D1 = dict_topcells_1x2_D1,
        dict_topcells_1x2_D2 = dict_topcells_1x2_D2,
        dict_topcells_1x2_R = dict_topcells_1x2_R,
        CDdotproduct = CDdotproduct,
        CDdotproduct_S = CDdotproduct_S,
        CDdotproduct_D2 = CDdotproduct_D2,
        CDdotproduct_1x2_S = CDdotproduct_1x2_S,
        CDdotproduct_1x2_D2 = CDdotproduct_1x2_D2,
        dict_within_selectivity_S = dict_within_selectivity_S,
        dict_module_activity_R = dict_module_activity_R,
        tix_sample = tix_sample,
        tix_response = tix_response,
        keys_within_context = par.keys_within_context,
        keys1 = par.keys1,
        keys2 = par.keys2,
)



#%%

# Compare CD dot product of sampling vs delay epochs
module_plots.plot_CDdotproduct_sampling_vs_delay(savefig=False)

# Heatmap of sensory, decision, response neuron activities
module_plots.plot_heatmap_SDR_neurons(savefig=False)

# Cell counts
module_plots.plot_cell_count(savefig=False)
module_plots.plot_cell_count_9groups(savefig=False)
module_plots.plot_cell_count_4groups(savefig=False)

# Sensory neurons: module activity
module_plots.plot_CDdotproduct_vs_module_activity_S(savefig=False)
module_plots.plot_module_activity_S(savefig=False)

# Response neurons: module activity
module_plots.plot_CDdotproduct_vs_module_activity_R(savefig=False)
module_plots.plot_module_activity_R(savefig=False)
    
