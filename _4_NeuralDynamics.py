#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import copy
import importlib
from utils import functions 
from utils import plot_fit_twosided_exp
from utils import plot_sequential_activity
from utils import plot_active_cells_dynamics

# %%
dirpath = '/Users/kimchm/Documents/KimNiNature2024/MatConverted2Python/' 
metadata = pd.read_csv(dirpath + 'EDF10d_info_2026_01_16.csv')

# Compare CD dot products of CK and JH
metadata['Merged_ID'] = (
    metadata['Mouse ID'].str.strip("'") + '_' + 
    metadata['FOV'].str.strip("'") + '_' + 
    metadata['Session1'].str.strip("'").str.replace("-", "_") + '_' + 
    metadata['Session2'].str.strip("'").str.replace("-", "_")
)
cols = ['Merged_ID'] + [c for c in metadata.columns if c != 'Merged_ID']
metadata = metadata[cols]
    
# merged id sorted by relearn speed
merged_ID     = metadata['Merged_ID']
# mouse group sorted by relearn speed
mouse_group   = metadata['Mouse Group']
# sort by relearns speed
colname_relearn_speed = 'Relative trials to reach\n75% performance'
relearn_time = metadata[colname_relearn_speed]
sorted_indices_by_relearn_speed = np.argsort(relearn_time)

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

nfile  = len(metadata)
CD_dotproduct  = np.zeros(nfile)
save_CD_dotproduct = True

# %%
P1_set1, P1_set2 = {}, {}
A1_set1, A1_set2 = {}, {}
A2_set1, A2_set2 = {}, {}
P2_set1, P2_set2 = {}, {}
sf={}


for fx in range(nfile):
    print(fx)
    
    fx_sorted = sorted_indices_by_relearn_speed[fx]

    #--- sorted by relearn speed ---#
    filepath   = dirpath + merged_ID[fx_sorted] + '.npy'
    data       = np.load(filepath, allow_pickle=True)
    data_type  = 'deconvolved'
    # data_type  = 'dFF0'
    opto_sess1 = data[data_type][0]
    opto_sess2 = data[data_type][1]

    # split data into two sets of trials
    opto_1R_set1, opto_1L_set1, opto_2R_set1, opto_2L_set1, \
    opto_1R_set2, opto_1L_set2, opto_2R_set2, opto_2L_set2 = functions.split_trials(data, data_type)

    # neural activity
    # trial set 1
    opto_1R_avg_set1 = np.mean(opto_1R_set1,axis=2) # neurons x time x trials
    opto_1L_avg_set1 = np.mean(opto_1L_set1,axis=2) # neurons x time x trials
    opto_2R_avg_set1 = np.mean(opto_2R_set1,axis=2) # neurons x time x trials
    opto_2L_avg_set1 = np.mean(opto_2L_set1,axis=2) # neurons x time x trials
    # trial set 2
    opto_1R_avg_set2 = np.mean(opto_1R_set2,axis=2) # neurons x time x trials
    opto_1L_avg_set2 = np.mean(opto_1L_set2,axis=2) # neurons x time x trials
    opto_2R_avg_set2 = np.mean(opto_2R_set2,axis=2) # neurons x time x trials
    opto_2L_avg_set2 = np.mean(opto_2L_set2,axis=2) # neurons x time x trials
    
    # neural activity - outlier neurons removed
    P1_set1[fx] = opto_1R_avg_set1
    A1_set1[fx] = opto_1L_avg_set1
    P2_set1[fx] = opto_2L_avg_set1
    A2_set1[fx] = opto_2R_avg_set1
    P1_set2[fx] = opto_1R_avg_set2
    A1_set2[fx] = opto_1L_avg_set2
    P2_set2[fx] = opto_2L_avg_set2
    A2_set2[fx] = opto_2R_avg_set2
    
    # CD dot product (use trial 2)
    CD_dotproduct[fx] = functions.compute_CD_dotproduct(opto_1R_set2,opto_1L_set2,opto_2R_set2,opto_2L_set2,tix_response)
    
    #############################################
    ## Takes a while to load spatial foot print #
    #############################################
    # # spatial footprints
    # _sf = data['spatial_footprints'][:,:,keep]
    # ncell  = np.sum(keep)
    # nspace = _sf.shape[0]
    # flat_idx = _sf.reshape(-1,ncell).argmax(axis=0)
    # _sf_fx = np.zeros((2,ncell))
    # _sf_fx[0], _sf_fx[1] = np.unravel_index(flat_idx,(nspace,nspace))
    # # save spatial footprint
    # sf[fx] = _sf_fx

if save_CD_dotproduct:
    datapath = 'data/CDdotproduct/'
    np.save(datapath + 'CDdotproduct.npy', CD_dotproduct)


#%%
############################
# Plot sequential activity #
############################
importlib.reload(plot_sequential_activity)

plot_sequential = plot_sequential_activity.SequentialActivityPlotter(
    P1_set1, A1_set1, P2_set1, A2_set1,
    tvec,
    tsample, tdelay, tresponse,
    tix_sample, tix_delay, tix_response,
    nfile,
    sf=sf
)

#%%
# Heat map of all four trial-types in sessions
plot_sequential.plot_all_sessions(sort_mode='own', savefig=True)
plot_sequential.plot_all_sessions(sort_mode='context1', savefig=True)
# 3D plot of a trial-type
plot_sequential.plot_3d(fx=29,act='A1',savefig=True)

 
#%%
'''
Fitting individual neuron activity:
    - Model: two-sided double exponential
    - Parameters:
        * A    : amplitude
        * b    : baseline activity
        * tp   : peak time
        * tau_r: rise time
        * tau_d: decay time
        
Data structure:
    - session number      : sess{number}
    - trial-type          : P1, A1, P2, A2
        * neuron activity : data
        * fit to model    : fit
        * fit parameters  : par (shown above)
        * error neurons   : err (this should be empty if all  neurons are fitted)
        
Peak time constrained to task activity:
    - Searched for the peak time of neuron activity during the task
    - Time interval = [t_sample, t_response + 1.2]
'''
dt = 1/6
tix_start = tix_sample
tix_end   = tix_response+8
ntimestep_fit = tix_end - tix_start
tstart = tvec[tix_start]
tend   = tvec[tix_end]

fitmodel = False

if fitmodel:        
    modelfit = functions.fit_exp_decay_to_neuron_activity(nfile, P1_set1, A1_set1, P2_set1, A2_set1, tvec, tix_start, tix_end)    
    # save the fitted model 
    datapath = 'data/fit_neuron_activity/'
    # np.save(datapath + 'modelfit.npy', modelfit)
               
#%%
#=======================#
# Load the fitted model #
#=======================#
datapath = 'data/fit_neuron_activity/'
modelfit = np.load(datapath + 'modelfit.npy', allow_pickle=True).item()

#==========================================#
# Collecte the fit summary of all sessions #
#==========================================#
acts = ['P1', 'A1', 'P2', 'A2']
actidx = {'P1':0,'A1':1,'P2':2,'A2':3}
nact = len(acts)

fit_summary = functions.get_fit_summary(nfile, modelfit, acts)


    
#%%
########################
# Plot the fit summary #
########################
importlib.reload(functions)
importlib.reload(plot_fit_twosided_exp)
figpath = 'figure/modelfit/'

plot_modelfit = plot_fit_twosided_exp.ModelFitPlotter(
    modelfit,
    fit_summary,
    tvec,
    tix_start,
    tix_end,
    tix_sample,
    tix_delay,
    tix_response,
    nfile,
    figpath
)
# collect variables to be plotted
plot_modelfit.collect_total_variables()
plot_modelfit.find_top_90_cells(threshold=0.9)
plot_modelfit.collect_tau_variables()

# plot model fit
plot_modelfit.plot_example_fits(fx=fx, act=act, savefig=True)
plot_modelfit.plot_variance_fraction(savefig=False)
plot_modelfit.plot_peak_time_hist(savefig=True)
plot_modelfit.plot_expvar_neuron_hist(savefig=True)
plot_modelfit.plot_expvar_pop_hist(savefig=True)
plot_modelfit.plot_tau_hist(savefig=True)
plot_modelfit.plot_tau_vs_top_cells(savefig=True)


#%%
'''
Dictionary of topcells
    * topcells
    * topcells_at_t
    # newcells_at_t
    * expvar_at_t
    * data_binary
Fit summary of topcells
    * goodness of fit
    * taur
    * taud
    * amp
    * expvar_at_t
    * taur_at_t
    * taud_at_t
    * amp_at_t
'''
importlib.reload(functions)

topcells_are_shuffled = False
thr_var = 0.8
t = tvec[tix_start:tix_end]
tgo = t - tvec[tix_response]
dt = t[1] - t[0]

dict_topcells = functions.get_dict_topcells(acts, nfile, thr_var, modelfit, fit_summary)

fit_summary_topcells = functions.get_fit_summary_topcells(t, nfile, acts, actidx, dict_topcells, fit_summary)
    
new_active_cells_id, new_active_cells_duration = functions.compute_new_active_cells(t, nact, nfile, acts, dict_topcells)

active_cells_decay_actual, active_cells_decay_norm, active_cells_decay_binary = functions.compute_active_cells_decay(nfile, nact, t, acts, dict_topcells, modelfit, fit_summary, topcells_are_shuffled)

# eff_avg, eff_sem, eff_err = functions.compute_neuron_time_constant(nfile, 'P1', idx_amp, idx_taud, new_active_cells_id['P1'], new_active_cells_duration['P1'], fit_summary, modelfit)

active_cell_activity_type = 'actual'
# active_cell_activity_type = 'normalized'
# active_cell_activity_type = 'binary'
if active_cell_activity_type == 'actual':
    active_cells_decay = np.copy(active_cells_decay_actual)
if active_cell_activity_type == 'normalized':
    active_cells_decay = np.copy(active_cells_decay_norm)
if active_cell_activity_type == 'binary':
    active_cells_decay = np.copy(active_cells_decay_binary)
    

# # check if topcells and the collection of new_active_cells_id are exactly the same
# for act in acts:
#     tfarray = np.zeros(nfile,dtype=bool)
#     for fx in range(nfile):
#         tfarray[fx] = np.all(np.sort(np.concatenate(list(new_active_cells_id[act][fx].values())))==dict_topcells[act]['topcells'][fx])
#     print(act, np.all(tfarray))

# # check if newcells_at_t and new_active_cells_id are exactly the same.
# for act in acts:
#     tfarray = np.zeros((nfile,24),dtype=bool)
#     for fx in range(nfile):
#         for ti in range(24):
#             tfarray[fx,ti] = np.all(np.sort(dict_topcells[act]['newcells_at_t'][fx][ti]) == np.sort(new_active_cells_id[act][fx][ti]))
#     print(act, np.all(tfarray))


#%%
importlib.reload(functions)
importlib.reload(plot_active_cells_dynamics)

act = 'A2'
for fx in range(nfile):
# for fx in range(5):    
    figpath = f'figure/neural_dynamics/active_cells/' + act + '/'

    ActiveCellsPlotter = plot_active_cells_dynamics.ActiveCellsPlotter(
        CD_dotproduct,
        new_active_cells_id[act],
        dict_topcells[act],
        active_cells_decay[actidx[act]],
        fit_summary_topcells[act],
        fit_summary,
        modelfit,
        t,
        tgo,
        nfile,
        act,
        figpath
    )
    
    savefig = True
    
    ActiveCellsPlotter.get_variables(fx)
    ActiveCellsPlotter.get_active_cell_dynamics(fx)
    ActiveCellsPlotter.plot_active_cell_sorted_activity(savefig=savefig,     dirname='activity/sorted')
    ActiveCellsPlotter.plot_active_cell_heatmap(savefig=savefig, dirname='activity/heatmap')
    ActiveCellsPlotter.plot_active_cell_time_constant(savefig=savefig, dirname='rate of change/decay constant')
    ActiveCellsPlotter.plot_active_cell_traces(savefig=savefig,        dirname='rate of change/traces')
    ActiveCellsPlotter.plot_active_cell_3d_traces(savefig=savefig,        dirname='rate of change/3d_traces')
    ActiveCellsPlotter.plot_average_summary(savefig=savefig, dirname='summary')
    # ActiveCellsPlotter.plot_numcell_to_cumvar(savefig)
    # ActiveCellsPlotter.plot_fit_neuron_taud_and_expvar(savefig=savefig, dirname='fit neuron/taud')
    # ActiveCellsPlotter.plot_fit_neuron_examples(time_picked=np.arange(len(t)),savefig=savefig, dirname='fit neuron/examples')
    # ActiveCellsPlotter.plot_fit_neuron_accuracy_vs_amp(savefig=savefig, dirname='fit neuron/accuracy')
    
#%%
importlib.reload(functions)
importlib.reload(plot_active_cells_dynamics)

savefig = True

ActiveCellsPlotter.get_active_cell_dynamics(fx=0)
ActiveCellsPlotter.plot_CDdotprod_vs_delay_activity(savefig, dirname='CDdotprod')
ActiveCellsPlotter.plot_CDdotprod_vs_sample_activity(savefig, dirname='CDdotprod')




