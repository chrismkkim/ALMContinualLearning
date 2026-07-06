#%%
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


# %%
dP = {}
dA = {}
dR = {}
dL = {}
P1_set1, P1_set2 = {}, {}
A1_set1, A1_set2 = {}, {}
A2_set1, A2_set2 = {}, {}
P2_set1, P2_set2 = {}, {}
S1 = {}
S2 = {}
dS = {}
cells_dS     = {}
cells_dS_neg = {}
cells_dS_pos = {}
cells_dS_neg_S1_pos = {}
cells_dS_neg_S1_neg = {}
cells_dS_pos_S1_pos = {}
cells_dS_pos_S1_neg = {}
cells_pos = {}
cells_neg = {}
topcells_dS_pos = {}
topcells_dS_neg = {}
num_totalcells = np.zeros(nfile)
num_outliers = np.zeros(nfile)
num_topcells = np.zeros(nfile)
sf={}


for fx in range(nfile):
    # print(fx)
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
    opto_1R_delay_set1 = np.mean(opto_1R_set1,axis=2) # neurons x time x trials
    opto_1L_delay_set1 = np.mean(opto_1L_set1,axis=2) # neurons x time x trials
    opto_2R_delay_set1 = np.mean(opto_2R_set1,axis=2) # neurons x time x trials
    opto_2L_delay_set1 = np.mean(opto_2L_set1,axis=2) # neurons x time x trials
    # trial set 2
    opto_1R_delay_set2 = np.mean(opto_1R_set2,axis=2) # neurons x time x trials
    opto_1L_delay_set2 = np.mean(opto_1L_set2,axis=2) # neurons x time x trials
    opto_2R_delay_set2 = np.mean(opto_2R_set2,axis=2) # neurons x time x trials
    opto_2L_delay_set2 = np.mean(opto_2L_set2,axis=2) # neurons x time x trials
    
    # neural activity - outlier neurons removed
    P1_set1[fx] = opto_1R_delay_set1
    A1_set1[fx] = opto_1L_delay_set1
    P2_set1[fx] = opto_2L_delay_set1
    A2_set1[fx] = opto_2R_delay_set1
    P1_set2[fx] = opto_1R_delay_set2
    A1_set2[fx] = opto_1L_delay_set2
    P2_set2[fx] = opto_2L_delay_set2
    A2_set2[fx] = opto_2R_delay_set2
    
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
#========================================================================#
# BELOW IS ABOUT FITTING TWO-SIDED EXPONENTIAL TO SINGLE NEURON ACTIVITY #
#========================================================================#
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
#######################################################
# Fit single neuron activity to two-sided exponential #
#######################################################
from scipy.optimize import curve_fit

def two_sided_exp(t, b, A, tp, tau_r, tau_d):
    return b + A * np.where(
        t < tp,
        np.exp((t - tp) / tau_r),
        np.exp(-(t - tp) / tau_d)
    )

dt = 1/6
tix_start = tix_sample
tix_end   = tix_response+8
ntimestep_fit = tix_end - tix_start
tstart = tvec[tix_start]
tend   = tvec[tix_end]

#%%
fitmodel = False

if fitmodel:        
    modelfit_sess = {'P1': {},'A1':{},'P2':{},'A2':{}}
    modelfit = {f'sess{i}': copy.deepcopy(modelfit_sess) for i in range(nfile)}
    max_retry = 10
    dataset = ['P1','A1','P2','A2']
    for fx in range(nfile):
        print('----- Session ', fx, '-----')        
        for _data in dataset:
            if _data == 'P1':
                data_fit = P1_set1[fx][:,tix_start:tix_end]
            if _data == 'A1':
                data_fit = A1_set1[fx][:,tix_start:tix_end]
            if _data == 'P2':
                data_fit = P2_set1[fx][:,tix_start:tix_end]
            if _data == 'A2':
                data_fit = A2_set1[fx][:,tix_start:tix_end]
                        
            print('Fit ', _data)
            
            ncell = data_fit.shape[0]
            expvar = np.zeros(ncell)
            y_data_save = np.zeros((ncell,ntimestep_fit))
            y_fit_save = np.zeros((ncell,ntimestep_fit))
            y_par_save = np.zeros((ncell,5))
            err_cell = []
            for celli in range(ncell): 
                # data to fit
                y = data_fit[celli]
                t = tvec[tix_start:tix_end]
                
                ############################################
                # Model parameters: b, A, tp, tau_r, tau_d #
                ############################################
                # Initial guesses
                b0 = np.percentile(y, 10)
                A0 = np.max(y) - b0
                tp0 = t[np.argmax(y)]
                tau_r0 = 0.5
                tau_d0 = 1.0
                p0 = [b0, A0, tp0, tau_r0, tau_d0]            
                # Bounds: b, A, tp, tau_r, tau_d
                bounds = (
                    [0, 0, t.min(), 0, 0],
                    [100, 100, t.max(), t[-1]-t[0], t[-1]-t[0]])
                
                success = False
                maxfev = 50000
                peak_shifted = -1
                # fit the data. Allow retries.
                for retry in range(max_retry):                
                    try:
                        popt, pcov = curve_fit(
                            two_sided_exp,
                            t,
                            y,
                            p0=p0,
                            bounds=bounds,
                            maxfev=maxfev
                        )
                        b, A, tp, tau_r, tau_d = popt

                        # Plot fit
                        y_fit = two_sided_exp(t, *popt)
                        explained_variance = 1 - np.var(y - y_fit) / np.var(y)
                        expvar[celli] = explained_variance        
                        
                        # save
                        y_data_save[celli] = y
                        y_fit_save[celli]  = y_fit
                        y_par_save[celli]  = [b,A,tp,tau_r,tau_d]
                        
                        success= True
                        break
                    except Exception:
                        print('retry fitting ', retry)
                        if retry > 5:
                            peak_shifted -= 1
                        tp0 = t[np.argsort(y)[peak_shifted]]
                        p0 = [b0, A0, tp0, tau_r0, tau_d0]
                        maxfev += 50000

                if not success:
                    err_cell.append(celli)
                    y_data_save[celli] = y

            # assert err_cnt == 0, "cells not fitted"
            modelfit[f'sess{fx}'][_data]['data'] = y_data_save
            modelfit[f'sess{fx}'][_data]['fit'] = y_fit_save
            modelfit[f'sess{fx}'][_data]['par'] = y_par_save
            modelfit[f'sess{fx}'][_data]['expvar'] = expvar
            modelfit[f'sess{fx}'][_data]['err'] = err_cell
    
    # save the fitted model 
    datapath = 'data/fit_neuron_activity/'
    # np.save(datapath + 'modelfit.npy', modelfit)
               
#%%
#########################
# Load the fitted model #
#########################
datapath = 'data/fit_neuron_activity/'
modelfit = np.load(datapath + 'modelfit.npy', allow_pickle=True).item()


#%%
############################################
# Collecte the fit summary of all sessions #
############################################
acts = ['P1', 'A1', 'P2', 'A2']
idx_amp = 1
idx_tp  = 2
idx_taur = 3
idx_taud = 4

fit_summary = {
    'var': {},
    'amp': {},
    'tp': {},
    'taur': {},
    'taud': {},
    'expvar_neuron': {},
    'expvar_pop': {},
    'nonoutlier': {},
}

for fx in range(nfile):
    sess = modelfit[f'sess{fx}']

    data = {act: sess[act]['data'] for act in acts}
    fit  = {act: sess[act]['fit']  for act in acts}
    amp  = {act: sess[act]['par'][:, idx_amp] for act in acts}
    tp   = {act: sess[act]['par'][:, idx_tp]  for act in acts}
    taur = {act: sess[act]['par'][:, idx_taur]  for act in acts}
    taud = {act: sess[act]['par'][:, idx_taud]  for act in acts}

    max_vals = [np.max(data[act], axis=1) for act in acts]
    keep_nonoutliers = functions.remove_outliers_alltrials_fixedtime(*max_vals, maxstd=5)
    keep_nonzero = np.logical_and.reduce([np.sum(data[act], axis=1) > 0 for act in acts])
    keep = keep_nonoutliers & keep_nonzero

    ncell = np.sum(keep)
    expvar_neuron = np.zeros((len(acts), ncell))
    expvar_pop    = np.zeros(len(acts))
    var_neuron    = np.zeros((len(acts), ncell))
    amp_neuron    = np.zeros((len(acts), ncell))
    tp_neuron     = np.zeros((len(acts), ncell))
    taur_neuron   = np.zeros((len(acts), ncell))
    taud_neuron   = np.zeros((len(acts), ncell))
    for ix, act in enumerate(acts):
        _data = data[act][keep]
        _fit  = fit[act][keep]

        expvar_neuron[ix] = 1 - np.var(_fit - _data, axis=1) / np.var(_data, axis=1)
        expvar_pop[ix]    = 1 - np.var(_fit - _data) / np.var(_data)
        var_neuron[ix]    = np.var(_data, axis=1)
        amp_neuron[ix]    = amp[act][keep]
        tp_neuron[ix]     = tp[act][keep]
        taur_neuron[ix]   = taur[act][keep]
        taud_neuron[ix]   = taud[act][keep]

    fit_summary['expvar_neuron'][fx] = expvar_neuron
    fit_summary['expvar_pop'][fx]    = expvar_pop
    fit_summary['var'][fx]    = var_neuron
    fit_summary['amp'][fx]    = amp_neuron
    fit_summary['tp'][fx]     = tp_neuron
    fit_summary['taur'][fx]   = taur_neuron
    fit_summary['taud'][fx]   = taud_neuron
    fit_summary['nonoutlier'][fx] = np.where(keep)[0]

    
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

#%%
# plot model fit
plot_modelfit.plot_example_fits(fx=fx, act=act, savefig=True)
plot_modelfit.plot_variance_fraction(savefig=False)
plot_modelfit.plot_peak_time_hist(savefig=True)
plot_modelfit.plot_expvar_neuron_hist(savefig=True)
plot_modelfit.plot_expvar_pop_hist(savefig=True)
plot_modelfit.plot_tau_hist(savefig=True)
plot_modelfit.plot_tau_vs_top_cells(savefig=True)




 
#%%
def sort_cells_by_var(_data):
    # _data_sq = _data**2
    _data_sq = (_data - np.mean(_data,axis=0))**2
    _data_norm = _data_sq / np.sum(_data_sq,axis=0)
    _cells_sorted_at_t = np.argsort(_data_norm,axis=0)[::-1]
    _data_sorted = np.take_along_axis(_data_norm, _cells_sorted_at_t, axis=0)
    _data_cumsum = np.cumsum(_data_sorted,axis=0)
    return _cells_sorted_at_t, _data_cumsum

def create_binary_data(_data, _thr_var):
    _ncell, _ntime = _data.shape
    _cells_sorted_at_t, _data_cumsum = sort_cells_by_var(_data)
    _idx_cross = np.argmax(_data_cumsum >= _thr_var, axis=0)
    _topcells_at_t = {i:[] for i in range(_ntime)}
    _newcells_at_t = {i:[] for i in range(_ntime)}
    _topcells_prev = []
    _data_binary = np.zeros_like(_data)
    for i in range(_ntime):
        # top cells at t
        _topcells_at_t[i] = _cells_sorted_at_t[:_idx_cross[i],i]
        _data_binary[_topcells_at_t[i],i] = 1    
        # only the new top cells added at t
        _newcells_at_t[i] = _topcells_at_t[i][~np.isin(_topcells_at_t[i],_topcells_prev)]
        # top cells accumulated over time
        _topcells_prev = np.concatenate((_topcells_prev,_newcells_at_t[i]))
    _expvar_at_t = np.array([_data_cumsum[_idx_cross[i],i] for i in range(_ntime)])
    return _expvar_at_t, _topcells_at_t, _newcells_at_t, _data_binary


#%%
'''
Active cells
    topcells
    topcells_at_t
    expvar_at_t
    data_binary
Statistics of active cells
    goodness of fit
    tau rise
    tau decay
    fraction of active cells
'''

thr_var = 0.8
act1 = 'P1'
trial_P1 = 0
t = tvec[tix_start:tix_end]
dt = t[1] - t[0]

topcells = {i:[] for i in range(nfile)}
topcells_at_t = {i:[] for i in range(nfile)}
newcells_at_t = {i:[] for i in range(nfile)}
expvar_at_t = {i:[] for i in range(nfile)}
data_binary = {i:[] for i in range(nfile)}
for fx in range(nfile):
    nonoutlier = fit_summary['nonoutlier'][fx]
    _data   = modelfit[f'sess{fx}'][act1]['data'][nonoutlier,:]
    _expvar_at_t_fx, _topcells_at_t_fx, _newcells_at_t_fx, _data_binary_fx = create_binary_data(_data,thr_var)
    _topcells_fx = np.unique(np.concatenate(list(_topcells_at_t_fx.values())))
    topcells[fx]      = _topcells_fx
    topcells_at_t[fx] = _topcells_at_t_fx
    newcells_at_t[fx] = _newcells_at_t_fx
    expvar_at_t[fx]   = _expvar_at_t_fx
    data_binary[fx]   = _data_binary_fx
    

goodfit_topcells = {i:[] for i in range(nfile)}
taur_topcells   = {i:[] for i in range(nfile)}
taud_topcells   = {i:[] for i in range(nfile)}
cells_sorted = {i:[] for i in range(nfile)}
goodfit_at_t = {i:{} for i in range(nfile)}
taur_at_t = {i:{} for i in range(nfile)}
taud_at_t = {i:{} for i in range(nfile)}
taur_at_t_newcells = {i:{} for i in range(nfile)}
taud_at_t_newcells = {i:{} for i in range(nfile)}
frac_at_t = {i:[] for i in range(nfile)}
for fx in range(nfile):            
    # all top cells
    ncell   = len(fit_summary['nonoutlier'][fx])
    goodfit = fit_summary['expvar_neuron'][fx][trial_P1]
    taur    = fit_summary['taur'][fx][trial_P1]
    taud    = fit_summary['taud'][fx][trial_P1]
    # top cells at t
    goodfit_at_t_fx = {i:[] for i in range(len(t))}
    taur_at_t_fx = {i:[] for i in range(len(t))}
    taud_at_t_fx = {i:[] for i in range(len(t))}
    taur_at_t_fx_newcells = {i:[] for i in range(len(t))}
    taud_at_t_fx_newcells = {i:[] for i in range(len(t))}
    frac_at_t_fx = np.zeros(len(t))
    for i in range(len(t)):
        goodfit_at_t_fx[i] = goodfit[topcells_at_t[fx][i]]
        taur_at_t_fx[i] = taur[topcells_at_t[fx][i]]
        taud_at_t_fx[i] = taud[topcells_at_t[fx][i]]
        taur_at_t_fx_newcells[i] = taur[newcells_at_t[fx][i]]
        taud_at_t_fx_newcells[i] = taud[newcells_at_t[fx][i]]
        frac_at_t_fx[i] = len(topcells_at_t[fx][i]) / ncell        
    
    # save
    goodfit_topcells[fx]   = goodfit[topcells[fx]]
    taur_topcells[fx]      = taur[topcells[fx]]
    taud_topcells[fx]      = taud[topcells[fx]]
    cells_sorted[fx]       = np.argsort(np.argmax(_data,axis=1))
    goodfit_at_t[fx]       = goodfit_at_t_fx
    taur_at_t[fx]          = taur_at_t_fx
    taud_at_t[fx]          = taud_at_t_fx
    taur_at_t_newcells[fx] = taur_at_t_fx_newcells
    taud_at_t_newcells[fx] = taud_at_t_fx_newcells
    frac_at_t[fx]    = frac_at_t_fx
    
intensity_topcells = {i:[] for i in range(nfile)}    
for fx in range(nfile):
    nonoutlier      = fit_summary['nonoutlier'][fx]
    _data_fx        = modelfit[f'sess{fx}'][act1]['data'][nonoutlier,:]
    _data_binary_fx = data_binary[fx]
    intensity_topcells[fx] = np.sum(_data_fx * _data_binary_fx, axis=0)
    

new_active_cells_id       = {i:{} for fx in range(nfile)}
new_active_cells_duration = {i:{} for fx in range(nfile)}
# frac_shared = {i:[] for i in range(len(t))}
# frac_shared_timeconstant = np.zeros((nfile,len(t)))
frac_shared_arr = np.zeros((nfile,len(t),len(t)))
frac_shared_arr_norm = np.zeros((nfile,len(t),len(t)))
for fx in range(nfile):
    _data_binary = data_binary[fx].astype(bool)
    _ncell = _data_binary.shape[0]
    _frac_shared_arr = np.zeros((len(t),len(t)))
    _frac_shared_arr_norm = np.zeros((len(t),len(t)))
    _pre_active_cells = np.zeros(_ncell).astype(bool)
    _new_active_cells_id = {i:[] for i in range(len(t))}
    _new_active_cells_duration = {i:[] for i in range(len(t))}
    for i in range(len(t)):
        _frac_shared = np.zeros(len(t)-i)
        _frac_shared_norm = np.zeros(len(t)-i)
        _pre_active_cells      += _data_binary[:,i-1] if i > 0 else False # accumulate cells activated previously
        _new_active_cells_at_i  = ~_pre_active_cells * _data_binary[:,i]
        for jix, j in enumerate(np.arange(i,len(t))):
            '''
            decay of newly activated group of neurons
            '''            
            #---- decay of new active cell group at t 
            _still_active_cells_at_j = _new_active_cells_at_i * _data_binary[:,j]
            _frac_shared[jix]        = np.sum(_still_active_cells_at_j) / np.sum(_ncell)
            _frac_shared_norm[jix]   = np.sum(_still_active_cells_at_j) / np.sum(_new_active_cells_at_i)            
            #---- decay of all active cells at t
            # _frac_shared[jix] = np.sum(_data_binary[:,i]*_data_binary[:,j]) / np.sum(_ncell)
            # _frac_shared[jix] = np.sum(_data_binary[:,i]*_data_binary[:,j]) / np.sum(_data_binary[:,i])
        # duration of new active cells at t
        '''
        duration of individual activated neurons (on to off time)
        '''        
        # duration of each active neuron (on to off time)
        _new_active_cells_id[i] = np.where(_new_active_cells_at_i)[0]
        _new_active_cells_duration[i] = np.array([idx_active_off[0] if (idx_active_off := np.where(~row)[0]).size else len(row) for row in _data_binary[_new_active_cells_at_i,i:]])
        # save the decay of new active neuron group
        # frac_shared[i] = _frac_shared
        _frac_shared_arr[i,i:] = _frac_shared
        _frac_shared_arr_norm[i,i:] = _frac_shared_norm
        # # time constant of active neurons at t
        # time_idx = np.where(_frac_shared < 0.1*_frac_shared[0])[0]
        # frac_shared_timeconstant[fx,i] = time_idx[0] if time_idx.size else np.nan
    '''
    CHECK HERE
    '''    
    new_active_cells_id[fx] = _new_active_cells_id
    new_active_cells_duration[fx] = _new_active_cells_duration
    frac_shared_arr[fx] = _frac_shared_arr
    frac_shared_arr_norm[fx] = _frac_shared_arr_norm
    
# check if topcells and the collection of new_active_cells_id are exactly the same
tfarray = np.zeros(nfile,dtype=bool)
for fx in range(nfile):
    tfarray[fx] = np.all(np.sort(np.concatenate(list(new_active_cells_id[fx].values())))==topcells[fx])
print(np.all(tfarray))

# check if newcells_at_t and new_active_cells_id are exactly the same.
tfarray = np.zeros((nfile,24),dtype=bool)
for fx in range(nfile):
    for ti in range(24):
        tfarray[fx,ti] = np.all(np.sort(newcells_at_t[fx][ti]) == np.sort(new_active_cells_id[fx][ti]))
print(np.all(tfarray))
    
#%%
def compute_neuron_time_constant(new_active_cells_id, new_active_cells_duration, fit_summary, modelfit):
    klist = np.linspace(0.01,0.1,num=10)
    dur_append = {i:[] for i in range(len(klist))}
    eff_append = {i:[] for i in range(len(klist))}
    for ix, k in enumerate(klist):
        _dur_append = np.array([])
        _eff_append = np.array([])
        for fx in range(nfile):    
            new_active_cells_id_fx       = new_active_cells_id[fx]
            new_active_cells_duration_fx = new_active_cells_duration[fx]
            nonoutlier                   = fit_summary['nonoutlier'][fx]
            _amp                         = modelfit[f'sess{fx}'][act1]['par'][nonoutlier,:][:,idx_amp]
            _taud                        = modelfit[f'sess{fx}'][act1]['par'][nonoutlier,:][:,idx_taud]    
            for tix in range(24):
                _cells_at_t    = new_active_cells_id_fx[tix]
                _duration_at_t = new_active_cells_duration_fx[tix]
                _amp_at_t      = _amp[_cells_at_t]
                _taud_at_t     = _taud[_cells_at_t]
                _eff_at_t      = -_taud_at_t * np.log(k/_amp_at_t)
                _dur_append    = np.concatenate((_dur_append,_duration_at_t))
                _eff_append    = np.concatenate((_eff_append,_eff_at_t))
        dur_append[ix] = _dur_append
        eff_append[ix] = _eff_append

    eff_avg = {i:[] for i in range(len(klist))}
    eff_sem = {i:[] for i in range(len(klist))}
    eff_err = np.zeros(len(klist))
    for ix in range(len(klist)):    
        maxduration = np.max(dur_append[ix]).astype(int)
        duration = np.arange(1,maxduration-3)
        duration_in_sec = duration / 6
        eff_avg[ix] = np.array([np.nanmean(eff_append[ix][dur_append[ix] == tix]) for tix in duration])
        eff_sem[ix] = np.array([np.nanstd(eff_append[ix][dur_append[ix] == tix] / np.sqrt(np.sum(dur_append[ix] == tix))) for tix in duration])
        eff_err[ix] = np.linalg.norm(duration_in_sec - eff_avg[ix])
    return eff_avg, eff_sem, eff_err

eff_avg, eff_sem, eff_err = compute_neuron_time_constant(new_active_cells_id, new_active_cells_duration, fit_summary, modelfit)

minix = np.argmin(eff_err)
minix=7
_eff_avg = eff_avg[minix]
_eff_sem = eff_sem[minix]
tlen = _eff_avg.size
duration_in_sec = np.arange(tlen) * dt
plt.figure(figsize=(3,3))
plt.plot(duration_in_sec, eff_avg[minix], c='k')
plt.fill_between(duration_in_sec,_eff_avg-_eff_sem,_eff_avg+_eff_sem,color='gray',alpha=0.3)
plt.plot(duration_in_sec,duration_in_sec,c='gray',linestyle='--')
plt.xlabel('duration of active neuron')
plt.ylabel('estimated decay time')
plt.tight_layout()

plt.figure(figsize=(3,3))
plt.plot(eff_err)

#%%
def event_times():
    plt.axvline(8,color='gray',linestyle='--')
    plt.axvline(16,color='gray',linestyle='--')

fx = 21
new_active_cells_id_fx = new_active_cells_id[fx]
topcells_sorted_by_activation_time = np.concatenate(list(new_active_cells_id_fx.values()))

num_topcells = len(topcells_sorted_by_activation_time)
frac_new_active_cells_fx = np.array([len(_active_cells_id_at_t) / num_topcells for _active_cells_id_at_t in list(new_active_cells_id_fx.values())])
frac_new_active_cells = np.zeros((nfile,24))
for fxx in range(nfile):
    frac_new_active_cells[fxx] = np.array([len(_active_cells_id_at_t) / num_topcells for _active_cells_id_at_t in list(new_active_cells_id[fxx].values())])
frac_new_active_cells_mean = np.mean(frac_new_active_cells,axis=0)
frac_new_active_cells_sem  = np.std(frac_new_active_cells,axis=0) / np.sqrt(nfile)
    
frac_shared_fx = frac_shared_arr[fx]
frac_shared_fx_norm = frac_shared_arr_norm[fx]
frac_shared_avg_norm = np.nanmean(frac_shared_arr_norm,axis=0)

active_cells_duration_fx_mean = np.array([np.mean(dur_at_t) if dur_at_t.size else 0 for dur_at_t in list(new_active_cells_duration[fx].values())])
active_cells_duration_all = np.zeros((nfile,24))
for fxx in range(nfile):
    active_cells_duration_all[fxx] = np.array([np.mean(dur_at_t) if dur_at_t.size else 0 for dur_at_t in list(new_active_cells_duration[fxx].values())])
active_cells_duration_all_mean = np.mean(active_cells_duration_all,axis=0)

nonoutlier = fit_summary['nonoutlier'][fx]
_data_fx      = modelfit[f'sess{fx}'][act1]['data'][nonoutlier,:]
_data_binary_fx = data_binary[fx]
_data_fx_topcells_sorted = _data_fx[topcells_sorted_by_activation_time]
_data_binary_fx_topcells_sorted = _data_binary_fx[topcells_sorted_by_activation_time]


plt.figure(figsize=(4,8))
plt.subplot(311)
plt.imshow(_data_fx_topcells_sorted,cmap='jet',vmin=0,vmax=0.2,aspect='auto')
event_times()
plt.subplot(312)
plt.imshow(_data_binary_fx_topcells_sorted,cmap='binary',vmin=0,vmax=1,aspect='auto',interpolation='None')
event_times()
plt.subplot(313)
rng = np.arange(2,24)
plt.plot(np.arange(24),0*np.arange(24), c='gray',linestyle='--')
plt.plot(rng,frac_new_active_cells_fx[rng], c='k')
plt.plot(rng,frac_new_active_cells_mean[rng], c='r', linestyle='--')
plt.fill_between(rng,frac_new_active_cells_mean[rng]-frac_new_active_cells_sem[rng],frac_new_active_cells_mean[rng]+frac_new_active_cells_sem[rng],color='r',alpha=0.3)
plt.ylabel('frac of newly activated cells')
event_times()
plt.tight_layout()


plt.figure(figsize=(4,6))
plt.subplot(211)
plt.imshow(frac_shared_fx_norm,cmap='copper',vmin=0,vmax=1,aspect='auto')
plt.xlabel('elapsed time')
plt.ylabel('activation time')
event_times()
plt.subplot(212)
plt.plot(active_cells_duration_fx_mean * dt, c='k')
plt.ylabel('mean duration of activated neurons (s)')
event_times()
plt.tight_layout()



plt.figure(figsize=(4,6))
plt.subplot(311)
plt.imshow(frac_shared_avg_norm,cmap='copper',vmin=0,vmax=1,aspect='auto')
event_times()
plt.xlabel('elapsed time')
plt.ylabel('activation time')
plt.subplot(312)
plt.plot(active_cells_duration_all_mean * dt, c='k')
plt.plot(rng,5*frac_new_active_cells_mean[rng], c='r', linestyle='--')
plt.xlabel('activation time')
plt.ylabel('mean duration (s)')
event_times()
plt.subplot(313)
for i in np.arange(0,8):
    telapsed = np.arange(i,len(t))
    plt.plot(telapsed, frac_shared_avg_norm[i,i:],c='k')
for i in np.arange(8,16):
    telapsed = np.arange(i,len(t))
    plt.plot(telapsed, frac_shared_avg_norm[i,i:],c='b')
for i in np.arange(16,24):
    telapsed = np.arange(i,len(t))
    plt.plot(telapsed, frac_shared_avg_norm[i,i:],c='r')
event_times()
plt.tight_layout()

    
#%%

fx = 23
frac_shared_fx = frac_shared_arr[fx]
new_active_cells_id_fx = new_active_cells_id[fx]
new_active_cells_duration_fx = new_active_cells_duration[fx]
taur_at_t_newcells_fx = taur_at_t_newcells[fx]
taud_at_t_newcells_fx = taud_at_t_newcells[fx]

nonoutlier = fit_summary['nonoutlier'][fx]
_data   = modelfit[f'sess{fx}'][act1]['data'][nonoutlier,:]
_fit    = modelfit[f'sess{fx}'][act1]['fit'][nonoutlier,:]
_amp    = modelfit[f'sess{fx}'][act1]['par'][nonoutlier,:][:,idx_amp]
_taud   = modelfit[f'sess{fx}'][act1]['par'][nonoutlier,:][:,idx_taud]
_data_bin_fx = data_binary[fx]


plt.figure()
plt.imshow(_data_bin_fx,cmap='binary',aspect='auto',interpolation='None')
plt.ylim([50,65])
plt.tight_layout()



plt.figure(figsize=(3,3))
plt.plot(eff_err)
plt.tight_layout()

#%%

tix = 15

plt.figure(figsize=(3,5))
plt.subplot(211)
plt.imshow(frac_shared_fx,cmap='copper',vmin=0,vmax=0.01,aspect='auto')
plt.axhline(tix, color='gray', alpha=0.5)
plt.xlabel('elapsed time')
plt.ylabel('activation time')
event_times()
plt.subplot(212)
plt.plot(frac_shared_timeconstant_fx)
plt.axvline(tix, color='gray', alpha=0.5)
plt.ylabel('time constant')
plt.tight_layout()
    

#%%
'''
(1) add: show the range of time constants at every t
(2) new: plot the fraction of total variance explained by top cells at every t
(3) new: mean intensity of active neurons at every t
'''

fx = 22
mean_taur_at_t_fx_topcells = np.array([np.mean(v) for k,v in taur_at_t[fx].items()])
mean_taud_at_t_fx_topcells = np.array([np.mean(v) for k,v in taud_at_t[fx].items()])
mean_taur_at_t_fx = np.array([np.mean(v) for k,v in taur_at_t_newcells[fx].items()])
mean_taud_at_t_fx = np.array([np.mean(v) for k,v in taud_at_t_newcells[fx].items()])
mean_goodfit_at_t_fx = np.array([np.mean(v) for k,v in goodfit_at_t[fx].items()])
intensity_topcells_fx = intensity_topcells[fx]
frac_at_t_fx   = frac_at_t[fx]
frac_shared_fx = frac_shared_arr[fx]
expvar_at_t_fx = expvar_at_t[fx]

nonoutlier = fit_summary['nonoutlier'][fx]
_data_topcells   = modelfit[f'sess{fx}'][act1]['data'][nonoutlier,:][topcells[fx]]
_topcells_sorted = np.argsort(np.argmax(_data_topcells,axis=1))

eff_taud = _taud[_newcells]/dt * np.log(_amp[_newcells]/0.01)

plt.figure(figsize=(10,10))
for fx in range(nfile):
    plt.subplot(6,6,fx+1)
    _x = frac_shared_timeconstant[fx]
    _y = np.array([np.mean(v) for k,v in taud_at_t_newcells[fx].items()])
    # _y = np.array([np.mean(v) for k,v in taud_at_t[fx].items()])
    mask = ~np.isnan(_x) & ~np.isnan(_y)
    cor = np.corrcoef(_x[mask],_y[mask])[0,1]
    plt.scatter(_x,_y,color='k')
    plt.title(str(np.round(cor,decimals=3)))
plt.tight_layout()



plt.figure(figsize=(3,7))
plt.subplot(311)
plt.plot(mean_taur_at_t_fx, label='rise')
plt.plot(mean_taud_at_t_fx, label='decay')
# plt.plot(mean_taur_at_t_fx_topcells, label='topcells')
plt.legend(frameon=False)
plt.ylim([0,1])
event_times()
plt.ylabel('neuron time constant (s)')
plt.subplot(312)
plt.imshow(frac_shared_fx,cmap='copper',vmin=0,vmax=1,aspect='auto')
plt.xlabel('elapsed time')
plt.ylabel('activation time')
event_times()
plt.subplot(313)
for i in np.arange(1,8):
    telapsed = np.arange(i,len(t))
    plt.plot(telapsed, frac_shared_fx[i,i:],c='k')
for i in np.arange(8,16):
    telapsed = np.arange(i,len(t))
    plt.plot(telapsed, frac_shared_fx[i,i:],c='b')
for i in np.arange(16,24):
    telapsed = np.arange(i,len(t))
    plt.plot(telapsed, frac_shared_fx[i,i:],c='r')
event_times()
plt.xlim([0,len(t)])        
plt.tight_layout()



plt.figure(figsize=(3,6))        
plt.subplot(311)
plt.imshow(_data_topcells[_topcells_sorted],cmap='jet',aspect='auto',vmin=0,vmax=0.1)
event_times()
plt.subplot(312)
plt.plot(frac_at_t_fx)
event_times()
plt.ylabel('frac of neurons\nto get 80% var')
plt.ylim([0,0.2])
plt.yticks([0,0.1,0.2])
plt.subplot(313)
plt.plot(intensity_topcells_fx)
plt.ylabel('mean intensity\nof active neurons')
event_times()
plt.tight_layout()



#%%
fx = 16
nonoutlier = fit_summary['nonoutlier'][fx]
_data_topcells   = modelfit[f'sess{fx}'][act1]['data'][nonoutlier,:][topcells[fx]]
_topcells_sorted = np.argsort(np.argmax(_data_topcells,axis=1))
frac_shared_fx = frac_shared_arr[fx]
plt.figure(figsize=(4,5.5))
plt.subplot(411)
plt.imshow(_data_topcells[_topcells_sorted],cmap='jet',aspect='auto',vmin=0,vmax=0.1)
event_times()
plt.subplot(412)
for i in np.arange(1,8):
    telapsed = np.arange(i,len(t))
    plt.plot(telapsed, frac_shared_fx[i,i:])
plt.xlim([0,len(t)])    
event_times()
plt.subplot(413)
for i in np.arange(8,16):
    telapsed = np.arange(i,len(t))
    plt.plot(telapsed, frac_shared_fx[i,i:])
plt.xlim([0,len(t)])        
event_times()
plt.subplot(414)
for i in np.arange(16,24):
    telapsed = np.arange(i,len(t))
    plt.plot(telapsed, frac_shared_fx[i,i:])
plt.xlim([0,len(t)])        
event_times()
plt.tight_layout()


frac_shared_avg = np.nanmean(frac_shared_arr,axis=0)
plt.figure(figsize=(4,4))
plt.subplot(311)
for i in np.arange(1,8):
    telapsed = np.arange(i,len(t))
    plt.plot(telapsed, frac_shared_avg[i,i:])
plt.xlim([0,len(t)])    
event_times()
plt.subplot(312)
for i in np.arange(8,16):
    telapsed = np.arange(i,len(t))
    plt.plot(telapsed, frac_shared_avg[i,i:])
plt.xlim([0,len(t)])        
event_times()
plt.subplot(313)
for i in np.arange(16,24):
    telapsed = np.arange(i,len(t))
    plt.plot(telapsed, frac_shared_avg[i,i:])
plt.xlim([0,len(t)])        
event_times()
plt.tight_layout()

#%%

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection='3d')

for i in np.arange(1, 24):
    telapsed = np.arange(i, len(t))
    y = np.full(len(telapsed), i)
    z = frac_shared_avg[i, i:]
    if i <=8:
        color='k'
    elif (i>8) and (i<=16):
        color='blue'
    else:
        color='red'
    ax.plot(telapsed, y, z, lw=0.8, c=color)

ax.set_xlim([0, len(t)])
ax.set_ylim([1, 24])
ax.set_xlabel('time')
ax.set_ylabel('start time index')
ax.set_zlabel('frac shared')

# event lines
for ev in [tix_sample-8, tix_delay-8, tix_response-8]:
    ax.plot(
        [ev, ev],
        [1, 24],
        [0, 0],
        c='gray',
        linestyle='--',
        lw=0.8
    )

# remove grid
ax.grid(False)

# remove ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# remove panes
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.fill = False
    axis.pane.set_edgecolor((1, 1, 1, 0))

# remove axis lines
ax.xaxis.line.set_color((1, 1, 1, 0))
ax.yaxis.line.set_color((1, 1, 1, 0))
ax.zaxis.line.set_color((1, 1, 1, 0))


ax.view_init(elev=25, azim=-75)
plt.tight_layout()


# plt.figure(figsize=(3,3))
# plt.imshow(np.mean(frac_shared_arr_all,axis=0),cmap='copper',vmin=0,vmax=1,aspect='auto')
# event_times()
# plt.tight_layout()




# plt.figure(figsize=(4,3))
# for fx in range(nfile):
#     plt.plot(expvar_all_t[fx])
# plt.plot(np.mean(expvar_all_t,axis=0),c='k',lw=2)    
# event_times()
# # plt.hist(expvar_all,bins=10,range=(0,1),histtype='step')
# plt.ylabel('explained variance')
# plt.tight_layout()


# plt.figure(figsize=(4,3))
# for fx in range(nfile):
#     plt.plot(taur_all_t[fx])
# plt.plot(np.mean(taur_all_t,axis=0),c='k')   
# event_times()
# plt.ylim([0,1]) 
# plt.ylabel('rise time')
# plt.tight_layout()

        
# plt.figure(figsize=(4,3))
# for fx in range(nfile):
#     plt.plot(taud_all_t[fx])
# plt.plot(np.mean(taud_all_t,axis=0),c='k')    
# event_times()
# plt.ylim([0,1]) 
# plt.ylabel('decay time')
# plt.tight_layout()


# plt.figure(figsize=(4,3))
# for fx in range(nfile):
#     plt.plot(frac_all_t[fx])
# plt.plot(np.mean(frac_all_t,axis=0),c='k')    
# event_times()
# plt.ylabel('frac cells')
# plt.tight_layout()

#%%

# plt.figure(figsize=(3,3))
# plt.imshow(np.mean(frac_shared_arr_all,axis=0),cmap='copper',vmin=0,vmax=1,aspect='auto')
# event_times()
# plt.tight_layout()


# plt.figure(figsize=(3,15))
# for i in range(15):
#     plt.subplot(15,1,i+1)
#     plt.plot(frac_shared[i])
#     plt.xlim([0,len(t)])
# plt.tight_layout()



        
#%%
plt.figure(figsize=(4,4))
plt.subplot(211)
plt.hist(expvar_topcells,bins=20,range=(0,1),histtype='step')
plt.axvline(np.mean(expvar[topcells]),color='gray',linestyle='--')
plt.xlabel('explained var')

plt.subplot(212)
plt.hist(taur_topcells,bins=20,range=(0,1),histtype='step')
plt.hist(taud_topcells,bins=20,range=(0,1),histtype='step')
plt.axvline(np.mean(taur[topcells]),color='gray',linestyle='--')
plt.axvline(np.mean(taud[topcells]),color='k',linestyle='--')
plt.xlabel('time constant')
plt.tight_layout()



plt.figure(figsize=(4,2))
plt.plot(taur_at_t)    
event_times()
plt.ylabel('time constant')
plt.tight_layout()

plt.figure(figsize=(4,2))
plt.plot(frac_at_t)    
event_times()
plt.ylabel('frac cells')
plt.tight_layout()


# data_binary_act1 = {}
# data_binary_act2 = {}
# cells_sorted_by_act1 = {}
# cells_sorted_by_act2 = {}
# frac_active_act1 = np.zeros((nfile,24))
# frac_active_act2 = np.zeros((nfile,24))
# frac_active_overlap = np.zeros((nfile,24))

# thr_cumsum = 0.9
# act1 = 'A1'
# act2 = 'P2'
# for fx in range(nfile):
#     _data_act1 = modelfit[f'sess{fx}'][act1]['data']
#     _data_act2 = modelfit[f'sess{fx}'][act2]['data']
#     _cells_sorted_by_act1 = np.argsort(np.argmax(_data_act1,axis=1))
#     _cells_sorted_by_act2 = np.argsort(np.argmax(_data_act2,axis=1))

#     _data_binary_act1, _ncell = create_binary_data(modelfit,act1,thr_cumsum)
#     _data_binary_act2, _ncell = create_binary_data(modelfit,act2,thr_cumsum)
#     _frac_active_act1 = np.sum(_data_binary_act1,axis=0) / _ncell
#     _frac_active_act2 = np.sum(_data_binary_act2,axis=0) / _ncell
#     _frac_active_overlap = np.sum(_data_binary_act1*_data_binary_act2,axis=0) / _ncell

#     data_binary_act1[fx] = _data_binary_act1
#     data_binary_act2[fx] = _data_binary_act2
#     cells_sorted_by_act1[fx]  = _cells_sorted_by_act1
#     cells_sorted_by_act2[fx]  = _cells_sorted_by_act2
#     frac_active_act1[fx]    = _frac_active_act1
#     frac_active_act2[fx]    = _frac_active_act2
#     frac_active_overlap[fx] = _frac_active_overlap
    
# overlap_act1_decision = np.mean(frac_active_overlap[:,12:16]/frac_active_act1[:,12:16],axis=1)
# overlap_act2_decision = np.mean(frac_active_overlap[:,12:16]/frac_active_act2[:,12:16],axis=1)
# overlap_act2_sample   = np.mean(frac_active_overlap[:,0:4]/frac_active_act2[:,0:4],axis=1)


# plt.figure()
# plt.subplot(311)
# for fx in range(nfile):
#     plt.plot(frac_active_overlap[fx]/frac_active_act1[fx])
# plt.plot(np.mean(frac_active_overlap,axis=0)/np.mean(frac_active_act1,axis=0),c='k',lw=2)
# plt.plot(np.mean(frac_active_overlap,axis=0)/np.mean(frac_active_act2,axis=0),c='b',lw=2)    
# plt.subplot(312)
# for fx in range(nfile):
#     plt.plot(frac_active_overlap[fx]/frac_active_act2[fx])
# plt.plot(np.mean(frac_active_overlap,axis=0)/np.mean(frac_active_act1,axis=0),c='k',lw=2)
# plt.plot(np.mean(frac_active_overlap,axis=0)/np.mean(frac_active_act2,axis=0),c='b',lw=2)    
# plt.subplot(313)
# for fx in range(nfile):
#     plt.plot(frac_active_overlap[fx]/(frac_active_act1[fx]*frac_active_act2[fx]))
# plt.ylim([0,10])
# plt.tight_layout()



# plt.figure()
# plt.subplot(211)
# plt.hist(overlap_act1_decision,bins=10,histtype='step', range=(0,1))
# plt.hist(overlap_act2_decision,bins=10,histtype='step', range=(0,1))
# # plt.hist(overlap_act2_sample,  bins=10,histtype='step', range=(0,1))
# plt.subplot(212)
# cnt, bins = np.histogram(overlap_act2_decision,bins=10,range=(0,1))
# cumcnt = np.cumsum(cnt) / np.sum(cnt)
# plt.plot(bins[:-1],cumcnt,marker='o')
# plt.tight_layout()



#%%
fx = 17
trial_P1 = 0
amp_neuron = fit_summary['amp'][fx][trial_P1]
var_neuron = fit_summary['var'][fx][trial_P1]
nonoutlier = fit_summary['nonoutlier'][fx]
_data   = modelfit[f'sess{fx}']['P1']['data'][nonoutlier,:]
_data_sq = _data**2
# _data_sq = (_data - np.mean(_data,axis=0))**2
_ncell = len(nonoutlier)

# select top cells
thresh_var = 0.8
t = tvec[tix_start:tix_end]
dt = t[1] - t[0]
tp_neuron  = fit_summary['tp'][fx][trial_P1]
tp_neuron_idx = np.round((tp_neuron - t[0]) / dt).astype(int)
_cells_at_t = {i:np.where(tp_neuron_idx==i)[0] for i in range(len(t))}
cells_at_t = {i:_cells_at_t[i][np.argsort(amp_neuron[_cells_at_t[i]])[::-1]] for i in range(len(t))}
tvar_at_t = {i:np.sum(_data_sq[:,i]) for i in range(len(t))}
var_at_t = {i:np.sum(_data_sq[cells_at_t[i],i]) / tvar_at_t[i] for i in range(len(t))}


cnt = np.zeros(len(t)).astype(int)
cnt_tf  = np.ones(len(t),dtype=bool)
var_at_tp = np.zeros(len(t))
while(np.any(cnt_tf)):
    for i in range(len(t)):
        if var_at_tp[i] < thresh_var:
            var_at_tp[i] = np.sum(_data_sq[cells_at_t[i][:cnt[i]],i]) / tvar_at_t[i]
            cnt[i] += 1
        else:
            cnt_tf[i] = False
topcells_at_tp = {i:cells_at_t[i][:cnt[i]] for i in range(len(t))}
topcells = np.concatenate(list(topcells_at_tp.values()))

expvar_topcells = fit_summary['expvar_neuron'][fx][trial_P1,topcells]
taur_topcells  = fit_summary['taur'][fx][trial_P1,topcells]

print('frac cells', len(topcells) / _ncell)

_data_sq_topcells = _data_sq[topcells] / np.sum(_data_sq, axis=0)
_cells_sorted = np.argsort(_data_sq_topcells,axis=0)[::-1]
_data_sorted = np.take_along_axis(_data_sq_topcells, _cells_sorted, axis=0)
_data_cumsum = np.cumsum(_data_sorted,axis=0)
_idx_cross = np.argmax(_data_cumsum >= thresh_var, axis=0)



def create_binary_data(_model,_act, _thr_cumsum):
    _data = _model[f'sess{fx}'][_act]['data']
    _ncell, _ntime = _data.shape
    _data_sq = _data**2
    _data_norm = _data_sq / np.sum(_data_sq,axis=0)
    _cells_sorted = np.argsort(_data_norm,axis=0)[::-1]
    _data_sorted = np.take_along_axis(_data_norm, _cells_sorted, axis=0)
    _data_cumsum = np.cumsum(_data_sorted,axis=0)
    _idx_cross = np.argmax(_data_cumsum >= _thr_cumsum, axis=0)
    _data_binary = np.zeros_like(_data)
    for col in range(_ntime):
        _data_binary[_cells_sorted[:_idx_cross[col],col],col] = 1
    return _data_binary, _ncell









#%%
plt.figure()
plt.imshow(_data[topcells],cmap='jet',vmin=0,vmax=0.05,aspect='auto')
plt.tight_layout()



plt.figure()
plt.subplot(211)
plt.hist(expvar_topcells,bins=20,range=(0,1),histtype='step')
plt.axvline(np.mean(expvar_topcells),c='gray',linestyle='--')
plt.xticks([0,0.5,1])

plt.subplot(212)
plt.hist(taur_topcells,bins=20, range=(0,0.5), histtype='step')
plt.axvline(np.mean(taur_topcells),color='gray',linestyle='--')
plt.tight_layout()



# plt.figure()
# plt.hist(tp_neuron_idx, bins=tix_end-tix_start,range=(0,tix_end-tix_start))
# plt.axvline(tix_delay-tix_sample,color='gray',linestyle='--')
# plt.axvline(tix_response-tix_sample,color='gray',linestyle='--')


#%%
amp_frac, var_frac, cells_top90 = functions.find_top_90_cells(amp_neuron,var_neuron,threshold=0.9)

print(len(cells_top90)/_ncell)

_ncell, _ntime = _data.shape
_data_sq = _data**2
_data_norm = _data_sq / np.sum(_data_sq,axis=0)
_data_cells_top90 = np.sum(_data_norm[cells_top90,:],axis=0)

plt.figure()
plt.plot(_data_cells_top90)
event_times()
plt.axhline(0.9,color='r')
plt.ylim([0.5,1])
plt.tight_layout()

plt.figure()
plt.hist(tp_neuron[cells_top90],histtype='step')
# plt.axvline(tvec[tix_sample],color='gray',linestyle='--')
plt.axvline(tvec[tix_delay],color='gray',linestyle='--')
plt.axvline(tvec[tix_response],color='gray',linestyle='--')
plt.tight_layout()

# _cells_sorted = np.argsort(_data_norm,axis=0)[::-1]
# _data_sorted = np.take_along_axis(_data_norm, _cells_sorted, axis=0)
# _data_cumsum = np.cumsum(_data_sorted,axis=0)
# _idx_cross = np.argmax(_data_cumsum >= _thr_cumsum, axis=0)
# _data_binary = np.zeros_like(_data)


# plt.figure()
# plt.plot(amp_frac,var_frac)
# plt.tight_layout()

