#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.stats import norm
from sklearn import linear_model
from sklearn.decomposition import PCA
import copy
import importlib
from utils import functions 


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

figpath = '/Users/kimchm/OneDrive - National Institutes of Health/NIH/research/ALM/code/figure/temp/'

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
dt = 1/6
tix_start = tix_sample
tix_end   = tix_response+8
ntimestep_fit = tix_end - tix_start
tstart = tvec[tix_start]
tend   = tvec[tix_end]

topcells_are_shuffled = False
thr_var = 0.8
t = tvec[tix_start:tix_end]
tgo = t - tvec[tix_response]
dt = t[1] - t[0]

dict_topcells = functions.get_dict_topcells(acts, nfile, thr_var, modelfit, fit_summary)

fit_summary_topcells = functions.get_fit_summary_topcells(t, nfile, acts, actidx, dict_topcells, fit_summary)

figpath = 'figure/neural_dynamics/cross_context/'
datapath = 'data/CDdotproduct/'
CDdotproduct = np.load(datapath + 'CDdotproduct.npy')

#%%

dict_topcells_2 = {
    'P2+A2-': np.array([]),
    'P2+A2+': np.array([]),
    'P2-A2+': np.array([])
}
dict_topcells_1x2_fx = {
    'P1+A1-': copy.deepcopy(dict_topcells_2),
    'P1+A1+': copy.deepcopy(dict_topcells_2),
    'P1-A1+': copy.deepcopy(dict_topcells_2),
}
dict_topcells_1x2 = {fx:copy.deepcopy(dict_topcells_1x2_fx) for fx in range(nfile)}

tix_delay = np.arange(8,16)
for fx in range(nfile):                
    topcells_P1 = np.unique(np.concatenate([dict_topcells['P1']['topcells_at_t'][fx][i] for i in tix_delay]))
    topcells_A1 = np.unique(np.concatenate([dict_topcells['A1']['topcells_at_t'][fx][i] for i in tix_delay]))
    topcells_P2 = np.unique(np.concatenate([dict_topcells['P2']['topcells_at_t'][fx][i] for i in tix_delay]))
    topcells_A2 = np.unique(np.concatenate([dict_topcells['A2']['topcells_at_t'][fx][i] for i in tix_delay]))
    P1_A1 = topcells_P1[~np.isin(topcells_P1,topcells_A1)]
    P1A1  = topcells_P1[ np.isin(topcells_P1,topcells_A1)]
    A1_P1 = topcells_A1[~np.isin(topcells_A1,topcells_P1)]
    P2_A2 = topcells_P2[~np.isin(topcells_P2,topcells_A2)]
    P2A2  = topcells_P2[ np.isin(topcells_P2,topcells_A2)]
    A2_P2 = topcells_A2[~np.isin(topcells_A2,topcells_P2)]
    
    dict_topcells_1x2[fx]['P1+A1-']['P2+A2-'] = P1_A1[np.isin(P1_A1,P2_A2)]
    dict_topcells_1x2[fx]['P1+A1-']['P2+A2+'] = P1_A1[np.isin(P1_A1,P2A2)]
    dict_topcells_1x2[fx]['P1+A1-']['P2-A2+'] = P1_A1[np.isin(P1_A1,A2_P2)]
    dict_topcells_1x2[fx]['P1+A1+']['P2+A2-'] =  P1A1[np.isin(P1A1, P2_A2)]
    dict_topcells_1x2[fx]['P1+A1+']['P2+A2+'] =  P1A1[np.isin(P1A1, P2A2)]
    dict_topcells_1x2[fx]['P1+A1+']['P2-A2+'] =  P1A1[np.isin(P1A1, A2_P2)]
    dict_topcells_1x2[fx]['P1-A1+']['P2+A2-'] = A1_P1[np.isin(A1_P1,P2_A2)]
    dict_topcells_1x2[fx]['P1-A1+']['P2+A2+'] = A1_P1[np.isin(A1_P1,P2A2)]
    dict_topcells_1x2[fx]['P1-A1+']['P2-A2+'] = A1_P1[np.isin(A1_P1,A2_P2)]


CDdp = np.zeros(nfile)
CDdp_1x2 = np.zeros(nfile)
dict_CDdp = copy.deepcopy(dict_topcells_1x2)
for fx in range(nfile):
    # topcells
    topcells_P1 = np.unique(np.concatenate([dict_topcells['P1']['topcells_at_t'][fx][i] for i in tix_delay]))
    topcells_A1 = np.unique(np.concatenate([dict_topcells['A1']['topcells_at_t'][fx][i] for i in tix_delay]))
    topcells_P2 = np.unique(np.concatenate([dict_topcells['P2']['topcells_at_t'][fx][i] for i in tix_delay]))
    topcells_A2 = np.unique(np.concatenate([dict_topcells['A2']['topcells_at_t'][fx][i] for i in tix_delay]))

    # shared & nonshared topcells
    topcells_1 = np.unique(np.concatenate((topcells_P1,topcells_A1)))
    topcells_2 = np.unique(np.concatenate((topcells_P2,topcells_A2)))
    topcells_1x2 = topcells_1[np.isin(topcells_1,topcells_2)]
    
    # compute CD1, CD2
    nonoutlier = fit_summary['nonoutlier'][fx]
    data_P1   = modelfit[f'sess{fx}']['P1']['data'][nonoutlier,:]
    data_A1   = modelfit[f'sess{fx}']['A1']['data'][nonoutlier,:]
    data_P2   = modelfit[f'sess{fx}']['P2']['data'][nonoutlier,:]
    data_A2   = modelfit[f'sess{fx}']['A2']['data'][nonoutlier,:]
    diff1 = np.mean((data_P1 - data_A1)[:,12:16],axis=1)
    diff2 = np.mean((data_P2 - data_A2)[:,12:16],axis=1)
    CD1 = diff1 / np.linalg.norm(diff1)
    CD2 = diff2 / np.linalg.norm(diff2)
    
    # approximate CD1CD2
    CDdp[fx] = np.inner(CD1,CD2)
    CDdp_1x2[fx] = np.inner(CD1[topcells_1x2],CD2[topcells_1x2])
    
    # Divide CD1CD2 into components
    dict_CDdp[fx]['P1+A1-']['P2+A2-'] = np.sum((CD1*CD2)[dict_topcells_1x2[fx]['P1+A1-']['P2+A2-']])
    dict_CDdp[fx]['P1+A1-']['P2+A2+'] = np.sum((CD1*CD2)[dict_topcells_1x2[fx]['P1+A1-']['P2+A2+']])
    dict_CDdp[fx]['P1+A1-']['P2-A2+'] = np.sum((CD1*CD2)[dict_topcells_1x2[fx]['P1+A1-']['P2-A2+']])
    dict_CDdp[fx]['P1+A1+']['P2+A2-'] = np.sum((CD1*CD2)[dict_topcells_1x2[fx]['P1+A1+']['P2+A2-']])
    dict_CDdp[fx]['P1+A1+']['P2+A2+'] = np.sum((CD1*CD2)[dict_topcells_1x2[fx]['P1+A1+']['P2+A2+']])
    dict_CDdp[fx]['P1+A1+']['P2-A2+'] = np.sum((CD1*CD2)[dict_topcells_1x2[fx]['P1+A1+']['P2-A2+']])
    dict_CDdp[fx]['P1-A1+']['P2+A2-'] = np.sum((CD1*CD2)[dict_topcells_1x2[fx]['P1-A1+']['P2+A2-']])
    dict_CDdp[fx]['P1-A1+']['P2+A2+'] = np.sum((CD1*CD2)[dict_topcells_1x2[fx]['P1-A1+']['P2+A2+']])
    dict_CDdp[fx]['P1-A1+']['P2-A2+'] = np.sum((CD1*CD2)[dict_topcells_1x2[fx]['P1-A1+']['P2-A2+']])

dict_CDdp_err = {
    'P1+A1-':np.zeros(nfile),
    'P1+A1+':np.zeros(nfile),
    'P1-A1+':np.zeros(nfile),
    'P1+A1-_P1+A1+':np.zeros(nfile),
    'P1-A1+_P1+A1+':np.zeros(nfile),
    'P1+A1-_P1-A1+':np.zeros(nfile),
    'all':np.zeros(nfile),
    'P2+A2-':np.zeros(nfile),
    'P2+A2+':np.zeros(nfile),
    'P2-A2+':np.zeros(nfile),    
    'P2+A2-_P2+A2+':np.zeros(nfile),
    'P2-A2+_P2+A2+':np.zeros(nfile),
    'P2+A2-_P2-A2+':np.zeros(nfile)    
}
for fx in range(nfile):
    dict_CDdp_err['P1+A1-'][fx] = np.abs(dict_CDdp[fx]['P1+A1-']['P2+A2-'] + dict_CDdp[fx]['P1+A1-']['P2+A2+'] + dict_CDdp[fx]['P1+A1-']['P2-A2+'] - CDdp[fx])
    dict_CDdp_err['P1+A1+'][fx] = np.abs(dict_CDdp[fx]['P1+A1+']['P2+A2-'] + dict_CDdp[fx]['P1+A1+']['P2+A2+'] + dict_CDdp[fx]['P1+A1+']['P2-A2+'] - CDdp[fx])
    dict_CDdp_err['P1-A1+'][fx] = np.abs(dict_CDdp[fx]['P1-A1+']['P2+A2-'] + dict_CDdp[fx]['P1-A1+']['P2+A2+'] + dict_CDdp[fx]['P1-A1+']['P2-A2+'] - CDdp[fx])
    dict_CDdp_err['P1+A1-_P1+A1+'][fx] = np.abs(dict_CDdp[fx]['P1+A1-']['P2+A2-'] + dict_CDdp[fx]['P1+A1-']['P2+A2+'] + dict_CDdp[fx]['P1+A1-']['P2-A2+'] + \
                                                dict_CDdp[fx]['P1+A1+']['P2+A2-'] + dict_CDdp[fx]['P1+A1+']['P2+A2+'] + dict_CDdp[fx]['P1+A1+']['P2-A2+'] - CDdp[fx])
    dict_CDdp_err['P1+A1-_P1-A1+'][fx] = np.abs(dict_CDdp[fx]['P1+A1-']['P2+A2-'] + dict_CDdp[fx]['P1+A1-']['P2+A2+'] + dict_CDdp[fx]['P1+A1-']['P2-A2+'] + \
                                                dict_CDdp[fx]['P1-A1+']['P2+A2-'] + dict_CDdp[fx]['P1-A1+']['P2+A2+'] + dict_CDdp[fx]['P1-A1+']['P2-A2+'] - CDdp[fx])
    dict_CDdp_err['P1-A1+_P1+A1+'][fx] = np.abs(dict_CDdp[fx]['P1-A1+']['P2+A2-'] + dict_CDdp[fx]['P1-A1+']['P2+A2+'] + dict_CDdp[fx]['P1-A1+']['P2-A2+'] + \
                                                dict_CDdp[fx]['P1+A1+']['P2+A2-'] + dict_CDdp[fx]['P1+A1+']['P2+A2+'] + dict_CDdp[fx]['P1+A1+']['P2-A2+'] - CDdp[fx])    
    dict_CDdp_err['all'][fx]           = np.abs(dict_CDdp[fx]['P1+A1-']['P2+A2-'] + dict_CDdp[fx]['P1+A1-']['P2+A2+'] + dict_CDdp[fx]['P1+A1-']['P2-A2+'] + \
                                                dict_CDdp[fx]['P1+A1+']['P2+A2-'] + dict_CDdp[fx]['P1+A1+']['P2+A2+'] + dict_CDdp[fx]['P1+A1+']['P2-A2+'] + \
                                                dict_CDdp[fx]['P1-A1+']['P2+A2-'] + dict_CDdp[fx]['P1-A1+']['P2+A2+'] + dict_CDdp[fx]['P1-A1+']['P2-A2+'] - CDdp[fx])

    dict_CDdp_err['P2+A2-'][fx] = np.abs(dict_CDdp[fx]['P1-A1+']['P2+A2-'] + dict_CDdp[fx]['P1+A1+']['P2+A2-'] - CDdp[fx])
    dict_CDdp_err['P2+A2+'][fx] = np.abs(dict_CDdp[fx]['P1-A1+']['P2+A2+'] + dict_CDdp[fx]['P1+A1+']['P2+A2+'] - CDdp[fx])
    dict_CDdp_err['P2-A2+'][fx] = np.abs(dict_CDdp[fx]['P1-A1+']['P2-A2+'] + dict_CDdp[fx]['P1+A1+']['P2-A2+'] - CDdp[fx])
    dict_CDdp_err['P2+A2-_P2+A2+'][fx] = np.abs(dict_CDdp[fx]['P1-A1+']['P2+A2-'] + dict_CDdp[fx]['P1+A1+']['P2+A2-'] + \
                                                dict_CDdp[fx]['P1-A1+']['P2+A2+'] + dict_CDdp[fx]['P1+A1+']['P2+A2+'] - CDdp[fx])
    dict_CDdp_err['P2-A2+_P2+A2+'][fx] = np.abs(dict_CDdp[fx]['P1-A1+']['P2-A2+'] + dict_CDdp[fx]['P1+A1+']['P2-A2+'] + \
                                                dict_CDdp[fx]['P1-A1+']['P2+A2+'] + dict_CDdp[fx]['P1+A1+']['P2+A2+'] - CDdp[fx])
    dict_CDdp_err['P2+A2-_P2-A2+'][fx] = np.abs(dict_CDdp[fx]['P1-A1+']['P2+A2-'] + dict_CDdp[fx]['P1+A1+']['P2+A2-'] + \
                                                dict_CDdp[fx]['P1-A1+']['P2-A2+'] + dict_CDdp[fx]['P1+A1+']['P2-A2+'] - CDdp[fx])

plt.figure(figsize=(8, 4))
keys = list(dict_CDdp_err.keys())
x = np.arange(len(keys))
bar_width = 0.6
xlabels = [r'$P_1^+A_1^-$', r'$P_1^+A_1^+$', r'$P_1^-A_1^+$', 
           r'$P_1^+A_1^- + P_1^+A_1^+$', r'$P_1^-A_1^+ + P_1^+A_1^+$', r'$P_1^+A_1^- + P_1^-A_1^+$',
           'all',
           r'$P_2^+A_2^-$', r'$P_2^+A_2^+$', r'$P_2^-A_2^+$', 
           r'$P_2^+A_2^- + P_2^+A_2^+$', r'$P_2^-A_2^+ + P_2^+A_2^+$', r'$P_2^+A_2^- + P_2^-A_2^+$']
for i, key in enumerate(keys):
    y = dict_CDdp_err[key]

    # Bar: mean
    plt.bar(
        i,
        np.mean(y),
        width=bar_width,
        color='lightgray',
        edgecolor='k',
        zorder=1
    )
    # Error bar (SEM)
    plt.errorbar(
        i,
        np.mean(y),
        yerr=np.std(y) / np.sqrt(len(y)),
        color='k',
        capsize=2,
        lw=1,
        zorder=2
    )
    # Scatter with horizontal jitter
    jitter = 0.1 * (2 * np.random.rand(len(y)) - 1)
    plt.scatter(
        i + jitter,
        y,
        s=12,
        facecolors='none',
        edgecolors='k',
        linewidths=0.6,
        zorder=3
    )    
    # Red downward arrow
    if i in [4, 6, 10]:
        plt.annotate(
            '',
            xy=(i, 0.5),        # arrow head
            xytext=(i, 0.65),   # arrow tail
            arrowprops=dict(
                arrowstyle='-|>',
                color='red',
                lw=2.5
            )
        )    
plt.xticks(x, xlabels, rotation=45, ha='right')
plt.ylabel(r'$\Delta$ CD dot product')
plt.tight_layout()
plt.savefig(figpath + 'CDdp_error.pdf')

#%%
dict_trialtypes = {
    'P1': np.zeros((nfile,24)),
    'A1': np.zeros((nfile,24)),
    'P2': np.zeros((nfile,24)),
    'A2': np.zeros((nfile,24))
}
dict_topcells_2 = {
    'P2+A2-': copy.deepcopy(dict_trialtypes),
    'P2+A2+': copy.deepcopy(dict_trialtypes),
    'P2-A2+': copy.deepcopy(dict_trialtypes)
}
dict_population_activity = {
    'P1+A1-': copy.deepcopy(dict_topcells_2),
    'P1+A1+': copy.deepcopy(dict_topcells_2),
    'P1-A1+': copy.deepcopy(dict_topcells_2),
}

for fx in range(nfile):
    nonoutlier = fit_summary['nonoutlier'][fx]
    data_P1   = modelfit[f'sess{fx}']['P1']['data'][nonoutlier,:]
    data_A1   = modelfit[f'sess{fx}']['A1']['data'][nonoutlier,:]
    data_P2   = modelfit[f'sess{fx}']['P2']['data'][nonoutlier,:]
    data_A2   = modelfit[f'sess{fx}']['A2']['data'][nonoutlier,:]

    for key1 in ['P1+A1-', 'P1+A1+', 'P1-A1+']:
        for key2 in ['P2+A2-', 'P2+A2+', 'P2-A2+']:
            _topcells = dict_topcells_1x2[fx][key1][key2]
            _P1 = np.mean(data_P1[_topcells],axis=0)
            _A1 = np.mean(data_A1[_topcells],axis=0)
            _P2 = np.mean(data_P2[_topcells],axis=0)
            _A2 = np.mean(data_A2[_topcells],axis=0)
            dict_population_activity[key1][key2]['P1'][fx] = _P1
            dict_population_activity[key1][key2]['A1'][fx] = _A1
            dict_population_activity[key1][key2]['P2'][fx] = _P2
            dict_population_activity[key1][key2]['A2'][fx] = _A2


fig = plt.figure(figsize=(6, 8))
gs = fig.add_gridspec(
    6, 3,
    hspace=0.3,   # small spacing overall
    wspace=0.5
)
keys1 = ['P1+A1-', 'P1+A1+', 'P1-A1+']
keys2 = ['P2+A2-', 'P2+A2+', 'P2-A2+']
titles1 = [r'$P_1^+A_1^-$', r'$P_1^+A_1^+$', r'$P_1^-A_1^+$']
labels2 = [r'$P_2^+A_2^-$', r'$P_2^+A_2^+$', r'$P_2^-A_2^+$']
for i2, key2 in enumerate(keys2):
    for i1, key1 in enumerate(keys1):
        # top: context 1
        ax1 = fig.add_subplot(gs[2*i2, i1])
        ax1.plot(
            np.nanmean(dict_population_activity[key1][key2]['P1'], axis=0),
            c='purple'
        )
        ax1.plot(
            np.nanmean(dict_population_activity[key1][key2]['A1'], axis=0),
            c='limegreen'
        )
        ax1.axvline(7, color='gray', linestyle='--')
        ax1.axvline(15, color='gray', linestyle='--')
        ax1.set_ylim([0, 0.3])
        ax1.set_xticklabels([])

        # bottom: context 2
        ax2 = fig.add_subplot(gs[2*i2 + 1, i1], sharex=ax1)
        ax2.plot(
            np.nanmean(dict_population_activity[key1][key2]['P2'], axis=0),
            c='purple',
            linestyle='--'
        )
        ax2.plot(
            np.nanmean(dict_population_activity[key1][key2]['A2'], axis=0),
            c='limegreen',
            linestyle='--'
        )
        ax2.axvline(7, color='gray', linestyle='--')
        ax2.axvline(15, color='gray', linestyle='--')
        ax2.set_ylim([0, 0.3])

        # Titles
        if i2 == 0:
            ax1.set_title(titles1[i1], fontsize=12)
        # Row labels
        if i1 == 0:
            ax1.set_ylabel(labels2[i2], fontsize=12)            
        # annotate context
        if i1 == 0 and i2 == 0:
            ax1.annotate('Context1', xy=(0.45,0.8), xycoords='axes fraction')
            ax2.annotate('Context2', xy=(0.45,0.8), xycoords='axes fraction')
plt.tight_layout()
plt.savefig(figpath + 'neuron_group_traces.pdf')



tix_delay = np.arange(8,16)
fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(
    3, 3,
    hspace=0.3,   # small spacing overall
    wspace=0.5
)
bar_width = 0.6
keys1 = ['P1+A1-', 'P1+A1+', 'P1-A1+']
keys2 = ['P2+A2-', 'P2+A2+', 'P2-A2+']
titles1 = [r'$P_1^+A_1^-$', r'$P_1^+A_1^+$', r'$P_1^-A_1^+$']
labels2 = [r'$P_2^+A_2^-$', r'$P_2^+A_2^+$', r'$P_2^-A_2^+$']
for i2, key2 in enumerate(keys2):
    print('key2', key2)
    for i1, key1 in enumerate(keys1):
        # top: context 1
        print('key1', key1)
        
        diff1 = np.nanmean(dict_population_activity[key1][key2]['P1'][:,tix_delay] - dict_population_activity[key1][key2]['A1'][:,tix_delay], axis=1)
        diff2 = np.nanmean(dict_population_activity[key1][key2]['P2'][:,tix_delay] - dict_population_activity[key1][key2]['A2'][:,tix_delay], axis=1)
        diff1_mean = np.nanmean(diff1)
        diff2_mean = np.nanmean(diff2)        
        diff1_std  = np.nanstd(diff1)
        diff2_std  = np.nanstd(diff2)
        
        if diff1_mean > 0:
            c1 = 'purple'
        else:
            c1 = 'limegreen'
        if diff2_mean > 0:
            c2 = 'purple'
        else:
            c2 = 'limegreen'            
        
        ax1 = fig.add_subplot(gs[i2, i1])
        # Bar: mean
        plt.bar(
            [0,1],
            [diff1_mean,diff2_mean],
            width=bar_width,
            color=[c1,c2],
            edgecolor=[c1,c2],
            zorder=1,
            alpha=0.5
        )
        # Scatter with horizontal jitter
        jitter = 0.1 * (2 * np.random.rand(len(diff1)) - 1)
        plt.scatter(
            0 + jitter,
            diff1,
            s=12,
            facecolors='none',
            edgecolors=c1,
            linewidths=0.5,
            alpha=1
        )    
        plt.scatter(
            1 + jitter,
            diff2,
            s=12,
            facecolors='none',
            edgecolors=c2,
            linewidths=0.5,
            alpha=0.8
        )    
        # Error bar (SEM)
        plt.errorbar(
            [0,1],
            [diff1_mean,diff2_mean],
            yerr=[diff1_std / np.sqrt(len(diff1)), diff2_std / np.sqrt(len(diff2))],
            color='k',
            capsize=8,
            lw=1.5,
            linestyle='none'
        )
        # plt.axvspan(-0.5,0.5,color='gray',alpha=0.3)
        # plt.axvspan(0.5,1.5,color='tab:cyan',alpha=0.3)
        plt.xlim([-0.5,1.5])
        plt.ylim([-0.3,0.3])
        # Titles
        if i2 == 0:
            ax1.set_title(titles1[i1], fontsize=12)
        # Row labels
        if i1 == 0:
            ax1.set_ylabel(labels2[i2], fontsize=12)            
        if i2 == 2:
            ax1.set_xticks([0,1])
            ax1.set_xticklabels(['Context 1', 'Context 2'], rotation=45)
        else:
            ax1.set_xticks([0,1])
            ax1.set_xticklabels([])
plt.tight_layout()
plt.savefig(figpath + 'neuron_group_individuals_within_context.pdf')




tix_delay = np.arange(8,16)
fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(
    3, 3,
    hspace=0.3,   # small spacing overall
    wspace=0.5
)
bar_width = 0.6
keys1 = ['P1+A1-', 'P1+A1+', 'P1-A1+']
keys2 = ['P2+A2-', 'P2+A2+', 'P2-A2+']
titles1 = [r'$P_1^+A_1^-$', r'$P_1^+A_1^+$', r'$P_1^-A_1^+$']
labels2 = [r'$P_2^+A_2^-$', r'$P_2^+A_2^+$', r'$P_2^-A_2^+$']
for i2, key2 in enumerate(keys2):
    print('key2', key2)
    for i1, key1 in enumerate(keys1):
        # top: context 1
        print('key1', key1)
        
        diff1 = np.nanmean(dict_population_activity[key1][key2]['P2'][:,tix_delay] - dict_population_activity[key1][key2]['P1'][:,tix_delay], axis=1)
        diff2 = np.nanmean(dict_population_activity[key1][key2]['A2'][:,tix_delay] - dict_population_activity[key1][key2]['A1'][:,tix_delay], axis=1)
        diff1_mean = np.nanmean(diff1)
        diff2_mean = np.nanmean(diff2)        
        diff1_std  = np.nanstd(diff1)
        diff2_std  = np.nanstd(diff2)
        
        if diff1_mean > 0:
            c1 = 'tab:cyan'
        else:
            c1 = 'gray'
        if diff2_mean > 0:
            c2 = 'tab:cyan'
        else:
            c2 = 'gray'            
        
        ax1 = fig.add_subplot(gs[i2, i1])
        # Bar: mean
        plt.bar(
            [0,1],
            [diff1_mean,diff2_mean],
            width=bar_width,
            color=[c1,c2],
            edgecolor=[c1,c2],
            zorder=1,
            alpha=0.5
        )
        # Scatter with horizontal jitter
        jitter = 0.1 * (2 * np.random.rand(len(diff1)) - 1)
        plt.scatter(
            0 + jitter,
            diff1,
            s=12,
            facecolors='none',
            edgecolors=c1,
            linewidths=0.5,
            alpha=1
        )    
        plt.scatter(
            1 + jitter,
            diff2,
            s=12,
            facecolors='none',
            edgecolors=c2,
            linewidths=0.5,
            alpha=1
        )    
        # Error bar (SEM)
        plt.errorbar(
            [0,1],
            [diff1_mean,diff2_mean],
            yerr=[diff1_std / np.sqrt(len(diff1)), diff2_std / np.sqrt(len(diff2))],
            color='k',
            capsize=8,
            lw=1.5,
            linestyle='none'
        )
        # plt.axvspan(-0.5,0.5,color='purple',alpha=0.2)
        # plt.axvspan(0.5,1.5,color='limegreen',alpha=0.2)
        plt.xlim([-0.5,1.5])
        plt.ylim([-0.3,0.3])
        # Titles
        if i2 == 0:
            ax1.set_title(titles1[i1], fontsize=12)
        # Row labels
        if i1 == 0:
            ax1.set_ylabel(labels2[i2], fontsize=12)            
        if i2 == 2:
            ax1.set_xticks([0,1])
            ax1.set_xticklabels([r'$P_2-P_1$', r'$A_2-A_1$'], rotation=45)
        else:
            ax1.set_xticks([0,1])
            ax1.set_xticklabels([])
plt.tight_layout()
plt.savefig(figpath + 'neuron_group_individuals_cross_context.pdf')



#%%

tix_context1 = np.arange(8,16)
tix_context2 = np.arange(8,16)
activation_matrix1 = np.zeros((nfile,8,8))
activation_matrix2 = np.zeros((nfile,8,8))
activation_matrix3 = np.zeros((nfile,8,8))
for fx in range(nfile):        
    for ix1, tx1 in enumerate(tix_context1):
        _topcells_P1 = np.unique(dict_topcells['P1']['topcells_at_t'][fx][tx1])
        _topcells_A1 = np.unique(dict_topcells['A1']['topcells_at_t'][fx][tx1])
        _topcells_P1_A1 = _topcells_P1[~np.isin(_topcells_P1,_topcells_A1)]
        for ix2, tx2 in enumerate(tix_context2):
            _topcells_P2 = np.unique(dict_topcells['P2']['topcells_at_t'][fx][tx2])
            _topcells_A2 = np.unique(dict_topcells['A2']['topcells_at_t'][fx][tx2])
            _topcells_P2_A2 = _topcells_P2[~np.isin(_topcells_P2,_topcells_A2)]
            _topcells_P2A2  = _topcells_P2[ np.isin(_topcells_P2,_topcells_A2)]
            _topcells_A2_P2 = _topcells_A2[~np.isin(_topcells_A2,_topcells_P2)]
            activation_matrix1[fx,ix1,ix2] = np.sum(np.isin(_topcells_P1_A1,_topcells_P2_A2)) / len(fit_summary['nonoutlier'][fx])
            activation_matrix2[fx,ix1,ix2] = np.sum(np.isin(_topcells_P1_A1,_topcells_P2A2))  / len(fit_summary['nonoutlier'][fx])
            activation_matrix3[fx,ix1,ix2] = np.sum(np.isin(_topcells_P1_A1,_topcells_A2_P2)) / len(fit_summary['nonoutlier'][fx])
activation_matrix1_norm = activation_matrix1
activation_matrix2_norm = activation_matrix2
activation_matrix3_norm = activation_matrix3
activation_matrix1_avg = np.mean(activation_matrix1_norm,axis=0)
activation_matrix2_avg = np.mean(activation_matrix2_norm,axis=0)
activation_matrix3_avg = np.mean(activation_matrix3_norm,axis=0)


activation_matrix4 = np.zeros((nfile,8,8))
activation_matrix5 = np.zeros((nfile,8,8))
activation_matrix6 = np.zeros((nfile,8,8))
for fx in range(nfile):        
    for ix1, tx1 in enumerate(tix_context1):
        _topcells_P1 = np.unique(dict_topcells['P1']['topcells_at_t'][fx][tx1])
        _topcells_A1 = np.unique(dict_topcells['A1']['topcells_at_t'][fx][tx1])
        _topcells_A1_P1 = _topcells_A1[~np.isin(_topcells_A1,_topcells_P1)]
        for ix2, tx2 in enumerate(tix_context2):
            _topcells_P2 = np.unique(dict_topcells['P2']['topcells_at_t'][fx][tx2])
            _topcells_A2 = np.unique(dict_topcells['A2']['topcells_at_t'][fx][tx2])
            _topcells_P2_A2 = _topcells_P2[~np.isin(_topcells_P2,_topcells_A2)]
            _topcells_P2A2  = _topcells_P2[ np.isin(_topcells_P2,_topcells_A2)]
            _topcells_A2_P2 = _topcells_A2[~np.isin(_topcells_A2,_topcells_P2)]
            activation_matrix4[fx,ix1,ix2] = np.sum(np.isin(_topcells_A1_P1,_topcells_P2_A2)) / len(fit_summary['nonoutlier'][fx])
            activation_matrix5[fx,ix1,ix2] = np.sum(np.isin(_topcells_A1_P1,_topcells_P2A2))  / len(fit_summary['nonoutlier'][fx])
            activation_matrix6[fx,ix1,ix2] = np.sum(np.isin(_topcells_A1_P1,_topcells_A2_P2)) / len(fit_summary['nonoutlier'][fx])
activation_matrix4_norm = activation_matrix4
activation_matrix5_norm = activation_matrix5
activation_matrix6_norm = activation_matrix6
activation_matrix4_avg = np.mean(activation_matrix4_norm,axis=0)
activation_matrix5_avg = np.mean(activation_matrix5_norm,axis=0)
activation_matrix6_avg = np.mean(activation_matrix6_norm,axis=0)


activation_matrix7 = np.zeros((nfile,8,8))
activation_matrix8 = np.zeros((nfile,8,8))
activation_matrix9 = np.zeros((nfile,8,8))
for fx in range(nfile):        
    for ix1, tx1 in enumerate(tix_context1):
        _topcells_P1 = np.unique(dict_topcells['P1']['topcells_at_t'][fx][tx1])
        _topcells_A1 = np.unique(dict_topcells['A1']['topcells_at_t'][fx][tx1])
        _topcells_P1A1 = _topcells_P1[np.isin(_topcells_P1,_topcells_A1)]
        for ix2, tx2 in enumerate(tix_context2):
            _topcells_P2 = np.unique(dict_topcells['P2']['topcells_at_t'][fx][tx2])
            _topcells_A2 = np.unique(dict_topcells['A2']['topcells_at_t'][fx][tx2])
            _topcells_P2_A2 = _topcells_P2[~np.isin(_topcells_P2,_topcells_A2)]
            _topcells_P2A2  = _topcells_P2[ np.isin(_topcells_P2,_topcells_A2)]
            _topcells_A2_P2 = _topcells_A2[~np.isin(_topcells_A2,_topcells_P2)]
            activation_matrix7[fx,ix1,ix2] = np.sum(np.isin(_topcells_P1A1,_topcells_P2_A2)) / len(fit_summary['nonoutlier'][fx])
            activation_matrix8[fx,ix1,ix2] = np.sum(np.isin(_topcells_P1A1,_topcells_P2A2))  / len(fit_summary['nonoutlier'][fx])
            activation_matrix9[fx,ix1,ix2] = np.sum(np.isin(_topcells_P1A1,_topcells_A2_P2)) / len(fit_summary['nonoutlier'][fx])
activation_matrix7_norm = activation_matrix7
activation_matrix8_norm = activation_matrix8
activation_matrix9_norm = activation_matrix9
activation_matrix7_avg = np.nanmean(activation_matrix7_norm,axis=0)
activation_matrix8_avg = np.mean(activation_matrix8_norm,axis=0)
activation_matrix9_avg = np.mean(activation_matrix9_norm,axis=0)

#%%

idline = np.arange(8)
plt.figure(figsize=(3,6.5))
plt.subplot(311)
plt.imshow(activation_matrix1_avg,cmap='jet',aspect='auto')
plt.plot(idline,idline,color='w',linestyle='--')
plt.gca().invert_yaxis()
plt.colorbar()
plt.title(r'$P_1^+A_2^-$',fontsize=12)
plt.ylabel(r'$P_2^+A_2^-$',fontsize=12)
plt.subplot(312)
plt.imshow(activation_matrix2_avg,cmap='jet',aspect='auto')
plt.plot(idline,idline,color='w',linestyle='--')
plt.gca().invert_yaxis()
plt.colorbar()
plt.ylabel(r'$P_2^+A_2^+$',fontsize=12)
plt.subplot(313)
plt.imshow(activation_matrix3_avg,cmap='jet',aspect='auto')
plt.plot(idline,idline,color='w',linestyle='--')
plt.gca().invert_yaxis()
plt.colorbar()
plt.ylabel(r'$P_2^-A_2^+$',fontsize=12)
plt.tight_layout()
plt.savefig(figpath + 'P1+_A1-.pdf')


idline = np.arange(8)
plt.figure(figsize=(3,6.5))
plt.subplot(311)
plt.imshow(activation_matrix4_avg,cmap='jet',aspect='auto')
plt.plot(idline,idline,color='w',linestyle='--')
plt.gca().invert_yaxis()
plt.colorbar()
plt.title(r'$P_1^-A_2^+$',fontsize=12)
plt.ylabel(r'$P_2^+A_2^-$',fontsize=12)
plt.subplot(312)
plt.imshow(activation_matrix5_avg,cmap='jet',aspect='auto')
plt.plot(idline,idline,color='w',linestyle='--')
plt.gca().invert_yaxis()
plt.colorbar()
plt.ylabel(r'$P_2^+A_2^+$',fontsize=12)
plt.subplot(313)
plt.imshow(activation_matrix6_avg,cmap='jet',aspect='auto')
plt.plot(idline,idline,color='w',linestyle='--')
plt.gca().invert_yaxis()
plt.colorbar()
plt.ylabel(r'$P_2^-A_2^+$',fontsize=12)
plt.tight_layout()
plt.savefig(figpath + 'P1-_A1+.pdf')



idline = np.arange(8)
plt.figure(figsize=(3,6.5))
plt.subplot(311)
plt.imshow(activation_matrix7_avg,cmap='jet',aspect='auto')
plt.plot(idline,idline,color='w',linestyle='--')
plt.gca().invert_yaxis()
plt.colorbar()
plt.title(r'$P_1^+A_2^+$', fontsize=12)
plt.ylabel(r'$P_2^+A_2^-$', fontsize=12)
plt.subplot(312)
plt.imshow(activation_matrix8_avg,cmap='jet',aspect='auto')
plt.plot(idline,idline,color='w',linestyle='--')
plt.gca().invert_yaxis()
plt.colorbar()
plt.ylabel(r'$P_2^+A_2^+$', fontsize=12)
plt.subplot(313)
plt.imshow(activation_matrix9_avg,cmap='jet',aspect='auto')
plt.plot(idline,idline,color='w',linestyle='--')
plt.gca().invert_yaxis()
plt.colorbar()
plt.ylabel(r'$P_2^-A_2^+$', fontsize=12)
plt.tight_layout()
plt.savefig(figpath + 'P1+_A1+.pdf')


#%%

act1_context1 = np.mean(activation_matrix1_norm,axis=(0,2))
act2_context1 = np.mean(activation_matrix2_norm,axis=(0,2))
act3_context1 = np.mean(activation_matrix3_norm,axis=(0,2))
act1_context2 = np.mean(activation_matrix1_norm,axis=(0,1))
act2_context2 = np.mean(activation_matrix2_norm,axis=(0,1))
act3_context2 = np.mean(activation_matrix3_norm,axis=(0,1))

act4_context1 = np.mean(activation_matrix4_norm,axis=(0,2))
act5_context1 = np.mean(activation_matrix5_norm,axis=(0,2))
act6_context1 = np.mean(activation_matrix6_norm,axis=(0,2))
act4_context2 = np.mean(activation_matrix4_norm,axis=(0,1))
act5_context2 = np.mean(activation_matrix5_norm,axis=(0,1))
act6_context2 = np.mean(activation_matrix6_norm,axis=(0,1))

act7_context1 = np.nanmean(activation_matrix7_norm,axis=(0,2))
act8_context1 = np.mean(activation_matrix8_norm,axis=(0,2))
act9_context1 = np.mean(activation_matrix9_norm,axis=(0,2))
act7_context2 = np.nanmean(activation_matrix7_norm,axis=(0,1))
act8_context2 = np.mean(activation_matrix8_norm,axis=(0,1))
act9_context2 = np.mean(activation_matrix9_norm,axis=(0,1))


fig = plt.figure(figsize=(2.3, 9))
plt.subplot(611)
plt.title(r'$P_1^+A_1^-$', fontsize=12)
plt.ylabel(r'$P_2^+A_2^-$', fontsize=12)
plt.plot(act1_context1)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.subplot(612)
plt.plot(act1_context2)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.subplot(613)
plt.ylabel(r'$P_2^+A_2^+$', fontsize=12)
plt.plot(act2_context1, label=r'$P_2^+A_2^+$')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.subplot(614)
plt.plot(act2_context2)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.subplot(615)
plt.ylabel(r'$P_2^-A_2^+$', fontsize=12)
plt.plot(act3_context1, label=r'$P_2^-A_2^+$')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.subplot(616)
plt.plot(act3_context2)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.xlabel('time',fontsize=12)
plt.tight_layout()
plt.savefig(figpath + 'P1+_A1-_hist.pdf')


fig = plt.figure(figsize=(2.3, 9))
plt.subplot(611)
plt.title(r'$P_1^-A_1^+$', fontsize=12)
plt.ylabel(r'$P_2^+A_2^-$', fontsize=12)
plt.plot(act4_context1)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.subplot(612)
plt.plot(act4_context2)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.subplot(613)
plt.ylabel(r'$P_2^+A_2^+$', fontsize=12)
plt.plot(act5_context1, label=r'$P_2^+A_2^+$')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.subplot(614)
plt.plot(act5_context2)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.subplot(615)
plt.ylabel(r'$P_2^-A_2^+$', fontsize=12)
plt.plot(act6_context1, label=r'$P_2^-A_2^+$')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.subplot(616)
plt.plot(act6_context2)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.xlabel('time',fontsize=12)
plt.tight_layout()
plt.savefig(figpath + 'P1-_A1+_hist.pdf')




fig = plt.figure(figsize=(2.3, 9))
plt.subplot(611)
plt.title(r'$P_1^+A_1^+$', fontsize=12)
plt.ylabel(r'$P_2^+A_2^-$', fontsize=12)
plt.plot(act7_context1)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.subplot(612)
plt.plot(act7_context2)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.subplot(613)
plt.ylabel(r'$P_2^+A_2^+$', fontsize=12)
plt.plot(act8_context1, label=r'$P_2^+A_2^+$')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.subplot(614)
plt.plot(act8_context2)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.subplot(615)
plt.ylabel(r'$P_2^-A_2^+$', fontsize=12)
plt.plot(act9_context1, label=r'$P_2^-A_2^+$')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.subplot(616)
plt.plot(act9_context2)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.xlabel('time',fontsize=12)
plt.tight_layout()
plt.savefig(figpath + 'P1+_A1+_hist.pdf')


#%%

keys1 = ['P1+A1-', 'P1+A1+', 'P1-A1+']
keys2 = ['P2+A2-', 'P2+A2+', 'P2-A2+']
popact_P1 = np.zeros((nfile,3,3,8,24))
popact_A1 = np.zeros((nfile,3,3,8,24))
popact_P2 = np.zeros((nfile,3,3,8,24))
popact_A2 = np.zeros((nfile,3,3,8,24))

for i1, k1 in enumerate(keys1):
    for i2, k2 in enumerate(keys2):        
        for fx in range(nfile):
            _cells = dict_topcells_1x2[fx][k1][k2]
            nonoutlier = fit_summary['nonoutlier'][fx]
            data_P1   = modelfit[f'sess{fx}']['P1']['data'][nonoutlier,:]
            data_A1   = modelfit[f'sess{fx}']['A1']['data'][nonoutlier,:]
            data_P2   = modelfit[f'sess{fx}']['P2']['data'][nonoutlier,:]
            data_A2   = modelfit[f'sess{fx}']['A2']['data'][nonoutlier,:]
            for ix, tx in enumerate(tix_context1):
                _cells_P1_tx = _cells[np.isin(_cells,dict_topcells['P1']['topcells_at_t'][fx][tx])]
                _cells_A1_tx = _cells[np.isin(_cells,dict_topcells['A1']['topcells_at_t'][fx][tx])]
                _cells_P2_tx = _cells[np.isin(_cells,dict_topcells['P2']['topcells_at_t'][fx][tx])]
                _cells_A2_tx = _cells[np.isin(_cells,dict_topcells['A2']['topcells_at_t'][fx][tx])]
                if len(_cells_P1_tx) > 0:
                    popact_P1[fx,i2,i1,ix] = np.mean(data_P1[_cells_P1_tx],axis=0)
                if len(_cells_A1_tx) > 0:
                    popact_A1[fx,i2,i1,ix] = np.mean(data_A1[_cells_A1_tx],axis=0)
                if len(_cells_P2_tx) > 0:
                    popact_P2[fx,i2,i1,ix] = np.mean(data_P2[_cells_P2_tx],axis=0)
                if len(_cells_A2_tx) > 0:
                    popact_A2[fx,i2,i1,ix] = np.mean(data_A2[_cells_A2_tx],axis=0)                
mean_popact_P1 = np.mean(popact_P1,axis=0)
mean_popact_A1 = np.mean(popact_A1,axis=0)
mean_popact_P2 = np.mean(popact_P2,axis=0)
mean_popact_A2 = np.mean(popact_A2,axis=0)


for i1, k1 in enumerate(keys1):
    for i2, k2 in enumerate(keys2):        
        
        mean_popact_P1_i2_i1 = mean_popact_P1[i2,i1]
        mean_popact_A1_i2_i1 = mean_popact_A1[i2,i1]
        mean_popact_P2_i2_i1 = mean_popact_P2[i2,i1]
        mean_popact_A2_i2_i1 = mean_popact_A2[i2,i1]
        
        vmax = np.max([np.max(mean_popact_P1_i2_i1),np.max(mean_popact_A1_i2_i1),np.max(mean_popact_P2_i2_i1),np.max(mean_popact_A2_i2_i1)])

        plt.figure(figsize=(4.3,3.8))    
        plt.subplot(221)
        plt.title(r'$P_1$',fontsize=12)
        plt.imshow(mean_popact_P1_i2_i1,cmap='jet',aspect='auto',vmin=0,vmax=vmax)
        plt.axvline(7,color='w',linestyle='--')
        plt.axvline(15,color='w',linestyle='--')
        plt.colorbar()
        plt.subplot(222)
        plt.title(r'$A_1$',fontsize=12)
        plt.imshow(mean_popact_A1_i2_i1,cmap='jet',aspect='auto',vmin=0,vmax=vmax)
        plt.axvline(7,color='w',linestyle='--')
        plt.axvline(15,color='w',linestyle='--')
        plt.colorbar()
        plt.subplot(223)
        plt.title(r'$P_2$',fontsize=12)
        plt.imshow(mean_popact_P2_i2_i1,cmap='jet',aspect='auto',vmin=0,vmax=vmax)
        plt.axvline(7,color='w',linestyle='--')
        plt.axvline(15,color='w',linestyle='--')
        plt.colorbar()
        plt.subplot(224)
        plt.title(r'$A_2$',fontsize=12)
        plt.imshow(mean_popact_A2_i2_i1,cmap='jet',aspect='auto',vmin=0,vmax=vmax)
        plt.axvline(7,color='w',linestyle='--')
        plt.axvline(15,color='w',linestyle='--')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(figpath + f'sequential/map_{i2}_{i1}_' + k2 + '_' + k1 + '.pdf')
        

        cmap = plt.cm.copper
        colors = cmap(np.linspace(0, 1, 8))        
        plt.figure(figsize=(3.3,3))
        plt.subplot(221)
        plt.title(r'$P_1$',fontsize=12)
        for i in range(8):
            plt.plot(mean_popact_P1_i2_i1[i], color=colors[i])
        plt.axvline(7,color='gray',linestyle='--')
        plt.axvline(15,color='gray',linestyle='--')
        plt.ylim([0,vmax])
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.subplot(222)
        plt.title(r'$A_1$',fontsize=12)
        for i in range(8):
            plt.plot(mean_popact_A1_i2_i1[i], color=colors[i])
        plt.axvline(7,color='gray',linestyle='--')
        plt.axvline(15,color='gray',linestyle='--')
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.ylim([0,vmax])
        plt.subplot(223)
        plt.title(r'$P_2$',fontsize=12)
        for i in range(8):
            plt.plot(mean_popact_P2_i2_i1[i], color=colors[i])
        plt.axvline(7,color='gray',linestyle='--')
        plt.axvline(15,color='gray',linestyle='--')
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.ylim([0,vmax])
        plt.subplot(224)
        plt.title(r'$A_2$',fontsize=12)
        for i in range(8):
            plt.plot(mean_popact_A2_i2_i1[i], color=colors[i])
        plt.axvline(7,color='gray',linestyle='--')
        plt.axvline(15,color='gray',linestyle='--')
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.ylim([0,vmax])
        plt.tight_layout()
        plt.savefig(figpath + f'sequential/trace_{i2}_{i1}_' + k2 + '_' + k1 + '.pdf')

#%%
def sort_data(x):
    tmax   = np.argmax(x[:,8:16],axis=1)
    cells_sorted = np.argsort(tmax)
    x_sorted = x[cells_sorted]
    return x_sorted

def find_tmax(x):
    tmax   = np.argmax(x[:,8:16],axis=1)
    return tmax

keys1 = ['P1+A1-', 'P1+A1+', 'P1-A1+']
keys2 = ['P2+A2-', 'P2+A2+', 'P2-A2+']
keys3 = ['P1','A1','P2','A2']
mean_activity_A1 = np.zeros((nfile,6,8))
mean_activity_P2 = np.zeros((nfile,6,8))

dict_trialtypes = {
    'P1': np.zeros((nfile,8)),
    'A1': np.zeros((nfile,8)),
    'P2': np.zeros((nfile,8)),
    'A2': np.zeros((nfile,8))
}
dict_topcells_2 = {
    'P2+A2-': copy.deepcopy(dict_trialtypes),
    'P2+A2+': copy.deepcopy(dict_trialtypes),
    'P2-A2+': copy.deepcopy(dict_trialtypes)
}
dict_mean_activity = {
    'P1+A1-': copy.deepcopy(dict_topcells_2),
    'P1+A1+': copy.deepcopy(dict_topcells_2),
    'P1-A1+': copy.deepcopy(dict_topcells_2),
}

for fx in range(nfile):        
    nonoutlier = fit_summary['nonoutlier'][fx]
    for k1 in keys1:
        for k2 in keys2:
            for k3 in keys3:                
                _data_k3 = modelfit[f'sess{fx}'][k3]['data'][nonoutlier,:]
                dict_mean_activity[k1][k2][k3][fx] = np.sum(_data_k3[dict_topcells_1x2[fx][k1][k2]],axis=0)[8:16]
                

dict_norm_activity = {
    'P1+A1-': copy.deepcopy(dict_topcells_2),
    'P1+A1+': copy.deepcopy(dict_topcells_2),
    'P1-A1+': copy.deepcopy(dict_topcells_2),
}
for fx in range(nfile):        
    for k3 in keys3:                
        _total_activity = np.zeros(8)
        for k1 in keys1:
            for k2 in keys2:
                _total_activity += dict_mean_activity[k1][k2][k3][fx]
        
        for k1 in keys1:
            for k2 in keys2:
                dict_norm_activity[k1][k2][k3][fx] = dict_mean_activity[k1][k2][k3][fx] / _total_activity
                
cmap = plt.cm.copper
colors = cmap(np.linspace(0, 1, nfile))
plt.figure(figsize=(8,8))
for i1, k1 in enumerate(keys1):
    for i2, k2 in enumerate(keys2):        
        plt.subplot(3,3,3*i2+i1+1)
        for fx in range(nfile):
            plt.plot(dict_norm_activity[k1][k2]['A1'][fx], color=colors[fx])
        plt.plot(np.mean(dict_norm_activity[k1][k2]['A1'],axis=0),lw=4)
        plt.ylim([0,0.6])
plt.tight_layout()

plt.figure(figsize=(8,8))
for i1, k1 in enumerate(keys1):
    for i2, k2 in enumerate(keys2):        
        plt.subplot(3,3,3*i2+i1+1)
        for fx in range(nfile):
            plt.plot(dict_norm_activity[k1][k2]['P2'][fx], color=colors[fx])
        plt.plot(np.mean(dict_norm_activity[k1][k2]['P2'],axis=0),lw=4)
        plt.ylim([0,0.6])
plt.tight_layout()


####### within context ########

# linear regression
dict_trialtypes = {
    'P1-A1': np.zeros((nfile,3)),
    'P2-A2': np.zeros((nfile,3)),
}
dict_topcells_2 = {
    'P2+A2-': copy.deepcopy(dict_trialtypes),
    'P2+A2+': copy.deepcopy(dict_trialtypes),
    'P2-A2+': copy.deepcopy(dict_trialtypes)
}
dict_linfit_within = {
    'P1+A1-': copy.deepcopy(dict_topcells_2),
    'P1+A1+': copy.deepcopy(dict_topcells_2),
    'P1-A1+': copy.deepcopy(dict_topcells_2),
}

keys_context = ['P1-A1','P2-A2']
linreg = linear_model.LinearRegression(fit_intercept=True)
for k1 in keys1:
    for k2 in keys2:
        for kc in keys_context:
            for fx in range(nfile):
                if kc == 'P1-A1':
                    _act  = dict_norm_activity[k1][k2]['P1'][fx] - dict_norm_activity[k1][k2]['A1'][fx]
                if kc == 'P2-A2':
                    _act  = dict_norm_activity[k1][k2]['P2'][fx] - dict_norm_activity[k1][k2]['A2'][fx]
                _mean = np.mean(_act)
                linreg.fit(np.arange(8).reshape(-1,1),_act)
                dict_linfit_within[k1][k2][kc][fx] = np.array([linreg.coef_.item(), linreg.intercept_.item(), _mean])

keys1 = ['P1+A1-', 'P1+A1+', 'P1-A1+']
keys2 = ['P2+A2-', 'P2+A2+', 'P2-A2+']
keys_context = ['P1-A1','P2-A2']

fig = plt.figure(figsize=(10, 6))

# 3 x 3 groups
outer = fig.add_gridspec(3, 3, wspace=0.35, hspace=0.4)

for i1, k1 in enumerate(keys1):
    for i2, k2 in enumerate(keys2):

        # two contexts placed close together
        inner = outer[i2, i1].subgridspec(1, 2, wspace=0.08)

        y1 = dict_linfit_within[k1][k2]['P1-A1'][:, 2]
        y2 = dict_linfit_within[k1][k2]['P2-A2'][:, 2]

        # common y range for direct comparison
        ymin = np.nanmin([np.nanmin(y1), np.nanmin(y2)]) - 0.05
        ymax = np.nanmax([np.nanmax(y1), np.nanmax(y2)]) + 0.05

        for ic, kc in enumerate(keys_context):

            ax = fig.add_subplot(inner[0, ic])
            ax.spines[['top', 'right']].set_visible(False)
            y = dict_linfit_within[k1][k2][kc][:, 2]
            ax.axhline(0,color='gray',linestyle='--')
            ax.scatter(
                CDdotproduct,
                y,
                facecolor='none',
                edgecolors='k'
            )

            _cor = np.corrcoef(CDdotproduct, y)[0, 1]

            ax.set_ylim([ymin, ymax])
            

            if ic == 0:
                ax.set_title(
                    # k1 + '\n' + r'$P_1-A_1$' +
                    # '\n' + 
                    r'$r=$' + str(np.round(_cor, 3)),
                    fontsize=9
                )

                # if i1 == 0:
                #     ax.set_ylabel(k2)
                None
            else:
                ax.set_title(
                    # r'$P_2-A_2$' +
                    # '\n' + 
                    r'$r=$' + str(np.round(_cor, 3)),
                    fontsize=9
                )

                # remove duplicate y labels/ticks
                ax.set_yticklabels([])

fig.tight_layout()
plt.savefig(figpath + 'CDdotprod_with_context.pdf')


####### across context ########

dict_trialtypes = {
    'P2-P1': np.zeros((nfile,3)),
    'A2-A1': np.zeros((nfile,3)),
}
dict_topcells_2 = {
    'P2+A2-': copy.deepcopy(dict_trialtypes),
    'P2+A2+': copy.deepcopy(dict_trialtypes),
    'P2-A2+': copy.deepcopy(dict_trialtypes)
}
dict_linfit_across = {
    'P1+A1-': copy.deepcopy(dict_topcells_2),
    'P1+A1+': copy.deepcopy(dict_topcells_2),
    'P1-A1+': copy.deepcopy(dict_topcells_2),
}
keys_context = ['P2-P1','A2-A1']
linreg = linear_model.LinearRegression(fit_intercept=True)
for k1 in keys1:
    for k2 in keys2:
        for kc in keys_context:
            for fx in range(nfile):
                if kc == 'P2-P1':
                    _act  = dict_norm_activity[k1][k2]['P2'][fx] - dict_norm_activity[k1][k2]['P1'][fx]
                if kc == 'A2-A1':
                    _act  = dict_norm_activity[k1][k2]['A2'][fx] - dict_norm_activity[k1][k2]['A1'][fx]
                _mean = np.mean(_act)
                linreg.fit(np.arange(8).reshape(-1,1),_act)
                dict_linfit_across[k1][k2][kc][fx] = np.array([linreg.coef_.item(), linreg.intercept_.item(), _mean])

keys1 = ['P1+A1-', 'P1+A1+', 'P1-A1+']
keys2 = ['P2+A2-', 'P2+A2+', 'P2-A2+']

fig = plt.figure(figsize=(10, 6))

# 3 x 3 groups
outer = fig.add_gridspec(3, 3, wspace=0.35, hspace=0.4)

for i1, k1 in enumerate(keys1):
    for i2, k2 in enumerate(keys2):

        # two contexts placed close together
        inner = outer[i2, i1].subgridspec(1, 2, wspace=0.08)

        y1 = dict_linfit_across[k1][k2]['P2-P1'][:, 2]
        y2 = dict_linfit_across[k1][k2]['A2-A1'][:, 2]

        # common y range for direct comparison
        ymin = np.nanmin([np.nanmin(y1), np.nanmin(y2)]) - 0.05
        ymax = np.nanmax([np.nanmax(y1), np.nanmax(y2)]) + 0.05

        for ic, kc in enumerate(keys_context):

            ax = fig.add_subplot(inner[0, ic])
            ax.spines[['top', 'right']].set_visible(False)
            y = dict_linfit_across[k1][k2][kc][:, 2]
            ax.axhline(0,color='gray',linestyle='--')
            ax.scatter(
                CDdotproduct,
                y,
                facecolor='none',
                edgecolors='k'
            )

            _cor = np.corrcoef(CDdotproduct, y)[0, 1]

            ax.set_ylim([ymin, ymax])
            

            if ic == 0:
                ax.set_title(
                    # k1 + '\n' + r'$P_1-A_1$' +
                    # '\n' + 
                    r'$r=$' + str(np.round(_cor, 3)),
                    fontsize=9
                )

                # if i1 == 0:
                #     ax.set_ylabel(k2)
                None
            else:
                ax.set_title(
                    # r'$P_2-A_2$' +
                    # '\n' + 
                    r'$r=$' + str(np.round(_cor, 3)),
                    fontsize=9
                )

                # remove duplicate y labels/ticks
                ax.set_yticklabels([])

fig.tight_layout()
plt.savefig(figpath + 'CDdotprod_across_context.pdf')



#%%
#-----------------------------------------

_mean_activity_A1 = np.mean(mean_activity_A1_norm, axis=0)  # shape: (6, time)
_mean_activity_P2 = np.mean(mean_activity_P2_norm, axis=0)  # shape: (6, time)
plt.figure(figsize=(4,4))
plt.stackplot(np.arange(_mean_activity_A1.shape[1]),_mean_activity_A1,labels=[str(i+1) for i in range(6)],edgecolor='k',linewidth=0.5)
plt.ylim([0, 1])
plt.legend()
plt.tight_layout()

plt.figure(figsize=(4,4))
plt.stackplot(np.arange(_mean_activity_P2.shape[1]),_mean_activity_P2,labels=[str(i+1) for i in range(6)],edgecolor='k',linewidth=0.5)
plt.ylim([0, 1])
plt.legend()
plt.tight_layout()




mean_activity_A1_avg = np.mean(mean_activity_A1_norm,axis=2)
mean_activity_P2_avg = np.mean(mean_activity_P2_norm,axis=2)

plt.figure(figsize=(9,6))
for grp in range(6):
    plt.subplot(2,3,grp+1)  
    plt.scatter(CDdotproduct,mean_activity_A1_avg[:,grp],color=f'C{grp}')
    _cor = np.corrcoef(CDdotproduct,mean_activity_A1_avg[:,grp])[0,1]
    plt.title(str(np.round(_cor,decimals=3)))
plt.tight_layout()

plt.figure(figsize=(9,6))
for grp in range(6):
    plt.subplot(2,3,grp+1)  
    plt.scatter(CDdotproduct,mean_activity_A1_avg[:,grp])
    _cor = np.corrcoef(CDdotproduct,mean_activity_P2_avg[:,grp])[0,1]
    plt.title(str(np.round(_cor,decimals=3)))
plt.tight_layout()





# plt.figure(figsize=(10,10))
# for fx in range(nfile):
#     plt.subplot(6,6,fx+1)
#     # for i in range(6):        
#     plt.plot(mean_activity_A1_norm[fx,grp], label=str(i))
#     plt.ylim([0,0.5])
# plt.legend()
# plt.tight_layout()


# plt.figure(figsize=(4,8))    
# for i in range(6):
#     plt.subplot(6,1,i+1)
#     plt.hist(tmax_A1_dist[i],bins=8,range=(0,8),histtype='step',density=True)
#     plt.ylim([0,0.4])
# plt.tight_layout()


# plt.figure(figsize=(6,8))
# plt.imshow(data_A1_sorted,cmap='jet',aspect='auto',vmin=0,vmax=0.1)
# plt.axhline(n0, color='w', linestyle='-')
# plt.axhline(n0+n1, color='w', linestyle='-')
# plt.axhline(n0+n1+n2, color='w', linestyle='-')
# plt.axhline(n0+n1+n2+n3, color='r', linestyle='-')
# plt.axhline(n0+n1+n2+n3+n4, color='r', linestyle='-')
# # plt.axhline(n0+n1+n2+n3+n4+n5, color='k', linestyle='-')
# plt.tight_layout()



#%%

frac_1x2_outof_totalcells = np.zeros(nfile)
frac_1x2_outof_topcells_1   = np.zeros(nfile)
CD1err = np.zeros(nfile)
CD2err = np.zeros(nfile)
CDdp = np.zeros(nfile)
CDdp_1x2 = np.zeros(nfile)
CDdp_1_2 = np.zeros(nfile)
CDdp_2_1 = np.zeros(nfile)
tix_latedelay = np.arange(12,16)
for fx in range(nfile):
    # topcells
    topcells_P1 = np.concatenate([dict_topcells['P1']['topcells_at_t'][fx][i] for i in tix_latedelay])
    topcells_A1 = np.concatenate([dict_topcells['A1']['topcells_at_t'][fx][i] for i in tix_latedelay])
    topcells_P2 = np.concatenate([dict_topcells['P2']['topcells_at_t'][fx][i] for i in tix_latedelay])
    topcells_A2 = np.concatenate([dict_topcells['A2']['topcells_at_t'][fx][i] for i in tix_latedelay])

    # shared & nonshared topcells
    topcells_1 = np.unique(np.concatenate((topcells_P1,topcells_A1)))
    topcells_2 = np.unique(np.concatenate((topcells_P2,topcells_A2)))
    topcells_1x2 = topcells_1[np.isin(topcells_1,topcells_2)]
    topcells_1_2 = topcells_1[~np.isin(topcells_1,topcells_2)]
    topcells_2_1 = topcells_2[~np.isin(topcells_2,topcells_1)]
    
    # compute CD1, CD2
    nonoutlier = fit_summary['nonoutlier'][fx]
    data_P1   = modelfit[f'sess{fx}']['P1']['data'][nonoutlier,:]
    data_A1   = modelfit[f'sess{fx}']['A1']['data'][nonoutlier,:]
    data_P2   = modelfit[f'sess{fx}']['P2']['data'][nonoutlier,:]
    data_A2   = modelfit[f'sess{fx}']['A2']['data'][nonoutlier,:]
    diff1 = np.mean((data_P1 - data_A1)[:,12:16],axis=1)
    diff2 = np.mean((data_P2 - data_A2)[:,12:16],axis=1)
    CD1 = diff1 / np.linalg.norm(diff1)
    CD2 = diff2 / np.linalg.norm(diff2)

    # number of topcells shared across contexts
    ncells = len(nonoutlier)
    frac_1x2_outof_totalcells[fx] = len(topcells_1x2)/ncells    
    frac_1x2_outof_topcells_1[fx] = len(topcells_1x2)/len(topcells_1)

    # approximate CD1 and CD2
    CD1aprx = np.zeros_like(CD1)
    CD2aprx = np.zeros_like(CD2)
    CD1aprx[topcells_1x2] = CD1[topcells_1x2]
    CD2aprx[topcells_1x2] = CD2[topcells_1x2]
    CD1err[fx] = np.linalg.norm(CD1aprx - CD1)
    CD2err[fx] = np.linalg.norm(CD2aprx - CD2)
    
    # approximate CD1CD2
    CDdp[fx] = np.inner(CD1,CD2)
    CDdp_1x2[fx] = np.inner(CD1[topcells_1x2],CD2[topcells_1x2])
    CDdp_1_2[fx] = np.inner(CD1[topcells_1_2],CD2[topcells_1_2])
    CDdp_2_1[fx] = np.inner(CD1[topcells_2_1],CD2[topcells_2_1])
    

#%%
def events():
    plt.axvline(8,color='gray',linestyle='--')
    plt.axvline(16,color='gray',linestyle='--')
    
#%%
'''
(1) Fraction of active neurons shared across Context 1 and Context 2
(2) CD1CD2 can be explained by the shared active neurons
'''

plt.figure()
plt.subplot(311)
plt.title('active neurons shared across contexts')
plt.plot(frac_1x2_outof_totalcells, label='total neurons')
plt.plot(frac_1x2_outof_topcells_1, label='context1 active')
plt.axhline(0.2,color='gray',linestyle='--')
plt.axhline(0.5,color='gray',linestyle='--')
plt.ylabel('frac of\nshared neurons')
plt.legend(title='out of')
plt.subplot(312)
plt.plot(CD1err, label='CD1')
plt.plot(CD2err, label='CD2')
plt.axhline(0.5,color='gray',linestyle='--')
plt.ylabel('CD error')
plt.legend()
plt.subplot(313)
plt.plot(CDdp, label='all neurons')
plt.plot(CDdp_1x2, c='k', label='shared neurons')
# plt.plot(CDdp_1_2, c='r', label='removed')
# plt.plot(CDdp_2_1, c='b', label='emerged')
plt.ylabel('CD1CD2')
plt.xlabel('session')
plt.legend()
plt.tight_layout()


#%%

tix_earlydelay = np.arange(9,13)
ntime = 24
time_slice = np.arange(ntime)

removed = np.zeros((nfile,3))
shared = np.zeros((nfile,3,3))
emerged = np.zeros((nfile,3))
NumCell = np.random.rand(nfile,4,4)
Smatrix = np.zeros((nfile,2,2))

dict_S = {
    'S1+':np.zeros(nfile),
    'S1-':np.zeros(nfile),
    'S2+':np.zeros(nfile),
    'S2-':np.zeros(nfile),
    'S1+S2+':np.zeros(nfile),
    'S1+S2-':np.zeros(nfile),
    'S1-S2+':np.zeros(nfile),
    'S1-S2-':np.zeros(nfile)    
    }
dict_topcells1x2 = {
    'P1+A1-':[],
    'P1+A1+':[],
    'P1-A1+':[],
    'P2+A2-':[],
    'P2+A2+':[],
    'P2-A2+':[]
}
S1_pos = np.zeros(nfile)
S1_neg = np.zeros(nfile)
S2_pos = np.zeros(nfile)
S2_neg = np.zeros(nfile)
S1_pos_S2_pos = np.zeros(nfile)
S1_pos_S2_neg = np.zeros(nfile)
S1_neg_S2_pos = np.zeros(nfile)
S1_neg_S2_neg = np.zeros(nfile)


# for tx in range(ntime):        
for fx in range(nfile):
                
    topcells_P1 = np.unique(np.concatenate([dict_topcells['P1']['topcells_at_t'][fx][i] for i in tix_earlydelay]))
    topcells_A1 = np.unique(np.concatenate([dict_topcells['A1']['topcells_at_t'][fx][i] for i in tix_earlydelay]))
    topcells_P2 = np.unique(np.concatenate([dict_topcells['P2']['topcells_at_t'][fx][i] for i in tix_earlydelay]))
    topcells_A2 = np.unique(np.concatenate([dict_topcells['A2']['topcells_at_t'][fx][i] for i in tix_earlydelay]))
    topcells_1 = np.unique(np.concatenate((topcells_P1,topcells_A1)))
    topcells_2 = np.unique(np.concatenate((topcells_P2,topcells_A2)))
    topcells_1_2 = topcells_1[~np.isin(topcells_1,topcells_2)]
    topcells_1x2 = topcells_1[np.isin(topcells_1,topcells_2)]
    topcells_2_1 = topcells_2[~np.isin(topcells_2,topcells_1)]
    N1 = len(topcells_1)
    N2 = len(topcells_2)

    P1_1_2 = np.isin(topcells_1_2,topcells_P1)
    A1_1_2 = np.isin(topcells_1_2,topcells_A1)
    topcells_1_2_P1A1_ = topcells_1_2[P1_1_2  & ~A1_1_2]
    topcells_1_2_P1A1  = topcells_1_2[P1_1_2  & A1_1_2]
    topcells_1_2_P1_A1 = topcells_1_2[~P1_1_2 & A1_1_2]

    P1_1x2 = np.isin(topcells_1x2, topcells_P1)
    A1_1x2 = np.isin(topcells_1x2, topcells_A1)
    P2_1x2 = np.isin(topcells_1x2, topcells_P2)
    A2_1x2 = np.isin(topcells_1x2, topcells_A2)
    topcells_1x2_P1A1_P2A2_ = topcells_1x2[P1_1x2 & ~A1_1x2 & P2_1x2  & ~A2_1x2]
    topcells_1x2_P1A1_P2A2  = topcells_1x2[P1_1x2 & ~A1_1x2 & P2_1x2  &  A2_1x2]
    topcells_1x2_P1A1_P2_A2 = topcells_1x2[P1_1x2 & ~A1_1x2 & ~P2_1x2 &  A2_1x2]
    topcells_1x2_P1A1P2A2_  = topcells_1x2[P1_1x2 & A1_1x2 & P2_1x2  & ~A2_1x2]
    topcells_1x2_P1A1P2A2   = topcells_1x2[P1_1x2 & A1_1x2 & P2_1x2  &  A2_1x2]
    topcells_1x2_P1A1P2_A2  = topcells_1x2[P1_1x2 & A1_1x2 & ~P2_1x2 &  A2_1x2]
    topcells_1x2_P1_A1P2A2_ = topcells_1x2[~P1_1x2 & A1_1x2 & P2_1x2  & ~A2_1x2]
    topcells_1x2_P1_A1P2A2  = topcells_1x2[~P1_1x2 & A1_1x2 & P2_1x2  &  A2_1x2]
    topcells_1x2_P1_A1P2_A2 = topcells_1x2[~P1_1x2 & A1_1x2 & ~P2_1x2 &  A2_1x2]

    P2_2_1 = np.isin(topcells_2_1,topcells_P2)
    A2_2_1 = np.isin(topcells_2_1,topcells_A2)
    topcells_2_1_P1A1_ = topcells_2_1[P2_2_1  & ~A2_2_1]
    topcells_2_1_P1A1  = topcells_2_1[P2_2_1  & A2_2_1]
    topcells_2_1_P1_A1 = topcells_2_1[~P2_2_1 & A2_2_1]

    _removed = np.array([len(topcells_1_2_P1A1_)/N1, 
                         len(topcells_1_2_P1A1)/N1,
                         len(topcells_1_2_P1_A1)/N1])

    _shared = np.array([[len(topcells_1x2_P1A1_P2A2_)/N1, len(topcells_1x2_P1A1P2A2_)/N1, len(topcells_1x2_P1_A1P2A2_)/N1],
                        [len(topcells_1x2_P1A1_P2A2)/N1,  len(topcells_1x2_P1A1P2A2)/N1,  len(topcells_1x2_P1_A1P2A2)/N1],
                        [len(topcells_1x2_P1A1_P2_A2)/N1, len(topcells_1x2_P1A1P2_A2)/N1, len(topcells_1x2_P1_A1P2_A2)/N1]])

    _emerged = np.array([len(topcells_2_1_P1A1_)/N2,
                         len(topcells_2_1_P1A1)/N2,
                         len(topcells_2_1_P1_A1)/N2])

    nonoutlier = fit_summary['nonoutlier'][fx]
    data_P1_fx = modelfit[f'sess{fx}']['P1']['data'][nonoutlier]
    data_P2_fx = modelfit[f'sess{fx}']['P2']['data'][nonoutlier]
    data_A1_fx = modelfit[f'sess{fx}']['A1']['data'][nonoutlier]
    data_A2_fx = modelfit[f'sess{fx}']['A2']['data'][nonoutlier]

    S1 = np.mean((data_P1_fx - data_A1_fx)[:,12:16],axis=1)
    S2 = np.mean((data_P2_fx - data_A2_fx)[:,12:16],axis=1)
    S1 = S1 / np.linalg.norm(S1)
    S2 = S2 / np.linalg.norm(S2)
    dS = S1*S2

    idx_S1_pos = S1 > 0
    idx_S1_neg = S1 < 0
    idx_S2_pos = S2 > 0
    idx_S2_neg = S2 < 0
    S1_pos = np.sum(S1[idx_S1_pos])
    S1_neg = np.sum(S1[idx_S1_neg])
    S2_pos = np.sum(S2[idx_S2_pos])
    S2_neg = np.sum(S2[idx_S2_neg])
    S1_pos_S2_pos = np.sum((S1*S2)[idx_S1_pos*idx_S2_pos])
    S1_pos_S2_neg = np.sum((S1*S2)[idx_S1_pos*idx_S2_neg])
    S1_neg_S2_pos = np.sum((S1*S2)[idx_S1_neg*idx_S2_pos])
    S1_neg_S2_neg = np.sum((S1*S2)[idx_S1_neg*idx_S2_neg])
    _Smatrix = np.array([[S1_pos_S2_pos, S1_neg_S2_pos],
                         [S1_pos_S2_neg, S1_neg_S2_neg]])

    removed[fx] = _removed
    shared[fx] = _shared
    emerged[fx] = _emerged
    NumCell[fx,:3,:3] = _shared
    NumCell[fx,3,:3] = _emerged
    NumCell[fx,:3,3] = _removed
    Smatrix[fx] = _Smatrix
    
    dict_S['S1+'][fx] = S1_pos
    dict_S['S1-'][fx] = S1_neg
    dict_S['S2+'][fx] = S2_pos
    dict_S['S2-'][fx] = S2_neg
    dict_S['S1+S2+'][fx] = S1_pos_S2_pos
    dict_S['S1+S2-'][fx] = S1_pos_S2_neg
    dict_S['S1-S2+'][fx] = S1_neg_S2_pos
    dict_S['S1-S2-'][fx] = S1_neg_S2_neg

#%%

NumCell_Sel = {
    'S1+':np.zeros((4,4)),
    'S1-':np.zeros((4,4)),
    'S2+':np.zeros((4,4)),
    'S2-':np.zeros((4,4)),
    'S1+S2+':np.zeros((4,4)),
    'S1+S2-':np.zeros((4,4)),
    'S1-S2+':np.zeros((4,4)),
    'S1-S2-':np.zeros((4,4)),
    'S1S2':np.zeros((4,4))
}
Numcell_S1pos = np.zeros((4,4))
Numcell_S1neg = np.zeros((4,4))
Numcell_S2pos = np.zeros((4,4))
Numcell_S2neg = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        NumCell_Sel['S1+'][i,j] = np.corrcoef((dict_S['S1+'],NumCell[:,i,j]))[0,1]
        NumCell_Sel['S1-'][i,j] = np.corrcoef((dict_S['S1-'],NumCell[:,i,j]))[0,1]
        NumCell_Sel['S2+'][i,j] = np.corrcoef((dict_S['S2+'],NumCell[:,i,j]))[0,1]
        NumCell_Sel['S2-'][i,j] = np.corrcoef((dict_S['S2-'],NumCell[:,i,j]))[0,1]        
        NumCell_Sel['S1+S2+'][i,j] = np.corrcoef((dict_S['S1+S2+'],NumCell[:,i,j]))[0,1]
        NumCell_Sel['S1+S2-'][i,j] = np.corrcoef((dict_S['S1+S2-'],NumCell[:,i,j]))[0,1]
        NumCell_Sel['S1-S2+'][i,j] = np.corrcoef((dict_S['S1-S2+'],NumCell[:,i,j]))[0,1]
        NumCell_Sel['S1-S2-'][i,j] = np.corrcoef((dict_S['S1-S2-'],NumCell[:,i,j]))[0,1]
        NumCell_Sel['S1S2'][i,j] = np.corrcoef((CDdotproduct,NumCell[:,i,j]))[0,1]

idx1 = {
    'P1+A1-':0,
    'P1+A1+':1,
    'P1-A1+':2
    }
idx2 = {
    'P2+A2-':0,
    'P2+A2+':1,
    'P2-A2+':2
    }

colors = (['tab:blue'] * 4 +
          ['tab:orange'] * 4 +
          ['tab:green'] * 4 +
          ['tab:red'] * 4)


selectivity_type = 'S1S2'

NumCell_Sel_flat = NumCell_Sel[selectivity_type].flatten(order='F')
plt.figure(figsize=(6,3))
plt.bar(np.arange(16), 
        NumCell_Sel_flat,
        width=0.2, alpha=0.3, color=colors)
plt.title(selectivity_type)
ax = plt.gca()
# Bottom labels
p2_labels = ['P2+A2-', 'P2+A2+', 'P2-A2+', 'P2-A2-'] * 4
ax.set_xticks(np.arange(16))
ax.set_xticklabels(p2_labels, rotation=45)

# Match bottom label colors to the bars
group_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for label, color in zip(ax.get_xticklabels(),
                        np.repeat(group_colors, 4)):
    label.set_color(color)

# Second axis for group labels
secax = ax.secondary_xaxis('bottom')
secax.set_xticks([1.5, 5.5, 9.5, 13.5])
secax.set_xticklabels(['P1+A1-', 'P1+A1+', 'P1-A1+', 'P1-A1-'])
secax.spines['bottom'].set_position(('outward', 45))
secax.tick_params(length=0)
# Match group label colors
for label, color in zip(secax.get_xticklabels(), group_colors):
    label.set_color(color)
plt.tight_layout()
plt.savefig(figpath + 'corr_early_late_delay.pdf')


#%%

fx =23
tix_latedelay = np.arange(12,16)
topcells_P1 = np.unique(np.concatenate([dict_topcells['P1']['topcells_at_t'][fx][i] for i in tix_latedelay]))
topcells_A1 = np.unique(np.concatenate([dict_topcells['A1']['topcells_at_t'][fx][i] for i in tix_latedelay]))
topcells_P2 = np.unique(np.concatenate([dict_topcells['P2']['topcells_at_t'][fx][i] for i in tix_latedelay]))
topcells_A2 = np.unique(np.concatenate([dict_topcells['A2']['topcells_at_t'][fx][i] for i in tix_latedelay]))

topcells_P1_P2 = topcells_P1[~np.isin(topcells_P1,topcells_P2)]
topcells_A2_P2 = topcells_A2[~np.isin(topcells_A2,topcells_P2)]
topcells_P2_A2 = topcells_P2[~np.isin(topcells_P2,topcells_A2)]

topcells_P1xP2 = topcells_P1[np.isin(topcells_P1,topcells_P2)]
topcells_P1xA2 = topcells_P1[np.isin(topcells_P1,topcells_A2_P2)]
topcells_A1xP2 = topcells_A1[np.isin(topcells_A1,topcells_P2_A2)]
topcells_A1xA2 = topcells_A1[np.isin(topcells_A1,topcells_A2)]

topcells_1 = np.unique(np.concatenate((topcells_P1,topcells_A1)))
topcells_2 = np.unique(np.concatenate((topcells_P2,topcells_A2)))
topcells_1x2 = topcells_1[np.isin(topcells_1,topcells_2)]
topcells_1_2 = topcells_1[~np.isin(topcells_1,topcells_2)]
topcells_2_1 = topcells_2[~np.isin(topcells_2,topcells_1)]
frac_1x2 = np.round(len(topcells_1x2)/len(topcells_1),decimals=3)

nonoutlier = fit_summary['nonoutlier'][fx]
data_P1   = modelfit[f'sess{fx}']['P1']['data'][nonoutlier,:]
data_A1   = modelfit[f'sess{fx}']['A1']['data'][nonoutlier,:]
data_P2   = modelfit[f'sess{fx}']['P2']['data'][nonoutlier,:]
data_A2   = modelfit[f'sess{fx}']['A2']['data'][nonoutlier,:]
diff1 = np.mean((data_P1 - data_A1)[:,12:16],axis=1)
diff2 = np.mean((data_P2 - data_A2)[:,12:16],axis=1)
CD1 = diff1 / np.linalg.norm(diff1)
CD2 = diff2 / np.linalg.norm(diff2)
CDdp = np.inner(CD1,CD2)


topcells_1x2_union = np.concatenate((topcells_P1xP2,topcells_P1xA2,topcells_A1xP2,topcells_A1xA2))
np.all(np.sort(topcells_1x2) == np.unique(np.sort(topcells_1x2_union)))

_CDdp_1x2 = np.sum((CD1*CD2)[topcells_1x2])
_CDdp_P1xP2 = np.sum((CD1*CD2)[topcells_P1xP2])
_CDdp_A1xA2 = np.sum((CD1*CD2)[topcells_A1xA2])
_CDdp_P1xA2 = np.sum((CD1*CD2)[topcells_P1xA2])
_CDdp_A1xP2 = np.sum((CD1*CD2)[topcells_A1xP2])
_CDdp_P1xP2_A1xA2 = np.sum((CD1*CD2)[np.unique(np.concatenate((topcells_P1xP2,topcells_A1xA2)))])
_CDdp_P1xP2_A1xA2_A1xP2 = np.sum((CD1*CD2)[np.unique(np.concatenate((topcells_P1xP2,topcells_A1xA2,topcells_A1xP2)))])


print('CD dot product: ', CDdp)
print('CD dot product 1x2: ', _CDdp_1x2)
print('CD dot product P1xP2: ', _CDdp_P1xP2)
print('CD dot product A1xA2: ', _CDdp_A1xA2)
print('CD dot product P1xA2: ', _CDdp_P1xA2)
print('CD dot product A1xP2: ', _CDdp_A1xP2)
print('CD dot product P1xP2 + A1xA2: ', _CDdp_P1xP2_A1xA2)
print('CD dot product P1xP2 + A1xA2 + A1xP2: ', _CDdp_P1xP2_A1xA2_A1xP2)

# print('fraction of P1xP2: ', len(topcells_P1xP2) / len(np.unique(topcells_P1)))
# print('fraction of P1-P2: ', len(topcells_P1_P2) / len(np.unique(topcells_P1)))



# plt.figure()
# for i in range(9):
#     plt.subplot(3,3,i+1)
#     ci = topcells_P1xP2[i]
#     plt.plot(data_P1[ci])
#     plt.plot(data_P2[ci])
#     events()
# plt.tight_layout()



# plt.figure()
# plt.scatter(diff1[topcells_P1xP2],diff2[topcells_P1xP2])
# # plt.plot((CD1*CD2)[topcells_P1xP2])
# plt.tight_layout()




#%%


fx =22
tix_latedelay = np.arange(12,16)
topcells_P1 = np.concatenate([dict_topcells['P1']['topcells_at_t'][fx][i] for i in tix_latedelay])
topcells_A1 = np.concatenate([dict_topcells['A1']['topcells_at_t'][fx][i] for i in tix_latedelay])
topcells_P2 = np.concatenate([dict_topcells['P2']['topcells_at_t'][fx][i] for i in tix_latedelay])
topcells_A2 = np.concatenate([dict_topcells['A2']['topcells_at_t'][fx][i] for i in tix_latedelay])


topcells_1 = np.unique(np.concatenate((topcells_P1,topcells_A1)))
topcells_2 = np.unique(np.concatenate((topcells_P2,topcells_A2)))
topcells_1x2 = topcells_1[np.isin(topcells_1,topcells_2)]
topcells_1_2 = topcells_1[~np.isin(topcells_1,topcells_2)]
topcells_2_1 = topcells_2[~np.isin(topcells_2,topcells_1)]
frac_1x2 = np.round(len(topcells_1x2)/len(topcells_1),decimals=3)

nonoutlier = fit_summary['nonoutlier'][fx]
data_P1   = modelfit[f'sess{fx}']['P1']['data'][nonoutlier,:]
data_A1   = modelfit[f'sess{fx}']['A1']['data'][nonoutlier,:]
data_P2   = modelfit[f'sess{fx}']['P2']['data'][nonoutlier,:]
data_A2   = modelfit[f'sess{fx}']['A2']['data'][nonoutlier,:]
diff1 = np.mean((data_P1 - data_A1)[:,12:16],axis=1)
diff2 = np.mean((data_P2 - data_A2)[:,12:16],axis=1)
CD1 = diff1 / np.linalg.norm(diff1)
CD2 = diff2 / np.linalg.norm(diff2)
CDdp = np.inner(CD1,CD2)

cells_sorted = np.argsort(CD1)
topcells_1_sorted = cells_sorted[np.isin(cells_sorted, topcells_1)]
topcells_1_sorted_idx = np.where(np.isin(cells_sorted, topcells_1))[0]
topcells_1x2_sorted = cells_sorted[np.isin(cells_sorted, topcells_1x2)]
topcells_1x2_sorted_idx = np.where(np.isin(cells_sorted, topcells_1x2))[0]
topcells_1_2_sorted = cells_sorted[np.isin(cells_sorted, topcells_1_2)]
topcells_1_2_sorted_idx = np.where(np.isin(cells_sorted, topcells_1_2))[0]
topcells_2_1_sorted = cells_sorted[np.isin(cells_sorted, topcells_2_1)]
topcells_2_1_sorted_idx = np.where(np.isin(cells_sorted, topcells_2_1))[0]
thresh = np.where(CD1[cells_sorted] > 0)[0][0]
ncell = data_P1.shape[0]


plt.figure(figsize=(8,10))
plt.subplot(311)
# plt.plot(CD2[cells_sorted],c='darkorange',marker='.',ms=2,linestyle='')
plt.plot(CD1[cells_sorted],c='limegreen',marker='.',ms=2,linestyle='')
plt.plot(topcells_1x2_sorted_idx,CD1[topcells_1x2_sorted],c='k',marker='.',ms=5,linestyle='')
plt.plot(topcells_1_2_sorted_idx,CD1[topcells_1_2_sorted],c='r',marker='.',ms=5,linestyle='')
plt.plot(topcells_2_1_sorted_idx,CD1[topcells_2_1_sorted],c='b',marker='.',ms=5,linestyle='')
plt.axvline(thresh,color='r',linestyle='--')
plt.title('fx ' + str(fx) + ', common ' + str(frac_1x2) + ', CD1CD2: ' + str(np.round(CDdotproduct[fx],decimals=2)))
plt.subplot(312)
plt.plot(CD2[cells_sorted],c='darkorange',marker='.',ms=2,linestyle='')
# plt.plot(CD1[cells_sorted],c='limegreen',marker='.',ms=2,linestyle='')
plt.plot(topcells_1x2_sorted_idx,CD2[topcells_1x2_sorted],c='k',marker='.',ms=5,linestyle='')
plt.plot(topcells_1_2_sorted_idx,CD2[topcells_1_2_sorted],c='r',marker='.',ms=5,linestyle='')
plt.plot(topcells_2_1_sorted_idx,CD2[topcells_2_1_sorted],c='b',marker='.',ms=5,linestyle='')
plt.axvline(thresh,color='r',linestyle='--')
plt.subplot(313)
plt.hist(topcells_1x2_sorted_idx,bins=100,range=(0,ncell), color='k', histtype='bar', label='preserved')
plt.hist(topcells_1_2_sorted_idx,bins=100,range=(0,ncell), color='r', histtype='step', lw=2, label='removed')
plt.hist(topcells_2_1_sorted_idx,bins=100,range=(0,ncell), color='b', histtype='step', lw=2, label='emerged')
plt.legend()
plt.axvline(thresh,color='r',linestyle='--')
plt.tight_layout()


#%%

'''
compare the fit accuracy of topcell_1x2 and topcell_1_2
'''

def fit_accuracy(_fx, _act, _cells, _dict_topcells):
    _fit_accuracy = np.array([])
    for i in tix_latedelay:
        _top_cells = _dict_topcells[_act]['topcells_at_t'][_fx][i]
        _cell_idx  = np.isin(_top_cells, _cells)
        _cell_num  = _top_cells[_cell_idx]
        _cells = _cells[~np.isin(_cells,_cell_num)]
        
        _fit_accuracy_tmp = fit_summary_topcells[_act]['expvar_neuron_at_t'][_fx][i]
        _fit_accuracy = np.concatenate((_fit_accuracy,_fit_accuracy_tmp[_cell_idx]))
    return _fit_accuracy
    

fit_accuracy_topcells_1x2_in_P1 = np.zeros(nfile)
fit_accuracy_topcells_1_2_in_P1 = np.zeros(nfile)
fit_accuracy_topcells_1x2_in_A1 = np.zeros(nfile)
fit_accuracy_topcells_1_2_in_A1 = np.zeros(nfile)

num_1x2 = np.zeros(nfile)
num_1 = np.zeros(nfile)

fx =15
for fx in range(nfile):
    tix_latedelay = np.arange(12,16)
    topcells_P1 = np.unique(np.concatenate([dict_topcells['P1']['topcells_at_t'][fx][i] for i in tix_latedelay]))
    topcells_A1 = np.unique(np.concatenate([dict_topcells['A1']['topcells_at_t'][fx][i] for i in tix_latedelay]))
    topcells_P2 = np.unique(np.concatenate([dict_topcells['P2']['topcells_at_t'][fx][i] for i in tix_latedelay]))
    topcells_A2 = np.unique(np.concatenate([dict_topcells['A2']['topcells_at_t'][fx][i] for i in tix_latedelay]))

    topcells_1 = np.unique(np.concatenate((topcells_P1,topcells_A1)))
    topcells_2 = np.unique(np.concatenate((topcells_P2,topcells_A2)))
    topcells_1x2 = topcells_1[np.isin(topcells_1,topcells_2)]
    topcells_1_2 = topcells_1[~np.isin(topcells_1,topcells_2)]

    _fit_accuracy_topcells_1_2_in_P1 = fit_accuracy(fx,'P1',topcells_1_2,dict_topcells)
    _fit_accuracy_topcells_1x2_in_P1 = fit_accuracy(fx,'P1',topcells_1x2,dict_topcells)
    _fit_accuracy_topcells_1_2_in_A1 = fit_accuracy(fx,'A1',topcells_1_2,dict_topcells)
    _fit_accuracy_topcells_1x2_in_A1 = fit_accuracy(fx,'A1',topcells_1x2,dict_topcells)

    fit_accuracy_topcells_1_2_in_P1[fx] = np.mean(_fit_accuracy_topcells_1_2_in_P1)
    fit_accuracy_topcells_1x2_in_P1[fx] = np.mean(_fit_accuracy_topcells_1x2_in_P1)
    fit_accuracy_topcells_1_2_in_A1[fx] = np.mean(_fit_accuracy_topcells_1_2_in_A1)
    fit_accuracy_topcells_1x2_in_A1[fx] = np.mean(_fit_accuracy_topcells_1x2_in_A1)
    
    num_1x2[fx] = len(topcells_1x2)
    num_1[fx] = len(topcells_1)


plt.figure()
plt.subplot(211)
plt.plot(fit_accuracy_topcells_1_2_in_P1,marker='o', label='top 1x2')
plt.plot(fit_accuracy_topcells_1x2_in_P1, label='top 1-2')
plt.legend()
plt.subplot(212)
plt.plot(fit_accuracy_topcells_1_2_in_A1,marker='o')
plt.plot(fit_accuracy_topcells_1x2_in_A1)
plt.tight_layout()


#%%

fx = 23

# tix_latedelay = np.arange(12,16)
tix_latedelay = [14]
topcells_P1 = np.unique(np.concatenate([dict_topcells['P1']['topcells_at_t'][fx][i] for i in tix_latedelay]))
topcells_A1 = np.unique(np.concatenate([dict_topcells['A1']['topcells_at_t'][fx][i] for i in tix_latedelay]))
topcells_P2 = np.unique(np.concatenate([dict_topcells['P2']['topcells_at_t'][fx][i] for i in tix_latedelay]))
topcells_A2 = np.unique(np.concatenate([dict_topcells['A2']['topcells_at_t'][fx][i] for i in tix_latedelay]))
topcells_1 = np.unique(np.concatenate((topcells_P1,topcells_A1)))
topcells_2 = np.unique(np.concatenate((topcells_P2,topcells_A2)))
topcells_1x2 = topcells_1[np.isin(topcells_1,topcells_2)]
topcells_1_2 = topcells_1[~np.isin(topcells_1,topcells_2)]

nonoutlier = fit_summary['nonoutlier'][fx]
data_P1_fx = modelfit[f'sess{fx}']['P1']['data'][nonoutlier]
data_P2_fx = modelfit[f'sess{fx}']['P2']['data'][nonoutlier]
data_A1_fx = modelfit[f'sess{fx}']['A1']['data'][nonoutlier]
data_A2_fx = modelfit[f'sess{fx}']['A2']['data'][nonoutlier]

S1 = np.mean((data_P1_fx - data_A1_fx)[:,12:16],axis=1)
S2 = np.mean((data_P2_fx - data_A2_fx)[:,12:16],axis=1)
dS = S1*S2
# cells_dS_neg_S1_pos = np.where((dS<0)*(S1>0))[0]
cells_dS_neg_S1_pos = np.where((dS>0)*(S1>0))[0]
topcells_1x2_dS_neg_S1_pos = cells_dS_neg_S1_pos[np.isin(cells_dS_neg_S1_pos, topcells_1x2)]

binary_P1_fx = dict_topcells['P1']['data_binary'][fx][topcells_1x2_dS_neg_S1_pos]
binary_A1_fx = dict_topcells['A1']['data_binary'][fx][topcells_1x2_dS_neg_S1_pos]
binary_P2_fx = dict_topcells['P2']['data_binary'][fx][topcells_1x2_dS_neg_S1_pos]
binary_A2_fx = dict_topcells['A2']['data_binary'][fx][topcells_1x2_dS_neg_S1_pos]

data_P1_fx_avg = np.mean(data_P1_fx[topcells_1x2_dS_neg_S1_pos],axis=0)
data_P2_fx_avg = np.mean(data_P2_fx[topcells_1x2_dS_neg_S1_pos],axis=0)
data_A1_fx_avg = np.mean(data_A1_fx[topcells_1x2_dS_neg_S1_pos],axis=0)
data_A2_fx_avg = np.mean(data_A2_fx[topcells_1x2_dS_neg_S1_pos],axis=0)

# plt.figure(figsize=(4,4))
# plt.plot(data_P1_fx_avg,c='b')
# plt.plot(data_A1_fx_avg,c='r')
# plt.plot(data_P2_fx_avg,c='b',linestyle=':')
# plt.plot(data_A2_fx_avg,c='r',linestyle=':')
# plt.axvline(8,color='gray',linestyle='--')
# plt.axvline(16,color='gray',linestyle='--')
# plt.tight_layout()

plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
plt.title('P1')
plt.imshow(binary_P1_fx,cmap='Blues',aspect='auto')
plt.subplot(2,2,3)
plt.title('P2')
plt.imshow(binary_P2_fx,cmap='Blues',aspect='auto')
plt.subplot(2,2,2)
plt.title('A1')
plt.imshow(binary_A1_fx,cmap='Reds',aspect='auto')
plt.subplot(2,2,4)
plt.title('A2')
plt.imshow(binary_A2_fx,cmap='Reds',aspect='auto')
plt.tight_layout()


# nplot = np.minimum(9,len(topcells_1x2_dS_neg_S1_pos))
# plt.figure(figsize=(6,6))
# for ix in range(nplot):
#     plt.subplot(3,3,ix+1)
#     ci = topcells_1x2_dS_neg_S1_pos[ix]
#     plt.plot(data_P1_fx[ci],c='b')
#     plt.plot(data_P2_fx[ci],c='b',linestyle=':')
#     plt.axvline(8,color='gray',linestyle='--')
#     plt.axvline(16,color='gray',linestyle='--')
# plt.tight_layout()


# plt.figure(figsize=(6,6))
# for ix in range(nplot):
#     plt.subplot(3,3,ix+1)
#     ci = topcells_1x2_dS_neg_S1_pos[ix]
#     plt.plot(data_A1_fx[ci],c='r')
#     plt.plot(data_A2_fx[ci],c='r',linestyle=':')
#     plt.axvline(8,color='gray',linestyle='--')
#     plt.axvline(16,color='gray',linestyle='--')
# plt.tight_layout()



#%%


# act = 'A1'
num_topcells_1 = np.zeros((nfile,len(t)))
num_topcells_2 = np.zeros((nfile,len(t)))
num_topcells_1x2 = np.zeros((nfile,len(t)))
num_common_1 = np.zeros((nfile,len(t)))
num_common_2 = np.zeros((nfile,len(t)))
for fx in range(nfile):
    binary_P1 = dict_topcells['P1']['data_binary'][fx].astype(bool)
    binary_A1 = dict_topcells['A1']['data_binary'][fx].astype(bool)
    binary_P2 = dict_topcells['P2']['data_binary'][fx].astype(bool)
    binary_A2 = dict_topcells['A2']['data_binary'][fx].astype(bool)

    binary_1 = binary_P1 + binary_A1
    binary_2 = binary_P2 + binary_A2
    binary_1x2 = binary_1 * binary_2

    num_topcells_1[fx] = np.sum(binary_1,axis=0)
    num_topcells_2[fx] = np.sum(binary_2,axis=0)
    num_topcells_1x2[fx] = np.sum(binary_1x2,axis=0)
    num_common_1[fx] = num_topcells_1x2[fx] / num_topcells_1[fx]
    num_common_2[fx] = num_topcells_1x2[fx] / num_topcells_2[fx]

num_common1_delay = np.mean(num_common_1[:,12:16],axis=1)
num_common2_delay = np.mean(num_common_2[:,12:16],axis=1)

num_common1_sample = np.mean(num_common_1[:,4:8],axis=1)
num_common2_sample = np.mean(num_common_2[:,4:8],axis=1)

#%%
plt.figure(figsize=(8,8))
for fx in range(nfile):
    plt.subplot(6,6,fx+1)
    plt.plot(num_topcells_1[fx])
    plt.plot(num_topcells_2[fx])
    plt.tight_layout()
    plt.axvline(8,color='gray',linestyle='--')
    plt.axvline(16,color='gray',linestyle='--')
    # plt.ylim([0,1])
plt.tight_layout()


plt.figure(figsize=(8,8))
for fx in range(nfile):
    plt.subplot(6,6,fx+1)
    plt.plot(num_common_1[fx])
    plt.plot(num_common_2[fx])
    plt.tight_layout()
    plt.axvline(8,color='gray',linestyle='--')
    plt.axvline(16,color='gray',linestyle='--')
    plt.ylim([0,1])
plt.tight_layout()


plt.figure(figsize=(3,3))
plt.subplot(211)
plt.hist(num_common1_sample,bins=20,histtype='step',range=(0,1))
plt.hist(num_common2_sample,bins=20,histtype='step',range=(0,1))
plt.axvline(np.mean(num_common1_sample),color='C0',linestyle='--')
plt.axvline(np.mean(num_common2_sample),color='C1',linestyle='--')
plt.subplot(212)
plt.hist(num_common1_delay,bins=20,histtype='step',range=(0,1))
plt.hist(num_common2_delay,bins=20,histtype='step',range=(0,1))
plt.axvline(np.mean(num_common1_delay),color='C0',linestyle='--')
plt.axvline(np.mean(num_common2_delay),color='C1',linestyle='--')
plt.tight_layout()



plt.figure(figsize=(3,3))
_corr = np.corrcoef(CDdotproduct,num_common1_delay)[0,1]
plt.scatter(CDdotproduct,num_common1_delay)
for fx in range(nfile):
    plt.annotate(str(fx),xy=(CDdotproduct[fx],num_common1_delay[fx]),xycoords='data')
# plt.scatter(CDdotproduct,num_common2_delay)
plt.title('cor:' + str(np.round(_corr,decimals=3)))
plt.tight_layout()
