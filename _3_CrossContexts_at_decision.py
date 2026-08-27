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
from utils import functions, functions_xcontext


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

keys1 = ['P1+A1-', 'P1+A1+', 'P1-A1+']
keys2 = ['P2+A2-', 'P2+A2+', 'P2-A2+']
keys3 = ['P1','A1','P2','A2']
keys_within_context = ['P1-A1','P2-A2']
keys_across_context = ['P2-P1','A2-A1']

#%%

importlib.reload(functions_xcontext)

#%%

tix_delay_range = np.arange(8,16)

dict_topcells_1x2 = functions_xcontext.create_dict_topcells_1x2(nfile, dict_topcells, tix_delay_range)

dict_CDdp, CDdp, CDdp_1x2 = functions_xcontext.create_dict_CDdotprod(nfile, dict_topcells, dict_topcells_1x2, tix_delay_range, fit_summary, modelfit)

dict_CDdp_err = functions_xcontext.create_dict_CDdotprod_err(nfile, dict_CDdp, CDdp)

dict_module_activity = functions_xcontext.create_dict_module_activity(nfile, dict_topcells_1x2, fit_summary, modelfit, keys1, keys2, keys3)

dict_normalized_activity = functions_xcontext.create_dict_normalized_activity(nfile, dict_module_activity, fit_summary, modelfit, dict_topcells_1x2, keys1, keys2, keys3)

sequential_P1, sequential_A1, sequential_P2, sequential_A2 = functions_xcontext.create_sequential_activity(nfile, dict_topcells_1x2, dict_topcells, fit_summary, modelfit, keys1, keys2)

dict_within_selectivity = functions_xcontext.create_dict_within_selectivity(nfile, dict_normalized_activity, tix_delay_range, keys1, keys2, keys_within_context, keys_across_context)
dict_across_activity    = functions_xcontext.create_dict_across_activity(nfile, dict_normalized_activity, tix_delay_range, keys1, keys2, keys_within_context, keys_across_context)



#%%

fx = 10
tx = 14
sum = 0
for k1 in keys1:
    for k2 in keys2:
       sum += dict_normalized_activity[k1][k2]['P1'][fx,tx]

print(sum)

meanrate = {i:np.array([]) for i in range(nfile)}

for fx in range(nfile):
    nonoutlier = fit_summary['nonoutlier'][fx]
    for k1 in keys1:
        for k2 in keys2:
            k3 = 'P1'
            _topcells_1x2 = dict_topcells_1x2[fx][k1][k2]
            _data_k3 = modelfit[f'sess{fx}'][k3]['data'][nonoutlier,:]
            if len(_topcells_1x2) > 0:
                _data_k3_topcells_1x2 = np.mean(_data_k3[_topcells_1x2,12:16],axis=1)
                meanrate[fx] = np.concatenate((meanrate[fx],_data_k3_topcells_1x2))

poprate = np.zeros(nfile)
for fx in range(nfile):
    poprate[fx] = np.mean((meanrate[fx])[meanrate[fx]>0])                
                

plt.figure(figsize=(2,2))                
plt.plot(poprate)
plt.tight_layout()

                
plt.figure(figsize=(6,6))                
for fx in np.arange(20,29):
    plt.subplot(3,3,fx-19)
    _meanrate = meanrate[fx][meanrate[fx] > 0]
    plt.hist(np.log10(_meanrate),bins=40,histtype='step', range=(-4,0), density=True)
    plt.axvline(np.log10(np.mean(_meanrate)))
plt.tight_layout()
                

#%%
#--------------#
# module size #
#--------------#
bar_width = 0.6
titles1 = [r'$P_1^+A_1^-$', r'$P_1^+A_1^+$', r'$P_1^-A_1^+$']
labels2 = [r'$P_2^+A_2^-$', r'$P_2^+A_2^+$', r'$P_2^-A_2^+$']

fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(
    3, 3,
    hspace=0.5,   # small spacing overall
    wspace=1
)
for i2, key2 in enumerate(keys2):
    for i1, key1 in enumerate(keys1):

        frac_cells = np.zeros(nfile)
        for fx in range(nfile):
            nonoutlier = fit_summary['nonoutlier'][fx]
            ntotal_cells = len(nonoutlier)
            frac_cells[fx] = len(dict_topcells_1x2[fx][key1][key2]) / ntotal_cells        
        frac_cells_mean = np.mean(frac_cells)
        frac_cells_std  = np.std(frac_cells)
        
        ax1 = fig.add_subplot(gs[i2, i1])
        # Bar: mean
        plt.bar(
            0,
            frac_cells_mean,
            width=bar_width,
            color='gray',
            alpha=0.5
        )
        # Scatter with horizontal jitter
        jitter = 0.1 * (2 * np.random.rand(len(frac_cells)) - 1)
        plt.scatter(
            0 + jitter,
            frac_cells,
            s=12,
            facecolors='none',
            edgecolors='gray',
            linewidths=0.5,
            alpha=1
        )    
        # Error bar (SEM)
        plt.errorbar(
            0,
            frac_cells_mean,
            yerr = frac_cells_std / np.sqrt(len(frac_cells)),
            color='k',
            capsize=8,
            lw=1.5,
            linestyle='none'
        )
        plt.xlim([-0.5,0.5])
        plt.ylim([0.0,0.13])
        # Titles
        if i2 == 0:
            ax1.set_title(titles1[i1], fontsize=12)
        # Row labels
        if i1 == 0:
            ax1.set_ylabel(labels2[i2], fontsize=12)            
        ax1.set_xticks([-0.5,0,0.5])
        ax1.set_xticklabels([])
        plt.gca().spines[['top', 'right']].set_visible(False)            
plt.tight_layout()
plt.savefig(figpath + 'module_sizes.pdf')



total_frac_cells = np.zeros(nfile)
total_number_cells = np.zeros(nfile)
for fx in range(nfile):
    nonoutlier   = fit_summary['nonoutlier'][fx]
    ntotal_cells = len(nonoutlier)
    for i2, key2 in enumerate(keys2):
        for i1, key1 in enumerate(keys1):
            total_frac_cells[fx] += len(dict_topcells_1x2[fx][key1][key2]) / ntotal_cells
            total_number_cells[fx] += len(dict_topcells_1x2[fx][key1][key2])
total_frac_cells_mean, total_frac_cells_std = np.mean(total_frac_cells), np.std(total_frac_cells)
total_number_cells_mean, total_number_cells_std = np.mean(total_number_cells), np.std(total_number_cells)
        
        
plt.figure(figsize=(2,2))        
# Bar: mean
plt.bar(
    0,
    total_frac_cells_mean,
    width=bar_width,
    color='gray',
    alpha=0.5
)
# Scatter with horizontal jitter
jitter = 0.1 * (2 * np.random.rand(nfile) - 1)
plt.scatter(
    0 + jitter,
    total_frac_cells,
    s=12,
    facecolors='none',
    edgecolors='gray',
    linewidths=0.5,
    alpha=1
)    
# Error bar (SEM)
plt.errorbar(
    0,
    total_frac_cells_mean,
    yerr = total_frac_cells_std / np.sqrt(nfile),
    color='k',
    capsize=8,
    lw=1.5,
    linestyle='none'
)
plt.xticks([-0.5,0,0.5],[])
plt.gca().spines[['top', 'right']].set_visible(False)
plt.ylabel('frac of neurons')
plt.tight_layout()
plt.savefig(figpath + 'module_sizes_total_frac.pdf')




plt.figure(figsize=(2,2))        
# Bar: mean
plt.bar(
    0,
    total_number_cells_mean,
    width=bar_width,
    color='gray',
    alpha=0.5
)
# Scatter with horizontal jitter
jitter = 0.1 * (2 * np.random.rand(nfile) - 1)
plt.scatter(
    0 + jitter,
    total_number_cells,
    s=12,
    facecolors='none',
    edgecolors='gray',
    linewidths=0.5,
    alpha=1
)    
# Error bar (SEM)
plt.errorbar(
    0,
    total_number_cells_mean,
    yerr = total_number_cells_std / np.sqrt(nfile),
    color='k',
    capsize=8,
    lw=1.5,
    linestyle='none'
)
plt.xticks([-0.5,0,0.5],[])
plt.yticks([0,100,200,300,400,500])
plt.gca().spines[['top', 'right']].set_visible(False)
plt.ylabel('# of neurons')
plt.tight_layout()
plt.savefig(figpath + 'module_sizes_total_number.pdf')



_cor = np.corrcoef(CDdotproduct, total_number_cells)[0,1]
plt.figure(figsize=(2,2))
plt.scatter(CDdotproduct, total_number_cells, fc='None', ec='k')
plt.xlabel('CD dot product')
plt.ylabel('# of neurons')
plt.gca().spines[['top','right']].set_visible(False)
plt.title('corr ' + str(np.round(_cor,decimals=3)))
plt.tight_layout()
plt.savefig(figpath + 'module_sizes_vs_CDdotprod.pdf')

#%%
#-----------------------------------#
# approximation of CD dot product
#-----------------------------------#

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
plt.savefig(figpath + 'CDdotprod_error.pdf')

#%%
#-----------------------------------#
# neural activity of modules
#-----------------------------------#

fig = plt.figure(figsize=(6, 8))
gs = fig.add_gridspec(
    6, 3,
    hspace=0.3,   # small spacing overall
    wspace=0.5
)
titles1 = [r'$P_1^+A_1^-$', r'$P_1^+A_1^+$', r'$P_1^-A_1^+$']
labels2 = [r'$P_2^+A_2^-$', r'$P_2^+A_2^+$', r'$P_2^-A_2^+$']
for i2, key2 in enumerate(keys2):
    for i1, key1 in enumerate(keys1):
        # top: context 1
        ax1 = fig.add_subplot(gs[2*i2, i1])
        ax1.plot(
            np.nanmean(dict_module_activity[key1][key2]['P1'], axis=0),
            c='purple'
        )
        ax1.plot(
            np.nanmean(dict_module_activity[key1][key2]['A1'], axis=0),
            c='limegreen'
        )
        ax1.axvline(7, color='gray', linestyle='--')
        ax1.axvline(15, color='gray', linestyle='--')
        ax1.set_ylim([0, 0.3])
        ax1.set_xticklabels([])

        # bottom: context 2
        ax2 = fig.add_subplot(gs[2*i2 + 1, i1], sharex=ax1)
        ax2.plot(
            np.nanmean(dict_module_activity[key1][key2]['P2'], axis=0),
            c='purple',
            linestyle='--'
        )
        ax2.plot(
            np.nanmean(dict_module_activity[key1][key2]['A2'], axis=0),
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
# plt.savefig(figpath + 'neuron_group_traces.pdf')



fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(
    3, 3,
    hspace=0.3,   # small spacing overall
    wspace=0.5
)
bar_width = 0.6
titles1 = [r'$P_1^+A_1^-$', r'$P_1^+A_1^+$', r'$P_1^-A_1^+$']
labels2 = [r'$P_2^+A_2^-$', r'$P_2^+A_2^+$', r'$P_2^-A_2^+$']
for i2, key2 in enumerate(keys2):
    for i1, key1 in enumerate(keys1):        
        diff1 = np.nanmean(dict_normalized_activity[key1][key2]['P1'][:,tix_delay_range] - dict_normalized_activity[key1][key2]['A1'][:,tix_delay_range], axis=1)
        diff2 = np.nanmean(dict_normalized_activity[key1][key2]['P2'][:,tix_delay_range] - dict_normalized_activity[key1][key2]['A2'][:,tix_delay_range], axis=1)
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
        plt.xlim([-0.5,1.5])
        # plt.ylim([-0.3,0.3])
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
# plt.savefig(figpath + 'neuron_group_individuals_within_context.pdf')




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
    # print('key2', key2)
    for i1, key1 in enumerate(keys1):
        # top: context 1
        # print('key1', key1)
        
        diff1 = np.nanmean(dict_normalized_activity[key1][key2]['P2'][:,tix_delay_range] - dict_normalized_activity[key1][key2]['P1'][:,tix_delay_range], axis=1)
        diff2 = np.nanmean(dict_normalized_activity[key1][key2]['A2'][:,tix_delay_range] - dict_normalized_activity[key1][key2]['A1'][:,tix_delay_range], axis=1)
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
        plt.xlim([-0.5,1.5])
        # plt.ylim([-0.3,0.3])
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
# plt.savefig(figpath + 'neuron_group_individuals_cross_context.pdf')




#%%
#-----------------------------------#
# sequential activity of modules
#-----------------------------------#

for i1, k1 in enumerate(keys1):
    for i2, k2 in enumerate(keys2):        
        
        sequential_P1_i2_i1 = sequential_P1[i2,i1]
        sequential_A1_i2_i1 = sequential_A1[i2,i1]
        sequential_P2_i2_i1 = sequential_P2[i2,i1]
        sequential_A2_i2_i1 = sequential_A2[i2,i1]
        
        vmax = np.max([np.max(sequential_P1_i2_i1),np.max(sequential_A1_i2_i1),np.max(sequential_P2_i2_i1),np.max(sequential_A2_i2_i1)])

        plt.figure(figsize=(4.3,3.8))    
        plt.subplot(221)
        plt.title(r'$P_1$',fontsize=12)
        plt.imshow(sequential_P1_i2_i1,cmap='jet',aspect='auto',vmin=0,vmax=vmax)
        plt.axvline(7,color='w',linestyle='--')
        plt.axvline(15,color='w',linestyle='--')
        plt.colorbar()
        plt.subplot(222)
        plt.title(r'$A_1$',fontsize=12)
        plt.imshow(sequential_A1_i2_i1,cmap='jet',aspect='auto',vmin=0,vmax=vmax)
        plt.axvline(7,color='w',linestyle='--')
        plt.axvline(15,color='w',linestyle='--')
        plt.colorbar()
        plt.subplot(223)
        plt.title(r'$P_2$',fontsize=12)
        plt.imshow(sequential_P2_i2_i1,cmap='jet',aspect='auto',vmin=0,vmax=vmax)
        plt.axvline(7,color='w',linestyle='--')
        plt.axvline(15,color='w',linestyle='--')
        plt.colorbar()
        plt.subplot(224)
        plt.title(r'$A_2$',fontsize=12)
        plt.imshow(sequential_A2_i2_i1,cmap='jet',aspect='auto',vmin=0,vmax=vmax)
        plt.axvline(7,color='w',linestyle='--')
        plt.axvline(15,color='w',linestyle='--')
        plt.colorbar()
        plt.tight_layout()
        # plt.savefig(figpath + f'sequential/map_{i2}_{i1}_' + k2 + '_' + k1 + '.pdf')
        

        cmap = plt.cm.copper
        colors = cmap(np.linspace(0, 1, 8))        
        plt.figure(figsize=(3.3,3))
        plt.subplot(221)
        plt.title(r'$P_1$',fontsize=12)
        for i in range(8):
            plt.plot(sequential_P1_i2_i1[i], color=colors[i])
        plt.axvline(7,color='gray',linestyle='--')
        plt.axvline(15,color='gray',linestyle='--')
        plt.ylim([0,vmax])
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.subplot(222)
        plt.title(r'$A_1$',fontsize=12)
        for i in range(8):
            plt.plot(sequential_A1_i2_i1[i], color=colors[i])
        plt.axvline(7,color='gray',linestyle='--')
        plt.axvline(15,color='gray',linestyle='--')
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.ylim([0,vmax])
        plt.subplot(223)
        plt.title(r'$P_2$',fontsize=12)
        for i in range(8):
            plt.plot(sequential_P2_i2_i1[i], color=colors[i])
        plt.axvline(7,color='gray',linestyle='--')
        plt.axvline(15,color='gray',linestyle='--')
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.ylim([0,vmax])
        plt.subplot(224)
        plt.title(r'$A_2$',fontsize=12)
        for i in range(8):
            plt.plot(sequential_A2_i2_i1[i], color=colors[i])
        plt.axvline(7,color='gray',linestyle='--')
        plt.axvline(15,color='gray',linestyle='--')
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.ylim([0,vmax])
        plt.tight_layout()
        # plt.savefig(figpath + f'sequential/trace_{i2}_{i1}_' + k2 + '_' + k1 + '.pdf')

#%%
#-----------------------------------#
# CD dot product vs module activity
#-----------------------------------#



fig = plt.figure(figsize=(10, 6))

# 3 x 3 groups
outer = fig.add_gridspec(3, 3, wspace=0.35, hspace=0.4)

for i1, k1 in enumerate(keys1):
    for i2, k2 in enumerate(keys2):

        # two contexts placed close together
        inner = outer[i2, i1].subgridspec(1, 2, wspace=0.08)

        y1 = dict_within_selectivity[k1][k2]['P1-A1']
        y2 = dict_within_selectivity[k1][k2]['P2-A2']

        # common y range for direct comparison
        ymin = np.nanmin([np.nanmin(y1), np.nanmin(y2)]) - 0.05
        ymax = np.nanmax([np.nanmax(y1), np.nanmax(y2)]) + 0.05

        for ic, kc in enumerate(keys_within_context):

            ax = fig.add_subplot(inner[0, ic])
            ax.spines[['top', 'right']].set_visible(False)
            y = dict_within_selectivity[k1][k2][kc]
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
                    r'$r=$' + str(np.round(_cor, 3)),
                    fontsize=9
                )

                None
            else:
                ax.set_title(
                    r'$r=$' + str(np.round(_cor, 3)),
                    fontsize=9
                )
                ax.set_yticklabels([])

fig.tight_layout()
# plt.savefig(figpath + 'CDdotprod_within_context.pdf')



####### CD dot product: across context ########

fig = plt.figure(figsize=(10, 6))

# 3 x 3 groups
outer = fig.add_gridspec(3, 3, wspace=0.35, hspace=0.4)

for i1, k1 in enumerate(keys1):
    for i2, k2 in enumerate(keys2):

        # two contexts placed close together
        inner = outer[i2, i1].subgridspec(1, 2, wspace=0.08)

        y1 = dict_across_activity[k1][k2]['P2-P1']
        y2 = dict_across_activity[k1][k2]['A2-A1']

        # common y range for direct comparison
        ymin = np.nanmin([np.nanmin(y1), np.nanmin(y2)]) - 0.05
        ymax = np.nanmax([np.nanmax(y1), np.nanmax(y2)]) + 0.05

        for ic, kc in enumerate(keys_across_context):

            ax = fig.add_subplot(inner[0, ic])
            ax.spines[['top', 'right']].set_visible(False)
            y = dict_across_activity[k1][k2][kc]
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
                    r'$r=$' + str(np.round(_cor, 3)),
                    fontsize=9
                )
                None
            else:
                ax.set_title(
                    r'$r=$' + str(np.round(_cor, 3)),
                    fontsize=9
                )

                # remove duplicate y labels/ticks
                ax.set_yticklabels([])

fig.tight_layout()
# plt.savefig(figpath + 'CDdotprod_across_context.pdf')










#---------------- below is optional --------------------#



#%%
#---------------------------------------------
# neuron counts (context 1 vs context 2)
#---------------------------------------------

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
# plt.savefig(figpath + 'P1+_A1-.pdf')


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
# plt.savefig(figpath + 'P1-_A1+.pdf')



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
# plt.savefig(figpath + 'P1+_A1+.pdf')


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
# plt.savefig(figpath + 'P1+_A1-_hist.pdf')


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
# plt.savefig(figpath + 'P1-_A1+_hist.pdf')




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
# plt.savefig(figpath + 'P1+_A1+_hist.pdf')



#%%
#-----------------------------------------

# _mean_activity_A1 = np.mean(mean_activity_A1_norm, axis=0)  # shape: (6, time)
# _mean_activity_P2 = np.mean(mean_activity_P2_norm, axis=0)  # shape: (6, time)
# plt.figure(figsize=(4,4))
# plt.stackplot(np.arange(_mean_activity_A1.shape[1]),_mean_activity_A1,labels=[str(i+1) for i in range(6)],edgecolor='k',linewidth=0.5)
# plt.ylim([0, 1])
# plt.legend()
# plt.tight_layout()

# plt.figure(figsize=(4,4))
# plt.stackplot(np.arange(_mean_activity_P2.shape[1]),_mean_activity_P2,labels=[str(i+1) for i in range(6)],edgecolor='k',linewidth=0.5)
# plt.ylim([0, 1])
# plt.legend()
# plt.tight_layout()




# mean_activity_A1_avg = np.mean(mean_activity_A1_norm,axis=2)
# mean_activity_P2_avg = np.mean(mean_activity_P2_norm,axis=2)

# plt.figure(figsize=(9,6))
# for grp in range(6):
#     plt.subplot(2,3,grp+1)  
#     plt.scatter(CDdotproduct,mean_activity_A1_avg[:,grp],color=f'C{grp}')
#     _cor = np.corrcoef(CDdotproduct,mean_activity_A1_avg[:,grp])[0,1]
#     plt.title(str(np.round(_cor,decimals=3)))
# plt.tight_layout()

# plt.figure(figsize=(9,6))
# for grp in range(6):
#     plt.subplot(2,3,grp+1)  
#     plt.scatter(CDdotproduct,mean_activity_A1_avg[:,grp])
#     _cor = np.corrcoef(CDdotproduct,mean_activity_P2_avg[:,grp])[0,1]
#     plt.title(str(np.round(_cor,decimals=3)))
# plt.tight_layout()





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

# frac_1x2_outof_totalcells = np.zeros(nfile)
# frac_1x2_outof_topcells_1   = np.zeros(nfile)
# CD1err = np.zeros(nfile)
# CD2err = np.zeros(nfile)
# CDdp = np.zeros(nfile)
# CDdp_1x2 = np.zeros(nfile)
# CDdp_1_2 = np.zeros(nfile)
# CDdp_2_1 = np.zeros(nfile)
# tix_latedelay = np.arange(12,16)
# for fx in range(nfile):
#     # topcells
#     topcells_P1 = np.concatenate([dict_topcells['P1']['topcells_at_t'][fx][i] for i in tix_latedelay])
#     topcells_A1 = np.concatenate([dict_topcells['A1']['topcells_at_t'][fx][i] for i in tix_latedelay])
#     topcells_P2 = np.concatenate([dict_topcells['P2']['topcells_at_t'][fx][i] for i in tix_latedelay])
#     topcells_A2 = np.concatenate([dict_topcells['A2']['topcells_at_t'][fx][i] for i in tix_latedelay])

#     # shared & nonshared topcells
#     topcells_1 = np.unique(np.concatenate((topcells_P1,topcells_A1)))
#     topcells_2 = np.unique(np.concatenate((topcells_P2,topcells_A2)))
#     topcells_1x2 = topcells_1[np.isin(topcells_1,topcells_2)]
#     topcells_1_2 = topcells_1[~np.isin(topcells_1,topcells_2)]
#     topcells_2_1 = topcells_2[~np.isin(topcells_2,topcells_1)]
    
#     # compute CD1, CD2
#     nonoutlier = fit_summary['nonoutlier'][fx]
#     data_P1   = modelfit[f'sess{fx}']['P1']['data'][nonoutlier,:]
#     data_A1   = modelfit[f'sess{fx}']['A1']['data'][nonoutlier,:]
#     data_P2   = modelfit[f'sess{fx}']['P2']['data'][nonoutlier,:]
#     data_A2   = modelfit[f'sess{fx}']['A2']['data'][nonoutlier,:]
#     diff1 = np.mean((data_P1 - data_A1)[:,12:16],axis=1)
#     diff2 = np.mean((data_P2 - data_A2)[:,12:16],axis=1)
#     CD1 = diff1 / np.linalg.norm(diff1)
#     CD2 = diff2 / np.linalg.norm(diff2)

#     # number of topcells shared across contexts
#     ncells = len(nonoutlier)
#     frac_1x2_outof_totalcells[fx] = len(topcells_1x2)/ncells    
#     frac_1x2_outof_topcells_1[fx] = len(topcells_1x2)/len(topcells_1)

#     # approximate CD1 and CD2
#     CD1aprx = np.zeros_like(CD1)
#     CD2aprx = np.zeros_like(CD2)
#     CD1aprx[topcells_1x2] = CD1[topcells_1x2]
#     CD2aprx[topcells_1x2] = CD2[topcells_1x2]
#     CD1err[fx] = np.linalg.norm(CD1aprx - CD1)
#     CD2err[fx] = np.linalg.norm(CD2aprx - CD2)
    
#     # approximate CD1CD2
#     CDdp[fx] = np.inner(CD1,CD2)
#     CDdp_1x2[fx] = np.inner(CD1[topcells_1x2],CD2[topcells_1x2])
#     CDdp_1_2[fx] = np.inner(CD1[topcells_1_2],CD2[topcells_1_2])
#     CDdp_2_1[fx] = np.inner(CD1[topcells_2_1],CD2[topcells_2_1])
    

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
