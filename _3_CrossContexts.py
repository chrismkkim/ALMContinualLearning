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
from utils import plot_stimulus 
from utils import plot_neuralstate_cd
from utils import plot_neuralstate_eucldist
from utils import functions 
from utils import plot_learned_activity


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
#########################
# Load the fitted model #
#########################
datapath = 'data/fit_neuron_activity/'
modelfit = np.load(datapath + 'dict_peaktime.npy', allow_pickle=True).item()


# %%

dP = {}
dA = {}
dR = {}
dL = {}
P1_set1, P1_set2 = {}, {}
A1_set1, A1_set2 = {}, {}
P2_set1, P2_set2 = {}, {}
A2_set1, A2_set2 = {}, {}
dP_peak = {}
dA_peak = {}
dR_peak = {}
dL_peak = {}
P1_peak, P2_peak = {}, {}
A1_peak, A2_peak = {}, {}
dP_trace = {}
dA_trace = {}
dR_trace = {}
dL_trace = {}
P1_trace, P2_trace = {}, {}
A1_trace, A2_trace = {}, {}
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

def trial_avg_at_delay(opto_1R,opto_1L,opto_2R,opto_2L):
    opto_1R_delay = np.mean(opto_1R[:,tix_response-4:tix_response,:],axis=(1,2)) # neurons x time x trials
    opto_1L_delay = np.mean(opto_1L[:,tix_response-4:tix_response,:],axis=(1,2)) # neurons x time x trials
    opto_2R_delay = np.mean(opto_2R[:,tix_response-4:tix_response,:],axis=(1,2)) # neurons x time x trials
    opto_2L_delay = np.mean(opto_2L[:,tix_response-4:tix_response,:],axis=(1,2)) # neurons x time x trials
    return opto_1R_delay, opto_1L_delay, opto_2R_delay, opto_2L_delay

def trial_avg_trace(opto_1R,opto_1L,opto_2R,opto_2L):
    opto_1R_trace = np.mean(opto_1R,axis=2) # neurons x time x trials
    opto_1L_trace = np.mean(opto_1L,axis=2) # neurons x time x trials
    opto_2R_trace = np.mean(opto_2R,axis=2) # neurons x time x trials
    opto_2L_trace = np.mean(opto_2L,axis=2) # neurons x time x trials
    return opto_1R_trace, opto_1L_trace, opto_2R_trace, opto_2L_trace

def post_sample_onset(x,tix_sample,tix_response):
    x_post_sample_onset = x[:,tix_sample:tix_response+8]
    return x_post_sample_onset

def find_peak_time_data(opto_1R,opto_1L,opto_2R,opto_2L,tix_sample,tix_response):
    opto_1R_trace, opto_1L_trace, opto_2R_trace, opto_2L_trace = trial_avg_trace(opto_1R,opto_1L,opto_2R,opto_2L)
    opto_1R_trace_post_sample_onset = post_sample_onset(opto_1R_trace,tix_sample,tix_response)
    opto_1L_trace_post_sample_onset = post_sample_onset(opto_1L_trace,tix_sample,tix_response)
    opto_2R_trace_post_sample_onset = post_sample_onset(opto_2R_trace,tix_sample,tix_response)
    opto_2L_trace_post_sample_onset = post_sample_onset(opto_2L_trace,tix_sample,tix_response)
    peak_time_1R = np.argmax(opto_1R_trace_post_sample_onset,axis=1) + tix_sample
    peak_time_1L = np.argmax(opto_1L_trace_post_sample_onset,axis=1) + tix_sample
    peak_time_2R = np.argmax(opto_2R_trace_post_sample_onset,axis=1) + tix_sample
    peak_time_2L = np.argmax(opto_2L_trace_post_sample_onset,axis=1) + tix_sample
    return peak_time_1R, peak_time_1L, peak_time_2R, peak_time_2L

def find_peak_time_model(modelfit,fx):
    peak_time_1R = np.argmax(modelfit[f'sess{fx}']['P1']['fit'],axis=1) + tix_sample
    peak_time_1L = np.argmax(modelfit[f'sess{fx}']['A1']['fit'],axis=1) + tix_sample
    peak_time_2R = np.argmax(modelfit[f'sess{fx}']['A2']['fit'],axis=1) + tix_sample
    peak_time_2L = np.argmax(modelfit[f'sess{fx}']['P2']['fit'],axis=1) + tix_sample
    return peak_time_1R, peak_time_1L, peak_time_2R, peak_time_2L

#%%
def two_sided_exp(t, b, A, tp, tau_r, tau_d):
    return b + A * np.where(
        t < tp,
        np.exp((t - tp) / tau_r),
        np.exp(-(t - tp) / tau_d)
    )

# model_peak = np.concatenate((model_peak_time_1R, model_peak_time_1L, model_peak_time_2R, model_peak_time_2L))
# data_peak = np.concatenate((peak_time_1R, peak_time_1L, peak_time_2R, peak_time_2L))
# plt.figure()
# plt.scatter(model_peak_time_1R, peak_time_1R)

# plt.figure()
# plt.plot(model_peak_time_1L[:100], marker='o', linestyle='')
# plt.plot(peak_time_1L[:100], marker='o', linestyle='')

frac_matched = np.sum(np.abs((model_peak_time_1R - peak_time_1R)<=1)) / len(model_peak_time_1R)
print(frac_matched)

t = tvec[tix_sample:tix_response+8]

cell_matched = np.where(np.abs(model_peak_time_1R - peak_time_1R)==0)[0]

model_trace = modelfit[f'sess{fx}']['P1']['fit']
plt.figure(figsize=(8,8))
for ci in range(4):
    fit = two_sided_exp(t,*modelfit[f'sess{fx}']['P1']['par'][cell_matched[ci]])
    
    plt.subplot(4,1,ci+1)
    plt.plot(tvec,opto_1R_trace[cell_matched[ci]])
    plt.plot(tvec[tix_sample:tix_response+8],model_trace[cell_matched[ci]])
    # plt.plot(t,fit)
    plt.axvline(tvec[peak_time_1R[cell_matched[ci]]],linestyle='--', color='C0')
    plt.axvline(tvec[model_peak_time_1R[cell_matched[ci]]],linestyle='--', color='C1')
    # plt.axvline(modelfit[f'sess{fx}']['P1']['par'][cell_matched[ci],2],linestyle='--', color='C2')
plt.tight_layout()

#%%
plt.figure()
plt.plot(_amp_frac,_var_frac)


#%%
for fx in range(nfile):
    # print(fx)
    fx_sorted = sorted_indices_by_relearn_speed[fx]

    #--- sorted by relearn speed ---#
    filepath   = dirpath + merged_ID[fx_sorted] + '.npy'
    data       = np.load(filepath, allow_pickle=True)

    # split data into two sets of trials
    data_type = 'deconvolved'
    opto_1R_set1, opto_1L_set1, opto_2R_set1, opto_2L_set1, \
    opto_1R_set2, opto_1L_set2, opto_2R_set2, opto_2L_set2 = functions.split_trials(data, data_type)

    # trial-averaged at delay
    opto_1R_delay_set1, opto_1L_delay_set1, opto_2R_delay_set1, opto_2L_delay_set1 = trial_avg_at_delay(opto_1R_set1,opto_1L_set1,opto_2R_set1,opto_2L_set1)
    opto_1R_delay_set2, opto_1L_delay_set2, opto_2R_delay_set2, opto_2L_delay_set2 = trial_avg_at_delay(opto_1R_set2,opto_1L_set2,opto_2R_set2,opto_2L_set2)

    # trial-averaged neural trace
    opto_1R_trace, opto_1L_trace, opto_2R_trace, opto_2L_trace = trial_avg_trace(opto_1R_set2,opto_1L_set2,opto_2R_set2,opto_2L_set2)

    # find peak time (Use fitted model. Can also directly use the data)
    peak_time_1R, peak_time_1L, peak_time_2R, peak_time_2L = find_peak_time_model(modelfit,fx)
    # peak_time_1R, peak_time_1L, peak_time_2R, peak_time_2L = find_peak_time_data(opto_1R_set2,opto_1L_set2,opto_2R_set2,opto_2L_set2,tix_sample,tix_response)        

    # remove outlier cells
    keep_set1 = functions.remove_outliers_alltrials_fixedtime(opto_1R_delay_set1,opto_1L_delay_set1,opto_2R_delay_set1,opto_2L_delay_set1,maxstd=5)
    keep_set2 = functions.remove_outliers_alltrials_fixedtime(opto_1R_delay_set2,opto_1L_delay_set2,opto_2R_delay_set2,opto_2L_delay_set2,maxstd=5)
    keep      = keep_set1 * keep_set2
    num_outliers[fx] = np.sum(~keep)
    
    ###################################################################
    # keep only the high amplitude cells that explain 90% of variance #
    ################################################################### 
    keep_idx = np.where(keep)[0]
    _idx_accu = np.array([])
    idx_amp = 1
    for act in ['P1','A1','P2','A2']:
        _amp_neuron =        modelfit[f'sess{fx}'][act]['par'][:,idx_amp][keep_idx]
        _var_neuron = np.var(modelfit[f'sess{fx}'][act]['data'],axis=1)[keep_idx]    
        _amp_frac, _var_frac, _idx_topamp = functions.find_top_90_cells(_amp_neuron,_var_neuron,percentage=0.9)
        _idx_accu = np.concatenate((_idx_accu,_idx_topamp))
    _idx_topamp = np.unique(_idx_accu).astype(int)
    keep_idx_topamp = keep_idx[_idx_topamp]
    keep = np.zeros_like(keep,dtype=bool)
    keep[keep_idx_topamp] = True
    
    
    # neural activity - outlier neurons removed
    P1_set1[fx] = opto_1R_delay_set1[keep]
    A1_set1[fx] = opto_1L_delay_set1[keep]
    P2_set1[fx] = opto_2L_delay_set1[keep]    
    A2_set1[fx] = opto_2R_delay_set1[keep]    
    P1_set2[fx] = opto_1R_delay_set2[keep]
    A1_set2[fx] = opto_1L_delay_set2[keep]
    P2_set2[fx] = opto_2L_delay_set2[keep]    
    A2_set2[fx] = opto_2R_delay_set2[keep]
    
    # peak time - outlier neurons removed
    P1_peak[fx] = peak_time_1R[keep]
    A1_peak[fx] = peak_time_1L[keep]
    P2_peak[fx] = peak_time_2L[keep]
    A2_peak[fx] = peak_time_2R[keep]

    # neural trace - outlier neurons removed
    P1_trace[fx] = opto_1R_trace[keep]
    A1_trace[fx] = opto_1L_trace[keep]
    P2_trace[fx] = opto_2L_trace[keep]
    A2_trace[fx] = opto_2R_trace[keep]
    
    '''
    Use Trial Set 1
        - Selectivity:    S1, S2
        - CD dot product: CD1*CD2
    '''
    # coding direction (single neuron)
    #   - use trial set 1
    S1[fx], S2[fx] = functions.compute_S1_S2(P1_set1[fx], P2_set1[fx], A1_set1[fx], A2_set1[fx])
    # CD dot product (population)
    CD_dotproduct[fx] = functions.compute_CD_dotproduct(opto_1R_set1,opto_1L_set1,opto_2R_set1,opto_2L_set1,tix_response)
    
    '''
    Use Trial Set 2
        - dP, dA
        - dR, dL
    '''
    # learned activity 
    #   - use trial set 2
    dP[fx], dA[fx], dR[fx], dL[fx] = functions.compute_dP_dA_dR_dL(P1_set2[fx], P2_set2[fx], A1_set2[fx], A2_set2[fx])
    
    # peak time
    dP_peak[fx], dA_peak[fx], dR_peak[fx], dL_peak[fx] = functions.compute_dP_dA_dR_dL_peak(P1_peak[fx], P2_peak[fx], A1_peak[fx], A2_peak[fx])

    # neural trace
    dP_trace[fx], dA_trace[fx], dR_trace[fx], dL_trace[fx] = functions.compute_dP_dA_dR_dL_peak(P1_trace[fx], P2_trace[fx], A1_trace[fx], A2_trace[fx])

#%%

(dS, cells_pos, cells_neg, 
topcells_dS_pos, topcells_dS_neg, 
cells_dS_neg_S1_pos, cells_dS_neg_S1_neg, 
cells_dS_pos_S1_pos, cells_dS_pos_S1_neg, 
cells_dS_neg, cells_dS_pos, cells_dS, 
num_totalcells, num_topcells) = functions.neuron_types(S1, S2, dS, 
                                                    cells_pos, cells_neg, 
                                                    topcells_dS_pos, topcells_dS_neg,
                                                    cells_dS_neg_S1_pos, cells_dS_neg_S1_neg, cells_dS_pos_S1_pos, cells_dS_pos_S1_neg,
                                                    cells_dS_neg, cells_dS_pos, cells_dS,
                                                    num_totalcells, num_topcells, nfile)

#%%
dict_sess = {'dS-': {'S1-': {},'S1+': {}},
             'dS+': {'S1-': {},'S1+': {}}}
learned_activity = {f'sess{i}': copy.deepcopy(dict_sess) for i in range(nfile)}
frac_cells       = {f'sess{i}': copy.deepcopy(dict_sess) for i in range(nfile)}
learned_activity_sum                 = copy.deepcopy(dict_sess)
frac_cells_sum                       = copy.deepcopy(dict_sess)
learned_activity, frac_cells = functions.compute_learned_activity(learned_activity, frac_cells, 
                                                                    dP, dA, dR, dL, 
                                                                    cells_dS_neg_S1_pos, cells_dS_neg_S1_neg, cells_dS_pos_S1_pos, cells_dS_pos_S1_neg,
                                                                    nfile)
learned_acitivty_sum, frac_cells_sum = functions.compute_learned_activity_sum(learned_activity, learned_activity_sum, frac_cells, frac_cells_sum, nfile)

#%%
dict_sess = {'dS-': {'S1-': {},'S1+': {}},
             'dS+': {'S1-': {},'S1+': {}}}
diff_peak_time = {f'sess{i}': copy.deepcopy(dict_sess) for i in range(nfile)}
diff_peak_time = functions.compute_diff_peak_time(diff_peak_time, 
                                                dP_peak, dA_peak, dR_peak, dL_peak, 
                                                cells_dS_neg_S1_pos, cells_dS_neg_S1_neg, cells_dS_pos_S1_pos, cells_dS_pos_S1_neg,
                                                nfile)

neural_trace = {f'sess{i}': copy.deepcopy(dict_sess) for i in range(nfile)}
neural_trace = functions.compute_diff_peak_time(neural_trace, 
                                                dP_trace, dA_trace, dR_trace, dL_trace, 
                                                cells_dS_neg_S1_pos, cells_dS_neg_S1_neg, cells_dS_pos_S1_pos, cells_dS_pos_S1_neg,
                                                nfile)

#%%

frac_pos_act_pos_peak = np.zeros(nfile)
frac_pos_act_neg_peak = np.zeros(nfile)
frac_neg_act_pos_peak = np.zeros(nfile)
frac_neg_act_neg_peak = np.zeros(nfile)

rev = 'dS+'
sel = 'S1+'
act = 'dR'
sgn = '-'
for fx in range(nfile):

    if sgn == '+':     
        _idx_la = learned_activity[f'sess{fx}'][rev][sel][act] > 0
    elif sgn == '-':
        _idx_la = learned_activity[f'sess{fx}'][rev][sel][act] < 0
    _diff_peak    = diff_peak_time[f'sess{fx}'][rev][sel][act][0][_idx_la]
    _peak_time_P1 = diff_peak_time[f'sess{fx}'][rev][sel][act][1][_idx_la]
    _peak_time_P2 = diff_peak_time[f'sess{fx}'][rev][sel][act][2][_idx_la]
    _la_neg       = learned_activity[f'sess{fx}'][rev][sel][act][_idx_la]
    _tr1          = neural_trace[f'sess{fx}'][rev][sel][act][1][_idx_la]
    _tr2          = neural_trace[f'sess{fx}'][rev][sel][act][2][_idx_la]
    
    cells_sorted = np.argsort(_peak_time_P1)
    # cells_sorted = np.argsort(_la_neg)

    vmax = 0.1
    num_neg = np.sum(_idx_la)
    plt.figure()
    plt.subplot(131)
    for ii in range(num_neg):
        ii_sorted = cells_sorted[ii]
        plt.plot(_peak_time_P1[ii_sorted], ii, c='k', marker='.')
        plt.plot(_peak_time_P2[ii_sorted], ii, c='r', marker='.')
        _diff = _peak_time_P2[ii_sorted] - _peak_time_P1[ii_sorted]
        if _diff > 0:
            plt.plot([_peak_time_P1[ii_sorted],_peak_time_P2[ii_sorted]], [ii,ii], c='r', alpha=0.3)
        else:
            plt.plot([_peak_time_P1[ii_sorted],_peak_time_P2[ii_sorted]], [ii,ii], c='k', alpha=0.3)
    plt.xlim([0,ntimestep])
    plt.ylim([-1,num_neg])
    plt.axvline(tix_sample, color='gray', linestyle='--', lw=0.5)
    plt.axvline(tix_delay, color='gray', linestyle='--', lw=0.5)
    plt.axvline(tix_response, color='gray', linestyle='--', lw=0.5)
    plt.subplot(132)
    plt.imshow(_tr1[cells_sorted,:], cmap='jet', aspect='auto', vmin=0, vmax=vmax, origin='lower')
    plt.axvline(tix_sample, color='w', linestyle='--', lw=0.5)
    plt.axvline(tix_delay, color='w', linestyle='--', lw=0.5)
    plt.axvline(tix_response, color='w', linestyle='--', lw=0.5)
    plt.subplot(133)
    plt.imshow(_tr2[cells_sorted,:], cmap='jet', aspect='auto', vmin=0, vmax=vmax, origin='lower')
    plt.axvline(tix_sample, color='w', linestyle='--', lw=0.5)
    plt.axvline(tix_delay, color='w', linestyle='--', lw=0.5)
    plt.axvline(tix_response, color='w', linestyle='--', lw=0.5)
    plt.tight_layout()    


#%%

from scipy.ndimage import gaussian_filter

step_size = 1
peak_time = np.arange(0,ntimestep,step=step_size)
nbins = len(peak_time)


reverse = ['dS+', 'dS-']
selectivity = ['S1+', 'S1-']
activity = [['dP','dA'],['dR','dL']]
signs = ['+','-']
for rev in reverse:
    for sel in selectivity:
        if rev == 'dS-':
            _activity = activity[0]
        elif rev == 'dS+':
            _activity = activity[1]
        for act in _activity:             
            for sgn in signs:           
                
                # dominant vs. nondominant
                dominant_mode = (
                    (rev == 'dS-' and sel == 'S1+' and act == 'dP' and sgn == '-') or
                    (rev == 'dS-' and sel == 'S1+' and act == 'dA' and sgn == '+') or
                    (rev == 'dS-' and sel == 'S1-' and act == 'dP' and sgn == '+') or
                    (rev == 'dS-' and sel == 'S1-' and act == 'dA' and sgn == '-') or
                    (rev == 'dS+' and sel == 'S1+' and act == 'dR' and sgn == '-') or
                    (rev == 'dS+' and sel == 'S1+' and act == 'dL' and sgn == '+') or
                    (rev == 'dS+' and sel == 'S1-' and act == 'dR' and sgn == '+') or
                    (rev == 'dS+' and sel == 'S1-' and act == 'dL' and sgn == '-')
                )
                if dominant_mode:
                    mode = 'dominant'
                else:
                    mode = 'non-dominant'

                
                P2_dict = {ii:[] for ii in np.arange(nbins)}
                for fx in range(nfile):                    
                    if sgn == '+':     
                        _idx_la = learned_activity[f'sess{fx}'][rev][sel][act] > 0
                    elif sgn == '-':
                        _idx_la = learned_activity[f'sess{fx}'][rev][sel][act] < 0
                    _peak_time_P1 = diff_peak_time[f'sess{fx}'][rev][sel][act][1][_idx_la]
                    _peak_time_P2 = diff_peak_time[f'sess{fx}'][rev][sel][act][2][_idx_la]
                    _tr1          = neural_trace[f'sess{fx}'][rev][sel][act][1][_idx_la]
                    _tr2          = neural_trace[f'sess{fx}'][rev][sel][act][2][_idx_la]

                    P1_idx = np.digitize(_peak_time_P1, peak_time)-1    
                    for key, val in zip(P1_idx, _peak_time_P2):
                        P2_dict[key].append(val)    

                P2_histogram = np.zeros((nbins,nbins-1))
                for ii in range(nbins):
                    cnt, bins = np.histogram(P2_dict[ii],bins=peak_time, density=False)
                    P2_histogram[ii] = cnt
                P2_histogram[P2_histogram==0] = 0

                min_tix = np.where(peak_time >= tix_sample)[0][0]
                max_tix = np.where(peak_time >= tix_response+8)[0][0]
                P2_histogram_trim = P2_histogram[min_tix:max_tix,min_tix:max_tix]
                # P2_histogram_trim = P2_histogram_trim / np.sum(P2_histogram_trim)
                _tvec = tvec[peak_time]
                _min_time = _tvec[min_tix]
                _max_time = _tvec[max_tix]
                _idline = _tvec[np.arange(min_tix,max_tix)]
                _tsample = _tvec[peak_time >= tix_sample][0]
                _tdelay  = _tvec[peak_time >= tix_delay][0]
                _tresponse = _tvec[peak_time >= tix_response][0]
        
                # sigma = 1  # controls the width of the Gaussian
                # img_smooth = gaussian_filter(P2_histogram_trim, sigma=sigma)
                vmax = 5*((np.std(P2_histogram_trim) * 3)//5 + 1)
                
                fig, ax = plt.subplots(figsize=(4,3))
                im = plt.imshow(P2_histogram_trim, cmap='jet', origin='lower', aspect='auto', vmin=0, vmax=vmax, extent=[_min_time,_max_time,_min_time,_max_time], rasterized=True)
                plt.plot(_idline,_idline,c='w',linestyle='--')
                plt.axvline(_tsample, color='w', linestyle='--', lw=0.8)
                plt.axvline(_tdelay, color='w', linestyle='--', lw=0.8)
                plt.axvline(_tresponse, color='w', linestyle='--', lw=0.8)
                plt.axhline(_tsample, color='w', linestyle='--', lw=0.8)
                plt.axhline(_tdelay, color='w', linestyle='--', lw=0.8)
                plt.axhline(_tresponse, color='w', linestyle='--', lw=0.8)
                plt.colorbar(im)
                plt.xlim([_min_time,_max_time])
                plt.ylim([_min_time,_max_time])
                plt.tight_layout()
                
                if mode == 'dominant':
                    plt.savefig('figure/neural_dynamics/diff_peak_time/relative_time/model_peak_time/dominant/' + rev + '_' + sel + '_' + act + sgn + '.png',dpi=300)
                    # plt.savefig('figure/neural_dynamics/diff_peak_time/relative_time/model_peak_time/dominant/' + rev + '_' + sel + '_' + act + sgn + '.pdf')
                elif mode == 'non-dominant':
                    plt.savefig('figure/neural_dynamics/diff_peak_time/relative_time/model_peak_time/nondominant/' + rev + '_' + sel + '_' + act + sgn + '.png',dpi=300)
                    # plt.savefig('figure/neural_dynamics/diff_peak_time/relative_time/model_peak_time/nondominant/' + rev + '_' + sel + '_' + act + sgn + '.pdf')                    
                plt.close()

#%%
plt.figure(figsize=(1.5,9))
for i in range(10):
    plt.subplot(10,1,i+1)
    ti = i +10
    plt.plot(P2_histogram_trim[ti])
plt.tight_layout()


ny, nx = P2_histogram_trim.shape

x = np.arange(nx)
y = np.arange(ny)

X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, P2_histogram_trim, cmap='jet',edgecolor='none',
    antialiased=True, alpha=1)
# ax.ploplot_wireframet_surface(X, Y, P2_histogram_trim)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=30, azim=-80)
plt.show()




# n_neurons, n_time = data_dict[act][fx].shape

# fig = plt.figure(figsize=(6, 5))
# ax = fig.add_subplot(111, projection='3d')
# t = np.arange(n_time)

# sort_data_fx = data_dict[act][fx]
# plot_data_fx = data_dict[act][fx]

# tpeak = np.argmax(sort_data_fx, axis=1)
# cell_sort_by_peak_time = np.argsort(tpeak)

# for ci in range(n_neurons):
#     neuron = cell_sort_by_peak_time[ci]

#     ax.plot(
#         t,
#         np.full(n_time, ci),
#         plot_data_fx[neuron],
#         color='k',
#         lw=0.4,
#         alpha=0.2
#     )

# self.axvline_3d(ax, self.tix_sample, n_neurons)
# self.axvline_3d(ax, self.tix_delay, n_neurons)
# self.axvline_3d(ax, self.tix_response, n_neurons)

# ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
#     axis.pane.fill = False
#     axis.pane.set_edgecolor((1, 1, 1, 0))
#     axis.line.set_color((1, 1, 1, 0))

# ax.set_xlabel('time')
# ax.set_ylabel('neuron')

# ax.view_init(elev=30, azim=-70)

# plt.tight_layout()

# idline = np.arange(50)
# plt.figure()
# for fx in range(9):
#     plt.subplot(3,3,fx+1)
#     plt.plot(diff_peak[fx], later_peak[fx],marker='.',linestyle='')
#     plt.plot(idline,idline,c='gray',linestyle='--')
#     plt.plot(-idline,idline,c='gray',linestyle='--')
# plt.tight_layout()
    

# diff_pos = diff_peak_time[f'sess{fx}'][rev][sel][act][0] > 0
# diff_neg = diff_peak_time[f'sess{fx}'][rev][sel][act][0] < 0
# diff_pos_P1 = diff_peak_time[f'sess{fx}'][rev][sel][act][1][diff_pos]
# diff_pos_P2 = diff_peak_time[f'sess{fx}'][rev][sel][act][2][diff_pos]
# diff_neg_P1 = diff_peak_time[f'sess{fx}'][rev][sel][act][1][diff_neg]
# diff_neg_P2 = diff_peak_time[f'sess{fx}'][rev][sel][act][2][diff_neg]

# num_pos = np.sum(diff_pos)
# num_neg = np.sum(diff_neg)
# plt.figure()
# plt.subplot(121)
# diff_pos_P1_sorted = np.argsort(diff_pos_P1)
# for ii in range(num_pos):
#     ii_sorted = diff_pos_P1_sorted[ii]
#     plt.plot(diff_pos_P1[ii_sorted], ii, c='k', marker='.')
#     plt.plot(diff_pos_P2[ii_sorted], ii, c='r', marker='.')
# plt.axvline(tix_sample, color='gray', linestyle='--')
# plt.axvline(tix_delay, color='gray', linestyle='--')
# plt.axvline(tix_response, color='gray', linestyle='--')
# plt.subplot(122)
# diff_neg_P1_sorted = np.argsort(diff_neg_P1)
# for ii in range(num_neg):
#     ii_sorted = diff_neg_P1_sorted[ii]
#     plt.plot(diff_neg_P1[ii_sorted], ii, c='k', marker='.')
#     plt.plot(diff_neg_P2[ii_sorted], ii, c='r', marker='.')
# plt.axvline(tix_sample, color='gray', linestyle='--')
# plt.axvline(tix_delay, color='gray', linestyle='--')
# plt.axvline(tix_response, color='gray', linestyle='--')
# plt.tight_layout()





# plt.figure(figsize=(12,12))
# for fx in range(nfile):
#     plt.subplot(6,6,fx+1)
#     _learned_activity = learned_activity[f'sess{fx}']['dS+']['S1-']['dL']
#     _diff_peak_time   =   diff_peak_time[f'sess{fx}']['dS+']['S1-']['dL']
#     _cor = np.corrcoef(_learned_activity,_diff_peak_time)[0,1]
#     plt.scatter(_learned_activity, _diff_peak_time, s=9)
#     plt.axhline(0,color='gray',linestyle='--')
#     plt.axvline(0,color='gray',linestyle='--')
    
#     idx_pos_act  = _learned_activity >= 0
#     idx_neg_act  = _learned_activity < 0
#     idx_pos_peak = _diff_peak_time >= 0
#     idx_neg_peak = _diff_peak_time < 0
#     frac_pos_act_pos_peak[fx] = np.sum(idx_pos_act * idx_pos_peak) / len(_diff_peak_time)
#     frac_neg_act_pos_peak[fx] = np.sum(idx_neg_act * idx_pos_peak) / len(_diff_peak_time)
#     frac_neg_act_neg_peak[fx] = np.sum(idx_neg_act * idx_neg_peak) / len(_diff_peak_time)
#     frac_pos_act_neg_peak[fx] = np.sum(idx_pos_act * idx_neg_peak) / len(_diff_peak_time)
# # plt.title('cor ' + str(np.round(_cor,decimals=3)))
# plt.tight_layout()



# reverse = ['dS+', 'dS-']
# selectivity = ['S1+', 'S1-']
# delta_activity = [['dP','dA'],['dR','dL']]
# for _rev in reverse:
#     for _sel in selectivity:
#         if _rev == 'dS-':
#             _delta_activity = delta_activity[0]
#         else:
#             _delta_activity = delta_activity[1]
#         for _del in _delta_activity:                        
#             for fx in range(nfile):
#                 _learned_activity = learned_activity[f'sess{fx}'][_rev][_sel][_del]
#                 _diff_peak_time   =   diff_peak_time[f'sess{fx}'][_rev][_sel][_del]
#                 idx_pos_act  = _learned_activity >= 0
#                 idx_neg_act  = _learned_activity < 0
#                 idx_pos_peak = _diff_peak_time >= 0
#                 idx_neg_peak = _diff_peak_time < 0
#                 frac_pos_act_pos_peak[fx] = np.sum(idx_pos_act * idx_pos_peak) / len(_diff_peak_time)
#                 frac_neg_act_pos_peak[fx] = np.sum(idx_neg_act * idx_pos_peak) / len(_diff_peak_time)
#                 frac_neg_act_neg_peak[fx] = np.sum(idx_neg_act * idx_neg_peak) / len(_diff_peak_time)
#                 frac_pos_act_neg_peak[fx] = np.sum(idx_pos_act * idx_neg_peak) / len(_diff_peak_time)
                
#             plt.figure(figsize=(4,4.5))
#             plt.suptitle(_rev + ', ' + _sel + ', ' + _del)
#             plt.subplot(222)
#             plt.plot(CD_dotproduct,frac_pos_act_pos_peak,marker='o',c='b',linestyle='',ms=4, mfc='None', mew=0.8)
#             _cor = np.corrcoef(CD_dotproduct,frac_pos_act_pos_peak)[0,1]
#             plt.annotate(r'$\rho=$' + str(np.round(_cor,decimals=3)),xy=(0.45,0.9),xycoords='axes fraction')
#             plt.ylim([-0.05,0.8])
#             plt.gca().spines[['top', 'right']].set_visible(False)
#             plt.title(_del + '+ ' + r'$\Delta t_p$' + '+')
#             plt.subplot(221)
#             plt.plot(CD_dotproduct,frac_neg_act_pos_peak,marker='o',c='b',linestyle='',ms=4, mfc='None', mew=0.8)
#             _cor = np.corrcoef(CD_dotproduct,frac_neg_act_pos_peak)[0,1]
#             plt.annotate(r'$\rho=$' + str(np.round(_cor,decimals=3)),xy=(0.45,0.9),xycoords='axes fraction')
#             plt.ylim([-0.05,0.8])
#             plt.gca().spines[['top', 'right']].set_visible(False)
#             plt.title(_del + '- ' + r'$\Delta t_p$' + '+')
#             plt.subplot(223)
#             plt.plot(CD_dotproduct,frac_neg_act_neg_peak,marker='o',c='r',linestyle='',ms=4, mfc='None', mew=0.8)
#             _cor = np.corrcoef(CD_dotproduct,frac_neg_act_neg_peak)[0,1]
#             plt.annotate(r'$\rho=$' + str(np.round(_cor,decimals=3)),xy=(0.45,0.9),xycoords='axes fraction')
#             plt.ylim([-0.05,0.8])
#             plt.gca().spines[['top', 'right']].set_visible(False)
#             plt.xlabel('CD dot product')
#             plt.ylabel('frac of cells')
#             plt.title(_del + '- ' + r'$\Delta t_p$' + '-')
#             plt.subplot(224)
#             plt.plot(CD_dotproduct,frac_pos_act_neg_peak,marker='o',c='r',linestyle='',ms=4, mfc='None', mew=0.8)
#             _cor = np.corrcoef(CD_dotproduct,frac_pos_act_neg_peak)[0,1]
#             plt.annotate(r'$\rho=$' + str(np.round(_cor,decimals=3)),xy=(0.45,0.9),xycoords='axes fraction')
#             plt.ylim([-0.05,0.8])
#             plt.gca().spines[['top', 'right']].set_visible(False)
#             plt.title(_del + '+ ' + r'$\Delta t_p$' + '-')
#             plt.tight_layout()
#             plt.savefig('figure/neural_dynamics/diff_peak_time/' + _rev + ', ' + _sel + ', ' + _del + '.pdf')
#             # plt.close()
            

#%%
'''
Shuffle within reversed and stable neurons
'''

# permute = 'paired'
permute = 'not_paired'    
shuff_P1_set1, shuff_P1_set2, shuff_A1_set1, shuff_A1_set2,\
shuff_P2_set1, shuff_P2_set2, shuff_A2_set1, shuff_A2_set2 = functions.shuffle_neural_data(
                                                            nfile, cells_dS, 
                                                            P1_set1, P1_set2, A1_set1, A1_set2, 
                                                            P2_set1, P2_set2, A2_set1, A2_set2, permute)

shuff_CD_dotproduct, \
shuff_S1, shuff_S2, \
shuff_dP, shuff_dA, shuff_dR, shuff_dL = functions.shuffle_compute_dS_dP_dA_dR_dL(
                                            nfile, 
                                            shuff_P1_set1, shuff_P1_set2, shuff_A1_set1, shuff_A1_set2, 
                                            shuff_P2_set1, shuff_P2_set2, shuff_A2_set1, shuff_A2_set2)    
    
shuff_learned_activity = {f'sess{i}': copy.deepcopy(dict_sess) for i in range(nfile)}
shuff_frac_cells       = {f'sess{i}': copy.deepcopy(dict_sess) for i in range(nfile)}    
shuff_learned_activity, shuff_frac_cells = functions.shuffle_compute_learned_activity(
                                                nfile, 
                                                shuff_learned_activity, shuff_frac_cells, 
                                                shuff_S1, shuff_S2, 
                                                shuff_dP, shuff_dA, shuff_dR, shuff_dL)

shuff_learned_activity_sum                       = copy.deepcopy(dict_sess)
shuff_frac_cells_sum                             = copy.deepcopy(dict_sess)
shuff_learned_activity_sum, shuff_frac_cells_sum = functions.compute_learned_activity_sum(shuff_learned_activity, shuff_learned_activity_sum, shuff_frac_cells, shuff_frac_cells_sum, nfile)


#%%
###########################
# mean and std of sessions
###########################
A1_mean = np.zeros(nfile)
A2_mean = np.zeros(nfile)
P1_mean = np.zeros(nfile)
P2_mean = np.zeros(nfile)
A1_std = np.zeros(nfile)
A2_std = np.zeros(nfile)
P1_std = np.zeros(nfile)
P2_std = np.zeros(nfile)
for fx in range(nfile):
    idx_A1 = A1_set2[fx] > 0
    idx_A2 = A2_set2[fx] > 0
    idx_P1 = P1_set2[fx] > 0
    idx_P2 = P2_set2[fx] > 0    
    A1_mean[fx] = np.mean(np.log10(A1_set2[fx][idx_A1]))
    A2_mean[fx] = np.mean(np.log10(A2_set2[fx][idx_A2]))
    P1_mean[fx] = np.mean(np.log10(P1_set2[fx][idx_P1]))
    P2_mean[fx] = np.mean(np.log10(P2_set2[fx][idx_P2]))

    A1_std[fx] = np.std(np.log10(A1_set2[fx][idx_A1]))
    A2_std[fx] = np.std(np.log10(A2_set2[fx][idx_A2]))
    P1_std[fx] = np.std(np.log10(P1_set2[fx][idx_P1]))
    P2_std[fx] = np.std(np.log10(P2_set2[fx][idx_P2]))
    
#######################################################
# Generate log normal data matching data statistics
#######################################################
shift_in_trial2 = -0.03
type1_dP, type1_dA, type2_dP, type2_dA, type3_dR, type3_dL, type4_dR, type4_dL, \
type1_frac_in_quad, type2_frac_in_quad, type3_frac_in_quad, type4_frac_in_quad = functions.gen_lognormal_rate(np.mean(A1_mean), np.mean(A1_std), shift_in_trial2)

synthetic_data = [shift_in_trial2, type1_dP, type1_dA, type2_dP, type2_dA, type3_dR, type3_dL, type4_dR, type4_dL, type1_frac_in_quad, type2_frac_in_quad, type3_frac_in_quad, type4_frac_in_quad]


#%%
'''
Class for plotting the results
'''
importlib.reload(plot_learned_activity)

plots = plot_learned_activity.LearnedActivityPlots(
    learned_activity,
    learned_activity_sum,
    frac_cells,
    frac_cells_sum,
    CD_dotproduct,
    relearn_time,
    shuff_learned_activity,
    shuff_learned_activity_sum,
    shuff_frac_cells,
    shuff_frac_cells_sum,    
    shuff_CD_dotproduct,
    synthetic_data,
    sorted_indices_by_relearn_speed,
    num_outliers,
    num_topcells,
    num_totalcells,    
    merged_ID,
    nfile,
    P1_set2, 
    P2_set2, 
    A1_set2, 
    A2_set2
)

#%%
'''
Plot analysis results
'''

###################################
# CD dot product vs. Relearn time 
###################################
plots.plot_CDdotproduct_learntime(savefig=False)

###############################################################
# fraction of outlier cells (removed) and top cells (analyzed)
###############################################################
plots.plot_outliers_topcells(savefig=False)

#####################################################
# fraction of cells satisfying the sign conditions
#####################################################
plots.plot_frac_with_signs_stat(savefig=False, shuffled=False)
plots.plot_frac_with_signs_quadrant(savefig=False, shuffled=False)

################################
# scatter plot of singl neurons
################################
# all sessions combined
plots.plot_dP_vs_dA_scatter(savefig=False, shuffled=False)
plots.plot_dR_vs_dL_scatter(savefig=False, shuffled=False)

# individual sessions
plots.plot_dP_vs_dA_sessions(savefig=False)
plots.plot_dR_vs_dL_sessions(savefig=False)

##########################################
# (1) CD dot product vs correlation of dP, dA.
# (2) CD dot product vs correlation of dR, dL.
##########################################
plots.plot_correlation_activity(savefig=False, shuffled=False)

#################################################################
# CD dot product vs fraction of satisfying the sign conditions
#################################################################
plots.plot_correlation_frac_cells(savefig=False, shuffled=False)

'''
Plot shuffled data (neurl and synthetic)
'''
##############################
# Shuffled neural data 
##############################
plots.plot_frac_with_signs_stat(savefig=True, shuffled=True)
plots.plot_frac_with_signs_quadrant(savefig=True, shuffled=True)

plots.plot_dP_vs_dA_scatter(savefig=True, shuffled=True)
plots.plot_dR_vs_dL_scatter(savefig=True, shuffled=True)
plots.plot_shuffled_data_scatter_jet(savefig=True)

plots.plot_correlation_activity(savefig=True, shuffled=True)
plots.plot_correlation_frac_cells(savefig=True, shuffled=True)

##############################
# Synthetic log normal data 
##############################
plots.plot_check_neuraldata_is_gaussian(savefig=True)

plots.plot_synthetic_data(savefig=True)



#%%
'''
Mathematical model of data distribution
'''
import numpy as np
from scipy.stats import lognorm
from scipy.integrate import quad

mu = 0.0
sigma = 1.0
Delta = 0.5

F = lognorm(s=sigma, scale=np.exp(mu))
G = lognorm(s=sigma, scale=np.exp(mu + Delta))

def integrand(t):
    Ft = F.cdf(t)
    Gt = G.cdf(t)

    f_t = F.pdf(t)
    g_t = G.pdf(t)

    return Ft * Gt * (f_t * (1 - Gt) + g_t * (1 - Ft))

num, err = quad(integrand, 0, np.inf)

prob = 4 * num

print(prob)
