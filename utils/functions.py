import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import matplotlib.colors as mcolors
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.linear_model import LinearRegression
import h5py 
import os
import copy
from scipy.optimize import curve_fit

def find_all_cells(CD_sess1_population, CD_sess2_population):    
    all_cells_1R, all_cells_2R = np.where(CD_sess1_population >= 0)[0], np.where(CD_sess2_population >= 0)[0]
    all_cells_1L, all_cells_2L = np.where(CD_sess1_population < 0)[0], np.where(CD_sess2_population < 0)[0]
    return all_cells_1R, all_cells_2R, all_cells_1L, all_cells_2L

def find_top_cells(CD_sess1_population, CD_sess2_population, ncell_selected_1L, ncell_selected_2L, ncell_selected_1R, ncell_selected_2R):
    sorted_cells_sess1 = np.argsort(CD_sess1_population)
    sorted_cells_sess2 = np.argsort(CD_sess2_population)
    top_cells_1L  = sorted_cells_sess1[:ncell_selected_1L]
    top_cells_2L  = sorted_cells_sess2[:ncell_selected_2L]
    top_cells_1R = sorted_cells_sess1[-ncell_selected_1R:]
    top_cells_2R = sorted_cells_sess2[-ncell_selected_2R:]
    return top_cells_1L, top_cells_2L, top_cells_1R, top_cells_2R
    
def sf_lick_left(all_cells_1L, all_cells_2L, sf_binary):    
    # cells shared by 1L, 2L
    mask_1L_in_2L       = np.isin(all_cells_1L, all_cells_2L)
    overlap_cells_1L_2L = all_cells_1L[mask_1L_in_2L]
    # cells not shared by 1L, 2L
    mask_1L_notin_2L   = np.isin(all_cells_1L, all_cells_2L, invert=True)
    mask_2L_notin_1L   = np.isin(all_cells_2L, all_cells_1L, invert=True)
    exclusive_cells_1L = all_cells_1L[mask_1L_notin_2L]
    exclusive_cells_2L = all_cells_2L[mask_2L_notin_1L]
    # spatial footprint
    sf_overlap_1L_2L = sf_binary[:,:,overlap_cells_1L_2L]
    sf_exclusive_1L  = sf_binary[:,:,exclusive_cells_1L]
    sf_exclusive_2L  = sf_binary[:,:,exclusive_cells_2L]
    return overlap_cells_1L_2L, exclusive_cells_1L, exclusive_cells_2L, sf_overlap_1L_2L, sf_exclusive_1L, sf_exclusive_2L
    
def sf_lick_right(all_cells_1R, all_cells_2R, sf_binary):    
    # cells shared by 1R, 2R
    mask_1R_in_2R       = np.isin(all_cells_1R, all_cells_2R)
    overlap_cells_1R_2R = all_cells_1R[mask_1R_in_2R]
    # cells not shared by 1R, 2R
    mask_1R_notin_2R   = np.isin(all_cells_1R, all_cells_2R, invert=True)
    mask_2R_notin_1R   = np.isin(all_cells_2R, all_cells_1R, invert=True)
    exclusive_cells_1R = all_cells_1R[mask_1R_notin_2R]
    exclusive_cells_2R = all_cells_2R[mask_2R_notin_1R]
    # spatial footprint
    sf_overlap_1R_2R = sf_binary[:,:,overlap_cells_1R_2R]
    sf_exclusive_1R  = sf_binary[:,:,exclusive_cells_1R]
    sf_exclusive_2R  = sf_binary[:,:,exclusive_cells_2R]
    return overlap_cells_1R_2R, exclusive_cells_1R, exclusive_cells_2R, sf_overlap_1R_2R, sf_exclusive_1R, sf_exclusive_2R

def create_plot_sf(cmap_type, sf):
    vlim = 2
    sf_sum = np.sum(sf, axis=2)    
    sf_sum[sf_sum > 0] = 1    
    vmin, vmax = -vlim, vlim    
    norm = mcolors.Normalize(vmin=vmin,vmax=vmax)
    cmap = plt.get_cmap(cmap_type)
    plot_sf = cmap(norm(sf_sum))
    plot_sf[sf_sum==0, 3] = 0
    return plot_sf    

def gen_trialavg_neural_data(dirpath, par):
    # trial-averaged neural data
    data_trial = {'P1': np.array([]), 'A1': np.array([]), 'P2': np.array([]),'A2': np.array([])}
    data_all = {fx: copy.deepcopy(data_trial) for fx in range(par.nfile)}
    for fx in range(par.nfile):
        print(fx)    
        # sorted by relearn speed 
        fx_sorted = par.sorted_indices_by_relearn_speed[fx]
        filepath   = dirpath + par.merged_ID[fx_sorted] + '.npy'
        data       = np.load(filepath, allow_pickle=True)
        data_type  = 'deconvolved'
        # data_type  = 'dFF0'

        # split data into two sets of trials
        opto_1R_set1, opto_1L_set1, opto_2R_set1, opto_2L_set1, \
        opto_1R_set2, opto_1L_set2, opto_2R_set2, opto_2L_set2 = split_trials(data, data_type)
        # neural activity
        opto_1R_avg_set1 = np.mean(opto_1R_set1,axis=2) # neurons x time x trials
        opto_1L_avg_set1 = np.mean(opto_1L_set1,axis=2) # neurons x time x trials
        opto_2R_avg_set1 = np.mean(opto_2R_set1,axis=2) # neurons x time x trials
        opto_2L_avg_set1 = np.mean(opto_2L_set1,axis=2) # neurons x time x trials
        # neural activity - outlier neurons removed
        data_all[fx]['P1'] = opto_1R_avg_set1[:,par.tix_start:par.tix_end]
        data_all[fx]['A1'] = opto_1L_avg_set1[:,par.tix_start:par.tix_end]
        data_all[fx]['P2'] = opto_2L_avg_set1[:,par.tix_start:par.tix_end]
        data_all[fx]['A2'] = opto_2R_avg_set1[:,par.tix_start:par.tix_end]
    return data_all

def gen_constrain_to_nonoutliers(par, data_all):
    # contstrain the data to nonoutlier neurons
    nonoutlier = {fx:np.array([]) for fx in range(par.nfile)}
    for fx in range(par.nfile):    
        sess = data_all[fx]
        data = {act: sess[act] for act in par.acts}    
        max_vals = [np.max(data[act], axis=1) for act in par.acts]
        keep_nonoutliers = remove_outliers_alltrials_fixedtime(*max_vals, maxstd=5)
        keep_nonzero = np.logical_or.reduce([np.sum(data[act], axis=1) > 0 for act in par.acts])
        keep = keep_nonoutliers & keep_nonzero
        nonoutlier[fx] = np.where(keep)[0]    
        if np.any(~keep_nonzero) > 0:
            print('-----')
            print(fx)
            print(np.where(~keep_nonzero==True))

    data_trial = {'P1': np.array([]), 'A1': np.array([]), 'P2': np.array([]),'A2': np.array([])}
    _data_nonoutlier = {fx: copy.deepcopy(data_trial) for fx in range(par.nfile)}
    for fx in range(par.nfile):
        for act in par.acts:
            _data_nonoutlier[fx][act] = data_all[fx][act][nonoutlier[fx]]
    return _data_nonoutlier
    
def gen_normalize_by_meanrate(par, _data_nonoutlier):
    # get the mean rate of each context
    meanrate_context1 = np.zeros(par.nfile)
    meanrate_context2 = np.zeros(par.nfile)
    for fx in range(par.nfile):
        meanrate_context1[fx] = np.mean(np.concatenate((_data_nonoutlier[fx]['P1'][:,12:16],_data_nonoutlier[fx]['A1'][:,12:16])))
        meanrate_context2[fx] = np.mean(np.concatenate((_data_nonoutlier[fx]['P2'][:,12:16],_data_nonoutlier[fx]['A2'][:,12:16])))
    meanrate = np.mean(np.concatenate((meanrate_context1,meanrate_context2)))    

    # normalize by mean rate
    data_trial = {'P1': np.array([]), 'A1': np.array([]), 'P2': np.array([]),'A2': np.array([])}
    data_nonoutlier = {fx: copy.deepcopy(data_trial) for fx in range(par.nfile)}
    for fx in range(par.nfile):
        for act in par.acts:
            if act == 'P1' or act == 'A1':
                data_nonoutlier[fx][act] = meanrate * _data_nonoutlier[fx][act] / meanrate_context1[fx]
            if act == 'P2' or act == 'A2':
                data_nonoutlier[fx][act] = meanrate * _data_nonoutlier[fx][act] / meanrate_context2[fx]
    return data_nonoutlier

def gen_topcells(thr_var, par, data_nonoutlier):    
    dict_topcells_format = {
        'topcells':                 {i:[] for i in range(par.nfile)},
        'topcells_at_t':            {i:[] for i in range(par.nfile)},
        'data_binary':              {i:[] for i in range(par.nfile)},
    }
    dict_topcells = {act: copy.deepcopy(dict_topcells_format) for act in par.acts}    
    for act in par.acts:
        for fx in range(par.nfile):
            _data   = data_nonoutlier[fx][act]
            _expvar_at_t_fx, _topcells_at_t_fx,  _newcells_at_t_fx, \
            _data_binary_fx, _cells_sorted_at_t, _data_cumsum = create_binary_data(_data,thr_var)
            _topcells_fx = np.unique(np.concatenate(list(_topcells_at_t_fx.values())))
            dict_topcells[act]['topcells'][fx]               = _topcells_fx
            dict_topcells[act]['topcells_at_t'][fx]          = _topcells_at_t_fx
            dict_topcells[act]['data_binary'][fx]            = _data_binary_fx    
    return dict_topcells


def split_trials(data, data_type):
    
    tsample   = 1.57
    tdelay    = 2.87
    tresponse = 4.17
    dt        = 1/6
    ntimestep = 47
    tvec      = dt * np.arange(ntimestep)
    tix_delay    = np.where(tvec > tdelay)[0][0]
    tix_response = np.where(tvec > tresponse)[0][0]

    opto_sess1 = data[data_type][0]
    opto_sess2 = data[data_type][1]
    ncat, ncell = opto_sess1.shape
    lickright = 0
    lickleft  = 1

    ntrials_1R = opto_sess1[lickright,0].shape[1]
    ntrials_1L = opto_sess1[lickleft,0].shape[1]
    ntrials_2R = opto_sess2[lickright,0].shape[1]
    ntrials_2L = opto_sess2[lickleft,0].shape[1]
    
    frac_train_trials = 0.5
    ntrials_train_1R = int(ntrials_1R * frac_train_trials)
    ntrials_train_1L = int(ntrials_1L * frac_train_trials)
    ntrials_train_2R = int(ntrials_2R * frac_train_trials)
    ntrials_train_2L = int(ntrials_2L * frac_train_trials)

    opto_1R_alltrials = np.zeros((ncell,ntimestep,ntrials_1R))
    opto_1L_alltrials = np.zeros((ncell,ntimestep,ntrials_1L))
    opto_2R_alltrials = np.zeros((ncell,ntimestep,ntrials_2R))
    opto_2L_alltrials = np.zeros((ncell,ntimestep,ntrials_2L))
    for cell in range(ncell):
        opto_1R_alltrials[cell] = opto_sess1[lickright,cell]
        opto_1L_alltrials[cell] = opto_sess1[lickleft,cell]
        opto_2R_alltrials[cell] = opto_sess2[lickright,cell]
        opto_2L_alltrials[cell] = opto_sess2[lickleft,cell]
        
    '''
    Compute Coding Direction
    '''
    # train, test trials
    trials_1R, trials_1L, trials_2R, trials_2L = np.arange(ntrials_1R), np.arange(ntrials_1L), np.arange(ntrials_2R), np.arange(ntrials_2L)
    train_trials_1R = np.sort(np.random.permutation(trials_1R)[:ntrials_train_1R])
    train_trials_1L = np.sort(np.random.permutation(trials_1L)[:ntrials_train_1L])
    train_trials_2R = np.sort(np.random.permutation(trials_2R)[:ntrials_train_2R])
    train_trials_2L = np.sort(np.random.permutation(trials_2L)[:ntrials_train_2L])    
    test_trials_1R  = trials_1R[~np.isin(trials_1R,train_trials_1R)]
    test_trials_1L  = trials_1L[~np.isin(trials_1L,train_trials_1L)]
    test_trials_2R  = trials_2R[~np.isin(trials_2R,train_trials_2R)]
    test_trials_2L  = trials_2L[~np.isin(trials_2L,train_trials_2L)]
    
    # neural activity - train, test trials
    opto_1R_traintrials = opto_1R_alltrials[:,:,train_trials_1R]
    opto_1L_traintrials = opto_1L_alltrials[:,:,train_trials_1L]
    opto_2R_traintrials = opto_2R_alltrials[:,:,train_trials_2R]
    opto_2L_traintrials = opto_2L_alltrials[:,:,train_trials_2L]
    
    opto_1R_testtrials  = opto_1R_alltrials[:,:,test_trials_1R]
    opto_1L_testtrials  = opto_1L_alltrials[:,:,test_trials_1L]
    opto_2R_testtrials  = opto_2R_alltrials[:,:,test_trials_2R]
    opto_2L_testtrials  = opto_2L_alltrials[:,:,test_trials_2L]    

    return (opto_1R_traintrials, opto_1L_traintrials, opto_2R_traintrials, opto_2L_traintrials,
            opto_1R_testtrials,  opto_1L_testtrials,  opto_2R_testtrials,  opto_2L_testtrials)
    

def compute_CD_dotproduct_compare_to_JH(data, data_type):
    
    tsample   = 1.57
    tdelay    = 2.87
    tresponse = 4.17
    dt        = 1/6
    ntimestep = 47
    tvec      = dt * np.arange(ntimestep)
    tix_delay    = np.where(tvec > tdelay)[0][0]
    tix_response = np.where(tvec > tresponse)[0][0]

    # organize neural data
    opto_sess1 = data[data_type][0]
    opto_sess2 = data[data_type][1]
    ncat, ncell = opto_sess1.shape
    lickright = 0
    lickleft  = 1

    ntrials_1R = opto_sess1[lickright,0].shape[1]
    ntrials_1L = opto_sess1[lickleft,0].shape[1]
    ntrials_2R = opto_sess2[lickright,0].shape[1]
    ntrials_2L = opto_sess2[lickleft,0].shape[1]

    opto_1R_alltrials = np.zeros((ncell,ntimestep,ntrials_1R))
    opto_1L_alltrials = np.zeros((ncell,ntimestep,ntrials_1L))
    opto_2R_alltrials = np.zeros((ncell,ntimestep,ntrials_2R))
    opto_2L_alltrials = np.zeros((ncell,ntimestep,ntrials_2L))
    for cell in range(ncell):
        opto_1R_alltrials[cell] = opto_sess1[lickright,cell]
        opto_1L_alltrials[cell] = opto_sess1[lickleft,cell]
        opto_2R_alltrials[cell] = opto_sess2[lickright,cell]
        opto_2L_alltrials[cell] = opto_sess2[lickleft,cell]
            
    # random trials        
    frac_train_trials = 0.5
    ntrials_1R_half = int(ntrials_1R * frac_train_trials)
    ntrials_1L_half = int(ntrials_1L * frac_train_trials)
    ntrials_2R_half = int(ntrials_2R * frac_train_trials)
    ntrials_2L_half = int(ntrials_2L * frac_train_trials)

    trials_1R, trials_1L, trials_2R, trials_2L = np.arange(ntrials_1R), np.arange(ntrials_1L), np.arange(ntrials_2R), np.arange(ntrials_2L)
    random_trials_1R_half = np.sort(np.random.permutation(trials_1R)[:ntrials_1R_half])
    random_trials_1L_half = np.sort(np.random.permutation(trials_1L)[:ntrials_1L_half])
    random_trials_2R_half = np.sort(np.random.permutation(trials_2R)[:ntrials_2R_half])
    random_trials_2L_half = np.sort(np.random.permutation(trials_2L)[:ntrials_2L_half])    

    # trial-averaged activity
    opto_1R_trialavg   = np.mean(opto_1R_alltrials[:,:,random_trials_1R_half],axis=2)
    opto_1L_trialavg   = np.mean(opto_1L_alltrials[:,:,random_trials_1L_half],axis=2)
    opto_2R_trialavg   = np.mean(opto_2R_alltrials[:,:,random_trials_2R_half],axis=2)
    opto_2L_trialavg   = np.mean(opto_2L_alltrials[:,:,random_trials_2L_half],axis=2)
                                    
    # compute CD                                
    CD_sess1            = opto_1R_trialavg - opto_1L_trialavg
    CD_sess2            = opto_2R_trialavg - opto_2L_trialavg
    CD_duration         = 4
    CD_sess1_population = np.mean(CD_sess1[:,tix_response-CD_duration:tix_response], axis=1)
    CD_sess2_population = np.mean(CD_sess2[:,tix_response-CD_duration:tix_response], axis=1)
    CD_dotproduct       = np.inner(CD_sess1_population, CD_sess2_population) / (np.linalg.norm(CD_sess1_population) * np.linalg.norm(CD_sess2_population))                    
    
    return CD_dotproduct

    
def compute_CD_dotproduct(opto_1R_set1,opto_1L_set1,opto_2R_set1,opto_2L_set1,tix_response):
    
    ncell     = opto_1R_set1.shape[0]
    ntimestep = opto_1R_set1.shape[1]
    opto_1R_trialavg   = np.zeros((ncell,ntimestep))
    opto_1L_trialavg   = np.zeros((ncell,ntimestep))
    opto_2R_trialavg   = np.zeros((ncell,ntimestep))
    opto_2L_trialavg   = np.zeros((ncell,ntimestep))        
    for cell in range(ncell):        
        # average over trials
        opto_1R_trialavg[cell] = np.mean(opto_1R_set1[cell,:,:],axis=1)
        opto_1L_trialavg[cell] = np.mean(opto_1L_set1[cell,:,:],axis=1)
        opto_2R_trialavg[cell] = np.mean(opto_2R_set1[cell,:,:],axis=1)
        opto_2L_trialavg[cell] = np.mean(opto_2L_set1[cell,:,:],axis=1)                                    
    CD_sess1            = opto_1R_trialavg - opto_1L_trialavg
    CD_sess2            = -(opto_2R_trialavg - opto_2L_trialavg)
    CD_duration         = 4
    CD_sess1_population = np.mean(CD_sess1[:,tix_response-CD_duration:tix_response], axis=1)
    CD_sess2_population = np.mean(CD_sess2[:,tix_response-CD_duration:tix_response], axis=1)
    CD_dotproduct       = np.inner(CD_sess1_population, CD_sess2_population) / (np.linalg.norm(CD_sess1_population) * np.linalg.norm(CD_sess2_population))    
    return CD_dotproduct

def compute_CD_dotproduct_fixed_time(opto_1R,opto_1L,opto_2R,opto_2L):    
    CD_sess1      = opto_1R - opto_1L
    CD_sess2      = -(opto_2R - opto_2L)
    CD_dotproduct = np.inner(CD_sess1, CD_sess2) / (np.linalg.norm(CD_sess1) * np.linalg.norm(CD_sess2))    
    return CD_dotproduct


def fraction_of(x,sgn):
    if sgn == '+':
        frac = np.sum(x>0) / len(x)
    elif sgn == '-':
        frac = np.sum(x<0) / len(x)
    return frac

def fraction_in_quad(x,y):
    ncell = len(x)
    q1    = (x>0) * (y>0)
    q2    = (x<0) * (y>0)
    q3    = (x<0) * (y<0)
    q4    = (x>0) * (y<0)    
    frac_q1 = np.sum(q1) / ncell
    frac_q2 = np.sum(q2) / ncell
    frac_q3 = np.sum(q3) / ncell
    frac_q4 = np.sum(q4) / ncell
    return frac_q1, frac_q2, frac_q3, frac_q4

def compute_learned_activity(learned_activity, frac_cells, 
                             dP, dA, dR, dL, 
                             dS_neg_S1_pos, dS_neg_S1_neg, dS_pos_S1_pos, dS_pos_S1_neg,
                             nfile):
    
    for fx in range(nfile):    
        learned_activity[f'sess{fx}']['dS-']['S1+']['dP'] = dP[fx][dS_neg_S1_pos[fx]]
        learned_activity[f'sess{fx}']['dS-']['S1+']['dA'] = dA[fx][dS_neg_S1_pos[fx]]
        learned_activity[f'sess{fx}']['dS-']['S1-']['dP'] = dP[fx][dS_neg_S1_neg[fx]]
        learned_activity[f'sess{fx}']['dS-']['S1-']['dA'] = dA[fx][dS_neg_S1_neg[fx]]  
        learned_activity[f'sess{fx}']['dS+']['S1+']['dR'] = dR[fx][dS_pos_S1_pos[fx]]
        learned_activity[f'sess{fx}']['dS+']['S1+']['dL'] = dL[fx][dS_pos_S1_pos[fx]]
        learned_activity[f'sess{fx}']['dS+']['S1-']['dR'] = dR[fx][dS_pos_S1_neg[fx]]
        learned_activity[f'sess{fx}']['dS+']['S1-']['dL'] = dL[fx][dS_pos_S1_neg[fx]]

        # marginal distribution
        frac_cells[f'sess{fx}']['dS-']['S1+']['dP<0'] = fraction_of(learned_activity[f'sess{fx}']['dS-']['S1+']['dP'],'-')
        frac_cells[f'sess{fx}']['dS-']['S1+']['dA>0'] = fraction_of(learned_activity[f'sess{fx}']['dS-']['S1+']['dA'],'+')
        frac_cells[f'sess{fx}']['dS-']['S1-']['dP>0'] = fraction_of(learned_activity[f'sess{fx}']['dS-']['S1-']['dP'],'+')
        frac_cells[f'sess{fx}']['dS-']['S1-']['dA<0'] = fraction_of(learned_activity[f'sess{fx}']['dS-']['S1-']['dA'],'-')    
        frac_cells[f'sess{fx}']['dS+']['S1+']['dR<0'] = fraction_of(learned_activity[f'sess{fx}']['dS+']['S1+']['dR'],'-')
        frac_cells[f'sess{fx}']['dS+']['S1+']['dL>0'] = fraction_of(learned_activity[f'sess{fx}']['dS+']['S1+']['dL'],'+')
        frac_cells[f'sess{fx}']['dS+']['S1-']['dR>0'] = fraction_of(learned_activity[f'sess{fx}']['dS+']['S1-']['dR'],'+')
        frac_cells[f'sess{fx}']['dS+']['S1-']['dL<0'] = fraction_of(learned_activity[f'sess{fx}']['dS+']['S1-']['dL'],'-')
        
        # all four quadrants
        frac_cells[f'sess{fx}']['dS-']['S1+']['Quad'] = fraction_in_quad(learned_activity[f'sess{fx}']['dS-']['S1+']['dP'],learned_activity[f'sess{fx}']['dS-']['S1+']['dA'])
        frac_cells[f'sess{fx}']['dS-']['S1-']['Quad'] = fraction_in_quad(learned_activity[f'sess{fx}']['dS-']['S1-']['dP'],learned_activity[f'sess{fx}']['dS-']['S1-']['dA'])
        frac_cells[f'sess{fx}']['dS+']['S1+']['Quad'] = fraction_in_quad(learned_activity[f'sess{fx}']['dS+']['S1+']['dR'],learned_activity[f'sess{fx}']['dS+']['S1+']['dL'])
        frac_cells[f'sess{fx}']['dS+']['S1-']['Quad'] = fraction_in_quad(learned_activity[f'sess{fx}']['dS+']['S1-']['dR'],learned_activity[f'sess{fx}']['dS+']['S1-']['dL'])    
    return learned_activity, frac_cells

def compute_learned_activity_sum(learned_activity, learned_activity_sum, frac_cells, frac_cells_sum, nfile):

    keys_frac_cells_marginal = [
        ('dS-', 'S1+', 'dP<0'),
        ('dS-', 'S1+', 'dA>0'),
        ('dS-', 'S1-', 'dP>0'),
        ('dS-', 'S1-', 'dA<0'),
        ('dS+', 'S1+', 'dR<0'),
        ('dS+', 'S1+', 'dL>0'),
        ('dS+', 'S1-', 'dR>0'),
        ('dS+', 'S1-', 'dL<0'),
    ]
    keys_frac_cells_quad = [
        ('dS-', 'S1+', 'Quad'),
        ('dS-', 'S1-', 'Quad'),
        ('dS+', 'S1+', 'Quad'),
        ('dS+', 'S1-', 'Quad'),
    ]            
    keys_learned_activity = [
        ('dS-', 'S1+', 'dP'),
        ('dS-', 'S1+', 'dA'),
        ('dS-', 'S1-', 'dP'),
        ('dS-', 'S1-', 'dA'),
        ('dS+', 'S1+', 'dR'),
        ('dS+', 'S1+', 'dL'),
        ('dS+', 'S1-', 'dR'),
        ('dS+', 'S1-', 'dL'),
    ]
    for _dcd, _cd1, _sign in keys_frac_cells_marginal:
        frac_cells_sum[_dcd][_cd1][_sign] = np.array([frac_cells[f'sess{fx}'][_dcd][_cd1][_sign] for fx in range(nfile)])
    for _dcd, _cd1, _sign in keys_frac_cells_quad:
        frac_cells_sum[_dcd][_cd1][_sign] = np.vstack([frac_cells[f'sess{fx}'][_dcd][_cd1][_sign] for fx in range(nfile)])        
    for _dcd, _cd1, _var in keys_learned_activity:
        learned_activity_sum[_dcd][_cd1][_var] = np.concatenate([learned_activity[f'sess{fx}'][_dcd][_cd1][_var] for fx in range(nfile)])
    return learned_activity_sum, frac_cells_sum


def compute_diff_peak_time(diff_peak_time, 
                             dP_peak, dA_peak, dR_peak, dL_peak, 
                             dS_neg_S1_pos, dS_neg_S1_neg, dS_pos_S1_pos, dS_pos_S1_neg,
                             nfile):
    for fx in range(nfile):    
        diff_peak_time[f'sess{fx}']['dS-']['S1+']['dP'] = tuple(arr[dS_neg_S1_pos[fx]] for arr in dP_peak[fx])
        diff_peak_time[f'sess{fx}']['dS-']['S1+']['dA'] = tuple(arr[dS_neg_S1_pos[fx]] for arr in dA_peak[fx])
        diff_peak_time[f'sess{fx}']['dS-']['S1-']['dP'] = tuple(arr[dS_neg_S1_neg[fx]] for arr in dP_peak[fx])
        diff_peak_time[f'sess{fx}']['dS-']['S1-']['dA'] = tuple(arr[dS_neg_S1_neg[fx]] for arr in dA_peak[fx])
        diff_peak_time[f'sess{fx}']['dS+']['S1+']['dR'] = tuple(arr[dS_pos_S1_pos[fx]] for arr in dR_peak[fx])
        diff_peak_time[f'sess{fx}']['dS+']['S1+']['dL'] = tuple(arr[dS_pos_S1_pos[fx]] for arr in dL_peak[fx])
        diff_peak_time[f'sess{fx}']['dS+']['S1-']['dR'] = tuple(arr[dS_pos_S1_neg[fx]] for arr in dR_peak[fx])
        diff_peak_time[f'sess{fx}']['dS+']['S1-']['dL'] = tuple(arr[dS_pos_S1_neg[fx]] for arr in dL_peak[fx])        
    return diff_peak_time

            
def compute_S1_S2(P1, P2, A1, A2):
    S1 = P1 - A1
    S2 = P2 - A2
    return S1, S2
                
def compute_dP_dA_dR_dL(P1, P2, A1, A2):
    dP = P2 - P1
    dA = A2 - A1
    dR = A2 - P1
    dL = P2 - A1
    return dP, dA, dR, dL    

def compute_dP_dA_dR_dL_peak(P1, P2, A1, A2):
    dP = (P2 - P1, P1, P2)
    dA = (A2 - A1, A1, A2)
    dR = (A2 - P1, P1, A2)
    dL = (P2 - A1, A1, P2)
    return dP, dA, dR, dL    

def find_topcells(dS, cells, thr):
    # trim cells
    dS_trim = np.abs(dS[cells])
    # sort the CD dot product (descending)
    cells_sorted = np.argsort(dS_trim)[::-1]
    # cumulative sum of the sorted CD dot product
    dS_cumsum = np.cumsum(dS_trim[cells_sorted])
    dS_cumsum_norm = dS_cumsum / dS_cumsum[-1]
    # find the top 99%
    topcells = cells[cells_sorted[:np.where(dS_cumsum_norm > thr)[0][0]]]
    return topcells
            
def neuron_types(S1, S2, dS, 
                 cells_pos, cells_neg, 
                 topcells_dS_pos, topcells_dS_neg,
                 cells_dS_neg_S1_pos, cells_dS_neg_S1_neg, 
                 cells_dS_pos_S1_pos, cells_dS_pos_S1_neg,
                 cells_dS_neg, cells_dS_pos, cells_dS,
                 num_totalcells, num_topcells, nfile):
    for fx in range(nfile):
        # find top cells
        dS[fx]              = S1[fx] * S2[fx]
        cells_pos[fx]       = np.where(dS[fx] > 0)[0]
        cells_neg[fx]       = np.where(dS[fx] < 0)[0]
        topcells_dS_pos_fx  = find_topcells(dS[fx], cells_pos[fx], thr=0.99)
        topcells_dS_neg_fx  = find_topcells(dS[fx], cells_neg[fx], thr=0.99)            
        topcells_dS_pos[fx] = topcells_dS_pos_fx
        topcells_dS_neg[fx] = topcells_dS_neg_fx 

        # four neuron types
        cells_dS_neg_S1_pos[fx] = topcells_dS_neg_fx[S1[fx][topcells_dS_neg_fx] > 0]
        cells_dS_neg_S1_neg[fx] = topcells_dS_neg_fx[S1[fx][topcells_dS_neg_fx] < 0]
        cells_dS_pos_S1_pos[fx] = topcells_dS_pos_fx[S1[fx][topcells_dS_pos_fx] > 0]
        cells_dS_pos_S1_neg[fx] = topcells_dS_pos_fx[S1[fx][topcells_dS_pos_fx] < 0]
        # reversed and stable neurons
        cells_dS_neg[fx]       = np.append(cells_dS_neg_S1_pos[fx],cells_dS_neg_S1_neg[fx])
        cells_dS_pos[fx]       = np.append(cells_dS_pos_S1_pos[fx],cells_dS_pos_S1_neg[fx])    
        # all cells
        cells_dS[fx]           = np.append(cells_dS_neg[fx],cells_dS_pos[fx])    

        # top cell count
        num_totalcells[fx] = len(S1[fx])    
        num_topcells[fx]   = len(topcells_dS_pos[fx]) + len(topcells_dS_neg[fx])        
    return (dS, cells_pos, cells_neg, 
            topcells_dS_pos, topcells_dS_neg, 
            cells_dS_neg_S1_pos, cells_dS_neg_S1_neg, 
            cells_dS_pos_S1_pos, cells_dS_pos_S1_neg, 
            cells_dS_neg, cells_dS_pos, cells_dS, 
            num_totalcells, num_topcells)
            
def compute_CD_delay(data, data_type, reference):
    
    tsample   = 1.57
    tdelay    = 2.87
    tresponse = 4.17
    dt        = 1/6
    ntimestep = 47
    tvec      = dt * np.arange(ntimestep)
    tix_delay    = np.where(tvec > tdelay)[0][0]
    tix_response = np.where(tvec > tresponse)[0][0]

    opto_sess1 = data[data_type][0]
    opto_sess2 = data[data_type][1]
    ncat, ncell = opto_sess1.shape
    lickright = 0
    lickleft  = 1

    ntrials_1R = opto_sess1[lickright,0].shape[1]
    ntrials_1L = opto_sess1[lickleft,0].shape[1]
    ntrials_2R = opto_sess2[lickright,0].shape[1]
    ntrials_2L = opto_sess2[lickleft,0].shape[1]
    
    frac_train_trials = 0.5
    ntrials_train_1R = int(ntrials_1R * frac_train_trials)
    ntrials_train_1L = int(ntrials_1L * frac_train_trials)
    ntrials_train_2R = int(ntrials_2R * frac_train_trials)
    ntrials_train_2L = int(ntrials_2L * frac_train_trials)

    opto_1R_alltrials = np.zeros((ncell,ntimestep,ntrials_1R))
    opto_1L_alltrials = np.zeros((ncell,ntimestep,ntrials_1L))
    opto_2R_alltrials = np.zeros((ncell,ntimestep,ntrials_2R))
    opto_2L_alltrials = np.zeros((ncell,ntimestep,ntrials_2L))
    for cell in range(ncell):
        opto_1R_alltrials[cell] = opto_sess1[lickright,cell]
        opto_1L_alltrials[cell] = opto_sess1[lickleft,cell]
        opto_2R_alltrials[cell] = opto_sess2[lickright,cell]
        opto_2L_alltrials[cell] = opto_sess2[lickleft,cell]
        
    '''
    Compute Coding Direction
    '''
    # train, test trials
    trials_1R, trials_1L, trials_2R, trials_2L = np.arange(ntrials_1R), np.arange(ntrials_1L), np.arange(ntrials_2R), np.arange(ntrials_2L)
    train_trials_1R = np.sort(np.random.permutation(trials_1R)[:ntrials_train_1R])
    train_trials_1L = np.sort(np.random.permutation(trials_1L)[:ntrials_train_1L])
    train_trials_2R = np.sort(np.random.permutation(trials_2R)[:ntrials_train_2R])
    train_trials_2L = np.sort(np.random.permutation(trials_2L)[:ntrials_train_2L])    
    test_trials_1R  = trials_1R[~np.isin(trials_1R,train_trials_1R)]
    test_trials_1L  = trials_1L[~np.isin(trials_1L,train_trials_1L)]
    test_trials_2R  = trials_2R[~np.isin(trials_2R,train_trials_2R)]
    test_trials_2L  = trials_2L[~np.isin(trials_2L,train_trials_2L)]
    
    # train trial neural activity
    opto_1R = np.zeros((ncell,ntimestep))
    opto_1L = np.zeros((ncell,ntimestep))
    opto_2R = np.zeros((ncell,ntimestep))
    opto_2L = np.zeros((ncell,ntimestep))
    for cell in range(ncell):        
        #------- first half trials ----------#
        opto_1R[cell] = np.mean(opto_1R_alltrials[cell,:,train_trials_1R].T,axis=1)
        opto_1L[cell] = np.mean(opto_1L_alltrials[cell,:,train_trials_1L].T,axis=1)
        opto_2R[cell] = np.mean(opto_2R_alltrials[cell,:,train_trials_2R].T,axis=1)
        opto_2L[cell] = np.mean(opto_2L_alltrials[cell,:,train_trials_2L].T,axis=1)    
            
    opto_1R_traintrials = opto_1R_alltrials[:,:,train_trials_1R]
    opto_1L_traintrials = opto_1L_alltrials[:,:,train_trials_1L]
    opto_2R_traintrials = opto_2R_alltrials[:,:,train_trials_2R]
    opto_2L_traintrials = opto_2L_alltrials[:,:,train_trials_2L]
            
    ########################################
    #   CD_sess1
    #   - output: right     - left
    #   - input:  posterior - anterior
    ########################################
    CD_sess1            = opto_1R - opto_1L
    if reference == 'output':
        ########################################
        #   CD_sess2
        #   - output: right    - left
        #   - input:  anterior - posterior
        ########################################
        CD_sess2 = opto_2R - opto_2L
    elif reference == 'input':
        ########################################
        #   CD_sess2
        #   - output: left      - right
        #   - input:  posterior - anterior
        ########################################
        CD_sess2 = -(opto_2R - opto_2L)
    CD_duration         = 4
    CD_sess1_population = np.mean(CD_sess1[:,tix_response-CD_duration:tix_response], axis=1)
    CD_sess2_population = np.mean(CD_sess2[:,tix_response-CD_duration:tix_response], axis=1)
    CD_dotproduct       = np.inner(CD_sess1_population, CD_sess2_population) / (np.linalg.norm(CD_sess1_population) * np.linalg.norm(CD_sess2_population))

    '''
    Project activity to coding dimension
    '''
    opto_1R_testtrials = opto_1R_alltrials[:,:,test_trials_1R]
    opto_1L_testtrials = opto_1L_alltrials[:,:,test_trials_1L]
    opto_2R_testtrials = opto_2R_alltrials[:,:,test_trials_2R]
    opto_2L_testtrials = opto_2L_alltrials[:,:,test_trials_2L]

    proj_1R = np.tensordot(CD_sess1_population, opto_1R_testtrials, axes=(0,0))
    proj_1L = np.tensordot(CD_sess1_population, opto_1L_testtrials, axes=(0,0))
    proj_2R = np.tensordot(CD_sess2_population, opto_2R_testtrials, axes=(0,0))
    proj_2L = np.tensordot(CD_sess2_population, opto_2L_testtrials, axes=(0,0))

    return (CD_dotproduct, CD_sess1_population, CD_sess2_population, 
            proj_1R, proj_1L, proj_2R, proj_2L,
            opto_1R_testtrials, opto_1L_testtrials, opto_2R_testtrials, opto_2L_testtrials,
            opto_1R_traintrials, opto_1L_traintrials, opto_2R_traintrials, opto_2L_traintrials)
    

def compute_CD_sample_update(data, data_type):
    
    tsample   = 1.57
    tdelay    = 2.87
    tresponse = 4.17
    dt        = 1/6
    ntimestep = 47
    tvec      = dt * np.arange(ntimestep)
    tix_sample   = np.where(tvec > tsample)[0][0]
    tix_delay    = np.where(tvec > tdelay)[0][0]
    tix_response = np.where(tvec > tresponse)[0][0]

    opto_sess1 = data[data_type][0]
    opto_sess2 = data[data_type][1]
    ncat, ncell = opto_sess1.shape
    lickright = 0
    lickleft  = 1

    ntrials_1P = opto_sess1[lickright,0].shape[1]
    ntrials_1A = opto_sess1[lickleft,0].shape[1]
    ntrials_2A = opto_sess2[lickright,0].shape[1]
    ntrials_2P = opto_sess2[lickleft,0].shape[1]
    
    frac_train_trials = 0.5
    ntrials_train_1P = int(ntrials_1P * frac_train_trials)
    ntrials_train_1A = int(ntrials_1A * frac_train_trials)
    ntrials_train_2P = int(ntrials_2P * frac_train_trials)
    ntrials_train_2A = int(ntrials_2A * frac_train_trials)

    opto_1P_alltrials = np.zeros((ncell,ntimestep,ntrials_1P))
    opto_1A_alltrials = np.zeros((ncell,ntimestep,ntrials_1A))
    opto_2P_alltrials = np.zeros((ncell,ntimestep,ntrials_2P))
    opto_2A_alltrials = np.zeros((ncell,ntimestep,ntrials_2A))
    for cell in range(ncell):
        opto_1P_alltrials[cell] = opto_sess1[lickright,cell]
        opto_1A_alltrials[cell] = opto_sess1[lickleft,cell]
        opto_2P_alltrials[cell] = opto_sess2[lickleft,cell]
        opto_2A_alltrials[cell] = opto_sess2[lickright,cell]
        
    '''
    Compute Coding Direction
    '''        
    # train, test trials
    trials_1P, trials_1A, trials_2P, trials_2A = np.arange(ntrials_1P), np.arange(ntrials_1A), np.arange(ntrials_2P), np.arange(ntrials_2A)
    train_trials_1P = np.sort(np.random.permutation(trials_1P)[:ntrials_train_1P])
    train_trials_1A = np.sort(np.random.permutation(trials_1A)[:ntrials_train_1A])
    train_trials_2P = np.sort(np.random.permutation(trials_2P)[:ntrials_train_2P])
    train_trials_2A = np.sort(np.random.permutation(trials_2A)[:ntrials_train_2A])    
    test_trials_1P  = trials_1P[~np.isin(trials_1P,train_trials_1P)]
    test_trials_1A  = trials_1A[~np.isin(trials_1A,train_trials_1A)]
    test_trials_2P  = trials_2P[~np.isin(trials_2P,train_trials_2P)]
    test_trials_2A  = trials_2A[~np.isin(trials_2A,train_trials_2A)]
    
    # train trial neural activity
    opto_1P = np.zeros((ncell,ntimestep))
    opto_1A = np.zeros((ncell,ntimestep))
    opto_2P = np.zeros((ncell,ntimestep))
    opto_2A = np.zeros((ncell,ntimestep))
    for cell in range(ncell):        
        #------- first half trials ----------#
        opto_1P[cell] = np.mean(opto_1P_alltrials[cell,:,train_trials_1P].T,axis=1)
        opto_1A[cell] = np.mean(opto_1A_alltrials[cell,:,train_trials_1A].T,axis=1)
        opto_2A[cell] = np.mean(opto_2P_alltrials[cell,:,train_trials_2P].T,axis=1)    
        opto_2P[cell] = np.mean(opto_2A_alltrials[cell,:,train_trials_2A].T,axis=1)
        
    # CD sample                
    opto_1P_sample         = np.mean(opto_1P[:,tix_sample:tix_sample+4],axis=1)
    opto_1A_sample         = np.mean(opto_1A[:,tix_sample:tix_sample+4],axis=1)
    opto_2P_sample         = np.mean(opto_2P[:,tix_sample:tix_sample+4],axis=1)
    opto_2A_sample         = np.mean(opto_2A[:,tix_sample:tix_sample+4],axis=1)
    opto_1_sample_mean     = np.mean(opto_1P_sample + opto_1A_sample)
    opto_2_sample_mean     = np.mean(opto_2P_sample + opto_2A_sample)
    norm_1P_sample         = opto_1P_sample / opto_1_sample_mean
    norm_1A_sample         = opto_1A_sample / opto_1_sample_mean
    norm_2P_sample         = opto_2P_sample / opto_2_sample_mean
    norm_2A_sample         = opto_2A_sample / opto_2_sample_mean    
    CD_sample_1_population = norm_1P_sample - norm_1A_sample
    CD_sample_2_population = norm_2P_sample - norm_2A_sample
    CD_sample_dotproduct   = np.inner(CD_sample_1_population, CD_sample_2_population) / (np.linalg.norm(CD_sample_1_population) * np.linalg.norm(CD_sample_2_population))

    return (CD_sample_dotproduct, CD_sample_1_population, CD_sample_2_population)
    
                
def compute_CD_sample(data, data_type):
    
    tsample   = 1.57
    tdelay    = 2.87
    tresponse = 4.17
    dt        = 1/6
    ntimestep = 47
    tvec      = dt * np.arange(ntimestep)
    tix_sample   = np.where(tvec > tsample)[0][0]
    tix_delay    = np.where(tvec > tdelay)[0][0]
    tix_response = np.where(tvec > tresponse)[0][0]

    opto_sess1 = data[data_type][0]
    opto_sess2 = data[data_type][1]
    ncat, ncell = opto_sess1.shape
    lickright = 0
    lickleft  = 1

    ntrials_1P = opto_sess1[lickright,0].shape[1]
    ntrials_1A = opto_sess1[lickleft,0].shape[1]
    ntrials_2A = opto_sess2[lickright,0].shape[1]
    ntrials_2P = opto_sess2[lickleft,0].shape[1]
    
    frac_train_trials = 0.5
    ntrials_train_1P = int(ntrials_1P * frac_train_trials)
    ntrials_train_1A = int(ntrials_1A * frac_train_trials)
    ntrials_train_2P = int(ntrials_2P * frac_train_trials)
    ntrials_train_2A = int(ntrials_2A * frac_train_trials)

    opto_1P_alltrials = np.zeros((ncell,ntimestep,ntrials_1P))
    opto_1A_alltrials = np.zeros((ncell,ntimestep,ntrials_1A))
    opto_2P_alltrials = np.zeros((ncell,ntimestep,ntrials_2P))
    opto_2A_alltrials = np.zeros((ncell,ntimestep,ntrials_2A))
    for cell in range(ncell):
        opto_1P_alltrials[cell] = opto_sess1[lickright,cell]
        opto_1A_alltrials[cell] = opto_sess1[lickleft,cell]
        opto_2P_alltrials[cell] = opto_sess2[lickleft,cell]
        opto_2A_alltrials[cell] = opto_sess2[lickright,cell]
        
    '''
    Compute Coding Direction
    '''
        
    # train, test trials
    trials_1P, trials_1A, trials_2P, trials_2A = np.arange(ntrials_1P), np.arange(ntrials_1A), np.arange(ntrials_2P), np.arange(ntrials_2A)
    train_trials_1P = np.sort(np.random.permutation(trials_1P)[:ntrials_train_1P])
    train_trials_1A = np.sort(np.random.permutation(trials_1A)[:ntrials_train_1A])
    train_trials_2P = np.sort(np.random.permutation(trials_2P)[:ntrials_train_2P])
    train_trials_2A = np.sort(np.random.permutation(trials_2A)[:ntrials_train_2A])    
    test_trials_1P  = trials_1P[~np.isin(trials_1P,train_trials_1P)]
    test_trials_1A  = trials_1A[~np.isin(trials_1A,train_trials_1A)]
    test_trials_2P  = trials_2P[~np.isin(trials_2P,train_trials_2P)]
    test_trials_2A  = trials_2A[~np.isin(trials_2A,train_trials_2A)]
    
    # train trial neural activity
    opto_1P = np.zeros((ncell,ntimestep))
    opto_1A = np.zeros((ncell,ntimestep))
    opto_2P = np.zeros((ncell,ntimestep))
    opto_2A = np.zeros((ncell,ntimestep))
    for cell in range(ncell):        
        #------- first half trials ----------#
        opto_1P[cell] = np.mean(opto_1P_alltrials[cell,:,train_trials_1P].T,axis=1)
        opto_1A[cell] = np.mean(opto_1A_alltrials[cell,:,train_trials_1A].T,axis=1)
        opto_2A[cell] = np.mean(opto_2P_alltrials[cell,:,train_trials_2P].T,axis=1)    
        opto_2P[cell] = np.mean(opto_2A_alltrials[cell,:,train_trials_2A].T,axis=1)
        
    # CD sample                
    CD_sample_1            = opto_1P - opto_1A
    CD_sample_2            = opto_2P - opto_2A
    CD_sample_1_population = np.mean(CD_sample_1[:,tix_sample:tix_sample+4], axis=1)
    CD_sample_2_population = np.mean(CD_sample_2[:,tix_sample:tix_sample+4], axis=1)
    CD_sample_dotproduct   = np.inner(CD_sample_1_population, CD_sample_2_population) / (np.linalg.norm(CD_sample_1_population) * np.linalg.norm(CD_sample_2_population))

    '''
    Project activity to coding dimension
    '''
    opto_1P_testtrials = opto_1P_alltrials[:,:,test_trials_1P]
    opto_1A_testtrials = opto_1A_alltrials[:,:,test_trials_1A]
    opto_2P_testtrials = opto_2P_alltrials[:,:,test_trials_2P]
    opto_2A_testtrials = opto_2A_alltrials[:,:,test_trials_2A]
    
    sample_1P_testtrials = np.mean(opto_1P_testtrials[:,tix_sample:tix_sample+4,:],axis=(1,2))
    sample_1A_testtrials = np.mean(opto_1A_testtrials[:,tix_sample:tix_sample+4,:],axis=(1,2))
    sample_2P_testtrials = np.mean(opto_2P_testtrials[:,tix_sample:tix_sample+4,:],axis=(1,2))
    sample_2A_testtrials = np.mean(opto_2A_testtrials[:,tix_sample:tix_sample+4,:],axis=(1,2))
    
    cossim_sample_1P = cosine_similarity(CD_sample_1_population, sample_1P_testtrials)
    cossim_sample_1A = cosine_similarity(CD_sample_1_population, sample_1A_testtrials)
    cossim_sample_2P = cosine_similarity(CD_sample_2_population, sample_2P_testtrials)
    cossim_sample_2A = cosine_similarity(CD_sample_2_population, sample_2A_testtrials)

    project_sample_1P = project_to(CD_sample_1_population, sample_1P_testtrials)
    project_sample_1A = project_to(CD_sample_1_population, sample_1A_testtrials)
    project_sample_2P = project_to(CD_sample_2_population, sample_2P_testtrials)
    project_sample_2A = project_to(CD_sample_2_population, sample_2A_testtrials)

    return (CD_sample_dotproduct, CD_sample_1_population, CD_sample_2_population,
            cossim_sample_1P, cossim_sample_1A, cossim_sample_2P, cossim_sample_2A,
            project_sample_1P, project_sample_1A, project_sample_2P, project_sample_2A,)
    

def compute_CD_sample_late(data, data_type, tix):
    
    tsample   = 1.57
    tdelay    = 2.87
    tresponse = 4.17
    dt        = 1/6
    ntimestep = 47
    tvec      = dt * np.arange(ntimestep)
    tix_sample   = np.where(tvec > tsample)[0][0]
    tix_delay    = np.where(tvec > tdelay)[0][0]
    tix_response = np.where(tvec > tresponse)[0][0]

    opto_sess1 = data[data_type][0]
    opto_sess2 = data[data_type][1]
    ncat, ncell = opto_sess1.shape
    lickright = 0
    lickleft  = 1

    ntrials_1P = opto_sess1[lickright,0].shape[1]
    ntrials_1A = opto_sess1[lickleft,0].shape[1]
    ntrials_2A = opto_sess2[lickright,0].shape[1]
    ntrials_2P = opto_sess2[lickleft,0].shape[1]
    
    frac_train_trials = 0.5
    ntrials_train_1P = int(ntrials_1P * frac_train_trials)
    ntrials_train_1A = int(ntrials_1A * frac_train_trials)
    ntrials_train_2P = int(ntrials_2P * frac_train_trials)
    ntrials_train_2A = int(ntrials_2A * frac_train_trials)

    opto_1P_alltrials = np.zeros((ncell,ntimestep,ntrials_1P))
    opto_1A_alltrials = np.zeros((ncell,ntimestep,ntrials_1A))
    opto_2P_alltrials = np.zeros((ncell,ntimestep,ntrials_2P))
    opto_2A_alltrials = np.zeros((ncell,ntimestep,ntrials_2A))
    for cell in range(ncell):
        opto_1P_alltrials[cell] = opto_sess1[lickright,cell]
        opto_1A_alltrials[cell] = opto_sess1[lickleft,cell]
        opto_2P_alltrials[cell] = opto_sess2[lickleft,cell]
        opto_2A_alltrials[cell] = opto_sess2[lickright,cell]
        
    '''
    Compute Coding Direction
    '''        
    # train, test trials
    trials_1P, trials_1A, trials_2P, trials_2A = np.arange(ntrials_1P), np.arange(ntrials_1A), np.arange(ntrials_2P), np.arange(ntrials_2A)
    train_trials_1P = np.sort(np.random.permutation(trials_1P)[:ntrials_train_1P])
    train_trials_1A = np.sort(np.random.permutation(trials_1A)[:ntrials_train_1A])
    train_trials_2P = np.sort(np.random.permutation(trials_2P)[:ntrials_train_2P])
    train_trials_2A = np.sort(np.random.permutation(trials_2A)[:ntrials_train_2A])    
    
    # train trial neural activity
    opto_1P = np.zeros((ncell,ntimestep))
    opto_1A = np.zeros((ncell,ntimestep))
    opto_2P = np.zeros((ncell,ntimestep))
    opto_2A = np.zeros((ncell,ntimestep))
    for cell in range(ncell):        
        #------- first half trials ----------#
        opto_1P[cell] = np.mean(opto_1P_alltrials[cell,:,train_trials_1P].T,axis=1)
        opto_1A[cell] = np.mean(opto_1A_alltrials[cell,:,train_trials_1A].T,axis=1)
        opto_2A[cell] = np.mean(opto_2P_alltrials[cell,:,train_trials_2P].T,axis=1)    
        opto_2P[cell] = np.mean(opto_2A_alltrials[cell,:,train_trials_2A].T,axis=1)
        
    # CD sample                
    tix_start            = tix 
    tix_end              = tix + 4
    opto_1P_sample_late  = np.mean(opto_1P[:,tix_start:tix_end],axis=1)
    opto_1A_sample_late  = np.mean(opto_1A[:,tix_start:tix_end],axis=1)
    opto_2P_sample_late  = np.mean(opto_2P[:,tix_start:tix_end],axis=1)
    opto_2A_sample_late  = np.mean(opto_2A[:,tix_start:tix_end],axis=1)
    CD_sample_late_1     = opto_1P_sample_late - opto_1A_sample_late
    CD_sample_late_2     = opto_2P_sample_late - opto_2A_sample_late
    CD_sample_late_1     = CD_sample_late_1 / np.linalg.norm(CD_sample_late_1)
    CD_sample_late_2     = CD_sample_late_2 / np.linalg.norm(CD_sample_late_2)
    CD_sample_late_dotproduct = np.inner(CD_sample_late_1, CD_sample_late_2) / (np.linalg.norm(CD_sample_late_1) * np.linalg.norm(CD_sample_late_2))

    return (CD_sample_late_1, CD_sample_late_2)
    
                
def compute_CD_context(data, data_type):
    
    tsample   = 1.57
    tdelay    = 2.87
    tresponse = 4.17
    dt        = 1/6
    ntimestep = 47
    tvec      = dt * np.arange(ntimestep)
    tix_sample   = np.where(tvec > tsample)[0][0]
    tix_delay    = np.where(tvec > tdelay)[0][0]
    tix_response = np.where(tvec > tresponse)[0][0]

    opto_sess1 = data[data_type][0]
    opto_sess2 = data[data_type][1]
    ncat, ncell = opto_sess1.shape
    lickright = 0
    lickleft  = 1

    ntrials_1P = opto_sess1[lickright,0].shape[1]
    ntrials_1A = opto_sess1[lickleft,0].shape[1]
    ntrials_2A = opto_sess2[lickright,0].shape[1]
    ntrials_2P = opto_sess2[lickleft,0].shape[1]
    
    frac_train_trials = 0.5
    ntrials_train_1P = int(ntrials_1P * frac_train_trials)
    ntrials_train_1A = int(ntrials_1A * frac_train_trials)
    ntrials_train_2A = int(ntrials_2A * frac_train_trials)
    ntrials_train_2P = int(ntrials_2P * frac_train_trials)

    opto_1P_alltrials = np.zeros((ncell,ntimestep,ntrials_1P))
    opto_1A_alltrials = np.zeros((ncell,ntimestep,ntrials_1A))
    opto_2A_alltrials = np.zeros((ncell,ntimestep,ntrials_2A))
    opto_2P_alltrials = np.zeros((ncell,ntimestep,ntrials_2P))
    for cell in range(ncell):
        opto_1P_alltrials[cell] = opto_sess1[lickright,cell]
        opto_1A_alltrials[cell] = opto_sess1[lickleft,cell]
        opto_2A_alltrials[cell] = opto_sess2[lickright,cell]
        opto_2P_alltrials[cell] = opto_sess2[lickleft,cell]
        
    '''
    Compute Coding Direction - Context
    '''
    opto_1P = np.zeros((ncell,ntimestep))
    opto_1A = np.zeros((ncell,ntimestep))
    opto_2A = np.zeros((ncell,ntimestep))
    opto_2P = np.zeros((ncell,ntimestep))
    for cell in range(ncell):        
        #------- first half trials ----------#
        opto_1P[cell] = np.mean(opto_1P_alltrials[cell,:,:ntrials_train_1P],axis=1)
        opto_1A[cell] = np.mean(opto_1A_alltrials[cell,:,:ntrials_train_1A],axis=1)
        opto_2A[cell] = np.mean(opto_2A_alltrials[cell,:,:ntrials_train_2A],axis=1)
        opto_2P[cell] = np.mean(opto_2P_alltrials[cell,:,:ntrials_train_2P],axis=1)    
        
    CD_context_P            = opto_1P - opto_2P
    CD_context_A            = opto_1A - opto_2A
    CD_context_P_population = np.mean(CD_context_P[:,tix_sample:tix_sample+5], axis=1)
    CD_context_A_population = np.mean(CD_context_A[:,tix_sample:tix_sample+5], axis=1)
    CD_context_dotproduct   = np.inner(CD_context_A_population, CD_context_P_population) / (np.linalg.norm(CD_context_A_population) * np.linalg.norm(CD_context_P_population))

    '''
    Project activity to coding dimension
    '''
    opto_1P_testtrials = opto_1P_alltrials[:,:,ntrials_train_1P:]
    opto_1A_testtrials = opto_1A_alltrials[:,:,ntrials_train_1A:]
    opto_2A_testtrials = opto_2A_alltrials[:,:,ntrials_train_2A:]
    opto_2P_testtrials = opto_2P_alltrials[:,:,ntrials_train_2P:]

    proj_1P = np.tensordot(CD_context_P_population, opto_1P_testtrials, axes=(0,0))
    proj_1A = np.tensordot(CD_context_A_population, opto_1A_testtrials, axes=(0,0))
    proj_2A = np.tensordot(CD_context_A_population, opto_2A_testtrials, axes=(0,0))
    proj_2P = np.tensordot(CD_context_P_population, opto_2P_testtrials, axes=(0,0))

    return (CD_context_dotproduct, CD_context_P_population, CD_context_A_population,
            proj_1P, proj_1A, proj_2A, proj_2P)
    
    
def compute_TD_cossim(
    CD_proj_1R, CD_proj_1L, CD_proj_2R, CD_proj_2L, 
    tix_response, tix_delay, 
    opto_1R_testtrials, opto_1L_testtrials, opto_2L_testtrials, opto_2R_testtrials,
    cossim_TD_1P_all, cossim_TD_1A_all, cossim_TD_2P_all, cossim_TD_2A_all,
    shuff, fx,
    ID_P, ID_A
    ):
    
    # Transient Direction 
    ydata_1P = CD_proj_1R[tix_response,:]
    ydata_1A = CD_proj_1L[tix_response,:]
    ydata_2P = CD_proj_2L[tix_response,:]    
    ydata_2A = CD_proj_2R[tix_response,:]
    
    tx = tix_delay
    xdata_1P = opto_1R_testtrials[:,tx,:].T
    xdata_1A = opto_1L_testtrials[:,tx,:].T
    xdata_2P = opto_2L_testtrials[:,tx,:].T
    xdata_2A = opto_2R_testtrials[:,tx,:].T        
    linreg = LinearRegression(fit_intercept=False)
    linreg.fit(xdata_1P,ydata_1P)
    TD_1P = linreg.coef_
    linreg.fit(xdata_1A,ydata_1A)
    TD_1A = linreg.coef_
    linreg.fit(xdata_2P,ydata_2P)
    TD_2P = linreg.coef_
    linreg.fit(xdata_2A,ydata_2A)
    TD_2A = linreg.coef_

    cossim_TD_1P_all[shuff,fx] = cosine_similarity(TD_1P, ID_P[tx])
    cossim_TD_1A_all[shuff,fx] = cosine_similarity(TD_1A, ID_A[tx])
    cossim_TD_2P_all[shuff,fx] = cosine_similarity(TD_2P, ID_P[tx])
    cossim_TD_2A_all[shuff,fx] = cosine_similarity(TD_2A, ID_A[tx])            
    
    return cossim_TD_1P_all, cossim_TD_1A_all, cossim_TD_2P_all, cossim_TD_2A_all


def decode_context(data, data_type):
    tsample   = 1.57
    tdelay    = 2.87
    tresponse = 4.17
    dt        = 1/6
    ntimestep = 47
    tvec      = dt * np.arange(ntimestep)
    tix_sample   = np.where(tvec > tsample)[0][0]
    tix_delay    = np.where(tvec > tdelay)[0][0]
    tix_response = np.where(tvec > tresponse)[0][0]

    opto_sess1  = data[data_type][0]
    opto_sess2  = data[data_type][1]
    ncat, ncell = opto_sess1.shape
    lickright = 0
    lickleft  = 1

    ntrials_1P = opto_sess1[lickright,0].shape[1]
    ntrials_1A = opto_sess1[lickleft,0].shape[1]
    ntrials_2A = opto_sess2[lickright,0].shape[1]
    ntrials_2P = opto_sess2[lickleft,0].shape[1]

    frac_train_trials = 0.5
    ntrials_train_1P = int(ntrials_1P * frac_train_trials)
    ntrials_train_1A = int(ntrials_1A * frac_train_trials)
    ntrials_train_2A = int(ntrials_2A * frac_train_trials)
    ntrials_train_2P = int(ntrials_2P * frac_train_trials)

    opto_1P_alltrials = np.zeros((ncell,ntimestep,ntrials_1P))
    opto_1A_alltrials = np.zeros((ncell,ntimestep,ntrials_1A))
    opto_2P_alltrials = np.zeros((ncell,ntimestep,ntrials_2P))
    opto_2A_alltrials = np.zeros((ncell,ntimestep,ntrials_2A))
    for cell in range(ncell):
        opto_1P_alltrials[cell] = opto_sess1[lickright,cell]
        opto_1A_alltrials[cell] = opto_sess1[lickleft,cell]
        opto_2P_alltrials[cell] = opto_sess2[lickleft,cell]
        opto_2A_alltrials[cell] = opto_sess2[lickright,cell]
        
    '''
    Compute Coding Direction - Context
    '''
    ndecode = 20
    pred_1P = np.zeros((ndecode,2))
    pred_1A = np.zeros((ndecode,2))
    pred_2P = np.zeros((ndecode,2))
    pred_2A = np.zeros((ndecode,2))
    for decode in range(ndecode):
            
        '''
        Divide train and test trials
        '''
        trials_1P, trials_1A, trials_2P, trials_2A = np.arange(ntrials_1P), np.arange(ntrials_1A), np.arange(ntrials_2P), np.arange(ntrials_2A)
        train_trials_1P = np.sort(np.random.permutation(trials_1P)[:ntrials_train_1P])
        train_trials_1A = np.sort(np.random.permutation(trials_1A)[:ntrials_train_1A])
        train_trials_2P = np.sort(np.random.permutation(trials_2P)[:ntrials_train_2P])
        train_trials_2A = np.sort(np.random.permutation(trials_2A)[:ntrials_train_2A])    
        test_trials_1P = trials_1P[~np.isin(trials_1P,train_trials_1P)]
        test_trials_1A = trials_1A[~np.isin(trials_1A,train_trials_1A)]
        test_trials_2P = trials_2P[~np.isin(trials_2P,train_trials_2P)]
        test_trials_2A = trials_2A[~np.isin(trials_2A,train_trials_2A)]

        '''
        Compute the CD that separtes context 1 and context 2
            - CD_context_P for comparing contexts when stimulus is posterior
            - CD_context_A for comparing contexts when stimulus is anterior
        '''
        opto_1P = np.zeros((ncell,ntimestep))
        opto_1A = np.zeros((ncell,ntimestep))
        opto_2P = np.zeros((ncell,ntimestep))
        opto_2A = np.zeros((ncell,ntimestep))
        for cell in range(ncell):        
            #------- first half trials ----------#
            opto_1P[cell] = np.mean(opto_1P_alltrials[cell,:,train_trials_1P].T,axis=1)
            opto_1A[cell] = np.mean(opto_1A_alltrials[cell,:,train_trials_1A].T,axis=1)
            opto_2P[cell] = np.mean(opto_2P_alltrials[cell,:,train_trials_2P].T,axis=1)    
            opto_2A[cell] = np.mean(opto_2A_alltrials[cell,:,train_trials_2A].T,axis=1)        
        CD_context_P            = opto_1P - opto_2P
        CD_context_A            = opto_1A - opto_2A
        CD_context_P_population = np.mean(CD_context_P[:,tix_sample:tix_sample+5], axis=1)
        CD_context_A_population = np.mean(CD_context_A[:,tix_sample:tix_sample+5], axis=1)

        '''
        Build a decoder
            - Project population activity to CD_context_P and CD_context_A
            - Use the activity of sample epoch to build LDAs (Posterior and Anterior)
        '''    
        opto_1P_traintrials = opto_1P_alltrials[:,:,train_trials_1P]
        opto_1A_traintrials = opto_1A_alltrials[:,:,train_trials_1A]
        opto_2P_traintrials = opto_2P_alltrials[:,:,train_trials_2P]
        opto_2A_traintrials = opto_2A_alltrials[:,:,train_trials_2A]
        proj_train_1P = np.tensordot(CD_context_P_population, opto_1P_traintrials, axes=(0,0))
        proj_train_1A = np.tensordot(CD_context_A_population, opto_1A_traintrials, axes=(0,0))
        proj_train_2P = np.tensordot(CD_context_P_population, opto_2P_traintrials, axes=(0,0))
        proj_train_2A = np.tensordot(CD_context_A_population, opto_2A_traintrials, axes=(0,0))
        x_train_1P    = np.mean(proj_train_1P[tix_sample:tix_sample+5,:],axis=0)
        x_train_1A    = np.mean(proj_train_1A[tix_sample:tix_sample+5,:],axis=0)
        x_train_2P    = np.mean(proj_train_2P[tix_sample:tix_sample+5,:],axis=0)
        x_train_2A    = np.mean(proj_train_2A[tix_sample:tix_sample+5,:],axis=0)
        x_train_P = np.vstack((x_train_1P.reshape(-1,1), x_train_2P.reshape(-1,1)))
        x_train_A = np.vstack((x_train_1A.reshape(-1,1), x_train_2A.reshape(-1,1)))
        y_train_P = np.concatenate((-np.ones(ntrials_train_1P), np.ones(ntrials_train_2P)))
        y_train_A = np.concatenate((-np.ones(ntrials_train_1A), np.ones(ntrials_train_2A)))

        lda_context_P = LinearDiscriminantAnalysis()
        lda_context_A = LinearDiscriminantAnalysis()
        lda_context_P.fit(x_train_P,y_train_P)
        lda_context_A.fit(x_train_A,y_train_A)

        '''
        Decode the test trials
            - LDA Posterior to Context 1 and Context 2 posterior test trials
            - LDA Anterior  to Context 1 and Context 2 anterior  test trials
        '''
        opto_1P_testtrials = opto_1P_alltrials[:,:,test_trials_1P]
        opto_2P_testtrials = opto_2P_alltrials[:,:,test_trials_2P]
        opto_1A_testtrials = opto_1A_alltrials[:,:,test_trials_1A]
        opto_2A_testtrials = opto_2A_alltrials[:,:,test_trials_2A]
        proj_test_1P = np.tensordot(CD_context_P_population, opto_1P_testtrials, axes=(0,0))
        proj_test_2P = np.tensordot(CD_context_P_population, opto_2P_testtrials, axes=(0,0))
        proj_test_1A = np.tensordot(CD_context_A_population, opto_1A_testtrials, axes=(0,0))
        proj_test_2A = np.tensordot(CD_context_A_population, opto_2A_testtrials, axes=(0,0))
        x_test_1P    = np.mean(proj_test_1P[tix_sample:tix_sample+5,:],axis=0)
        x_test_2P    = np.mean(proj_test_2P[tix_sample:tix_sample+5,:],axis=0)
        x_test_1A    = np.mean(proj_test_1A[tix_sample:tix_sample+5,:],axis=0)
        x_test_2A    = np.mean(proj_test_2A[tix_sample:tix_sample+5,:],axis=0)

        y_pred_1P = lda_context_P.predict(x_test_1P.reshape(-1,1))
        y_pred_2P = lda_context_P.predict(x_test_2P.reshape(-1,1))
        y_pred_1A = lda_context_A.predict(x_test_1A.reshape(-1,1))
        y_pred_2A = lda_context_A.predict(x_test_2A.reshape(-1,1))
        
        pred_1P[decode,0] += np.sum(y_pred_1P==-1) 
        pred_2P[decode,0] += np.sum(y_pred_2P==1) 
        pred_1A[decode,0] += np.sum(y_pred_1A==-1) 
        pred_2A[decode,0] += np.sum(y_pred_2A==1) 
        pred_1P[decode,1] += len(y_pred_1P)
        pred_2P[decode,1] += len(y_pred_2P)
        pred_1A[decode,1] += len(y_pred_1A)
        pred_2A[decode,1] += len(y_pred_2A)
        
    pred_1P_correct, pred_1P_all = np.sum(pred_1P,axis=0)
    pred_2P_correct, pred_2P_all = np.sum(pred_2P,axis=0)
    pred_1A_correct, pred_1A_all = np.sum(pred_1A,axis=0)
    pred_2A_correct, pred_2A_all = np.sum(pred_2A,axis=0)
    pred_1P_accuracy = pred_1P_correct / pred_1P_all
    pred_2P_accuracy = pred_2P_correct / pred_2P_all
    pred_1A_accuracy = pred_1A_correct / pred_1A_all
    pred_2A_accuracy = pred_2A_correct / pred_2A_all
    
    return pred_1P_accuracy, pred_1A_accuracy, pred_2P_accuracy, pred_2A_accuracy
    
        
def find_spatial_cluster(all_cells_R, all_cells_L, spatial_coord_R, spatial_coord_L):
    
    ndist=200
    ncell_R   = all_cells_R.shape[0]    
    ncell_L   = all_cells_L.shape[0]    
    cluster_R = _count_cells_over_distance(ncell_R, spatial_coord_R, spatial_coord_L, ndist, 'right_centered')
    cluster_L = _count_cells_over_distance(ncell_L, spatial_coord_R, spatial_coord_L, ndist, 'left_centered')
    
    return cluster_R, cluster_L

def _count_cells_over_distance(ncell, spatial_coord_R, spatial_coord_L, ndist, center):
    
    cluster = np.zeros((ncell,ndist))
    for cix in range(ncell):
        if center == 'right_centered':
            sc_cell = spatial_coord_R[cix]
        elif center == 'left_centered':
            sc_cell = spatial_coord_L[cix]
        sc_dist_1R = np.sqrt(np.sum((spatial_coord_R - sc_cell)**2,axis=1))
        sc_dist_1L = np.sqrt(np.sum((spatial_coord_L - sc_cell)**2,axis=1))        
        for dist in range(ndist):
            num_1R = np.sum(sc_dist_1R < dist)
            num_1L = np.sum(sc_dist_1L < dist)
            if (num_1R > 0) or (num_1L > 0):
                if center == 'right_centered':
                    cluster[cix,dist] = num_1R / (num_1R + num_1L)
                elif center == 'left_centered':
                    cluster[cix,dist] = num_1L / (num_1R + num_1L)
            else:
                cluster[cix,dist] = 1                
    return cluster

def cosine_similarity(x,y):
    cosine_similarity = np.inner(x,y) / np.linalg.norm(x) / np.linalg.norm(y)
    return cosine_similarity


def project_to(x,y):
    projection = np.inner(x,y) / np.linalg.norm(x)
    return projection
    
def rank_mouse_groups(mouse_group, CD_dotproduct_all, nfile):
    
    unique_mouse_group = np.unique(mouse_group)
    num_unique         = len(unique_mouse_group)
    color_index        = np.zeros(num_unique)
    mouse_group_ranks  = np.zeros(nfile)
    # Average of group's CD dot product
    CD_dotproduct_of_mouse_group                          = np.array([np.mean(CD_dotproduct_all[mouse_group == m]) for m in unique_mouse_group])
    # Rank the groups by CD dot product
    color_index[np.argsort(CD_dotproduct_of_mouse_group)] = np.arange(num_unique) + 1
    # Rank individual mice by CD dot product
    for cx, m in enumerate(unique_mouse_group):
        mouse_group_ranks[mouse_group == m] = color_index[cx]
    return mouse_group_ranks
    
def divide_into_two_groups(mouse_group_ranks, numfirstGroup, nfile):

    # include numfirstGroup groups in the first group
    firstGroup = np.array([],dtype=int)
    for m in range(numfirstGroup):
        firstGroup = np.concatenate((firstGroup,np.where(mouse_group_ranks == m+1)[0]))
    # the remaining mice in the second group
    secondGroup = np.delete(np.arange(nfile),firstGroup)    
    return firstGroup, secondGroup

def correlation_CDdotprod_vs_ID_along_CD(nshuff, ntimestep, CD_dotproduct_all, 
                                          ID_P_along_CD1, ID_A_along_CD1, 
                                          ID_P_along_CD2, ID_A_along_CD2
                                          ):
    
    cor_CDdot_IDP_CD1              = np.zeros((nshuff,ntimestep))
    cor_CDdot_IDA_CD1              = np.zeros((nshuff,ntimestep))
    cor_CDdot_IDP_CD2              = np.zeros((nshuff,ntimestep))
    cor_CDdot_IDA_CD2              = np.zeros((nshuff,ntimestep))    
    for shuff in range(nshuff):
        CD_dotproduct_shuff = CD_dotproduct_all[shuff]
        for tx in range(ntimestep):
            cor_CDdot_IDP_CD1[shuff,tx] = np.corrcoef(CD_dotproduct_shuff, ID_P_along_CD1[shuff,:,tx])[0,1]
            cor_CDdot_IDA_CD1[shuff,tx] = np.corrcoef(CD_dotproduct_shuff, ID_A_along_CD1[shuff,:,tx])[0,1]
            cor_CDdot_IDP_CD2[shuff,tx] = np.corrcoef(CD_dotproduct_shuff, ID_P_along_CD2[shuff,:,tx])[0,1]
            cor_CDdot_IDA_CD2[shuff,tx] = np.corrcoef(CD_dotproduct_shuff, ID_A_along_CD2[shuff,:,tx])[0,1]
                
    return (cor_CDdot_IDP_CD1, cor_CDdot_IDA_CD1, 
            cor_CDdot_IDP_CD2, cor_CDdot_IDA_CD2)

def correlation_CDdotprod_vs_distance_btw_contexts(nshuff, ntimestep, nfile, CD_dotproduct_all, mouse_group,
                                                   distance_P_all, distance_A_all):

    cor_CDdot_distP_firstGroup  = np.zeros((nshuff,ntimestep))
    cor_CDdot_distA_firstGroup  = np.zeros((nshuff,ntimestep))
    cor_CDdot_distP_secondGroup = np.zeros((nshuff,ntimestep))
    cor_CDdot_distA_secondGroup = np.zeros((nshuff,ntimestep))
    cor_CDdot_distP         = np.zeros((nshuff,ntimestep))
    cor_CDdot_distA         = np.zeros((nshuff,ntimestep))    
    for shuff in range(nshuff):
        CD_dotproduct_shuff     = CD_dotproduct_all[shuff]
        numfirstGroup           = 5
        mouse_group_ranks       = rank_mouse_groups(mouse_group, CD_dotproduct_shuff, nfile)
        firstGroup, secondGroup = divide_into_two_groups(mouse_group_ranks, numfirstGroup, nfile)
        for tx in range(ntimestep):
            cor_CDdot_distP_firstGroup[shuff,tx]  = np.corrcoef(CD_dotproduct_shuff[firstGroup],  distance_P_all[shuff,firstGroup,tx])[0,1]
            cor_CDdot_distA_firstGroup[shuff,tx]  = np.corrcoef(CD_dotproduct_shuff[firstGroup],  distance_A_all[shuff,firstGroup,tx])[0,1]
            cor_CDdot_distP_secondGroup[shuff,tx] = np.corrcoef(CD_dotproduct_shuff[secondGroup], distance_P_all[shuff,secondGroup,tx])[0,1]
            cor_CDdot_distA_secondGroup[shuff,tx] = np.corrcoef(CD_dotproduct_shuff[secondGroup], distance_A_all[shuff,secondGroup,tx])[0,1]
            cor_CDdot_distP[shuff,tx]             = np.corrcoef(CD_dotproduct_shuff,              distance_P_all[shuff,:,tx])[0,1]
            cor_CDdot_distA[shuff,tx]             = np.corrcoef(CD_dotproduct_shuff,              distance_A_all[shuff,:,tx])[0,1]
    
    return (cor_CDdot_distP_firstGroup,  cor_CDdot_distA_firstGroup, 
            cor_CDdot_distP_secondGroup, cor_CDdot_distA_secondGroup, 
            cor_CDdot_distP,             cor_CDdot_distA)
    
def stimulus_distance(nshuff, nfile, CD_sample_1, CD_sample_2):
    
    CD_sample_1_distance = np.zeros((nshuff,nfile))
    CD_sample_2_distance = np.zeros((nshuff,nfile))
    CD_sample_1_std      = np.zeros((nshuff,nfile))
    CD_sample_2_std      = np.zeros((nshuff,nfile))
    for shuff in range(nshuff):
        for fx in range(nfile):
            CD_sample_1_distance[shuff,fx] = np.sqrt(np.mean(CD_sample_1[shuff,fx]**2))
            CD_sample_2_distance[shuff,fx] = np.sqrt(np.mean(CD_sample_2[shuff,fx]**2))
            CD_sample_1_std[shuff,fx] = np.std(CD_sample_1[shuff,fx])
            CD_sample_2_std[shuff,fx] = np.std(CD_sample_2[shuff,fx])
            
    return (CD_sample_1_distance, CD_sample_2_distance, 
            CD_sample_1_std,      CD_sample_2_std)

def stimulus_selective_cells(nshuff, nfile, nthresh, thresh, CD_sample_1, CD_sample_2, CD_sample_1_std, CD_sample_2_std):
    
    CD_sample_1_selective = np.zeros((nshuff,nfile,nthresh))
    CD_sample_2_selective = np.zeros((nshuff,nfile,nthresh))
    for shuff in range(nshuff):
        std1   = np.mean(CD_sample_1_std[shuff])
        std2   = np.mean(CD_sample_2_std[shuff])
        for fx in range(nfile):
            # std   = CD_sample_1_std[shuff,fx]
            ncell = len(CD_sample_1[shuff,fx])
            for thx, thr in enumerate(thresh):
                CD_sample_1_selective[shuff,fx,thx] = np.sum(np.abs(CD_sample_1[shuff,fx]) > thr * std1) / ncell
                CD_sample_2_selective[shuff,fx,thx] = np.sum(np.abs(CD_sample_2[shuff,fx]) > thr * std2) / ncell        
    
    return CD_sample_1_selective, CD_sample_2_selective


def gen_lognormal_rate(mean, std, shift_in_trial2):
    
    def convert_to_bas10(x):
        x_convert = 10**(np.log(x))
        return x_convert

    # shift_in_trial2 = 0.0
    # shift_in_trial2 = 0.025
    P1 = convert_to_bas10(np.random.lognormal(mean=mean, sigma=std, size=5000))
    A1 = convert_to_bas10(np.random.lognormal(mean=mean, sigma=std, size=5000))
    P2 = convert_to_bas10(np.random.lognormal(mean=mean, sigma=std, size=5000)) + shift_in_trial2
    A2 = convert_to_bas10(np.random.lognormal(mean=mean, sigma=std, size=5000)) + shift_in_trial2

    _dP = P2 - P1
    _dA = A2 - A1
    _dR = A2 - P1
    _dL = P2 - A1

    _S1 = P1 - A1
    _S2 = P2 - A2
    _S1S2 = _S1 * _S2

    type1 = (_S1S2 < 0) * (_S1 > 0)
    type2 = (_S1S2 < 0) * (_S1 < 0)
    type3 = (_S1S2 > 0) * (_S1 > 0)
    type4 = (_S1S2 > 0) * (_S1 < 0)

    type1_dP, type1_dA = _dP[type1], _dA[type1]
    type2_dP, type2_dA = _dP[type2], _dA[type2]
    type3_dR, type3_dL = _dR[type3], _dL[type3]
    type4_dR, type4_dL = _dR[type4], _dL[type4]

    type1_frac_in_quad = fraction_in_quad(type1_dP, type1_dA)
    type2_frac_in_quad = fraction_in_quad(type2_dP, type2_dA)
    type3_frac_in_quad = fraction_in_quad(type3_dR, type3_dL)
    type4_frac_in_quad = fraction_in_quad(type4_dR, type4_dL)
    
    return type1_dP, type1_dA, type2_dP, type2_dA, type3_dR, type3_dL, type4_dR, type4_dL, \
           type1_frac_in_quad, type2_frac_in_quad, type3_frac_in_quad, type4_frac_in_quad
    
def shuffle_neural_data(nfile, cells_dS, P1_set1, P1_set2, A1_set1, A1_set2, P2_set1, P2_set2, A2_set1, A2_set2, permute):
            
    shuff_cells_P1     = {}
    shuff_cells_A1     = {}
    shuff_cells_P2     = {}
    shuff_cells_A2     = {}
    '''
    If paired:
        (P1,A1) and (P2,A2) are shuffled. The pair is kept the same.
        
    If not paired:
        P1, A1, P2, A2 are shuffled separately

    Shuffled cells for each activity     
        shuff_cells_P1     = P1 (set 1 & set 2)
        shuff_cells_A1     = A1 (set 1 & set 2)
        shuff_cells_P2     = P2 (set 1 & set 2)
        shuff_cells_A2     = A2 (set 1 & set 2)
    '''
    if permute == 'paired':    
        for fx in range(nfile):    
            _shuff_cells_dS1 = np.random.permutation(cells_dS[fx])
            _shuff_cells_dS2 = np.random.permutation(cells_dS[fx])
            shuff_cells_P1[fx] = _shuff_cells_dS1
            shuff_cells_A1[fx] = _shuff_cells_dS1
            shuff_cells_P2[fx] = _shuff_cells_dS2
            shuff_cells_A2[fx] = _shuff_cells_dS2
    elif permute == 'not_paired':    
        for fx in range(nfile):    
            shuff_cells_P1[fx] = np.random.permutation(cells_dS[fx])
            shuff_cells_A1[fx] = np.random.permutation(cells_dS[fx])
            shuff_cells_P2[fx] = np.random.permutation(cells_dS[fx])
            shuff_cells_A2[fx] = np.random.permutation(cells_dS[fx])

    shuff_P1_set1, shuff_P1_set2 = {}, {}
    shuff_A1_set1, shuff_A1_set2 = {}, {}
    shuff_P2_set1, shuff_P2_set2 = {}, {}
    shuff_A2_set1, shuff_A2_set2 = {}, {}
    for fx in range(nfile):
        shuff_P1_set1[fx] = P1_set1[fx][shuff_cells_P1[fx]]
        shuff_P1_set2[fx] = P1_set2[fx][shuff_cells_P1[fx]]        
        shuff_A1_set1[fx] = A1_set1[fx][shuff_cells_A1[fx]]
        shuff_A1_set2[fx] = A1_set2[fx][shuff_cells_A1[fx]]    
        shuff_P2_set1[fx] = P2_set1[fx][shuff_cells_P2[fx]]
        shuff_P2_set2[fx] = P2_set2[fx][shuff_cells_P2[fx]]        
        shuff_A2_set1[fx] = A2_set1[fx][shuff_cells_A2[fx]]        
        shuff_A2_set2[fx] = A2_set2[fx][shuff_cells_A2[fx]]        
        
    return shuff_P1_set1, shuff_P1_set2, shuff_A1_set1, shuff_A1_set2, shuff_P2_set1, shuff_P2_set2, shuff_A2_set1, shuff_A2_set2
        
def shuffle_compute_dS_dP_dA_dR_dL(nfile, shuff_P1_set1, shuff_P1_set2, shuff_A1_set1, shuff_A1_set2, shuff_P2_set1, shuff_P2_set2, shuff_A2_set1, shuff_A2_set2):
    
    shuff_CD_dotproduct = np.zeros(nfile)
    shuff_S1, shuff_S2 = {}, {}
    shuff_dP, shuff_dA = {}, {}
    shuff_dR, shuff_dL = {}, {}
    for fx in range(nfile):
        # trial set 1
        _shuff_P1_set1 = shuff_P1_set1[fx]
        _shuff_A1_set1 = shuff_A1_set1[fx]
        _shuff_P2_set1 = shuff_P2_set1[fx]
        _shuff_A2_set1 = shuff_A2_set1[fx]
        # CD dot product
        shuff_CD_dotproduct[fx] = compute_CD_dotproduct_fixed_time(_shuff_P1_set1,_shuff_A1_set1,_shuff_A2_set1,_shuff_P2_set1)        
        # selectivity
        shuff_S1[fx], shuff_S2[fx] = compute_S1_S2(_shuff_P1_set1, _shuff_P2_set1, _shuff_A1_set1, _shuff_A2_set1)
        
        # trial set 2
        _shuff_P1_set2 = shuff_P1_set2[fx]
        _shuff_A1_set2 = shuff_A1_set2[fx]
        _shuff_P2_set2 = shuff_P2_set2[fx]
        _shuff_A2_set2 = shuff_A2_set2[fx]    
        # learned activity
        shuff_dP[fx], shuff_dA[fx], shuff_dR[fx], shuff_dL[fx] = compute_dP_dA_dR_dL(_shuff_P1_set2, _shuff_P2_set2, _shuff_A1_set2, _shuff_A2_set2)
    return shuff_CD_dotproduct, shuff_S1, shuff_S2, shuff_dP, shuff_dA, shuff_dR, shuff_dL

def shuffle_compute_learned_activity(nfile, shuff_learned_activity, shuff_frac_cells, shuff_S1, shuff_S2, shuff_dP, shuff_dA, shuff_dR, shuff_dL):
    
    for fx in range(nfile):  
        # selectivity
        _shuff_dS = shuff_S1[fx] * shuff_S2[fx]
        _cells_dS_neg = np.where(_shuff_dS < 0)[0]
        _cells_dS_pos = np.where(_shuff_dS > 0)[0]
        _cells_dS_neg_S1_pos_fx = _cells_dS_neg[shuff_S1[fx][_cells_dS_neg]>0]
        _cells_dS_neg_S1_neg_fx = _cells_dS_neg[shuff_S1[fx][_cells_dS_neg]<0]
        _cells_dS_pos_S1_pos_fx = _cells_dS_pos[shuff_S1[fx][_cells_dS_pos]>0]
        _cells_dS_pos_S1_neg_fx = _cells_dS_pos[shuff_S1[fx][_cells_dS_pos]<0]
        
        # learned activity
        shuff_learned_activity[f'sess{fx}']['dS-']['S1+']['dP'], shuff_learned_activity[f'sess{fx}']['dS-']['S1+']['dA'] = shuff_dP[fx][_cells_dS_neg_S1_pos_fx], shuff_dA[fx][_cells_dS_neg_S1_pos_fx]
        shuff_learned_activity[f'sess{fx}']['dS-']['S1-']['dP'], shuff_learned_activity[f'sess{fx}']['dS-']['S1-']['dA'] = shuff_dP[fx][_cells_dS_neg_S1_neg_fx], shuff_dA[fx][_cells_dS_neg_S1_neg_fx]
        shuff_learned_activity[f'sess{fx}']['dS+']['S1+']['dR'], shuff_learned_activity[f'sess{fx}']['dS+']['S1+']['dL'] = shuff_dR[fx][_cells_dS_pos_S1_pos_fx], shuff_dL[fx][_cells_dS_pos_S1_pos_fx]
        shuff_learned_activity[f'sess{fx}']['dS+']['S1-']['dR'], shuff_learned_activity[f'sess{fx}']['dS+']['S1-']['dL'] = shuff_dR[fx][_cells_dS_pos_S1_neg_fx], shuff_dL[fx][_cells_dS_pos_S1_neg_fx]

        # marginal distribution
        shuff_frac_cells[f'sess{fx}']['dS-']['S1+']['dP<0'] = fraction_of(shuff_learned_activity[f'sess{fx}']['dS-']['S1+']['dP'],'-')
        shuff_frac_cells[f'sess{fx}']['dS-']['S1+']['dA>0'] = fraction_of(shuff_learned_activity[f'sess{fx}']['dS-']['S1+']['dA'],'+')
        shuff_frac_cells[f'sess{fx}']['dS-']['S1-']['dP>0'] = fraction_of(shuff_learned_activity[f'sess{fx}']['dS-']['S1-']['dP'],'+')
        shuff_frac_cells[f'sess{fx}']['dS-']['S1-']['dA<0'] = fraction_of(shuff_learned_activity[f'sess{fx}']['dS-']['S1-']['dA'],'-')    
        shuff_frac_cells[f'sess{fx}']['dS+']['S1+']['dR<0'] = fraction_of(shuff_learned_activity[f'sess{fx}']['dS+']['S1+']['dR'],'-')
        shuff_frac_cells[f'sess{fx}']['dS+']['S1+']['dL>0'] = fraction_of(shuff_learned_activity[f'sess{fx}']['dS+']['S1+']['dL'],'+')
        shuff_frac_cells[f'sess{fx}']['dS+']['S1-']['dR>0'] = fraction_of(shuff_learned_activity[f'sess{fx}']['dS+']['S1-']['dR'],'+')
        shuff_frac_cells[f'sess{fx}']['dS+']['S1-']['dL<0'] = fraction_of(shuff_learned_activity[f'sess{fx}']['dS+']['S1-']['dL'],'-')
        
        # all four quadrants
        shuff_frac_cells[f'sess{fx}']['dS-']['S1+']['Quad'] = fraction_in_quad(shuff_learned_activity[f'sess{fx}']['dS-']['S1+']['dP'],shuff_learned_activity[f'sess{fx}']['dS-']['S1+']['dA'])
        shuff_frac_cells[f'sess{fx}']['dS-']['S1-']['Quad'] = fraction_in_quad(shuff_learned_activity[f'sess{fx}']['dS-']['S1-']['dP'],shuff_learned_activity[f'sess{fx}']['dS-']['S1-']['dA'])
        shuff_frac_cells[f'sess{fx}']['dS+']['S1+']['Quad'] = fraction_in_quad(shuff_learned_activity[f'sess{fx}']['dS+']['S1+']['dR'],shuff_learned_activity[f'sess{fx}']['dS+']['S1+']['dL'])
        shuff_frac_cells[f'sess{fx}']['dS+']['S1-']['Quad'] = fraction_in_quad(shuff_learned_activity[f'sess{fx}']['dS+']['S1-']['dR'],shuff_learned_activity[f'sess{fx}']['dS+']['S1-']['dL'])    

    return shuff_learned_activity, shuff_frac_cells

def remove_outliers_alltrials_fixedtime(x1R,x1L,x2R,x2L,maxstd):
    x1   = np.append(x1R,x1L)
    x2   = np.append(x2R,x2L)
    z1R  = (x1R - np.mean(x1)) / np.std(x1)
    z1L  = (x1L - np.mean(x1)) / np.std(x1)
    z2R  = (x2R - np.mean(x2)) / np.std(x2)
    z2L  = (x2L - np.mean(x2)) / np.std(x2)
    keep = (np.abs(z1R)<maxstd) * (np.abs(z1L)<maxstd) * (np.abs(z2R)<maxstd) * (np.abs(z2L)<maxstd)
    return keep


def remove_outliers_alltrials_timeseries(x1R,x1L,x2R,x2L,maxstd):
    # x: neuron x time
    x1R = np.max(x1R,axis=1)
    x1L = np.max(x1L,axis=1)
    x2R = np.max(x2R,axis=1)
    x2L = np.max(x2L,axis=1)
    
    x1   = np.append(x1R,x1L)
    x2   = np.append(x2R,x2L)
    z1R  = (x1R - np.mean(x1)) / np.std(x1)
    z1L  = (x1L - np.mean(x1)) / np.std(x1)
    z2R  = (x2R - np.mean(x2)) / np.std(x2)
    z2L  = (x2L - np.mean(x2)) / np.std(x2)
    keep = (np.abs(z1R)<maxstd) * (np.abs(z1L)<maxstd) * (np.abs(z2R)<maxstd) * (np.abs(z2L)<maxstd)
    return keep

def find_top_cells_of(amp_neuron, percentage):
    _ncell       = len(amp_neuron)
    _frac_cells  = int(_ncell * percentage)
    cells_sorted = np.argsort(amp_neuron)[::-1]
    top_cells    = cells_sorted[:_frac_cells]
    return top_cells

def find_top_90_cells(amp_neuron, var_neuron, threshold):    
    amp_frac = np.arange(0.01, 1.0, step=0.01)
    var_frac = np.zeros(len(amp_frac))
    for ix, frac in enumerate(amp_frac):
        _topcells    = find_top_cells_of(amp_neuron, frac)
        var_frac[ix] = np.sum(var_neuron[_topcells]) / np.sum(var_neuron)
        
    amp_frac_to_get_var90 = amp_frac[np.where(var_frac > threshold)[0][0]]
    cells_top90           = find_top_cells_of(amp_neuron, amp_frac_to_get_var90)    
    return amp_frac, var_frac, cells_top90

def two_sided_exp(t, b, A, tp, tau_r, tau_d):
    return b + A * np.where(
        t < tp,
        np.exp((t - tp) / tau_r),
        np.exp(-(t - tp) / tau_d)
    )

def fit_exp_decay_to_neuron_activity(nfile, P1_set1, A1_set1, P2_set1, A2_set1,
                                     tvec, tix_start, tix_end):
    ntimestep_fit = tix_end - tix_start
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
    return modelfit

def get_fit_summary(nfile, modelfit, acts):
    
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
        keep_nonoutliers = remove_outliers_alltrials_fixedtime(*max_vals, maxstd=5)
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
    return fit_summary
    
def get_fit_summary_topcells(t, nfile, acts, actidx, dict_topcells, fit_summary):

    fit_summary_topcells_fotmat = {
        'expvar_neuron':        {i:[] for i in range(nfile)},
        'taur':                 {i:[] for i in range(nfile)},
        'taud':                 {i:[] for i in range(nfile)},
        'amp':                  {i:[] for i in range(nfile)},
        'expvar_neuron_at_t':   {i:{} for i in range(nfile)},
        'taur_at_t':            {i:{} for i in range(nfile)},
        'taud_at_t':            {i:{} for i in range(nfile)},
        'amp_at_t':             {i:{} for i in range(nfile)}
    }
    fit_summary_topcells = {act: copy.deepcopy(fit_summary_topcells_fotmat) for act in acts}
    
    for act in acts:
        for fx in range(nfile):            
            # all top cells
            nonoutlier  = fit_summary['nonoutlier'][fx]
            ncell       = len(nonoutlier)
            expvar      = fit_summary['expvar_neuron'][fx][actidx[act]]
            taur        = fit_summary['taur'][fx][actidx[act]]
            taud        = fit_summary['taud'][fx][actidx[act]]
            amp         = fit_summary['amp'][fx][actidx[act]]
            # top cells at t
            expvar_at_t_fx = {i:[] for i in range(len(t))}
            taur_at_t_fx   = {i:[] for i in range(len(t))}
            taud_at_t_fx   = {i:[] for i in range(len(t))}
            amp_at_t_fx    = {i:[] for i in range(len(t))}
            for i in range(len(t)):
                expvar_at_t_fx[i] = expvar[dict_topcells[act]['topcells_at_t'][fx][i]]
                taur_at_t_fx[i]   = taur[dict_topcells[act]['topcells_at_t'][fx][i]]
                taud_at_t_fx[i]   = taud[dict_topcells[act]['topcells_at_t'][fx][i]]    
                amp_at_t_fx[i]    = amp[dict_topcells[act]['topcells_at_t'][fx][i]]
            # save
            fit_summary_topcells[act]['expvar_neuron'][fx]       = expvar[dict_topcells[act]['topcells'][fx]]
            fit_summary_topcells[act]['taur'][fx]                = taur[dict_topcells[act]['topcells'][fx]]
            fit_summary_topcells[act]['taud'][fx]                = taud[dict_topcells[act]['topcells'][fx]]
            fit_summary_topcells[act]['amp'][fx]                 = amp[dict_topcells[act]['topcells'][fx]]
            fit_summary_topcells[act]['expvar_neuron_at_t'][fx]  = expvar_at_t_fx
            fit_summary_topcells[act]['taur_at_t'][fx]           = taur_at_t_fx
            fit_summary_topcells[act]['taud_at_t'][fx]           = taud_at_t_fx        
            fit_summary_topcells[act]['amp_at_t'][fx]            = amp_at_t_fx                    
    return fit_summary_topcells

def shuffle_topcells_at_t(_topcells_fx,_topcells_at_t_fx):
    # mapping from topcells to shuffled topcells
    _topcells_fx_shuffled = np.random.permutation(_topcells_fx)
    _mapping = dict(zip(_topcells_fx,_topcells_fx_shuffled))
    # shuffle topcells at t
    _topcells_at_t_fx_shuffled = copy.deepcopy(_topcells_at_t_fx)
    _ntime = len(_topcells_at_t_fx)
    # apply the mapping to the topcells at every t
    for ti in range(_ntime):
        _topcells_at_t_fx_shuffled[ti] = np.array([_mapping[v] for v in _topcells_at_t_fx[ti]])
    return _topcells_at_t_fx_shuffled

def sort_cells_by_var(_data):
    '''
    variance vs. second-moment
    '''    
    _data_sq = _data**2 # second-moment
    # _data_sq = (_data - np.mean(_data,axis=0))**2 # variance
    # normalize population activity at every t
    _data_norm = _data_sq / np.sum(_data_sq,axis=0)
    # sort the neurons by its contribution to variance
    _cells_sorted_at_t = np.argsort(_data_norm,axis=0)[::-1]
    _data_sorted = np.take_along_axis(_data_norm, _cells_sorted_at_t, axis=0)
    # cumulative sum at every t
    _data_cumsum = np.cumsum(_data_sorted,axis=0)
    return _cells_sorted_at_t, _data_cumsum

def create_binary_data(_data, _thr_var):
    _ncell, _ntime = _data.shape
    _cells_sorted_at_t, _data_cumsum = sort_cells_by_var(_data)
    _idx_cross = np.argmax(_data_cumsum >= _thr_var, axis=0)
    _topcells_at_t = {i:[] for i in range(_ntime)}
    _newcells_at_t = {i:[] for i in range(_ntime)}
    _topcells_at_t_shuffled = {i:[] for i in range(_ntime)}
    _topcells_prev = []
    _data_binary = np.zeros_like(_data)
    for i in range(_ntime):
        # top cells at t
        _topcells_at_t[i] = _cells_sorted_at_t[:_idx_cross[i],i]
        # create binary data
        _data_binary[_topcells_at_t[i],i] = 1    
        # only the new top cells added at t
        _newcells_at_t[i] = _topcells_at_t[i][~np.isin(_topcells_at_t[i],_topcells_prev)]
        # top cells accumulated over time
        _topcells_prev = np.concatenate((_topcells_prev,_newcells_at_t[i]))
    _expvar_at_t = np.array([_data_cumsum[_idx_cross[i],i] for i in range(_ntime)])
    return _expvar_at_t, _topcells_at_t, _newcells_at_t, _data_binary, _cells_sorted_at_t, _data_cumsum

def get_dict_topcells(acts, nfile, thr_var, modelfit, fit_summary):
    
    dict_topcells_format = {
        'topcells':                 {i:[] for i in range(nfile)},
        'topcells_at_t':            {i:[] for i in range(nfile)},
        'topcells_at_t_shuffled':   {i:[] for i in range(nfile)},
        'newcells_at_t':            {i:[] for i in range(nfile)},
        'expvar_at_t':              {i:[] for i in range(nfile)},
        'data_binary':              {i:[] for i in range(nfile)},
        'cells_sorted_at_t':        {i:[] for i in range(nfile)},
        'data_cumsum':              {i:[] for i in range(nfile)}
    }
    dict_topcells = {act: copy.deepcopy(dict_topcells_format) for act in acts}    
    
    for act in acts:
        for fx in range(nfile):
            nonoutlier = fit_summary['nonoutlier'][fx]
            _data   = modelfit[f'sess{fx}'][act]['data'][nonoutlier,:]
            _expvar_at_t_fx, _topcells_at_t_fx,  _newcells_at_t_fx, \
            _data_binary_fx, _cells_sorted_at_t, _data_cumsum = create_binary_data(_data,thr_var)
            _topcells_fx = np.unique(np.concatenate(list(_topcells_at_t_fx.values())))
            dict_topcells[act]['expvar_at_t'][fx]            = _expvar_at_t_fx
            dict_topcells[act]['topcells_at_t'][fx]          = _topcells_at_t_fx
            dict_topcells[act]['topcells_at_t_shuffled'][fx] = shuffle_topcells_at_t(_topcells_fx,_topcells_at_t_fx)
            dict_topcells[act]['newcells_at_t'][fx]          = _newcells_at_t_fx
            dict_topcells[act]['topcells'][fx]               = _topcells_fx
            dict_topcells[act]['data_binary'][fx]            = _data_binary_fx
            dict_topcells[act]['cells_sorted_at_t'][fx]      = _cells_sorted_at_t
            dict_topcells[act]['data_cumsum'][fx]            = _data_cumsum                
    return dict_topcells    

def exp_decay(t, tau, A, b):
    return A * np.exp(-t / tau) + b

def exp_rise(t, tau, A, b):
    return A * np.exp(t / tau) + b

def fit_decay_of_active_cells_linear(t, _data_to_fit):

    n_time = len(t)
    linear_params = np.full((n_time, 2), np.nan)   # slope, intercept
    linear_r2     = np.full(n_time, np.nan)
    fit_trace      = {i: [] for i in range(n_time)}

    for i in range(n_time):

        telapsed = t[i:]
        y = _data_to_fit[i, i:]

        # remove NaNs
        keep = np.isfinite(telapsed) & np.isfinite(y)
        xfit = telapsed[keep] - t[i]   # elapsed time starts at 0
        yfit = y[keep]

        if len(yfit) < 2:
            continue

        try:
            # Linear regression: y = slope*x + intercept
            slope, intercept = np.polyfit(xfit, yfit, 1)

            linear_params[i] = [slope, intercept]

            ypred = slope * xfit + intercept

            if np.var(yfit) > 0:
                linear_r2[i] = 1 - np.var(yfit - ypred) / np.var(yfit)
            else:
                linear_r2[i] = np.nan

            fit_trace[i] = np.vstack((yfit, ypred))

        except Exception:
            pass

    return linear_params, linear_r2, fit_trace

def fit_decay_of_active_cells_exponential(t,_data_to_fit):
        
    n_time = len(t)
    dt = t[1]-t[0]
    exp_params = np.full((n_time, 3), np.nan)   # A, tau, b
    exp_r2     = np.full(n_time, np.nan)
    fit_trace  = {i:[] for i in range(n_time)}
    for i in range(n_time):

        telapsed = t[i:]
        y = _data_to_fit[i, i:]

        # remove NaNs
        keep = np.isfinite(telapsed) & np.isfinite(y)
        xfit = telapsed[keep] - t[i]   # elapsed time starts at 0
        yfit = y[keep]

        if len(yfit) < 2:
            continue

        try:
            p0 = [dt* len(yfit) / 3, yfit[0] - yfit[-1], yfit[-1]]
            bounds = ([1e-6, 0, -np.inf], [dt* 20, 2, 2])

            popt, pcov = curve_fit(
                exp_decay,
                xfit,
                yfit,
                p0=p0,
                bounds=bounds,
                maxfev=20000
            )

            tau, A, b = popt
            exp_params[i] = [tau, A, b]

            ypred = exp_decay(xfit, *popt)
            exp_r2[i] = 1 - np.var(yfit - ypred) / np.var(yfit)
            fit_trace[i] = np.vstack((yfit,ypred))
        except Exception:
            pass
    return exp_params, exp_r2, fit_trace

def exp_two_phase(x, A1, tau1, b1, A2, tau2, tbreak):
    ybreak = exp_rise(tbreak, A1, tau1, b1)
    dt = x[1] - x[0]
    return np.where(
        x <= tbreak,
        exp_rise(x, A1, tau1, b1),
        exp_rise(x - tbreak, A2, tau2, ybreak-A2)
    )

def fit_rise_of_active_cells_linear(t, _data_to_fit):

    tix_delay = 8
    tix_response = 16
    tstart = tix_delay - 1
    tref = np.arange(tix_delay, tix_response)

    pre_linear_params = np.full((len(tref), 2), np.nan)   # slope, intercept
    pre_linear_r2     = np.full(len(tref), np.nan)
    pre_fit_trace     = np.zeros((len(tref), 2, tstart + 1))

    peri_linear_params = np.full((len(tref), 2), np.nan)  # slope, intercept
    peri_linear_r2     = np.full(len(tref), np.nan)
    peri_fit_trace     = {i: [] for i in range(len(tref))}

    fit_duration = ['pre_delay', 'peri_delay']

    for dur in fit_duration:
        for i, tr in enumerate(tref):

            if dur == 'pre_delay':
                tid_span = np.arange(tstart + 1)

            if dur == 'peri_delay':
                tid_span = np.arange(tstart, tr + 1)

            tspan = t[tid_span]
            yfit = _data_to_fit[tr, tid_span]
            xfit = tspan - tspan[0]

            if len(yfit) < 2:
                continue

            try:
                # y = slope * x + intercept
                slope, intercept = np.polyfit(xfit, yfit, 1)
                ypred = slope * xfit + intercept

                if np.var(yfit) > 0:
                    r2 = 1 - np.var(yfit - ypred) / np.var(yfit)
                else:
                    r2 = np.nan

                if dur == 'pre_delay':
                    pre_linear_params[i] = [slope, intercept]
                    pre_linear_r2[i] = r2
                    pre_fit_trace[i] = np.vstack((yfit, ypred))

                if dur == 'peri_delay':
                    peri_linear_params[i] = [slope, intercept]
                    peri_linear_r2[i] = r2
                    peri_fit_trace[i] = np.vstack((yfit, ypred))

            except Exception:
                pass

    return (
        pre_linear_params,
        pre_linear_r2,
        pre_fit_trace,
        peri_linear_params,
        peri_linear_r2,
        peri_fit_trace
    )
        
# def fit_rise_of_active_cells(t,_data_to_fit):
#     n_time = len(t)
#     dt = t[1] - t[0]

#     tref = np.arange(8, 16)
#     tstart = 7

#     exp_params = np.full((len(tref), 5), np.nan)   # A1, tau1, b1, A2, tau2
#     exp_r2 = np.full(len(tref), np.nan)
#     fit_trace = {i: [] for i in range(len(tref))}

#     for i, tr in enumerate(tref):

#         tid_span = np.arange(tr + 1)

#         tspan = t[tid_span]
#         yfit = _data_to_fit[tr, tid_span]
#         xfit = tspan - tspan[0]

#         # Break between pre-delay and peri-delay
#         tbreak = xfit[tstart]

#         if len(yfit) < 4:
#             continue

#         try:
#             p0 = [
#                 yfit[tstart] - yfit[0],        # A1
#                 dt * (tstart + 1) / 3,         # tau1
#                 yfit[0],                       # b1
#                 yfit[-1] - yfit[tstart],       # A2
#                 dt * (tr - tstart + 1) / 3     # tau2
#             ]

#             bounds = (
#                 [1e-2, 1e-6, -np.inf, 1e-2, 1e-6],
#                 [2, dt * 20, 10, 2, dt * 20]
#             )

#             popt, pcov = curve_fit(
#                 lambda x, A1, tau1, b1, A2, tau2:
#                     exp_two_phase(x, A1, tau1, b1, A2, tau2, tbreak),
#                 xfit,
#                 yfit,
#                 p0=p0,
#                 bounds=bounds,
#                 maxfev=20000
#             )

#             ypred = exp_two_phase(xfit, *popt, tbreak)

#             exp_params[i] = popt
#             exp_r2[i] = 1 - np.var(yfit - ypred) / np.var(yfit)
#             fit_trace[i] = np.vstack((yfit, ypred))

#         except Exception:
#             pass    
#     return exp_params, exp_r2, fit_trace

# def fit_rise_of_active_cells(t,_data_to_fit):
        
#     n_time = len(t)
#     dt = t[1]-t[0]
#     tref = np.arange(8,16) # 9...16
#     tstart = 7
#     pre_exp_params = np.full((len(tref), 3), np.nan)   # A, tau, b
#     pre_exp_r2     = np.full(len(tref), np.nan)
#     pre_fit_trace  = np.zeros((len(tref),2,tstart+1))
#     peri_exp_params = np.full((len(tref), 3), np.nan)   # A, tau, b
#     peri_exp_r2     = np.full(len(tref), np.nan)
#     peri_fit_trace  = {i:[] for i in range(len(tref))}
#     fit_duration = ['pre_delay', 'peri_delay']
#     for dur in fit_duration:
#         for i, tr in enumerate(tref):

#             if dur == 'pre_delay':
#                 tid_span = np.arange(tstart+1)
#             if dur == 'peri_delay':
#                 tid_span = np.arange(tstart,tr+1)
#             tspan = t[tid_span]
#             yfit = _data_to_fit[tr,tid_span]
#             xfit = tspan - tspan[0]   # elapsed time starts at 0

#             if len(yfit) < 2:
#                 continue

#             try:
#                 # A, tau, b
#                 p0 = [yfit[-1] - yfit[0], dt* len(yfit) / 3, yfit[0]]
#                 bounds = ([0, 1e-6, -np.inf], [10, dt* 200, 10])

#                 popt, pcov = curve_fit(
#                     exp_rise,
#                     xfit,
#                     yfit,
#                     p0=p0,
#                     bounds=bounds,
#                     maxfev=20000
#                 )

#                 A, tau, b = popt
#                 ypred = exp_rise(xfit, *popt)
                
#                 if dur == 'pre_delay':
#                     pre_exp_params[i] = [A, tau, b]
#                     pre_exp_r2[i] = 1 - np.var(yfit - ypred) / np.var(yfit)
#                     pre_fit_trace[i] = np.vstack((yfit,ypred))
#                 if dur == 'peri_delay':
#                     peri_exp_params[i] = [A, tau, b]
#                     peri_exp_r2[i] = 1 - np.var(yfit - ypred) / np.var(yfit)
#                     peri_fit_trace[i] = np.vstack((yfit,ypred))

#             except Exception:
#                 pass
#     return peri_exp_params, peri_exp_r2, peri_fit_trace, pre_exp_params, pre_exp_r2, pre_fit_trace
    
def topcell_activity(t, fx, dict_topcells, _data_fx_norm, topcells_are_shuffled):
    # topcells at t: shuffled or not shuffled
    if not topcells_are_shuffled:
        _topcells_at_t_fx = dict_topcells['topcells_at_t'][fx]
    elif topcells_are_shuffled:
        _topcells_at_t_fx = dict_topcells['topcells_at_t_shuffled'][fx]
    # average activity of topcells
    data_fx_top = np.zeros((len(t),len(t)))
    data_fx_top_norm = np.zeros((len(t),len(t)))
    for i in range(len(t)):
        data_fx_top[i]      = np.mean(_data_fx_norm[_topcells_at_t_fx[i]],axis=0)
        data_fx_top_norm[i] = data_fx_top[i] / data_fx_top[i,i]
        # data_fx_top_norm[i] = data_fx_top[i] 
    return data_fx_top_norm

def compute_active_cells_decay(nfile, nact, t, acts, dict_topcells, modelfit, fit_summary, topcells_are_shuffled):

    active_cells_decay_actual  = np.zeros((nact,nfile,len(t),len(t)))
    active_cells_decay_norm    = np.zeros((nact,nfile,len(t),len(t)))
    active_cells_decay_binary  = np.zeros((nact,nfile,len(t),len(t)))
    # active_cells_decay_shuffle          = np.zeros((nfile,len(t),len(t)))
    for ax, act in enumerate(acts):
        for fx in range(nfile):
            nonoutlier = fit_summary['nonoutlier'][fx]
            data_fx = modelfit[f'sess{fx}'][act]['data'][nonoutlier, :]
            # actual
            data_fx_actual = np.copy(data_fx)
            # normalized
            data_fx_norm = data_fx / np.linalg.norm(data_fx,axis=0)
            # data_fx_norm = (data_fx / np.linalg.norm(data_fx,axis=0))**2
            # binary
            data_fx_binary = dict_topcells[act]['data_binary'][fx]
            # # binary shuffle
            # data_fx_shuffle = np.zeros_like(data_fx_binary)
            # for ci in range(data_fx_binary.shape[0]):
            #     numones = np.sum(data_fx_binary[ci]).astype(int)
            #     randtimes = np.random.permutation(np.arange(len(t)))[:numones]
            #     data_fx_shuffle[ci,randtimes] = 1

            active_cells_decay_actual[ax,fx]   = topcell_activity(t, fx, dict_topcells[act], data_fx_actual, topcells_are_shuffled)
            active_cells_decay_norm[ax,fx]     = topcell_activity(t, fx, dict_topcells[act], data_fx_norm,   topcells_are_shuffled)
            active_cells_decay_binary[ax,fx]   = topcell_activity(t, fx, dict_topcells[act], data_fx_binary, topcells_are_shuffled)
            # active_cells_decay_shuffle[fx] = topcell_activity(t,fx, dict_topcells, data_fx_shuffle)
        
    return active_cells_decay_actual, active_cells_decay_norm, active_cells_decay_binary
        
    
def compute_new_active_cells(t, nact, nfile, acts, dict_topcells):
    active_cells_decay          = np.zeros((nact,nfile,len(t),len(t)))
    active_cells_decay_norm     = np.zeros((nact,nfile,len(t),len(t)))
    new_active_cells_id         = {act:{i:{} for i in range(nfile)} for act in acts}
    new_active_cells_duration   = {act:{i:{} for i in range(nfile)} for act in acts}
    for ax, act in enumerate(acts):
        for fx in range(nfile):
            _data_binary = dict_topcells[act]['data_binary'][fx].astype(bool)
            _ncell = _data_binary.shape[0]
            _active_cells_decay = np.zeros((len(t),len(t)))
            _active_cells_decay_norm = np.zeros((len(t),len(t)))
            _pre_active_cells = np.zeros(_ncell).astype(bool)
            _new_active_cells_id = {i:[] for i in range(len(t))}
            _new_active_cells_duration = {i:[] for i in range(len(t))}
            for i in range(len(t)):
                _active_cells_decay_at_t = np.zeros(len(t)-i)
                _active_cells_decay_norm_at_t = np.zeros(len(t)-i)
                _pre_active_cells      += _data_binary[:,i-1] if i > 0 else False # accumulate cells activated previously
                _new_active_cells_at_i  = ~_pre_active_cells * _data_binary[:,i]
                for jix, j in enumerate(np.arange(i,len(t))):
                    #---- decay of all active cells at t
                    _active_cells_decay_at_t[jix] = np.sum(_data_binary[:,i]*_data_binary[:,j]) / np.sum(_ncell)
                    _active_cells_decay_norm_at_t[jix] = np.sum(_data_binary[:,i]*_data_binary[:,j]) / np.sum(_data_binary[:,i])
                    # #---- decay of activated cell group at t 
                    # _still_active_cells_at_j = _new_active_cells_at_i * _data_binary[:,j]
                    # _active_cells_decay_at_t[jix]        = np.sum(_still_active_cells_at_j) / np.sum(_ncell)
                    # _active_cells_decay_norm_at_t[jix]   = np.sum(_still_active_cells_at_j) / np.sum(_new_active_cells_at_i)            
                # duration of each active neuron (on to off time)
                _new_active_cells_id[i] = np.where(_new_active_cells_at_i)[0]
                _new_active_cells_duration[i] = np.array([idx_active_off[0] if (idx_active_off := np.where(~row)[0]).size else len(row) for row in _data_binary[_new_active_cells_at_i,i:]])
                # save the decay of active neurons at t
                _active_cells_decay[i,i:] = _active_cells_decay_at_t
                _active_cells_decay_norm[i,i:] = _active_cells_decay_norm_at_t
            active_cells_decay[ax,fx]          = _active_cells_decay
            active_cells_decay_norm[ax,fx]     = _active_cells_decay_norm
            new_active_cells_id[act][fx]       = _new_active_cells_id
            new_active_cells_duration[act][fx] = _new_active_cells_duration
    return new_active_cells_id, new_active_cells_duration

def compute_neuron_time_constant(nfile, act1, idx_amp, idx_taud, new_active_cells_id, new_active_cells_duration, fit_summary, modelfit):
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

