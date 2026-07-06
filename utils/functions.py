import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import matplotlib.colors as mcolors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
import h5py 
import os


# def compute_CD_dotproduct(data, data_type):
    
#     tsample   = 1.57
#     tdelay    = 2.87
#     tresponse = 4.17
#     dt        = 1/6
#     ntimestep = 47
#     tvec      = dt * np.arange(ntimestep)
#     tix_delay    = np.where(tvec > tdelay)[0][0]
#     tix_response = np.where(tvec > tresponse)[0][0]

#     opto_sess1 = data[data_type][0]
#     opto_sess2 = data[data_type][1]
#     ncat, ncell = opto_sess1.shape
#     lickright = 0
#     lickleft  = 1

#     ntrials_1R = opto_sess1[lickright,0].shape[1]
#     ntrials_1L = opto_sess1[lickleft,0].shape[1]
#     ntrials_2R = opto_sess2[lickright,0].shape[1]
#     ntrials_2L = opto_sess2[lickleft,0].shape[1]
    
#     frac_train_trials = 0.5
#     ntrials_train_1R = int(ntrials_1R * frac_train_trials)
#     ntrials_train_1L = int(ntrials_1L * frac_train_trials)
#     ntrials_train_2R = int(ntrials_2R * frac_train_trials)
#     ntrials_train_2L = int(ntrials_2L * frac_train_trials)

#     opto_1R = np.zeros((ncell,ntimestep))
#     opto_1L = np.zeros((ncell,ntimestep))
#     opto_2R = np.zeros((ncell,ntimestep))
#     opto_2L = np.zeros((ncell,ntimestep))
#     for cell in range(ncell):        
#         #------- first half trials ----------#
#         opto_1R[cell] = np.mean(opto_sess1[lickright,cell][:,:ntrials_train_1R],axis=1)
#         opto_1L[cell] = np.mean(opto_sess1[lickleft,cell][:,:ntrials_train_1L],axis=1)
#         opto_2R[cell] = np.mean(opto_sess2[lickright,cell][:,:ntrials_train_2R],axis=1)
#         opto_2L[cell] = np.mean(opto_sess2[lickleft,cell][:,:ntrials_train_2L],axis=1)    

#         # #------- random half trials ---------#
#         # opto_1R[cell] = np.mean(opto_sess1[lickright,cell][:,np.sort(np.random.randint(0, ntrials_1R, ntrials_train_1R))],axis=1)
#         # opto_1L[cell] = np.mean(opto_sess1[lickleft,cell][:,np.sort(np.random.randint(0, ntrials_1L, ntrials_train_1L))],axis=1)
#         # opto_2R[cell] = np.mean(opto_sess2[lickright,cell][:,np.sort(np.random.randint(0, ntrials_2R, ntrials_train_2R))],axis=1)
#         # opto_2L[cell] = np.mean(opto_sess2[lickleft,cell][:,np.sort(np.random.randint(0, ntrials_2L, ntrials_train_2L))],axis=1)    
#     CD_sess1            = opto_1R - opto_1L
#     CD_sess2            = opto_2R - opto_2L
#     CD_sess1_population = np.mean(CD_sess1[:,tix_response-4:tix_response], axis=1)
#     CD_sess2_population = np.mean(CD_sess2[:,tix_response-4:tix_response], axis=1)    
#     # CD_sess1_population = np.mean(CD_sess1[:,tix_delay:tix_response], axis=1)
#     # CD_sess2_population = np.mean(CD_sess2[:,tix_delay:tix_response], axis=1)
#     CD_dotproduct       = np.inner(CD_sess1_population, CD_sess2_population) / (np.linalg.norm(CD_sess1_population) * np.linalg.norm(CD_sess2_population))

#     return CD_dotproduct, CD_sess1_population, CD_sess2_population

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
    
    # Compute coding direction - use train trials
    opto_1R = np.zeros((ncell,ntimestep))
    opto_1L = np.zeros((ncell,ntimestep))
    opto_2R = np.zeros((ncell,ntimestep))
    opto_2L = np.zeros((ncell,ntimestep))    
    for cell in range(ncell):        
        # use train trials
        opto_1R[cell] = np.mean(opto_1R_traintrials[cell,:,:],axis=1)
        opto_1L[cell] = np.mean(opto_1L_traintrials[cell,:,:],axis=1)
        opto_2R[cell] = np.mean(opto_2R_traintrials[cell,:,:],axis=1)
        opto_2L[cell] = np.mean(opto_2L_traintrials[cell,:,:],axis=1)                    
                
    CD_sess1            = opto_1R - opto_1L
    CD_sess2            = -(opto_2R - opto_2L)
    CD_duration         = 4
    CD_sess1_population = np.mean(CD_sess1[:,tix_response-CD_duration:tix_response], axis=1)
    CD_sess2_population = np.mean(CD_sess2[:,tix_response-CD_duration:tix_response], axis=1)
    CD_dotproduct       = np.inner(CD_sess1_population, CD_sess2_population) / (np.linalg.norm(CD_sess1_population) * np.linalg.norm(CD_sess2_population))


    return (opto_1R_traintrials, opto_1L_traintrials, opto_2R_traintrials, opto_2L_traintrials,
            opto_1R_testtrials,  opto_1L_testtrials,  opto_2R_testtrials,  opto_2L_testtrials)
    
def compute_CD_dotproduct(opto_1R_set1,opto_1L_set1,opto_2R_set1,opto_2L_set1,tix_response):
    
    ncell     = opto_1R_set1.shape[0]
    ntimestep = opto_1R_set1.shape[1]
    opto_1R   = np.zeros((ncell,ntimestep))
    opto_1L   = np.zeros((ncell,ntimestep))
    opto_2R   = np.zeros((ncell,ntimestep))
    opto_2L   = np.zeros((ncell,ntimestep))        
    for cell in range(ncell):        
        # use train trials
        opto_1R[cell] = np.mean(opto_1R_set1[cell,:,:],axis=1)
        opto_1L[cell] = np.mean(opto_1L_set1[cell,:,:],axis=1)
        opto_2R[cell] = np.mean(opto_2R_set1[cell,:,:],axis=1)
        opto_2L[cell] = np.mean(opto_2L_set1[cell,:,:],axis=1)                                    
    CD_sess1            = opto_1R - opto_1L
    CD_sess2            = -(opto_2R - opto_2L)
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
