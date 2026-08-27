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

def create_dict_topcells_1x2(nfile, dict_topcells, tix_delay_range):
    
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

    for fx in range(nfile):                
        topcells_P1 = np.unique(np.concatenate([dict_topcells['P1']['topcells_at_t'][fx][i] for i in tix_delay_range]))
        topcells_A1 = np.unique(np.concatenate([dict_topcells['A1']['topcells_at_t'][fx][i] for i in tix_delay_range]))
        topcells_P2 = np.unique(np.concatenate([dict_topcells['P2']['topcells_at_t'][fx][i] for i in tix_delay_range]))
        topcells_A2 = np.unique(np.concatenate([dict_topcells['A2']['topcells_at_t'][fx][i] for i in tix_delay_range]))
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

    return dict_topcells_1x2

def create_dict_CDdotprod(nfile, dict_topcells, dict_topcells_1x2, tix_delay_range, fit_summary, modelfit):
    
    CDdp = np.zeros(nfile)
    CDdp_1x2 = np.zeros(nfile)
    dict_CDdp = copy.deepcopy(dict_topcells_1x2)
    tix_late_delay = np.arange(start=12,stop=16)
    for fx in range(nfile):
        # topcells
        topcells_P1 = np.unique(np.concatenate([dict_topcells['P1']['topcells_at_t'][fx][i] for i in tix_delay_range]))
        topcells_A1 = np.unique(np.concatenate([dict_topcells['A1']['topcells_at_t'][fx][i] for i in tix_delay_range]))
        topcells_P2 = np.unique(np.concatenate([dict_topcells['P2']['topcells_at_t'][fx][i] for i in tix_delay_range]))
        topcells_A2 = np.unique(np.concatenate([dict_topcells['A2']['topcells_at_t'][fx][i] for i in tix_delay_range]))

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
        diff1 = np.mean((data_P1 - data_A1)[:,tix_late_delay],axis=1)
        diff2 = np.mean((data_P2 - data_A2)[:,tix_late_delay],axis=1)
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

    return dict_CDdp, CDdp, CDdp_1x2

def create_dict_CDdotprod_err(nfile, dict_CDdp, CDdp):
    
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

    return dict_CDdp_err

def create_dict_module_activity(nfile, dict_topcells_1x2, fit_summary, modelfit, keys1, keys2, keys3):
    
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
    dict_module_activity = {
        'P1+A1-': copy.deepcopy(dict_topcells_2),
        'P1+A1+': copy.deepcopy(dict_topcells_2),
        'P1-A1+': copy.deepcopy(dict_topcells_2),
    }

    for fx in range(nfile):
        nonoutlier = fit_summary['nonoutlier'][fx]
        for k1 in keys1:
            for k2 in keys2:
                _topcells_1x2 = dict_topcells_1x2[fx][k1][k2]
                for k3 in keys3:
                    _data_k3 = modelfit[f'sess{fx}'][k3]['data'][nonoutlier,:]
                    if len(_topcells_1x2) > 0:
                        _data_k3_module = np.sum(_data_k3[_topcells_1x2],axis=0)
                    else:
                        _data_k3_module = np.zeros(_data_k3.shape[1])
                    dict_module_activity[k1][k2][k3][fx] = _data_k3_module
                    
    return dict_module_activity



def create_dict_normalized_activity(nfile, dict_module_activity, fit_summary, modelfit, dict_topcells_1x2, keys1, keys2, keys3):
    
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
    dict_normalized_activity = {
        'P1+A1-': copy.deepcopy(dict_topcells_2),
        'P1+A1+': copy.deepcopy(dict_topcells_2),
        'P1-A1+': copy.deepcopy(dict_topcells_2),
    }
    
    #---- different versions of normalization ---#
    # # (1)
    # total_activity = {
    #     'P1': np.zeros((nfile,24)),
    #     'A1': np.zeros((nfile,24)),
    #     'P2': np.zeros((nfile,24)),
    #     'A2': np.zeros((nfile,24))
    # }
    # for fx in range(nfile):
    #     for k3 in keys3:
    #         for k2 in keys2:
    #             for k1 in keys1:
    #                 # total_activity[k3][fx] += dict_module_activity[k1][k2][k3][fx]
    #                 total_activity[k3][fx] += np.mean((dict_module_activity[k1][k2][k3][fx])[12:16])
                
    # # normalized activity
    # for fx in range(nfile):
    #     for k3 in keys3:
    #         for k2 in keys2:
    #             for k1 in keys1:                
    #                 dict_normalized_activity[k1][k2][k3][fx] = dict_module_activity[k1][k2][k3][fx] / total_activity[k3][fx]
    
    
    # (2)
    total_activity = {
        'P1A1': np.zeros(nfile),
        'P2A2': np.zeros(nfile),
    }

    for fx in range(nfile):
        nonoutlier = fit_summary['nonoutlier'][fx]
        for k1 in keys1:
            for k2 in keys2:
                _topcells_1x2 = dict_topcells_1x2[fx][k1][k2]
                if len(_topcells_1x2) > 0:
                    # context 1
                    _data_P1 = modelfit[f'sess{fx}']['P1']['data'][nonoutlier,:]
                    _data_A1 = modelfit[f'sess{fx}']['A1']['data'][nonoutlier,:]
                    _data_P1_topcells = np.mean(_data_P1[_topcells_1x2,12:16],axis=1)
                    _data_A1_topcells = np.mean(_data_A1[_topcells_1x2,12:16],axis=1)
                    total_activity['P1A1'][fx] += np.sum((_data_P1_topcells - _data_A1_topcells)**2)
                    # context 2
                    _data_P2 = modelfit[f'sess{fx}']['P2']['data'][nonoutlier,:]
                    _data_A2 = modelfit[f'sess{fx}']['A2']['data'][nonoutlier,:]
                    _data_P2_topcells = np.mean(_data_P2[_topcells_1x2,12:16],axis=1)
                    _data_A2_topcells = np.mean(_data_A2[_topcells_1x2,12:16],axis=1)
                    total_activity['P2A2'][fx] += np.sum((_data_P2_topcells - _data_A2_topcells)**2)                    
                else:
                    total_activity['P1A1'][fx] += 0
                    total_activity['P2A2'][fx] += 0
        total_activity['P2A2'][fx] = np.sqrt(total_activity['P2A2'][fx])
        
    # normalized activity
    for fx in range(nfile):
        for k3 in keys3:
            for k2 in keys2:
                for k1 in keys1:                
                    if k3 == 'P1' or k3 == 'A1':
                        dict_normalized_activity[k1][k2][k3][fx] = dict_module_activity[k1][k2][k3][fx] / total_activity['P1A1'][fx]
                    if k3 == 'P2' or k3 == 'A2':
                        dict_normalized_activity[k1][k2][k3][fx] = dict_module_activity[k1][k2][k3][fx] / total_activity['P2A2'][fx]

        
    return dict_normalized_activity


def create_sequential_activity(nfile, dict_topcells_1x2, dict_topcells, fit_summary, modelfit, keys1, keys2):
        
    tix_context1 = np.arange(8,16)
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
    
    return mean_popact_P1, mean_popact_A1, mean_popact_P2, mean_popact_A2


def create_dict_within_selectivity(nfile, dict_module_activity, tix_delay_range, keys1, keys2, keys_within_context, keys_across_context):    
    
    dict_trialtypes = {
        'P1-A1': np.zeros(nfile),
        'P2-A2': np.zeros(nfile),
    }
    dict_topcells_2 = {
        'P2+A2-': copy.deepcopy(dict_trialtypes),
        'P2+A2+': copy.deepcopy(dict_trialtypes),
        'P2-A2+': copy.deepcopy(dict_trialtypes)
    }
    dict_within_selectivity = {
        'P1+A1-': copy.deepcopy(dict_topcells_2),
        'P1+A1+': copy.deepcopy(dict_topcells_2),
        'P1-A1+': copy.deepcopy(dict_topcells_2),
    }

    for k1 in keys1:
        for k2 in keys2:
            for kc in keys_within_context:
                for fx in range(nfile):
                    if kc == 'P1-A1':
                        _act  = dict_module_activity[k1][k2]['P1'][fx] - dict_module_activity[k1][k2]['A1'][fx]
                    if kc == 'P2-A2':
                        _act  = dict_module_activity[k1][k2]['P2'][fx] - dict_module_activity[k1][k2]['A2'][fx]
                    _act = _act[tix_delay_range]
                    dict_within_selectivity[k1][k2][kc][fx] = np.mean(_act)

    return dict_within_selectivity


def create_dict_across_activity(nfile, dict_module_activity, tix_delay_range, keys1, keys2, keys_within_context, keys_across_context):        
        
    dict_trialtypes = {
        'P2-P1': np.zeros(nfile),
        'A2-A1': np.zeros(nfile),
    }
    dict_topcells_2 = {
        'P2+A2-': copy.deepcopy(dict_trialtypes),
        'P2+A2+': copy.deepcopy(dict_trialtypes),
        'P2-A2+': copy.deepcopy(dict_trialtypes)
    }
    dict_across_activity = {
        'P1+A1-': copy.deepcopy(dict_topcells_2),
        'P1+A1+': copy.deepcopy(dict_topcells_2),
        'P1-A1+': copy.deepcopy(dict_topcells_2),
    }
    
    for k1 in keys1:
        for k2 in keys2:
            for kc in keys_across_context:
                for fx in range(nfile):
                    if kc == 'P2-P1':
                        _act  = dict_module_activity[k1][k2]['P2'][fx] - dict_module_activity[k1][k2]['P1'][fx]
                    if kc == 'A2-A1':
                        _act  = dict_module_activity[k1][k2]['A2'][fx] - dict_module_activity[k1][k2]['A1'][fx]
                    _act = _act[tix_delay_range]
                    dict_across_activity[k1][k2][kc][fx] = np.mean(_act)
    
    return dict_across_activity

