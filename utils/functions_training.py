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

def create_dict_topcells_4groups(par, dict_topcells_1x2_X):
    
    condition_groups = {
        'Group 0': [
            ('P1+A1-', 'P2+A2-'),
            ('P1+A1-', 'P2+A2+'),
        ],
        'Group 1': [
            ('P1+A1-', 'P2-A2+'),
        ],
        'Group 2': [
            ('P1+A1+', 'P2-A2+'),
            ('P1-A1+', 'P2-A2+'),
        ],
        'Group 3': [
            ('P1+A1+', 'P2+A2-'),
            ('P1+A1+', 'P2+A2+'),
            ('P1-A1+', 'P2+A2-'),
            ('P1-A1+', 'P2+A2+'),
        ],
    }
    dict_topcells_1x2_fx = {
        'Group0': np.array([]),
        'Group1': np.array([]),
        'Group2': np.array([]),
        'Group3': np.array([])
    }
    dict_topcells_X_4groups = {fx:copy.deepcopy(dict_topcells_1x2_fx) for fx in range(par.nfile)}

    for fx in range(par.nfile):
        for ig, condition_pairs in enumerate(condition_groups.values()):
            cells_group = [
                dict_topcells_1x2_X[fx][key1][key2]
                for key1, key2 in condition_pairs
            ]
            cells_group = [
                cells for cells in cells_group
                if len(cells) > 0
            ]
            if len(cells_group) > 0:
                dict_topcells_X_4groups[fx][f'Group{ig}'] = np.unique(np.concatenate(cells_group))
    
    return dict_topcells_X_4groups

def create_activity_4groups(par, dict_topcells_X_4groups, data_nonoutlier):

    dict_trialtypes = {
        'P1': np.array([]),
        'A1': np.array([]),
        'P2': np.array([]),
        'A2': np.array([])
    }
    dict_activity_X_4groups = {
        fx: copy.deepcopy(dict_trialtypes) 
        for fx in range(par.nfile)
    }

    for fx in range(par.nfile):
        for k3 in par.keys3:
            _data_fx_k3 = data_nonoutlier[fx][k3]
            for ig in range(4):
                _topcells_in_group = dict_topcells_X_4groups[fx][f'Group{ig}']
                if len(_topcells_in_group) > 0:
                    _data_fx_k3_group = _data_fx_k3[_topcells_in_group]
                    dict_activity_X_4groups[fx][k3] = np.concatenate(
                        (dict_activity_X_4groups[fx][k3].reshape(-1, _data_fx_k3.shape[1]),
                        _data_fx_k3_group),
                        axis=0
                    )            
                

    return dict_activity_X_4groups

def combine_topcells_SDR(par, dict_topcells_S_4groups, dict_topcells_D_4groups, dict_topcells_R_4groups, nonoutlier, num_og_cells): 
    dict_group_type = {
        'Group0': np.array([]),
        'Group1': np.array([]),
        'Group2': np.array([]),
        'Group3': np.array([])
    }
    dict_neuron_type = {
        'num_og_cells': np.array([]),
        'nonoutlier': np.array([]),
        'S': copy.deepcopy(dict_group_type),
        'D': copy.deepcopy(dict_group_type),
        'R': copy.deepcopy(dict_group_type)
    }
    dict_topcells_4groups = {fx:copy.deepcopy(dict_neuron_type) for fx in range(par.nfile)}

    for fx in range(par.nfile):
        for neutype in ['num_og_cells','nonoutlier','S','D','R']:
            if neutype == 'num_og_cells':
                dict_topcells_4groups[fx][neutype] = num_og_cells[fx]
            if neutype == 'nonoutlier':
                dict_topcells_4groups[fx][neutype] = nonoutlier[fx]
            if neutype == 'S':
                for ig in range(4):
                    dict_topcells_4groups[fx][neutype][f'Group{ig}'] = dict_topcells_S_4groups[fx][f'Group{ig}']
            if neutype == 'D':
                for ig in range(4):
                    dict_topcells_4groups[fx][neutype][f'Group{ig}'] = dict_topcells_D_4groups[fx][f'Group{ig}']
            if neutype == 'R':
                for ig in range(4):
                    dict_topcells_4groups[fx][neutype][f'Group{ig}'] = dict_topcells_R_4groups[fx][f'Group{ig}']
    return dict_topcells_4groups

def combine_activity_SDR(par, dict_activity_S_4groups, dict_activity_D_4groups, dict_activity_R_4groups):
    dict_trialtypes = {
        'P1': np.array([]),
        'A1': np.array([]),
        'P2': np.array([]),
        'A2': np.array([])
    }
    dict_activity_4groups = {
        fx: copy.deepcopy(dict_trialtypes) 
        for fx in range(par.nfile)
    }

    for fx in range(par.nfile):
        for k3 in par.keys3:
            dict_activity_4groups[fx][k3] = np.concatenate(
                (dict_activity_S_4groups[fx][k3],
                dict_activity_D_4groups[fx][k3],
                dict_activity_R_4groups[fx][k3]),
                axis=0
                )
    return dict_activity_4groups


