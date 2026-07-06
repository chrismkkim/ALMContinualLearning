#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from utils import functions
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import copy
from utils import plot_stimulus 
from utils import plot_neuralstate_cd
from utils import plot_neuralstate_eucldist
plt.style.use('pyplot_setting.mplstyle')

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
relearn_speed = metadata[colname_relearn_speed]
sorted_indices_by_relearn_speed = np.argsort(relearn_speed)


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

nshuff = 15
nfile  = len(metadata)
CD_dotproduct_all  = np.zeros((nshuff,nfile))
CD_sample_1       = np.empty((nshuff,nfile),dtype=object)
CD_sample_2       = np.empty((nshuff,nfile),dtype=object)


# %%

corr_PA_pos = np.zeros(nfile)
corr_PA_neg = np.zeros(nfile)
corr_RL_pos = np.zeros(nfile)
corr_RL_neg = np.zeros(nfile)
dP = {}
dA = {}
dR = {}
dL = {}
R1 = {}
L1 = {}
R2 = {}
L2 = {}
CD1 = {}
CD2 = {}
CDdotprod = {}
cells_pos = {}
cells_neg = {}
topcells_pos = {}
topcells_neg = {}
num_totalcells = np.zeros(nfile)
num_outliers = np.zeros(nfile)
num_topcells = np.zeros(nfile)
shuff = 0
fx = 1

def find_topcells(CDdotprod, cells, thr):
    # trim cells
    CDdotprod_trim = np.abs(CDdotprod[cells])
    # sort the CD dot product (descending)
    cells_sorted = np.argsort(CDdotprod_trim)[::-1]
    # cumulative sum of the sorted CD dot product
    CDdotprod_cumsum = np.cumsum(CDdotprod_trim[cells_sorted])
    CDdotprod_cumsum_norm = CDdotprod_cumsum / CDdotprod_cumsum[-1]
    # find the top 99%
    topcells = cells[cells_sorted[:np.where(CDdotprod_cumsum_norm > thr)[0][0]]]
    return topcells

def remove_outliers(x1R,x1L,x2R,x2L,maxstd):
    x1   = np.append(x1R,x1L)
    x2   = np.append(x2R,x2L)
    z1R  = (x1R - np.mean(x1)) / np.std(x1)
    z1L  = (x1L - np.mean(x1)) / np.std(x1)
    z2R  = (x2R - np.mean(x2)) / np.std(x2)
    z2L  = (x2L - np.mean(x2)) / np.std(x2)
    keep = (np.abs(z1R)<maxstd) * (np.abs(z1L)<maxstd) * (np.abs(z2R)<maxstd) * (np.abs(z2L)<maxstd)
    return keep

for fx in range(nfile):
    # print(fx)
    fx_sorted = sorted_indices_by_relearn_speed[fx]

    #--- sorted by relearn speed ---#
    filepath   = dirpath + merged_ID[fx_sorted] + '.npy'
    data       = np.load(filepath, allow_pickle=True)
    opto_sess1 = data['deconvolved'][0]
    opto_sess2 = data['deconvolved'][1]
    ncell      = opto_sess1.shape[1]

    # Compute Coding Direction @ delay
    data_type = 'deconvolved'

    CD_dotproduct, CD_sess1_population, CD_sess2_population, \
    proj_CD_1R, proj_CD_1L, proj_CD_2R, proj_CD_2L,\
    opto_1R_testtrials, opto_1L_testtrials, opto_2R_testtrials, opto_2L_testtrials, \
    opto_1R_traintrials, opto_1L_traintrials, opto_2R_traintrials, opto_2L_traintrials = functions.compute_CD_delay(data, data_type, 'input')

    CD_dotproduct_all[shuff,fx] = CD_dotproduct

    # neural activity (train trials)
    opto_1P_traintrials     = opto_1R_traintrials
    opto_1P_traintrials_avg = np.mean(opto_1P_traintrials,axis=2) # neurons x time

    # neural activity (test trials)
    opto_1R_testtrials_avg = np.mean(opto_1R_testtrials,axis=2) # neurons x time
    opto_1L_testtrials_avg = np.mean(opto_1L_testtrials,axis=2) # neurons x time
    opto_2R_testtrials_avg = np.mean(opto_2R_testtrials,axis=2) # neurons x time
    opto_2L_testtrials_avg = np.mean(opto_2L_testtrials,axis=2) # neurons x time        
    opto_1R_delay = np.mean(opto_1R_testtrials_avg[:,tix_response-4:tix_response],axis=1)
    opto_1L_delay = np.mean(opto_1L_testtrials_avg[:,tix_response-4:tix_response],axis=1)
    opto_2R_delay = np.mean(opto_2R_testtrials_avg[:,tix_response-4:tix_response],axis=1)
    opto_2L_delay = np.mean(opto_2L_testtrials_avg[:,tix_response-4:tix_response],axis=1)

    # remove outlier cells
    keep             = remove_outliers(opto_1R_delay,opto_1L_delay,opto_2R_delay,opto_2L_delay,maxstd=5)
    
    # learned activity
    R1[fx] = opto_1R_delay[keep]
    L1[fx] = opto_1L_delay[keep]
    R2[fx] = opto_2R_delay[keep]
    L2[fx] = opto_2L_delay[keep]
    
    dP[fx] = L2[fx] - R1[fx]
    dA[fx] = R2[fx] - L1[fx]
    dR[fx] = R2[fx] - R1[fx]
    dL[fx] = L2[fx] - L1[fx]
    
    # coding direction
    CD1[fx]       =   R1[fx] - L1[fx]
    CD2[fx]       = -(R2[fx] - L2[fx])
    CDdotprod[fx] = CD1[fx] * CD2[fx]
    
    # include top cells only
    cells_pos[fx]    = np.where(CDdotprod[fx] > 0)[0]
    cells_neg[fx]    = np.where(CDdotprod[fx] < 0)[0]
    topcells_pos[fx] = find_topcells(CDdotprod[fx], cells_pos[fx], thr=0.99)
    topcells_neg[fx] = find_topcells(CDdotprod[fx], cells_neg[fx], thr=0.99)            
    
    # cell counts
    num_totalcells[fx] = ncell
    num_outliers[fx]   = np.sum(~keep)
    num_topcells[fx]   = len(topcells_pos[fx]) + len(topcells_neg[fx])


#%%
# fraction of outliers (removed) and top cells (analyzed)
plt.figure(figsize=(1.8,2.0))
plt.subplot(211)
plt.plot(num_outliers / num_totalcells)
plt.ylabel('frac. outliers')
plt.ylim([0,0.03])
plt.xticks([0,10,20,30])
plt.subplot(212)
plt.plot(num_topcells / num_totalcells)
plt.xlabel('Experiment sess')
plt.ylabel('frac. top cells')
plt.ylim([0.4,0.8])
plt.xticks([0,10,20,30])
plt.tight_layout()

#%%
def fraction_of(x,sgn):
    if sgn == '+':
        frac = np.sum(x>0) / len(x)
    elif sgn == '-':
        frac = np.sum(x<0) / len(x)
    return frac

dict_sess = {
            'dCD-': {
                'CD1-': {},
                'CD1+': {}
                    },
            'dCD+': {
                'CD1-': {},
                'CD1+': {}
                }
            }

learned_activity = {f'sess{i}': copy.deepcopy(dict_sess) for i in range(nfile)}
frac_cells = {f'sess{i}': copy.deepcopy(dict_sess) for i in range(nfile)}

for fx in range(nfile):
    dCD_neg_CD1_pos = CD1[fx][topcells_neg[fx]] > 0
    dCD_neg_CD1_neg = CD1[fx][topcells_neg[fx]] < 0
    dCD_pos_CD1_pos = CD1[fx][topcells_pos[fx]] > 0
    dCD_pos_CD1_neg = CD1[fx][topcells_pos[fx]] < 0

    learned_activity[f'sess{fx}']['dCD-']['CD1+']['dP'] = dP[fx][topcells_neg[fx][dCD_neg_CD1_pos]]
    learned_activity[f'sess{fx}']['dCD-']['CD1+']['dA'] = dA[fx][topcells_neg[fx][dCD_neg_CD1_pos]]
    learned_activity[f'sess{fx}']['dCD-']['CD1-']['dP'] = dP[fx][topcells_neg[fx][dCD_neg_CD1_neg]]
    learned_activity[f'sess{fx}']['dCD-']['CD1-']['dA'] = dA[fx][topcells_neg[fx][dCD_neg_CD1_neg]]    
    learned_activity[f'sess{fx}']['dCD+']['CD1+']['dR'] = dR[fx][topcells_pos[fx][dCD_pos_CD1_pos]]
    learned_activity[f'sess{fx}']['dCD+']['CD1+']['dL'] = dL[fx][topcells_pos[fx][dCD_pos_CD1_pos]]
    learned_activity[f'sess{fx}']['dCD+']['CD1-']['dR'] = dR[fx][topcells_pos[fx][dCD_pos_CD1_neg]]
    learned_activity[f'sess{fx}']['dCD+']['CD1-']['dL'] = dL[fx][topcells_pos[fx][dCD_pos_CD1_neg]]    

    frac_cells[f'sess{fx}']['dCD-']['CD1+']['dP<0'] = fraction_of(learned_activity[f'sess{fx}']['dCD-']['CD1+']['dP'],'-')
    frac_cells[f'sess{fx}']['dCD-']['CD1+']['dA>0'] = fraction_of(learned_activity[f'sess{fx}']['dCD-']['CD1+']['dA'],'+')
    frac_cells[f'sess{fx}']['dCD-']['CD1-']['dP>0'] = fraction_of(learned_activity[f'sess{fx}']['dCD-']['CD1-']['dP'],'+')
    frac_cells[f'sess{fx}']['dCD-']['CD1-']['dA<0'] = fraction_of(learned_activity[f'sess{fx}']['dCD-']['CD1-']['dA'],'-')    
    frac_cells[f'sess{fx}']['dCD+']['CD1+']['dR<0'] = fraction_of(learned_activity[f'sess{fx}']['dCD+']['CD1+']['dR'],'-')
    frac_cells[f'sess{fx}']['dCD+']['CD1+']['dL>0'] = fraction_of(learned_activity[f'sess{fx}']['dCD+']['CD1+']['dL'],'+')
    frac_cells[f'sess{fx}']['dCD+']['CD1-']['dR>0'] = fraction_of(learned_activity[f'sess{fx}']['dCD+']['CD1-']['dR'],'+')
    frac_cells[f'sess{fx}']['dCD+']['CD1-']['dL<0'] = fraction_of(learned_activity[f'sess{fx}']['dCD+']['CD1-']['dL'],'-')


frac_cells_sum = copy.deepcopy(dict_sess)
for fx in range(nfile):    
    if fx == 0:
        frac_cells_sum['dCD-']['CD1+']['dP<0'] = np.array([])
        frac_cells_sum['dCD-']['CD1+']['dA>0'] = np.array([])
        frac_cells_sum['dCD-']['CD1-']['dP>0'] = np.array([])
        frac_cells_sum['dCD-']['CD1-']['dA<0'] = np.array([])
        frac_cells_sum['dCD+']['CD1+']['dR<0'] = np.array([])
        frac_cells_sum['dCD+']['CD1+']['dL>0'] = np.array([])
        frac_cells_sum['dCD+']['CD1-']['dR>0'] = np.array([])
        frac_cells_sum['dCD+']['CD1-']['dL<0'] = np.array([])
    
    frac_cells_sum['dCD-']['CD1+']['dP<0'] = np.append(frac_cells_sum['dCD-']['CD1+']['dP<0'], frac_cells[f'sess{fx}']['dCD-']['CD1+']['dP<0'])
    frac_cells_sum['dCD-']['CD1+']['dA>0'] = np.append(frac_cells_sum['dCD-']['CD1+']['dA>0'], frac_cells[f'sess{fx}']['dCD-']['CD1+']['dA>0'])
    frac_cells_sum['dCD-']['CD1-']['dP>0'] = np.append(frac_cells_sum['dCD-']['CD1-']['dP>0'], frac_cells[f'sess{fx}']['dCD-']['CD1-']['dP>0'])
    frac_cells_sum['dCD-']['CD1-']['dA<0'] = np.append(frac_cells_sum['dCD-']['CD1-']['dA<0'], frac_cells[f'sess{fx}']['dCD-']['CD1-']['dA<0'])
    frac_cells_sum['dCD+']['CD1+']['dR<0'] = np.append(frac_cells_sum['dCD+']['CD1+']['dR<0'], frac_cells[f'sess{fx}']['dCD+']['CD1+']['dR<0'])
    frac_cells_sum['dCD+']['CD1+']['dL>0'] = np.append(frac_cells_sum['dCD+']['CD1+']['dL>0'], frac_cells[f'sess{fx}']['dCD+']['CD1+']['dL>0'])
    frac_cells_sum['dCD+']['CD1-']['dR>0'] = np.append(frac_cells_sum['dCD+']['CD1-']['dR>0'], frac_cells[f'sess{fx}']['dCD+']['CD1-']['dR>0'])
    frac_cells_sum['dCD+']['CD1-']['dL<0'] = np.append(frac_cells_sum['dCD+']['CD1-']['dL<0'], frac_cells[f'sess{fx}']['dCD+']['CD1-']['dL<0'])
    
learned_activity_sum = copy.deepcopy(dict_sess)
for fx in range(nfile):    
    if fx == 0:
        learned_activity_sum['dCD-']['CD1+']['dP'] = np.array([])
        learned_activity_sum['dCD-']['CD1+']['dA'] = np.array([])
        learned_activity_sum['dCD-']['CD1-']['dP'] = np.array([])
        learned_activity_sum['dCD-']['CD1-']['dA'] = np.array([])
        learned_activity_sum['dCD+']['CD1+']['dR'] = np.array([])
        learned_activity_sum['dCD+']['CD1+']['dL'] = np.array([])
        learned_activity_sum['dCD+']['CD1-']['dR'] = np.array([])
        learned_activity_sum['dCD+']['CD1-']['dL'] = np.array([])
    
    learned_activity_sum['dCD-']['CD1+']['dP'] = np.append(learned_activity_sum['dCD-']['CD1+']['dP'], learned_activity[f'sess{fx}']['dCD-']['CD1+']['dP'])
    learned_activity_sum['dCD-']['CD1+']['dA'] = np.append(learned_activity_sum['dCD-']['CD1+']['dA'], learned_activity[f'sess{fx}']['dCD-']['CD1+']['dA'])
    learned_activity_sum['dCD-']['CD1-']['dP'] = np.append(learned_activity_sum['dCD-']['CD1-']['dP'], learned_activity[f'sess{fx}']['dCD-']['CD1-']['dP'])
    learned_activity_sum['dCD-']['CD1-']['dA'] = np.append(learned_activity_sum['dCD-']['CD1-']['dA'], learned_activity[f'sess{fx}']['dCD-']['CD1-']['dA'])
    learned_activity_sum['dCD+']['CD1+']['dR'] = np.append(learned_activity_sum['dCD+']['CD1+']['dR'], learned_activity[f'sess{fx}']['dCD+']['CD1+']['dR'])
    learned_activity_sum['dCD+']['CD1+']['dL'] = np.append(learned_activity_sum['dCD+']['CD1+']['dL'], learned_activity[f'sess{fx}']['dCD+']['CD1+']['dL'])
    learned_activity_sum['dCD+']['CD1-']['dR'] = np.append(learned_activity_sum['dCD+']['CD1-']['dR'], learned_activity[f'sess{fx}']['dCD+']['CD1-']['dR'])
    learned_activity_sum['dCD+']['CD1-']['dL'] = np.append(learned_activity_sum['dCD+']['CD1-']['dL'], learned_activity[f'sess{fx}']['dCD+']['CD1-']['dL'])    

#%%
    
dP_frac_cells_mean = np.array([np.mean(frac_cells_sum['dCD-']['CD1+']['dP<0'],axis=0), np.mean(frac_cells_sum['dCD-']['CD1-']['dP>0'],axis=0)])
dA_frac_cells_mean = np.array([np.mean(frac_cells_sum['dCD-']['CD1+']['dA>0'],axis=0), np.mean(frac_cells_sum['dCD-']['CD1-']['dA<0'],axis=0)])
dR_frac_cells_mean = np.array([np.mean(frac_cells_sum['dCD+']['CD1+']['dR<0'],axis=0), np.mean(frac_cells_sum['dCD+']['CD1-']['dR>0'],axis=0)])
dL_frac_cells_mean = np.array([np.mean(frac_cells_sum['dCD+']['CD1+']['dL>0'],axis=0), np.mean(frac_cells_sum['dCD+']['CD1-']['dL<0'],axis=0)])

c1, c2, c3, c4 = 'darkviolet', 'blue', 'darkorange', 'limegreen'
x1, x2, x3, x4 = 0.5, 0.85, 1.35, 1.7
plt.figure(figsize=(2.5,2.5))
plt.subplot(211)
plt.bar([x1,x2,x3,x4], np.array([dP_frac_cells_mean[0],dA_frac_cells_mean[0],dP_frac_cells_mean[1],dA_frac_cells_mean[1]]),width=0.2,color=[c1,c1,c2,c2],edgecolor=[c1,c1,c2,c2], alpha=0.3)
plt.plot(x1+0.2*np.random.rand(nfile)-0.1, frac_cells_sum['dCD-']['CD1+']['dP<0'], marker='.', linestyle='', c=c1)
plt.plot(x2+0.2*np.random.rand(nfile)-0.1, frac_cells_sum['dCD-']['CD1+']['dA>0'], marker='.', linestyle='', c=c1)
plt.plot(x3+0.2*np.random.rand(nfile)-0.1, frac_cells_sum['dCD-']['CD1-']['dP>0'], marker='.', linestyle='', c=c2)
plt.plot(x4+0.2*np.random.rand(nfile)-0.1, frac_cells_sum['dCD-']['CD1-']['dA<0'], marker='.', linestyle='', c=c2)
plt.xticks([x1,x2,x3,x4],[r'$\Delta P < 0$', r'$\Delta A > 0$', r'$\Delta P > 0$', r'$\Delta A < 0$'])
plt.annotate(r'$S_1 > 0$', xy=(0.55,1.1), xycoords='data', fontsize=8)
plt.annotate(r'$S_1 < 0$', xy=(1.40,1.1), xycoords='data', fontsize=8)
plt.title(r'$S_1 \cdot S_2 < 0$', fontsize=8)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim([0,1.2])
plt.ylabel('frac of cells', fontsize=8)

plt.subplot(212)
plt.bar([x1,x2,x3,x4], np.array([dR_frac_cells_mean[0],dL_frac_cells_mean[0],dR_frac_cells_mean[1],dL_frac_cells_mean[1]]),width=0.2,color=[c3,c3,c4,c4],edgecolor=[c3,c3,c4,c4], alpha=0.3)
plt.plot(x1+0.2*np.random.rand(nfile)-0.1, frac_cells_sum['dCD+']['CD1+']['dR<0'], marker='.', linestyle='', c=c3)
plt.plot(x2+0.2*np.random.rand(nfile)-0.1, frac_cells_sum['dCD+']['CD1+']['dL>0'], marker='.', linestyle='', c=c3)
plt.plot(x3+0.2*np.random.rand(nfile)-0.1, frac_cells_sum['dCD+']['CD1-']['dR>0'], marker='.', linestyle='', c=c4)
plt.plot(x4+0.2*np.random.rand(nfile)-0.1, frac_cells_sum['dCD+']['CD1-']['dL<0'], marker='.', linestyle='', c=c4)
plt.xticks([x1,x2,x3,x4],[r'$\Delta R < 0$', r'$\Delta L > 0$', r'$\Delta R > 0$', r'$\Delta L < 0$'])
plt.annotate(r'$S_1 > 0$', xy=(0.55,1.1), xycoords='data', fontsize=8)
plt.annotate(r'$S_1 < 0$', xy=(1.40,1.1), xycoords='data', fontsize=8)
plt.title(r'$S_1 \cdot S_2 > 0$', fontsize=8)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim([0,1.2])
plt.ylabel('frac of cells')
plt.tight_layout()

plt.savefig('figure/learned_activity/session_all/frac_with_signs_stat.pdf')
plt.close()

#%%

xlim = 0.5
plt.figure(figsize=(2.0,1.8))
plt.plot(learned_activity_sum['dCD-']['CD1+']['dP'], learned_activity_sum['dCD-']['CD1+']['dA'], marker='.', linestyle='', c=c1, mfc='None', mew=0.3, label=r'$CD1>0$', ms=4)
plt.plot(learned_activity_sum['dCD-']['CD1-']['dP'], learned_activity_sum['dCD-']['CD1-']['dA'], marker='.', linestyle='', c=c2, mfc='None', mew=0.3, label=r'$CD1<0$', ms=4)
plt.axhline(0, color='gray', linestyle='--', lw=0.5)
plt.axvline(0, color='gray', linestyle='--', lw=0.5)
plt.xlabel(r'$\Delta P$')
plt.ylabel(r'$\Delta A$')
plt.xticks([-xlim,0,xlim])
plt.yticks([-xlim,0,xlim])
plt.xlim([-xlim,xlim])
plt.ylim([-xlim,xlim])
plt.title(r'$S_1 \cdot S_2<0$', fontsize=8)
legend_elements = [Line2D([0], [0], marker='o', color=c1, label=r'$S_1>0$', linestyle='None', ms=3), Line2D([0], [0], marker='o', color=c2, label=r'$S_1<0$', linestyle='None', ms=3)]
plt.legend(frameon=False, handles=legend_elements, loc=3, fontsize=6)
plt.tight_layout()

plt.savefig('figure/learned_activity/session_all/dP_vs_dA_scatter.pdf')
plt.close()


xlim = 0.5
plt.figure(figsize=(2.0,1.8))
plt.plot(learned_activity_sum['dCD+']['CD1+']['dR'], learned_activity_sum['dCD+']['CD1+']['dL'], marker='.', linestyle='', c=c3, mfc='None', mew=0.3, label=r'$CD1>0$', ms=4)
plt.plot(learned_activity_sum['dCD+']['CD1-']['dR'], learned_activity_sum['dCD+']['CD1-']['dL'], marker='.', linestyle='', c=c4, mfc='None', mew=0.3, label=r'$CD1<0$', ms=4)
plt.axhline(0, color='gray', linestyle='--', lw=0.5)
plt.axvline(0, color='gray', linestyle='--', lw=0.5)
plt.tight_layout()
plt.xlabel(r'$\Delta R$')
plt.ylabel(r'$\Delta L$')
plt.xticks([-xlim,0,xlim])
plt.yticks([-xlim,0,xlim])
plt.xlim([-xlim,xlim])
plt.ylim([-xlim,xlim])
plt.title(r'$S_1 \cdot S_2>0$', fontsize=8)
legend_elements = [Line2D([0], [0], marker='o', color=c3, label=r'$S_1>0$', linestyle='None', ms=3), Line2D([0], [0], marker='o', color=c4, label=r'$S_1<0$', linestyle='None', ms=3)]
plt.legend(frameon=False, handles=legend_elements, loc=3, fontsize=6)
plt.tight_layout()

plt.savefig('figure/learned_activity/session_all/dR_vs_dL_scatter.pdf')
plt.close()


#%%
from scipy.stats import gaussian_kde

def density_scatter(x, y, cmap='jet', s=5, label=None):
    xy = np.vstack([x, y])    
    kde = gaussian_kde(xy, bw_method=1)
    z = kde(xy)    
    # draw sparse points first
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(
        x, y,
        c=z,
        cmap=cmap,
        s=s,
        edgecolors='none',
        label=label
    )

# CD1 > 0
density_scatter(
    learned_activity_sum['dCD-']['CD1+']['dP'],
    learned_activity_sum['dCD-']['CD1+']['dA'],
    cmap='jet',
    label=r'$CD1>0$'
)
# CD1 < 0
density_scatter(
    learned_activity_sum['dCD-']['CD1-']['dP'],
    learned_activity_sum['dCD-']['CD1-']['dA'],
    cmap='jet',
    label=r'$CD1<0$'
)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.legend()
plt.xlabel(r'$\Delta P$')
plt.ylabel(r'$\Delta A$')

#%%
'''
dP vs. dA of sessions with difference CD1*CD2
'''
file_sorted = np.argsort(CD_dotproduct_all[0])
xlim = 0.5
for ii, fx in enumerate(file_sorted):    
    plt.figure(figsize=(1.8,1.8))
    fx_sorted = sorted_indices_by_relearn_speed[fx]
    # plt.subplot(1,3,ii+1)
    plt.plot(learned_activity[f'sess{fx}']['dCD-']['CD1+']['dP'], learned_activity[f'sess{fx}']['dCD-']['CD1+']['dA'], marker='.', linestyle='', c=c1, mfc='None', mew=0.3, label=r'$CD1>0$', ms=4)
    plt.plot(learned_activity[f'sess{fx}']['dCD-']['CD1-']['dP'], learned_activity[f'sess{fx}']['dCD-']['CD1-']['dA'], marker='.', linestyle='', c=c2, mfc='None', mew=0.3, label=r'$CD1<0$', ms=4)
    plt.axhline(0, color='gray', linestyle='--', lw=0.5)
    plt.axvline(0, color='gray', linestyle='--', lw=0.5)
    plt.title(merged_ID[fx_sorted][6:16] + '\n' + r'$CD_1 \cdot CD_2$ =' + str(np.round(CD_dotproduct_all[0,fx],decimals=2)), fontsize=8)
    # correlation of dP vs dA
    _dP = np.append(learned_activity[f'sess{fx}']['dCD-']['CD1+']['dP'],learned_activity[f'sess{fx}']['dCD-']['CD1-']['dP'])
    _dA = np.append(learned_activity[f'sess{fx}']['dCD-']['CD1+']['dA'],learned_activity[f'sess{fx}']['dCD-']['CD1-']['dA'])
    _cor = np.corrcoef(_dP,_dA)[0,1]
    plt.annotate(r'$\rho$ =' + str(np.round(_cor,decimals=3)), xy=(0.02,0.05), xycoords='axes fraction', fontsize=6)
    plt.xticks([-xlim,0,xlim])
    plt.yticks([-xlim,0,xlim])
    plt.xlim([-xlim,xlim])
    plt.ylim([-xlim,xlim])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # if ii == 0:
    plt.xlabel(r'$\Delta P$', fontsize=8)
    plt.ylabel(r'$\Delta A$', fontsize=8) 
    plt.tight_layout()
    plt.savefig('figure/learned_activity/session_each/dP_dA/' + str(ii) + '_dP_vs_dA_sessions.pdf')
    plt.close()

#%%
'''
dR vs. dL of sessions with difference CD1*CD2
'''
file_sorted = np.argsort(CD_dotproduct_all[0])
xlim = 0.3
for ii, fx in enumerate(file_sorted):    
    plt.figure(figsize=(1.8,1.8))
    fx_sorted = sorted_indices_by_relearn_speed[fx]
    # plt.subplot(1,3,ii+1)
    plt.plot(learned_activity[f'sess{fx}']['dCD+']['CD1+']['dR'], learned_activity[f'sess{fx}']['dCD+']['CD1+']['dL'], marker='.', linestyle='', c=c3, mfc='None', mew=0.3, label=r'$CD1>0$', ms=4)
    plt.plot(learned_activity[f'sess{fx}']['dCD+']['CD1-']['dR'], learned_activity[f'sess{fx}']['dCD+']['CD1-']['dL'], marker='.', linestyle='', c=c4, mfc='None', mew=0.3, label=r'$CD1<0$', ms=4)
    plt.axhline(0, color='gray', linestyle='--', lw=0.5)
    plt.axvline(0, color='gray', linestyle='--', lw=0.5)
    plt.title(merged_ID[fx_sorted][6:16] + '\n' + r'$CD_1 \cdot CD_2$ =' + str(np.round(CD_dotproduct_all[0,fx],decimals=2)), fontsize=8)
    # correlation of dR vs dL
    _dR = np.append(learned_activity[f'sess{fx}']['dCD+']['CD1+']['dR'],learned_activity[f'sess{fx}']['dCD+']['CD1-']['dR'])
    _dL = np.append(learned_activity[f'sess{fx}']['dCD+']['CD1+']['dL'],learned_activity[f'sess{fx}']['dCD+']['CD1-']['dL'])
    _cor = np.corrcoef(_dR,_dL)[0,1]
    plt.annotate(r'$\rho$ =' + str(np.round(_cor,decimals=3)), xy=(0.02,0.05), xycoords='axes fraction', fontsize=6)
    plt.xticks([-xlim,0,xlim])
    plt.yticks([-xlim,0,xlim])
    plt.xlim([-xlim,xlim])
    plt.ylim([-xlim,xlim])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # if ii == 0:
    plt.xlabel(r'$\Delta R$', fontsize=8)
    plt.ylabel(r'$\Delta L$', fontsize=8) 
    plt.tight_layout()

    plt.savefig('figure/learned_activity/session_each/dR_dL/' + str(ii) + '_dR_vs_dL_sessions.pdf')
    plt.close()


 
#%%
'''
The correlation between dP and dA are stronger if CD1*CD2 < 0
'''

def z_score(x):
    zscore = (x - np.mean(x)) / np.std(x)
    return zscore
    
corr_dP_vs_dA = np.zeros(nfile)
corr_dR_vs_dL = np.zeros(nfile)
_dP = np.array([])
_dA = np.array([])
_dR = np.array([])
_dL = np.array([])
for fx in range(nfile):
    _dP = np.append(learned_activity[f'sess{fx}']['dCD-']['CD1+']['dP'], learned_activity[f'sess{fx}']['dCD-']['CD1-']['dP'])
    _dA = np.append(learned_activity[f'sess{fx}']['dCD-']['CD1+']['dA'], learned_activity[f'sess{fx}']['dCD-']['CD1-']['dA'])
    _dR = np.append(learned_activity[f'sess{fx}']['dCD+']['CD1+']['dR'], learned_activity[f'sess{fx}']['dCD+']['CD1-']['dR'])
    _dL = np.append(learned_activity[f'sess{fx}']['dCD+']['CD1+']['dL'], learned_activity[f'sess{fx}']['dCD+']['CD1-']['dL'])
    corr_dP_vs_dA[fx] = np.corrcoef(_dP,_dA)[0,1] 
    corr_dR_vs_dL[fx] = np.corrcoef(_dR,_dL)[0,1] 

plt.figure(figsize=(1.8,1.5))
plt.plot(CD_dotproduct_all[0], corr_dP_vs_dA, marker='o', linestyle='', c='k', ms=4, mfc='None', mew=0.5)
plt.xlabel(r'$CD_1 \cdot CD_2$')
plt.ylabel(r'corr of $\Delta P$ and $\Delta A$')
_corr = np.corrcoef(CD_dotproduct_all[0], corr_dP_vs_dA)[0,1]
plt.annotate(r'$\rho = $' + str(np.round(_corr,decimals=3)), xy=(0.4,0.05), xycoords='axes fraction', fontsize=8)
plt.ylim([-1,0.2])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

plt.savefig('figure/learned_activity/correlation_activity/dP_vs_dA_correlation.pdf')
plt.close()



plt.figure(figsize=(1.8,1.5))
plt.plot(CD_dotproduct_all[0], corr_dR_vs_dL, marker='o', linestyle='', c='k', ms=4, mfc='None', mew=0.5)
plt.xlabel(r'$CD_1 \cdot CD_2$')
plt.ylabel(r'corr of $\Delta R$ and $\Delta L$')
_corr = np.corrcoef(CD_dotproduct_all[0], corr_dR_vs_dL)[0,1]
plt.annotate(r'$\rho = $' + str(np.round(_corr,decimals=3)), xy=(0.4,0.05), xycoords='axes fraction', fontsize=8)
plt.ylim([-1,0.2])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

plt.savefig('figure/learned_activity/correlation_activity/dR_vs_dL_correlation.pdf')
plt.close()


#%%

    
corr_dP_vs_dA = np.zeros(nfile)
frac_dP = np.zeros((2,nfile))
frac_dA = np.zeros((2,nfile))
frac_dR = np.zeros((2,nfile))
frac_dL = np.zeros((2,nfile))
for fx in range(nfile):
    frac_dP[0,fx] = frac_cells[f'sess{fx}']['dCD-']['CD1+']['dP<0']
    frac_dP[1,fx] = frac_cells[f'sess{fx}']['dCD-']['CD1-']['dP>0']
    frac_dA[0,fx] = frac_cells[f'sess{fx}']['dCD-']['CD1+']['dA>0']
    frac_dA[1,fx] = frac_cells[f'sess{fx}']['dCD-']['CD1-']['dA<0']

    frac_dR[0,fx] = frac_cells[f'sess{fx}']['dCD+']['CD1+']['dR<0']
    frac_dR[1,fx] = frac_cells[f'sess{fx}']['dCD+']['CD1-']['dR>0']
    frac_dL[0,fx] = frac_cells[f'sess{fx}']['dCD+']['CD1+']['dL>0']
    frac_dL[1,fx] = frac_cells[f'sess{fx}']['dCD+']['CD1-']['dL<0']

dP_dA0 = np.append(frac_dP[0],frac_dA[0])
dP_dA1 = np.append(frac_dP[1],frac_dA[1])
dR_dL0 = np.append(frac_dR[0],frac_dL[0])
dR_dL1 = np.append(frac_dR[1],frac_dL[1])
_CDdotprod = np.append(CD_dotproduct_all[0],CD_dotproduct_all[0])
corr_dP_dA_CD1_pos = np.corrcoef(_CDdotprod, dP_dA0)[0,1]
corr_dP_dA_CD1_neg = np.corrcoef(_CDdotprod, dP_dA1)[0,1]
corr_dR_dL_CD1_pos = np.corrcoef(_CDdotprod, dR_dL0)[0,1]
corr_dR_dL_CD1_neg = np.corrcoef(_CDdotprod, dR_dL1)[0,1]


plt.figure(figsize=(2.1,1.5))
plt.plot(CD_dotproduct_all[0], frac_dP[0], marker='o', linestyle='', c=c1, ms=3, mfc='lightgray', mew=0.5)
plt.plot(CD_dotproduct_all[0], frac_dP[0], marker='o', linestyle='', c=c1, ms=3, mfc='None', mew=0.5)
plt.plot(CD_dotproduct_all[0], frac_dA[0], marker='o', linestyle='', c=c1, ms=3, mfc='None', mew=0.5)
plt.annotate(r'$\rho = $' + str(np.round(corr_dP_dA_CD1_pos,decimals=3)), xy=(0.05,0.05), xycoords='axes fraction', fontsize=8)
plt.xlabel(r'$CD_1 \cdot CD_2$')
plt.ylabel('frac of cells')
plt.xticks([-0.5,0])
plt.ylim([0.7,1.02])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
legend_elements = [Line2D([0], [0], marker='o', color=c1, label=r'$\Delta P<0$', linestyle='None', ms=3, mfc='None', mew=0.5), Line2D([0], [0], marker='o', mec=c1, mfc='lightgray', label=r'$\Delta A>0$', linestyle='None', ms=3, mew=0.5)]
plt.legend(frameon=False, handles=legend_elements, fontsize=6, bbox_to_anchor=[1,1])
plt.tight_layout()

plt.savefig('figure/learned_activity/frac_cells/frac_dCD_correlation_dP.pdf')
plt.close()


plt.figure(figsize=(2.1,1.5))
plt.plot(CD_dotproduct_all[0], frac_dP[1], marker='o', linestyle='', c=c2, ms=3, mfc='lightgray', mew=0.5)
plt.plot(CD_dotproduct_all[0], frac_dP[1], marker='o', linestyle='', c=c2, ms=3, mfc='None', mew=0.5)
plt.plot(CD_dotproduct_all[0], frac_dA[1], marker='o', linestyle='', c=c2, ms=3, mfc='None', mew=0.5)
plt.annotate(r'$\rho = $' + str(np.round(corr_dP_dA_CD1_neg,decimals=3)), xy=(0.05,0.05), xycoords='axes fraction', fontsize=8)
plt.xlabel(r'$CD_1 \cdot CD_2$')
plt.ylabel('frac of cells')
plt.xticks([-0.5,0])
plt.ylim([0.7,1.02])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
legend_elements = [Line2D([0], [0], marker='o', color=c2, label=r'$\Delta P>0$', linestyle='None', ms=3, mfc='None', mew=0.5), Line2D([0], [0], marker='o', mec=c2, mfc='lightgray', label=r'$\Delta A<0$', linestyle='None', ms=3, mew=0.5)]
plt.legend(frameon=False, handles=legend_elements, fontsize=6, bbox_to_anchor=[1,1])
plt.tight_layout()

plt.savefig('figure/learned_activity/frac_cells/frac_dCD_correlation_dA.pdf')
plt.close()
   

plt.figure(figsize=(2.1,1.5))
plt.plot(CD_dotproduct_all[0], frac_dR[0], marker='o', linestyle='', c=c3, ms=3, mfc='lightgray', mew=0.5)
plt.plot(CD_dotproduct_all[0], frac_dR[0], marker='o', linestyle='', c=c3, ms=3, mfc='None', mew=0.5)
plt.plot(CD_dotproduct_all[0], frac_dL[0], marker='o', linestyle='', c=c3, ms=3, mfc='None', mew=0.5)
plt.annotate(r'$\rho = $' + str(np.round(corr_dR_dL_CD1_pos,decimals=3)), xy=(0.05,0.05), xycoords='axes fraction', fontsize=8)
plt.xlabel(r'$CD_1 \cdot CD_2$')
plt.ylabel('frac of cells')
plt.xticks([-0.5,0])
plt.ylim([0.6,1.02])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
legend_elements = [Line2D([0], [0], marker='o', color=c3, label=r'$\Delta R<0$', linestyle='None', ms=3, mfc='None', mew=0.5), Line2D([0], [0], marker='o', mec=c3, mfc='lightgray', label=r'$\Delta L>0$', linestyle='None', ms=3, mew=0.5)]
plt.legend(frameon=False, handles=legend_elements, fontsize=6, bbox_to_anchor=[1,1])
plt.tight_layout()
   
plt.savefig('figure/learned_activity/frac_cells/frac_dCD_correlation_dR.pdf')
plt.close()


plt.figure(figsize=(2.1,1.5))
plt.plot(CD_dotproduct_all[0], frac_dR[1], marker='o', linestyle='', c=c4, ms=3, mfc='lightgray', mew=0.5)
plt.plot(CD_dotproduct_all[0], frac_dR[1], marker='o', linestyle='', c=c4, ms=3, mfc='None', mew=0.5)
plt.plot(CD_dotproduct_all[0], frac_dL[1], marker='o', linestyle='', c=c4, ms=3, mfc='None', mew=0.5)
plt.annotate(r'$\rho = $' + str(np.round(corr_dR_dL_CD1_neg,decimals=3)), xy=(0.05,0.05), xycoords='axes fraction', fontsize=8)
plt.xlabel(r'$CD_1 \cdot CD_2$')
plt.ylabel('frac of cells')
plt.xticks([-0.5,0])
plt.ylim([0.6,1.02])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
legend_elements = [Line2D([0], [0], marker='o', color=c4, label=r'$\Delta R>0$', linestyle='None', ms=3, mfc='None', mew=0.5), Line2D([0], [0], marker='o', mec=c4, mfc='lightgray', label=r'$\Delta L<0$', linestyle='None', ms=3, mew=0.5)]
plt.legend(frameon=False, handles=legend_elements, fontsize=6, bbox_to_anchor=[1,1])
plt.tight_layout()
   
plt.savefig('figure/learned_activity/frac_cells/frac_dCD_correlation_dL.pdf')
plt.close()
   
 