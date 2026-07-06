#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import functions
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from utils import plot_stimulus 
from utils import plot_neuralstate_cd
from utils import plot_neuralstate_eucldist

# %%
dirpath = '/Users/kimchm/Documents/KimNiNature2024/MatConverted2Python/' 
metadata = pd.read_csv(dirpath + 'EDF10d_info_2026_01_16.csv')
# CDdotprod_python = pd.read_csv('CDdotprod_python_deconvolved.txt')


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

hadamard_xcont_2P_all = np.empty((nshuff,nfile,2),dtype=object)
hadamard_xcont_2A_all = np.empty((nshuff,nfile,2),dtype=object)
similarity_xcont_2P_all = np.zeros((nshuff,nfile,2,ntimestep))
similarity_xcont_2A_all = np.zeros((nshuff,nfile,2,ntimestep))
similarity_xcont_1P_all = np.zeros((nshuff,nfile,2,ntimestep))

# %%
frac_L2_R1_L_sel = np.zeros(nfile)
frac_L2_R1_R_sel = np.zeros(nfile)
frac_L2_L1_L_sel = np.zeros(nfile)
frac_L2_L1_R_sel = np.zeros(nfile)
frac_R2_R1_L_sel = np.zeros(nfile)
frac_R2_R1_R_sel = np.zeros(nfile)
frac_R2_L1_L_sel = np.zeros(nfile)
frac_R2_L1_R_sel = np.zeros(nfile)

shuff = 0
fx = 0

for fx in range(nfile):
        
    # for shuff in range(nshuff):
        
    #     print(str(shuff) + ' / ' + str(nshuff))
        
    #     for fx in range(nfile):
            
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
    opto_1P_testtrials = opto_1R_testtrials
    opto_1A_testtrials = opto_1L_testtrials
    opto_2P_testtrials = opto_2L_testtrials
    opto_2A_testtrials = opto_2R_testtrials                

    opto_1R_testtrials_avg = np.mean(opto_1R_testtrials,axis=2) # neurons x time
    opto_1L_testtrials_avg = np.mean(opto_1L_testtrials,axis=2) # neurons x time
    opto_2R_testtrials_avg = np.mean(opto_2R_testtrials,axis=2) # neurons x time
    opto_2L_testtrials_avg = np.mean(opto_2L_testtrials,axis=2) # neurons x time        

    opto_1R_delay = opto_1R_testtrials_avg[:,tix_response-2]
    opto_1L_delay = opto_1L_testtrials_avg[:,tix_response-2]
    opto_2R_delay = opto_2R_testtrials_avg[:,tix_response-2]
    opto_2L_delay = opto_2L_testtrials_avg[:,tix_response-2]

    l2_r1 = opto_2L_delay - opto_1R_delay
    l2_l1   = opto_2L_delay - opto_1L_delay
    r2_l1 = opto_2R_delay - opto_1L_delay
    r2_r1   = opto_2R_delay - opto_1R_delay
    C1_selectivity = opto_1R_delay - opto_1L_delay

    cells_sorted_l2_r1 = np.argsort(l2_r1)        
    cells_sorted_l2_l1   = np.argsort(l2_l1)        

    # L2_cells_sorted       = np.argsort(l2_r1)        
    L2_congruent_sorted   = l2_l1[cells_sorted_l2_r1]
    L2_incongruent_sorted = l2_r1[cells_sorted_l2_r1]

    L2_R1 = opto_2L_delay[cells_sorted_l2_r1] - opto_1R_delay[cells_sorted_l2_r1]
    L2_L1 = opto_2L_delay[cells_sorted_l2_r1] - opto_1L_delay[cells_sorted_l2_r1]
    R2_R1 = opto_2R_delay[cells_sorted_l2_r1] - opto_1R_delay[cells_sorted_l2_r1]
    R2_L1 = opto_2R_delay[cells_sorted_l2_r1] - opto_1L_delay[cells_sorted_l2_r1]
    R1_L1 = opto_1R_delay[cells_sorted_l2_r1] - opto_1L_delay[cells_sorted_l2_r1]

    cells_L2_R1_pos = np.where(L2_R1 > 0)[0]
    cells_L2_R1_neg = np.where(L2_R1 < 0)[0]
    cells_L2_L1_pos = np.where(L2_L1 > 0)[0]
    cells_L2_L1_neg = np.where(L2_L1 < 0)[0]
    frac_L2_R1_L_sel[fx] = np.sum(R1_L1[cells_L2_R1_pos] < 0) / len(cells_L2_R1_pos) 
    frac_L2_R1_R_sel[fx] = np.sum(R1_L1[cells_L2_R1_neg] > 0) / len(cells_L2_R1_neg) 
    frac_L2_L1_R_sel[fx] = np.sum(R1_L1[cells_L2_L1_pos] > 0) / len(cells_L2_L1_pos) 
    frac_L2_L1_L_sel[fx] = np.sum(R1_L1[cells_L2_L1_neg] < 0) / len(cells_L2_L1_neg) 
    
    cells_R2_R1_pos = np.where(R2_R1 > 0)[0]
    cells_R2_R1_neg = np.where(R2_R1 < 0)[0]
    cells_R2_L1_pos = np.where(R2_L1 > 0)[0]
    cells_R2_L1_neg = np.where(R2_L1 < 0)[0]
    frac_R2_R1_L_sel[fx] = np.sum(R1_L1[cells_R2_R1_pos] < 0) / len(cells_R2_R1_pos) 
    frac_R2_R1_R_sel[fx] = np.sum(R1_L1[cells_R2_R1_neg] > 0) / len(cells_R2_R1_neg) 
    frac_R2_L1_R_sel[fx] = np.sum(R1_L1[cells_R2_L1_pos] > 0) / len(cells_R2_L1_pos) 
    frac_R2_L1_L_sel[fx] = np.sum(R1_L1[cells_R2_L1_neg] < 0) / len(cells_R2_L1_neg)     
    
#%%

plt.figure()
plt.subplot(221)
plt.scatter(CD_dotproduct_all[0],frac_L2_R1_L_sel)
cor = np.corrcoef(CD_dotproduct_all[0], frac_L2_R1_L_sel)[0,1]
plt.title('cor ' + str(np.round(cor,decimals=3)))
plt.ylabel('L2 > R1, L selective')
plt.subplot(222)
plt.scatter(CD_dotproduct_all[0],frac_L2_R1_R_sel)
cor = np.corrcoef(CD_dotproduct_all[0], frac_L2_R1_R_sel)[0,1]
plt.title('cor ' + str(np.round(cor,decimals=3)))
plt.ylabel('L2 < R1, R selective')
plt.subplot(223)
plt.scatter(CD_dotproduct_all[0],frac_L2_L1_L_sel)
cor = np.corrcoef(CD_dotproduct_all[0], frac_L2_L1_L_sel)[0,1]
plt.title('cor ' + str(np.round(cor,decimals=3)))
plt.ylabel('L2 < L1, L selective')
plt.subplot(224)
plt.scatter(CD_dotproduct_all[0],frac_L2_L1_R_sel)
cor = np.corrcoef(CD_dotproduct_all[0], frac_L2_L1_R_sel)[0,1]
plt.title('cor ' + str(np.round(cor,decimals=3)))
plt.ylabel('L2 > L1, R selective')
plt.tight_layout()



plt.figure()
plt.subplot(221)
plt.scatter(CD_dotproduct_all[0],frac_R2_R1_L_sel)
cor = np.corrcoef(CD_dotproduct_all[0], frac_R2_R1_L_sel)[0,1]
plt.title('cor ' + str(np.round(cor,decimals=3)))
plt.ylabel('R2 > R1, L selective')
plt.subplot(222)
plt.scatter(CD_dotproduct_all[0],frac_R2_R1_R_sel)
cor = np.corrcoef(CD_dotproduct_all[0], frac_R2_R1_R_sel)[0,1]
plt.title('cor ' + str(np.round(cor,decimals=3)))
plt.ylabel('R2 < R1, R selective')
plt.subplot(223)
plt.scatter(CD_dotproduct_all[0],frac_R2_L1_L_sel)
cor = np.corrcoef(CD_dotproduct_all[0], frac_R2_L1_L_sel)[0,1]
plt.title('cor ' + str(np.round(cor,decimals=3)))
plt.ylabel('R2 < L1, L selective')
plt.subplot(224)
plt.scatter(CD_dotproduct_all[0],frac_R2_L1_R_sel)
cor = np.corrcoef(CD_dotproduct_all[0], frac_R2_L1_R_sel)[0,1]
plt.title('cor ' + str(np.round(cor,decimals=3)))
plt.ylabel('R2 > L1, R selective')
plt.tight_layout()



# plt.figure()
# plt.subplot(211)
# cor = np.corrcoef(CD_dotproduct_all[0], frac_L2_L1_L_sel)[0,1]
# plt.scatter(CD_dotproduct_all[0], frac_L2_L1_L_sel)    
# plt.title('cor ' + str(np.round(cor,decimals=3)))
# plt.ylabel('frac L selective')
# # plt.ylim([0.5,0.9])

# plt.subplot(212)
# cor = np.corrcoef(CD_dotproduct_all[0], frac_L2_L1_R_sel)[0,1]
# plt.scatter(CD_dotproduct_all[0], frac_L2_L1_R_sel)    
# plt.title('cor ' + str(np.round(cor,decimals=3)))
# plt.ylabel('frac R selective')
# plt.xlabel('CD dot product')
# # plt.ylim([0.5,0.9])
# plt.tight_layout()




# plt.figure()
# plt.hist(R1_L1[cells_L2_R1_pos], histtype='step')
# plt.hist(R1_L1[cells_L2_R1_neg], histtype='step')
# plt.tight_layout()

# idline = np.linspace(-0.1,0.1,10)
# plt.figure()
# plt.plot(L2_R1, L2_L1, marker='.', linestyle='')
# plt.plot(idline, idline, c='r', linestyle='--')
# plt.axvline(0,color='gray',linestyle='--')
# plt.axhline(0,color='gray',linestyle='--')
# plt.xlabel('L2 - R1')
# plt.ylabel('L2 - L1')
# plt.tight_layout()


# plt.figure()
# plt.plot(L2_R1, R1_L1, marker='.', linestyle='')
# plt.axvline(0,color='gray',linestyle='--')
# plt.axhline(0,color='gray',linestyle='--')
# plt.xlabel('L2 - R1')
# plt.ylabel('R1 - L1')
# plt.tight_layout()


#%%
CD1_population = np.mean(opto_1R_traintrials[:,tix_response-4:tix_response,:],axis=(1,2)) - np.mean(opto_1L_traintrials[:,tix_response-4:tix_response,:],axis=(1,2))
CD2_population = np.mean(opto_2R_traintrials[:,tix_response-4:tix_response,:],axis=(1,2)) - np.mean(opto_2L_traintrials[:,tix_response-4:tix_response,:],axis=(1,2))

ncells = CD1_population.shape[0]
abs_CD1_population  = np.abs(CD1_population)
abs_CD2_population  = np.abs(CD2_population)
cells_sorted_by_CD1 = np.argsort(abs_CD1_population)
cells_sorted_by_CD2 = np.argsort(abs_CD2_population)

nfrac = 50
frac_correct_1 = np.zeros(nfrac)
frac_correct_2 = np.zeros(nfrac)
frac_removed = np.linspace(0,1.0,nfrac)

#%%

def cell_avg(x):
    sz = 20
    ngroup = x.shape[0]//sz
    groups = sz * np.arange(ngroup)
    xavg = np.zeros(ngroup)
    for gi in range(ngroup):
        xavg[gi] = np.mean(x[gi*sz:(gi+1)*sz])
    return xavg, groups
    

#%%
plt.figure()
plt.subplot(311)
plt.plot(opto_2L_delay[L2_cells_sorted], marker='.', linestyle='')
plt.title('L2')
# plt.ylim([-0.05,0.3])
plt.subplot(312)
plt.plot(opto_1R_delay[L2_cells_sorted], marker='.', linestyle='')
plt.title('R1')
plt.ylim([-0.05,0.3])
plt.subplot(313)
plt.plot(opto_1L_delay[L2_cells_sorted], marker='.', linestyle='')
# plt.plot(opto_2L_delay[L2_cells_sorted] - opto_1R_delay[L2_cells_sorted], marker='.', linestyle='')
# plt.plot(opto_2L_delay[L2_cells_sorted] - opto_1L_delay[L2_cells_sorted], marker='.', linestyle='')
plt.title('L1')
# plt.ylim([-0.05,0.3])
plt.tight_layout()


#%%

C1_sel_incongruent, groups = cell_avg(opto_1R_delay[cells_sorted_l2_r1] - opto_1L_delay[cells_sorted_l2_r1])
C1_sel_congruent, groups = cell_avg(opto_1R_delay[cells_sorted_l2_l1] - opto_1L_delay[cells_sorted_l2_l1])

plt.figure(figsize=(8,10))
plt.subplot(321)
plt.plot(opto_2L_delay[cells_sorted_l2_r1] - opto_1R_delay[cells_sorted_l2_r1], marker='.', linestyle='')
plt.axhline(0, color='r', linestyle='--')
plt.title('L2 - R1')
# plt.ylim([-0.05,0.3])
plt.subplot(323)
plt.plot(opto_2L_delay[cells_sorted_l2_r1] - opto_1L_delay[cells_sorted_l2_r1], marker='.', linestyle='')
plt.axhline(0, color='r', linestyle='--')
plt.title('L2 - L1')
plt.subplot(325)
plt.plot(opto_1R_delay[cells_sorted_l2_r1] - opto_1L_delay[cells_sorted_l2_r1], marker='.', linestyle='')
plt.plot(groups, C1_sel_incongruent, c='magenta')
plt.axhline(0, color='r', linestyle='--')
plt.title('R1 - L1')


plt.subplot(322)
plt.plot(opto_2L_delay[cells_sorted_l2_l1] - opto_1R_delay[cells_sorted_l2_l1], marker='.', linestyle='')
plt.axhline(0, color='r', linestyle='--')
plt.title('L2 - R1')
# plt.ylim([-0.05,0.3])
plt.subplot(324)
plt.plot(opto_2L_delay[cells_sorted_l2_l1] - opto_1L_delay[cells_sorted_l2_l1], marker='.', linestyle='')
plt.axhline(0, color='r', linestyle='--')
plt.title('L2 - L1')
plt.subplot(326)
plt.plot(opto_1R_delay[cells_sorted_l2_l1] - opto_1L_delay[cells_sorted_l2_l1], marker='.', linestyle='')
# plt.plot(opto_1R_delay[cells_sorted_l2_l1], marker='.', linestyle='')
# plt.plot(- opto_1L_delay[cells_sorted_l2_l1], marker='.', linestyle='')
plt.plot(groups, C1_sel_congruent, c='magenta')
plt.axhline(0, color='r', linestyle='--')
plt.title('R1 - L1')

plt.tight_layout()


#%%

plt.figure(figsize=(8,10))
plt.subplot(321)
plt.plot(opto_2L_delay[cells_sorted_l2_r1] - opto_1R_delay[cells_sorted_l2_r1], marker='.', linestyle='')
plt.axhline(0, color='r', linestyle='--')
plt.title('L2 - R1')
# plt.ylim([-0.05,0.3])
plt.subplot(323)
plt.plot(opto_2L_delay[cells_sorted_l2_r1] - opto_1L_delay[cells_sorted_l2_r1], marker='.', linestyle='')
plt.axhline(0, color='r', linestyle='--')
plt.title('L2 - L1')
plt.subplot(325)
plt.plot(opto_1R_delay[cells_sorted_l2_r1] - opto_1L_delay[cells_sorted_l2_r1], marker='.', linestyle='')
plt.plot(groups, C1_sel_incongruent, c='magenta')
plt.axhline(0, color='r', linestyle='--')
plt.title('R1 - L1')


plt.subplot(322)
plt.plot(opto_2R_delay[cells_sorted_l2_r1] - opto_1R_delay[cells_sorted_l2_r1], marker='.', linestyle='')
plt.axhline(0, color='r', linestyle='--')
plt.title('R2 - R1')
# plt.ylim([-0.05,0.3])
plt.subplot(324)
plt.plot(opto_2R_delay[cells_sorted_l2_r1] - opto_1L_delay[cells_sorted_l2_r1], marker='.', linestyle='')
plt.axhline(0, color='r', linestyle='--')
plt.title('R2 - L1')
plt.subplot(326)
plt.plot(opto_2R_delay[cells_sorted_l2_r1] - opto_2L_delay[cells_sorted_l2_r1], marker='.', linestyle='')
plt.plot(groups, C1_sel_congruent, c='magenta')
plt.axhline(0, color='r', linestyle='--')
plt.title('R2 - L2')

plt.tight_layout()


#%%

plt.figure(figsize=(8,10))
plt.subplot(321)
plt.plot(opto_2L_delay[cells_sorted_l2_l1] - opto_1R_delay[cells_sorted_l2_l1], marker='.', linestyle='')
plt.axhline(0, color='r', linestyle='--')
plt.title('L2 - R1')
# plt.ylim([-0.05,0.3])
plt.subplot(323)
plt.plot(opto_2L_delay[cells_sorted_l2_l1] - opto_1L_delay[cells_sorted_l2_l1], marker='.', linestyle='')
plt.axhline(0, color='r', linestyle='--')
plt.title('L2 - L1')
plt.subplot(325)
plt.plot(opto_1R_delay[cells_sorted_l2_l1] - opto_1L_delay[cells_sorted_l2_l1], marker='.', linestyle='')
# plt.plot(groups, C1_sel_incongruent, c='magenta')
plt.axhline(0, color='r', linestyle='--')
plt.title('R1 - L1')


plt.subplot(322)
plt.plot(opto_2R_delay[cells_sorted_l2_l1] - opto_1R_delay[cells_sorted_l2_l1], marker='.', linestyle='')
plt.axhline(0, color='r', linestyle='--')
plt.title('R2 - R1')
# plt.ylim([-0.05,0.3])
plt.subplot(324)
plt.plot(opto_2R_delay[cells_sorted_l2_l1] - opto_1L_delay[cells_sorted_l2_l1], marker='.', linestyle='')
plt.axhline(0, color='r', linestyle='--')
plt.title('R2 - L1')
plt.subplot(326)
plt.plot(opto_2R_delay[cells_sorted_l2_l1] - opto_2L_delay[cells_sorted_l2_l1], marker='.', linestyle='')
# plt.plot(groups, C1_sel_congruent, c='magenta')
plt.axhline(0, color='r', linestyle='--')
plt.title('R2 - L2')

plt.tight_layout()


#%%
for ix, frac in enumerate(frac_removed):
    frac = 0.95
    ncells_deleted = np.floor(ncells * frac).astype(int)
    
    cells_for_decoding_1 = cells_sorted_by_CD1[ncells_deleted:]
    cells_for_decoding_2 = cells_sorted_by_CD2[ncells_deleted:]

    CD1_population_trim = CD1_population[cells_for_decoding_1]
    CD2_population_trim = CD2_population[cells_for_decoding_2]
    opto_1R_testtrials_trim = opto_1R_testtrials[cells_for_decoding_1]
    opto_1L_testtrials_trim = opto_1L_testtrials[cells_for_decoding_1]
    opto_2R_testtrials_trim = opto_2R_testtrials[cells_for_decoding_2]
    opto_2L_testtrials_trim = opto_2L_testtrials[cells_for_decoding_2]

    proj_1R = np.tensordot(CD1_population_trim, opto_1R_testtrials_trim, axes=(0,0))
    proj_1L = np.tensordot(CD1_population_trim, opto_1L_testtrials_trim, axes=(0,0))
    proj_2R = np.tensordot(CD2_population_trim, opto_2R_testtrials_trim, axes=(0,0))
    proj_2L = np.tensordot(CD2_population_trim, opto_2L_testtrials_trim, axes=(0,0))

    ntrial_1R, ntrial_1L = proj_1R.shape[1], proj_1L.shape[1]
    ntrial_2R, ntrial_2L = proj_2R.shape[1], proj_2L.shape[1]

    ntrial_1R_correct = np.sum(np.mean(proj_1R[tix_response-4:tix_response,:],axis=0) > 0)
    ntrial_1L_correct = np.sum(np.mean(proj_1L[tix_response-4:tix_response,:],axis=0) < 0)
    ntrial_2R_correct = np.sum(np.mean(proj_2R[tix_response-4:tix_response,:],axis=0) > 0)
    ntrial_2L_correct = np.sum(np.mean(proj_2L[tix_response-4:tix_response,:],axis=0) < 0)

    frac_correct_1[ix] = (ntrial_1R_correct + ntrial_1L_correct) / (ntrial_1R + ntrial_1L)
    frac_correct_2[ix] = (ntrial_2R_correct + ntrial_2L_correct) / (ntrial_2R + ntrial_2L)


#%%
plt.figure()
plt.plot(frac_removed, frac_correct_1, marker='o')
plt.plot(frac_removed, frac_correct_2, marker='o')
plt.tight_layout()


plt.figure()
for triali in range(ntrial_2R):
    plt.plot(tvec, proj_2R[:,triali], c='b', alpha=0.3)
for triali in range(ntrial_2L):
    plt.plot(tvec, proj_2L[:,triali], c='r', alpha=0.3)
plt.axvline(tvec[tix_sample], color='gray', linestyle='--')
plt.axvline(tvec[tix_delay], color='gray', linestyle='--')
plt.axvline(tvec[tix_response], color='gray', linestyle='--')
plt.tight_layout()

    
    
    
# # %%
# plt.figure()
# plt.hist(l2_l1, bins=50, histtype='step')
# plt.hist(l2_r1, bins=50, histtype='step')
# plt.hist(C1_selectivity, bins=50, histtype='step')



# plt.figure()
# plt.hist(C1_selectivity[pos_cells], histtype='step', bins=20)



# plt.figure(figsize=(8,5))
# plt.subplot(211)
# L2_cells_sorted       = np.argsort(l2_r1)        
# L2_congruent_sorted   = l2_l1[L2_cells_sorted]
# L2_incongruent_sorted = l2_r1[L2_cells_sorted]
# plt.plot(L2_incongruent_sorted, marker='.', linestyle='', label='L2-R1', c='purple')
# plt.plot(L2_congruent_sorted, marker='.', linestyle='', label='L2-L1', c='limegreen')
# plt.legend()

# plt.subplot(212)
# L2_cells_sorted       = np.argsort(l2_l1)        
# L2_congruent_sorted   = l2_l1[L2_cells_sorted]
# L2_incongruent_sorted = l2_r1[L2_cells_sorted]
# plt.plot(L2_congruent_sorted, marker='.', linestyle='', label='L2-L1', c='limegreen')
# plt.plot(L2_incongruent_sorted, marker='.', linestyle='', label='L2-R1', c='purple')
# plt.legend()
# plt.tight_layout()
# # plt.savefig('figure/temp/relearn_' + str(fx) + '_compare_P1_A1_' + merged_ID[fx] + '.png',dpi=300)
# # plt.close()
        
# # plt.subplot(212)
# # A2_cells_sorted = np.argsort(A2_incongruent)        
# # A2_congruent_sorted = A2_congruent[A2_cells_sorted]
# # A2_incongruent_sorted = A2_incongruent[A2_cells_sorted]
# # plt.plot(A2_incongruent_sorted, marker='.', linestyle='', label='A2-A1', c='limegreen')
# # plt.plot(A2_congruent_sorted, marker='.', linestyle='', label='A2-P1', c='purple')
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig('figure/temp/relearn_' + str(fx) + '_compare_P1_A1_' + merged_ID[fx] + '.png',dpi=300)
# # plt.close()
        
# x=1

#         # end of mouse loop
#     # end of shuffle loop
            
# x=1
   
   
