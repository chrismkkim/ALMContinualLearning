import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import functions
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


dirpath = '/Users/kimchm/Documents/KimNiNature2024/MatConverted2Python/' 
metadata = pd.read_csv(dirpath + 'EDF10d_info_2026_01_16.csv')
CDdotprod_python = pd.read_csv('CDdotprod_python_deconvolved.txt')


# Compare CD dot products of CK and JH
metadata['Merged_ID'] = (
    metadata['Mouse ID'].str.strip("'") + '_' + 
    metadata['FOV'].str.strip("'") + '_' + 
    metadata['Session1'].str.strip("'").str.replace("-", "_") + '_' + 
    metadata['Session2'].str.strip("'").str.replace("-", "_")
)
cols = ['Merged_ID'] + [c for c in metadata.columns if c != 'Merged_ID']
metadata = metadata[cols]
metadata = metadata.sort_values(by='Merged_ID')
metadata = metadata.reset_index(drop=True)

filtered_CDdotprod_python = CDdotprod_python[CDdotprod_python['ID'].isin(metadata['Merged_ID'])].reset_index(drop=True)

is_aligned = metadata['Merged_ID'].equals(filtered_CDdotprod_python['ID'])
print('is aligned: ', is_aligned)
    
# sort by relearns speed
colname_relearn_speed = 'Relative trials to reach\n75% performance'
relearn_speed = metadata[colname_relearn_speed]
sorted_indices_by_relearn_speed = np.argsort(relearn_speed)
relearn_speed_sorted = np.zeros_like(relearn_speed)

# sort by CD dot product
colname_CDdotproduct = 'CDdotproduct'
CDdotproduct = filtered_CDdotprod_python[colname_CDdotproduct]
sorted_indices_by_CDdotproduct = np.argsort(CDdotproduct)
# filtered_CDdotprod_sorted = filtered_CDdotprod_python.sort_values(by='CDdotproduct').reset_index(drop=True)

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

nfile = len(filtered_CDdotprod_python)
CD_dotproduct_all = np.zeros(nfile)
P_dotproduct_all  = np.zeros(nfile)
A_dotproduct_all  = np.zeros(nfile)
P_distance_all  = np.zeros(nfile)
A_distance_all  = np.zeros(nfile)
for ix in range(nfile):
    # ix = 0
    print(ix)

    #--- sorted by relearn speed ---#
    fidx     = sorted_indices_by_relearn_speed[ix]    
    filename = filtered_CDdotprod_python['ID'][fidx]

    # #--- sorted by CD dot product ---#
    # fidx = sorted_indices_by_CDdotproduct[ix]
    # filename = filtered_CDdotprod_python['ID'][fidx]

    filepath = dirpath + filename + '.npy'
    data = np.load(filepath, allow_pickle=True)
    opto_sess1 = data['deconvolved'][0]
    opto_sess2 = data['deconvolved'][1]
    ncell      = opto_sess1.shape[1]

    # compute coding direction delay
    data_type = 'deconvolved'
    CD_dotproduct, CD_sess1_population, CD_sess2_population, \
    proj_1R, proj_1L, proj_2R, proj_2L,\
    decision_1R, decision_1L, decision_2R, decision_2L = functions.compute_CD_delay(data, data_type, 'input')

    # compute similarity between posterior (or anterior)
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

    ntrials_1P = opto_sess1[lickright,0].shape[1]
    ntrials_1A = opto_sess1[lickleft,0].shape[1]
    ntrials_2P = opto_sess2[lickleft,0].shape[1]
    ntrials_2A = opto_sess2[lickright,0].shape[1]

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
    opto_1P = np.zeros((ncell,ntimestep))
    opto_1A = np.zeros((ncell,ntimestep))
    opto_2P = np.zeros((ncell,ntimestep))
    opto_2A = np.zeros((ncell,ntimestep))    
    trials_1P, trials_1A, trials_2P, trials_2A = np.arange(ntrials_1P), np.arange(ntrials_1A), np.arange(ntrials_2P), np.arange(ntrials_2A)
    train_trials_1P = np.sort(np.random.permutation(trials_1P)[:ntrials_train_1P])
    train_trials_1A = np.sort(np.random.permutation(trials_1A)[:ntrials_train_1A])
    train_trials_2P = np.sort(np.random.permutation(trials_2P)[:ntrials_train_2P])
    train_trials_2A = np.sort(np.random.permutation(trials_2A)[:ntrials_train_2A])    
    
    for cell in range(ncell):        
        #------- first half trials ----------#
        opto_1P[cell] = np.mean(opto_1P_alltrials[cell,:,train_trials_1P].T,axis=1)
        opto_1A[cell] = np.mean(opto_1A_alltrials[cell,:,train_trials_1A].T,axis=1)
        opto_2P[cell] = np.mean(opto_2P_alltrials[cell,:,train_trials_2P].T,axis=1)
        opto_2A[cell] = np.mean(opto_2A_alltrials[cell,:,train_trials_2A].T,axis=1)    
        
    opto_1P_delay = np.mean(opto_1P[:,tix_response-4:tix_response],axis=1)
    opto_1A_delay = np.mean(opto_1A[:,tix_response-4:tix_response],axis=1)
    opto_2P_delay = np.mean(opto_2P[:,tix_response-4:tix_response],axis=1)
    opto_2A_delay = np.mean(opto_2A[:,tix_response-4:tix_response],axis=1)
        
    P_dotproduct = np.inner(opto_1P_delay, opto_2P_delay) / (np.linalg.norm(opto_1P_delay) * np.linalg.norm(opto_2P_delay))
    A_dotproduct = np.inner(opto_1A_delay, opto_2A_delay) / (np.linalg.norm(opto_1A_delay) * np.linalg.norm(opto_2A_delay))    
    P_distance   = np.mean((opto_1P_delay - opto_2P_delay)**2) / (np.var(opto_1P_delay) + np.var(opto_2P_delay))
    A_distance   = np.mean((opto_1A_delay - opto_2A_delay)**2) / (np.var(opto_1A_delay) + np.var(opto_2A_delay))

    # save population vectors
    CD_dotproduct_all[ix] = CD_dotproduct
    P_dotproduct_all[ix]  = P_dotproduct
    A_dotproduct_all[ix]  = A_dotproduct
    P_distance_all[ix]    = P_distance
    A_distance_all[ix]    = A_distance

    # save relearn speed sorted
    relearn_speed_sorted[ix] = relearn_speed[fidx]


x = 1

P_corr_dotprod = np.corrcoef(CD_dotproduct_all,P_dotproduct_all)[0,1]
A_corr_dotprod = np.corrcoef(CD_dotproduct_all,A_dotproduct_all)[0,1]
P_corr_distance = np.corrcoef(CD_dotproduct_all,P_distance_all)[0,1]
A_corr_distance = np.corrcoef(CD_dotproduct_all,A_distance_all)[0,1]
P_corr_distance_relearn = np.corrcoef(relearn_speed_sorted,P_distance_all)[0,1]
A_corr_distance_relearn = np.corrcoef(relearn_speed_sorted,A_distance_all)[0,1]
P_corr_dotprod_relearn = np.corrcoef(relearn_speed_sorted,P_dotproduct_all)[0,1]
A_corr_dotprod_relearn = np.corrcoef(relearn_speed_sorted,A_dotproduct_all)[0,1]
corr_CD_relearn = np.corrcoef(relearn_speed_sorted,CD_dotproduct_all)[0,1]


plt.figure(figsize=(3,3))
# plt.subplot(121)
plt.plot(CD_dotproduct_all, relearn_speed_sorted, marker='o', c='k', linestyle='')
# plt.plot(xvec, P_slope*xvec+P_intercept, c='gray', linestyle='--')
plt.xlabel('CD dot product')
plt.ylabel('Learning time')
# plt.title('slope ' + str(np.round(P_slope[0],decimals=2)) + ',  R2 ' + str(np.round(P_r2,decimals=2)))
plt.title('correlation ' + str(np.round(corr_CD_relearn,decimals=2)))
plt.tight_layout()
plt.savefig('figure/learn_time/CDdotprod_learnTime.pdf')



plt.figure(figsize=(6,3))
plt.subplot(121)
plt.plot(CD_dotproduct_all, P_dotproduct_all, marker='o', c='purple', linestyle='')
# plt.plot(xvec, P_slope*xvec+P_intercept, c='gray', linestyle='--')
plt.xlabel('CD dot product')
plt.ylabel('Posterior dot product')
# plt.title('slope ' + str(np.round(P_slope[0],decimals=2)) + ',  R2 ' + str(np.round(P_r2,decimals=2)))
plt.title('correlation ' + str(np.round(P_corr_dotprod,decimals=2)))
# plt.xlim([-0.5,1])
plt.ylim([0,0.9])
plt.subplot(122)
plt.plot(CD_dotproduct_all, A_dotproduct_all, marker='o', c='limegreen', linestyle='')
# plt.plot(xvec, A_slope*xvec+A_intercept, c='gray', linestyle='--')
plt.xlabel('CD dot product')
plt.ylabel('Anterior dot product')
# plt.title('slope ' + str(np.round(A_slope[0],decimals=2)) + ',  R2 ' + str(np.round(A_r2,decimals=2)))
plt.title('correlation ' + str(np.round(A_corr_dotprod,decimals=2)))
# plt.xlim([-0.5,1])
plt.ylim([0,0.9])
plt.tight_layout()
plt.savefig('figure/learn_time/CDdotprod_PAdotprod.pdf')



plt.figure(figsize=(6,3))
plt.subplot(121)
plt.plot(CD_dotproduct_all, P_distance_all, marker='o', c='purple', linestyle='')
# plt.plot(xvec, P_slope*xvec+P_intercept, c='gray', linestyle='--')
plt.xlabel('CD dot product')
plt.ylabel('Posterior distance')
# plt.title('slope ' + str(np.round(P_slope[0],decimals=2)) + ',  R2 ' + str(np.round(P_r2,decimals=2)))
plt.title('correlation ' + str(np.round(P_corr_distance,decimals=2)))
# plt.xlim([-0.5,1])
# plt.ylim([0,0.1])
plt.subplot(122)
plt.plot(CD_dotproduct_all, A_distance_all, marker='o', c='limegreen', linestyle='')
# plt.plot(xvec, A_slope*xvec+A_intercept, c='gray', linestyle='--')
plt.xlabel('CD dot product')
plt.ylabel('Anterior distance')
# plt.title('slope ' + str(np.round(A_slope[0],decimals=2)) + ',  R2 ' + str(np.round(A_r2,decimals=2)))
plt.title('correlation ' + str(np.round(A_corr_distance,decimals=2)))
# plt.xlim([-0.5,1])
# plt.ylim([0,0.1])
plt.tight_layout()
plt.savefig('figure/learn_time/CDdotprod_PAdistance.pdf')


plt.figure(figsize=(6,3))
plt.subplot(121)
plt.plot(P_distance_all, relearn_speed_sorted, marker='o', c='purple', linestyle='')
# plt.plot(P_distance_avg, relearn_unique, marker='o', c='purple', linestyle='')
plt.xlabel('Posterior distance')
plt.ylabel('Learning time')
plt.title('correlation ' + str(np.round(P_corr_distance_relearn,decimals=2)))
plt.subplot(122)
plt.plot(A_distance_all, relearn_speed_sorted, marker='o', c='limegreen', linestyle='')
# plt.plot(A_distance_avg, relearn_unique, marker='o', c='limegreen', linestyle='')
plt.xlabel('Anterior distance')
plt.ylabel('Learning time')
plt.title('correlation ' + str(np.round(A_corr_distance_relearn,decimals=2)))
plt.tight_layout()
plt.savefig('figure/learn_time/PAdistance_learnTime.pdf')



plt.figure(figsize=(6,3))
plt.subplot(121)
plt.plot(P_dotproduct_all, relearn_speed_sorted, marker='o', c='purple', linestyle='')
# plt.plot(xvec, P_relearn_slope*xvec+P_relearn_intercept, c='gray', linestyle='--')
plt.xlabel('Posterior dot product')
plt.ylabel('Learning time')
# plt.title('slope ' + str(np.round(P_slope[0],decimals=2)) + ',  R2 ' + str(np.round(P_r2,decimals=2)))
plt.title('correlation ' + str(np.round(P_corr_dotprod_relearn,decimals=2)))
# plt.xlim([-0.5,1])
# plt.ylim([0,0.9])
plt.subplot(122)
plt.plot(A_dotproduct_all, relearn_speed_sorted, marker='o', c='limegreen', linestyle='')
# plt.plot(xvec, A_relearn_slope*xvec+A_relearn_intercept, c='gray', linestyle='--')
plt.xlabel('Anterior dot product')
plt.ylabel('Learning time')
# plt.title('slope ' + str(np.round(A_slope[0],decimals=2)) + ',  R2 ' + str(np.round(A_r2,decimals=2)))
plt.title('correlation ' + str(np.round(A_corr_dotprod_relearn,decimals=2)))
# plt.xlim([-0.5,1])
# plt.ylim([0,0.9])
plt.tight_layout()
plt.savefig('figure/learn_time/PAdotprod_learnTime.pdf')





# linreg_P = LinearRegression()
# linreg_A = LinearRegression()
# linreg_P.fit(CD_dotproduct_all.reshape(-1,1),P_dotproduct_all)
# linreg_A.fit(CD_dotproduct_all.reshape(-1,1),A_dotproduct_all)
# P_slope, P_intercept = linreg_P.coef_, linreg_P.intercept_
# A_slope, A_intercept = linreg_A.coef_, linreg_A.intercept_
# P_r2 = linreg_P.score(CD_dotproduct_all.reshape(-1,1),P_dotproduct_all)
# A_r2 = linreg_A.score(CD_dotproduct_all.reshape(-1,1),A_dotproduct_all)
# xvec = np.linspace(-0.5,1.0,num=10)
# print(P_r2)
# print(A_r2)


# linreg_P_relearn = LinearRegression()
# linreg_A_relearn = LinearRegression()
# linreg_P_relearn.fit(P_dotproduct_all.reshape(-1,1), relearn_speed_sorted)
# linreg_A_relearn.fit(A_dotproduct_all.reshape(-1,1), relearn_speed_sorted)
# P_relearn_slope, P_relearn_intercept = linreg_P_relearn.coef_, linreg_P_relearn.intercept_
# A_relearn_slope, A_relearn_intercept = linreg_A_relearn.coef_, linreg_A_relearn.intercept_
# P_relearn_r2 = linreg_P_relearn.score(P_dotproduct_all.reshape(-1,1),relearn_speed_sorted)
# A_relearn_r2 = linreg_A_relearn.score(A_dotproduct_all.reshape(-1,1),relearn_speed_sorted)
# xvec = np.linspace(0,1.0,num=10)
# print(P_relearn_r2)
# print(A_relearn_r2)





# model_P = sm.OLS(P_dotproduct_all,sm.add_constant(CD_dotproduct_all))
# results_P = model_P.fit()
# print(results_P.summary())

# model_A = sm.OLS(A_dotproduct_all,sm.add_constant(CD_dotproduct_all))
# results_A = model_A.fit()
# print(results_A.summary())