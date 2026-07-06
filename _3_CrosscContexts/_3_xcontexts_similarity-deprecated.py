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
for shuff in range(nshuff):
    
    print(str(shuff) + ' / ' + str(nshuff))
    
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
        opto_1P_testtrials = opto_1R_testtrials
        opto_1A_testtrials = opto_1L_testtrials
        opto_2P_testtrials = opto_2L_testtrials
        opto_2A_testtrials = opto_2R_testtrials                

        opto_1P_testtrials_avg = np.mean(opto_1P_testtrials,axis=2) # neurons x time
        opto_1A_testtrials_avg = np.mean(opto_1A_testtrials,axis=2) # neurons x time
        opto_2P_testtrials_avg = np.mean(opto_2P_testtrials,axis=2) # neurons x time
        opto_2A_testtrials_avg = np.mean(opto_2A_testtrials,axis=2) # neurons x time        
        
        # similarity of 2P vs. 1P and 1A
        opto_1P_norm_train = opto_1P_traintrials_avg / np.linalg.norm(opto_1P_traintrials_avg,axis=0).reshape(1,-1)
        opto_1P_norm = opto_1P_testtrials_avg / np.linalg.norm(opto_1P_testtrials_avg,axis=0).reshape(1,-1)
        opto_1A_norm = opto_1A_testtrials_avg / np.linalg.norm(opto_1A_testtrials_avg,axis=0).reshape(1,-1)
        opto_2P_norm = opto_2P_testtrials_avg / np.linalg.norm(opto_2P_testtrials_avg,axis=0).reshape(1,-1)
        opto_2A_norm = opto_2A_testtrials_avg / np.linalg.norm(opto_2A_testtrials_avg,axis=0).reshape(1,-1)
        
        hadamard_1P_2P                      = opto_1P_norm * opto_2P_norm
        hadamard_1A_2P                      = opto_1A_norm * opto_2P_norm
        hadamard_1P_1P                      = opto_1P_norm * opto_1P_norm_train
        hadamard_1P_1A                      = opto_1P_norm * opto_1A_norm
        hadamard_xcont_2P_all[shuff,fx,0]   = hadamard_1P_2P
        hadamard_xcont_2P_all[shuff,fx,1]   = hadamard_1A_2P        
        similarity_2P_1P                    = np.sum(hadamard_1P_2P, axis=0)
        similarity_2P_1A                    = np.sum(hadamard_1A_2P, axis=0)
        similarity_1P_1P                    = np.sum(hadamard_1P_1P, axis=0)
        similarity_1P_1A                    = np.sum(hadamard_1P_1A, axis=0)
        similarity_xcont_2P_all[shuff,fx,0] = similarity_2P_1P
        similarity_xcont_2P_all[shuff,fx,1] = similarity_2P_1A
        similarity_xcont_1P_all[shuff,fx,0] = similarity_1P_1P
        similarity_xcont_1P_all[shuff,fx,1] = similarity_1P_1A


        P2_incongruent = (opto_2P_testtrials_avg - opto_1P_testtrials_avg)[:,tix_response-2]
        P2_congruent   = (opto_2P_testtrials_avg - opto_1A_testtrials_avg)[:,tix_response-2]
        A2_incongruent = (opto_2A_testtrials_avg - opto_1A_testtrials_avg)[:,tix_response-2]
        A2_congruent   = (opto_2A_testtrials_avg - opto_1P_testtrials_avg)[:,tix_response-2]

        x = 1
        # plt.figure(figsize=(8,5))
        # plt.subplot(211)
        # P2_cells_sorted = np.argsort(P2_incongruent)        
        # P2_congruent_sorted = P2_congruent[P2_cells_sorted]
        # P2_incongruent_sorted = P2_incongruent[P2_cells_sorted]
        # plt.plot(P2_incongruent_sorted, marker='.', linestyle='', label='P2-P1', c='purple')
        # plt.plot(P2_congruent_sorted, marker='.', linestyle='', label='P2-A1', c='limegreen')
        # plt.legend()

        # plt.subplot(212)
        # P2_cells_sorted = np.argsort(P2_congruent)        
        # P2_congruent_sorted = P2_congruent[P2_cells_sorted]
        # P2_incongruent_sorted = P2_incongruent[P2_cells_sorted]
        # plt.plot(P2_congruent_sorted, marker='.', linestyle='', label='P2-A1', c='limegreen')
        # plt.plot(P2_incongruent_sorted, marker='.', linestyle='', label='P2-P1', c='purple')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig('figure/temp/relearn_' + str(fx) + '_compare_P1_A1_' + merged_ID[fx] + '.png',dpi=300)
        # plt.close()
                
        # plt.subplot(212)
        # A2_cells_sorted = np.argsort(A2_incongruent)        
        # A2_congruent_sorted = A2_congruent[A2_cells_sorted]
        # A2_incongruent_sorted = A2_incongruent[A2_cells_sorted]
        # plt.plot(A2_incongruent_sorted, marker='.', linestyle='', label='A2-A1', c='limegreen')
        # plt.plot(A2_congruent_sorted, marker='.', linestyle='', label='A2-P1', c='purple')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig('figure/temp/relearn_' + str(fx) + '_compare_P1_A1_' + merged_ID[fx] + '.png',dpi=300)
        # plt.close()
        
    x=1

        # end of mouse loop
    # end of shuffle loop
            
x=1
   
   






CD_dotproduct_avg = np.mean(CD_dotproduct_all,axis=0)
similarity_xcont_2P_avg = np.mean(similarity_xcont_2P_all,axis=0)
similarity_xcont_1P_avg = np.mean(similarity_xcont_1P_all,axis=0)
        
plt.figure()
plt.subplot(211)
for fx in range(nfile):
    plt.plot(tvec,similarity_xcont_2P_avg[fx,0], alpha=0.5)
# plt.plot(tvec,np.mean(similarity_xcont_2P_avg[:,0,:],axis=0),c='k',lw=3.5)            
plt.plot(tvec,np.mean(similarity_xcont_2P_avg[:,0,:],axis=0),c='purple',lw=3)        
plt.plot(tvec,np.mean(similarity_xcont_1P_avg[:,0,:],axis=0),c='b',lw=2)        
plt.axvline(tvec[tix_sample],color='r', linestyle='--')
plt.axvline(tvec[tix_delay],color='r', linestyle='--')
plt.axvline(tvec[tix_response],color='r', linestyle='--')
plt.title('Similarity of P2 vs. P1')
plt.subplot(212)
for fx in range(nfile):
    plt.plot(tvec,similarity_xcont_2P_avg[fx,1], alpha=0.5)
plt.plot(tvec,np.mean(similarity_xcont_2P_avg[:,1,:],axis=0),c='k',lw=3.5)   
plt.plot(tvec,np.mean(similarity_xcont_2P_avg[:,1,:],axis=0),c='limegreen',lw=2)        
plt.plot(tvec,np.mean(similarity_xcont_1P_avg[:,1,:],axis=0),c='r',lw=2)        
plt.axvline(tvec[tix_sample],color='r', linestyle='--')
plt.axvline(tvec[tix_delay],color='r', linestyle='--')
plt.axvline(tvec[tix_response],color='r', linestyle='--')
plt.title('Similarity of P2 vs. A1')
plt.xlabel('time (s)')
plt.tight_layout()
plt.savefig('figure/temp/proj_xcontext_similarity.png')



corr_time_P1 = np.zeros(ntimestep-2)
corr_time_A1 = np.zeros(ntimestep-2)
corr_time_P1_P1 = np.zeros(ntimestep-2)
corr_time_P1_A1 = np.zeros(ntimestep-2)
for tix in range(ntimestep-2):
    corr_time_P1[tix] = np.corrcoef(CD_dotproduct_avg, similarity_xcont_2P_avg[:,0,tix])[0,1]
    corr_time_A1[tix] = np.corrcoef(CD_dotproduct_avg, similarity_xcont_2P_avg[:,1,tix])[0,1]
    corr_time_P1_P1[tix] = np.corrcoef(CD_dotproduct_avg, similarity_xcont_1P_avg[:,0,tix])[0,1]
    corr_time_P1_A1[tix] = np.corrcoef(CD_dotproduct_avg, similarity_xcont_1P_avg[:,1,tix])[0,1]
    


plt.figure(figsize=(8,8))
for ti, tix in enumerate(np.arange(tix_sample,tix_sample+16)):
    plt.subplot(4,4,ti+1)
    corr0 = np.corrcoef(CD_dotproduct_avg, similarity_xcont_2P_avg[:,0,tix])[0,1]
    plt.scatter(CD_dotproduct_avg, similarity_xcont_2P_avg[:,0,tix],color='purple')
    plt.title('t=' + str(np.round(tvec[tix],decimals=2)) + ', ' + str(np.round(corr0,decimals=3)))
    plt.ylim([0,1])
plt.tight_layout()
plt.savefig('figure/temp/proj_xcontext_similarity_corrP2P1.png')


plt.figure(figsize=(8,8))
for ti, tix in enumerate(np.arange(tix_sample,tix_sample+16)):
    plt.subplot(4,4,ti+1)
    corr1 = np.corrcoef(CD_dotproduct_avg, similarity_xcont_2P_avg[:,1,tix])[0,1]
    plt.scatter(CD_dotproduct_avg, similarity_xcont_2P_avg[:,1,tix],color='limegreen')
    plt.title('t=' + str(np.round(tvec[tix],decimals=2)) + ', ' + str(np.round(corr1,decimals=3)))
    plt.ylim([0,1])
plt.tight_layout()
plt.savefig('figure/temp/proj_xcontext_similarity_corrP2A1.png')



plt.figure(figsize=(8,8))
for ti, tix in enumerate(np.arange(tix_sample,tix_sample+16)):
    plt.subplot(4,4,ti+1)
    corr1 = np.corrcoef(CD_dotproduct_avg, similarity_xcont_1P_avg[:,0,tix])[0,1]
    plt.scatter(CD_dotproduct_avg, similarity_xcont_1P_avg[:,0,tix],color='b')
    plt.title('t=' + str(np.round(tvec[tix],decimals=2)) + ', ' + str(np.round(corr1,decimals=3)))
    plt.ylim([0,1.1])
plt.tight_layout()
plt.savefig('figure/temp/proj_xcontext_similarity_corrP1P1.png')


plt.figure(figsize=(8,8))
for ti, tix in enumerate(np.arange(tix_sample,tix_sample+16)):
    plt.subplot(4,4,ti+1)
    corr1 = np.corrcoef(CD_dotproduct_avg, similarity_xcont_1P_avg[:,1,tix])[0,1]
    plt.scatter(CD_dotproduct_avg, similarity_xcont_1P_avg[:,1,tix],color='r')
    plt.title('t=' + str(np.round(tvec[tix],decimals=2)) + ', ' + str(np.round(corr1,decimals=3)))
    plt.ylim([0,1])
plt.tight_layout()
plt.savefig('figure/temp/proj_xcontext_similarity_corrP1A1.png')



plt.figure(figsize=(6,6))
plt.subplot(211)
plt.plot(tvec[:-2],corr_time_P1, c='purple', label='P2-P1', marker='.')
plt.plot(tvec[:-2],corr_time_A1, c='limegreen', label='P2-A1', marker='.')
plt.axvline(tvec[tix_sample],color='gray', linestyle='--')
plt.axvline(tvec[tix_delay],color='gray', linestyle='--')
plt.axvline(tvec[tix_response],color='gray', linestyle='--')
plt.axhline(0, color='gray', alpha=0.5)
plt.legend()
plt.subplot(212)
plt.plot(tvec[:-2],corr_time_P1_P1, c='b', label='P1-P1', marker='.')
plt.plot(tvec[:-2],corr_time_P1_A1, c='r', label='P1-A1', marker='.')
plt.axvline(tvec[tix_sample],color='gray', linestyle='--')
plt.axvline(tvec[tix_delay],color='gray', linestyle='--')
plt.axvline(tvec[tix_response],color='gray', linestyle='--')
plt.axhline(0, color='gray', alpha=0.5)
plt.xlabel('time (s)')
plt.ylabel('Correlation to CD dot product')
plt.legend()
plt.savefig('figure/temp/proj_xcontext_similarity_corr.png')



file_sorted = np.argsort(CD_dotproduct_avg)

shuff = 0
tix   = tix_sample + 2

plt.figure(figsize=(15,10))
for fx in range(nfile):
    plt.subplot(6,6,fx+1)
    hadamard_P1 = hadamard_xcont_2P_all[shuff,file_sorted[fx],0][:,tix]
    hadamard_A1 = hadamard_xcont_2P_all[shuff,file_sorted[fx],1][:,tix]
    plt.plot(hadamard_P1,c='purple')
    plt.plot(hadamard_A1,c='limegreen')
    plt.title('CD dotprod ' + str(np.round(CD_dotproduct_avg[file_sorted[fx]],decimals=3)))
    plt.ylim([0,0.2])    
plt.tight_layout()


plt.figure()
plt.imshow(hadamard_xcont_2P_all[shuff,file_sorted[0],0],cmap='jet', vmin=0, vmax=0.01, aspect='auto')
plt.axvline(tix_sample, color='w')
plt.axvline(tix_delay, color='w')
plt.axvline(tix_response, color='w')
plt.colorbar()
plt.tight_layout()



overlapcells    = np.empty((nshuff,nfile,ntimestep),dtype=object)
frac_overlap    = np.zeros((nshuff,nfile,ntimestep))
frac_overlap_P1 = np.zeros((nshuff,nfile,ntimestep))
frac_overlap_A1 = np.zeros((nshuff,nfile,ntimestep))
topcells_P1     = np.zeros((nshuff,nfile,ntimestep))
topcells_A1     = np.zeros((nshuff,nfile,ntimestep))
frac_similarity_overlap_P1 = np.zeros((nshuff,nfile,ntimestep))
frac_similarity_overlap_A1 = np.zeros((nshuff,nfile,ntimestep))

for shuff in range(nshuff):
    for fx in range(nfile):
        ncell = len(hadamard_xcont_2P_all[0,file_sorted[fx],0][:,0])
        for tix in range(ntimestep):
            _hadamard_P1        = hadamard_xcont_2P_all[shuff,file_sorted[fx],0][:,tix]
            _hadamard_A1        = hadamard_xcont_2P_all[shuff,file_sorted[fx],1][:,tix]
            _P1_sorted         = np.argsort(_hadamard_P1)[::-1]
            _A1_sorted         = np.argsort(_hadamard_A1)[::-1]
            hadamard_P1_cumsum = np.cumsum(_hadamard_P1[_P1_sorted]) / np.sum(_hadamard_P1)
            hadamard_A1_cumsum = np.cumsum(_hadamard_A1[_A1_sorted]) / np.sum(_hadamard_A1)
            
            _topcells_P1       = _P1_sorted[:np.where(hadamard_P1_cumsum > 0.8)[0][0]+1]
            _topcells_A1       = _A1_sorted[:np.where(hadamard_A1_cumsum > 0.8)[0][0]+1]
            _overlapcells       = _topcells_P1[np.isin(_topcells_P1, _topcells_A1)]            
            
            overlapcells[shuff,fx,tix]    = _overlapcells
            frac_overlap[shuff,fx,tix]    = len(_overlapcells) / ncell
            frac_overlap_P1[shuff,fx,tix] = len(_overlapcells) / len(_topcells_P1)
            frac_overlap_A1[shuff,fx,tix] = len(_overlapcells) / len(_topcells_A1)
            topcells_P1[shuff,fx,tix]     = len(_topcells_P1) / ncell
            topcells_A1[shuff,fx,tix]     = len(_topcells_A1) / ncell
            frac_similarity_overlap_P1[shuff,fx,tix] = np.sum(_hadamard_P1[_overlapcells]) / np.sum(_hadamard_P1)
            frac_similarity_overlap_A1[shuff,fx,tix] = np.sum(_hadamard_A1[_overlapcells]) / np.sum(_hadamard_A1)                        
        # assert ncell == _hadamard_A1.shape[0]


# plot the indices of shared cells
shuff = 0
fx = 11
plt.figure()
for tix in range(ntimestep):
    overlap_tix = overlapcells[shuff,fx,tix]
    _ncell = len(overlap_tix)
    plt.plot(tvec[tix]*np.ones(_ncell),overlap_tix,marker='o',linestyle='')
plt.axvline(tvec[tix_sample],color='gray', linestyle='--')
plt.axvline(tvec[tix_delay],color='gray', linestyle='--')
plt.axvline(tvec[tix_response],color='gray', linestyle='--')    
plt.tight_layout()




shuff = 0
plt.figure(figsize=(6,5))
plt.subplot(211)
for fx in range(nfile):
    # plt.subplot(6,6,fx+1)
    plt.plot(tvec, frac_overlap_P1[shuff,fx], label='P1-P2', alpha=0.3)
plt.plot(tvec, np.mean(frac_overlap_P1[shuff,:,:],axis=0), c='k', lw=3.5)
plt.plot(tvec, np.mean(frac_overlap_P1[shuff,:,:],axis=0), c='purple', label='P1-P2', lw=2)
plt.axvline(tvec[tix_sample],color='gray', linestyle='--')
plt.axvline(tvec[tix_delay],color='gray', linestyle='--')
plt.axvline(tvec[tix_response],color='gray', linestyle='--')    
plt.title('P2 vs. P1')
plt.ylim([0,1])

plt.subplot(212)
for fx in range(nfile):
    plt.plot(tvec, frac_overlap_A1[shuff,fx], label='A1-P2', alpha=0.3)
plt.plot(tvec, np.mean(frac_overlap_A1[shuff,:,:],axis=0), c='k', label='A1-P2', lw=3.5)    
plt.plot(tvec, np.mean(frac_overlap_A1[shuff,:,:],axis=0), c='limegreen', label='A1-P2', lw=2)
plt.axvline(tvec[tix_sample],color='gray', linestyle='--')
plt.axvline(tvec[tix_delay],color='gray', linestyle='--')
plt.axvline(tvec[tix_response],color='gray', linestyle='--')    
plt.title('P2 vs. A1')
plt.ylim([0,1])
plt.xlabel('time (s)')
plt.ylabel('frac cells shared btw P1/A1')
plt.tight_layout()
plt.savefig('figure/temp/frac_shared_cells.png')



shuff = 0
plt.figure(figsize=(6,5))
plt.subplot(211)
for fx in range(nfile):
    # plt.subplot(6,6,fx+1)
    plt.plot(tvec, topcells_P1[shuff,fx], alpha=0.3)
plt.plot(tvec, np.mean(topcells_P1[shuff,:,:],axis=0), c='k', lw=3.5)    
plt.plot(tvec, np.mean(topcells_P1[shuff,:,:],axis=0), c='purple', label='P2-P1', lw=2)
plt.plot(tvec, np.mean(frac_overlap[shuff,:,:],axis=0), c='k', lw=3.5)
plt.plot(tvec, np.mean(frac_overlap[shuff,:,:],axis=0), c='darkorange', label='shared',  lw=2)
plt.axvline(tvec[tix_sample],color='gray', linestyle='--')
plt.axvline(tvec[tix_delay],color='gray', linestyle='--')
plt.axvline(tvec[tix_response],color='gray', linestyle='--')    
plt.title('P2 vs. P1')
plt.legend()
plt.ylim([-0.05,0.3])

plt.subplot(212)
for fx in range(nfile):
    plt.plot(tvec, topcells_A1[shuff,fx], alpha=0.3)
plt.plot(tvec, np.mean(topcells_A1[shuff,:,:],axis=0), c='k', lw=3.5)    
plt.plot(tvec, np.mean(topcells_A1[shuff,:,:],axis=0), c='limegreen', label='P2-A1', lw=2)
plt.plot(tvec, np.mean(frac_overlap[shuff,:,:],axis=0), c='k', lw=3.5)
plt.plot(tvec, np.mean(frac_overlap[shuff,:,:],axis=0), c='darkorange', label='shared', lw=2)
plt.axvline(tvec[tix_sample],color='gray', linestyle='--')
plt.axvline(tvec[tix_delay],color='gray', linestyle='--')
plt.axvline(tvec[tix_response],color='gray', linestyle='--')    
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('frac cells to get 0.8 of dot prod')
plt.title('P2 vs. A1')
plt.ylim([-0.05,0.3])
plt.tight_layout()
plt.savefig('figure/temp/frac_cells_to_0.8.png')




plt.figure(figsize=(6,5))
plt.subplot(211)
for fx in range(nfile):
    # plt.subplot(6,6,fx+1)
    plt.plot(tvec, frac_similarity_overlap_P1[shuff,fx], label='P1-P2', alpha=0.3)
plt.plot(tvec, np.mean(frac_similarity_overlap_P1[shuff,:,:],axis=0), c='k', lw=3.5)
plt.plot(tvec, np.mean(frac_similarity_overlap_P1[shuff,:,:],axis=0), c='purple', label='P1-P2', lw=2)
plt.axvline(tvec[tix_sample],color='gray', linestyle='--')
plt.axvline(tvec[tix_delay],color='gray', linestyle='--')
plt.axvline(tvec[tix_response],color='gray', linestyle='--')    
plt.title('P2 vs. P1')
plt.ylim([0,1])

plt.subplot(212)
for fx in range(nfile):
    plt.plot(tvec, frac_similarity_overlap_A1[shuff,fx], label='A1-P2', alpha=0.3)
plt.plot(tvec, np.mean(frac_similarity_overlap_A1[shuff,:,:],axis=0), c='k', label='A1-P2', lw=3.5)    
plt.plot(tvec, np.mean(frac_similarity_overlap_A1[shuff,:,:],axis=0), c='limegreen', label='A1-P2', lw=2)
plt.axvline(tvec[tix_sample],color='gray', linestyle='--')
plt.axvline(tvec[tix_delay],color='gray', linestyle='--')
plt.axvline(tvec[tix_response],color='gray', linestyle='--')    
plt.title('P2 vs. A1')
plt.ylim([0,1])
plt.xlabel('time (s)')
plt.ylabel('frac similarity by shared cells')
plt.tight_layout()
plt.savefig('figure/temp/frac_shared_cells_similarity.png')


shuff = 0
count_accum_overlapcells = np.zeros((nshuff,nfile,ntimestep))
for shuff in range(nshuff):
    for fx in range(nfile):
        accum_overlapcells = np.array([])
        ncell = len(hadamard_xcont_2P_all[0,file_sorted[fx],0][:,0])
        for tix in range(ntimestep):
            _current_overlapcells = overlapcells[shuff,fx,tix]
            _new_overlapcells = _current_overlapcells[~np.isin(_current_overlapcells,accum_overlapcells)]
            accum_overlapcells = np.append(accum_overlapcells,_new_overlapcells)
            assert len(accum_overlapcells) == np.unique(accum_overlapcells).shape[0]
            
            count_accum_overlapcells[shuff,fx,tix] = len(accum_overlapcells) / ncell


shuff = 0
plt.figure()
for fx in range(nfile):
    plt.plot(tvec, count_accum_overlapcells[shuff,fx] - count_accum_overlapcells[shuff,fx,tix_sample])
plt.axvline(tvec[tix_sample],color='gray', linestyle='--')
plt.axvline(tvec[tix_delay],color='gray', linestyle='--')
plt.axvline(tvec[tix_response],color='gray', linestyle='--')    
plt.xlim(tvec[tix_sample],tvec[tix_response])
plt.ylim([-0.01,0.2])
plt.xlabel('time (s)')
plt.ylabel('accumulated cells')
plt.savefig('figure/temp/accumulated_cells.png')


slope_sample = count_accum_overlapcells[:,:,tix_delay] - count_accum_overlapcells[:,:,tix_sample]
slope_delay  = count_accum_overlapcells[:,:,tix_response] - count_accum_overlapcells[:,:,tix_delay]


idx = np.linspace(0.01,0.08,10)
plt.figure()
plt.scatter(slope_sample[shuff], slope_delay[shuff])
plt.plot(idx, idx, c='gray', linestyle='--')
plt.xlabel('slope of delay epoch')
plt.ylabel('slope of sample epoch')
plt.savefig('figure/temp/accumulated_cells_slope.png')


shuff = 0
plt.figure()
corr = np.corrcoef(CD_dotproduct_avg[file_sorted], slope_delay[shuff])[0,1]
plt.scatter(CD_dotproduct_avg[file_sorted], slope_delay[shuff])
plt.title('cor ' + str(np.round(corr,decimals=3)))
plt.xlabel('CD dot product')
plt.ylabel('slope of delay')
plt.savefig('figure/temp/accumulated_cells_slope_CDdotprod.png')



corall = np.zeros(nshuff)
for shuff in range(nshuff):
    corall[shuff] = np.corrcoef(CD_dotproduct_avg[file_sorted], slope_delay[shuff])[0,1]
plt.figure()
plt.hist(corall)



# corr = np.zeros((nshuff,ntimestep))
# for shuff in range(nshuff):
#     for ti in range((ntimestep)):
#         corr[shuff,ti] = np.corrcoef(CD_dotproduct_avg, frac_overlap_A1[shuff,:,ti])[0,1]
    
# plt.figure()
# plt.plot(tvec,np.mean(corr,axis=0))
# plt.axvline(tvec[tix_sample],color='gray', linestyle='--')
# plt.axvline(tvec[tix_delay],color='gray', linestyle='--')
# plt.axvline(tvec[tix_response],color='gray', linestyle='--')    




        
# fx = 30        
# plt.figure(figsize=(10,10))
# for ci in range(25):
#     plt.subplot(5,5,ci+1)
#     cnum = np.random.randint(hadamard_xcont_2P_all[shuff,file_sorted[fx],0].shape[0])
#     cell_P1 = hadamard_xcont_2P_all[shuff,file_sorted[fx],0][cnum,:]
#     cell_A1 = hadamard_xcont_2P_all[shuff,file_sorted[fx],1][cnum,:]
#     plt.plot(tvec, cell_P1, c='purple')
#     plt.plot(tvec, cell_A1, c='limegreen')
#     plt.axvline(tvec[tix_sample],color='gray', linestyle='--')
#     plt.axvline(tvec[tix_delay],color='gray', linestyle='--')
#     plt.axvline(tvec[tix_response],color='gray', linestyle='--')    
# plt.tight_layout()

        
        
# plt.figure(figsize=(8,8))
# for ti, tix in enumerate(np.arange(tix_sample,tix_sample+16)):
#     plt.subplot(4,4,ti+1)
#     corr = np.corrcoef(similarity_xcont_2P_avg[:,0,tix], similarity_xcont_2P_avg[:,1,tix])[0,1]
#     plt.scatter(similarity_xcont_2P_avg[:,0,tix], similarity_xcont_2P_avg[:,1,tix])
#     plt.title('t=' + str(np.round(tvec[tix],decimals=2)) + ', ' + str(np.round(corr,decimals=3)))
#     plt.ylim([0,1])
# plt.tight_layout()

#------------------------------------------------------#        
proj_xcont_P_avg = np.mean(proj_xcont_P_all,axis=0)
proj_xcont_A_avg = np.mean(proj_xcont_A_all,axis=0)

plt.figure()
plt.imshow(proj_xcont_P_avg, cmap='jet', aspect='auto')    
plt.axvline(tix_sample,color='white')
plt.axvline(tix_delay,color='white')
plt.axvline(tix_response,color='white')
plt.colorbar()
plt.tight_layout()
plt.savefig('figure/temp/proj_xcontext_heatmap_P.png')


plt.figure()
plt.imshow(proj_xcont_A_avg, cmap='jet', aspect='auto')    
plt.axvline(tix_sample,color='white')
plt.axvline(tix_delay,color='white')
plt.axvline(tix_response,color='white')
plt.colorbar()
plt.tight_layout()
plt.savefig('figure/temp/proj_xcontext_heatmap_A.png')



corr_P = np.zeros((nCD))
corr_A = np.zeros((nCD))
for tix in range(nCD):
    tix_start = tix_sample+4+tix
    tix_end   = tix_sample+4+tix + 4
    corr_P[tix] = np.corrcoef(np.mean(proj_xcont_P_all[:,tix,tix_start:tix_end],axis=1),CD_dotproduct_all[0])[0,1]
    corr_A[tix] = np.corrcoef(np.mean(proj_xcont_A_all[:,tix,tix_start:tix_end],axis=1),CD_dotproduct_all[0])[0,1]


plt.figure()
plt.plot(corr_P)
plt.plot(corr_A)
plt.tight_layout()

plt.figure(figsize=(8,8))
for tix in range(nCD):
    plt.subplot(4,4,tix+1)
    plt.scatter(CD_dotproduct_all[0], np.mean(proj_xcont_P_all[:,tix,tix_sample+4+tix:tix_sample+4+tix+4],axis=1))
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('cor ' + str(np.round(corr_P[tix],decimals=3)))
    plt.ylim([-0.3,0.7])
plt.subplot(4,4,tix+2)
plt.plot(corr_P)
plt.axhline(0, color='gray', linestyle='--')
plt.ylabel('corr')
plt.tight_layout()
plt.savefig('figure/temp/proj_xcontext_corrP.png')



plt.figure(figsize=(8,8))
for tix in range(nCD):
    plt.subplot(4,4,tix+1)
    plt.scatter(CD_dotproduct_all[0], np.mean(proj_xcont_A_all[:,tix,tix_sample+4+tix:tix_sample+4+tix+4],axis=1))
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('cor ' + str(np.round(corr_A[tix],decimals=3)))
    plt.ylim([-0.6,0.7])
plt.subplot(4,4,tix+2)
plt.plot(corr_A)
plt.axhline(0, color='gray', linestyle='--')
plt.ylabel('corr')
plt.tight_layout()
plt.savefig('figure/temp/proj_xcontext_corrA.png')




plt.figure()
# for fx in range(nfile):
#     # plt.subplot(6,6,fx+1)
#     plt.plot(tvec, proj_xcont_P_all[fx])
plt.plot(tvec, np.mean(proj_xcont_P_all,axis=0),c='k',lw=3)    
plt.axvline(tsample, color="gray", linestyle="--")
plt.axvline(tdelay, color="gray", linestyle="--")
plt.axvline(tresponse, color="gray", linestyle="--")
plt.tight_layout()


plt.figure(figsize=(10,10))
for fx in range(nfile):
    plt.subplot(6,6,fx+1)
    plt.plot(tvec, proj_xcont_P_all[fx])
    plt.plot(tvec, np.mean(proj_xcont_P_all,axis=0),c='k',lw=3)    
    plt.axvline(tsample, color="gray", linestyle="--")
    plt.axvline(tdelay, color="gray", linestyle="--")
    plt.axvline(tresponse, color="gray", linestyle="--")
    plt.title(str(np.round(CD_dotproduct_all[0,fx],decimals=3)),fontsize=10)
plt.tight_layout()
plt.savefig('figure/temp/proj_sample_late.png')
# plt.ylim([-2,3])


proj_avg = np.mean(proj_xcont_P_all[:,tix_delay-4:tix_delay],axis=1)
corr = np.corrcoef(proj_avg,CD_dotproduct_all[0])[0,1]
plt.figure()
plt.scatter(CD_dotproduct_all[0], proj_avg)
plt.title('corr ' + str(np.round(corr,decimals=3)))
plt.savefig('figure/temp/proj_sample_late_corr.png')


timex = tix_delay

'''
Divide the mice into two groups
'''
numfirstGroup           = 5
CD_dotproduct_avg       = np.mean(CD_dotproduct_all,axis=0)
mouse_group_ranks       = functions.rank_mouse_groups(mouse_group, CD_dotproduct_avg, nfile)
firstGroup, secondGroup = functions.divide_into_two_groups(mouse_group_ranks, numfirstGroup, nfile)

'''
Correlation between CDdotproduct vs. Change in neural state across contexts
'''
cor_CDdot_IDP_CD1, cor_CDdot_IDA_CD1, \
cor_CDdot_IDP_CD2, cor_CDdot_IDA_CD2 = functions.correlation_CDdotprod_vs_ID_along_CD(
                                        nshuff, ntimestep, CD_dotproduct_all, 
                                        ID_P_along_CD1, ID_A_along_CD1, 
                                        ID_P_along_CD2, ID_A_along_CD2
                                        )
cor_CDdot_distP_firstGroup,  cor_CDdot_distA_firstGroup, \
cor_CDdot_distP_secondGroup, cor_CDdot_distA_secondGroup, \
cor_CDdot_distP,             cor_CDdot_distA = functions.correlation_CDdotprod_vs_distance_btw_contexts(
                                                nshuff, ntimestep, nfile, CD_dotproduct_all, mouse_group,
                                                distance_P_all, distance_A_all
                                                )


'''
Stimulus strength vs. CD similarity
Stimulus strength vs. Change in neural state
'''
nthresh = 4
thresh = np.linspace(0.1,0.4,nthresh)
CD_sample_1_distance, CD_sample_2_distance, CD_sample_1_std, CD_sample_2_std = functions.stimulus_distance(nshuff, nfile, CD_sample_1, CD_sample_2)
CD_sample_1_selective, CD_sample_2_selective = functions.stimulus_selective_cells(nshuff, nfile, nthresh, thresh, CD_sample_1, CD_sample_2, CD_sample_1_std, CD_sample_2_std)
CD_sample_1_distance_avg = np.mean(CD_sample_1_distance,axis=0)
CD_sample_2_distance_avg = np.mean(CD_sample_2_distance,axis=0)
CD_sample_1_selective_avg = np.mean(CD_sample_1_selective,axis=0)        
CD_sample_2_selective_avg = np.mean(CD_sample_2_selective,axis=0)        


'''
CD similarity          vs. Stimulus encoding
Change in neural state vs. Stimulus encoding
'''        
StimEncodingPlotter = plot_stimulus.StimulusEncodingPlotter(
    CD_dotproduct_avg=CD_dotproduct_avg,
    CD_sample_1_distance_avg=CD_sample_1_distance_avg,
    CD_sample_2_distance_avg=CD_sample_2_distance_avg,
    CD_sample_1_selective_avg=CD_sample_1_selective_avg,
    CD_sample_2_selective_avg=CD_sample_2_selective_avg,
    thresh=thresh,
    ID_P_along_CD1=ID_P_along_CD1,
    ID_A_along_CD1=ID_A_along_CD1,
    tix_delay=tix_delay,
    save_dir="figure/stimulus",
)
StimEncodingPlotter.plot_cd_similarity_vs_frac_selective(context=1,save=False)
StimEncodingPlotter.plot_cd_similarity_vs_frac_selective(context=2,save=False)
StimEncodingPlotter.plot_cd_similarity_vs_distance(save=False)
StimEncodingPlotter.plot_neural_state_vs_frac_selective(stimtype="P",save=False)
StimEncodingPlotter.plot_neural_state_vs_frac_selective(stimtype="A",save=False)
StimEncodingPlotter.plot_neural_state_vs_distance(save=False)



'''
Change in neural states (from context 1 to context 2) along CD1
'''
NeuralStateAlongCDPlotter = plot_neuralstate_cd.NeuralStateAlongCDPlotter(
    CD_dotproduct_avg=CD_dotproduct_avg,
    mouse_group_ranks=mouse_group_ranks,
    ID_P_along_CD1=ID_P_along_CD1,
    ID_A_along_CD1=ID_A_along_CD1,
    ID_P_along_CD2=ID_P_along_CD2,
    ID_A_along_CD2=ID_A_along_CD2,
    cor_CDdot_IDP_CD1=cor_CDdot_IDP_CD1,
    cor_CDdot_IDA_CD1=cor_CDdot_IDA_CD1,
    cor_CDdot_IDP_CD2=cor_CDdot_IDP_CD2,
    cor_CDdot_IDA_CD2=cor_CDdot_IDA_CD2,
    tvec=tvec,
    tsample=tsample,
    tdelay=tdelay,
    tresponse=tresponse,
    tix_sample=tix_sample,
    tix_delay=tix_delay,
    tix_response=tix_response,
    firstGroup=firstGroup,
    secondGroup=secondGroup,
)
NeuralStateAlongCDPlotter.plot_cd_dotproduct_vs_neuralstate_along_CD(cd=1, stimtype="P", save=False)
NeuralStateAlongCDPlotter.plot_cd_dotproduct_vs_neuralstate_along_CD(cd=1, stimtype="A", save=False)
NeuralStateAlongCDPlotter.plot_corr_time(cd=1, save=False)
NeuralStateAlongCDPlotter.plot_corr_time(cd=2, save=False)
NeuralStateAlongCDPlotter.plot_mean_time(cd=1, save=False)
NeuralStateAlongCDPlotter.plot_mean_time(cd=2, save=False)

'''
Euclidean Distance between contexts
'''
NeuralStateDistancePlotter = plot_neuralstate_eucldist.NeuralStateDistancePlotter(
    CD_dotproduct_avg=CD_dotproduct_avg,
    mouse_group_ranks=mouse_group_ranks,
    distance_P_all=distance_P_all,
    cor_CDdot_distP_firstGroup=cor_CDdot_distP_firstGroup,
    cor_CDdot_distP_secondGroup=cor_CDdot_distP_secondGroup,
    ID_P_along_TD1=ID_P_along_TD1,
    ID_A_along_TD1=ID_A_along_TD1,
    tvec=tvec,
    tsample=tsample,
    tdelay=tdelay,
    tresponse=tresponse,
    tix_sample=tix_sample,
    tix_delay=tix_delay,
    tix_response=tix_response,
    firstGroup=firstGroup,
    secondGroup=secondGroup,
)
NeuralStateDistancePlotter.plot_distance_scatter_time(save=False)
NeuralStateDistancePlotter.plot_distance_corr_time(save=False)
NeuralStateDistancePlotter.plot_distance_mean_time(save=False)

NeuralStateDistancePlotter.plot_transient_direction(save=False)



