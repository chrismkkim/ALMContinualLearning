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

nshuff = 20
nfile  = len(metadata)
CD_dotproduct_all  = np.zeros((nshuff,nfile))
ID_P_along_TD1      = np.zeros((nshuff,nfile))
ID_A_along_TD1      = np.zeros((nshuff,nfile))
ID_P_along_TD2      = np.zeros((nshuff,nfile))
ID_A_along_TD2      = np.zeros((nshuff,nfile))
ID_P_along_CD1      = np.zeros((nshuff,nfile,ntimestep))
ID_A_along_CD1      = np.zeros((nshuff,nfile,ntimestep))
ID_P_along_CD2      = np.zeros((nshuff,nfile,ntimestep))
ID_A_along_CD2      = np.zeros((nshuff,nfile,ntimestep))
ID_P_project_CD1     = np.zeros((nshuff,nfile,ntimestep))
ID_A_project_CD1     = np.zeros((nshuff,nfile,ntimestep))
ID_P_project_CD2     = np.zeros((nshuff,nfile,ntimestep))
ID_A_project_CD2     = np.zeros((nshuff,nfile,ntimestep))
distance_P_all        = np.zeros((nshuff,nfile,ntimestep))
distance_A_all        = np.zeros((nshuff,nfile,ntimestep))
CD_sample_1       = np.empty((nshuff,nfile),dtype=object)
CD_sample_2       = np.empty((nshuff,nfile),dtype=object)
cor_CD_1P_firstGroup       = np.zeros((nshuff,ntimestep))
cor_CD_1A_firstGroup       = np.zeros((nshuff,ntimestep))
cor_CD_2P_firstGroup       = np.zeros((nshuff,ntimestep))
cor_CD_2A_firstGroup       = np.zeros((nshuff,ntimestep))
cor_CD_1P_secondGroup      = np.zeros((nshuff,ntimestep))
cor_CD_1A_secondGroup      = np.zeros((nshuff,ntimestep))
cor_CD_2P_secondGroup      = np.zeros((nshuff,ntimestep))
cor_CD_2A_secondGroup      = np.zeros((nshuff,ntimestep))

for shuff in range(nshuff):
    
    print(str(shuff) + ' / ' + str(nshuff))
    
    for fx in range(nfile):
        
        print(fx)

        #--- sorted by relearn speed ---#
        filepath   = dirpath + merged_ID[fx] + '.npy'
        data       = np.load(filepath, allow_pickle=True)
        opto_sess1 = data['deconvolved'][0]
        opto_sess2 = data['deconvolved'][1]
        ncell      = opto_sess1.shape[1]

        # Compute Coding Direction @ delay
        data_type = 'deconvolved'
        CD_dotproduct, CD_sess1_population, CD_sess2_population, \
        proj_CD_1R, proj_CD_1L, proj_CD_2R, proj_CD_2L,\
        opto_1R_testtrials, opto_1L_testtrials, opto_2R_testtrials, opto_2L_testtrials = functions.compute_CD_delay(data, data_type, 'input')

        CD_dotproduct_all[shuff,fx] = CD_dotproduct
        
        # neural activity (test trials)
        opto_1P_testtrials = opto_1R_testtrials
        opto_1A_testtrials = opto_1L_testtrials
        opto_2P_testtrials = opto_2L_testtrials
        opto_2A_testtrials = opto_2R_testtrials                

        # Initial Direction
        opto_1P_testtrials_avg = np.mean(opto_1P_testtrials,axis=2) # neurons x time
        opto_2P_testtrials_avg = np.mean(opto_2P_testtrials,axis=2) # neurons x time
        opto_1A_testtrials_avg = np.mean(opto_1A_testtrials,axis=2) # neurons x time
        opto_2A_testtrials_avg = np.mean(opto_2A_testtrials,axis=2) # neurons x time
        ID_P = (opto_2P_testtrials_avg - opto_1P_testtrials_avg).T  # time x neurons
        ID_A = (opto_2A_testtrials_avg - opto_1A_testtrials_avg).T  # time x neurons

        # Euclidean distance between contexts in time
        distance_P_all[shuff,fx] = np.mean((opto_2P_testtrials_avg - opto_1P_testtrials_avg)**2,axis=0) / (np.var(opto_1P_testtrials_avg,axis=0) + np.var(opto_2P_testtrials_avg,axis=0))
        distance_A_all[shuff,fx] = np.mean((opto_2A_testtrials_avg - opto_1A_testtrials_avg)**2,axis=0) / (np.var(opto_1A_testtrials_avg,axis=0) + np.var(opto_2A_testtrials_avg,axis=0))
        
        # Coding Direction
        for tx in range(ntimestep):
            ID_P_along_CD1[shuff,fx,tx] = functions.cosine_similarity(CD_sess1_population, ID_P[tx])
            ID_A_along_CD1[shuff,fx,tx] = functions.cosine_similarity(CD_sess1_population, ID_A[tx])
            ID_P_along_CD2[shuff,fx,tx] = functions.cosine_similarity(CD_sess2_population, ID_P[tx])
            ID_A_along_CD2[shuff,fx,tx] = functions.cosine_similarity(CD_sess2_population, ID_A[tx])

        # Transient Direction 
        ID_P_along_TD1, ID_A_along_TD1, ID_P_along_TD2, ID_A_along_TD2 = functions.compute_TD_cossim(
                    proj_CD_1R, proj_CD_1L, proj_CD_2R, proj_CD_2L, 
                    tix_response, tix_delay, 
                    opto_1R_testtrials, opto_1L_testtrials, opto_2L_testtrials, opto_2R_testtrials,
                    ID_P_along_TD1, ID_A_along_TD1, ID_P_along_TD2, ID_A_along_TD2,
                    shuff, fx,
                    ID_P, ID_A
                    )
        
        # Sample Direction
        CD_sample_dotproduct, CD_sample_1_population, CD_sample_2_population = functions.compute_CD_sample_update(data,data_type)        
        CD_sample_1[shuff,fx] = CD_sample_1_population
        CD_sample_2[shuff,fx] = CD_sample_2_population
        # end of mouse loop
    # end of shuffle loop
        
    


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



#------------------------------#


# plt.figure(figsize=(6,3))
# plt.subplot(121)
# plt.plot(CD_dotproduct_all, ID_P_project_CD1[timex], marker='o', linestyle='', c='purple')
# plt.plot(CD_dotproduct_all, ID_A_project_CD1[timex], marker='o', linestyle='', c='limegreen')
# plt.axvline(0, color='gray', linestyle='--')
# plt.axhline(0, color='gray', linestyle='--')
# plt.title('Context 1')
# plt.xlabel('CD dot product')
# plt.ylabel('Project ID to CD')
# plt.ylim([-1.2,1.2])
# plt.subplot(122)
# plt.plot(CD_dotproduct_all, ID_P_project_CD2[timex], marker='o', linestyle='', c='purple')
# plt.plot(CD_dotproduct_all, ID_A_project_CD2[timex], marker='o', linestyle='', c='limegreen')
# plt.axvline(0, color='gray', linestyle='--')
# plt.axhline(0, color='gray', linestyle='--')
# plt.title('Context 2')
# plt.xlabel('CD dot product')
# plt.ylabel('Project ID to CD')
# plt.ylim([-1.2,1.2])
# plt.tight_layout()
# plt.savefig('figure/summary/CDdotprod_Project_ID_to_CD.pdf')



# plt.figure(figsize=(6,3))
# plt.subplot(121)
# plt.plot(CD_dotproduct_all, ID_P_along_TD1, marker='o', linestyle='', c='purple')
# plt.plot(CD_dotproduct_all, ID_A_along_TD1, marker='o', linestyle='', c='limegreen')
# plt.axvline(0, color='gray', linestyle='--')
# plt.axhline(0, color='gray', linestyle='--')
# plt.title('Context 1')
# plt.xlabel('CD dot product')
# plt.ylabel('TD dot ID')
# plt.subplot(122)
# plt.plot(CD_dotproduct_all, ID_P_along_TD2, marker='o', linestyle='', c='purple')
# plt.plot(CD_dotproduct_all, ID_A_along_TD2, marker='o', linestyle='', c='limegreen')
# plt.axvline(0, color='gray', linestyle='--')
# plt.axhline(0, color='gray', linestyle='--')
# plt.title('Context 2')
# plt.xlabel('CD dot product')
# plt.ylabel('TD dot ID')
# plt.tight_layout()
# plt.savefig('figure/summary/CDdotprod_TDIDdotprod.pdf')





# '''
# Initial Direction between contexts along CD1
# '''

# grpid = np.arange(nfile)
# cmap = plt.cm.jet
# norm = plt.Normalize(vmin=np.min(mouse_group_ranks), vmax=np.max(mouse_group_ranks))
# time_points = [tix_sample, tix_delay, tix_response]
# plt.figure(figsize=(7,6))
# for ix, tx in enumerate(time_points):
#     plt.subplot(2,3,ix+1)
#     # plt.axvspan(CD_dotproduct_all[file_sorted_by_CDdotprod[0]], CD_dotproduct_all[file_sorted_by_CDdotprod[15]], color='dodgerblue', alpha=0.3)
#     # plt.axvspan(CD_dotproduct_all[file_sorted_by_CDdotprod[15]], CD_dotproduct_all[file_sorted_by_CDdotprod[-1]], color='darkorange', alpha=0.3)
#     plt.scatter(CD_dotproduct_avg[grpid], np.mean(ID_P_along_CD1[:,grpid,tx],axis=0), marker='o', linestyle='', c=cmap(norm(mouse_group_ranks[grpid])))
#     plt.axvline(0, color='gray', linestyle='--')
#     plt.axhline(0, color='gray', linestyle='--')
#     plt.xlabel('CD dot product')
#     if ix == 0:
#         plt.ylabel('Posterior IDir along CD1')
#     plt.title('t=' + str(np.round(tvec[tx],decimals=2)))
#     plt.ylim([-0.3,0.8])
    
#     plt.subplot(2,3,3+ix+1)
#     plt.scatter(CD_dotproduct_avg[grpid], np.mean(ID_A_along_CD1[:,grpid,tx],axis=0), marker='o', linestyle='', c=cmap(norm(mouse_group_ranks[grpid])))
#     plt.axvline(0, color='gray', linestyle='--')
#     plt.axhline(0, color='gray', linestyle='--')
#     plt.xlabel('CD dot product')
#     if ix == 0:
#         plt.ylabel('Anterior IDir along CD1')
#     plt.title('t=' + str(np.round(tvec[tx],decimals=2)))
#     plt.ylim([-0.8,0.3])
    
# plt.tight_layout()
# # plt.savefig('figure/summary/distance_CDdotprod_update.pdf')





# file_sorted_by_CDdotprod = np.argsort(CD_dotproduct_all)
# plt.figure()
# plt.subplot(211)
# for fx in range(5):
#     fx_sort = file_sorted_by_CDdotprod[fx]
#     plt.plot(tvec, distance_P_all[:,fx_sort], c=jet[fx], marker='o')
# plt.axvline(tsample, color='gray', linestyle='--')
# plt.axvline(tdelay, color='gray', linestyle='--')
# plt.axvline(tresponse, color='gray', linestyle='--')
# plt.title('Sessions of CD dot prod < 0')

# plt.subplot(212)
# for fx in np.arange(nfile-5,nfile):
#     fx_sort = file_sorted_by_CDdotprod[fx]
#     plt.plot(tvec, distance_P_all[:,fx_sort], c=jet[fx], marker='o')
# plt.axvline(tsample, color='gray', linestyle='--')
# plt.axvline(tdelay, color='gray', linestyle='--')
# plt.axvline(tresponse, color='gray', linestyle='--')
# plt.xlabel('time (s)')
# plt.ylabel('Distance between contexts')
# plt.title('Sessions of CD dot prod > 0')
# plt.tight_layout()
# plt.savefig('figure/summary/distance_time.pdf')


# plt.figure(figsize=(6,3))
# plt.subplot(121)
# plt.plot(CD_dotproduct_all, ID_P_along_TD1, marker='o', linestyle='')
# plt.plot(CD_dotproduct_all, ID_P_along_TD2, marker='o', linestyle='')
# plt.axvline(0, color='gray', linestyle='--')
# plt.axhline(0, color='gray', linestyle='--')
# plt.subplot(122)
# plt.plot(CD_dotproduct_all, ID_A_along_TD1, marker='o', linestyle='')
# plt.plot(CD_dotproduct_all, ID_A_along_TD2, marker='o', linestyle='')
# plt.axvline(0, color='gray', linestyle='--')
# plt.axhline(0, color='gray', linestyle='--')
# plt.tight_layout()




# plt.figure(figsize=(4,3))
# plt.plot(tvec, cor_TD_1P_all[:,0], label='1P', c='purple')
# plt.plot(tvec, cor_TD_1A_all[:,0], label='1A', c='limegreen')
# plt.plot(tvec, cor_TD_2P_all[:,0], label='2P', c='purple', linestyle='--')
# plt.plot(tvec, cor_TD_2A_all[:,0], label='2A', c='limegreen', linestyle='--')
# plt.axvline(tsample, color='gray', linestyle='--')
# plt.axvline(tdelay, color='gray', linestyle='--')
# plt.axvline(tresponse, color='gray', linestyle='--')
# plt.xlabel('time (s)')
# plt.ylabel('correlation')
# plt.legend()
# plt.tight_layout()
# plt.savefig('figure/summary/corr_TD_all_time.pdf')


# plt.figure(figsize=(4,3))
# # plt.plot(tvec, cor_CDdot_distP[:,2], label='1P', c='purple')
# # plt.plot(tvec, cor_CDdot_distA[:,2], label='1A', c='limegreen')
# plt.plot(tvec, np.mean(cor_CDdot_distP_firstGroup,axis=0), label='P Grp1', c='purple')
# plt.plot(tvec, np.mean(cor_CDdot_distP_secondGroup,axis=0), label='P Grp2', c='purple', linestyle='--')
# plt.plot(tvec, np.mean(cor_CDdot_distA_firstGroup,axis=0), label='A Grp1', c='limegreen')
# plt.plot(tvec, np.mean(cor_CDdot_distA_secondGroup,axis=0), label='A Grp2', c='limegreen', linestyle='--')

# plt.axvline(tsample, color='gray', linestyle='--')
# plt.axvline(tdelay, color='gray', linestyle='--')
# plt.axvline(tresponse, color='gray', linestyle='--')
# plt.xlabel('time (s)')
# plt.ylabel('correlation')
# plt.legend()
# plt.tight_layout()
# # plt.savefig('figure/summary/distance_corr_time_firstNsecond.pdf')


# plt.figure(figsize=(6,3))
# plt.subplot(121)
# plt.hist(cor_TD_1P_all, bins=20, range=(-1,1), histtype='step', color='purple', label='Pos')
# plt.hist(cor_TD_1A_all, bins=20, range=(-1,1), histtype='step', color='limegreen', label='Ant')
# plt.title('TD of Context 1')
# plt.xlabel('cor of TD and ID')
# plt.legend(frameon=False)
# # plt.ylabel(r'$TD_{1R} \cdot ID_R$')
# # plt.xlim([-1,0.5])
# # plt.ylim([-0.5,0.5])
# plt.subplot(122)
# plt.hist(cor_TD_2P_all, bins=20, range=(-1,1), histtype='step', color='purple')
# plt.hist(cor_TD_2A_all, bins=20, range=(-1,1), histtype='step', color='limegreen')
# plt.title('TD of Context 2')
# plt.xlabel('cor of TD and ID')
# # plt.ylabel(r'$TD_{1L} \cdot ID_L$')
# # plt.xlim([-1,0.5])
# # plt.ylim([-0.5,0.5])
# plt.tight_layout()
# plt.savefig('figure/summary/corr_TD_ID.pdf')



# plt.figure(figsize=(6,3))
# plt.subplot(121)
# plt.hist(cor_CDdot_IDP_CD1, bins=20, range=(-1,1), histtype='step', color='purple', label='Pos')
# plt.hist(cor_CDdot_IDA_CD1, bins=20, range=(-1,1), histtype='step', color='limegreen', label='Ant')
# plt.title('CD of Context 1')
# plt.xlabel('cor of CD and ID')
# plt.legend(frameon=False)
# # plt.ylabel(r'$TD_{1R} \cdot ID_R$')
# # plt.xlim([-1,0.5])
# # plt.ylim([-0.5,0.5])
# plt.subplot(122)
# plt.hist(cor_CDdot_IDP_CD2, bins=20, range=(-1,1), histtype='step', color='purple')
# plt.hist(cor_CDdot_IDA_CD2, bins=20, range=(-1,1), histtype='step', color='limegreen')
# plt.title('CD of Context 2')
# plt.xlabel('cor of CD and ID')
# # plt.ylabel(r'$TD_{1L} \cdot ID_L$')
# # plt.xlim([-1,0.5])
# # plt.ylim([-0.5,0.5])
# plt.tight_layout()
# plt.savefig('figure/summary/corr_CD_ID.pdf')



# plt.figure(figsize=(6,3))
# plt.subplot(121)
# plt.hist(cor_CD_TD_1P_all.flatten(), bins=20, range=(-0.5,0.5), histtype='step', color='purple', label='Pos')
# plt.hist(cor_CD_TD_1A_all.flatten(), bins=20, range=(-0.5,0.5), histtype='step', color='limegreen', label='Ant')
# plt.title('Context 1')
# plt.xlabel('cor of CD and TD')
# plt.legend(frameon=False)
# # plt.ylabel(r'$TD_{1R} \cdot ID_R$')
# # plt.xlim([-1,0.5])
# # plt.ylim([-0.5,0.5])
# plt.subplot(122)
# plt.hist(cor_CD_TD_2P_all.flatten(), bins=20, range=(-0.5,0.5), histtype='step', color='purple')
# plt.hist(cor_CD_TD_2A_all.flatten(), bins=20, range=(-0.5,0.5), histtype='step', color='limegreen')
# plt.title('Context 2')
# plt.xlabel('cor of CD and TD')
# # plt.ylabel(r'$TD_{1L} \cdot ID_L$')
# # plt.xlim([-1,0.5])
# # plt.ylim([-0.5,0.5])
# plt.tight_layout()
# plt.savefig('figure/summary/corr_CD_TD.pdf')


