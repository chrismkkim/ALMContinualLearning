import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
from scipy.stats import norm

class LearnedActivityPlots:
    def __init__(
        self,
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
    ):
        self.learned_activity = learned_activity
        self.learned_activity_sum = learned_activity_sum
        self.frac_cells = frac_cells
        self.frac_cells_sum = frac_cells_sum
        self.CD_dotproduct = CD_dotproduct
        self.relearn_time = relearn_time
        self.shuff_learned_activity = shuff_learned_activity
        self.shuff_learned_activity_sum = shuff_learned_activity_sum
        self.shuff_frac_cells = shuff_frac_cells
        self.shuff_frac_cells_sum = shuff_frac_cells_sum            
        self.shuff_CD_dotproduct = shuff_CD_dotproduct
        self.synthetic_data = synthetic_data
        self.sorted_indices_by_relearn_speed = sorted_indices_by_relearn_speed
        self.merged_ID = merged_ID
        self.num_outliers = num_outliers
        self.num_topcells = num_topcells
        self.num_totalcells = num_totalcells
        self.nfile = nfile
        self.P1_set2 = P1_set2
        self.P2_set2 = P2_set2
        self.A1_set2 = A1_set2
        self.A2_set2 = A2_set2
        self.c1 = 'darkviolet' 
        self.c2 = 'blue' 
        self.c3 = 'darkorange' 
        self.c4 = 'limegreen'
        plt.style.use('pyplot_setting.mplstyle')

    def plot_outliers_topcells(self, savefig):
        
        num_outliers = self.num_outliers
        num_topcells = self.num_topcells
        num_totalcells = self.num_totalcells
        
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
        
        if savefig:
            plt.savefig('figure/learned_activity/session_all/frac_outliers_topcells.pdf')
            plt.close()
            
    def plot_CDdotproduct_learntime(self, savefig):
        CD_dotproduct = self.CD_dotproduct
        relearn_time  = self.relearn_time
        relearn_time_sorted = np.sort(relearn_time)
        
        plt.figure(figsize=(1.6,1.5))
        plt.hist(CD_dotproduct, bins=10, range=(-1,0.5), histtype='step', color='k')
        plt.xlabel(r'$CD_1 \cdot CD_2$')
        plt.ylabel('mouse/FOV count')
        plt.xticks([-1,-0.5,0,0.5])
        plt.tight_layout()
        if savefig:
            plt.savefig('figure/learned_activity/CDdotproduct/CDdotproduct.pdf')
            plt.close()
        

        plt.figure(figsize=(1.9,1.6))
        plt.plot(CD_dotproduct, relearn_time_sorted, marker='o', linestyle='', c='k', ms=4, mfc='None', mew=0.5)
        plt.xlabel(r'$CD_1 \cdot CD_2$')
        plt.ylabel('relative trials to reach \n 75% performance')
        _corr = np.corrcoef(CD_dotproduct, relearn_time)[0,1]
        plt.annotate(r'$\rho = $' + str(np.round(_corr,decimals=3)), xy=(0.45,0.85), xycoords='axes fraction', fontsize=8)
        plt.xticks([-1,-0.5,0,0.5])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        if savefig:
            plt.savefig('figure/learned_activity/CDdotproduct/CDdotproduct_vs_relearn_time.pdf')
            plt.close()


    def plot_frac_with_signs_stat(self, savefig, shuffled):
        if not shuffled:
            frac_cells_sum = self.frac_cells_sum
        if shuffled:
            frac_cells_sum = self.shuff_frac_cells_sum
        nfile = self.nfile            

        dP_frac_cells_mean = np.array([np.mean(frac_cells_sum['dS-']['S1+']['dP<0'],axis=0), np.mean(frac_cells_sum['dS-']['S1-']['dP>0'],axis=0)])
        dA_frac_cells_mean = np.array([np.mean(frac_cells_sum['dS-']['S1+']['dA>0'],axis=0), np.mean(frac_cells_sum['dS-']['S1-']['dA<0'],axis=0)])
        dR_frac_cells_mean = np.array([np.mean(frac_cells_sum['dS+']['S1+']['dR<0'],axis=0), np.mean(frac_cells_sum['dS+']['S1-']['dR>0'],axis=0)])
        dL_frac_cells_mean = np.array([np.mean(frac_cells_sum['dS+']['S1+']['dL>0'],axis=0), np.mean(frac_cells_sum['dS+']['S1-']['dL<0'],axis=0)])

        print('dP: ', dP_frac_cells_mean)
        print('dA: ', dA_frac_cells_mean)
        print('dR: ', dR_frac_cells_mean)
        print('dL: ', dL_frac_cells_mean)
        
        c1, c2, c3, c4 = self.c1, self.c2, self.c3, self.c4
        x1, x2, x3, x4 = 0.5, 0.85, 1.35, 1.7
        plt.figure(figsize=(2.5,2.5))
        plt.subplot(211)
        plt.bar([x1,x2,x3,x4], np.array([dP_frac_cells_mean[0],dA_frac_cells_mean[0],dP_frac_cells_mean[1],dA_frac_cells_mean[1]]),width=0.2,color=[c1,c1,c2,c2],edgecolor=[c1,c1,c2,c2], alpha=0.3)
        plt.plot(x1+0.2*np.random.rand(nfile)-0.1, frac_cells_sum['dS-']['S1+']['dP<0'], marker='.', linestyle='', c=c1)
        plt.plot(x2+0.2*np.random.rand(nfile)-0.1, frac_cells_sum['dS-']['S1+']['dA>0'], marker='.', linestyle='', c=c1)
        plt.plot(x3+0.2*np.random.rand(nfile)-0.1, frac_cells_sum['dS-']['S1-']['dP>0'], marker='.', linestyle='', c=c2)
        plt.plot(x4+0.2*np.random.rand(nfile)-0.1, frac_cells_sum['dS-']['S1-']['dA<0'], marker='.', linestyle='', c=c2)
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
        plt.plot(x1+0.2*np.random.rand(nfile)-0.1, frac_cells_sum['dS+']['S1+']['dR<0'], marker='.', linestyle='', c=c3)
        plt.plot(x2+0.2*np.random.rand(nfile)-0.1, frac_cells_sum['dS+']['S1+']['dL>0'], marker='.', linestyle='', c=c3)
        plt.plot(x3+0.2*np.random.rand(nfile)-0.1, frac_cells_sum['dS+']['S1-']['dR>0'], marker='.', linestyle='', c=c4)
        plt.plot(x4+0.2*np.random.rand(nfile)-0.1, frac_cells_sum['dS+']['S1-']['dL<0'], marker='.', linestyle='', c=c4)
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

        if savefig:
            if not shuffled:
                plt.savefig('figure/learned_activity/session_all/frac_with_signs_stat.pdf')
            if shuffled:
                plt.savefig('figure/learned_activity/shuffled/frac_with_signs_stats_shuffled.pdf')
            plt.close()


    def plot_frac_with_signs_quadrant(self, savefig, shuffled):
        if not shuffled:
            frac_cells_sum = self.frac_cells_sum
        if shuffled:
            frac_cells_sum = self.shuff_frac_cells_sum
        nfile = self.nfile        

        frac_cells_dSn_S1p = np.mean(frac_cells_sum['dS-']['S1+']['Quad'],axis=0)
        frac_cells_dSn_S1n = np.mean(frac_cells_sum['dS-']['S1-']['Quad'],axis=0)
        frac_cells_dSp_S1p = np.mean(frac_cells_sum['dS+']['S1+']['Quad'],axis=0)
        frac_cells_dSp_S1n = np.mean(frac_cells_sum['dS+']['S1-']['Quad'],axis=0)

        print('dP: ', frac_cells_dSn_S1p)
        print('dA: ', frac_cells_dSn_S1n)
        print('dR: ', frac_cells_dSp_S1p)
        print('dL: ', frac_cells_dSp_S1n)

        c1, c2, c3, c4 = 'darkviolet', 'blue', 'darkorange', 'limegreen'
        x1, x2, x3, x4 = 0.2, 0.4, 0.6, 0.8
        x5, x6, x7, x8 = 1.2, 1.4, 1.6, 1.8
        ms = 1
        plt.figure(figsize=(2.8,2.5))
        plt.subplot(211)
        clr = c1
        plt.bar([x1,x2,x3,x4], np.array([frac_cells_dSn_S1p[0],frac_cells_dSn_S1p[1],frac_cells_dSn_S1p[2],frac_cells_dSn_S1p[3]]),width=0.1,color=[clr,clr,clr,clr],edgecolor=[clr,clr,clr,clr], alpha=0.3)
        plt.plot(x1+0.1*np.random.rand(nfile)-0.05, frac_cells_sum['dS-']['S1+']['Quad'][:,0], marker='.', linestyle='', c=clr, ms=ms)
        plt.plot(x2+0.1*np.random.rand(nfile)-0.05, frac_cells_sum['dS-']['S1+']['Quad'][:,1], marker='.', linestyle='', c=clr, ms=ms)
        plt.plot(x3+0.1*np.random.rand(nfile)-0.05, frac_cells_sum['dS-']['S1+']['Quad'][:,2], marker='.', linestyle='', c=clr, ms=ms)
        plt.plot(x4+0.1*np.random.rand(nfile)-0.05, frac_cells_sum['dS-']['S1+']['Quad'][:,3], marker='.', linestyle='', c=clr, ms=ms)
        plt.annotate(r'$S_1 > 0$', xy=(x2,0.9), xycoords='data', fontsize=8)
        plt.annotate(r'$S_1 < 0$', xy=(x6,0.9), xycoords='data', fontsize=8)
        plt.title(r'$S_1 \cdot S_2 < 0$', fontsize=8)

        clr = c2
        plt.bar([x5,x6,x7,x8], np.array([frac_cells_dSn_S1n[0],frac_cells_dSn_S1n[1],frac_cells_dSn_S1n[2],frac_cells_dSn_S1n[3]]),width=0.1,color=[clr,clr,clr,clr],edgecolor=[clr,clr,clr,clr], alpha=0.3)
        plt.plot(x5+0.1*np.random.rand(nfile)-0.05, frac_cells_sum['dS-']['S1-']['Quad'][:,0], marker='.', linestyle='', c=clr, ms=ms)
        plt.plot(x6+0.1*np.random.rand(nfile)-0.05, frac_cells_sum['dS-']['S1-']['Quad'][:,1], marker='.', linestyle='', c=clr, ms=ms)
        plt.plot(x7+0.1*np.random.rand(nfile)-0.05, frac_cells_sum['dS-']['S1-']['Quad'][:,2], marker='.', linestyle='', c=clr, ms=ms)
        plt.plot(x8+0.1*np.random.rand(nfile)-0.05, frac_cells_sum['dS-']['S1-']['Quad'][:,3], marker='.', linestyle='', c=clr, ms=ms)
        q1_name = r'$Q1$'
        q2_name = r'$Q2$'
        q3_name = r'$Q3$'
        q4_name = r'$Q4$'
        q5_name = r'$Q1$'
        q6_name = r'$Q2$'
        q7_name = r'$Q3$'
        q8_name = r'$Q4$'
        plt.xticks([x1,x2,x3,x4,x5,x6,x7,x8],[q1_name, q2_name, q3_name, q4_name, q5_name, q6_name, q7_name, q8_name],fontsize=6)
        plt.ylim([0,1])

        plt.subplot(212)
        clr = c3
        plt.bar([x1,x2,x3,x4], np.array([frac_cells_dSp_S1p[0],frac_cells_dSp_S1p[1],frac_cells_dSp_S1p[2],frac_cells_dSp_S1p[3]]),width=0.1,color=[clr,clr,clr,clr],edgecolor=[clr,clr,clr,clr], alpha=0.3)
        plt.plot(x1+0.1*np.random.rand(nfile)-0.05, frac_cells_sum['dS+']['S1+']['Quad'][:,0], marker='.', linestyle='', c=clr, ms=ms)
        plt.plot(x2+0.1*np.random.rand(nfile)-0.05, frac_cells_sum['dS+']['S1+']['Quad'][:,1], marker='.', linestyle='', c=clr, ms=ms)
        plt.plot(x3+0.1*np.random.rand(nfile)-0.05, frac_cells_sum['dS+']['S1+']['Quad'][:,2], marker='.', linestyle='', c=clr, ms=ms)
        plt.plot(x4+0.1*np.random.rand(nfile)-0.05, frac_cells_sum['dS+']['S1+']['Quad'][:,3], marker='.', linestyle='', c=clr, ms=ms)
        plt.annotate(r'$S_1 > 0$', xy=(x2,0.9), xycoords='data', fontsize=8)
        plt.annotate(r'$S_1 < 0$', xy=(x6,0.9), xycoords='data', fontsize=8)
        plt.title(r'$S_1 \cdot S_2 > 0$', fontsize=8)

        clr = c4
        plt.bar([x5,x6,x7,x8], np.array([frac_cells_dSp_S1n[0],frac_cells_dSp_S1n[1],frac_cells_dSp_S1n[2],frac_cells_dSp_S1n[3]]),width=0.1,color=[clr,clr,clr,clr],edgecolor=[clr,clr,clr,clr], alpha=0.3)
        plt.plot(x5+0.1*np.random.rand(nfile)-0.05, frac_cells_sum['dS+']['S1-']['Quad'][:,0], marker='.', linestyle='', c=clr, ms=ms)
        plt.plot(x6+0.1*np.random.rand(nfile)-0.05, frac_cells_sum['dS+']['S1-']['Quad'][:,1], marker='.', linestyle='', c=clr, ms=ms)
        plt.plot(x7+0.1*np.random.rand(nfile)-0.05, frac_cells_sum['dS+']['S1-']['Quad'][:,2], marker='.', linestyle='', c=clr, ms=ms)
        plt.plot(x8+0.1*np.random.rand(nfile)-0.05, frac_cells_sum['dS+']['S1-']['Quad'][:,3], marker='.', linestyle='', c=clr, ms=ms)
        q1_name = r'$Q1$' + '\n' + r'$\Delta P +$' + '\n' + r'$\Delta A +$'
        q2_name = r'$Q2$' + '\n' + r'$\Delta P -$' + '\n' + r'$\Delta A +$'
        q3_name = r'$Q3$' + '\n' + r'$\Delta P -$' + '\n' + r'$\Delta A -$'
        q4_name = r'$Q4$' + '\n' + r'$\Delta P -$' + '\n' + r'$\Delta A -$'
        q5_name = r'$Q1$' + '\n' + r'$\Delta P +$' + '\n' + r'$\Delta A +$'
        q6_name = r'$Q2$' + '\n' + r'$\Delta P -$' + '\n' + r'$\Delta A +$'
        q7_name = r'$Q3$' + '\n' + r'$\Delta P -$' + '\n' + r'$\Delta A -$'
        q8_name = r'$Q4$' + '\n' + r'$\Delta P -$' + '\n' + r'$\Delta A -$'
        plt.xticks([x1,x2,x3,x4,x5,x6,x7,x8],[q1_name, q2_name, q3_name, q4_name, q5_name, q6_name, q7_name, q8_name],fontsize=6)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.ylim([0,1])
        plt.ylabel('frac of cells', fontsize=8)
        plt.tight_layout()

        if savefig:
            if not shuffled:
                plt.savefig('figure/learned_activity/session_all/frac_with_signs_quadrant.pdf')
            if shuffled:
                plt.savefig('figure/learned_activity/shuffled/frac_with_signs_quadrant_shuffled.pdf')
            plt.close()


    def plot_dP_vs_dA_scatter(self, savefig, shuffled):
        if not shuffled:
            learned_activity_sum = self.learned_activity_sum
        if shuffled:
            learned_activity_sum = self.shuff_learned_activity_sum

        c1, c2, c3, c4 = self.c1, self.c2, self.c3, self.c4

        xlim = 0.5
        plt.figure(figsize=(3.0,1.8))
        plt.suptitle(r'$S_1 \cdot S_2 < 0$', fontsize=8, y=0.85)
        plt.subplot(121)
        plt.plot(learned_activity_sum['dS-']['S1+']['dP'], learned_activity_sum['dS-']['S1+']['dA'], marker='.', linestyle='', c=c1, mfc='None', mew=0.3, label=r'$S1>0$', ms=4)
        # plt.plot(learned_activity_sum['dS-']['S1-']['dP'], learned_activity_sum['dS-']['S1-']['dA'], marker='.', linestyle='', c=c2, mfc='None', mew=0.3, label=r'$S1<0$', ms=4)
        plt.axhline(0, color='gray', linestyle='--', lw=0.5)
        plt.axvline(0, color='gray', linestyle='--', lw=0.5)
        plt.xlabel(r'$\Delta P$')
        plt.ylabel(r'$\Delta A$')
        plt.xticks([-xlim,0,xlim])
        plt.yticks([-xlim,0,xlim])
        plt.xlim([-xlim,xlim])
        plt.ylim([-xlim,xlim])
        legend_elements = [Line2D([0], [0], marker='o', color=c1, label=r'$S_1>0$', linestyle='None', ms=3)]
        plt.legend(frameon=False, handles=legend_elements, loc=3, fontsize=6)
        
        plt.subplot(122)
        # plt.plot(learned_activity_sum['dS-']['S1+']['dP'], learned_activity_sum['dS-']['S1+']['dA'], marker='.', linestyle='', c=c1, mfc='None', mew=0.3, label=r'$S1>0$', ms=4)
        plt.plot(learned_activity_sum['dS-']['S1-']['dP'], learned_activity_sum['dS-']['S1-']['dA'], marker='.', linestyle='', c=c2, mfc='None', mew=0.3, label=r'$S1<0$', ms=4)
        plt.axhline(0, color='gray', linestyle='--', lw=0.5)
        plt.axvline(0, color='gray', linestyle='--', lw=0.5)
        plt.xlabel(r'$\Delta P$')
        # plt.ylabel(r'$\Delta A$')
        plt.xticks([-xlim,0,xlim])
        plt.yticks([-xlim,0,xlim],['','',''])
        plt.xlim([-xlim,xlim])
        plt.ylim([-xlim,xlim])
        legend_elements = [Line2D([0], [0], marker='o', color=c2, label=r'$S_1<0$', linestyle='None', ms=3)]
        plt.legend(frameon=False, handles=legend_elements, loc=3, fontsize=6)                
        plt.tight_layout()

        if savefig:
            if not shuffled:
                plt.savefig('figure/learned_activity/session_all/scatter_dP_vs_dA.pdf')
            if shuffled:
                plt.savefig('figure/learned_activity/shuffled/scatter_dP_vs_dA_shuffled.pdf')
            plt.close()

    def plot_dR_vs_dL_scatter(self, savefig, shuffled):
        if not shuffled:            
            learned_activity_sum = self.learned_activity_sum
        if shuffled:
            learned_activity_sum = self.shuff_learned_activity_sum

        c1, c2, c3, c4 = self.c1, self.c2, self.c3, self.c4

        xlim = 0.5
        plt.figure(figsize=(3.0,1.8))
        plt.suptitle(r'$S_1 \cdot S_2 > 0$', fontsize=8, y=0.85)
        plt.subplot(121)
        plt.plot(learned_activity_sum['dS+']['S1+']['dR'], learned_activity_sum['dS+']['S1+']['dL'], marker='.', linestyle='', c=c3, mfc='None', mew=0.3, label=r'$S1>0$', ms=4)
        # plt.plot(learned_activity_sum['dS+']['S1-']['dR'], learned_activity_sum['dS+']['S1-']['dL'], marker='.', linestyle='', c=c4, mfc='None', mew=0.3, label=r'$S1<0$', ms=4)
        plt.axhline(0, color='gray', linestyle='--', lw=0.5)
        plt.axvline(0, color='gray', linestyle='--', lw=0.5)
        plt.tight_layout()
        plt.xlabel(r'$\Delta R$')
        plt.ylabel(r'$\Delta L$')
        plt.xticks([-xlim,0,xlim])
        plt.yticks([-xlim,0,xlim])
        plt.xlim([-xlim,xlim])
        plt.ylim([-xlim,xlim])
        # plt.title(r'$S_1 \cdot S_2>0$', fontsize=8)
        legend_elements = [Line2D([0], [0], marker='o', color=c3, label=r'$S_1>0$', linestyle='None', ms=3)]
        plt.legend(frameon=False, handles=legend_elements, loc=3, fontsize=6)

        plt.subplot(122)
        # plt.plot(learned_activity_sum['dS+']['S1+']['dR'], learned_activity_sum['dS+']['S1+']['dL'], marker='.', linestyle='', c=c3, mfc='None', mew=0.3, label=r'$S1>0$', ms=4)
        plt.plot(learned_activity_sum['dS+']['S1-']['dR'], learned_activity_sum['dS+']['S1-']['dL'], marker='.', linestyle='', c=c4, mfc='None', mew=0.3, label=r'$S1<0$', ms=4)
        plt.axhline(0, color='gray', linestyle='--', lw=0.5)
        plt.axvline(0, color='gray', linestyle='--', lw=0.5)
        plt.tight_layout()
        plt.xlabel(r'$\Delta R$')
        # plt.ylabel(r'$\Delta L$')
        plt.xticks([-xlim,0,xlim])
        plt.yticks([-xlim,0,xlim],['','',''])
        plt.xlim([-xlim,xlim])
        plt.ylim([-xlim,xlim])
        # plt.title(r'$S_1 \cdot S_2>0$', fontsize=8)
        legend_elements = [Line2D([0], [0], marker='o', color=c4, label=r'$S_1<0$', linestyle='None', ms=3)]
        plt.legend(frameon=False, handles=legend_elements, loc=3, fontsize=6)

        plt.tight_layout()

        if savefig:
            if not shuffled:
                plt.savefig('figure/learned_activity/session_all/scatter_dR_vs_dL.pdf')
            if shuffled:
                plt.savefig('figure/learned_activity/shuffled/scatter_dR_vs_dL_shuffled.pdf')
            plt.close()

    def plot_dP_vs_dA_sessions(self, savefig):
        learned_activity = self.learned_activity
        CD_dotproduct = self.CD_dotproduct
        sorted_indices_by_relearn_speed = self.sorted_indices_by_relearn_speed
        merged_ID = self.merged_ID

        c1, c2, c3, c4 = self.c1, self.c2, self.c3, self.c4

        file_sorted = np.argsort(CD_dotproduct)
        xlim = 0.5
        for ii, fx in enumerate(file_sorted):    
            plt.figure(figsize=(1.8,1.8))
            fx_sorted = sorted_indices_by_relearn_speed[fx]
            plt.plot(learned_activity[f'sess{fx}']['dS-']['S1+']['dP'], learned_activity[f'sess{fx}']['dS-']['S1+']['dA'], marker='.', linestyle='', c=c1, mfc='None', mew=0.3, label=r'$S1>0$', ms=4)
            plt.plot(learned_activity[f'sess{fx}']['dS-']['S1-']['dP'], learned_activity[f'sess{fx}']['dS-']['S1-']['dA'], marker='.', linestyle='', c=c2, mfc='None', mew=0.3, label=r'$S1<0$', ms=4)
            plt.axhline(0, color='gray', linestyle='--', lw=0.5)
            plt.axvline(0, color='gray', linestyle='--', lw=0.5)
            plt.title(merged_ID[fx_sorted][6:16] + '\n' + r'$CD_1 \cdot CD_2$ =' + str(np.round(CD_dotproduct[fx],decimals=2)), fontsize=8)
            # correlation of dP vs dA
            _dP = np.append(learned_activity[f'sess{fx}']['dS-']['S1+']['dP'],learned_activity[f'sess{fx}']['dS-']['S1-']['dP'])
            _dA = np.append(learned_activity[f'sess{fx}']['dS-']['S1+']['dA'],learned_activity[f'sess{fx}']['dS-']['S1-']['dA'])
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
            
            if savefig:
                plt.savefig('figure/learned_activity/session_each/dP_dA/' + str(ii) + '_dP_vs_dA_sessions.pdf')
                plt.close()

    def plot_dR_vs_dL_sessions(self, savefig):
        learned_activity = self.learned_activity
        CD_dotproduct = self.CD_dotproduct
        sorted_indices_by_relearn_speed = self.sorted_indices_by_relearn_speed
        merged_ID = self.merged_ID

        c1, c2, c3, c4 = self.c1, self.c2, self.c3, self.c4

        file_sorted = np.argsort(CD_dotproduct)
        xlim = 0.3
        for ii, fx in enumerate(file_sorted):    
            plt.figure(figsize=(1.8,1.8))
            fx_sorted = sorted_indices_by_relearn_speed[fx]
            plt.plot(learned_activity[f'sess{fx}']['dS+']['S1+']['dR'], learned_activity[f'sess{fx}']['dS+']['S1+']['dL'], marker='.', linestyle='', c=c3, mfc='None', mew=0.3, label=r'$S1>0$', ms=4)
            plt.plot(learned_activity[f'sess{fx}']['dS+']['S1-']['dR'], learned_activity[f'sess{fx}']['dS+']['S1-']['dL'], marker='.', linestyle='', c=c4, mfc='None', mew=0.3, label=r'$S1<0$', ms=4)
            plt.axhline(0, color='gray', linestyle='--', lw=0.5)
            plt.axvline(0, color='gray', linestyle='--', lw=0.5)
            plt.title(merged_ID[fx_sorted][6:16] + '\n' + r'$CD_1 \cdot CD_2$ =' + str(np.round(CD_dotproduct[fx],decimals=2)), fontsize=8)
            # correlation of dR vs dL
            _dR = np.append(learned_activity[f'sess{fx}']['dS+']['S1+']['dR'],learned_activity[f'sess{fx}']['dS+']['S1-']['dR'])
            _dL = np.append(learned_activity[f'sess{fx}']['dS+']['S1+']['dL'],learned_activity[f'sess{fx}']['dS+']['S1-']['dL'])
            _cor = np.corrcoef(_dR,_dL)[0,1]
            plt.annotate(r'$\rho$ =' + str(np.round(_cor,decimals=3)), xy=(0.02,0.05), xycoords='axes fraction', fontsize=6)
            plt.xticks([-xlim,0,xlim])
            plt.yticks([-xlim,0,xlim])
            plt.xlim([-xlim,xlim])
            plt.ylim([-xlim,xlim])
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xlabel(r'$\Delta R$', fontsize=8)
            plt.ylabel(r'$\Delta L$', fontsize=8) 
            plt.tight_layout()

            if savefig:
                plt.savefig('figure/learned_activity/session_each/dR_dL/' + str(ii) + '_dR_vs_dL_sessions.pdf')
                plt.close()

    def plot_correlation_activity(self, savefig, shuffled):
        if not shuffled:
            learned_activity = self.learned_activity
            CD_dotproduct = self.CD_dotproduct
        if shuffled:
            learned_activity = self.shuff_learned_activity
            CD_dotproduct = self.shuff_CD_dotproduct            
        nfile = self.nfile

        c1, c2, c3, c4 = self.c1, self.c2, self.c3, self.c4

        corr_dP_vs_dA = np.zeros(nfile)
        corr_dR_vs_dL = np.zeros(nfile)
        _dP = np.array([])
        _dA = np.array([])
        _dR = np.array([])
        _dL = np.array([])
        for fx in range(nfile):
            _dP = np.append(learned_activity[f'sess{fx}']['dS-']['S1+']['dP'], learned_activity[f'sess{fx}']['dS-']['S1-']['dP'])
            _dA = np.append(learned_activity[f'sess{fx}']['dS-']['S1+']['dA'], learned_activity[f'sess{fx}']['dS-']['S1-']['dA'])
            _dR = np.append(learned_activity[f'sess{fx}']['dS+']['S1+']['dR'], learned_activity[f'sess{fx}']['dS+']['S1-']['dR'])
            _dL = np.append(learned_activity[f'sess{fx}']['dS+']['S1+']['dL'], learned_activity[f'sess{fx}']['dS+']['S1-']['dL'])
            corr_dP_vs_dA[fx] = np.corrcoef(_dP,_dA)[0,1] 
            corr_dR_vs_dL[fx] = np.corrcoef(_dR,_dL)[0,1] 

        plt.figure(figsize=(1.8,1.5))
        plt.plot(CD_dotproduct, corr_dP_vs_dA, marker='o', linestyle='', c='k', ms=4, mfc='None', mew=0.5)
        plt.xlabel(r'$CD_1 \cdot CD_2$')
        plt.ylabel(r'corr of $\Delta P$ and $\Delta A$')
        _corr = np.corrcoef(CD_dotproduct, corr_dP_vs_dA)[0,1]
        plt.annotate(r'$\rho = $' + str(np.round(_corr,decimals=3)), xy=(0.4,0.05), xycoords='axes fraction', fontsize=8)
        plt.xlim([-1,0.5])
        plt.ylim([-1,0.4])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        if savefig:
            if not shuffled:
                plt.savefig('figure/learned_activity/correlation_activity/corr_dP_vs_dA.pdf')
                plt.close()
            elif shuffled:
                plt.savefig('figure/learned_activity/shuffled/corr_dP_vs_dA_shuffled.pdf')
                plt.close()                

        plt.figure(figsize=(1.8,1.5))
        plt.plot(CD_dotproduct, corr_dR_vs_dL, marker='o', linestyle='', c='k', ms=4, mfc='None', mew=0.5)
        plt.xlabel(r'$CD_1 \cdot CD_2$')
        plt.ylabel(r'corr of $\Delta R$ and $\Delta L$')
        _corr = np.corrcoef(CD_dotproduct, corr_dR_vs_dL)[0,1]
        plt.annotate(r'$\rho = $' + str(np.round(_corr,decimals=3)), xy=(0.4,0.05), xycoords='axes fraction', fontsize=8)
        plt.xlim([-1,0.5])
        plt.ylim([-1,0.4])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        if savefig:
            if not shuffled:
                plt.savefig('figure/learned_activity/correlation_activity/corr_dR_vs_dL.pdf')
                plt.close()
            elif shuffled: 
                plt.savefig('figure/learned_activity/shuffled/corr_dR_vs_dL_shuffled.pdf')
                plt.close()
                
            
        plt.figure(figsize=(1.8,1.5))
        plt.plot(corr_dP_vs_dA, corr_dR_vs_dL, marker='o', linestyle='', c='k', ms=4, mfc='None', mew=0.5)
        plt.xlabel(r'corr of $\Delta P$ and $\Delta A$')
        plt.ylabel(r'corr of $\Delta R$ and $\Delta L$')
        _corr = np.corrcoef(corr_dP_vs_dA, corr_dR_vs_dL)[0,1]
        plt.annotate(r'$\rho = $' + str(np.round(_corr,decimals=3)), xy=(0.05,0.05), xycoords='axes fraction', fontsize=8)        
        plt.xlim([-1,0.5])
        plt.ylim([-1,0.4])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        
        if savefig:
            if not shuffled:
                plt.savefig('figure/learned_activity/correlation_activity/corr_dPdA_vs_dRdL.pdf')
                plt.close()
            elif shuffled: 
                plt.savefig('figure/learned_activity/shuffled/corr_dPdA_vs_dRdL_shuffled.pdf')
                plt.close()
        

    def plot_correlation_frac_cells(self, savefig, shuffled):
        if not shuffled:
            frac_cells = self.frac_cells
            CD_dotproduct = self.CD_dotproduct
        if shuffled:
            frac_cells = self.shuff_frac_cells
            CD_dotproduct = self.shuff_CD_dotproduct            
        nfile = self.nfile

        c1, c2, c3, c4 = self.c1, self.c2, self.c3, self.c4

        corr_dP_vs_dA = np.zeros(nfile)
        frac_dP = np.zeros((2,nfile))
        frac_dA = np.zeros((2,nfile))
        frac_dR = np.zeros((2,nfile))
        frac_dL = np.zeros((2,nfile))
        for fx in range(nfile):
            frac_dP[0,fx] = frac_cells[f'sess{fx}']['dS-']['S1+']['dP<0']
            frac_dA[0,fx] = frac_cells[f'sess{fx}']['dS-']['S1+']['dA>0']
            frac_dP[1,fx] = frac_cells[f'sess{fx}']['dS-']['S1-']['dP>0']
            frac_dA[1,fx] = frac_cells[f'sess{fx}']['dS-']['S1-']['dA<0']

            frac_dR[0,fx] = frac_cells[f'sess{fx}']['dS+']['S1+']['dR<0']
            frac_dL[0,fx] = frac_cells[f'sess{fx}']['dS+']['S1+']['dL>0']
            frac_dR[1,fx] = frac_cells[f'sess{fx}']['dS+']['S1-']['dR>0']
            frac_dL[1,fx] = frac_cells[f'sess{fx}']['dS+']['S1-']['dL<0']

        dP_dA0 = np.append(frac_dP[0],frac_dA[0])
        dP_dA1 = np.append(frac_dP[1],frac_dA[1])
        dR_dL0 = np.append(frac_dR[0],frac_dL[0])
        dR_dL1 = np.append(frac_dR[1],frac_dL[1])
        _CDdotprod = np.append(CD_dotproduct,CD_dotproduct)
        corr_dP_dA_CD1_pos = np.corrcoef(_CDdotprod, dP_dA0)[0,1]
        corr_dP_dA_CD1_neg = np.corrcoef(_CDdotprod, dP_dA1)[0,1]
        corr_dR_dL_CD1_pos = np.corrcoef(_CDdotprod, dR_dL0)[0,1]
        corr_dR_dL_CD1_neg = np.corrcoef(_CDdotprod, dR_dL1)[0,1]

        plt.figure(figsize=(2.1,1.5))
        plt.plot(CD_dotproduct, frac_dP[0], marker='o', linestyle='', c=c1, ms=3, mfc='lightgray', mew=0.5)
        plt.plot(CD_dotproduct, frac_dP[0], marker='o', linestyle='', c=c1, ms=3, mfc='None', mew=0.5)
        plt.plot(CD_dotproduct, frac_dA[0], marker='o', linestyle='', c=c1, ms=3, mfc='None', mew=0.5)
        plt.annotate(r'$\rho = $' + str(np.round(corr_dP_dA_CD1_pos,decimals=3)), xy=(0.05,0.05), xycoords='axes fraction', fontsize=8)
        plt.xlabel(r'$CD_1 \cdot CD_2$')
        plt.ylabel('frac of cells')
        plt.xticks([-1,0,0.5])
        plt.xlim([-1,0.5])
        plt.ylim([0.3,1.02])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        legend_elements = [Line2D([0], [0], marker='o', color=c1, label=r'$\Delta P<0$', linestyle='None', ms=3, mfc='None', mew=0.5), Line2D([0], [0], marker='o', mec=c1, mfc='lightgray', label=r'$\Delta A>0$', linestyle='None', ms=3, mew=0.5)]
        plt.legend(frameon=False, handles=legend_elements, fontsize=6, bbox_to_anchor=[1,0.8])
        plt.tight_layout()

        if savefig:
            if not shuffled:
                plt.savefig('figure/learned_activity/frac_cells/frac_dCD_vs_dP.pdf')
                plt.close()
            if shuffled:
                plt.savefig('figure/learned_activity/shuffled/frac_dCD_vs_dP_shuffled.pdf')
                plt.close()
                                

        plt.figure(figsize=(2.1,1.5))
        plt.plot(CD_dotproduct, frac_dP[1], marker='o', linestyle='', c=c2, ms=3, mfc='lightgray', mew=0.5)
        plt.plot(CD_dotproduct, frac_dP[1], marker='o', linestyle='', c=c2, ms=3, mfc='None', mew=0.5)
        plt.plot(CD_dotproduct, frac_dA[1], marker='o', linestyle='', c=c2, ms=3, mfc='None', mew=0.5)
        plt.annotate(r'$\rho = $' + str(np.round(corr_dP_dA_CD1_neg,decimals=3)), xy=(0.05,0.05), xycoords='axes fraction', fontsize=8)
        plt.xlabel(r'$CD_1 \cdot CD_2$')
        plt.ylabel('frac of cells')
        plt.xticks([-1,0,0.5])
        plt.xlim([-1,0.5])
        plt.ylim([0.3,1.02])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        legend_elements = [Line2D([0], [0], marker='o', color=c2, label=r'$\Delta P>0$', linestyle='None', ms=3, mfc='None', mew=0.5), Line2D([0], [0], marker='o', mec=c2, mfc='lightgray', label=r'$\Delta A<0$', linestyle='None', ms=3, mew=0.5)]
        plt.legend(frameon=False, handles=legend_elements, fontsize=6, bbox_to_anchor=[1,0.8])
        plt.tight_layout()

        if savefig:
            if not shuffled:
                plt.savefig('figure/learned_activity/frac_cells/frac_dCD_vs_dA.pdf')
                plt.close()
            elif shuffled:
                plt.savefig('figure/learned_activity/shuffled/frac_dCD_vs_dA_shuffled.pdf')
                plt.close()
           

        plt.figure(figsize=(2.1,1.5))
        plt.plot(CD_dotproduct, frac_dR[0], marker='o', linestyle='', c=c3, ms=3, mfc='lightgray', mew=0.5)
        plt.plot(CD_dotproduct, frac_dR[0], marker='o', linestyle='', c=c3, ms=3, mfc='None', mew=0.5)
        plt.plot(CD_dotproduct, frac_dL[0], marker='o', linestyle='', c=c3, ms=3, mfc='None', mew=0.5)
        plt.annotate(r'$\rho = $' + str(np.round(corr_dR_dL_CD1_pos,decimals=3)), xy=(0.05,0.05), xycoords='axes fraction', fontsize=8)
        plt.xlabel(r'$CD_1 \cdot CD_2$')
        plt.ylabel('frac of cells')
        plt.xticks([-1,0,0.5])
        plt.xlim([-1,0.5])
        plt.ylim([0.3,1.02])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        legend_elements = [Line2D([0], [0], marker='o', color=c3, label=r'$\Delta R<0$', linestyle='None', ms=3, mfc='None', mew=0.5), Line2D([0], [0], marker='o', mec=c3, mfc='lightgray', label=r'$\Delta L>0$', linestyle='None', ms=3, mew=0.5)]
        plt.legend(frameon=False, handles=legend_elements, fontsize=6, bbox_to_anchor=[1,0.8])
        plt.tight_layout()
        
        if savefig:
            if not shuffled:
                plt.savefig('figure/learned_activity/frac_cells/frac_dCD_vs_dR.pdf')
                plt.close()
            elif shuffled:
                plt.savefig('figure/learned_activity/shuffled/frac_dCD_vs_dR_shuffled.pdf')
                plt.close()
                
        plt.figure(figsize=(2.1,1.5))
        plt.plot(CD_dotproduct, frac_dR[1], marker='o', linestyle='', c=c4, ms=3, mfc='lightgray', mew=0.5)
        plt.plot(CD_dotproduct, frac_dR[1], marker='o', linestyle='', c=c4, ms=3, mfc='None', mew=0.5)
        plt.plot(CD_dotproduct, frac_dL[1], marker='o', linestyle='', c=c4, ms=3, mfc='None', mew=0.5)
        plt.annotate(r'$\rho = $' + str(np.round(corr_dR_dL_CD1_neg,decimals=3)), xy=(0.05,0.05), xycoords='axes fraction', fontsize=8)
        plt.xlabel(r'$CD_1 \cdot CD_2$')
        plt.ylabel('frac of cells')
        plt.xticks([-1,0,0.5])
        plt.xlim([-1,0.5])
        plt.ylim([0.3,1.02])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        legend_elements = [Line2D([0], [0], marker='o', color=c4, label=r'$\Delta R>0$', linestyle='None', ms=3, mfc='None', mew=0.5), Line2D([0], [0], marker='o', mec=c4, mfc='lightgray', label=r'$\Delta L<0$', linestyle='None', ms=3, mew=0.5)]
        plt.legend(frameon=False, handles=legend_elements, fontsize=6, bbox_to_anchor=[1,0.8])
        plt.tight_layout()
           
        if savefig:
            if not shuffled:
                plt.savefig('figure/learned_activity/frac_cells/frac_dCD_vs_dL.pdf')
                plt.close()
            if shuffled:
                plt.savefig('figure/learned_activity/shuffled/frac_dCD_vs_dL_shuffled.pdf')
                plt.close()
                

    def plot_shuffled_data_scatter_jet(self, savefig):
        
        learned_activity_sum       = self.learned_activity_sum
        shuff_learned_activity_sum = self.shuff_learned_activity_sum
        
        ###############################
        # scatter plot - original data
        ###############################
        plt.figure(figsize=(3.5,2.0))
        plt.suptitle(r'$S_1 \cdot S_2 < 0$', fontsize=8, y=0.9)
        plt.subplot(121)
        plt.hist2d(
            learned_activity_sum['dS-']['S1+']['dP'], learned_activity_sum['dS-']['S1+']['dA'],
            bins=100,
            cmap='jet',
            range = [[-0.3,0.3],[-0.3,0.3]],
            norm=LogNorm()
        )
        plt.axhline(0, color='gray', linestyle='--', lw=0.5)
        plt.axvline(0, color='gray', linestyle='--', lw=0.5)
        plt.xlabel(r'$\Delta P$', fontsize=8)
        plt.ylabel(r'$\Delta A$', fontsize=8)
        plt.title(r'$S_1 > 0$', fontsize=8)
        # plt.colorbar()

        plt.subplot(122)
        plt.hist2d(
            learned_activity_sum['dS-']['S1-']['dP'], learned_activity_sum['dS-']['S1-']['dA'],
            bins=100,
            cmap='jet',
            range = [[-0.3,0.3],[-0.3,0.3]],
            norm=LogNorm()
        )
        plt.axhline(0, color='gray', linestyle='--', lw=0.5)
        plt.axvline(0, color='gray', linestyle='--', lw=0.5)
        plt.xlabel(r'$\Delta P$', fontsize=8)
        plt.ylabel(r'$\Delta A$', fontsize=8)
        plt.title(r'$S_1 < 0$', fontsize=8)
        # plt.colorbar()
        plt.tight_layout()
        if savefig:
            plt.savefig('figure/learned_activity/shuffled/jet_dP_vs_dA_original.pdf')
            plt.close()


        plt.figure(figsize=(3.5,2.0))
        plt.suptitle(r'$S_1 \cdot S_2 > 0$', fontsize=8, y=0.9)
        plt.subplot(121)
        plt.hist2d(
            learned_activity_sum['dS+']['S1+']['dR'], learned_activity_sum['dS+']['S1+']['dL'],
            bins=100,
            cmap='jet',
            range = [[-0.3,0.3],[-0.3,0.3]],
            norm=LogNorm()
        )
        plt.axhline(0, color='gray', linestyle='--', lw=0.5)
        plt.axvline(0, color='gray', linestyle='--', lw=0.5)
        plt.xlabel(r'$\Delta R$', fontsize=8)
        plt.ylabel(r'$\Delta L$', fontsize=8)
        plt.title(r'$S_1 > 0$', fontsize=8)
        # plt.colorbar()

        plt.subplot(122)
        plt.hist2d(
            learned_activity_sum['dS+']['S1-']['dR'], learned_activity_sum['dS+']['S1-']['dL'],
            bins=100,
            cmap='jet',
            range = [[-0.3,0.3],[-0.3,0.3]],
            norm=LogNorm()
        )
        plt.axhline(0, color='gray', linestyle='--', lw=0.5)
        plt.axvline(0, color='gray', linestyle='--', lw=0.5)
        plt.xlabel(r'$\Delta R$', fontsize=8)
        plt.ylabel(r'$\Delta L$', fontsize=8)
        plt.title(r'$S_1 < 0$', fontsize=8)
        # plt.colorbar()
        plt.tight_layout()

        if savefig:
            plt.savefig('figure/learned_activity/shuffled/jet_dR_vs_dL_original.pdf')
            plt.close()

        ###############################
        # scatter plot - shuffled data
        ###############################
        plt.figure(figsize=(3.5,2.0))
        plt.suptitle(r'$S_1 \cdot S_2 < 0$', fontsize=8, y=0.9)
        plt.subplot(121)
        plt.hist2d(
            shuff_learned_activity_sum['dS-']['S1+']['dP'], shuff_learned_activity_sum['dS-']['S1+']['dA'],
            bins=100,
            cmap='jet',
            range = [[-0.3,0.3],[-0.3,0.3]],
            norm=LogNorm()
        )
        plt.axhline(0, color='gray', linestyle='--', lw=0.5)
        plt.axvline(0, color='gray', linestyle='--', lw=0.5)
        plt.xlabel(r'$\Delta P$', fontsize=8)
        plt.ylabel(r'$\Delta A$', fontsize=8)
        plt.title(r'$S_1 > 0$', fontsize=8)
        # plt.colorbar()

        plt.subplot(122)
        plt.hist2d(
            shuff_learned_activity_sum['dS-']['S1-']['dP'], shuff_learned_activity_sum['dS-']['S1-']['dA'],
            bins=100,
            cmap='jet',
            range = [[-0.3,0.3],[-0.3,0.3]],
            norm=LogNorm()
        )
        plt.axhline(0, color='gray', linestyle='--', lw=0.5)
        plt.axvline(0, color='gray', linestyle='--', lw=0.5)
        plt.xlabel(r'$\Delta P$', fontsize=8)
        plt.ylabel(r'$\Delta A$', fontsize=8)
        plt.title(r'$S_1 < 0$', fontsize=8)
        # plt.colorbar()
        plt.tight_layout()
        if savefig:
            plt.savefig('figure/learned_activity/shuffled/jet_dP_vs_dA_shuffled.pdf')
            plt.close()


        plt.figure(figsize=(3.5,2.0))
        plt.suptitle(r'$S_1 \cdot S_2 > 0$', fontsize=8, y=0.9)
        plt.subplot(121)
        plt.hist2d(
            shuff_learned_activity_sum['dS+']['S1+']['dR'], shuff_learned_activity_sum['dS+']['S1+']['dL'],
            bins=100,
            cmap='jet',
            range = [[-0.3,0.3],[-0.3,0.3]],
            norm=LogNorm()
        )
        plt.axhline(0, color='gray', linestyle='--', lw=0.5)
        plt.axvline(0, color='gray', linestyle='--', lw=0.5)
        plt.xlabel(r'$\Delta R$', fontsize=8)
        plt.ylabel(r'$\Delta L$', fontsize=8)
        plt.title(r'$S_1 > 0$', fontsize=8)
        # plt.colorbar()

        plt.subplot(122)
        plt.hist2d(
            shuff_learned_activity_sum['dS+']['S1-']['dR'], shuff_learned_activity_sum['dS+']['S1-']['dL'],
            bins=100,
            cmap='jet',
            range = [[-0.3,0.3],[-0.3,0.3]],
            norm=LogNorm()
        )
        plt.axhline(0, color='gray', linestyle='--', lw=0.5)
        plt.axvline(0, color='gray', linestyle='--', lw=0.5)
        plt.xlabel(r'$\Delta R$', fontsize=8)
        plt.ylabel(r'$\Delta L$', fontsize=8)
        plt.title(r'$S_1 < 0$', fontsize=8)
        # plt.colorbar()
        plt.tight_layout()        
        plt.tight_layout()
        if savefig:
            plt.savefig('figure/learned_activity/shuffled/jet_dR_vs_dL_shuffled.pdf')
            plt.close()
        

    def plot_synthetic_data(self, savefig):
        
        shift_in_trial2, \
        type1_dP, type1_dA, type2_dP, type2_dA, type3_dR, type3_dL, type4_dR, type4_dL, \
        type1_frac_in_quad, type2_frac_in_quad, type3_frac_in_quad, type4_frac_in_quad = self.synthetic_data

        ########################
        # scatter plot
        ########################
        xlim = 0.3
        ms = 0.3
        plt.figure(figsize=(2.8,1.7))
        plt.suptitle(r'$S_1S_2 < 0$', fontsize=8, y=0.9)
        plt.subplot(121)
        plt.plot(type1_dP, type1_dA, marker='.', c='purple', linestyle='', ms=ms)
        plt.axhline(0,color='gray',linestyle='--', lw=0.5)
        plt.axvline(0,color='gray',linestyle='--', lw=0.5)
        plt.xlim([-xlim,xlim])
        plt.ylim([-xlim,xlim])
        plt.xlabel(r'$\Delta P$', fontsize=8)
        plt.ylabel(r'$\Delta A$', fontsize=8)
        plt.xticks([-xlim,0,xlim])
        plt.yticks([-xlim,0,xlim])
        plt.title(r'$S_1 > 0$', fontsize=8)
        plt.subplot(122)
        plt.plot(type2_dP, type2_dA, marker='.', c='blue', linestyle='', ms=ms)
        plt.axhline(0,color='gray',linestyle='--', lw=0.5)
        plt.axvline(0,color='gray',linestyle='--', lw=0.5)
        plt.xlim([-xlim,xlim])
        plt.ylim([-xlim,xlim])
        plt.xlabel(r'$\Delta P$', fontsize=8)
        plt.ylabel(r'$\Delta A$', fontsize=8)
        plt.xticks([-xlim,0,xlim])
        plt.yticks([-xlim,0,xlim])
        plt.title(r'$S_1 < 0$', fontsize=8)
        plt.tight_layout()
        
        if savefig:
            plt.savefig('figure/learned_activity/shuffled/synthetic_dP_dA_shift' + str(shift_in_trial2) + '.pdf')
            plt.close()

        plt.figure(figsize=(2.8,1.7))
        plt.suptitle(r'$S_1S_2 < 0$', fontsize=8, y=0.9)
        plt.subplot(121)
        plt.plot(type3_dR, type3_dL, marker='.', c='darkorange', linestyle='', ms=ms)
        plt.axhline(0,color='gray',linestyle='--', lw=0.5)
        plt.axvline(0,color='gray',linestyle='--', lw=0.5)
        plt.xlim([-xlim,xlim])
        plt.ylim([-xlim,xlim])
        plt.xlabel(r'$\Delta P$', fontsize=8)
        plt.ylabel(r'$\Delta A$', fontsize=8)
        plt.xticks([-xlim,0,xlim])
        plt.yticks([-xlim,0,xlim])
        plt.title(r'$S_1 > 0$', fontsize=8)
        plt.subplot(122)
        plt.plot(type4_dR, type4_dL, marker='.', c='limegreen', linestyle='', ms=ms)
        plt.axhline(0,color='gray',linestyle='--', lw=0.5)
        plt.axvline(0,color='gray',linestyle='--', lw=0.5)
        plt.xlim([-xlim,xlim])
        plt.ylim([-xlim,xlim])
        plt.xlabel(r'$\Delta P$', fontsize=8)
        plt.ylabel(r'$\Delta A$', fontsize=8)
        plt.xticks([-xlim,0,xlim])
        plt.yticks([-xlim,0,xlim])
        plt.title(r'$S_1 < 0$', fontsize=8)
        plt.tight_layout()

        if savefig:
            plt.savefig('figure/learned_activity/shuffled/synthetic_dR_dL_shift' + str(shift_in_trial2) + '.pdf')
            plt.close()
                
        ################################
        # fraction of cells in quad
        ################################
        c1, c2, c3, c4 = 'darkviolet', 'blue', 'darkorange', 'limegreen'
        x1, x2, x3, x4 = 0.2, 0.4, 0.6, 0.8
        x5, x6, x7, x8 = 1.2, 1.4, 1.6, 1.8
        ms = 1
        plt.figure(figsize=(2.8,2.5))
        plt.subplot(211)
        clr = c1
        plt.bar([x1,x2,x3,x4], np.array([type1_frac_in_quad[0],type1_frac_in_quad[1],type1_frac_in_quad[2],type1_frac_in_quad[3]]),width=0.1,color=[clr,clr,clr,clr],edgecolor=[clr,clr,clr,clr], alpha=0.3)
        plt.annotate(r'$S_1 > 0$', xy=(x2,0.9), xycoords='data', fontsize=8)
        plt.annotate(r'$S_1 < 0$', xy=(x6,0.9), xycoords='data', fontsize=8)
        plt.title(r'$S_1 \cdot S_2 < 0$', fontsize=8)
        clr = c2
        plt.bar([x5,x6,x7,x8], np.array([type2_frac_in_quad[0],type2_frac_in_quad[1],type2_frac_in_quad[2],type2_frac_in_quad[3]]),width=0.1,color=[clr,clr,clr,clr],edgecolor=[clr,clr,clr,clr], alpha=0.3)
        q1_name = r'$Q1$'
        q2_name = r'$Q2$'
        q3_name = r'$Q3$'
        q4_name = r'$Q4$'
        q5_name = r'$Q1$'
        q6_name = r'$Q2$'
        q7_name = r'$Q3$'
        q8_name = r'$Q4$'
        plt.xticks([x1,x2,x3,x4,x5,x6,x7,x8],[q1_name, q2_name, q3_name, q4_name, q5_name, q6_name, q7_name, q8_name],fontsize=6)
        plt.ylim([0,1])
        plt.ylabel('frac of cells', fontsize=8)
        plt.subplot(212)
        clr = c3
        plt.bar([x1,x2,x3,x4], np.array([type3_frac_in_quad[0],type3_frac_in_quad[1],type3_frac_in_quad[2],type3_frac_in_quad[3]]),width=0.1,color=[clr,clr,clr,clr],edgecolor=[clr,clr,clr,clr], alpha=0.3)
        plt.annotate(r'$S_1 > 0$', xy=(x2,0.9), xycoords='data', fontsize=8)
        plt.annotate(r'$S_1 < 0$', xy=(x6,0.9), xycoords='data', fontsize=8)
        plt.title(r'$S_1 \cdot S_2 > 0$', fontsize=8)
        clr = c4
        plt.bar([x5,x6,x7,x8], np.array([type4_frac_in_quad[0],type4_frac_in_quad[1],type4_frac_in_quad[2],type4_frac_in_quad[3]]),width=0.1,color=[clr,clr,clr,clr],edgecolor=[clr,clr,clr,clr], alpha=0.3)
        q1_name = r'$Q1$' + '\n' + r'$\Delta P +$' + '\n' + r'$\Delta A +$'
        q2_name = r'$Q2$' + '\n' + r'$\Delta P -$' + '\n' + r'$\Delta A +$'
        q3_name = r'$Q3$' + '\n' + r'$\Delta P -$' + '\n' + r'$\Delta A -$'
        q4_name = r'$Q4$' + '\n' + r'$\Delta P -$' + '\n' + r'$\Delta A -$'
        q5_name = r'$Q1$' + '\n' + r'$\Delta P +$' + '\n' + r'$\Delta A +$'
        q6_name = r'$Q2$' + '\n' + r'$\Delta P -$' + '\n' + r'$\Delta A +$'
        q7_name = r'$Q3$' + '\n' + r'$\Delta P -$' + '\n' + r'$\Delta A -$'
        q8_name = r'$Q4$' + '\n' + r'$\Delta P -$' + '\n' + r'$\Delta A -$'
        plt.xticks([x1,x2,x3,x4,x5,x6,x7,x8],[q1_name, q2_name, q3_name, q4_name, q5_name, q6_name, q7_name, q8_name],fontsize=6)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.ylim([0,1])
        plt.ylabel('frac of cells', fontsize=8)
        plt.tight_layout()

        if savefig:
            plt.savefig('figure/learned_activity/shuffled/synthetic_frac_with_signs_quadrant_shift' + str(shift_in_trial2) + '.pdf')
            plt.close()
            
    def plot_check_neuraldata_is_gaussian(self, savefig):
        
        nfile = self.nfile
        P1_set2 = self.P1_set2
        P2_set2 = self.P2_set2
        A1_set2 = self.A1_set2
        A2_set2 = self.A2_set2

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
        # CV of means
        P1_mean_CV = np.std(P1_mean) / np.abs(np.mean(P1_mean))
        P2_mean_CV = np.std(P2_mean) / np.abs(np.mean(P2_mean))
        A1_mean_CV = np.std(A1_mean) / np.abs(np.mean(A1_mean))
        A2_mean_CV = np.std(A2_mean) / np.abs(np.mean(A2_mean))
        # CV of stds
        P1_std_CV = np.std(P1_std) / np.abs(np.mean(P1_std))
        P2_std_CV = np.std(P2_std) / np.abs(np.mean(P2_std))
        A1_std_CV = np.std(A1_std) / np.abs(np.mean(A1_std))
        A2_std_CV = np.std(A2_std) / np.abs(np.mean(A2_std))

        #################################
        # Show neural data is log normal
        # histogram
        #################################
        nbins = 100
        activity_all = ['P1', 'P2', 'A1', 'A2']
        for act in activity_all:
            if act == 'P1':
                data = P1_set2
            elif act == 'P2':
                data = P2_set2
            elif act == 'A1':
                data = A1_set2
            elif act == 'A2':
                data = A2_set2
                
            plt.figure(figsize=(6,6))
            for fx in range(nfile):
                plt.subplot(6,6,fx+1)
                idx = data[fx] > 0
                data_fx = np.log10(data[fx][idx])
                mu = np.mean(data_fx)
                sig = np.std(data_fx)
                xx = np.linspace(np.min(data_fx),np.max(data_fx),100)
                plt.hist(data_fx, bins=nbins, range=(-4,-0.5), histtype='step', density=True, lw=1.5)
                plt.plot(xx, norm.pdf(xx,mu,sig),c='r', lw=0.8)
                if fx == 30:
                    plt.xlabel('log (activity)')
                    plt.ylabel('density')
                if fx == 0:
                    plt.title('sess' + str(fx))
                else:
                    plt.title(str(fx))
                plt.xticks([-4,-2])
                plt.xlim([-4,0])
            plt.tight_layout()
            
            if savefig:
                plt.savefig('figure/learned_activity/shuffled/check_gaussian_' + act + '_hist.pdf')
                plt.close()



        from scipy import stats

        #################################
        # Show neural data is log normal
        # Q-Q plot
        #################################
        for act in activity_all:
            if act == 'P1':
                data = P1_set2
            elif act == 'P2':
                data = P2_set2
            elif act == 'A1':
                data = A1_set2
            elif act == 'A2':
                data = A2_set2
                
            plt.figure(figsize=(6,6))
            for fx in range(nfile):
                ax = plt.subplot(6,6,fx+1)
                idx = data[fx] > 0
                xx = np.log10(data[fx][idx])
                res = stats.probplot(xx, dist="norm", plot=plt, rvalue=False)
                R2 = res[1][2]
                plt.annotate('R2:' + str(np.round(R2,decimals=3)),xy=(0.2,0.05),xycoords='axes fraction', fontsize=6)
                if fx != 30:
                    plt.xlabel('')
                    plt.ylabel('')
                else:
                    plt.ylabel('data values')
                plt.xticks([-2,0,2])
                if fx == 0:
                    plt.title('session' + str(fx), fontsize=8)
                else:
                    plt.title(str(fx), fontsize=8)
                # first line = points
                ax.lines[0].set_markersize(1)
                ax.lines[0].set_markeredgewidth(0.5)
                # second line = regression line
                ax.lines[1].set_linewidth(1)    
            plt.tight_layout()
            
            if savefig:
                plt.savefig('figure/learned_activity/shuffled/check_gaussian_' + act + '_qq.pdf')
                plt.close()
            
        #################################
        # distribution of session means
        #################################
        plt.figure(figsize=(3,1.5))
        plt.subplot(121)
        plt.hist(P1_mean, bins=50, range=(-4,-0.5), histtype='step', color='purple', label='P1')
        plt.hist(P2_mean, bins=50, range=(-4,-0.5), histtype='step', color='blue', label='P2')
        plt.annotate('P1 CV:' + str(np.round(P1_mean_CV,decimals=2)),xy=(0.02,0.8),xycoords='axes fraction', fontsize=6)
        plt.annotate('P2 CV:' + str(np.round(P2_mean_CV,decimals=2)),xy=(0.02,0.7),xycoords='axes fraction', fontsize=6)
        plt.legend(fontsize=6, frameon=False)
        plt.xlim([-4,0])
        plt.xlabel('session mean')
        plt.ylabel('session count')
        plt.subplot(122)
        plt.hist(A1_mean, bins=50, range=(-4,-0.5), histtype='step', color='limegreen', label='A1')
        plt.hist(A2_mean, bins=50, range=(-4,-0.5), histtype='step', color='darkorange', label='A2')
        plt.annotate('A1 CV:' + str(np.round(A1_mean_CV,decimals=2)),xy=(0.02,0.8),xycoords='axes fraction', fontsize=6)
        plt.annotate('A2 CV:' + str(np.round(A2_mean_CV,decimals=2)),xy=(0.02,0.7),xycoords='axes fraction', fontsize=6)
        plt.legend(fontsize=6, frameon=False)
        plt.xlim([-4,0])
        plt.tight_layout()

        if savefig:
            plt.savefig('figure/learned_activity/shuffled/check_gaussian_session_mean.pdf')
            plt.close()

        #################################
        # distribution of session std
        #################################
        plt.figure(figsize=(3,1.5))
        plt.subplot(121)
        plt.hist(P1_std, bins=50, range=(0,1), histtype='step', color='purple', label='P1')
        plt.hist(P2_std, bins=50, range=(0,1), histtype='step', color='blue', label='P2')
        plt.annotate('P1 CV:' + str(np.round(P1_std_CV,decimals=2)),xy=(0.02,0.8),xycoords='axes fraction', fontsize=6)
        plt.annotate('P2 CV:' + str(np.round(P2_std_CV,decimals=2)),xy=(0.02,0.7),xycoords='axes fraction', fontsize=6)
        plt.legend(fontsize=6, frameon=False)
        # plt.xlim([-4,0])
        plt.xlabel('session std')
        plt.ylabel('session count')
        plt.subplot(122)
        plt.hist(A1_std, bins=50, range=(0,1), histtype='step', color='limegreen', label='A1')
        plt.hist(A2_std, bins=50, range=(0,1), histtype='step', color='darkorange', label='A2')
        plt.annotate('A1 CV:' + str(np.round(A1_std_CV,decimals=2)),xy=(0.02,0.8),xycoords='axes fraction', fontsize=6)
        plt.annotate('A2 CV:' + str(np.round(A2_std_CV,decimals=2)),xy=(0.02,0.7),xycoords='axes fraction', fontsize=6)
        plt.legend(fontsize=6, frameon=False)
        # plt.xlim([-4,0])
        plt.tight_layout()
        
        if savefig:
            plt.savefig('figure/learned_activity/shuffled/check_gaussian_session_std.pdf')
            plt.close()
        


    