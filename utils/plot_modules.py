import numpy as np
import matplotlib.pyplot as plt
from utils import functions
import os
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl

import numpy as np
import matplotlib.pyplot as plt


class ModulePlots:

    def __init__(
        self,
        figpath,
        nfile,
        data_nonoutlier,
        dict_topcells_1x2,
        CDdotproduct,
        dict_CDdp_err,
        dict_module_activity,
        tix_late_delay,
        sequential_P1,
        sequential_A1,
        sequential_P2,
        sequential_A2,
        dict_within_selectivity,
        dict_within_selectivity_neurons,
        dict_across_activity,
        dict_across_activity_neurons,
        dict_across_context_and_trial_activity,
        keys_within_context,
        keys_across_context,
        keys1,
        keys2,
    ):

        self.figpath = figpath
        self.nfile = nfile
        self.data_nonoutlier = data_nonoutlier
        # self.multi_data_nonoutlier = multi_data_nonoutlier
        # self.multi_CDdotproduct = multi_CDdotproduct
        # self.multi_module_activity = multi_module_activity
        # self.multi_within_selectivity = multi_within_selectivity
        # self.multi_across_activity = multi_across_activity
        
        self.dict_topcells_1x2 = dict_topcells_1x2
        self.CDdotproduct = CDdotproduct
        self.dict_CDdp_err = dict_CDdp_err
        self.dict_module_activity = dict_module_activity
        self.tix_late_delay = tix_late_delay

        self.sequential_P1 = sequential_P1
        self.sequential_A1 = sequential_A1
        self.sequential_P2 = sequential_P2
        self.sequential_A2 = sequential_A2

        self.dict_within_selectivity = dict_within_selectivity
        self.dict_across_activity = dict_across_activity
        self.dict_across_context_and_trial_activity = dict_across_context_and_trial_activity

        self.dict_within_selectivity_neurons = dict_within_selectivity_neurons
        self.dict_across_activity_neurons = dict_across_activity_neurons


        self.keys_within_context = keys_within_context
        self.keys_across_context = keys_across_context

        if keys1 is None:
            keys1 = ['P1+A1-', 'P1+A1+', 'P1-A1+']

        if keys2 is None:
            keys2 = ['P2+A2-', 'P2+A2+', 'P2-A2+']

        self.keys1 = keys1
        self.keys2 = keys2


    # ==============================================================
    # module size
    # ==============================================================

    def plot_module_size(self, savefig):

        figpath = self.figpath
        nfile = self.nfile
        data_nonoutlier = self.data_nonoutlier
        dict_topcells_1x2 = self.dict_topcells_1x2
        CDdotproduct = self.CDdotproduct
        keys1 = self.keys1
        keys2 = self.keys2

        bar_width = 0.6
        titles1 = [r'$P_1^+A_1^-$', r'$P_1^+A_1^+$', r'$P_1^-A_1^+$']
        labels2 = [r'$P_2^+A_2^-$', r'$P_2^+A_2^+$', r'$P_2^-A_2^+$']

        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(
            3, 3,
            hspace=0.5,
            wspace=1
        )

        for i2, key2 in enumerate(keys2):
            for i1, key1 in enumerate(keys1):

                frac_cells = np.zeros(nfile)

                for fx in range(nfile):
                    ntotal_cells = data_nonoutlier[fx]['P1'].shape[0]

                    frac_cells[fx] = (
                        len(dict_topcells_1x2[fx][key1][key2])
                        / ntotal_cells
                    )

                frac_cells_mean = np.mean(frac_cells)
                frac_cells_std = np.std(frac_cells)

                ax1 = fig.add_subplot(gs[i2, i1])

                plt.bar(
                    0,
                    frac_cells_mean,
                    width=bar_width,
                    color='gray',
                    alpha=0.5
                )

                jitter = 0.1 * (2 * np.random.rand(len(frac_cells)) - 1)

                plt.scatter(
                    0 + jitter,
                    frac_cells,
                    s=12,
                    facecolors='none',
                    edgecolors='gray',
                    linewidths=0.5,
                    alpha=1
                )

                plt.errorbar(
                    0,
                    frac_cells_mean,
                    yerr=frac_cells_std / np.sqrt(len(frac_cells)),
                    color='k',
                    capsize=8,
                    lw=1.5,
                    linestyle='none'
                )

                plt.xlim([-0.5, 0.5])
                plt.ylim([0.0, 0.1])

                if i2 == 0:
                    ax1.set_title(titles1[i1], fontsize=12)

                if i1 == 0:
                    ax1.set_ylabel(labels2[i2], fontsize=12)

                ax1.set_xticks([-0.5, 0, 0.5])
                ax1.set_xticklabels([])
                ax1.set_yticks([0, 0.05, 0.1])

                plt.gca().spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        if savefig:
            plt.savefig(figpath + 'module_sizes.pdf')


        # ----------------------------
        # total module size
        # ----------------------------

        total_frac_cells = np.zeros(nfile)
        total_number_cells = np.zeros(nfile)

        for fx in range(nfile):

            ntotal_cells = data_nonoutlier[fx]['P1'].shape[0]

            for i2, key2 in enumerate(keys2):
                for i1, key1 in enumerate(keys1):

                    total_frac_cells[fx] += (
                        len(dict_topcells_1x2[fx][key1][key2])
                        / ntotal_cells
                    )

                    total_number_cells[fx] += (
                        len(dict_topcells_1x2[fx][key1][key2])
                    )

        total_frac_cells_mean = np.mean(total_frac_cells)
        total_frac_cells_std = np.std(total_frac_cells)

        total_number_cells_mean = np.mean(total_number_cells)
        total_number_cells_std = np.std(total_number_cells)


        plt.figure(figsize=(2, 2))

        plt.bar(
            0,
            total_frac_cells_mean,
            width=bar_width,
            color='gray',
            alpha=0.5
        )

        jitter = 0.1 * (2 * np.random.rand(nfile) - 1)

        plt.scatter(
            0 + jitter,
            total_frac_cells,
            s=12,
            facecolors='none',
            edgecolors='gray',
            linewidths=0.5,
            alpha=1
        )

        plt.errorbar(
            0,
            total_frac_cells_mean,
            yerr=total_frac_cells_std / np.sqrt(nfile),
            color='k',
            capsize=8,
            lw=1.5,
            linestyle='none'
        )

        plt.xticks([-0.5, 0, 0.5], [])
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.ylabel('frac of neurons')
        plt.tight_layout()

        if savefig:
            plt.savefig(figpath + 'module_sizes_total_frac.pdf')


        plt.figure(figsize=(2, 2))

        plt.bar(
            0,
            total_number_cells_mean,
            width=bar_width,
            color='gray',
            alpha=0.5
        )

        jitter = 0.1 * (2 * np.random.rand(nfile) - 1)

        plt.scatter(
            0 + jitter,
            total_number_cells,
            s=12,
            facecolors='none',
            edgecolors='gray',
            linewidths=0.5,
            alpha=1
        )

        plt.errorbar(
            0,
            total_number_cells_mean,
            yerr=total_number_cells_std / np.sqrt(nfile),
            color='k',
            capsize=8,
            lw=1.5,
            linestyle='none'
        )

        plt.xticks([-0.5, 0, 0.5], [])
        plt.yticks([0, 100, 200, 300, 400])
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.ylabel('# of neurons')
        plt.tight_layout()

        if savefig:
            plt.savefig(figpath + 'module_sizes_total_number.pdf')


        _cor = np.corrcoef(
            CDdotproduct,
            total_number_cells
        )[0, 1]

        plt.figure(figsize=(2, 2))

        plt.scatter(
            CDdotproduct,
            total_frac_cells,
            fc='None',
            ec='k'
        )

        plt.xlabel('CD dot product')
        plt.ylabel('frac of neurons')

        plt.gca().spines[['top', 'right']].set_visible(False)

        plt.title(
            'corr ' + str(np.round(_cor, decimals=3))
        )

        plt.tight_layout()

        if savefig:
            plt.savefig(figpath + 'module_sizes_vs_CDdotprod.pdf')

        # save for later use if needed
        self.total_frac_cells = total_frac_cells
        self.total_number_cells = total_number_cells


    # ==============================================================
    # approximation of CD dot product
    # ==============================================================

    def plot_CDdotprod_error(self, savefig):

        figpath = self.figpath
        dict_CDdp_err = self.dict_CDdp_err
        
        plt.figure(figsize=(8, 4))

        keys = list(dict_CDdp_err.keys())
        x = np.arange(len(keys))

        bar_width = 0.6

        xlabels = [
            r'$P_1^+A_1^-$',
            r'$P_1^+A_1^+$',
            r'$P_1^-A_1^+$',
            r'$P_1^+A_1^- + P_1^+A_1^+$',
            r'$P_1^-A_1^+ + P_1^+A_1^+$',
            r'$P_1^+A_1^- + P_1^-A_1^+$',
            'all',
            r'$P_2^+A_2^-$',
            r'$P_2^+A_2^+$',
            r'$P_2^-A_2^+$',
            r'$P_2^+A_2^- + P_2^+A_2^+$',
            r'$P_2^-A_2^+ + P_2^+A_2^+$',
            r'$P_2^+A_2^- + P_2^-A_2^+$'
        ]

        for i, key in enumerate(keys):

            y = dict_CDdp_err[key]

            plt.bar(
                i,
                np.mean(y),
                width=bar_width,
                color='lightgray',
                edgecolor='k',
                zorder=1
            )

            plt.errorbar(
                i,
                np.mean(y),
                yerr=np.std(y) / np.sqrt(len(y)),
                color='k',
                capsize=2,
                lw=1,
                zorder=2
            )

            jitter = 0.1 * (2 * np.random.rand(len(y)) - 1)

            plt.scatter(
                i + jitter,
                y,
                s=12,
                facecolors='none',
                edgecolors='k',
                linewidths=0.6,
                zorder=3
            )

            if i in [4, 6, 10]:

                plt.annotate(
                    '',
                    xy=(i, 0.5),
                    xytext=(i, 0.65),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color='red',
                        lw=2.5
                    )
                )

        plt.xticks(
            x,
            xlabels,
            rotation=45,
            ha='right'
        )

        plt.ylabel(r'$\Delta$ CD dot product')

        plt.tight_layout()

        if savefig:
            plt.savefig(figpath + 'CDdotprod_error.pdf')


    # ==============================================================
    # neural activity of modules
    # ==============================================================

    def plot_module_activity(self, savefig):

        figpath = self.figpath
        dict_module_activity = self.dict_module_activity
        data_nonoutlier = self.data_nonoutlier
        tix_late_delay = self.tix_late_delay
        keys1 = self.keys1
        keys2 = self.keys2
        keys_within_context = self.keys_within_context 
        nfile = self.nfile
        
        bar_width = 0.6

        titles1 = [
            r'$P_1^+A_1^-$',
            r'$P_1^+A_1^+$',
            r'$P_1^-A_1^+$'
        ]

        labels2 = [
            r'$P_2^+A_2^-$',
            r'$P_2^+A_2^+$',
            r'$P_2^-A_2^+$'
        ]


        # ----------------------------------------------------------
        # within context
        # ----------------------------------------------------------

        fig = plt.figure(figsize=(6, 6))

        gs = fig.add_gridspec(
            3, 3,
            hspace=0.3,
            wspace=0.5
        )

        for i2, key2 in enumerate(keys2):
            for i1, key1 in enumerate(keys1):

                diff1 = np.stack([
                    0 if len(self.dict_topcells_1x2[fx][key1][key2]) == 0 else
                    np.mean((data_nonoutlier[fx]['P1'] - data_nonoutlier[fx]['A1'])[self.dict_topcells_1x2[fx][key1][key2]][:,tix_late_delay])
                    for fx in range(nfile)
                ])
                diff2 = np.stack([
                    0 if len(self.dict_topcells_1x2[fx][key1][key2]) == 0 else
                    np.mean((data_nonoutlier[fx]['P2'] - data_nonoutlier[fx]['A2'])[self.dict_topcells_1x2[fx][key1][key2]][:,tix_late_delay])
                    for fx in range(nfile)
                ])

                diff1_mean = np.nanmean(diff1)
                diff2_mean = np.nanmean(diff2)

                diff1_std = np.nanstd(diff1)
                diff2_std = np.nanstd(diff2)

                ymax = np.max(np.concatenate((diff1, diff2))) + 0.02
                ymin = np.min(np.concatenate((diff1, diff2))) - 0.02                

                if diff1_mean > 0:
                    c1 = 'purple'
                else:
                    c1 = 'limegreen'

                if diff2_mean > 0:
                    c2 = 'purple'
                else:
                    c2 = 'limegreen'

                ax1 = fig.add_subplot(gs[i2, i1])

                plt.bar(
                    [0, 1],
                    [diff1_mean, diff2_mean],
                    width=bar_width,
                    color=[c1, c2],
                    edgecolor=[c1, c2],
                    zorder=1,
                    alpha=0.5
                )

                jitter = 0.1 * (2 * np.random.rand(len(diff1)) - 1)

                plt.scatter(
                    0 + jitter,
                    diff1,
                    s=12,
                    facecolors='none',
                    edgecolors=c1,
                    linewidths=0.5,
                    alpha=1
                )

                plt.scatter(
                    1 + jitter,
                    diff2,
                    s=12,
                    facecolors='none',
                    edgecolors=c2,
                    linewidths=0.5,
                    alpha=0.8
                )

                plt.errorbar(
                    [0, 1],
                    [diff1_mean, diff2_mean],
                    yerr=[
                        diff1_std / np.sqrt(len(diff1)),
                        diff2_std / np.sqrt(len(diff2))
                    ],
                    color='k',
                    capsize=8,
                    lw=1.5,
                    linestyle='none'
                )

                plt.xlim([-0.5, 1.5])
                plt.ylim([ymin, ymax])

                if i2 == 0:
                    ax1.set_title(titles1[i1], fontsize=12)

                if i1 == 0:
                    ax1.set_ylabel(labels2[i2], fontsize=12)

                if i2 == 2:
                    ax1.set_xticks([0, 1])
                    ax1.set_xticklabels(
                        ['Context 1', 'Context 2'],
                        rotation=45
                    )
                else:
                    ax1.set_xticks([0, 1])
                    ax1.set_xticklabels([])

        plt.tight_layout()

        if savefig:
            plt.savefig(figpath + 'module_within_context.pdf')


        # ----------------------------------------------------------
        # across context
        # ----------------------------------------------------------

        fig = plt.figure(figsize=(6, 6))

        gs = fig.add_gridspec(
            3, 3,
            hspace=0.3,
            wspace=0.5
        )

        for i2, key2 in enumerate(keys2):
            for i1, key1 in enumerate(keys1):

                diff1 = np.stack([
                    0 if len(self.dict_topcells_1x2[fx][key1][key2]) == 0 else
                    np.mean((data_nonoutlier[fx]['P2'] - data_nonoutlier[fx]['P1'])[self.dict_topcells_1x2[fx][key1][key2]][:,tix_late_delay])
                    for fx in range(nfile)
                ])
                diff2 = np.stack([
                    0 if len(self.dict_topcells_1x2[fx][key1][key2]) == 0 else
                    np.mean((data_nonoutlier[fx]['A2'] - data_nonoutlier[fx]['A1'])[self.dict_topcells_1x2[fx][key1][key2]][:,tix_late_delay])
                    for fx in range(nfile)
                ])

                diff1_mean = np.nanmean(diff1)
                diff2_mean = np.nanmean(diff2)

                diff1_std = np.nanstd(diff1)
                diff2_std = np.nanstd(diff2)

                ymax = np.max(np.concatenate((diff1, diff2))) + 0.02
                ymin = np.min(np.concatenate((diff1, diff2))) - 0.02

                if diff1_mean > 0:
                    c1 = 'tab:cyan'
                else:
                    c1 = 'gray'

                if diff2_mean > 0:
                    c2 = 'tab:cyan'
                else:
                    c2 = 'gray'

                ax1 = fig.add_subplot(gs[i2, i1])

                plt.bar(
                    [0, 1],
                    [diff1_mean, diff2_mean],
                    width=bar_width,
                    color=[c1, c2],
                    edgecolor=[c1, c2],
                    zorder=1,
                    alpha=0.5
                )

                jitter = 0.1 * (2 * np.random.rand(len(diff1)) - 1)

                plt.scatter(
                    0 + jitter,
                    diff1,
                    s=12,
                    facecolors='none',
                    edgecolors=c1,
                    linewidths=0.5,
                    alpha=1
                )

                plt.scatter(
                    1 + jitter,
                    diff2,
                    s=12,
                    facecolors='none',
                    edgecolors=c2,
                    linewidths=0.5,
                    alpha=1
                )

                plt.errorbar(
                    [0, 1],
                    [diff1_mean, diff2_mean],
                    yerr=[
                        diff1_std / np.sqrt(len(diff1)),
                        diff2_std / np.sqrt(len(diff2))
                    ],
                    color='k',
                    capsize=8,
                    lw=1.5,
                    linestyle='none'
                )

                plt.xlim([-0.5, 1.5])
                plt.ylim([ymin, ymax])

                if i2 == 0:
                    ax1.set_title(titles1[i1], fontsize=12)

                if i1 == 0:
                    ax1.set_ylabel(labels2[i2], fontsize=12)

                if i2 == 2:
                    ax1.set_xticks([0, 1])
                    ax1.set_xticklabels(
                        [r'$P_2-P_1$', r'$A_2-A_1$'],
                        rotation=45
                    )
                else:
                    ax1.set_xticks([0, 1])
                    ax1.set_xticklabels([])

        plt.tight_layout()

        if savefig:
            plt.savefig(figpath + 'module_cross_context.pdf')


        # # ----------------------------------------------------------
        # # across context and trial
        # # ----------------------------------------------------------

        # fig = plt.figure(figsize=(6, 6))

        # gs = fig.add_gridspec(
        #     3, 3,
        #     hspace=0.3,
        #     wspace=0.5
        # )

        # for i2, key2 in enumerate(keys2):
        #     for i1, key1 in enumerate(keys1):

        #         diff1 = np.nanmean(
        #             dict_module_activity[key1][key2]['P2'][:, tix_late_delay]
        #             - dict_module_activity[key1][key2]['A1'][:, tix_late_delay],
        #             axis=1
        #         )

        #         diff2 = np.nanmean(
        #             dict_module_activity[key1][key2]['A2'][:, tix_late_delay]
        #             - dict_module_activity[key1][key2]['P1'][:, tix_late_delay],
        #             axis=1
        #         )

        #         diff1_mean = np.nanmean(diff1)
        #         diff2_mean = np.nanmean(diff2)

        #         diff1_std = np.nanstd(diff1)
        #         diff2_std = np.nanstd(diff2)

        #         ymax = np.max(np.concatenate((diff1, diff2))) + 0.02
        #         ymin = np.min(np.concatenate((diff1, diff2))) - 0.02

        #         if diff1_mean > 0:
        #             c1 = 'tab:olive'
        #         else:
        #             c1 = 'gray'

        #         if diff2_mean > 0:
        #             c2 = 'tab:olive'
        #         else:
        #             c2 = 'gray'

        #         ax1 = fig.add_subplot(gs[i2, i1])

        #         plt.bar(
        #             [0, 1],
        #             [diff1_mean, diff2_mean],
        #             width=bar_width,
        #             color=[c1, c2],
        #             edgecolor=[c1, c2],
        #             zorder=1,
        #             alpha=0.5
        #         )

        #         jitter = 0.1 * (2 * np.random.rand(len(diff1)) - 1)

        #         plt.scatter(
        #             0 + jitter,
        #             diff1,
        #             s=12,
        #             facecolors='none',
        #             edgecolors=c1,
        #             linewidths=0.5,
        #             alpha=1
        #         )

        #         plt.scatter(
        #             1 + jitter,
        #             diff2,
        #             s=12,
        #             facecolors='none',
        #             edgecolors=c2,
        #             linewidths=0.5,
        #             alpha=1
        #         )

        #         plt.errorbar(
        #             [0, 1],
        #             [diff1_mean, diff2_mean],
        #             yerr=[
        #                 diff1_std / np.sqrt(len(diff1)),
        #                 diff2_std / np.sqrt(len(diff2))
        #             ],
        #             color='k',
        #             capsize=8,
        #             lw=1.5,
        #             linestyle='none'
        #         )

        #         plt.xlim([-0.5, 1.5])
        #         plt.ylim([ymin, ymax])

        #         if i2 == 0:
        #             ax1.set_title(titles1[i1], fontsize=12)

        #         if i1 == 0:
        #             ax1.set_ylabel(labels2[i2], fontsize=12)

        #         if i2 == 2:
        #             ax1.set_xticks([0, 1])
        #             ax1.set_xticklabels(
        #                 [r'$P_2-A_1$', r'$A_2-P_1$'],
        #                 rotation=45
        #             )
        #         else:
        #             ax1.set_xticks([0, 1])
        #             ax1.set_xticklabels([])

        # plt.tight_layout()

        # if savefig:
        #     plt.savefig(figpath + 'module_cross_context_and_trial.pdf')


        # ----------------------------------------------------------
        # module traces
        # ----------------------------------------------------------

        fig = plt.figure(figsize=(6, 8))

        gs = fig.add_gridspec(
            6, 3,
            hspace=0.3,
            wspace=0.5
        )

        for i2, key2 in enumerate(keys2):
            for i1, key1 in enumerate(keys1):

                ax1 = fig.add_subplot(gs[2 * i2, i1])

                activity_all_P1 = np.stack([
                    np.nanmean(
                        data_nonoutlier[fx]['P1'][self.dict_topcells_1x2[fx][key1][key2], :],
                        axis=0
                    )
                    for fx in range(nfile)
                ])
                activity_all_A1 = np.stack([
                    np.nanmean(
                        data_nonoutlier[fx]['A1'][self.dict_topcells_1x2[fx][key1][key2], :],
                        axis=0
                    )
                    for fx in range(nfile)
                ])                
                mean_activity_P1 = np.nanmean(activity_all_P1, axis=0)
                mean_activity_A1 = np.nanmean(activity_all_A1, axis=0)
                
                ax1.plot(mean_activity_P1,c='purple')
                ax1.plot(mean_activity_A1,c='limegreen')

                ax1.axvline(7, color='gray', linestyle='--')
                ax1.axvline(15, color='gray', linestyle='--')

                # ax1.set_ylim([0, 0.3])
                ax1.set_xticklabels([])

                ax2 = fig.add_subplot(
                    gs[2 * i2 + 1, i1],
                    sharex=ax1
                )


                activity_all_P2 = np.stack([
                    np.nanmean(
                        data_nonoutlier[fx]['P2'][self.dict_topcells_1x2[fx][key1][key2], :],
                        axis=0
                    )
                    for fx in range(nfile)
                ])
                activity_all_A2 = np.stack([
                    np.nanmean(
                        data_nonoutlier[fx]['A2'][self.dict_topcells_1x2[fx][key1][key2], :],
                        axis=0
                    )
                    for fx in range(nfile)
                ])                
                mean_activity_P2 = np.nanmean(activity_all_P2, axis=0)
                mean_activity_A2 = np.nanmean(activity_all_A2, axis=0)
                
                ax2.plot(mean_activity_P2,c='purple',linestyle='--')
                ax2.plot(mean_activity_A2,c='limegreen',linestyle='--')                

                ax2.axvline(7, color='gray', linestyle='--')
                ax2.axvline(15, color='gray', linestyle='--')

                # ax2.set_ylim([0, 0.3])

                if i2 == 0:
                    ax1.set_title(
                        titles1[i1],
                        fontsize=12
                    )

                if i1 == 0:
                    ax1.set_ylabel(
                        labels2[i2],
                        fontsize=12
                    )

                if i1 == 0 and i2 == 0:
                    ax1.annotate(
                        'Context1',
                        xy=(0.45, 0.8),
                        xycoords='axes fraction'
                    )

                    ax2.annotate(
                        'Context2',
                        xy=(0.45, 0.8),
                        xycoords='axes fraction'
                    )

        plt.tight_layout()

        if savefig:
            plt.savefig(figpath + 'module_traces.pdf')


        # # ----------------------------------------------------------
        # # module heatmaps (P1-A1, P2-A2)
        # # ----------------------------------------------------------

        # nfile = len(data_nonoutlier.keys())
        # y1save = np.zeros((3,3,nfile))
        # y2save = np.zeros((3,3,nfile))
        # for fx in range(nfile):
                
        #     fig = plt.figure(figsize=(10, 6))

        #     outer = fig.add_gridspec(
        #         3, 3,
        #         wspace=0.35,
        #         hspace=0.4
        #     )
        
        #     for i1, k1 in enumerate(keys1):
        #         for i2, k2 in enumerate(keys2):

        #             inner = outer[i2, i1].subgridspec(
        #                 1, 2,
        #                 wspace=0.08
        #             )

        #             y1 = (data_nonoutlier[fx]['P1'] - data_nonoutlier[fx]['A1'])[self.dict_topcells_1x2[fx][k1][k2],:]
        #             y2 = (data_nonoutlier[fx]['P2'] - data_nonoutlier[fx]['A2'])[self.dict_topcells_1x2[fx][k1][k2],:]                    
        #             y1_sorted = np.argsort(np.argmax(y1,axis=1))
                    
        #             for ic, kc in enumerate(keys_within_context):

        #                 ax = fig.add_subplot(inner[0, ic])

        #                 ax.spines[
        #                     ['top', 'right']
        #                 ].set_visible(False)

        #                 if ic == 0:
        #                     plt.imshow(y1[y1_sorted,:],cmap='PiYG_r',aspect='auto',vmin=-0.2,vmax=0.2,interpolation='None')
        #                     plt.axvline(8,color='k',linestyle='--',lw=0.5)
        #                     plt.axvline(16,color='k',linestyle='--',lw=0.5)
        #                     plt.annotate(str(np.round(np.mean(y1[:,tix_late_delay]),decimals=3)),xy=(0.5,0.8),xycoords='axes fraction')
        #                 if ic == 1:
        #                     plt.imshow(y2[y1_sorted,:],cmap='PiYG_r',aspect='auto',vmin=-0.2,vmax=0.2,interpolation='None')
        #                     plt.axvline(8,color='k',linestyle='--',lw=0.5)
        #                     plt.axvline(16,color='k',linestyle='--',lw=0.5)
        #                     plt.annotate(str(np.round(np.mean(y2[:,tix_late_delay]),decimals=3)),xy=(0.5,0.8),xycoords='axes fraction')
        #                     ax.set_yticklabels([])                            
        #                 if i2 == 0 and ic == 0:
        #                     ax.set_title(
        #                         titles1[i1],
        #                         fontsize=12
        #                     )

        #                 if i1 == 0 and ic == 0:
        #                     ax.set_ylabel(
        #                         labels2[i2],
        #                         fontsize=12
        #                     )                            
                            
        #                 if ic == 0:
        #                     y1save[i2,i1,fx] = np.mean(y1[:,tix_late_delay])
        #                 elif ic == 1:
        #                     y2save[i2,i1,fx] = np.mean(y2[:,tix_late_delay])

        #     fig.tight_layout()

        #     if savefig:
        #         plt.savefig(figpath + 'module_heatmaps/selectivity/' f'module_heatmaps_{fx}.png',dpi=600)
        #         plt.close()

        #     # module selectivity
        #     plt.figure(figsize=(6,6))
        #     for i1 in range(3):
        #         for i2 in range(3):
        #             plt.subplot(3,3,3*i2+i1+1)
                                        
        #             jitter = 0.1 * (2 * np.random.rand(len(diff1)) - 1)

        #             plt.scatter(
        #                 0 + jitter,
        #                 y1save[i2,i1],
        #                 s=12,
        #                 facecolors='none',
        #                 edgecolors='k',
        #                 linewidths=0.5,
        #                 alpha=1
        #             )

        #             plt.scatter(
        #                 1 + jitter,
        #                 y2save[i2,i1],
        #                 s=12,
        #                 facecolors='none',
        #                 edgecolors='k',
        #                 linewidths=0.5,
        #                 alpha=1
        #             )
        #             plt.axhline(0,color='gray',linestyle='--')
        #     plt.tight_layout()
            
        #     if savefig:
        #         plt.savefig(figpath + 'module_heatmaps/selectivity/module_selectivity.pdf')
        #         plt.close()




        # ----------------------------------------------------------
        # module heatmaps (P1,A1,P2,A2)
        # ----------------------------------------------------------

        nfile = len(data_nonoutlier.keys())
        
        for fx in range(nfile):
                
            for i3, k3 in enumerate(['P1A1','P2A2']):
                    
                fig = plt.figure(figsize=(10, 6))

                outer = fig.add_gridspec(
                    3, 3,
                    wspace=0.35,
                    hspace=0.4
                )
            
                for i1, k1 in enumerate(keys1):
                    for i2, k2 in enumerate(keys2):

                        inner = outer[i2, i1].subgridspec(
                            1, 2,
                            wspace=0.08
                        )

                        if k3 == 'P1A1':
                            y1 = data_nonoutlier[fx]['P1'][self.dict_topcells_1x2[fx][k1][k2],:]
                            y2 = data_nonoutlier[fx]['A1'][self.dict_topcells_1x2[fx][k1][k2],:]                    
                        elif k3 == 'P2A2':
                            y1 = data_nonoutlier[fx]['P2'][self.dict_topcells_1x2[fx][k1][k2],:]
                            y2 = data_nonoutlier[fx]['A2'][self.dict_topcells_1x2[fx][k1][k2],:]                    
                        sorted1 = np.argsort(np.argmax(y1,axis=1))
                        sorted2 = np.argsort(np.argmax(y2,axis=1))
                            
                        
                        for ic, kc in enumerate(keys_within_context):

                            ax = fig.add_subplot(inner[0, ic])

                            ax.spines[
                                ['top', 'right']
                            ].set_visible(False)

                            if ic == 0:
                                plt.imshow(y1[sorted1,:],cmap='jet',aspect='auto',vmin=0,vmax=0.2,interpolation='None')
                                plt.axvline(8,color='w',linestyle='--',lw=0.5)
                                plt.axvline(16,color='w',linestyle='--',lw=0.5)
                            if ic == 1:
                                plt.imshow(y2[sorted1,:],cmap='jet',aspect='auto',vmin=0,vmax=0.2,interpolation='None')
                                plt.axvline(8,color='w',linestyle='--',lw=0.5)
                                plt.axvline(16,color='w',linestyle='--',lw=0.5)
                                ax.set_yticklabels([])                            
                            if i2 == 0 and ic == 0:
                                ax.set_title(
                                    titles1[i1],
                                    fontsize=12
                                )

                            if i1 == 0 and ic == 0:
                                ax.set_ylabel(
                                    labels2[i2],
                                    fontsize=12
                                )                            

                fig.tight_layout()

                if savefig:
                    dirpath = figpath + f'module_heatmaps/trials/{fx}/'
                    os.makedirs(dirpath, exist_ok=True)
                    plt.savefig(dirpath + k3 + f'_module_heatmaps_{fx}.png',dpi=600)
                    plt.close()


    # ==============================================================
    # sequential activity of modules
    # ==============================================================

    def plot_sequential_activity(self, savefig=False):

        figpath = self.figpath
        keys1 = self.keys1
        keys2 = self.keys2

        sequential_P1 = self.sequential_P1
        sequential_A1 = self.sequential_A1
        sequential_P2 = self.sequential_P2
        sequential_A2 = self.sequential_A2

        for i1, k1 in enumerate(keys1):
            for i2, k2 in enumerate(keys2):

                sequential_P1_i2_i1 = sequential_P1[i2, i1]
                sequential_A1_i2_i1 = sequential_A1[i2, i1]
                sequential_P2_i2_i1 = sequential_P2[i2, i1]
                sequential_A2_i2_i1 = sequential_A2[i2, i1]

                vmax = np.max([
                    np.max(sequential_P1_i2_i1),
                    np.max(sequential_A1_i2_i1),
                    np.max(sequential_P2_i2_i1),
                    np.max(sequential_A2_i2_i1)
                ])

                plt.figure(figsize=(4.3, 3.8))

                plt.subplot(221)
                plt.title(r'$P_1$', fontsize=12)
                plt.imshow(
                    sequential_P1_i2_i1,
                    cmap='jet',
                    aspect='auto',
                    vmin=0,
                    vmax=vmax
                )
                plt.axvline(7, color='w', linestyle='--')
                plt.axvline(15, color='w', linestyle='--')
                plt.colorbar()

                plt.subplot(222)
                plt.title(r'$A_1$', fontsize=12)
                plt.imshow(
                    sequential_A1_i2_i1,
                    cmap='jet',
                    aspect='auto',
                    vmin=0,
                    vmax=vmax
                )
                plt.axvline(7, color='w', linestyle='--')
                plt.axvline(15, color='w', linestyle='--')
                plt.colorbar()

                plt.subplot(223)
                plt.title(r'$P_2$', fontsize=12)
                plt.imshow(
                    sequential_P2_i2_i1,
                    cmap='jet',
                    aspect='auto',
                    vmin=0,
                    vmax=vmax
                )
                plt.axvline(7, color='w', linestyle='--')
                plt.axvline(15, color='w', linestyle='--')
                plt.colorbar()

                plt.subplot(224)
                plt.title(r'$A_2$', fontsize=12)
                plt.imshow(
                    sequential_A2_i2_i1,
                    cmap='jet',
                    aspect='auto',
                    vmin=0,
                    vmax=vmax
                )
                plt.axvline(7, color='w', linestyle='--')
                plt.axvline(15, color='w', linestyle='--')
                plt.colorbar()

                plt.tight_layout()

                if savefig:
                    plt.savefig(figpath + f'sequential/map_{i2}_{i1}_' + k2 + '_' + k1 + '.pdf')


                cmap = plt.cm.copper
                colors = cmap(np.linspace(0, 1, 8))

                plt.figure(figsize=(3.3, 3))

                plt.subplot(221)
                plt.title(r'$P_1$', fontsize=12)

                for i in range(8):
                    plt.plot(
                        sequential_P1_i2_i1[i],
                        color=colors[i]
                    )

                plt.axvline(7, color='gray', linestyle='--')
                plt.axvline(15, color='gray', linestyle='--')
                plt.ylim([0, vmax])
                plt.gca().spines[['top', 'right']].set_visible(False)

                plt.subplot(222)
                plt.title(r'$A_1$', fontsize=12)

                for i in range(8):
                    plt.plot(
                        sequential_A1_i2_i1[i],
                        color=colors[i]
                    )

                plt.axvline(7, color='gray', linestyle='--')
                plt.axvline(15, color='gray', linestyle='--')
                plt.gca().spines[['top', 'right']].set_visible(False)
                plt.ylim([0, vmax])

                plt.subplot(223)
                plt.title(r'$P_2$', fontsize=12)

                for i in range(8):
                    plt.plot(
                        sequential_P2_i2_i1[i],
                        color=colors[i]
                    )

                plt.axvline(7, color='gray', linestyle='--')
                plt.axvline(15, color='gray', linestyle='--')
                plt.gca().spines[['top', 'right']].set_visible(False)
                plt.ylim([0, vmax])

                plt.subplot(224)
                plt.title(r'$A_2$', fontsize=12)

                for i in range(8):
                    plt.plot(
                        sequential_A2_i2_i1[i],
                        color=colors[i]
                    )

                plt.axvline(7, color='gray', linestyle='--')
                plt.axvline(15, color='gray', linestyle='--')
                plt.gca().spines[['top', 'right']].set_visible(False)
                plt.ylim([0, vmax])

                plt.tight_layout()

                if savefig:
                    plt.savefig(figpath + f'sequential/trace_{i2}_{i1}_' + k2 + '_' + k1 + '.pdf')


    # ==============================================================
    # CD dot product vs module activity
    # ==============================================================

    def plot_CDdotproduct_vs_module_activity(self, savefig):

        figpath = self.figpath
        CDdotproduct = self.CDdotproduct
        keys1 = self.keys1
        keys2 = self.keys2
        keys_within_context = self.keys_within_context
        keys_across_context = self.keys_across_context

        dict_within_selectivity = self.dict_within_selectivity
        dict_across_activity = self.dict_across_activity
        dict_across_context_and_trial_activity = self.dict_across_context_and_trial_activity

        dict_within_selectivity_neurons = self.dict_within_selectivity_neurons
        dict_across_activity_neurons = self.dict_across_activity_neurons

        # multipel datasets
        multi_within_selectivity = self.multi_within_selectivity
        multi_across_activity = self.multi_across_activity
        multi_CDdotproduct = self.multi_CDdotproduct

        # ----------------------------------------------------------
        # within context
        # ----------------------------------------------------------

        fig = plt.figure(figsize=(10, 6))

        outer = fig.add_gridspec(
            3, 3,
            wspace=0.35,
            hspace=0.4
        )

        for i1, k1 in enumerate(keys1):
            for i2, k2 in enumerate(keys2):

                inner = outer[i2, i1].subgridspec(
                    1, 2,
                    wspace=0.08
                )

                y1 = dict_within_selectivity[k1][k2]['P1-A1']
                y2 = dict_within_selectivity[k1][k2]['P2-A2']

                ymin = np.nanmin([
                    np.nanmin(y1),
                    np.nanmin(y2)
                ]) - 0.05

                ymax = np.nanmax([
                    np.nanmax(y1),
                    np.nanmax(y2)
                ]) + 0.05

                for ic, kc in enumerate(keys_within_context):

                    ax = fig.add_subplot(inner[0, ic])

                    ax.spines[
                        ['top', 'right']
                    ].set_visible(False)

                    y = dict_within_selectivity[k1][k2][kc]

                    ax.axhline(
                        0,
                        color='gray',
                        linestyle='--'
                    )

                    ax.scatter(
                        CDdotproduct,
                        y,
                        facecolor='none',
                        edgecolors='k'
                    )

                    _cor = np.corrcoef(
                        CDdotproduct,
                        y
                    )[0, 1]

                    ax.set_ylim([ymin, ymax])

                    ax.set_title(
                        r'$r=$' + str(np.round(_cor, 3)),
                        fontsize=9
                    )

                    if ic == 1:
                        ax.set_yticklabels([])

        fig.tight_layout()

        if savefig:
            plt.savefig(figpath + 'CDdotprod_within_context.pdf')


        # ----------------------------------------------------------
        # across context
        # ----------------------------------------------------------

        fig = plt.figure(figsize=(10, 6))

        outer = fig.add_gridspec(
            3, 3,
            wspace=0.35,
            hspace=0.4
        )

        for i1, k1 in enumerate(keys1):
            for i2, k2 in enumerate(keys2):

                inner = outer[i2, i1].subgridspec(
                    1, 2,
                    wspace=0.08
                )

                y1 = dict_across_activity[k1][k2]['P2-P1']
                y2 = dict_across_activity[k1][k2]['A2-A1']

                ymin = np.nanmin([
                    np.nanmin(y1),
                    np.nanmin(y2)
                ]) - 0.05

                ymax = np.nanmax([
                    np.nanmax(y1),
                    np.nanmax(y2)
                ]) + 0.05

                for ic, kc in enumerate(keys_across_context):

                    ax = fig.add_subplot(inner[0, ic])

                    ax.spines[
                        ['top', 'right']
                    ].set_visible(False)

                    y = dict_across_activity[k1][k2][kc]

                    ax.axhline(
                        0,
                        color='gray',
                        linestyle='--'
                    )

                    ax.scatter(
                        CDdotproduct,
                        y,
                        facecolor='none',
                        edgecolors='k'
                    )

                    _cor = np.corrcoef(
                        CDdotproduct,
                        y
                    )[0, 1]

                    ax.set_ylim([ymin, ymax])

                    ax.set_title(
                        r'$r=$' + str(np.round(_cor, 3)),
                        fontsize=9
                    )

                    if ic == 1:
                        ax.set_yticklabels([])

        fig.tight_layout()

        if savefig:
            plt.savefig(figpath + 'CDdotprod_across_context.pdf')


        # ----------------------------------------------------------
        # within context (multiple datasets)
        # ----------------------------------------------------------

        fig = plt.figure(figsize=(10, 6))

        outer = fig.add_gridspec(
            3, 3,
            wspace=0.35,
            hspace=0.4
        )

        for i1, k1 in enumerate(keys1):
            for i2, k2 in enumerate(keys2):

                inner = outer[i2, i1].subgridspec(
                    1, 2,
                    wspace=0.08
                )

                y1 = dict_within_selectivity[k1][k2]['P1-A1']
                y2 = dict_within_selectivity[k1][k2]['P2-A2']

                ymin = np.nanmin([
                    np.nanmin(y1),
                    np.nanmin(y2)
                ]) - 0.05

                ymax = np.nanmax([
                    np.nanmax(y1),
                    np.nanmax(y2)
                ]) + 0.05

                for ic, kc in enumerate(keys_within_context):

                    ax = fig.add_subplot(inner[0, ic])

                    ax.spines[
                        ['top', 'right']
                    ].set_visible(False)

                    y = dict_within_selectivity[k1][k2][kc]

                    ax.axhline(
                        0,
                        color='gray',
                        linestyle='--'
                    )

                    ax.scatter(
                        CDdotproduct,
                        y,
                        facecolor='none',
                        edgecolors='k'
                    )

                    _cor = np.corrcoef(
                        CDdotproduct,
                        y
                    )[0, 1]

                    ax.set_ylim([ymin, ymax])

                    ax.set_title(
                        r'$r=$' + str(np.round(_cor, 3)),
                        fontsize=9
                    )

                    if ic == 1:
                        ax.set_yticklabels([])

        fig.tight_layout()

        if savefig:
            plt.savefig(figpath + 'CDdotprod_within_context.pdf')



        # # ----------------------------------------------------------
        # # within context (all neurons in the modules)
        # # ----------------------------------------------------------

        # nfile = 33
        # nbins = 20
        # within_selectivity_context1 = np.zeros((nfile,nbins))
        # within_selectivity_context2 = np.zeros((nfile,nbins))

        # within_selectivity_context1_avg = np.zeros(nfile)
        # within_selectivity_context2_avg = np.zeros(nfile)
        
        # fig = plt.figure(figsize=(10, 6))

        # outer = fig.add_gridspec(
        #     3, 3,
        #     wspace=0.35,
        #     hspace=0.4
        # )

        # for i1, k1 in enumerate(keys1):
        #     for i2, k2 in enumerate(keys2):

        #         inner = outer[i2, i1].subgridspec(
        #             1, 2,
        #             wspace=0.08
        #         )

        #         for fx in range(nfile):
        #             y1 = dict_within_selectivity_neurons[k1][k2]['P1-A1'][fx]
        #             y2 = dict_within_selectivity_neurons[k1][k2]['P2-A2'][fx]
        #             y1cnt, y1edg = np.histogram(y1,bins=nbins,range=(-0.2,0.2))
        #             y2cnt, y2edg = np.histogram(y2,bins=nbins,range=(-0.2,0.2))
        #             within_selectivity_context1[fx] = y1cnt
        #             within_selectivity_context2[fx] = y2cnt

        #             within_selectivity_context1_avg[fx] = np.mean(y1)
        #             within_selectivity_context2_avg[fx] = np.mean(y2)            

        #         for ic, kc in enumerate(keys_within_context):

        #             ax = fig.add_subplot(inner[0, ic])

        #             ax.spines[
        #                 ['top', 'right']
        #             ].set_visible(False)

        #             if ic == 0:
        #                 plt.imshow(within_selectivity_context1,cmap='jet',aspect='auto')
        #                 # plt.plot(y1edg[:-1],np.mean(within_selectivity_context1,axis=0),c='k')
        #                 # plt.hist(within_selectivity_context1_avg,bins=20,range=(-0.2,0.2),histtype='step')
        #             if ic == 1:
        #                 plt.imshow(within_selectivity_context2,cmap='jet',aspect='auto')
        #                 # plt.plot(y1edg[:-1],np.mean(within_selectivity_context2,axis=0),c='k')
        #                 # plt.hist(within_selectivity_context2_avg,bins=20,range=(-0.2,0.2),histtype='step')
        #             plt.axvline(nbins//2,color='w',linestyle='--',lw=0.5)
        #             # plt.axvline(y1edg[nbins//2],color='r',linestyle='--')

        #             # ax.set_title(
        #             #     r'$r=$' + str(np.round(_cor, 3)),
        #             #     fontsize=9
        #             # )

        #             if ic == 1:
        #                 ax.set_yticklabels([])

        # fig.tight_layout()

        # if savefig:
        #     plt.savefig(figpath + 'CDdotprod_within_context_neurons.pdf')
