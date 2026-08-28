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
        fit_summary,
        dict_topcells_1x2,
        CDdotproduct,
        dict_CDdp_err,
        dict_module_activity,
        tix_delay_range,
        sequential_P1,
        sequential_A1,
        sequential_P2,
        sequential_A2,
        dict_within_selectivity,
        dict_across_activity,
        dict_across_context_and_trial_activity,
        keys_within_context,
        keys_across_context,
        keys1=None,
        keys2=None,
    ):

        self.figpath = figpath
        self.nfile = nfile
        self.fit_summary = fit_summary
        self.dict_topcells_1x2 = dict_topcells_1x2
        self.CDdotproduct = CDdotproduct
        self.dict_CDdp_err = dict_CDdp_err
        self.dict_module_activity = dict_module_activity
        self.tix_delay_range = tix_delay_range

        self.sequential_P1 = sequential_P1
        self.sequential_A1 = sequential_A1
        self.sequential_P2 = sequential_P2
        self.sequential_A2 = sequential_A2

        self.dict_within_selectivity = dict_within_selectivity
        self.dict_across_activity = dict_across_activity
        self.dict_across_context_and_trial_activity = dict_across_context_and_trial_activity

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
        fit_summary = self.fit_summary
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
                    nonoutlier = fit_summary['nonoutlier'][fx]
                    ntotal_cells = len(nonoutlier)

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

            nonoutlier = fit_summary['nonoutlier'][fx]
            ntotal_cells = len(nonoutlier)

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
            total_number_cells,
            fc='None',
            ec='k'
        )

        plt.xlabel('CD dot product')
        plt.ylabel('# of neurons')

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
        tix_delay_range = self.tix_delay_range
        keys1 = self.keys1
        keys2 = self.keys2

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

                diff1 = np.nanmean(
                    dict_module_activity[key1][key2]['P1'][:, tix_delay_range]
                    - dict_module_activity[key1][key2]['A1'][:, tix_delay_range],
                    axis=1
                )

                diff2 = np.nanmean(
                    dict_module_activity[key1][key2]['P2'][:, tix_delay_range]
                    - dict_module_activity[key1][key2]['A2'][:, tix_delay_range],
                    axis=1
                )

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

                diff1 = np.nanmean(
                    dict_module_activity[key1][key2]['P2'][:, tix_delay_range]
                    - dict_module_activity[key1][key2]['P1'][:, tix_delay_range],
                    axis=1
                )

                diff2 = np.nanmean(
                    dict_module_activity[key1][key2]['A2'][:, tix_delay_range]
                    - dict_module_activity[key1][key2]['A1'][:, tix_delay_range],
                    axis=1
                )

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


        # ----------------------------------------------------------
        # across context and trial
        # ----------------------------------------------------------

        fig = plt.figure(figsize=(6, 6))

        gs = fig.add_gridspec(
            3, 3,
            hspace=0.3,
            wspace=0.5
        )

        for i2, key2 in enumerate(keys2):
            for i1, key1 in enumerate(keys1):

                diff1 = np.nanmean(
                    dict_module_activity[key1][key2]['P2'][:, tix_delay_range]
                    - dict_module_activity[key1][key2]['A1'][:, tix_delay_range],
                    axis=1
                )

                diff2 = np.nanmean(
                    dict_module_activity[key1][key2]['A2'][:, tix_delay_range]
                    - dict_module_activity[key1][key2]['P1'][:, tix_delay_range],
                    axis=1
                )

                diff1_mean = np.nanmean(diff1)
                diff2_mean = np.nanmean(diff2)

                diff1_std = np.nanstd(diff1)
                diff2_std = np.nanstd(diff2)

                ymax = np.max(np.concatenate((diff1, diff2))) + 0.02
                ymin = np.min(np.concatenate((diff1, diff2))) - 0.02

                if diff1_mean > 0:
                    c1 = 'tab:olive'
                else:
                    c1 = 'gray'

                if diff2_mean > 0:
                    c2 = 'tab:olive'
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
                        [r'$P_2-A_1$', r'$A_2-P_1$'],
                        rotation=45
                    )
                else:
                    ax1.set_xticks([0, 1])
                    ax1.set_xticklabels([])

        plt.tight_layout()

        if savefig:
            plt.savefig(figpath + 'module_cross_context_and_trial.pdf')


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

                ax1.plot(
                    np.nanmean(
                        dict_module_activity[key1][key2]['P1'],
                        axis=0
                    ),
                    c='purple'
                )

                ax1.plot(
                    np.nanmean(
                        dict_module_activity[key1][key2]['A1'],
                        axis=0
                    ),
                    c='limegreen'
                )

                ax1.axvline(7, color='gray', linestyle='--')
                ax1.axvline(15, color='gray', linestyle='--')

                ax1.set_ylim([0, 0.3])
                ax1.set_xticklabels([])

                ax2 = fig.add_subplot(
                    gs[2 * i2 + 1, i1],
                    sharex=ax1
                )

                ax2.plot(
                    np.nanmean(
                        dict_module_activity[key1][key2]['P2'],
                        axis=0
                    ),
                    c='purple',
                    linestyle='--'
                )

                ax2.plot(
                    np.nanmean(
                        dict_module_activity[key1][key2]['A2'],
                        axis=0
                    ),
                    c='limegreen',
                    linestyle='--'
                )

                ax2.axvline(7, color='gray', linestyle='--')
                ax2.axvline(15, color='gray', linestyle='--')

                ax2.set_ylim([0, 0.3])

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

        dict_within_selectivity = self.dict_within_selectivity
        dict_across_activity = self.dict_across_activity
        dict_across_context_and_trial_activity = self.dict_across_context_and_trial_activity

        keys_within_context = self.keys_within_context
        keys_across_context = self.keys_across_context


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
        # across context and trial
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

                y1 = (
                    dict_across_context_and_trial_activity
                    [k1][k2]['P2-P1']
                )

                y2 = (
                    dict_across_context_and_trial_activity
                    [k1][k2]['A2-A1']
                )

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

                    y = (
                        dict_across_context_and_trial_activity
                        [k1][k2][kc]
                    )

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
            plt.savefig(figpath + 'CDdotprod_across_context_and_trial.pdf')

