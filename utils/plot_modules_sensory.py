import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ModulePlots:

    def __init__(
        self,
        figpath,
        nfile,
        acts,
        data_nonoutlier,
        dict_topcells_S,
        dict_topcells_D2,
        dict_topcells_R,
        dict_topcells_1x2_S,
        dict_topcells_1x2_D1,
        dict_topcells_1x2_D2,
        dict_topcells_1x2_R,
        CDdotproduct,
        CDdotproduct_S,
        CDdotproduct_D2,
        CDdotproduct_1x2_S,
        CDdotproduct_1x2_D2,
        dict_within_selectivity_S,
        dict_module_activity_R,
        tix_sample,
        tix_response,
        keys_within_context,
        keys1,
        keys2,
    ):

        self.figpath = figpath
        self.nfile = nfile
        self.acts = acts
        self.data_nonoutlier = data_nonoutlier

        self.dict_topcells_S = dict_topcells_S
        self.dict_topcells_D2 = dict_topcells_D2
        self.dict_topcells_R = dict_topcells_R

        self.dict_topcells_1x2_S = dict_topcells_1x2_S
        self.dict_topcells_1x2_D1 = dict_topcells_1x2_D1
        self.dict_topcells_1x2_D2 = dict_topcells_1x2_D2
        self.dict_topcells_1x2_R = dict_topcells_1x2_R

        self.CDdotproduct = CDdotproduct
        self.CDdotproduct_S = CDdotproduct_S
        self.CDdotproduct_D2 = CDdotproduct_D2
        self.CDdotproduct_1x2_S = CDdotproduct_1x2_S
        self.CDdotproduct_1x2_D2 = CDdotproduct_1x2_D2

        self.dict_within_selectivity_S = dict_within_selectivity_S
        self.dict_module_activity_R = dict_module_activity_R

        self.tix_sample = tix_sample
        self.tix_response = tix_response
        self.keys_within_context = keys_within_context
        self.keys1 = keys1
        self.keys2 = keys2


    # ==============================================================
    # CD dot product: sampling vs delay
    # ==============================================================

    def plot_CDdotproduct_sampling_vs_delay(self, savefig):

        figpath = self.figpath
        CDdotproduct_S = self.CDdotproduct_S
        CDdotproduct_D2 = self.CDdotproduct_D2
        CDdotproduct_1x2_S = self.CDdotproduct_1x2_S
        CDdotproduct_1x2_D2 = self.CDdotproduct_1x2_D2

        plt.figure(figsize=(4, 5.5))
        plt.subplot(311)
        plt.hist(
            CDdotproduct_D2,
            bins=20,
            range=(-1, 1),
            histtype='step',
            label='late delay'
        )
        plt.hist(
            CDdotproduct_S,
            bins=20,
            range=(-1, 1),
            histtype='step',
            label='sampling'
        )
        plt.xlabel('CD dot product')
        plt.ylabel('session count')
        plt.legend(bbox_to_anchor=[1, 1], frameon=False)

        plt.subplot(312)
        plt.plot(CDdotproduct_S, c='k', label='all cells')
        plt.plot(CDdotproduct_1x2_S, c='r', label='top cells (S\D2)')
        plt.axhline(0, color='gray', linestyle='--')
        plt.legend(bbox_to_anchor=[1, 1], frameon=False)
        plt.title('sampling')

        plt.subplot(313)
        plt.plot(CDdotproduct_D2, c='k', label='all cells')
        plt.plot(CDdotproduct_1x2_D2, c='r', label='top cells (D2)')
        plt.axhline(0, color='gray', linestyle='--')
        plt.title('late delay')
        plt.xlabel('session')
        plt.ylabel('CD dot product')
        plt.legend(bbox_to_anchor=[1, 1], frameon=False)
        plt.tight_layout()

        if savefig:
            plt.savefig(figpath + '/sensory_input/CDdotprod_sampling_vs_delay.pdf')


    # ==============================================================
    # cell count: sampling, delay, response
    # ==============================================================

    def plot_cell_count(self, savefig):

        figpath = self.figpath
        nfile = self.nfile
        data_nonoutlier = self.data_nonoutlier
        dict_topcells_1x2_S = self.dict_topcells_1x2_S
        dict_topcells_1x2_D1 = self.dict_topcells_1x2_D1
        dict_topcells_1x2_D2 = self.dict_topcells_1x2_D2
        dict_topcells_1x2_R = self.dict_topcells_1x2_R
        keys1 = self.keys1
        keys2 = self.keys2

        cellcnt_S = np.zeros(nfile)
        cellcnt_D1 = np.zeros(nfile)
        cellcnt_D2 = np.zeros(nfile)
        cellcnt_R = np.zeros(nfile)
        for fx in range(nfile):
            ncell = data_nonoutlier[fx]['P1'].shape[0]
            for i1, k1 in enumerate(keys1):
                for i2, k2 in enumerate(keys2):
                    cellcnt_S[fx] += len(dict_topcells_1x2_S[fx][k1][k2]) / ncell
                    cellcnt_D1[fx] += len(dict_topcells_1x2_D1[fx][k1][k2]) / ncell
                    cellcnt_D2[fx] += len(dict_topcells_1x2_D2[fx][k1][k2]) / ncell
                    cellcnt_R[fx] += len(dict_topcells_1x2_R[fx][k1][k2]) / ncell

        plt.figure(figsize=(4, 5))
        plt.subplot(211)
        plt.plot(cellcnt_S, label='S', c='C0')
        plt.plot(cellcnt_D2, label='D2', c='C1')
        plt.plot(cellcnt_R, label='R', c='C2')
        plt.plot(cellcnt_D1, label='D1', c='r')
        plt.legend(bbox_to_anchor=[1, 1])
        plt.ylabel('frac of neurons')
        plt.subplot(212)
        plt.plot(cellcnt_D1 / cellcnt_S, c='r')
        plt.xlabel('sessions')
        plt.ylabel('|D1| / |S|')
        plt.tight_layout()

        if savefig:
            plt.savefig(figpath + '/sensory_input/cellcnt_small_D1.pdf')


    # ==============================================================
    # heatmap of sensory, decision, response neuron activities
    # ==============================================================

    def plot_heatmap_SDR_neurons(self, savefig):

        figpath = self.figpath
        nfile = self.nfile
        acts = self.acts
        data_nonoutlier = self.data_nonoutlier
        dict_topcells_S = self.dict_topcells_S
        dict_topcells_D2 = self.dict_topcells_D2
        dict_topcells_R = self.dict_topcells_R

        for fx in range(nfile):
            plt.figure(figsize=(6, 6))
            for ix, act in enumerate(acts):
                plt.subplot(2, 2, ix + 1)
                cells_S = dict_topcells_S[act][fx]
                cells_D = dict_topcells_D2[act][fx]
                cells_R = dict_topcells_R[act][fx]

                cells_S_sorted = cells_S[
                    np.argsort(np.argmax(data_nonoutlier[fx][act][cells_S], axis=1))
                ]
                cells_D_sorted = cells_D[
                    np.argsort(np.argmax(data_nonoutlier[fx][act][cells_D], axis=1))
                ]
                cells_R_sorted = cells_R[
                    np.argsort(np.argmax(data_nonoutlier[fx][act][cells_R], axis=1))
                ]
                cells_sorted = np.concatenate((cells_S_sorted, cells_D_sorted, cells_R_sorted))

                ncells_S = len(cells_S_sorted)
                ncells_D = len(cells_D_sorted)

                plt.imshow(
                    data_nonoutlier[fx][act][cells_sorted],
                    cmap='jet',
                    aspect='auto',
                    vmin=0,
                    vmax=0.2
                )
                plt.axvline(8, color='w', linestyle='--')
                plt.axvline(16, color='w', linestyle='--')
                plt.axhline(ncells_S, color='w', linestyle='--')
                plt.axhline(ncells_S + ncells_D, color='w', linestyle='--')
            plt.tight_layout()

            if savefig:
                plt.savefig(figpath + f'/sensory_input/heatmap/heatmap{fx}')
            plt.close()


    # ==============================================================
    # cell counts by 9 groups
    # ==============================================================

    def plot_cell_count_9groups(self, savefig):

        figpath = self.figpath
        nfile = self.nfile
        keys1 = self.keys1
        keys2 = self.keys2

        dict_topcells_compare = {
            'S': self.dict_topcells_1x2_S,
            'D2': self.dict_topcells_1x2_D2,
            'R': self.dict_topcells_1x2_R,
        }
        condition_labels = [
            f'{k1} x {k2}'
            for k1 in keys1
            for k2 in keys2
        ]

        topcell_count_rows = []
        topcell_counts = {
            epoch: np.zeros((nfile, len(keys1), len(keys2)), dtype=int)
            for epoch in dict_topcells_compare
        }

        for epoch, dict_topcells_epoch in dict_topcells_compare.items():
            for fx in range(nfile):
                for i1, k1 in enumerate(keys1):
                    for i2, k2 in enumerate(keys2):
                        n_topcells = len(dict_topcells_epoch[fx][k1][k2])
                        topcell_counts[epoch][fx, i1, i2] = n_topcells
                        topcell_count_rows.append({
                            'epoch': epoch,
                            'fx': fx,
                            'condition1': k1,
                            'condition2': k2,
                            'condition': f'{k1} x {k2}',
                            'n_neurons': n_topcells,
                        })

        df_topcell_counts = pd.DataFrame(topcell_count_rows)
        mean_topcell_counts = df_topcell_counts.pivot_table(
            index='condition',
            columns='epoch',
            values='n_neurons',
            aggfunc='mean',
        ).loc[condition_labels]
        print(mean_topcell_counts)

        bar_width = 0.6
        titles1 = [r'$P_1^+A_1^-$', r'$P_1^+A_1^+$', r'$P_1^-A_1^+$']
        labels2 = [r'$P_2^+A_2^-$', r'$P_2^+A_2^+$', r'$P_2^-A_2^+$']

        for epoch in dict_topcells_compare:
            fig = plt.figure(figsize=(6, 6))
            gs = fig.add_gridspec(
                3, 3,
                hspace=0.5,
                wspace=1
            )

            ymax = np.max(topcell_counts[epoch])
            if ymax == 0:
                ymax = 1

            for i2, key2 in enumerate(keys2):
                for i1, key1 in enumerate(keys1):
                    n_cells = topcell_counts[epoch][:, i1, i2]
                    n_cells_mean = np.mean(n_cells)
                    n_cells_std = np.std(n_cells)

                    ax1 = fig.add_subplot(gs[i2, i1])

                    plt.bar(
                        0,
                        n_cells_mean,
                        width=bar_width,
                        color='gray',
                        alpha=0.5
                    )

                    jitter = 0.1 * (2 * np.random.rand(len(n_cells)) - 1)

                    plt.scatter(
                        0 + jitter,
                        n_cells,
                        s=12,
                        facecolors='none',
                        edgecolors='gray',
                        linewidths=0.5,
                        alpha=1
                    )

                    plt.errorbar(
                        0,
                        n_cells_mean,
                        yerr=n_cells_std / np.sqrt(len(n_cells)),
                        color='k',
                        capsize=8,
                        lw=1.5,
                        linestyle='none'
                    )

                    plt.xlim([-0.5, 0.5])
                    # plt.ylim([0, 1.15 * ymax])

                    if i2 == 0:
                        ax1.set_title(titles1[i1], fontsize=12)

                    if i1 == 0:
                        ax1.set_ylabel(labels2[i2], fontsize=12)

                    ax1.set_xticks([-0.5, 0, 0.5])
                    ax1.set_xticklabels([])

                    plt.gca().spines[['top', 'right']].set_visible(False)

            fig.suptitle(f'{epoch}: number of neurons', fontsize=12)
            plt.tight_layout()

            if savefig:
                plt.savefig(figpath + '/sensory_input/cellcnt_9groups_' + epoch + '.pdf')


    # ==============================================================
    # cell counts by 4 groups
    # ==============================================================

    def plot_cell_count_4groups(self, savefig):

        figpath = self.figpath
        nfile = self.nfile

        condition_groups = {
            'Group 1': [
                ('P1+A1-', 'P2+A2-'),
                ('P1+A1-', 'P2+A2+'),
            ],
            'Group 2': [
                ('P1+A1-', 'P2-A2+'),
            ],
            'Group 3': [
                ('P1+A1+', 'P2-A2+'),
                ('P1-A1+', 'P2-A2+'),
            ],
            'Group 4': [
                ('P1+A1+', 'P2+A2-'),
                ('P1+A1+', 'P2+A2+'),
                ('P1-A1+', 'P2+A2-'),
                ('P1-A1+', 'P2+A2+'),
            ],
        }

        dict_topcells_group_compare = {
            'S': self.dict_topcells_1x2_S,
            'D': self.dict_topcells_1x2_D2,
            'R': self.dict_topcells_1x2_R,
        }

        topcell_group_counts = {
            epoch: np.zeros((nfile, len(condition_groups)), dtype=int)
            for epoch in dict_topcells_group_compare
        }

        for epoch, dict_topcells_epoch in dict_topcells_group_compare.items():
            for fx in range(nfile):
                for ig, condition_pairs in enumerate(condition_groups.values()):
                    cells_group = [
                        dict_topcells_epoch[fx][key1][key2]
                        for key1, key2 in condition_pairs
                    ]
                    cells_group = [
                        cells for cells in cells_group
                        if len(cells) > 0
                    ]
                    if len(cells_group) > 0:
                        topcell_group_counts[epoch][fx, ig] = len(
                            np.unique(np.concatenate(cells_group))
                        )

        bar_width = 0.6

        for epoch in dict_topcells_group_compare:
            fig = plt.figure(figsize=(6, 6))
            gs = fig.add_gridspec(
                2, 2,
                hspace=0.5,
                wspace=0.5
            )

            ymax = np.max(topcell_group_counts[epoch])
            if ymax == 0:
                ymax = 1

            for ig, group_label in enumerate(condition_groups):
                irow = ig % 2
                icol = ig // 2
                n_cells = topcell_group_counts[epoch][:, ig]
                n_cells_mean = np.mean(n_cells)
                n_cells_std = np.std(n_cells)

                ax1 = fig.add_subplot(gs[irow, icol])

                plt.bar(
                    0,
                    n_cells_mean,
                    width=bar_width,
                    color='gray',
                    alpha=0.5
                )

                jitter = 0.1 * (2 * np.random.rand(len(n_cells)) - 1)

                plt.scatter(
                    0 + jitter,
                    n_cells,
                    s=12,
                    facecolors='none',
                    edgecolors='gray',
                    linewidths=0.5,
                    alpha=1
                )

                plt.errorbar(
                    0,
                    n_cells_mean,
                    yerr=n_cells_std / np.sqrt(len(n_cells)),
                    color='k',
                    capsize=8,
                    lw=1.5,
                    linestyle='none'
                )

                plt.xlim([-0.5, 0.5])
                # plt.ylim([0, 1.15 * ymax])

                ax1.set_title(group_label, fontsize=12)
                ax1.set_xticks([-0.5, 0, 0.5])
                ax1.set_xticklabels([])

                if icol == 0:
                    ax1.set_ylabel('number of neurons')

                plt.gca().spines[['top', 'right']].set_visible(False)

            fig.suptitle(f'{epoch}: grouped condition neuron counts', fontsize=12)
            plt.tight_layout()

            if savefig:
                plt.savefig(figpath + '/sensory_input/cellcnt_4groups_' + epoch + '.pdf')


    # ==============================================================
    # CD dot product vs sensory-module activity
    # ==============================================================

    def plot_CDdotproduct_vs_module_activity_S(self, savefig):

        figpath = self.figpath
        CDdotproduct = self.CDdotproduct
        dict_within_selectivity_S = self.dict_within_selectivity_S
        keys_within_context = self.keys_within_context
        keys1 = self.keys1
        keys2 = self.keys2

        fig = plt.figure(figsize=(10, 6))
        outer = fig.add_gridspec(
            3, 3,
            wspace=0.35,
            hspace=0.4
        )

        titles1 = [r'$P_1^+A_1^-$', r'$P_1^+A_1^+$', r'$P_1^-A_1^+$']
        labels2 = [r'$P_2^+A_2^-$', r'$P_2^+A_2^+$', r'$P_2^-A_2^+$']

        for i1, k1 in enumerate(keys1):
            for i2, k2 in enumerate(keys2):
                inner = outer[i2, i1].subgridspec(
                    1, 2,
                    wspace=0.08
                )

                y1 = dict_within_selectivity_S[k1][k2]['P1-A1']
                y2 = dict_within_selectivity_S[k1][k2]['P2-A2']

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
                    ax.spines[['top', 'right']].set_visible(False)

                    y = dict_within_selectivity_S[k1][k2][kc]

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

                    valid = ~np.isnan(CDdotproduct) & ~np.isnan(y)
                    _cor = np.corrcoef(
                        CDdotproduct[valid],
                        y[valid]
                    )[0, 1]

                    ax.set_ylim([ymin, ymax])

                    ax.set_title(
                        r'$r=$' + str(np.round(_cor, 3)),
                        fontsize=9
                    )

                    if i2 == 0:
                        ax.set_xlabel(kc, fontsize=8)
                        ax.xaxis.set_label_position('top')

                    if ic == 1:
                        ax.set_yticklabels([])

                    if ic == 0 and i1 == 0:
                        ax.set_ylabel(labels2[i2], fontsize=12)

                if i2 == 0:
                    fig.add_subplot(outer[i2, i1], frameon=False)
                    plt.tick_params(
                        labelcolor='none',
                        top=False,
                        bottom=False,
                        left=False,
                        right=False
                    )
                    plt.title(titles1[i1], fontsize=12, pad=30)

        fig.tight_layout()

        if savefig:
            plt.savefig(figpath + 'sensory_input/within_context_CDdotprod_S.pdf')


    # ==============================================================
    # sensory module activity
    # ==============================================================

    def plot_module_activity_S(self, savefig):

        figpath = self.figpath
        nfile = self.nfile
        data_nonoutlier = self.data_nonoutlier
        dict_topcells_1x2_S = self.dict_topcells_1x2_S
        tix_sample = self.tix_sample
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

        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(
            3, 3,
            hspace=0.3,
            wspace=0.5
        )

        for i2, key2 in enumerate(keys2):
            for i1, key1 in enumerate(keys1):
                diff1 = np.stack([
                    0 if len(dict_topcells_1x2_S[fx][key1][key2]) == 0 else
                    np.mean(
                        (
                            data_nonoutlier[fx]['P1']
                            - data_nonoutlier[fx]['A1']
                        )[dict_topcells_1x2_S[fx][key1][key2]][:, tix_sample]
                    )
                    for fx in range(nfile)
                ])

                diff2 = np.stack([
                    0 if len(dict_topcells_1x2_S[fx][key1][key2]) == 0 else
                    np.mean(
                        (
                            data_nonoutlier[fx]['P2']
                            - data_nonoutlier[fx]['A2']
                        )[dict_topcells_1x2_S[fx][key1][key2]][:, tix_sample]
                    )
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
            plt.savefig(figpath + 'sensory_input/within_context_module_activity_S.pdf')


    # ==============================================================
    # CD dot product vs response-module activity
    # ==============================================================

    def plot_CDdotproduct_vs_module_activity_R(self, savefig):

        figpath = self.figpath
        CDdotproduct = self.CDdotproduct
        dict_module_activity_R = self.dict_module_activity_R
        tix_response = self.tix_response
        keys_within_context = self.keys_within_context
        keys1 = self.keys1
        keys2 = self.keys2

        fig = plt.figure(figsize=(10, 6))
        outer = fig.add_gridspec(
            3, 3,
            wspace=0.35,
            hspace=0.4
        )

        titles1 = [r'$P_1^+A_1^-$', r'$P_1^+A_1^+$', r'$P_1^-A_1^+$']
        labels2 = [r'$P_2^+A_2^-$', r'$P_2^+A_2^+$', r'$P_2^-A_2^+$']

        for i1, k1 in enumerate(keys1):
            for i2, k2 in enumerate(keys2):
                inner = outer[i2, i1].subgridspec(
                    1, 2,
                    wspace=0.08
                )

                y1 = np.nanmean(
                    dict_module_activity_R[k1][k2]['P1'][:, tix_response]
                    - dict_module_activity_R[k1][k2]['A1'][:, tix_response],
                    axis=1
                )
                y2 = np.nanmean(
                    dict_module_activity_R[k1][k2]['P2'][:, tix_response]
                    - dict_module_activity_R[k1][k2]['A2'][:, tix_response],
                    axis=1
                )

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
                    ax.spines[['top', 'right']].set_visible(False)

                    if kc == 'P1-A1':
                        y = y1
                    if kc == 'P2-A2':
                        y = y2

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

                    valid = ~np.isnan(CDdotproduct) & ~np.isnan(y)
                    _cor = np.corrcoef(
                        CDdotproduct[valid],
                        y[valid]
                    )[0, 1]

                    ax.set_ylim([ymin, ymax])

                    ax.set_title(
                        r'$r=$' + str(np.round(_cor, 3)),
                        fontsize=9
                    )

                    if i2 == 0:
                        ax.set_xlabel(kc, fontsize=8)
                        ax.xaxis.set_label_position('top')

                    if ic == 1:
                        ax.set_yticklabels([])

                    if ic == 0 and i1 == 0:
                        ax.set_ylabel(labels2[i2], fontsize=12)

                if i2 == 0:
                    fig.add_subplot(outer[i2, i1], frameon=False)
                    plt.tick_params(
                        labelcolor='none',
                        top=False,
                        bottom=False,
                        left=False,
                        right=False
                    )
                    plt.title(titles1[i1], fontsize=12, pad=30)

        fig.tight_layout()

        if savefig:
            plt.savefig(figpath + 'sensory_input/within_context_CDdotprod_R.pdf')


    # ==============================================================
    # response module activity
    # ==============================================================

    def plot_module_activity_R(self, savefig):

        figpath = self.figpath
        nfile = self.nfile
        data_nonoutlier = self.data_nonoutlier
        dict_topcells_1x2_R = self.dict_topcells_1x2_R
        tix_response = self.tix_response
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

        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(
            3, 3,
            hspace=0.3,
            wspace=0.5
        )

        for i2, key2 in enumerate(keys2):
            for i1, key1 in enumerate(keys1):
                diff1 = np.stack([
                    0 if len(dict_topcells_1x2_R[fx][key1][key2]) == 0 else
                    np.mean(
                        (
                            data_nonoutlier[fx]['P1']
                            - data_nonoutlier[fx]['A1']
                        )[dict_topcells_1x2_R[fx][key1][key2]][:, tix_response]
                    )
                    for fx in range(nfile)
                ])

                diff2 = np.stack([
                    0 if len(dict_topcells_1x2_R[fx][key1][key2]) == 0 else
                    np.mean(
                        (
                            data_nonoutlier[fx]['P2']
                            - data_nonoutlier[fx]['A2']
                        )[dict_topcells_1x2_R[fx][key1][key2]][:, tix_response]
                    )
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
            plt.savefig(figpath + 'sensory_input/within_context_module_activity_R.pdf')
