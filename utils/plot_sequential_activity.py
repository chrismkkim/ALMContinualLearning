import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SequentialActivityPlotter:
    def __init__(
        self,
        P1_set1, A1_set1, P2_set1, A2_set1,
        tvec,
        tsample, tdelay, tresponse,
        tix_sample, tix_delay, tix_response,
        nfile,
        sf=None
    ):
        self.P1_set1 = P1_set1
        self.A1_set1 = A1_set1
        self.P2_set1 = P2_set1
        self.A2_set1 = A2_set1

        self.tvec = tvec
        self.tsample = tsample
        self.tdelay = tdelay
        self.tresponse = tresponse

        self.tix_sample = tix_sample
        self.tix_delay = tix_delay
        self.tix_response = tix_response

        self.nfile = nfile
        self.sf = sf

    def plot_all_sessions(self, sort_mode='own', savefig=True):
        tstart = self.tsample - 0.5
        tend   = self.tresponse + 2.0
        tix_start = np.where(self.tvec > tstart)[0][0]
        tix_end   = np.where(self.tvec > tend)[0][0]
        vmax = 0.05

        for fx in range(self.nfile):
            titles = ['P1', 'A1', 'P2', 'A2']

            plt.figure(figsize=(5, 1.5))
            ncell = self.P1_set1[fx].shape[0]
            data_all = [
                self.P1_set1[fx],
                self.A1_set1[fx],
                self.P2_set1[fx],
                self.A2_set1[fx]
            ]
            
            if sort_mode == 'context1':
                sort_P = np.argsort(np.argmax(self.P1_set1[fx], axis=1))
                sort_A = np.argsort(np.argmax(self.A1_set1[fx], axis=1))
                sort_all = [sort_P, sort_A, sort_P, sort_A]

            for ii in range(4):
                plot_data = data_all[ii]

                if sort_mode == 'own':
                    tpeak = np.argmax(plot_data, axis=1)
                    cell_sorted = np.argsort(tpeak)
                elif sort_mode == 'context1':
                    cell_sorted = sort_all[ii]                
                    
                plot_data = plot_data[cell_sorted, tix_start:tix_end]

                plt.subplot(1, 4, ii + 1)
                plt.imshow(
                    plot_data,
                    cmap='jet',
                    vmin=0,
                    vmax=vmax,
                    aspect='auto',
                    extent=[tstart, tend, 0, ncell]
                )
                plt.axvline(self.tsample, color='w', linestyle='--')
                plt.axvline(self.tdelay, color='w', linestyle='--')
                plt.axvline(self.tresponse, color='w', linestyle='--')
                plt.xticks([])
                plt.yticks([])
                plt.gca().spines[['top', 'right', 'bottom', 'left']].set_visible(False)
                plt.title(titles[ii], fontsize=8)
            plt.tight_layout()

            if savefig:
                if sort_mode == 'own':
                    plt.savefig('figure/neural_dynamics/sequential_activity/sort_mode_' + sort_mode + f'/session_{fx}_sorted_by_own.pdf')                    
                elif sort_mode == 'context1':
                    plt.savefig('figure/neural_dynamics/sequential_activity/sort_mode_' + sort_mode + f'/session_{fx}_sorted_by_context1.pdf')
                plt.close()

    def axvline_3d(self, ax, t0, n_neurons):
        ax.plot(
            [t0, t0],
            [0, n_neurons - 1],
            [0, 0],
            c='r',
            lw=0.8,
            linestyle='--',
            alpha=0.3
        )

    def plot_3d(self, fx=29, act='A1', savefig=False):
        data_dict = {
            'P1': self.P1_set1,
            'A1': self.A1_set1,
            'P2': self.P2_set1,
            'A2': self.A2_set1,
        }

        n_neurons, n_time = data_dict[act][fx].shape

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        t = np.arange(n_time)

        sort_data_fx = data_dict[act][fx]
        plot_data_fx = data_dict[act][fx]

        tpeak = np.argmax(sort_data_fx, axis=1)
        cell_sort_by_peak_time = np.argsort(tpeak)

        for ci in range(n_neurons):
            neuron = cell_sort_by_peak_time[ci]

            ax.plot(
                t,
                np.full(n_time, ci),
                plot_data_fx[neuron],
                color='k',
                lw=0.4,
                alpha=0.2
            )

        self.axvline_3d(ax, self.tix_sample, n_neurons)
        self.axvline_3d(ax, self.tix_delay, n_neurons)
        self.axvline_3d(ax, self.tix_response, n_neurons)

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
            axis.pane.set_edgecolor((1, 1, 1, 0))
            axis.line.set_color((1, 1, 1, 0))

        ax.set_xlabel('time')
        ax.set_ylabel('neuron')

        ax.view_init(elev=30, azim=-70)

        plt.tight_layout()
        
        if savefig:
            plt.savefig('figure/neural_dynamics/sequential_activity/3D/' + f'session_{fx}_' + act + '.pdf')
            plt.close()

    def plot_no_wave(self, savefig=True):
        if self.sf is None:
            raise ValueError("sf must be provided to plot_no_wave().")

        tstart = 0
        tend   = self.tvec[-2]
        tix_start = np.where(self.tvec > tstart)[0][0]
        tix_end   = np.where(self.tvec > tend)[0][0]
        vmax = 0.1

        if savefig:
            os.makedirs('figure/neural_dynamics/no_wave', exist_ok=True)

        for fx in range(self.nfile):
            ncell = self.A1_set1[fx].shape[0]
            data_fx = self.A1_set1[fx]

            tpeak = np.argmax(data_fx, axis=1)
            cell_sort_by_peak_time = np.argsort(tpeak)

            plt.figure(figsize=(4, 7))

            plt.subplot(311)
            plt.imshow(
                data_fx[cell_sort_by_peak_time, tix_start:tix_end],
                cmap='jet',
                vmin=0,
                vmax=vmax,
                aspect='auto',
                extent=[tstart, tend, 0, ncell]
            )
            plt.xlabel('time (s)')
            plt.ylabel('cells sorted by peak time')

            plt.subplot(312)
            sf_fx = self.sf[fx]
            sf_fx_x_sorted = sf_fx[1][cell_sort_by_peak_time]
            sf_fx_y_sorted = sf_fx[0][cell_sort_by_peak_time]

            ms = 2
            plt.plot(sf_fx_x_sorted, ms=ms, marker='.', linestyle='')
            plt.xlabel('cells sorted by peak time')
            plt.ylabel('x coord')

            plt.subplot(313)
            plt.plot(sf_fx_y_sorted, ms=ms, marker='.', linestyle='')
            plt.xlabel('cells sorted by peak time')
            plt.ylabel('y coord')

            plt.tight_layout()

            if savefig:
                plt.savefig(
                    'figure/neural_dynamics/no_wave/session' + str(fx) + '.pdf'
                )
                plt.close()