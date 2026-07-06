import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import functions

class ModelFitPlotter:
    def __init__(
        self,
        modelfit,
        fit_summary,
        tvec,
        tix_start,
        tix_end,
        tix_sample,
        tix_delay,
        tix_response,
        nfile,
        figpath
    ):
        self.modelfit = modelfit
        self.fit_summary = fit_summary
        self.tvec = tvec
        self.tix_start = tix_start
        self.tix_end = tix_end
        self.tix_sample = tix_sample
        self.tix_delay = tix_delay
        self.tix_response = tix_response
        self.nfile = nfile
        self.t = tvec[tix_start:tix_end]
        self.figpath = figpath
        plt.style.use('pyplot_setting.mplstyle')

    def find_top_cells_of(self, amp_neuron, percentage):
        _ncell       = len(amp_neuron)
        _frac_cells  = int(_ncell * percentage)
        cells_sorted = np.argsort(amp_neuron)[::-1]
        top_cells    = cells_sorted[:_frac_cells]
        return top_cells

    def remove_top_right_axes(self):
        plt.gca().spines[['top', 'right']].set_visible(False)

    def add_event_lines(self):
        plt.axvline(self.tvec[self.tix_sample], c='gray', linestyle='--')
        plt.axvline(self.tvec[self.tix_delay], c='gray', linestyle='--')
        plt.axvline(self.tvec[self.tix_response], c='gray', linestyle='--')

    def collect_total_variables(self):
        self.total_expvar_neuron = np.array([])
        self.total_expvar_pop    = np.zeros((self.nfile, 4))
        self.total_var_neuron    = np.array([])
        self.total_amp_neuron    = np.array([])
        self.total_tp_neuron     = np.array([])

        for fx in range(self.nfile):
            self.total_expvar_neuron = np.concatenate((
                self.total_expvar_neuron,
                self.fit_summary['expvar_neuron'][fx].flatten()
            ))
            self.total_expvar_pop[fx] = self.fit_summary['expvar_pop'][fx]
            self.total_var_neuron = np.concatenate((
                self.total_var_neuron,
                self.fit_summary['var'][fx].flatten()
            ))
            self.total_amp_neuron = np.concatenate((
                self.total_amp_neuron,
                self.fit_summary['amp'][fx].flatten()
            ))
            self.total_tp_neuron = np.concatenate((
                self.total_tp_neuron,
                self.fit_summary['tp'][fx].flatten()
            ))

    def find_top_90_cells(self, threshold=0.9):
        
        self.amp_frac, self.var_frac, self._cells_top90 = functions.find_top_90_cells(self.total_amp_neuron, self.total_var_neuron, threshold)
        
        # self.amp_frac = np.arange(0.01, 1.0, step=0.01)
        # self.var_frac = np.zeros(len(self.amp_frac))

        # for ix, frac in enumerate(self.amp_frac):
        #     _topcells = self.find_top_cells(self.total_amp_neuron, frac)
        #     self.var_frac[ix] = (
        #         np.sum(self.total_var_neuron[_topcells])
        #         / np.sum(self.total_var_neuron)
        #     )

        # self.amp_frac_get_var90 = self.amp_frac[np.where(self.var_frac > 0.9)[0][0]]
        # self.amp_frac_get_var95 = self.amp_frac[np.where(self.var_frac > 0.95)[0][0]]

        # self._cells_top90 = self.find_top_cells(
        #     self.total_amp_neuron,
        #     self.amp_frac_get_var90
        # )


    def plot_example_fits(self, fx=14, act='P1', savefig=False):
        idx_rise = 3
        idx_decay = 4

        _ydata = self.modelfit[f'sess{fx}'][act]['data'][self.fit_summary['nonoutlier_neuron'][fx]]
        _yfit = self.modelfit[f'sess{fx}'][act]['fit'][self.fit_summary['nonoutlier_neuron'][fx]]
        _expvar = self.modelfit[f'sess{fx}'][act]['expvar'][self.fit_summary['nonoutlier_neuron'][fx]]
        _tau_r = self.modelfit[f'sess{fx}'][act]['par'][self.fit_summary['nonoutlier_neuron'][fx], idx_rise]
        _tau_d = self.modelfit[f'sess{fx}'][act]['par'][self.fit_summary['nonoutlier_neuron'][fx], idx_decay]

        ncell = len(_expvar)
        expvar_sorted = np.argsort(_expvar)
        cell_bad_fit  = expvar_sorted[:20]
        cell_okay_fit = expvar_sorted[ncell // 3:ncell // 3 + 20]
        cell_good_fit = expvar_sorted[-20:]

        plt.figure(figsize=(6, 4.5))
        for ii in range(3):
            for row, cells in enumerate([cell_bad_fit, cell_okay_fit, cell_good_fit]):
                plt.subplot(3, 3, ii + 1 + 3 * row)
                celli = cells[ii]
                plt.plot(self.t, _ydata[celli], lw=1.5, c='k')
                plt.plot(self.t, _yfit[celli], lw=1, c='r')
                self.add_event_lines()
                _exvar = np.round(_expvar[celli], decimals=2)
                _taur  = np.round(_tau_r[celli], decimals=2)
                _taud  = np.round(_tau_d[celli], decimals=2)
                plt.legend(
                    [Line2D([], [], linestyle='None')],
                    [rf'EV={_exvar:.2f}' '\n'
                    rf'$\tau_r={_taur:.2f}$' '\n'
                    rf'$\tau_d={_taud:.2f}$'],
                    # loc='upper right',
                    frameon=False,
                    handlelength=0,
                    handletextpad=0,
                    fontsize=7,
                    bbox_to_anchor=[1,1]
                )  
                self.remove_top_right_axes()
        plt.tight_layout()
        
        if savefig:
            plt.savefig(self.figpath + 'example_fits.pdf')

    def plot_variance_fraction(self, savefig=False):
        plt.figure(figsize=(1.8, 1.5))
        plt.plot(self.amp_frac, self.var_frac, c='k')
        plt.axhline(0.9, color='gray', linestyle='--')
        plt.axvline(0.2, color='gray', linestyle='--')
        plt.xlabel('% of top cells by amp')
        plt.ylabel('frac of total var')
        plt.xticks([0, 0.2, 0.5, 1.0])
        plt.yticks([0, 0.5, 0.9])
        self.remove_top_right_axes()
        plt.tight_layout()

        if savefig:
            plt.savefig(self.figpath + 'variance_fraction.pdf')

    def plot_peak_time_hist(self, savefig=False):
        plt.figure(figsize=(1.8, 1.5))
        plt.hist(
            self.total_tp_neuron,
            bins=25,
            range=(0, 5),
            histtype='step',
            density=True,
            color='k'
        )
        self.add_event_lines()
        plt.xlim([
            self.tvec[self.tix_sample - 4],
            self.tvec[self.tix_response + 6]
        ])
        plt.xlabel('peak time (s)')
        plt.ylabel('neuron density')
        self.remove_top_right_axes()
        plt.tight_layout()

        if savefig:
            plt.savefig(self.figpath + 'peak_time.pdf')

    def plot_expvar_neuron_hist(self, savefig=False):
        total_expvar_neuron_top90 = self.total_expvar_neuron[self._cells_top90]
        mean_total_expvar = np.mean(self.total_expvar_neuron)
        mean_total_expvar_top90 = np.mean(total_expvar_neuron_top90)

        plt.figure(figsize=(2.5, 1.5))
        plt.hist(
            self.total_expvar_neuron,
            bins=50,
            range=(0, 1),
            histtype='step',
            label='all cells',
            density=True,
            color='k'
        )
        plt.hist(
            total_expvar_neuron_top90,
            bins=50,
            range=(0, 1),
            histtype='step',
            label='top cells',
            density=True,
            color='C1'
        )
        plt.scatter(mean_total_expvar, 0.4, s=12, marker='v', clip_on=False, color='k')
        plt.scatter(mean_total_expvar_top90, 0.4, s=12, marker='v', clip_on=False, color='C1')
        plt.legend(bbox_to_anchor=[1, 1], frameon=False)
        plt.xlabel('var explained by model')
        plt.ylabel('neuron density')
        self.remove_top_right_axes()
        plt.tight_layout()
        
        if savefig:
            plt.savefig(self.figpath + 'expvar_neuron_hist.pdf')
        

    def plot_expvar_pop_hist(self, savefig=False):
        plt.figure(figsize=(1.8, 1.5))
        plt.hist(
            self.total_expvar_pop.flatten(),
            bins=25,
            range=(0.5, 1),
            histtype='step',
            color='k'
        )
        plt.scatter(
            np.mean(self.total_expvar_pop),
            2,
            s=12,
            marker='v',
            clip_on=False,
            color='k'
        )
        plt.xlabel('var explained by model')
        plt.ylabel('sess. count')
        self.remove_top_right_axes()
        plt.tight_layout()

        if savefig:
            plt.savefig(self.figpath + 'expvar_pop_hist.pdf')

    def collect_tau_variables(self):
        idx_amp = 1
        idx_rise = 3
        idx_decay = 4

        self.amp = np.array([])
        self.tau_r = np.array([])
        self.tau_d = np.array([])
        for fx in range(self.nfile):
            for act in ['P1', 'A1', 'P2', 'A2']:
                self.amp = np.concatenate((
                    self.amp,
                    self.modelfit[f'sess{fx}'][act]['par'][
                        self.fit_summary['nonoutlier'][fx], idx_amp
                    ]
                ))
                self.tau_r = np.concatenate((
                    self.tau_r,
                    self.modelfit[f'sess{fx}'][act]['par'][
                        self.fit_summary['nonoutlier'][fx], idx_rise
                    ]
                ))
                self.tau_d = np.concatenate((
                    self.tau_d,
                    self.modelfit[f'sess{fx}'][act]['par'][
                        self.fit_summary['nonoutlier'][fx], idx_decay
                    ]
                ))

        self.tau = np.concatenate((self.tau_r, self.tau_d))
        self.tau_top = np.concatenate((
            self.tau_r[self._cells_top90],
            self.tau_d[self._cells_top90]
        ))

    def plot_tau_hist(self, savefig=False):
        plt.figure(figsize=(1.8, 1.5))
        plt.hist(
            np.log10(self.tau_top),
            bins=50,
            histtype='step',
            range=(-3, 1),
            density=True,
            label='all',
            color='C1'
        )
        plt.scatter(
            np.log10(np.mean(self.tau_top)),
            0.06,
            s=12,
            marker='v',
            clip_on=False,
            color='C1'
        )
        plt.xlabel('log of time const (s)')
        plt.ylabel('neuron density')
        plt.xticks([-3, -2, -1, 0, 1])
        self.remove_top_right_axes()
        plt.tight_layout()
        
        if savefig:
            plt.savefig(self.figpath + 'tau_hist.pdf')
        

    def plot_tau_vs_top_cells(self, savefig=False):
        cell_fracs = np.arange(0.1, 1.1, step=0.1)
        tau_mean = np.zeros(len(cell_fracs))
        for ix, frac in enumerate(cell_fracs):
            _cells_top = self.find_top_cells_of(self.total_amp_neuron, frac)
            tau_mean[ix] = np.mean(np.concatenate((self.tau_r[_cells_top],self.tau_d[_cells_top])))

        plt.figure(figsize=(1.8, 1.5))
        plt.plot(cell_fracs, tau_mean, marker='o', ms=2.5, c='k')
        plt.xlabel('% of top cells by amp')
        plt.ylabel('time constant (s)')
        plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        plt.yticks([0.2,0.3,0.4,0.5])
        self.remove_top_right_axes()
        plt.tight_layout()
        
        if savefig:
            plt.savefig(self.figpath + 'tau_vs_top_cells.pdf')
        

    def plot_all(self, fx=14, act='P1'):
        self.collect_total_variables()
        self.compute_var_fraction()
        self.collect_tau_variables()

        self.plot_example_fits(fx=fx, act=act)
        self.plot_variance_fraction()
        self.plot_peak_time_hist()
        self.plot_expvar_neuron_hist()
        self.plot_expvar_pop_hist()
        self.plot_tau_hist()
        self.plot_tau_vs_top_cells()