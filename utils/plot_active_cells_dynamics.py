import numpy as np
import matplotlib.pyplot as plt
from utils import functions
import os
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl

class ActiveCellsPlotter:
    def __init__(
        self,
        CD_dotproduct,
        new_active_cells_id,
        dict_topcells,
        active_cells_decay,
        fit_summary_topcells,
        fit_summary,
        modelfit,
        t,
        tgo,
        nfile,
        act,
        figpath
    ):
        self.CD_dotproduct = CD_dotproduct
        self.new_active_cells_id = new_active_cells_id
        self.dict_topcells = dict_topcells
        self.active_cells_decay = active_cells_decay
        self.fit_summary_topcells = fit_summary_topcells
        self.fit_summary = fit_summary
        self.modelfit = modelfit
        self.t = t
        self.tgo = tgo
        self.nfile = nfile
        self.act = act
        self.figpath = figpath
        
        plt.style.use('pyplot_setting.mplstyle')
        os.makedirs(self.figpath, exist_ok=True)

    def event_times(self, t):
        plt.axvline(t[8], color='gray', linestyle='--', lw=0.5)
        plt.axvline(t[16], color='gray', linestyle='--', lw=0.5)

    def event_times_horizontal(self, t):
        plt.axhline(t[8], color='gray', linestyle='--', lw=0.5)
        plt.axhline(t[16], color='gray', linestyle='--', lw=0.5)

    def get_variables(self, fx=3):
        self.fx = fx

        # collect topcells sorted by activation time
        new_active_cells_id_fx = self.new_active_cells_id[self.fx]
        self.topcells_sorted_by_activation_time = np.concatenate(
            list(new_active_cells_id_fx.values())
        )

        self.num_topcells = len(self.topcells_sorted_by_activation_time)

        # fraction of new active cells at t (not used in the code)        
        self.frac_new_active_cells_fx = np.array([
            len(_active_cells_id_at_t) / self.num_topcells
            for _active_cells_id_at_t in list(new_active_cells_id_fx.values())
        ])
        self.frac_new_active_cells = np.zeros((self.nfile, 24))
        for fxx in range(self.nfile):
            self.frac_new_active_cells[fxx] = np.array([
                len(_active_cells_id_at_t) / self.num_topcells
                for _active_cells_id_at_t in list(self.new_active_cells_id[fxx].values())
            ])
        self.frac_new_active_cells_mean = np.mean(self.frac_new_active_cells, axis=0)
        self.frac_new_active_cells_sem = np.std(self.frac_new_active_cells, axis=0) / np.sqrt(self.nfile)

        # fraction of active cells (include cells activated earlier in time)
        self.frac_active_cells = np.zeros((self.nfile, 24))
        for fxx in range(self.nfile):
            self.frac_active_cells[fxx] = np.array([
                len(_active_cells_id_at_t) / self.num_topcells
                for _active_cells_id_at_t in list(self.dict_topcells['topcells_at_t'][fxx].values())
            ])
        self.frac_active_cells_fx = self.frac_active_cells[self.fx]
        self.frac_active_cells_mean = np.mean(self.frac_active_cells, axis=0)

        # decay trace of active cells
        self.active_cells_decay_norm_fx = self.active_cells_decay[self.fx]
        self.active_cells_decay_norm_avg = np.nanmean(self.active_cells_decay, axis=0)

        # rise time constant of active cells
        self.linearfit_param_rise_delay_pre_fx, self.linearfit_r2_rise_delay_pre_fx, self.linearfit_trace_rise_delay_pre_fx, \
        self.linearfit_param_rise_delay_peri_fx, self.linearfit_r2_rise_delay_peri_fx, self.linearfit_trace_rise_delay_peri_fx = (
            functions.fit_rise_of_active_cells_linear(self.t, self.active_cells_decay_norm_fx)
        )        

        # exponential fit of decay trace of active cells
        self.expfit_param_decay_fx, self.expfit_r2_decay_fx, self.expfit_trace_decay_fx = (
            functions.fit_decay_of_active_cells_exponential(self.t, self.active_cells_decay_norm_fx)
        )
        self.expfit_param_decay_avg, self.expfit_r2_decay_avg, _ = (
            functions.fit_decay_of_active_cells_exponential(self.t, self.active_cells_decay_norm_avg)
        )

        # linear fit of decay trace of active cells
        self.linearfit_param_decay_fx, self.linearfit_r2_decay_fx, self.linearfit_trace_decay_fx = (
            functions.fit_decay_of_active_cells_linear(self.t, self.active_cells_decay_norm_fx)
        )
        

        # decay and amplitude of top cells (obtained from two-sided exponential)
        self.taud_topcells_at_t_fx = self.fit_summary_topcells['taud_at_t'][self.fx]
        self.amp_topcells_at_t_fx = self.fit_summary_topcells['amp_at_t'][self.fx]

        # decay time constant of topcells at t (obtained from two-sided exponential)
        self.taud_topcells_at_t_fx_avg = np.array([
            np.mean(taud_at_t)
            for taud_at_t in list(self.fit_summary_topcells['taud_at_t'][self.fx].values())
        ])        
        self.efftaud_at_t_fx = np.zeros(len(self.t))
        self.amp_at_t_fx = np.zeros(len(self.t))
        for i in range(len(self.t)):
            _taud_at_t = self.fit_summary_topcells['taud_at_t'][self.fx][i]
            _amp_at_t  = self.fit_summary_topcells['amp_at_t'][self.fx][i]
            self.efftaud_at_t_fx[i] = np.mean(_taud_at_t * np.log(_amp_at_t / 0.04))
            self.amp_at_t_fx[i]     = np.mean(_amp_at_t)

        # fit accuracy of two-sided exponential
        self.expvar_topcells_at_t_fx = self.fit_summary_topcells['expvar_neuron_at_t'][self.fx]
        self.expvar_topcells_at_t_fx_avg = np.array([
            np.mean(expvar_at_t)
            for expvar_at_t in list(self.fit_summary_topcells['expvar_neuron_at_t'][self.fx].values())
        ])

        # neural data
        nonoutlier = self.fit_summary['nonoutlier'][self.fx]
        self.data_fx = self.modelfit[f'sess{self.fx}'][self.act]['data'][nonoutlier, :]
        self.fit_fx  = self.modelfit[f'sess{self.fx}'][self.act]['fit'][nonoutlier, :]
        # binary neural data
        self._data_binary_fx = self.dict_topcells['data_binary'][self.fx]

        self._data_fx_topcells_sorted        = self.data_fx[self.topcells_sorted_by_activation_time]
        self._data_binary_fx_topcells_sorted = self._data_binary_fx[self.topcells_sorted_by_activation_time]

    def get_active_cell_dynamics(self, fx):
        
        # increase / decay rate of active neurons
        self.rate_of_rise_delay_pre = np.zeros((self.nfile,8))
        self.rate_of_rise_delay_peri = np.zeros((self.nfile,8))
        self.rate_of_decay_linear = np.zeros((self.nfile,len(self.t)))
        self.rate_of_decay_exp = np.zeros((self.nfile,len(self.t)))
        self.fit_trace_rise_pre = {i:{} for i in range(self.nfile)}
        self.fit_trace_rise_peri = {i:{} for i in range(self.nfile)}
        self.fit_trace_decay_linear = {i:{} for i in range(self.nfile)}
        self.fit_trace_decay_exp = {i:{} for i in range(self.nfile)}
        for _fx in range(self.nfile):
            self.get_variables(_fx)
            self.rate_of_rise_delay_pre[_fx] = self.linearfit_param_rise_delay_pre_fx[:,0]
            self.rate_of_rise_delay_peri[_fx] = self.linearfit_param_rise_delay_peri_fx[:,0]
            self.rate_of_decay_linear[_fx] = self.linearfit_param_decay_fx[:,0]
            self.rate_of_decay_exp[_fx] = self.expfit_param_decay_fx[:,0]
            
            self.fit_trace_decay_linear[_fx] = self.linearfit_trace_decay_fx
            self.fit_trace_decay_exp[_fx] = self.expfit_trace_decay_fx
            self.fit_trace_rise_pre[_fx] = self.linearfit_trace_rise_delay_pre_fx
            self.fit_trace_rise_peri[_fx] = self.linearfit_trace_rise_delay_peri_fx        
            
        # center of mass of active neuron traces
        _data_plot    = self.active_cells_decay[0]
        _nref, _ntime = _data_plot.shape
        self.CofM     = np.zeros((self.nfile,_nref))
        for _fx in range(self.nfile):
            _data_plot = self.active_cells_decay[_fx]
            _data_norm = _data_plot / np.sum(_data_plot,axis=1,keepdims=True)
            self.CofM[_fx] = np.sum(np.repeat(np.arange(_ntime).reshape(1,-1),_ntime,axis=0) * _data_norm, axis=1)
            
        # reset get_variables
        self.fx = fx
        self.get_variables(fx)
            

    def plot_numcell_to_cumvar(self, savefig):
        _data_cumsum_fx = self.dict_topcells['data_cumsum'][self.fx]
        _ncell = _data_cumsum_fx.shape[0]
        frac_ncell = np.linspace(0,1,num=_ncell)
        plt.figure(figsize=(2.0,1.5))
        for i in np.arange(len(self.t))[::5]:
            plt.plot(frac_ncell,_data_cumsum_fx[:,i],label=f'{i}')
        plt.legend(frameon=False,title='t idx',fontsize=6,bbox_to_anchor=[1,1])
        plt.axhline(0.8,color='gray',linestyle='--')
        plt.axhline(0.9,color='gray',linestyle='--')
        plt.xlabel('frac of cells')
        plt.ylabel('cumulative var')
        plt.tight_layout()
        
        if savefig:
            plt.savefig(self.figpath + f'sess{self.fx}' +'_0_ncell_to_cumvar.pdf',dpi=1200)
            plt.close()

    def plot_active_cell_sorted_activity(self, savefig, dirname):
        plt.figure(figsize=(1.8, 4))

        extent = [
            self.tgo[0],
            self.tgo[-1],
            -0.5,
            self._data_fx_topcells_sorted.shape[0] - 0.5
        ]

        plt.subplot(311)
        plt.imshow(
            self._data_fx_topcells_sorted,
            cmap='jet',
            vmin=0,
            vmax=0.05,
            aspect='auto',
            extent=extent,
            interpolation='gaussian'
        )
        self.event_times(self.tgo)
        plt.ylabel('neuron')

        plt.subplot(312)
        ncell, nt = self._data_binary_fx_topcells_sorted.shape

        dt = self.tgo[1] - self.tgo[0]
        x = np.concatenate(([self.tgo[0] - dt/2], self.tgo + dt/2))
        y = np.arange(ncell + 1)        
        plt.pcolormesh(
            x,
            y,
            self._data_binary_fx_topcells_sorted,
            cmap='binary',
            shading='flat'
        )
        plt.gca().invert_yaxis()
        self.event_times(self.tgo)
        plt.ylabel('neuron')

        plt.subplot(313)
        ymax = np.max(self.frac_active_cells_fx[1:])
        plt.plot(self.tgo, self.frac_active_cells_fx, c='k', marker='.', label='active')
        plt.xlabel('time to go cue')
        plt.ylabel('frac active neurons')
        self.event_times(self.tgo)
        plt.ylim([0, ymax + 0.01])
        plt.tight_layout()
        
        
        if savefig:
            os.makedirs(self.figpath + dirname, exist_ok=True)
            plt.savefig(self.figpath + dirname + f'/sorted_activity_sess{self.fx}.pdf')
            plt.close()
            
    def plot_active_cell_heatmap(self, savefig, dirname):
        plt.figure(figsize=(2.0, 1.6))

        dt = self.tgo[1] - self.tgo[0]
        extent = [
            self.tgo[0] - dt / 2,
            self.tgo[-1] + dt / 2,
            self.tgo[-1] + dt / 2,
            self.tgo[0] - dt / 2
        ]

        plt.imshow(
            self.active_cells_decay_norm_fx,
            cmap='jet',
            vmin=0,
            vmax=1,
            aspect='auto',
            extent=extent,
            origin='upper'
        )
        plt.colorbar()
        self.event_times(self.tgo)
        self.event_times_horizontal(self.tgo)
        plt.xlabel('elapsed time')
        plt.ylabel('reference time')
        plt.gca().spines[:].set_visible(False)
        plt.tight_layout()
        
        if savefig:
            os.makedirs(self.figpath + dirname, exist_ok=True)
            plt.savefig(self.figpath + dirname + f'/active_cell_heatmap_sess{self.fx}.pdf')
            plt.close()

    def plot_active_cell_3d_traces(self, savefig, dirname):
        
        _data_plot  = self.active_cells_decay_norm_fx
        nref, ntime = self.active_cells_decay_norm_fx.shape
        CofM_fx     = self.CofM[self.fx]
        
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(111, projection='3d')
        _t = np.arange(ntime)
        alpha = np.linspace(0.2,0.5,num=8)
        
        #-----------------------------
        # center of mass
        #-----------------------------
        ax.plot(
            np.arange(ntime),CofM_fx,np.zeros(ntime),
            c='gray',
            linestyle='-', 
            marker='o',
            mfc='None',
            mew=0.4,
            ms=2,
            lw=0.4,
            alpha=0.5
            )

        #-------------------------------------
        # divide into sample, delay, response
        #-------------------------------------        
        for ci in range(8):
            ax.plot(
                np.full(ntime, ci),
                _t,
                _data_plot[ci],
                color='k',
                lw=1,
                alpha=alpha[ci]
            )
        for i, ci in enumerate(np.arange(8,16)):
            ax.plot(
                np.full(ntime, ci),
                _t,
                _data_plot[ci],
                color='b',
                lw=1,
                alpha=alpha[i]
            )    
        for i, ci in enumerate(np.arange(16,ntime)):
            ax.plot(
                np.full(ntime, ci),
                _t,
                _data_plot[ci],
                color='r',
                lw=1,
                alpha=alpha[i]
            )   
        
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
            axis.pane.set_edgecolor((1, 1, 1, 0))
            axis.line.set_color((1, 1, 1, 0))
        # Viewing angle
        ax.view_init(elev=50, azim=-10)
        ax.set_box_aspect((3, 1.5, 1))
        plt.tight_layout()
        
        if savefig:
            os.makedirs(self.figpath + dirname, exist_ok=True)
            plt.savefig(self.figpath + dirname + f'/3d_traces_sess{self.fx}.pdf')
            plt.close()
            

        # #--------------------------------
        # # use jet to show z values
        # #--------------------------------
        # zmin = 0 #np.nanmin(active_cells_intensity)
        # zmax = 1 #np.nanmax(active_cells_intensity)

        # cmap = plt.cm.jet
        # norm = mpl.colors.Normalize(vmin=zmin, vmax=zmax)

        # for ci in range(nref):

        #     z = _data_plot[ci]
        #     x = np.full(ntime, ci)
        #     y = _t

        #     # Consecutive 3D points forming line segments
        #     points = np.column_stack((x, y, z))
        #     segments = np.stack((points[:-1], points[1:]), axis=1)

        #     # Value used to color each segment
        #     z_segment = (z[:-1] + z[1:]) / 2

        #     lc = Line3DCollection(
        #         segments,
        #         cmap=cmap,
        #         norm=norm,
        #         linewidth=1,
        #         alpha=0.5
        #     )
        #     lc.set_array(z_segment)

        #     ax.add_collection3d(lc)

        # # Line3DCollection does not automatically set axis limits
        # ax.set_xlim(0, nref - 1)
        # ax.set_ylim(0, ntime - 1)
        # ax.set_zlim(zmin, zmax)

        # # Colorbar
        # sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])

        # # cbar = fig.colorbar(
        # #     sm,
        # #     ax=ax,
        # #     shrink=0.6,
        # #     pad=0.02
        # # )

        # ax.grid(False)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        #     axis.pane.fill = False
        #     axis.pane.set_edgecolor((1, 1, 1, 0))
        #     axis.line.set_color((1, 1, 1, 0))            

        # ax.view_init(elev=50, azim=-10)
        # ax.set_box_aspect((3, 1.5, 1))

        # plt.tight_layout()
        
        
        

    def plot_active_cell_time_constant(self, savefig, dirname):
        plt.figure(figsize=(1.8, 1.5))
        plt.plot(self.tgo, self.expfit_param_decay_fx[:, 0], c='k', marker='.')
        plt.plot(self.tgo, 0 * self.tgo, c='None')
        plt.xlabel('reference time')
        plt.ylabel('decay rate of\nactive neurons (s)')
        self.event_times(self.tgo)
        plt.tight_layout()
        
        if savefig:
            os.makedirs(self.figpath + dirname, exist_ok=True)
            plt.savefig(self.figpath + dirname + f'/decay_time_constant_sess{self.fx}.pdf')
            plt.close()


    def plot_active_cell_traces(self, savefig, dirname):

        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, 8))


        # rise trace
        plt.figure(figsize=(1.8, 3))
        plt.subplot(311)
        for ix, i in enumerate(np.arange(0, 8)):
            telapsed = self.tgo[i:]
            plt.plot(self.tgo[:i+1], self.active_cells_decay_norm_fx[i,:i+1], c=colors[ix], lw=0.8)
            plt.plot(telapsed[0], self.active_cells_decay_norm_fx[i, i], c=colors[ix], marker='o', markersize=1)
        plt.plot(self.tgo, 0 * self.tgo, c='None')
        self.event_times(self.tgo)
        # plt.ylim([0,1.2])
        plt.subplot(312)
        for ix, i in enumerate(np.arange(8, 16)):
            telapsed = self.tgo[i:]
            plt.plot(self.tgo[:i+1], self.active_cells_decay_norm_fx[i,:i+1], c=colors[ix], lw=0.8)
            plt.plot(telapsed[0], self.active_cells_decay_norm_fx[i, i], c=colors[ix], marker='o', markersize=1)
        plt.plot(self.tgo, 0 * self.tgo, c='None')
        self.event_times(self.tgo)
        # plt.ylim([0,1.2])
        plt.subplot(313)
        for ix, i in enumerate(np.arange(16, 24)):
            telapsed = self.tgo[i:]
            plt.plot(self.tgo[:i+1], self.active_cells_decay_norm_fx[i,:i+1], c=colors[ix], lw=0.8)
            plt.plot(telapsed[0], self.active_cells_decay_norm_fx[i, i], c=colors[ix], marker='o', markersize=1)
        plt.plot(self.tgo, 0 * self.tgo, c='None')
        self.event_times(self.tgo)
        # plt.ylim([0,1.2])
        plt.xlabel('elapsed time')
        plt.ylabel('mean activity of\nactive neurons')
        plt.tight_layout()

        if savefig:
            os.makedirs(self.figpath + dirname, exist_ok=True)
            plt.savefig(self.figpath + dirname + f'/rise_traces_sess{self.fx}.pdf')            
            plt.close()
            
            
        # decay trace
        plt.figure(figsize=(1.8, 3))
        plt.subplot(311)
        for ix, i in enumerate(np.arange(0, 8)):
            telapsed = self.tgo[i:]
            plt.plot(telapsed, self.active_cells_decay_norm_fx[i,i:], c=colors[ix], lw=0.8)
            plt.plot(telapsed[0], self.active_cells_decay_norm_fx[i, i], c=colors[ix], marker='o', markersize=1)
        plt.plot(self.tgo, 0 * self.tgo, c='None')
        self.event_times(self.tgo)
        plt.ylim([0,1.2])
        plt.subplot(312)
        for ix, i in enumerate(np.arange(8, 16)):
            telapsed = self.tgo[i:]
            plt.plot(telapsed, self.active_cells_decay_norm_fx[i,i:], c=colors[ix], lw=0.8)
            plt.plot(telapsed[0], self.active_cells_decay_norm_fx[i, i], c=colors[ix], marker='o', markersize=1)
        plt.plot(self.tgo, 0 * self.tgo, c='None')
        self.event_times(self.tgo)
        plt.ylim([0,1.2])
        plt.subplot(313)
        for ix, i in enumerate(np.arange(16, 24)):
            telapsed = self.tgo[i:]
            plt.plot(telapsed, self.active_cells_decay_norm_fx[i,i:], c=colors[ix], lw=0.8)
            plt.plot(telapsed[0], self.active_cells_decay_norm_fx[i, i], c=colors[ix], marker='o', markersize=1)
        plt.plot(self.tgo, 0 * self.tgo, c='None')
        self.event_times(self.tgo)
        plt.ylim([0,1.2])
        plt.xlabel('elapsed time')
        plt.ylabel('mean activity of\nactive neurons')
        # plt.ylabel('frac of neuron at ref\nstaying active')
        plt.tight_layout()

        if savefig:
            os.makedirs(self.figpath + dirname, exist_ok=True)
            plt.savefig(self.figpath + dirname + f'/decay_traces_sess{self.fx}.pdf')
            plt.close()
            
            
    def plot_fit_neuron_taud_and_expvar(self, savefig, dirname):
        plt.figure(figsize=(1.8, 4))

        plt.subplot(311)
        plt.plot(self.tgo, self.efftaud_at_t_fx, c='k', marker='.',label='eff')
        plt.plot(self.tgo,self.taud_topcells_at_t_fx_avg,c='gray',marker='.',alpha=0.5,label='actual')
        plt.plot(self.tgo, 0 * self.tgo, c='None')
        plt.xlabel('reference time')
        plt.ylabel('decay const (s)')
        self.event_times(self.tgo)
        plt.ylim([0.0, 2.0])
        plt.yticks([0.0, 1.0, 2.0])
        plt.legend(frameon=False)

        plt.subplot(312)
        ymax = np.max(self.amp_at_t_fx)
        plt.plot(self.tgo, self.amp_at_t_fx, c='k', marker='.',label='amp')
        plt.plot(self.tgo, 0 * self.tgo, c='None')
        plt.xlabel('reference time')
        plt.ylabel('amplitude')
        self.event_times(self.tgo)
        plt.ylim([0.0, ymax+0.1])
        plt.yticks([0.0, ymax//1])

        plt.subplot(313)
        plt.plot(self.tgo, self.expvar_topcells_at_t_fx_avg, c='k', marker='.')
        plt.xlabel('time')
        plt.ylabel('fit accuracy')
        plt.ylim([0.5, 1])
        plt.yticks([0.5, 1.0])
        self.event_times(self.tgo)
        plt.tight_layout()
        
        if savefig:
            os.makedirs(self.figpath + dirname, exist_ok=True)
            plt.savefig(self.figpath + dirname + f'/eff_tau_and_expvar_sess{self.fx}.pdf')
            plt.close()

    def plot_fit_neuron_examples(self, time_picked=[3, 7, 11, 15], savefig=False, dirname='fit neuron'):
        for ti in time_picked:
            _topcells_at_t = self.dict_topcells['topcells_at_t'][self.fx][ti]
            _expvar_neuron_at_t_fx = self.fit_summary_topcells['expvar_neuron_at_t'][self.fx][ti]
            _expvar_idx_sorted = np.argsort(_expvar_neuron_at_t_fx)

            figpth_to_fit_example = self.figpath + dirname + f'/sess{self.fx}/'
            os.makedirs(figpth_to_fit_example, exist_ok=True)            
                
            ncell_plot = 64
            ncell = len(_expvar_idx_sorted)
            for plot_i, start in enumerate(range(0, ncell, ncell_plot)):

                plt.figure(figsize=(6, 6))
                stop = min(start + ncell_plot, ncell)
                for i, _idx in enumerate(_expvar_idx_sorted[start:stop]):
                    plt.subplot(8, 8, i + 1)

                    ci = _topcells_at_t[_idx]

                    plt.plot(self.tgo, self.data_fx[ci], c='k', lw=1.0)
                    plt.plot(self.tgo, self.fit_fx[ci], c='r', lw=0.5)
                    plt.axvline(self.tgo[ti], color='gray', linestyle='--', lw=0.5)
                    plt.annotate(
                        str(np.round(_expvar_neuron_at_t_fx[_idx], decimals=3)),
                        xy=(0.5, 0.9),
                        xycoords='axes fraction',
                        fontsize=6,
                        color='b'
                    )
                    ymax = np.max(self.data_fx[ci])
                    ylim = 0.1 if ymax <= 0.1 else (0.5 if ymax <= 0.5 else (1 if ymax <= 1 else 2))
                    ax = plt.gca()
                    ax.spines[['top', 'right']].set_visible(False)
                    ax.set_xticks([-2, 0])
                    ax.set_ylim([0, ylim])
                    ax.set_yticks([ylim])
                    ax.set_yticklabels([str(ylim)])
                    ax.tick_params(axis='x', length=2, labelsize=6)
                    ax.tick_params(axis='y', length=2, labelsize=6)
                plt.tight_layout()            
                
                if savefig:
                    os.makedirs(self.figpath + dirname, exist_ok=True)
                    plt.savefig(figpth_to_fit_example + f'/fit_examples_sess{self.fx}' + f'_time{ti}' + f'_{plot_i}' + '.pdf')
                    plt.close()
                    
            plt.figure(figsize=(1.8,1.5))
            plt.hist(_expvar_neuron_at_t_fx, bins=20, range=(0,1), histtype='step', color='k')
            plt.axvline(np.mean(_expvar_neuron_at_t_fx), color='gray', linestyle='--', lw=0.5)
            plt.xlabel('variance explained')
            plt.ylabel('neuron count')
            plt.tight_layout()
            if savefig:
                os.makedirs(self.figpath + dirname, exist_ok=True)
                plt.savefig(figpth_to_fit_example + f'/fit_examples_sess{self.fx}' + f'_time{ti}' + '_histogram.pdf')
                plt.close()

    def plot_fit_neuron_accuracy_vs_amp(self, savefig, dirname):
        if self.act == 'P1':
            trial_num = 0
            
        _topcells_fx  = self.dict_topcells['topcells'][self.fx]
        _topcells_ev  = self.fit_summary['expvar_neuron'][self.fx][trial_num,_topcells_fx]
        _topcells_amp = self.fit_summary['amp'][self.fx][trial_num,_topcells_fx]
        _topcells_taud = self.fit_summary['taud'][self.fx][trial_num,_topcells_fx]
        
        plt.figure(figsize=(1.8,1.5))
        plt.scatter(np.log10(_topcells_amp),_topcells_ev,color='k',s=2,facecolors='None',linewidths=0.3)
        plt.xlabel('amplitude (log)')
        plt.ylabel('fit accuracy')
        plt.tight_layout()
        
        if savefig:
            os.makedirs(self.figpath + dirname, exist_ok=True)
            plt.savefig(self.figpath + dirname + f'/fit_vs_amp_sess{self.fx}.pdf')
            plt.close()


    def plot_average_summary(self, savefig, dirname):
        plt.figure(figsize=(2.0, 4))

        dt = self.tgo[1] - self.tgo[0]
        extent = [
            self.tgo[0] - dt / 2,
            self.tgo[-1] + dt / 2,
            self.tgo[-1] + dt / 2,
            self.tgo[0] - dt / 2
        ]

        plt.subplot(311)
        plt.imshow(
            self.active_cells_decay_norm_avg,
            cmap='jet',
            vmin=0,
            vmax=1,
            aspect='auto',
            extent=extent,
            origin='upper'
        )
        self.event_times(self.tgo)
        self.event_times_horizontal(self.tgo)
        plt.xlabel('elapsed time')
        plt.ylabel('reference time')
        plt.colorbar()

        plt.subplot(312)
        plt.plot(self.tgo, self.expfit_param_decay_avg[:, 0], c='k', marker='.')
        self.event_times(self.tgo)
        plt.xlabel('reference time')
        plt.ylabel('decay time of\nactive neurons (s)')

        plt.subplot(313)
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, 8))

        for ix, i in enumerate(np.arange(0, 8)):
            telapsed = self.tgo[i:]
            plt.plot(telapsed, self.active_cells_decay_norm_avg[i, i:], c=colors[ix], lw=0.8)
            plt.plot(telapsed[0], self.active_cells_decay_norm_avg[i, i], c=colors[ix], marker='o', markersize=1)

        for ix, i in enumerate(np.arange(8, 16)):
            telapsed = self.tgo[i:]
            plt.plot(telapsed, self.active_cells_decay_norm_avg[i, i:], c=colors[ix], lw=0.8)
            plt.plot(telapsed[0], self.active_cells_decay_norm_avg[i, i], c=colors[ix], marker='o', markersize=1)

        for ix, i in enumerate(np.arange(16, 24)):
            telapsed = self.tgo[i:]
            plt.plot(telapsed, self.active_cells_decay_norm_avg[i, i:], c=colors[ix], lw=0.8)
            plt.plot(telapsed[0], self.active_cells_decay_norm_avg[i, i], c=colors[ix], marker='o', markersize=1)

        self.event_times(self.tgo)
        plt.xlabel('elapsed time')
        plt.ylabel('frac of neuron at ref\nstaying active')
        plt.tight_layout()
        
        if savefig:
            os.makedirs(self.figpath + dirname, exist_ok=True)
            plt.savefig(self.figpath + dirname + f'/average_summary.pdf')
            plt.close()

    def plot_CDdotprod_vs_delay_activity(self, savefig, dirname):
        
        '''
        Compute correlation between CD dot product vs. Rate of increase during delay epoch (slope)
        '''        
        corr_CDdp_rise = np.zeros(8)
        for i in range(8):
            corr_CDdp_rise[i] = np.corrcoef(self.CD_dotproduct,self.rate_of_rise_delay_peri[:,i])[0,1]

        '''
        (1) Examples of the mean activity of active cells @ ref time = end of delay
            Show the slopes of linear regression fitting
        '''
        
        # Sort by peri-delay slope
        tref_rise = 7
        fx_sorted = np.argsort(self.rate_of_rise_delay_peri[:, tref_rise])
        # slope_sorted = self.rate_of_rise_delay_peri[fx_sorted, tref_rise]

        # Colormap based on actual slope values
        cmap = plt.cm.jet
        colors = cmap(np.linspace(0, 1, self.nfile))


        fig = plt.figure(figsize=(2.8, 1.3))
        ax1 = plt.subplot(121)
        for fx, color in zip(fx_sorted, colors):
            ax1.plot(self.tgo,self.active_cells_decay[fx][8 + tref_rise, :],color=color,lw=0.3)
        ax1.axvline(self.tgo[8], color='gray', linestyle='--')
        ax1.axvline(self.tgo[16], color='gray', linestyle='--')
        ax1.set_xlabel('time to go cue')
        ax1.set_ylabel('mean activity')

        ax2 = plt.subplot(122)
        ax2.plot(self.tgo[:8],self.fit_trace_rise_pre[fx][tref_rise, 0, :],c='darkgrey', lw=1.5)
        ax2.plot(self.tgo[:8],self.fit_trace_rise_pre[fx][tref_rise, 1, :],c='magenta', linestyle='--')
        ax2.plot(self.tgo[7:tref_rise + 9],self.fit_trace_rise_peri[fx][tref_rise][0, :],c='k', lw=1.5)
        ax2.plot(self.tgo[7:tref_rise + 9],self.fit_trace_rise_peri[fx][tref_rise][1, :],c='r', linestyle='--')
        ax2.plot(self.tgo,self.active_cells_decay[fx][8 + tref_rise, :],c='k',lw=0.2)
        ax2.axvline(self.tgo[8], color='gray', linestyle='--')
        ax2.axvline(self.tgo[16], color='gray', linestyle='--')
        ax2.set_xlabel('time to go cue')
        plt.tight_layout()
        
        if savefig:
            os.makedirs(self.figpath + dirname, exist_ok=True)
            plt.savefig(self.figpath + dirname + f'/delay-slopes_of_rising_trace.pdf')
            plt.close()


        '''
        (2) Compare slopes during sample vs. delay
        '''        
        plt.figure(figsize=(1.5,1.3))
        plt.scatter(
            self.rate_of_rise_delay_pre[fx_sorted,tref_rise], 
            self.rate_of_rise_delay_peri[fx_sorted,tref_rise], 
            facecolors='none',
            edgecolors=colors,
            s=20,
            linewidths=0.6
        )
        plt.plot(np.linspace(0,0.5,num=10),np.linspace(0,0.5,num=10),c='gray',linestyle='--')
        plt.xlabel('slope (sample)')
        plt.ylabel('slope (delay)')
        plt.xlim([-0.1,0.7])
        plt.ylim([-0.1,0.7])
        plt.tight_layout()

        if savefig:
            os.makedirs(self.figpath + dirname, exist_ok=True)
            plt.savefig(self.figpath + dirname + f'/delay-slopes_of_pre_vs_peri.pdf')
            plt.close()

        '''
        (3) CD dot product vs. Slope during delay
        '''
        fx_sorted = np.argsort(self.rate_of_rise_delay_peri[:,tref_rise])
        cmap = plt.cm.jet
        colors = cmap(np.linspace(0, 1, self.nfile))        
        plt.figure(figsize=(3.5,1.5))
        plt.subplot(121)
        plt.scatter(
            self.CD_dotproduct[fx_sorted],
            self.rate_of_rise_delay_peri[fx_sorted, tref_rise],
            facecolors='none',
            edgecolors=colors,
            s=20,
            linewidths=0.6
        )
        plt.annotate(r'$\rho=$' + str(np.round(corr_CDdp_rise[tref_rise],decimals=3)), xy=(0.4,0.9),xycoords='axes fraction')
        plt.xlabel(r'$CD_1CD_2$')
        plt.ylabel('slope (delay)')
        plt.subplot(122)
        plt.plot(self.tgo[8:16], corr_CDdp_rise,marker='o',c='k')
        plt.plot(self.tgo[15], corr_CDdp_rise[tref_rise],marker='o',c='r',ms=2)
        plt.xlabel('time to go cue')
        plt.ylabel(r'Corr btw $CD_1CD_2$' '\n' 'and slope (delay)')
        plt.ylim([-0.4,0.2])
        plt.yticks([-0.4,-0.2,0,0.2])
        plt.tight_layout()
        
        if savefig:
            os.makedirs(self.figpath + dirname, exist_ok=True)
            plt.savefig(self.figpath + dirname + f'/delay-CDdotprod_vs_slope.pdf')
            plt.close()
        
        
    def plot_CDdotprod_vs_sample_activity(self, savefig, dirname):

        tref_decay = 0
        corr_CDdp_decay = np.zeros(8)
        for i in range(8):
            corr_CDdp_decay[i] = np.corrcoef(self.CD_dotproduct, self.rate_of_decay_linear[:,i])[0,1]

        dt = self.tgo[1] - self.tgo[0]
        
        '''
        (1) Examples of the mean activity of active cells @ ref time = start of sample
            Show the slopes of linear regression fitting
        '''
        
        # Colormap based on ranking of rate of decay
        fx_sorted = np.argsort(self.rate_of_decay_linear[:,tref_decay])
        cmap = plt.cm.jet
        colors = cmap(np.linspace(0, 1, self.nfile))
        
        fig = plt.figure(figsize=(2.8,1.3))
        ax1 = plt.subplot(121)
        for fx, color in enumerate(colors):
            _fx_sorted = fx_sorted[fx]
            ax1.plot(self.tgo, self.active_cells_decay[_fx_sorted][tref_decay,:],c=color,lw=0.3)
        ax1.axvline(self.tgo[8],color='gray',linestyle='--')
        ax1.axvline(self.tgo[16],color='gray',linestyle='--')
        ax1.set_ylim([-0.1,2.5])
        ax1.set_xlabel('time to go cue')
        ax1.set_ylabel('mean activity')
        
        plt.subplot(122)
        plt.plot(self.tgo[tref_decay:], self.fit_trace_decay_linear[fx][tref_decay][0,:], c='k', lw=1.5)
        plt.plot(self.tgo[tref_decay:], self.fit_trace_decay_linear[fx][tref_decay][1,:], c='r', linestyle='--')
        plt.plot(self.tgo, self.active_cells_decay[fx][tref_decay,:],c='k',lw=0.2)
        plt.axvline(self.tgo[8],color='gray',linestyle='--')
        plt.axvline(self.tgo[16],color='gray',linestyle='--')
        plt.tight_layout()
        plt.xlabel('time to go cue')
        
        if savefig:
            os.makedirs(self.figpath + dirname, exist_ok=True)
            plt.savefig(self.figpath + dirname + f'/sample-slopes_of_decay_trace.pdf')
            plt.close()

            
        plt.figure(figsize=(3.5,1.5))
        plt.subplot(121)
        plt.scatter(self.CD_dotproduct[fx_sorted], self.rate_of_decay_linear[fx_sorted,tref_decay],facecolors='none',edgecolors=colors,s=20,linewidths=0.6)
        plt.annotate(r'$\rho=$' + str(np.round(corr_CDdp_decay[tref_decay],decimals=3)), xy=(0.4,0.1),xycoords='axes fraction')
        plt.xlabel(r'$CD_1CD_2$')
        plt.ylabel('slope')
        plt.subplot(122)
        plt.plot(dt*np.arange(8), corr_CDdp_decay, marker='o', c='k')
        plt.plot(dt*0, corr_CDdp_decay[tref_decay], marker='o', c='r', ms=2)
        plt.xlabel('time from stim on')
        plt.ylabel(r'Corr btw $CD_1CD_2$' '\n' 'and slope')
        plt.yticks([-0.1,0.3])
        plt.tight_layout()

        if savefig:
            os.makedirs(self.figpath + dirname, exist_ok=True)
            plt.savefig(self.figpath + dirname + f'/sample-CDdotprod_vs_slope.pdf')
            plt.close()


        