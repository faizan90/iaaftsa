'''
Created on Aug 7, 2022

@author: Faizan3800X-Uni
'''
import matplotlib as mpl
# Has to be big enough to accomodate all plotted values.
mpl.rcParams['agg.path.chunksize'] = 50000

from timeit import default_timer
from multiprocessing import Pool

import h5py
import numpy as np
import matplotlib.pyplot as plt; plt.ioff()

from gnrctsgenr import (
    get_mpl_prms,
    set_mpl_prms,
    GTGPlotBase,
    GTGPlotOSV,
    GTGPlotSingleSite,
    GTGPlotMultiSite,
    GTGPlotSingleSiteQQ,
    GenericTimeSeriesGeneratorPlot,
    )

from gnrctsgenr.misc import print_sl, print_el


class IAAFTSAPlot(
        GTGPlotBase,
        GTGPlotOSV,
        GTGPlotSingleSite,
        GTGPlotMultiSite,
        GTGPlotSingleSiteQQ,
        GenericTimeSeriesGeneratorPlot):

    def __init__(self, verbose):

        GTGPlotBase.__init__(self, verbose)
        GTGPlotOSV.__init__(self)
        GTGPlotSingleSite.__init__(self)
        GTGPlotMultiSite.__init__(self)
        GTGPlotSingleSiteQQ.__init__(self)
        GenericTimeSeriesGeneratorPlot.__init__(self)

        self._plt_sett_phs_red_rates = self._default_line_sett
        self._plt_sett_idxs = self._default_line_sett
        return

    def _cmpr_iaafted_and_final_sers(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_ts_probs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = 'ss__ts_data_cmpr'

        data_labels = tuple(h5_hdl['data_ref'].attrs['data_ref_labels'])

        n_data_labels = h5_hdl['data_ref'].attrs['data_ref_n_labels']

        loop_prod = np.arange(n_data_labels)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        best_rltzn_labs = self._get_best_obj_vals_srtd_sim_labs(sim_grp_main)

        plt.figure()

        plt.figure()
        for data_lab_idx in loop_prod:

            ref_ts_data = h5_hdl[f'data_ref/data_ref_rltzn'][:, data_lab_idx]

            plt.plot(
                ref_ts_data,
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_1,
                label='ref')

            plt.grid(True)

            plt.gca().set_axisbelow(True)

            plt.legend(framealpha=0.7)

            plt.ylabel('Magnitude')
            plt.xlabel('Time step')

            fig_name = (
                f'{out_name_pref}_{data_labels[data_lab_idx]}_ref.png')

            plt.savefig(str(self._ss_dir / fig_name), bbox_inches='tight')

            plt.clf()

            plot_ctr = 0
            for rltzn_lab in best_rltzn_labs:
                iaaft_ts_data = sim_grp_main[
                    f'{rltzn_lab}/init_data'][:, data_lab_idx]

                final_ts_data = sim_grp_main[
                    f'{rltzn_lab}/data'][:, data_lab_idx]

                plt.plot(
                    final_ts_data,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_2,
                    lw=plt_sett.lw_1,
                    label='final')

                plt.plot(
                    iaaft_ts_data,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label='init')

                plt.grid(True)

                plt.gca().set_axisbelow(True)

                plt.legend(framealpha=0.7)

                plt.ylabel('Magnitude')
                plt.xlabel('Time step')

                fig_name = (
                    f'{out_name_pref}_{data_labels[data_lab_idx]}_'
                    f'{rltzn_lab}.png')

                plt.savefig(str(self._ss_dir / fig_name), bbox_inches='tight')

                plt.clf()

                plot_ctr += 1

                if plot_ctr == self._plt_max_n_sim_plots:
                    break

        plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site probability time series '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_osv_iaaftsa(self, vname, xlabel, fname, tlabel):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_phs_red_rates

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        best_rltzn_lab = self._get_best_obj_vals_srtd_sim_labs(sim_grp_main)[0]

        plt.figure()

        min_y_lim = +np.inf
        max_y_lim = -np.inf

        for rltzn_lab in sim_grp_main:
            if rltzn_lab == best_rltzn_lab:
                continue

            osv_variable = sim_grp_main[f'{rltzn_lab}/{vname}']

            if osv_variable.ndim == 2:
                plt.plot(
                    osv_variable[:, 0],
                    osv_variable[:, 1],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1)

                min_y = osv_variable[:, 1].min()
                max_y = osv_variable[:, 1].max()

            elif osv_variable.ndim == 1:
                plt.plot(
                    osv_variable[:],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1)

                min_y = osv_variable[:].min()
                max_y = osv_variable[:].max()

            else:
                raise NotImplementedError(osv_variable.ndim)

            if min_y < min_y_lim:
                min_y_lim = min_y

            if max_y > max_y_lim:
                max_y_lim = max_y

        osv_variable = sim_grp_main[f'{best_rltzn_lab}/{vname}']

        label = 'best'

        if osv_variable.ndim == 2:
            plt.plot(
                osv_variable[:, 0],
                osv_variable[:, 1],
                alpha=plt_sett.alpha_3,
                color=plt_sett.lc_3,
                lw=plt_sett.lw_3,
                label=label)

            min_y = osv_variable[:, 1].min()
            max_y = osv_variable[:, 1].max()

        elif osv_variable.ndim == 1:
            plt.plot(
                osv_variable[:],
                alpha=plt_sett.alpha_3,
                color=plt_sett.lc_3,
                lw=plt_sett.lw_3,
                label=label)

            min_y = osv_variable[:].min()
            max_y = osv_variable[:].max()

        else:
            raise NotImplementedError(osv_variable.ndim)

        if min_y < min_y_lim:
            min_y_lim = min_y

        if max_y > max_y_lim:
            max_y_lim = max_y

        if (min_y_lim >= 0) and (max_y_lim <= 1):

            plt.ylim(-0.1, +1.1)

        plt.legend()

        plt.xlabel('Iteration')

        plt.ylabel(f'{xlabel}')

        plt.grid()

        plt.gca().set_axisbelow(True)

        plt.savefig(
            str(self._osv_dir / f'osv__{fname}.png'), bbox_inches='tight')

        plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting optimization {tlabel} '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def plot(self):

        if self._vb:
            print_sl()

            print('Plotting...')

        assert self._plt_verify_flag, 'Plot in an unverified state!'

        ftns_args = []

        self._fill_osv_args_gnrc(ftns_args)

        # Variables specific to IAAFTSA.
        if self._plt_osv_flag:
            ftns_args.extend([
                (self._plot_osv_iaaftsa,
                 ('mxn_ratio_red_rates',
                  'Mixing ratio increment reduction rate',
                  'mxn_ratio_red_rates',
                  'mixing reduction reduction rates')),
                (self._plot_osv_iaaftsa,
                 ('mxn_ratios_probs',
                  'Probability series mixing ratio',
                  'mxn_ratios_probs',
                  'probability series mixing ratio')),
                (self._plot_osv_iaaftsa,
                 ('mxn_ratios_margs',
                  'Marginals series mixing ratio',
                  'mxn_ratios_margs',
                  'marginals series mixing ratio')),
                (self._plot_osv_iaaftsa,
                 ('n_levelss',
                  'Number of series division levels',
                  'n_levelss',
                  'number of series division levels')),
                (self._plot_osv_iaaftsa,
                 ('max_shift_exps',
                  'Maximum step shift exponent',
                  'max_shift_exps',
                  'maximum step shift exponent')),
                (self._plot_osv_iaaftsa,
                 ('max_shifts',
                  'Maximum step shifts',
                  'max_shifts',
                  'maximum step shifts')),
                (self._plot_osv_iaaftsa,
                 ('pre_vals_ratios',
                  'Step preservation ratio',
                  'pre_vals_ratios',
                  'step preservation ratio')),
                (self._plot_osv_iaaftsa,
                 ('asymm_n_iterss',
                  'Asymmetrize function iterations',
                  'asymm_n_iterss',
                  'asymmetrize function iterations')),
                (self._plot_osv_iaaftsa,
                 ('prob_centers',
                  'Probability center',
                  'prob_centers',
                  'probability center')),
                ])

        self._fill_ss_args_gnrc(ftns_args)

        if self._plt_ss_flag:
            ftns_args.extend([
                (self._cmpr_iaafted_and_final_sers, []),
                ])

        self._fill_ms_args_gnrc(ftns_args)

        if self._plt_ms_flag:
            pass

        self._fill_qq_args_gnrc(ftns_args)

        if self._plt_qq_flag:
            pass

        assert ftns_args

        n_cpus = min(self._n_cpus, len(ftns_args))

        if n_cpus == 1:
            for ftn_arg in ftns_args:
                self._exec(ftn_arg)

        else:
            mp_pool = Pool(n_cpus)

            # NOTE:
            # imap_unordered does not show exceptions, map does.

            # mp_pool.imap_unordered(self._exec, ftns_args)

            mp_pool.map(self._exec, ftns_args, chunksize=1)

            mp_pool.close()
            mp_pool.join()

            mp_pool = None

        if self._vb:
            print('Done plotting.')

            print_el()

        return
