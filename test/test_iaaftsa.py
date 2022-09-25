'''
@author: Faizan-Uni-Stuttgart

Aug 7, 2022

11:02:10 AM

'''

import os
import sys

if ('P:\\Synchronize\\Python3Codes' not in sys.path):
    sys.path.append('P:\\Synchronize\\Python3Codes')

import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd

from iaaftsa import IAAFTSAMain, IAAFTSAPlot

DEBUG_FLAG = False


def main():

    # TODO: Formally implement the constants used in the _set_cdf_wts method.
    # TODO: Move computations of running variables under the temp update
    # section.
    # TODO: Decide which ftn connected to dist obj ftns computation.
    # TODO: Communicate with running threads through a text file.
    # TODO: Label wts for pairs in multisite obj ftns.
    # TODO: Scaling exp for auto obj wts.
    # TODO: Two values for asymms in 2D. One above and one below the diagonal.
    # Otherwise, it seems to bias the whole thing to one side and not
    # care about the other.
    # TODO: A class for handling ranks efficiently. That can help us
    # recompute ranks faster by subtracting 1 from higher ranks. But do this
    # outside in a script and compare the performance gain for a long
    # series. Handle repeating values as well. Write tests for correctness.
    # TODO: For cases where, acpt_Rate is low and the interpolated init_temp
    # is way out of bounds, draw a line tracing the point cloud and
    # take the init_temp from that.
    # TODO: Instead of linear control of temperature update iters, use
    # logarithmic to update faster as the optimization moves along.
    # TODO: An automatic search for the best bounds of parameters.
    # The same principle to get obj wts can be taken.
    # TODO: For the values on the ecopula diagonal, check what happens to
    # the phases. See if they have similarities.

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\iaaftsa')
    os.chdir(main_dir)

#==============================================================================
#    Daily HBV sim
#==============================================================================
    # in_file_path = Path(r'hbv_sim__1963_2015_2.csv')
    #
    # sim_label = 'test_temp_prec_02'  # next:
    #
    # labels = 'temp;prec'.split(';')  # pet;q_obs;prec;
    #
    # time_fmt = '%Y-%m-%d'
    #
    # beg_time = '1964-10-01'
    # end_time = '1971-11-30'

#==============================================================================
#    Daily ppt.
#==============================================================================
    # in_file_path = Path(r'precipitation_bw_1961_2015_10cps.csv')
    #
    # sim_label = 'test_10cps_ppt_02'  # next:
    #
    # labels = ['P1162', 'P1197', 'cp']
    # # labels = ['P1162']
    #
    # time_fmt = '%Y-%m-%d'
    #
    # beg_time = '1991-01-01'
    # end_time = '1995-12-31'

#==============================================================================
#    Hourly ppt.
#==============================================================================
    # in_file_path = Path(r'neckar_1hr_ppt_data_20km_buff_Y2004_2020_10cps.pkl')
    #
    # sim_label = 'test_data_tfm_probs_02'  # next:
    #
    # labels = ['P1176', 'P1290', 'cp']  # , 'P13674' , 'P13698', 'P1937', 'P2159', 'P2292', ]
    #
    # time_fmt = '%Y-%m-%d'
    #
    # beg_time = '2009-01-01'
    # end_time = '2009-12-31'

    # From Prof.
    # in_file_path = Path(r'BW_dwd_stns_60min_1995_2020_data.csv')
    #
    # sim_label = 'test_lim_perturb_06'  # next:
    #
    # # labels = ['P1176', 'P1290', 'P13674']  # , 'P13698', 'P1937', 'P2159', 'P2292', ]
    #
    # # labels = ['P02787', 'P02575', 'P01216']
    #
    # # All these have no missing values in BW fro 2010 to 2014.
    # # labels = [
    # #     'P00071', 'P00257', 'P00279', 'P00498', 'P00684', 'P00757', 'P00931',
    # #     'P01089', 'P01216', 'P01224', 'P01255', 'P01290', 'P01584', 'P01602', ]
    #     # 'P01711', 'P01937', 'P02388', 'P02575', 'P02638', 'P02787', 'P02814',
    #     # 'P02880', 'P03278', 'P03362', 'P03519', 'P03761', 'P03925', 'P03927',
    #     # 'P04160', 'P04175', 'P04294', 'P04300', 'P04315', 'P04349', 'P04623',
    #     # 'P04710', 'P04881', 'P04928', 'P05229', 'P05664', 'P05711', 'P05724',
    #     # 'P05731', 'P06258', 'P06263', 'P06275', 'P07138', 'P07187', 'P07331',
    #     # 'P13672', 'P13698', 'P13965']
    #
    # # labels = 'P13698;P07331;P13672;P02575;P02814;P00279;P06275;P02787;P05711;P03278;P03761'.split(';')
    # labels = 'P13698;P07331;P13672'.split(';')
    #
    # time_fmt = '%Y-%m-%dT%H:%M:%S'
    #
    # beg_time = '2010-01-01'
    # end_time = '2010-12-31'  # '2014-07-25 15:00:00'  #

#==============================================================================
#    Daily discharge.
#==============================================================================
    in_file_path = Path(r'neckar_q_data_combined_20180713.csv')

    # in_file_path = Path(
    #     r'neckar_q_data_combined_20180713_10cps.csv')

    sim_label = 'test_maiden_432'  # next:

    labels = ['420']  # , '3421']  # , 'cp']  #, '427'

    time_fmt = '%Y-%m-%d'

    beg_time = '1963-07-01'
    end_time = '1973-08-31'

#==============================================================================

#==============================================================================
#    Hourly
#==============================================================================
    # in_file_path = Path(r'hourly_bw_discharge__2008__2019.csv')
    #
    # sim_label = 'cmpr_with_fftm1_06_hourly'
    #
    # labels = ['420']  # '3470', '3465']
    #
    # time_fmt = '%Y-%m-%d-%H'
    #
    # beg_time = '2009-01-01-00'
    # end_time = '2009-12-31-23'

#==============================================================================

#==============================================================================
#    FFTMA - Noise
#==============================================================================
    # in_file_path = Path(
    #     r'neckar_norm_cop_infill_discharge_1961_2015_20190118__fftma_noise.csv')
    #
    # sim_label = 'test_gnrctsgenr_05'
    #
    # labels = ['420', '3421']  # '3470', '3465']
    #
    # time_fmt = '%Y-%m-%d'
    #
    # beg_time = '2000-01-01'
    # end_time = '2001-12-31'

#==============================================================================

    sep = ';'

    verbose = True
    # verbose = False

    h5_name = 'iaaftsa.h5'

    gen_rltzns_flag = True
    # gen_rltzns_flag = False

    plt_flag = True
    # plt_flag = False

    long_test_flag = True
    # long_test_flag = False

    auto_init_temperature_flag = True
    # auto_init_temperature_flag = False

    wts_flag = True
    # wts_flag = False

    n_reals = 16  # A multiple of n_cpus.
    outputs_dir = main_dir / sim_label
    n_cpus = 'auto'

    scorr_flag = True
    asymm_type_1_flag = True
    asymm_type_2_flag = True
    ecop_dens_flag = True
    ecop_etpy_flag = True
    nth_order_diffs_flag = True
    cos_sin_dist_flag = True
    pcorr_flag = True
    asymm_type_1_ms_flag = True
    asymm_type_2_ms_flag = True
    ecop_dens_ms_flag = True
    match_data_ft_flag = True
    match_probs_ft_flag = True
    asymm_type_1_ft_flag = True
    asymm_type_2_ft_flag = True
    nth_order_ft_flag = True
    asymm_type_1_ms_ft_flag = True
    asymm_type_2_ms_ft_flag = True
    etpy_ft_flag = True
    etpy_ms_ft_flag = True
    scorr_ms_flag = True
    etpy_ms_flag = True
    match_data_ms_ft_flag = True
    match_probs_ms_ft_flag = True
    match_data_ms_pair_ft_flag = True
    match_probs_ms_pair_ft_flag = True

    scorr_flag = False
    # asymm_type_1_flag = False
    # asymm_type_2_flag = False
    ecop_dens_flag = False
    ecop_etpy_flag = False
    nth_order_diffs_flag = False
    cos_sin_dist_flag = False
    pcorr_flag = False
    asymm_type_1_ms_flag = False
    asymm_type_2_ms_flag = False
    ecop_dens_ms_flag = False
    # match_data_ft_flag = False
    # match_probs_ft_flag = False
    asymm_type_1_ft_flag = False
    asymm_type_2_ft_flag = False
    nth_order_ft_flag = False
    asymm_type_1_ms_ft_flag = False
    asymm_type_2_ms_ft_flag = False
    etpy_ft_flag = False
    etpy_ms_ft_flag = False
    scorr_ms_flag = False
    etpy_ms_flag = False
    match_data_ms_ft_flag = False
    match_probs_ms_ft_flag = False
    match_data_ms_pair_ft_flag = False
    match_probs_ms_pair_ft_flag = False

    lag_steps = np.arange(1, 21)
    ecop_bins = 25
    nth_ords = np.arange(1, 6)
    lag_steps_vld = np.arange(1, 41)
    nth_ords_vld = np.arange(1, 6)

    use_dists_in_obj_flag = True
    use_dists_in_obj_flag = False

    use_dens_ftn_flag = True
    use_dens_ftn_flag = False

    ratio_per_dens_bin = 0.01

    mixing_ratio_reduction_rate_type = 3
    mixing_ratio_reduction_rate_min = 1e-4
    iaaft_n_iterations_max = 10  # This can be an optimization parameter.

    use_asymmetrize_function_flag = True
    # use_asymmetrize_function_flag = False
    asymmetrize_type = 2
    n_levels_bds = (0, 200)
    max_shift_exp_bds = (0.0, 100.0)
    max_shift_bds = (0, 500)
    pre_values_ratio_bds = (0.0, 1.0)
    asymmetrize_iterations_bds = (1, 3)
    prob_center_bds = (0.0, 1.0)
    pre_val_exp_bds = (0.0, 5.05)
    crt_val_exp_bds = (0.0, 5.05)
    level_thresh_cnst_bds = (-100000, +100000)
    level_thresh_slp_bds = (-100000.0, +100000.0)
    rand_err_sclr_cnst_bds = (-3.5, +3.5)
    rand_err_sclr_rel_bds = (-1.0, +1.0)

    # weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.005], dtype=np.float64)
    # auto_wts_set_flag = False
    # wts_n_iters = None
    # obj_wts_exp = None

    weights = None
    auto_wts_set_flag = True
    wts_n_iters = 500
    obj_wts_exp = 0.65

    min_period = None
    max_period = 90
    keep_beyond_flag = True
    # keep_beyond_flag = False

    lags_nths_wts_flag = True
    lags_nths_wts_flag = False
    lags_nths_exp = 2.0
    lags_nths_n_iters = 1000
    lags_nths_cumm_wts_contrib = 0.9999
    lags_nths_n_thresh = max(lag_steps.size, nth_ords.size)

    label_wts_flag = True
    label_wts_flag = False
    label_exp = 2.0
    label_n_iters = 100

    cdf_penalt_flag = True
    cdf_penalt_flag = False
    n_vals_thresh = 1
    n_vals_penlt = 3

    prt_cdf_calib_flag = True
    prt_cdf_calib_flag = False
    lower_threshold = 0.2
    upper_threshold = 0.8
    inside_flag = False

    stop_criteria_labels = (
        'Maximum iteration completion',
        'Iterations without acceptance',
        'Running objective function tolerance',
        'Annealing temperature',
        'Mixing ratio reduction rate',
        'Running acceptance rate',
        'Iterations without updating the global minimum',
        'Maximum parameter search attempts')

    plt_osv_flag = True
    plt_ss_flag = True
    plt_ms_flag = True
    plt_qq_flag = True

    # plt_osv_flag = False
    # plt_ss_flag = False
    # plt_ms_flag = False
    plt_qq_flag = False

    max_sims_to_plot = 5
    max_lags_to_plot = 4

    if long_test_flag:
        initial_annealing_temperature = 1e5
        temperature_reduction_ratio = 0.9
        update_at_every_iteration_no = 100
        maximum_iterations = int(1e7)
        maximum_without_change_iterations = 20000
        objective_tolerance = 1e-3
        objective_tolerance_iterations = update_at_every_iteration_no * 30
        stop_acpt_rate = 1e-2
        maximum_iterations_without_updating_best = 20000

        temperature_lower_bound = 1e3
        temperature_upper_bound = 5e9
        n_iterations_per_attempt = update_at_every_iteration_no
        acceptance_lower_bound = 0.75
        acceptance_upper_bound = 0.85
        target_acpt_rate = 0.80
        ramp_rate = 1.2
        mixing_ratio_reduction_rate = 0.01

        acceptance_rate_iterations = update_at_every_iteration_no * 30
        acceptance_threshold_ratio = 1e-2

    else:
        initial_annealing_temperature = 1e-10
        temperature_reduction_ratio = 0.99
        update_at_every_iteration_no = 501
        maximum_iterations = 2000  # int(2.5e3)
        maximum_without_change_iterations = maximum_iterations
        objective_tolerance = 1e-15
        objective_tolerance_iterations = maximum_iterations + 1
        stop_acpt_rate = 1e-15
        maximum_iterations_without_updating_best = maximum_iterations + 1

        temperature_lower_bound = 1e0
        temperature_upper_bound = 5e9
        n_iterations_per_attempt = update_at_every_iteration_no
        acceptance_lower_bound = 0.6
        acceptance_upper_bound = 0.7
        target_acpt_rate = 0.65
        ramp_rate = 1.2

        mixing_ratio_reduction_rate = 0.05

        acceptance_rate_iterations = maximum_iterations + 1
        acceptance_threshold_ratio = 1e-2

    if gen_rltzns_flag:
        if in_file_path.suffix == '.csv':
            in_df = pd.read_csv(in_file_path, sep=sep, index_col=0)
            in_df.index = pd.to_datetime(in_df.index, format=time_fmt)

        elif in_file_path.suffix == '.pkl':
            in_df = pd.read_pickle(in_file_path)

        else:
            raise NotImplementedError(
                f'Unknown extension of in_data_file: {in_file_path.suffix}!')

        sub_df = in_df.loc[beg_time:end_time, labels]

        in_vals = sub_df.values

        iaaftsa_cls = IAAFTSAMain(verbose)

        iaaftsa_cls.set_reference_data(in_vals, list(labels))

        iaaftsa_cls.set_objective_settings(
            scorr_flag,
            asymm_type_1_flag,
            asymm_type_2_flag,
            ecop_dens_flag,
            ecop_etpy_flag,
            nth_order_diffs_flag,
            cos_sin_dist_flag,
            lag_steps,
            ecop_bins,
            nth_ords,
            use_dists_in_obj_flag,
            pcorr_flag,
            lag_steps_vld,
            nth_ords_vld,
            asymm_type_1_ms_flag,
            asymm_type_2_ms_flag,
            ecop_dens_ms_flag,
            match_data_ft_flag,
            match_probs_ft_flag,
            asymm_type_1_ft_flag,
            asymm_type_2_ft_flag,
            nth_order_ft_flag,
            asymm_type_1_ms_ft_flag,
            asymm_type_2_ms_ft_flag,
            etpy_ft_flag,
            use_dens_ftn_flag,
            ratio_per_dens_bin,
            etpy_ms_ft_flag,
            scorr_ms_flag,
            etpy_ms_flag,
            match_data_ms_ft_flag,
            match_probs_ms_ft_flag,
            match_data_ms_pair_ft_flag,
            match_probs_ms_pair_ft_flag,
            )

        iaaftsa_cls.set_annealing_settings(
            initial_annealing_temperature,
            temperature_reduction_ratio,
            update_at_every_iteration_no,
            maximum_iterations,
            maximum_without_change_iterations,
            objective_tolerance,
            objective_tolerance_iterations,
            acceptance_rate_iterations,
            stop_acpt_rate,
            maximum_iterations_without_updating_best,
            acceptance_threshold_ratio)

        iaaftsa_cls.set_iaaftsa_sa_settings(
            mixing_ratio_reduction_rate_type,
            mixing_ratio_reduction_rate,
            mixing_ratio_reduction_rate_min,
            iaaft_n_iterations_max)

        if use_asymmetrize_function_flag:
            iaaftsa_cls.set_asymmetrize_settings(
                asymmetrize_type,
                n_levels_bds,
                max_shift_exp_bds,
                max_shift_bds,
                pre_values_ratio_bds,
                asymmetrize_iterations_bds,
                prob_center_bds,
                pre_val_exp_bds,
                crt_val_exp_bds,
                level_thresh_cnst_bds,
                level_thresh_slp_bds,
                rand_err_sclr_cnst_bds,
                rand_err_sclr_rel_bds)

        iaaftsa_cls.set_internal_data_transform_to_use_settings('data')

        if auto_init_temperature_flag:
            iaaftsa_cls.set_annealing_auto_temperature_settings(
                temperature_lower_bound,
                temperature_upper_bound,
                n_iterations_per_attempt,
                acceptance_lower_bound,
                acceptance_upper_bound,
                target_acpt_rate,
                ramp_rate)

        if wts_flag:
            iaaftsa_cls.set_objective_weights_settings(
                weights, auto_wts_set_flag, wts_n_iters, obj_wts_exp)

        if np.any([min_period, max_period]):
            iaaftsa_cls.set_preserve_coefficients_subset_settings(
                min_period, max_period, keep_beyond_flag)

        if lags_nths_wts_flag:
            iaaftsa_cls.set_lags_nths_weights_settings(
                lags_nths_exp,
                lags_nths_n_iters,
                lags_nths_cumm_wts_contrib,
                lags_nths_n_thresh)

        if label_wts_flag:
            iaaftsa_cls.set_label_weights_settings(label_exp, label_n_iters)

        if cdf_penalt_flag:
            iaaftsa_cls.set_cdf_penalty_settings(n_vals_thresh, n_vals_penlt)

        if prt_cdf_calib_flag:
            iaaftsa_cls.set_partial_cdf_calibration_settings(
                lower_threshold, upper_threshold, inside_flag)

        iaaftsa_cls.set_misc_settings(n_reals, outputs_dir, n_cpus)

        iaaftsa_cls.set_stop_criteria_labels(stop_criteria_labels)

        iaaftsa_cls.update_h5_file_name(h5_name)

        iaaftsa_cls.verify()

        iaaftsa_cls.prepare()

        iaaftsa_cls.simulate()

    if plt_flag:
        iaaftsa_plt_cls = IAAFTSAPlot(verbose)

        iaaftsa_plt_cls.set_input(
            outputs_dir / h5_name,
            n_cpus,
            plt_osv_flag,
            plt_ss_flag,
            plt_ms_flag,
            plt_qq_flag,
            max_sims_to_plot,
            max_lags_to_plot)

        iaaftsa_plt_cls.set_output(outputs_dir)

        iaaftsa_plt_cls.verify()

        iaaftsa_plt_cls.plot()

    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
