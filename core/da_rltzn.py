'''
Created on Aug 7, 2022

@author: Faizan3800X-Uni
'''
from time import asctime
from copy import deepcopy
from collections import deque
from timeit import default_timer

import numpy as np
from kde.kernels import triangular_kern, gaussian_kern, tricube_kern

# from fcopulas import asymmetrize_type_1_cy, asymmetrize_type_2_cy
from fcopulas import asymmetrize_type_1_cy, asymmetrize_type_3_cy

from gnrctsgenr import (
    GTGBase,
    GTGAlgRealization,
    )

from gnrctsgenr.misc import print_sl, print_el


class OPTVARS:

    def __init__(self):

        # For IAAFT.
        self.mxn_ratio_margs = None
        self.mxn_ratio_probs = None

        # For asymmetrize.
        self.n_levels = None
        self.max_shift_exp = None
        self.max_shift = None
        self.pre_vals_ratio = None
        self.asymm_n_iters = None
        self.asymms_rand_err = None
        self.prob_center = None
        return


class IAAFTSARealization(GTGAlgRealization):

    def __init__(self):

        self._rltzn_prm_max_srch_atpts_flag = False

        GTGAlgRealization.__init__(self)
        return

    def _update_wts(self):

        if any([
            self._sett_wts_lags_nths_set_flag,
            self._sett_wts_label_set_flag,
            self._sett_wts_obj_auto_set_flag]):

            opt_vars_cls_old = OPTVARS()

            self._init_iaaft_opt_vars(opt_vars_cls_old, True)

            self._set_init_iaaft(opt_vars_cls_old)

            self._update_wts_iaaftsa(opt_vars_cls_old)

        return

    def _update_wts_iaaftsa(self, opt_vars_cls):

        if self._sett_wts_lags_nths_set_flag:

            if self._vb:
                print_sl()

                print(f'Computing lag and nths weights...')

            self._set_lag_nth_wts(opt_vars_cls)

            if self._vb:

                self._show_lag_nth_wts()

                print(f'Done computing lag and nths weights.')

                print_el()

        if self._sett_wts_label_set_flag:

            if self._vb:
                print_sl()

                print(f'Computing label weights...')

            self._set_label_wts(opt_vars_cls)

            if self._vb:
                self._show_label_wts()

                print(f'Done computing label weights.')

                print_el()

        if self._sett_wts_obj_auto_set_flag:
            if self._vb:
                print_sl()

                print(f'Computing individual objective function weights...')

            self._set_auto_obj_wts(opt_vars_cls)

            if self._vb:
                self._show_obj_wts()

                print(f'Done computing individual objective function weights.')

                print_el()

        return

    def _show_rltzn_situ(
            self,
            iter_ctr,
            rltzn_iter,
            iters_wo_acpt,
            tol,
            temp,
            mxn_ratio_red_rate,
            acpt_rate,
            new_obj_val,
            obj_val_min,
            iter_wo_min_updt,
            stopp_criteria):

        c1 = self._sett_ann_max_iters >= 10000
        c2 = not (iter_ctr % (0.05 * self._sett_ann_max_iters))
        c3 = all(stopp_criteria)

        if (c1 and c2) or (iter_ctr == 1) or (not c3):
            with self._lock:
                print_sl()

                if c3:
                    print(
                        f'Realization {rltzn_iter} finished {iter_ctr} out of '
                        f'{self._sett_ann_max_iters} iterations on {asctime()}.')

                else:
                    print(f'Realization {rltzn_iter} finished on {asctime()}.')

                # right_align_chars
                rac = max([len(lab) for lab in self._alg_cnsts_stp_crit_labs])

                obj_label_str = 'Current objective function value'
                obj_min_label_str = 'Running minimum objective function value'

                rac = max(rac, len(obj_label_str))
                rac = max(rac, len(obj_min_label_str))

                print(f'{obj_label_str:>{rac}}: {new_obj_val:.2E}')
                print(f'{obj_min_label_str:>{rac}}: {obj_val_min:.2E}\n')

                iter_wo_min_updt_ratio = (
                    iter_wo_min_updt / self._sett_ann_max_iter_wo_min_updt)

                print(
                    f'Stopping criteria variables:\n'
                    f'{self._alg_cnsts_stp_crit_labs[0]:>{rac}}: '
                    f'{iter_ctr/self._sett_ann_max_iters:.2%}\n'
                    f'{self._alg_cnsts_stp_crit_labs[1]:>{rac}}: '
                    f'{iters_wo_acpt/self._sett_ann_max_iter_wo_chng:.2%}\n'
                    f'{self._alg_cnsts_stp_crit_labs[2]:>{rac}}: {tol:.2E}\n'
                    f'{self._alg_cnsts_stp_crit_labs[3]:>{rac}}: {temp:.2E}\n'
                    f'{self._alg_cnsts_stp_crit_labs[4]:>{rac}}: '
                    f'{mxn_ratio_red_rate:.3%}\n'
                    f'{self._alg_cnsts_stp_crit_labs[5]:>{rac}}: '
                    f'{acpt_rate:.3%}\n'
                    f'{self._alg_cnsts_stp_crit_labs[6]:>{rac}}: '
                    f'{iter_wo_min_updt_ratio:.2%}')

                print_el()
        return

    def _get_stopp_criteria(self, test_vars):

        (iter_ctr,
         iters_wo_acpt,
         tol,
         temp,
         mxn_ratio_red_rate,
         acpt_rate,
         iter_wo_min_updt) = test_vars

        stopp_criteria = (
            (iter_ctr < self._sett_ann_max_iters),
            (iters_wo_acpt < self._sett_ann_max_iter_wo_chng),
            (tol > self._sett_ann_obj_tol),
            (temp > self._alg_cnsts_almost_zero),
            (mxn_ratio_red_rate > self._sett_ann_min_mxn_ratio_red_rate),
            (acpt_rate > self._sett_ann_stop_acpt_rate),
            (iter_wo_min_updt < self._sett_ann_max_iter_wo_min_updt),
            (not self._rltzn_prm_max_srch_atpts_flag),
            )

        if iter_ctr <= 1:
            assert len(self._alg_cnsts_stp_crit_labs) == len(stopp_criteria), (
                'stopp_criteria and its labels are not of the '
                'same length!')

        return stopp_criteria

    def _get_mxn_ratio_red_rate(
            self, iter_ctr, acpt_rate, old_mxn_ratio_red_rate):

        _ = old_mxn_ratio_red_rate  # To avoid the annoying unused warning.

        if self._alg_ann_runn_auto_init_temp_search_flag:
            mxn_ratio_red_rate = self._sett_ann_auto_init_temp_trgt_acpt_rate

        else:
            if self._sett_ann_mxn_ratio_red_rate_type == 0:
                mxn_ratio_red_rate = 1.0

            elif self._sett_ann_mxn_ratio_red_rate_type == 1:
                mxn_ratio_red_rate = (
                    1.0 - (iter_ctr / self._sett_ann_max_iters))

            elif self._sett_ann_mxn_ratio_red_rate_type == 2:
                mxn_ratio_red_rate = float(
                    self._sett_ann_mxn_ratio_red_rate **
                    (iter_ctr / self._sett_ann_upt_evry_iter))

            elif self._sett_ann_mxn_ratio_red_rate_type == 3:

                # An unstable mean of acpts_rjts_dfrntl is a
                # problem. So, it has to be long enough.

                # Why the min(acpt_rate, old_mxn_ratio_red_rate) was used?

                # Not using min might result in instability as acpt_rate will
                # oscillate when mxn_ratio_red_rate oscillates but this
                # is taken care of by the maximum iterations without
                # updating the global minimum.

                # Also, it might get stuck in a local minimum by taking min.

                # Normally, it moves very slowly if min is used after some
                # iterations. The accpt_rate stays high due to this slow
                # movement after hitting a low. This becomes a substantial
                # part of the time taken to finish annealing which doesn't
                # bring much improvement to the global minimum.

                # mxn_ratio_red_rate = max(
                #     self._sett_ann_min_mxn_ratio_red_rate,
                #     min(acpt_rate, old_mxn_ratio_red_rate))

                mxn_ratio_red_rate = acpt_rate

            else:
                raise NotImplemented(
                    'Unknown _sett_ann_phs_red_rate_type:',
                    self._sett_ann_phs_red_rate_type)

            assert mxn_ratio_red_rate >= 0.0, 'Invalid phs_red_rate!'

        return mxn_ratio_red_rate

    @GTGBase._timer_wrap
    def _get_next_iter_vars(self, mxn_ratio_red_rate, opt_vars_cls_old):

        assert 0.0 <= opt_vars_cls_old.mxn_ratio_probs <= 1.0, (
            opt_vars_cls_old.mxn_ratio_probs)

        assert 0.0 <= opt_vars_cls_old.mxn_ratio_margs <= 1.0, (
            opt_vars_cls_old.mxn_ratio_margs)

        if self._sett_asymm_set_flag:
            assert (self._sett_asymm_n_levels_lbd <=
                    opt_vars_cls_old.n_levels <=
                    self._sett_asymm_n_levels_ubd), opt_vars_cls_old.n_levels

            assert (self._sett_asymm_max_shift_exp_lbd <=
                    opt_vars_cls_old.max_shift_exp <=
                    self._sett_asymm_max_shift_exp_ubd), (
                        opt_vars_cls_old.max_shift_exp)

            assert (self._sett_asymm_max_shift_lbd <=
                    opt_vars_cls_old.max_shift <=
                    self._sett_asymm_max_shift_ubd), opt_vars_cls_old.max_shift

            assert (self._sett_asymm_pre_vals_ratio_lbd <=
                    opt_vars_cls_old.pre_vals_ratio <=
                    self._sett_asymm_pre_vals_ratio_ubd), (
                        opt_vars_cls_old.pre_vals_ratio)

            assert (self._sett_asymm_n_iters_lbd <=
                    opt_vars_cls_old.asymm_n_iters <=
                    self._sett_asymm_n_iters_ubd), (
                        opt_vars_cls_old.asymm_n_iters)

            assert (self._sett_asymm_prob_center_lbd <=
                    opt_vars_cls_old.prob_center <=
                    self._sett_asymm_prob_center_ubd), (
                        opt_vars_cls_old.prob_center)
        #======================================================================

        opt_vars_cls_new = OPTVARS()

        n_vars_to_choose_from = 1

        if self._sett_asymm_set_flag:
            if self._sett_asymm_type == 1:
                n_vars_to_choose_from += 5

            elif self._sett_asymm_type == 2:
                n_vars_to_choose_from += 6

            else:
                raise NotImplementedError

        max_search_atpts = 1000
        search_atpts = 0

        var_updt_flag = False

        while not var_updt_flag:

            if search_atpts == max_search_atpts:
                self._rltzn_prm_max_srch_atpts_flag = True
                break

            var_to_updt = np.random.choice(np.arange(n_vars_to_choose_from))

            # Mixing variables.
            if var_to_updt == 0:

                ratio_diff_probs = -0.5 + (1 * np.random.random())

                mxn_ratio_probs = opt_vars_cls_old.mxn_ratio_probs + (
                    mxn_ratio_red_rate * ratio_diff_probs)

                mxn_ratio_probs = max(0.0, mxn_ratio_probs)
                mxn_ratio_probs = min(1.0, mxn_ratio_probs)

                mxn_ratio_margs = 1.0 - mxn_ratio_probs

                if opt_vars_cls_old.mxn_ratio_probs == mxn_ratio_probs:
                    search_atpts += 1
                    continue

                opt_vars_cls_new.mxn_ratio_probs = mxn_ratio_probs
                opt_vars_cls_new.mxn_ratio_margs = mxn_ratio_margs

                var_updt_flag = True

            else:
                opt_vars_cls_new.mxn_ratio_probs = (
                    opt_vars_cls_old.mxn_ratio_probs)

                opt_vars_cls_new.mxn_ratio_margs = (
                    opt_vars_cls_old.mxn_ratio_margs)

            # Asymmetrize variables.
            # n_levels.
            if (var_to_updt == 1) and (
                (self._sett_asymm_n_levels_ubd -
                 self._sett_asymm_n_levels_lbd) > 0):

                if False:
                    n_levels_probs = [
                        0.5 * mxn_ratio_red_rate,
                        1.0 - mxn_ratio_red_rate,
                        0.5 * mxn_ratio_red_rate]

                    n_levels_diff = np.random.choice(
                        [-1, 0, 1], p=n_levels_probs)

                    n_levels = opt_vars_cls_old.n_levels + n_levels_diff

                else:
                    n_levels = opt_vars_cls_old.n_levels + int(
                        mxn_ratio_red_rate *
                        (self._sett_asymm_n_levels_ubd -
                         self._sett_asymm_n_levels_lbd) * (
                        -0.5 + (1 * np.random.random())))

                n_levels = max(self._sett_asymm_n_levels_lbd, n_levels)
                n_levels = min(self._sett_asymm_n_levels_ubd, n_levels)

                n_levels = int(n_levels)

                if opt_vars_cls_old.n_levels == n_levels:
                    search_atpts += 1
                    continue

                opt_vars_cls_new.n_levels = n_levels

                var_updt_flag = True

            else:
                opt_vars_cls_new.n_levels = opt_vars_cls_old.n_levels

            # max_shift_exp.
            if var_to_updt == 2 and (
                (self._sett_asymm_max_shift_exp_ubd -
                 self._sett_asymm_max_shift_exp_lbd) > 0):

                max_shift_exp_diff = -0.5 + (1 * np.random.random())
                max_shift_exp_diff *= mxn_ratio_red_rate

                max_shift_exp = (
                    opt_vars_cls_old.max_shift_exp + max_shift_exp_diff)

                # max_shift_exp = opt_vars_cls_old.max_shift_exp + (
                #     (self._sett_asymm_max_shift_exp_ubd -
                #      self._sett_asymm_max_shift_exp_lbd) * (
                #     -1 + (2 * np.random.random())))

                max_shift_exp = max(
                    self._sett_asymm_max_shift_exp_lbd, max_shift_exp)

                max_shift_exp = min(
                    self._sett_asymm_max_shift_exp_ubd, max_shift_exp)

                if opt_vars_cls_old.max_shift_exp == max_shift_exp:
                    search_atpts += 1
                    continue

                opt_vars_cls_new.max_shift_exp = max_shift_exp

                var_updt_flag = True

            else:
                opt_vars_cls_new.max_shift_exp = opt_vars_cls_old.max_shift_exp

            # max_shift.
            if var_to_updt == 3 and (
                (self._sett_asymm_max_shift_ubd -
                 self._sett_asymm_max_shift_lbd) > 0):

                if False:
                    max_shift_probs = [
                        0.5 * mxn_ratio_red_rate,
                        1.0 - mxn_ratio_red_rate,
                        0.5 * mxn_ratio_red_rate]

                    max_shift_diff = np.random.choice(
                        [-1, 0, 1], p=max_shift_probs)

                    max_shift = opt_vars_cls_old.max_shift + max_shift_diff

                else:
                    max_shift = opt_vars_cls_old.max_shift + int(
                        mxn_ratio_red_rate *
                        (self._sett_asymm_max_shift_ubd -
                         self._sett_asymm_max_shift_lbd) * (
                        -0.5 + (1 * np.random.random())))

                max_shift = max(self._sett_asymm_max_shift_lbd, max_shift)
                max_shift = min(self._sett_asymm_max_shift_ubd, max_shift)

                max_shift = int(max_shift)

                if opt_vars_cls_old.max_shift == max_shift:
                    search_atpts += 1
                    continue

                opt_vars_cls_new.max_shift = max_shift

                var_updt_flag = True

            else:
                opt_vars_cls_new.max_shift = opt_vars_cls_old.max_shift

            # pre_vals_ratio.
            if var_to_updt == 4 and (
                (self._sett_asymm_pre_vals_ratio_ubd -
                 self._sett_asymm_pre_vals_ratio_lbd) > 0):

                ratio_diff_pre_vals = -0.5 + (1 * np.random.random())

                pre_vals_ratio = opt_vars_cls_old.pre_vals_ratio + (
                    mxn_ratio_red_rate * ratio_diff_pre_vals)

                # pre_vals_ratio = opt_vars_cls_old.pre_vals_ratio + (
                #     (self._sett_asymm_pre_vals_ratio_ubd -
                #      self._sett_asymm_pre_vals_ratio_lbd) * (
                #     -1 + (2 * np.random.random())))

                pre_vals_ratio = max(
                    self._sett_asymm_pre_vals_ratio_lbd, pre_vals_ratio)

                pre_vals_ratio = min(
                    self._sett_asymm_pre_vals_ratio_ubd, pre_vals_ratio)

                if opt_vars_cls_old.pre_vals_ratio == pre_vals_ratio:
                    search_atpts += 1
                    continue

                opt_vars_cls_new.pre_vals_ratio = pre_vals_ratio

                var_updt_flag = True

            else:
                opt_vars_cls_new.pre_vals_ratio = (
                    opt_vars_cls_old.pre_vals_ratio)

            # asymm_n_iters
            if var_to_updt == 5 and (
                (self._sett_asymm_n_iters_ubd -
                 self._sett_asymm_n_iters_lbd) > 0):

                if False:
                    asymm_n_iters_probs = [
                        0.5 * mxn_ratio_red_rate,
                        1.0 - mxn_ratio_red_rate,
                        0.5 * mxn_ratio_red_rate]

                    asymm_n_iters_diff = np.random.choice(
                        [-1, 0, 1], p=asymm_n_iters_probs)

                    asymm_n_iters = (
                        opt_vars_cls_old.asymm_n_iters + asymm_n_iters_diff)

                else:
                    asymm_n_iters = opt_vars_cls_old.asymm_n_iters + int(
                        mxn_ratio_red_rate *
                        (self._sett_asymm_n_iters_ubd -
                         self._sett_asymm_n_iters_lbd) * (
                        -0.5 + (1 * np.random.random())))

                asymm_n_iters = max(
                    self._sett_asymm_n_iters_lbd, asymm_n_iters)

                asymm_n_iters = min(
                    self._sett_asymm_n_iters_ubd, asymm_n_iters)

                asymm_n_iters = int(asymm_n_iters)

                if opt_vars_cls_old.asymm_n_iters == asymm_n_iters:
                    search_atpts += 1
                    continue

                opt_vars_cls_new.asymm_n_iters = asymm_n_iters

                var_updt_flag = True

            else:
                opt_vars_cls_new.asymm_n_iters = opt_vars_cls_old.asymm_n_iters

            # prob_center.
            if var_to_updt == 6 and (
                (self._sett_asymm_prob_center_ubd -
                 self._sett_asymm_prob_center_lbd) > 0):

                diff_pre_vals = -0.5 + (1 * np.random.random())

                prob_center = opt_vars_cls_old.prob_center + (
                    mxn_ratio_red_rate * diff_pre_vals)

                prob_center = max(
                    self._sett_asymm_prob_center_lbd, prob_center)

                prob_center = min(
                    self._sett_asymm_prob_center_ubd, prob_center)

                if opt_vars_cls_old.prob_center == prob_center:
                    search_atpts += 1
                    continue

                opt_vars_cls_new.prob_center = prob_center

                var_updt_flag = True

            else:
                opt_vars_cls_new.prob_center = (
                    opt_vars_cls_old.prob_center)

            search_atpts += 1

        opt_vars_cls_new.asymms_rand_err = opt_vars_cls_old.asymms_rand_err

        return opt_vars_cls_new

    def _update_sim_no_prms(self):

        data = np.fft.irfft(self._rs.ft, axis=0)

        probs = self._get_probs(data, True)

        self._rs.data = np.empty_like(
            self._data_ref_rltzn_srtd, dtype=np.float64)

        for i in range(self._data_ref_n_labels):
            self._rs.data[:, i] = self._data_ref_rltzn_srtd[
                np.argsort(np.argsort(probs[:, i])), i]

        self._rs.probs = probs

        if True:
            # This needs to be done once more, in case _rs.ft variable is not
            # having the correct value due to some overlooking.
            self._rs.ft = np.fft.rfft(self._rs.data, axis=0)
            self._rs.mag_spec = np.abs(self._rs.ft)
            self._rs.phs_spec = np.angle(self._rs.ft)

        self._rs.data_ft_coeffs = np.fft.rfft(self._rs.data, axis=0)
        self._rs.data_ft_coeffs_mags = np.abs(self._rs.data_ft_coeffs)
        self._rs.data_ft_coeffs_phss = np.angle(self._rs.data_ft_coeffs)

        self._rs.probs_ft_coeffs = np.fft.rfft(self._rs.probs, axis=0)
        self._rs.probs_ft_coeffs_mags = np.abs(self._rs.probs_ft_coeffs)
        self._rs.probs_ft_coeffs_phss = np.angle(self._rs.probs_ft_coeffs)

        self._update_obj_vars('sim')
        return

    @GTGBase._timer_wrap
    def _update_sim(self, opt_vars_cls, load_snapshot_flag):

        if load_snapshot_flag:
            self._load_snapshot()

        else:
            self._run_iaaft(opt_vars_cls, False)
            self._update_sim_no_prms()

        return

    def _init_iaaft_opt_vars(self, opt_vars_cls, set_mean_flag=False):

        if set_mean_flag:
            opt_vars_cls.mxn_ratio_probs = 0.5

            opt_vars_cls.mxn_ratio_margs = 1.0 - opt_vars_cls.mxn_ratio_probs

            if self._sett_asymm_set_flag:
                opt_vars_cls.n_levels = int(
                    self._sett_asymm_n_levels_lbd + (
                        (self._sett_asymm_n_levels_ubd -
                         self._sett_asymm_n_levels_lbd) * 0.5))

                opt_vars_cls.max_shift_exp = (
                    self._sett_asymm_max_shift_exp_lbd + (
                        self._sett_asymm_max_shift_exp_ubd -
                        self._sett_asymm_max_shift_exp_lbd) * 0.5)

                # Can be zero but is changed later on when computing new
                # parameters.

                opt_vars_cls.max_shift = int(
                    self._sett_asymm_max_shift_lbd + (
                        self._sett_asymm_max_shift_ubd -
                        self._sett_asymm_max_shift_lbd) * 0.5)

                opt_vars_cls.pre_vals_ratio = (
                    self._sett_asymm_pre_vals_ratio_lbd + (
                        self._sett_asymm_pre_vals_ratio_ubd -
                        self._sett_asymm_pre_vals_ratio_lbd) * 0.5)

                opt_vars_cls.asymm_n_iters = int(
                    self._sett_asymm_n_iters_lbd + (
                        self._sett_asymm_n_iters_ubd -
                        self._sett_asymm_n_iters_lbd) * 0.5)

                if self._sett_asymm_type == 2:
                    opt_vars_cls.prob_center = (
                        self._sett_asymm_prob_center_lbd + (
                            self._sett_asymm_prob_center_ubd -
                            self._sett_asymm_prob_center_lbd) * 0.5)

        else:
            opt_vars_cls.mxn_ratio_probs = np.random.random()

            opt_vars_cls.mxn_ratio_margs = 1.0 - opt_vars_cls.mxn_ratio_probs

            if self._sett_asymm_set_flag:
                opt_vars_cls.n_levels = int(
                    self._sett_asymm_n_levels_lbd + (
                        (self._sett_asymm_n_levels_ubd -
                         self._sett_asymm_n_levels_lbd) *
                        0.5 * np.random.random()))

                opt_vars_cls.max_shift_exp = (
                    self._sett_asymm_max_shift_exp_lbd + (
                        self._sett_asymm_max_shift_exp_ubd -
                        self._sett_asymm_max_shift_exp_lbd) *
                    1.0 * np.random.random())

                # Can be zero but is changed later on when computing new
                # parameters.

                opt_vars_cls.max_shift = int(
                    self._sett_asymm_max_shift_lbd + (
                        self._sett_asymm_max_shift_ubd -
                        self._sett_asymm_max_shift_lbd) *
                     1.0 * np.random.random())

                opt_vars_cls.pre_vals_ratio = (
                    self._sett_asymm_pre_vals_ratio_lbd + (
                        self._sett_asymm_pre_vals_ratio_ubd -
                        self._sett_asymm_pre_vals_ratio_lbd) *
                    1.0 * np.random.random())

                opt_vars_cls.asymm_n_iters = int(
                    self._sett_asymm_n_iters_lbd + (
                        self._sett_asymm_n_iters_ubd -
                        self._sett_asymm_n_iters_lbd) *
                    1.0 * np.random.random())

                if self._sett_asymm_type == 2:
                    opt_vars_cls.prob_center = (
                        self._sett_asymm_prob_center_lbd + (
                            self._sett_asymm_prob_center_ubd -
                            self._sett_asymm_prob_center_lbd) *
                        1.0 * np.random.random())

        # If data has magnitudes of values smaller than
        # the minimum here, then we have the problem of randomizing
        # everything.
        inc = 0.001  # self._rr.data.min() * 0.0

        asymms_rand_err = -inc + (
            (2.0 * inc) * np.random.random(self._data_ref_shape[0]))

        # asymms_rand_err[
        #     (asymms_rand_err > (0.3 * -inc)) &
        #     (asymms_rand_err < (0.3 * +inc))] *= 0.01

        opt_vars_cls.asymms_rand_err = asymms_rand_err
        return

    @GTGBase._timer_wrap
    def _asymmetrize_type_1(self, data, probs, opt_vars_cls):

        n_levels = opt_vars_cls.n_levels
        max_shift_exp = opt_vars_cls.max_shift_exp
        max_shift = opt_vars_cls.max_shift
        pre_vals_ratio = opt_vars_cls.pre_vals_ratio
        asymms_rand_err = opt_vars_cls.asymms_rand_err
        asymm_n_iters = opt_vars_cls.asymm_n_iters

        out_data = asymmetrize_type_1_cy(
            data,
            probs,
            n_levels,
            max_shift_exp,
            max_shift,
            pre_vals_ratio,
            asymm_n_iters,
            asymms_rand_err)

        return out_data

    @GTGBase._timer_wrap
    def _asymmetrize_type_2(self, data, probs, opt_vars_cls):

        n_levels = opt_vars_cls.n_levels
        max_shift_exp = opt_vars_cls.max_shift_exp
        max_shift = opt_vars_cls.max_shift
        pre_vals_ratio = opt_vars_cls.pre_vals_ratio
        asymms_rand_err = opt_vars_cls.asymms_rand_err
        asymm_n_iters = opt_vars_cls.asymm_n_iters
        prob_center = opt_vars_cls.prob_center

        out_data = asymmetrize_type_3_cy(
            data,
            probs,
            n_levels,
            max_shift_exp,
            max_shift,
            pre_vals_ratio,
            asymm_n_iters,
            asymms_rand_err,
            prob_center)

        return out_data

    # @GTGBase._timer_wrap
    # def _asymmetrize_type_1(self, data, probs, opt_vars_cls):
    #
    #     n_levels = opt_vars_cls.n_levels
    #     max_shift_exp = opt_vars_cls.max_shift_exp
    #     max_shift = opt_vars_cls.max_shift
    #     pre_vals_ratio = opt_vars_cls.pre_vals_ratio
    #     asymms_rand_err = opt_vars_cls.asymms_rand_err
    #
    #     out_data = np.empty_like(data)
    #
    #     for i in range(data.shape[1]):
    #
    #         data_i = data[:, i].copy()
    #         probs_i = probs[:, i].copy()
    #
    #         vals_sort = np.sort(data_i)
    #
    #         levels = (probs_i * n_levels).astype(int)
    #
    #         asymm_vals = data_i.copy()
    #         for level in range(n_levels):
    #             asymm_vals_i = asymm_vals.copy()
    #
    #             max_shift_level = (
    #                 max_shift -
    #                 (max_shift * ((level / n_levels) ** max_shift_exp)))
    #
    #             # max_shift_level = round(max_shift_level)
    #             max_shift_level = int(max_shift_level)
    #
    #             if max_shift_level <= 0:
    #                 shifts = range(max_shift_level, 0, 1)
    #
    #             else:
    #                 shifts = range(1, max_shift_level + 1)
    #
    #             for shift in shifts:
    #                 asymm_vals_i = np.roll(asymm_vals_i, shift)
    #
    #                 idxs_to_shift = levels <= level
    #
    #                 asymm_vals[idxs_to_shift] = (
    #                     asymm_vals[idxs_to_shift] * pre_vals_ratio +
    #                     asymm_vals_i[idxs_to_shift] * (1.0 - pre_vals_ratio))
    #
    #         # if asymms_rand_err is not None:
    #         asymm_vals += asymms_rand_err
    #
    #         asymm_vals = vals_sort[np.argsort(np.argsort(asymm_vals))]
    #
    #         out_data[:, i] = asymm_vals
    #
    #     return out_data

    def _get_smoothed_array(self, data, half_window_size):

        n_vals = data.size

        smoothed_arr = np.empty(n_vals, dtype=float)

        data_padded = np.concatenate((
            data[::-1][:half_window_size], data, data[:half_window_size]))

        rel_dists = np.concatenate((
            np.arange(half_window_size, -1, -1.0,),
            np.arange(1.0, half_window_size + 1.0)))

        rel_dists /= rel_dists.max() + 1.0

        if True:
            if False:
                kern_ftn = np.vectorize(triangular_kern)

            elif False:
                kern_ftn = np.vectorize(gaussian_kern)

            elif True:
                kern_ftn = np.vectorize(tricube_kern)

            else:
                raise Exception

            window_wts = kern_ftn(rel_dists)
            window_wts /= window_wts.sum()

        else:
            window_wts = np.array([0.1, 0.8, 0.1])

        for i in range(n_vals):
            smoothed_arr[i] = np.nansum(
                data_padded[i: i + 2 * half_window_size + 1] * window_wts)

        return smoothed_arr

    @GTGBase._timer_wrap
    def _adjust_init_spec(self, opt_vars_cls):

        data = self._rs.init_data.copy()
        probs = self._rs.init_probs.copy()

        data_init = data.copy()

        plot_flag = True

        if plot_flag:
            import matplotlib.pyplot as plt

            ft_data_ref_corr = self._get_cumm_ft_corr(
                self._rr.data_ft_coeffs, self._rr.data_ft_coeffs)[:, 0]

            ft_probs_ref_corr = self._get_cumm_ft_corr(
                self._rr.probs_ft_coeffs, self._rr.probs_ft_coeffs)[:, 0]

            ref_periods = (ft_data_ref_corr.size * 2) / (
                np.arange(1, ft_data_ref_corr.size + 1))

            plt.figure(figsize=(13, 7))

            plt.semilogx(
                ref_periods,
                ft_data_ref_corr,
                alpha=0.75,
                color='r',
                lw=1.5,
                label='ref_data',
                zorder=3)

            plt.semilogx(
                ref_periods,
                ft_probs_ref_corr,
                alpha=0.75,
                color='r',
                lw=1.5,
                ls='--',
                label='ref_probs',
                zorder=3)

        best_obj_val = np.inf

        best_ft = self._rs.ft.copy()

        best_updt_ctr = 0
        max_not_best_updt_iters = 2  # Consecutive.

        # phs_data_init = None
        # phs_probs_init = None

        opt_vars_cls.mxn_ratio_margs = 1.0
        opt_vars_cls.mxn_ratio_probs = 0.0

        opt_vars_cls_rev = deepcopy(opt_vars_cls)

        opt_vars_cls_rev.max_shift *= -1
        opt_vars_cls_rev.max_shift += 1

        ft_iter = 0
        max_ft_iters = 5
        while True:
            ft_data_init = np.fft.rfft(data_init, axis=0)
            # mag_data_init = np.abs(ft_data_init)

            # if phs_data_init is None:
                # phs_data_init = np.angle(ft_data_init)

            ft_probs_init = np.fft.rfft(probs, axis=0)
            # mag_probs_init = np.abs(ft_probs_init)

            # if phs_probs_init is None:
                # phs_probs_init = np.angle(ft_probs_init)

            if plot_flag:
                ft_data_init_corr = self._get_cumm_ft_corr(
                    ft_data_init, ft_data_init)[:, 0]

                ft_probs_init_corr = self._get_cumm_ft_corr(
                    ft_probs_init, ft_probs_init)[:, 0]

            # Using probs as data works slightly better
            # than using data.
            if self._sett_asymm_type == 1:
                data = self._asymmetrize_type_1(data_init, probs, opt_vars_cls)

            elif self._sett_asymm_type == 2:
                data = self._asymmetrize_type_2(data_init, probs, opt_vars_cls)

            else:
                raise NotImplementedError

            probs = self._get_probs(data, True)

            # data = np.empty_like(self._data_ref_rltzn_srtd, dtype=np.float64)
            #
            # for i in range(self._data_ref_n_labels):
            #     data[:, i] = self._data_ref_rltzn_srtd[
            #         np.argsort(np.argsort(probs[:, i])), i]
            #
            #     data[:, i] -= data[:, i].mean()
            #     data[:, i] /= data[:, i].std()
            #
            #     probs[:, i] -= probs[:, i].mean()
            #     probs[:, i] /= probs[:, i].std()
            #
            # probs = self._get_probs(
            #     (data * opt_vars_cls.mxn_ratio_margs) +
            #     (probs * opt_vars_cls.mxn_ratio_probs),
            #     True)
            #==================================================================

            data = np.empty_like(
                self._data_ref_rltzn_srtd, dtype=np.float64)

            for i in range(self._data_ref_n_labels):
                data[:, i] = self._data_ref_rltzn_srtd[
                    np.argsort(np.argsort(probs[:, i])), i]

            ft_data_asym = np.fft.rfft(data, axis=0)
            # mag_data_asym = np.abs(ft_data_asym)
            phs_data_asym = np.angle(ft_data_asym)

            # ft_probs_asym = np.fft.rfft(probs, axis=0)
            # mag_probs_asym = np.abs(ft_probs_asym)
            # phs_probs_asym = np.angle(ft_probs_asym)

            ft_data_asym = np.empty_like(ft_data_init)
            ft_data_asym.real[:] = np.cos(phs_data_asym) * self._rr.data_ft_coeffs_mags
            ft_data_asym.imag[:] = np.sin(phs_data_asym) * self._rr.data_ft_coeffs_mags
            #
            # ft_probs_asym = np.empty_like(ft_probs_init)
            # ft_probs_asym.real[:] = np.cos(phs_probs_asym) * self._rr.probs_ft_coeffs_mags
            # ft_probs_asym.imag[:] = np.sin(phs_probs_asym) * self._rr.probs_ft_coeffs_mags

            self._rs.ft = ft_data_asym
            self._rs.mag_spec = np.abs(self._rs.ft)
            self._rs.phs_spec = np.angle(self._rs.ft)

            self._update_sim_no_prms()

            obj_val = self._get_obj_ftn_val().sum()

            if obj_val < best_obj_val:

                best_obj_val = obj_val

                best_ft = ft_data_asym.copy()

                best_updt_ctr = 0

            else:
                best_updt_ctr += 1

            if best_updt_ctr == max_not_best_updt_iters:
                break
            #==================================================================

            # sq_diff = ((self._rr.data_ft_coeffs_mags - mag_data_asym)[1:] /
            #            self._rr.data_ft_coeffs_mags[1:].sum()) ** 2
            #
            # sq_diff += ((self._rr.probs_ft_coeffs_mags - mag_probs_asym)[1:] /
            #            self._rr.probs_ft_coeffs_mags[1:].sum()) ** 2
            #
            # obj_val = sq_diff.sum()
            #
            # if obj_val < best_obj_val:
            #     best_obj_val = obj_val
            #
            #     best_ft = ft_data_asym.copy()
            #     # best_ft = ft_data_init.copy()

            if plot_flag:
                print(ft_iter, f'{obj_val:0.3E}')

            # mag_data_init = mag_data_init + (self._rr.mag_spec - mag_data_asym)

            # mag_data_init += (self._rr.data_ft_coeffs_mags - mag_data_asym)
            # mag_probs_init += (self._rr.probs_ft_coeffs_mags - mag_probs_asym)

            # ft_data_init += (self._rr.data_ft_coeffs - ft_data_asym)
            # ft_probs_init += (self._rr.probs_ft_coeffs - ft_probs_asym)

            # ft_data_init += (ft_data_asym - self._rr.data_ft_coeffs)
            # ft_probs_init += (ft_probs_asym - self._rr.probs_ft_coeffs)

            # ft_data_init[~self._rr.prsrv_coeffs_idxs] *= -1 + -1j
            # ft_probs_init[~self._rr.prsrv_coeffs_idxs] *= -1 + -1j

            # ft_data_init[self._rr.prsrv_coeffs_idxs] = 0
            # ft_probs_init[self._rr.prsrv_coeffs_idxs] = 0

            # if True:
            #     mag_data_init[mag_data_init < 0] = 0
            #     mag_probs_init[mag_probs_init < 0] = 0

            # if self._sett_prsrv_coeffs_set_flag:
            #     mag_data_init[self._rr.prsrv_coeffs_idxs,:] = (
            #         self._rr.data_ft_coeffs_mags[self._rr.prsrv_coeffs_idxs,:])
            #
            #     mag_probs_init[self._rr.prsrv_coeffs_idxs,:] = (
            #         self._rr.probs_ft_coeffs_mags[self._rr.prsrv_coeffs_idxs,:])

            if self._sett_asymm_type == 1:
                data = self._asymmetrize_type_1(
                    self._rs.probs,
                    self._rs.probs,
                    opt_vars_cls_rev)

            elif self._sett_asymm_type == 2:
                data = self._asymmetrize_type_2(
                    self._rs.probs,
                    self._rs.probs,
                    opt_vars_cls_rev)

            else:
                raise NotImplementedError

            probs = self._get_probs(data, True)

            data_init = np.empty_like(
                self._data_ref_rltzn_srtd, dtype=np.float64)

            for i in range(self._data_ref_n_labels):
                data_init[:, i] = self._data_ref_rltzn_srtd[
                    np.argsort(np.argsort(probs[:, i])), i]

            # ft_data_init = np.empty_like(ft_data_init)
            # ft_data_init.real[:] = np.cos(phs_data_init) * mag_data_init
            # ft_data_init.imag[:] = np.sin(phs_data_init) * mag_data_init
            #
            # ft_probs_init = np.empty_like(ft_probs_init)
            # ft_probs_init.real[:] = np.cos(phs_probs_init) * mag_probs_init
            # ft_probs_init.imag[:] = np.sin(phs_probs_init) * mag_probs_init
            #
            # data_init = np.fft.irfft(ft_data_init, axis=0)
            # data_init -= data_init.mean(axis=0)
            # data_init /= data_init.std(axis=0)
            #
            # probs_init = np.fft.irfft(ft_probs_init, axis=0)
            # probs_init -= probs_init.mean(axis=0)
            # probs_init /= probs_init.std(axis=0)

            # probs = self._get_probs(
            #     (data_init * opt_vars_cls.mxn_ratio_margs) +
            #     (probs_init * opt_vars_cls.mxn_ratio_probs),
            #     True)

            # data_init = np.empty_like(
            #     self._data_ref_rltzn_srtd, dtype=np.float64)
            #
            # for i in range(self._data_ref_n_labels):
            #     data_init[:, i] = self._data_ref_rltzn_srtd[
            #         np.argsort(np.argsort(probs[:, i])), i]

            if plot_flag:
                if ft_iter == 0:
                    lab_init_data = 'init_data'
                    lab_asym_data = 'asym_data'
                    lab_init_probs = 'init_probs'
                    lab_asym_probs = 'asym_probs'

                else:
                    lab_init_data = lab_asym_data = None
                    lab_init_probs = lab_asym_probs = None

                if ft_iter == (max_ft_iters - 1):
                    lw = 2

                    alpha = 0.75

                else:
                    lw = 1

                    alpha = 0.1

                ft_data_asym_corr = self._get_cumm_ft_corr(
                    self._rs.data_ft_coeffs, self._rs.data_ft_coeffs)[:, 0]

                ft_probs_asym_corr = self._get_cumm_ft_corr(
                    self._rs.probs_ft_coeffs, self._rs.probs_ft_coeffs)[:, 0]

                plt.semilogx(
                    ref_periods,
                    ft_data_init_corr,
                    alpha=alpha,
                    color='g',
                    lw=lw,
                    label=lab_init_data,
                    zorder=1)

                plt.semilogx(
                    ref_periods,
                    ft_data_asym_corr,
                    alpha=alpha,
                    color='b',
                    lw=lw,
                    label=lab_asym_data,
                    zorder=2)

                plt.semilogx(
                    ref_periods,
                    ft_probs_init_corr,
                    alpha=alpha,
                    color='g',
                    lw=lw,
                    ls='--',
                    label=lab_init_probs,
                    zorder=1)

                plt.semilogx(
                    ref_periods,
                    ft_probs_asym_corr,
                    alpha=alpha,
                    color='b',
                    lw=lw,
                    ls='--',
                    label=lab_asym_probs,
                    zorder=2)

            ft_iter += 1

            if ft_iter == max_ft_iters:
                break

        if plot_flag:
            plt.grid()

            plt.ylim(-0.05, +1.05)

            plt.gca().set_axisbelow(True)

            plt.legend(framealpha=0.7)

            plt.title(
                f'n_levels: {opt_vars_cls.n_levels}, '
                f'max_shift_exp: {opt_vars_cls.max_shift_exp:0.2f}, '
                f'max_shift: {opt_vars_cls.max_shift}\n'
                f'pre_vals_ratio: {opt_vars_cls.pre_vals_ratio:0.2f}, '
                f'asymm_n_iters: {opt_vars_cls.asymm_n_iters}, '
                f'prob_center: {opt_vars_cls.prob_center:0.2f}')

            plt.ylabel('Cummulative FT correlation')

            plt.xlabel(f'Period (steps)')

            plt.xlim(plt.xlim()[::-1])

            out_name = f'P:/Downloads/ss__ft_data_init_asym6.png'

            plt.savefig(str(out_name), bbox_inches='tight')

            plt.close()

            raise Exception

        return best_ft

    @GTGBase._timer_wrap
    def _run_iaaft(self, opt_vars_cls, plain_iaaft_flag=False):

        phs_spec_data = self._rr.data_ft_coeffs_phss.copy()
        phs_spec_probs = self._rr.probs_ft_coeffs_phss.copy()

        iaaft_n_iters = self._sett_ann_iaaft_n_iters

        if plain_iaaft_flag:
            data = self._rs.data.copy()
            probs = self._rs.probs.copy()

            iaaft_n_iters = self._sett_ann_iaaft_n_iters

            # Better this way.
            opt_vars_cls.mxn_ratio_margs = 1.0
            opt_vars_cls.mxn_ratio_probs = 0.0

            mag_spec_data = self._rr.data_ft_coeffs_mags.copy()
            mag_spec_probs = self._rr.probs_ft_coeffs_mags.copy()

        else:
            if True:
                data = self._rs.init_data.copy()
                probs = self._rs.init_probs.copy()

                mag_spec_data = self._rr.data_ft_coeffs_mags.copy()
                mag_spec_probs = self._rr.probs_ft_coeffs_mags.copy()

                # self._rs.ft = np.fft.rfft(data, axis=0)
                # self._rs.mag_spec = np.abs(self._rs.ft)
                # self._rs.phs_spec = np.angle(self._rs.ft)
                #
                # self._update_sim_no_prms()
                #
                # best_ft = self._rs.ft.copy()
                # best_obj_val = self._get_obj_ftn_val().sum()

                iaaft_n_iters = self._sett_ann_iaaft_n_iters

        # else:
        #     if True:
        #         data = self._rs.init_data.copy()
        #         probs = self._rs.init_probs.copy()
        #
        #         if self._sett_asymm_set_flag:
        #
        #             # Using probs as data works slightly better
        #             # than using data.
        #
        #             if self._sett_asymm_type == 1:
        #                 data = self._asymmetrize_type_1(
        #                     data, probs, opt_vars_cls)
        #
        #             elif self._sett_asymm_type == 2:
        #                 data = self._asymmetrize_type_2(
        #                     data, probs, opt_vars_cls)
        #
        #             else:
        #                 raise NotImplementedError
        #
        #             if False:
        #                 assert self._rs.data.shape[1] == 1, (
        #                     'self._get_smoothed_array requires 1 column only!')
        #
        #                 data = self._get_smoothed_array(
        #                     data.ravel(), 4).reshape(-1, 1)
        #
        #             probs = self._get_probs(data, True)
        #
        #             self._rs.probs = probs
        #
        #             self._rs.data = np.empty_like(
        #                 self._data_ref_rltzn_srtd, dtype=np.float64)
        #
        #             for i in range(self._data_ref_n_labels):
        #                 self._rs.data[:, i] = self._data_ref_rltzn_srtd[
        #                     np.argsort(np.argsort(probs[:, i])), i]
        #
        #             data = self._rs.data
        #
        #             self._rs.ft = np.fft.rfft(data, axis=0)
        #             self._rs.mag_spec = np.abs(self._rs.ft)
        #             self._rs.phs_spec = np.angle(self._rs.ft)
        #
        #             self._update_sim_no_prms()
        #
        #             # Only needed here.
        #             best_ft = self._rs.ft.copy()
        #             best_obj_val = self._get_obj_ftn_val().sum()
        #
        #             # This has to be after the asymmetrizing.
        #             if False:
        #                 mag_spec_data = self._rs.data_ft_coeffs_mags.copy()
        #
        #             else:
        #                 mag_spec_data = self._rr.data_ft_coeffs_mags.copy()
        #
        #             if False:
        #                 mag_spec_probs = self._rs.probs_ft_coeffs_mags.copy()
        #
        #             else:
        #                 mag_spec_probs = self._rr.probs_ft_coeffs_mags.copy()
        #
        #             iaaft_n_iters = 0  # self._sett_ann_iaaft_n_iters
        #
        #     else:
        #         best_ft = self._adjust_init_spec(opt_vars_cls)
        #
        #         iaaft_n_iters = 0  # self._sett_ann_iaaft_n_iters

        order_sdiffs = np.full(iaaft_n_iters, np.nan)

        # obj_vals = np.full(iaaft_n_iters, np.inf)

        auto_spec_flag = True

        if self._rs.shape[1] > 1:
            cross_spec_flag = True

        else:
            cross_spec_flag = False

        reorder_idxs_old = np.random.randint(
            0, self._rs.data.shape[0], size=self._rs.data.shape)

        # best_updt_ctr = 0
        # max_not_best_updt_iters = 2  # Consecutive.

        # if self._sett_prsrv_coeffs_set_flag:
        #     mag_spec_data[self._rr.prsrv_coeffs_idxs,:] = (
        #         self._rr.data_ft_coeffs_mags[self._rr.prsrv_coeffs_idxs,:])
        #
        #     mag_spec_probs[self._rr.prsrv_coeffs_idxs,:] = (
        #         self._rr.probs_ft_coeffs_mags[self._rr.prsrv_coeffs_idxs,:])

        stn_ctr = 0
        for i in range(iaaft_n_iters):

            if opt_vars_cls.mxn_ratio_margs:
                sim_ft_margs = np.fft.rfft(data, axis=0)

            if opt_vars_cls.mxn_ratio_probs:
                sim_ft_probs = np.fft.rfft(probs, axis=0)

            # Marginals auto.
            if opt_vars_cls.mxn_ratio_margs and auto_spec_flag:
                sim_phs_margs = np.angle(sim_ft_margs)

                if self._sett_prsrv_coeffs_set_flag:
                    sim_phs_margs[self._rr.prsrv_coeffs_idxs,:] = (
                        phs_spec_data[self._rr.prsrv_coeffs_idxs,:])

                sim_ft_new = np.empty_like(sim_ft_margs)

                sim_ft_new.real[:] = np.cos(sim_phs_margs) * mag_spec_data
                sim_ft_new.imag[:] = np.sin(sim_phs_margs) * mag_spec_data

                sim_ft_new[0,:] = 0

                sim_ift_margs_auto = np.fft.irfft(sim_ft_new, axis=0)

                # reorder_idxs_new = np.empty_like(reorder_idxs_old)
                # for k in range(self._data_ref_n_labels):
                #     reorder_idxs_new[:, k] = np.argsort(
                #         np.argsort(sim_ift_margs_auto[:, k]))
                #
                # for k in range(self._data_ref_n_labels):
                #     sim_ift_margs_auto[:, k] = (
                #         self._data_ref_rltzn_srtd[reorder_idxs_new[:, k], k])

                # sim_ift_margs_auto -= sim_ift_margs_auto.mean(axis=0)
                # sim_ift_margs_auto /= sim_ift_margs_auto.std(axis=0)

            else:
                sim_ift_margs_auto = 0.0

            # Ranks auto.
            if opt_vars_cls.mxn_ratio_probs and auto_spec_flag:
                sim_phs = np.angle(sim_ft_probs)

                if self._sett_prsrv_coeffs_set_flag:
                    sim_phs[self._rr.prsrv_coeffs_idxs,:] = (
                        phs_spec_probs[self._rr.prsrv_coeffs_idxs,:])

                sim_ft_new = np.empty_like(sim_ft_probs)

                sim_ft_new.real[:] = np.cos(sim_phs) * mag_spec_probs

                sim_ft_new.imag[:] = np.sin(sim_phs) * mag_spec_probs

                # sim_ft_new[self._rr.prsrv_coeffs_idxs,:] = 0

                sim_ft_new[0,:] = 0

                sim_ift_probs_auto = np.fft.irfft(sim_ft_new, axis=0)

                # sim_ift_probs_auto = self._get_probs(sim_ift_probs_auto, True)

                # sim_ift_probs_auto -= sim_ift_probs_auto.mean(axis=0)
                # sim_ift_probs_auto /= sim_ift_probs_auto.std(axis=0)

            else:
                sim_ift_probs_auto = 0.0
            #==================================================================

            # Marginals cross.
            if opt_vars_cls.mxn_ratio_margs and cross_spec_flag:
                sim_mag = self._rr.data_ft_coeffs_mags.copy()

                sim_phs = (
                    np.angle(sim_ft_margs[:, [stn_ctr]]) +
                    phs_spec_data - phs_spec_data[:, [stn_ctr]])

                sim_phs[0,:] = phs_spec_data[0,:]

                if self._sett_prsrv_coeffs_set_flag:
                    sim_phs[self._rr.prsrv_coeffs_idxs,:] = (
                        phs_spec_data[self._rr.prsrv_coeffs_idxs,:])

                sim_ft_new = np.empty_like(sim_ft_margs)

                sim_ft_new.real[:] = np.cos(sim_phs) * sim_mag
                sim_ft_new.imag[:] = np.sin(sim_phs) * sim_mag

                sim_ft_new[0,:] = 0

                sim_ift_margs_cross = np.fft.irfft(sim_ft_new, axis=0)

                # reorder_idxs_new = np.empty_like(reorder_idxs_old)
                # for k in range(self._data_ref_n_labels):
                #     reorder_idxs_new[:, k] = np.argsort(
                #         np.argsort(sim_ift_margs_cross[:, k]))
                #
                # for k in range(self._data_ref_n_labels):
                #     sim_ift_margs_cross[:, k] = (
                #         self._data_ref_rltzn_srtd[reorder_idxs_new[:, k], k])
                #
                # sim_ift_margs_cross -= sim_ift_margs_cross.mean(axis=0)
                sim_ift_margs_cross /= sim_ift_margs_cross.std(axis=0)

            else:
                sim_ift_margs_cross = 0.0

            # Ranks cross.
            if opt_vars_cls.mxn_ratio_probs and cross_spec_flag:
                sim_mag = self._rr.probs_ft_coeffs_mags.copy()

                sim_phs = (
                    np.angle(sim_ft_probs[:, [stn_ctr]]) +
                    phs_spec_probs - phs_spec_probs[:, [stn_ctr]])

                sim_phs[0,:] = phs_spec_probs[0,:]

                if self._sett_prsrv_coeffs_set_flag:
                    sim_phs[self._rr.prsrv_coeffs_idxs,:] = (
                        phs_spec_probs[self._rr.prsrv_coeffs_idxs,:])

                sim_ft_new = np.empty_like(sim_ft_probs)

                sim_ft_new.real[:] = np.cos(sim_phs) * sim_mag
                sim_ft_new.imag[:] = np.sin(sim_phs) * sim_mag

                sim_ft_new[0,:] = 0

                sim_ift_probs_cross = np.fft.irfft(sim_ft_new, axis=0)

                # sim_ift_probs_cross = self._get_probs(
                #     sim_ift_probs_cross, True)
                #
                # sim_ift_probs_cross -= sim_ift_probs_cross.mean(axis=0)
                sim_ift_probs_cross /= sim_ift_probs_cross.std(axis=0)

            else:
                sim_ift_probs_cross = 0.0
            #==================================================================

            # Their sum.
            sim_ift = (
                (opt_vars_cls.mxn_ratio_margs * sim_ift_margs_auto) +
                (opt_vars_cls.mxn_ratio_probs * sim_ift_probs_auto) +
                (opt_vars_cls.mxn_ratio_margs * sim_ift_margs_cross) +
                (opt_vars_cls.mxn_ratio_probs * sim_ift_probs_cross)
                )

            assert isinstance(sim_ift, np.ndarray), type(sim_ift)

            reorder_idxs_new = np.empty_like(reorder_idxs_old)
            for k in range(self._data_ref_n_labels):
                reorder_idxs_new[:, k] = np.argsort(np.argsort(sim_ift[:, k]))

            order_sdiff = 1 - np.corrcoef(
                reorder_idxs_old.ravel(), reorder_idxs_new.ravel())[0, 1]

            assert order_sdiff >= 0, order_sdiff

            order_sdiffs[i] = order_sdiff

            reorder_idxs_old = reorder_idxs_new

            data = np.empty_like(data)

            for k in range(self._data_ref_n_labels):
                data[:, k] = self._data_ref_rltzn_srtd[
                    reorder_idxs_old[:, k], k]

            probs = self._get_probs(data, True)

            if order_sdiff <= (1e-3):
                # Nothing changed.
                break

            stn_ctr += 1
            if stn_ctr == data.shape[1]:
                stn_ctr = 0
            #==================================================================

            # if not plain_iaaft_flag:
            #     self._rs.ft = np.fft.rfft(data, axis=0)
            #     self._rs.mag_spec = np.abs(self._rs.ft)
            #     self._rs.phs_spec = np.angle(self._rs.ft)
            #
            #     self._update_sim_no_prms()
            #
            #     obj_val = self._get_obj_ftn_val().sum()
            #
            #     obj_vals[i] = obj_val
            #
            #     if obj_val < best_obj_val:
            #
            #         best_obj_val = obj_val
            #
            #         best_ft = self._rs.ft.copy()
            #
            #         best_updt_ctr = 0
            #
            #     else:
            #         best_updt_ctr += 1
            #
            #     if best_updt_ctr == max_not_best_updt_iters:
            #         break
            #==================================================================

        if self._sett_asymm_set_flag and not plain_iaaft_flag:

            # Using probs as data works slightly better
            # than using data.

            if self._sett_asymm_type == 1:
                data = self._asymmetrize_type_1(
                    probs, probs, opt_vars_cls)

            elif self._sett_asymm_type == 2:
                data = self._asymmetrize_type_2(
                    probs, probs, opt_vars_cls)

            else:
                raise NotImplementedError

            if False:
                assert self._rs.data.shape[1] == 1, (
                    'self._get_smoothed_array requires 1 column only!')

                data = self._get_smoothed_array(
                    data.ravel(), 4).reshape(-1, 1)

            probs = self._get_probs(data, True)

            self._rs.probs = probs

            self._rs.data = np.empty_like(
                self._data_ref_rltzn_srtd, dtype=np.float64)

            for i in range(self._data_ref_n_labels):
                self._rs.data[:, i] = self._data_ref_rltzn_srtd[
                    np.argsort(np.argsort(probs[:, i])), i]

            data = self._rs.data

            self._rs.ft = np.fft.rfft(data, axis=0)
            # self._rs.mag_spec = np.abs(self._rs.ft)
            # self._rs.phs_spec = np.angle(self._rs.ft)
            #
            # self._update_sim_no_prms()

            # Only needed here.
            best_ft = self._rs.ft.copy()

        elif not plain_iaaft_flag:
            # self._rs.ft = np.fft.rfft(data, axis=0)
            # self._rs.mag_spec = np.abs(self._rs.ft)
            # self._rs.phs_spec = np.angle(self._rs.ft)

            best_ft = np.fft.rfft(data, axis=0)

        if plain_iaaft_flag:
            best_ft = np.fft.rfft(data, axis=0)

        self._rs.ft = best_ft
        self._rs.mag_spec = np.abs(self._rs.ft)
        self._rs.phs_spec = np.angle(self._rs.ft)
        return

    def _set_init_iaaft(self, opt_vars_cls):

        self._run_iaaft(opt_vars_cls, True)

        self._update_sim_no_prms()

        self._rs.init_data = self._rs.data.copy()
        self._rs.init_probs = self._rs.probs.copy()
        return

    def _gen_gnrc_rltzn(self, args):

        (rltzn_iter,
         init_temp,
        ) = args

        assert self._alg_verify_flag, 'Call verify first!'

        beg_time = default_timer()

        assert isinstance(rltzn_iter, int), 'rltzn_iter not integer!'

        if self._alg_ann_runn_auto_init_temp_search_flag:
            temp = init_temp

        else:
            # _alg_rltzn_iter should be only set when annealing is started.
            self._alg_rltzn_iter = rltzn_iter

            assert 0 <= rltzn_iter < self._sett_misc_n_rltzns, (
                'Invalid rltzn_iter!')

            temp = self._sett_ann_init_temp

        if self._data_ref_rltzn.ndim != 2:
            raise NotImplementedError('Implemention for 2D only!')

        # Start with a scrambled series.
        self._gen_sim_aux_data()

        # IAAFTSA variable.
        # Depends on the number of mixing ratio variables.
        # For now, two only.
        opt_vars_cls_old = OPTVARS()

        self._init_iaaft_opt_vars(
            opt_vars_cls_old,
            self._alg_ann_runn_auto_init_temp_search_flag)

        # IAAFTSA variable.
        self._set_init_iaaft(opt_vars_cls_old)

        # IAAFTSA variable.
        self._update_sim(opt_vars_cls_old, False)

        old_obj_val = self._get_obj_ftn_val().sum()

        # Initialize sim anneal variables.
        iter_ctr = 0

        if self._sett_ann_auto_init_temp_trgt_acpt_rate is not None:
            acpt_rate = self._sett_ann_auto_init_temp_trgt_acpt_rate

        else:
            acpt_rate = 1.0

        # IAAFTSA variable.
        mxn_ratio_red_rate = self._get_mxn_ratio_red_rate(
            iter_ctr, acpt_rate, 1.0)

        if self._alg_ann_runn_auto_init_temp_search_flag:
            stopp_criteria = (
                (iter_ctr <= self._sett_ann_auto_init_temp_niters),
                )

        else:
            iters_wo_acpt = 0
            tol = np.inf

            iter_wo_min_updt = 0

            tols_dfrntl = deque(maxlen=self._sett_ann_obj_tol_iters)

            acpts_rjts_dfrntl = deque(maxlen=self._sett_ann_acpt_rate_iters)

            stopp_criteria = self._get_stopp_criteria(
                (iter_ctr,
                 iters_wo_acpt,
                 tol,
                 temp,
                 mxn_ratio_red_rate,
                 acpt_rate,
                 iter_wo_min_updt))

        self._update_snapshot()

        # Initialize diagnostic variables.
        acpts_rjts_all = []

        if not self._alg_ann_runn_auto_init_temp_search_flag:
            tols = []

            obj_vals_all = []

            obj_val_min = old_obj_val
            obj_vals_min = []

            # IAAFTSA variable.
            mxn_ratio_red_rates = [[iter_ctr, mxn_ratio_red_rate]]

            # IAAFTSA variable.
            mxn_ratios_probs = []

            # IAAFTSA variable.
            mxn_ratios_margs = []

            # IAAFTSA variable.
            n_levelss = []

            # IAAFTSA variable.
            max_shift_exps = []

            # IAAFTSA variable.
            max_shifts = []

            # IAAFTSA variable.
            pre_vals_ratios = []

            # IAAFTSA variable.
            asymm_n_iterss = []

            # IAAFTSA variable.
            prob_centers = []

            temps = [[iter_ctr, temp]]

            acpt_rates_dfrntl = [[iter_ctr, acpt_rate]]

            obj_vals_all_indiv = []

        else:
            pass

        self._rs.ft_best = self._rs.ft.copy()

        while all(stopp_criteria):

            #==============================================================
            # Simulated annealing start.
            #==============================================================

            # IAAFTSA variable.
            opt_vars_cls_new = self._get_next_iter_vars(
                mxn_ratio_red_rate, opt_vars_cls_old)

            # IAAFTSA variable.
            self._update_sim(opt_vars_cls_new, False)

            new_obj_val_indiv = self._get_obj_ftn_val()
            new_obj_val = new_obj_val_indiv.sum()

            old_new_diff = old_obj_val - new_obj_val

            old_new_adj_diff = old_new_diff - (
                self._sett_ann_acpt_thresh * old_obj_val)

            if old_new_adj_diff > 0:
                accept_flag = True

            else:
                rand_p = np.random.random()

                boltz_p = np.exp(old_new_adj_diff / temp)

                if rand_p < boltz_p:
                    accept_flag = True

                else:
                    accept_flag = False

            if self._alg_force_acpt_flag:
                accept_flag = True
                self._alg_force_acpt_flag = False

            if accept_flag:
                # IAAFTSA variable.
                opt_vars_cls_old = opt_vars_cls_new

                old_obj_val = new_obj_val

                self._update_snapshot()

            else:
                # IAAFTSA variable.
                self._update_sim(opt_vars_cls_old, True)

            iter_ctr += 1

            #==============================================================
            # Simulated annealing end.
            #==============================================================

            acpts_rjts_all.append(accept_flag)

            if self._alg_ann_runn_auto_init_temp_search_flag:
                stopp_criteria = (
                    (iter_ctr <= self._sett_ann_auto_init_temp_niters),
                    )

            else:
                obj_vals_all_indiv.append(new_obj_val_indiv)

                # IAAFTSA variable.
                mxn_ratios_probs.append(opt_vars_cls_new.mxn_ratio_probs)

                # IAAFTSA variable.
                mxn_ratios_margs.append(opt_vars_cls_new.mxn_ratio_margs)

                # IAAFTSA variable.
                n_levelss.append(opt_vars_cls_new.n_levels)

                # IAAFTSA variable.
                max_shift_exps.append(opt_vars_cls_new.max_shift_exp)

                # IAAFTSA variable.
                max_shifts.append(opt_vars_cls_new.max_shift)

                # IAAFTSA variable.
                pre_vals_ratios.append(opt_vars_cls_new.pre_vals_ratio)

                # IAAFTSA variable.
                asymm_n_iterss.append(opt_vars_cls_new.asymm_n_iters)

                # IAAFTSA variable.
                prob_centers.append(opt_vars_cls_new.prob_center)

                if new_obj_val < obj_val_min:
                    iter_wo_min_updt = 0

                    self._rs.ft_best = self._rs.ft.copy()

                else:
                    iter_wo_min_updt += 1

                tols_dfrntl.append(abs(old_new_diff))

                obj_val_min = min(obj_val_min, new_obj_val)

                obj_vals_min.append(obj_val_min)
                obj_vals_all.append(new_obj_val)

                acpts_rjts_dfrntl.append(accept_flag)

                if iter_ctr >= tols_dfrntl.maxlen:
                    tol = sum(tols_dfrntl) / float(tols_dfrntl.maxlen)

                    assert np.isfinite(tol), 'Invalid tol!'

                    tols.append(tol)

                if accept_flag:
                    iters_wo_acpt = 0

                else:
                    iters_wo_acpt += 1

                if iter_ctr >= acpts_rjts_dfrntl.maxlen:
                    acpt_rates_dfrntl.append([iter_ctr - 1, acpt_rate])

                    acpt_rate = (
                        sum(acpts_rjts_dfrntl) /
                        float(acpts_rjts_dfrntl.maxlen))

                    acpt_rates_dfrntl.append([iter_ctr, acpt_rate])

                if (iter_ctr % self._sett_ann_upt_evry_iter) == 0:

                    # Temperature.
                    temps.append([iter_ctr - 1, temp])

                    temp *= self._sett_ann_temp_red_rate

                    assert temp >= 0.0, 'Invalid temp!'

                    temps.append([iter_ctr, temp])

                    # IAAFTSA variable.
                    mxn_ratio_red_rates.append(
                        [iter_ctr - 1, mxn_ratio_red_rate])

                    # IAAFTSA variable.
                    mxn_ratio_red_rate = self._get_mxn_ratio_red_rate(
                        iter_ctr, acpt_rate, mxn_ratio_red_rate)

                    # IAAFTSA variable.
                    mxn_ratio_red_rates.append([iter_ctr, mxn_ratio_red_rate])

                if self._vb:
                    self._show_rltzn_situ(
                        iter_ctr,
                        rltzn_iter,
                        iters_wo_acpt,
                        tol,
                        temp,
                        mxn_ratio_red_rate,  # IAAFTSA variable.
                        acpt_rate,
                        new_obj_val,
                        obj_val_min,
                        iter_wo_min_updt,
                        stopp_criteria)

                stopp_criteria = self._get_stopp_criteria(
                    (iter_ctr,
                     iters_wo_acpt,
                     tol,
                     temp,
                     mxn_ratio_red_rate,  # IAAFTSA variable.
                     acpt_rate,
                     iter_wo_min_updt))

        if self._vb and not self._alg_ann_runn_auto_init_temp_search_flag:
            self._show_rltzn_situ(
                iter_ctr,
                rltzn_iter,
                iters_wo_acpt,
                tol,
                temp,
                mxn_ratio_red_rate,  # IAAFTSA variable.
                acpt_rate,
                new_obj_val,
                obj_val_min,
                iter_wo_min_updt,
                stopp_criteria)

        # Manual update of timer because this function writes timings
        # to the HDF5 file before it returns.
        if '_gen_gnrc_rltzn' not in self._dur_tmr_cumm_call_times:
            self._dur_tmr_cumm_call_times['_gen_gnrc_rltzn'] = 0.0
            self._dur_tmr_cumm_n_calls['_gen_gnrc_rltzn'] = 0.0

        self._dur_tmr_cumm_call_times['_gen_gnrc_rltzn'] += (
            default_timer() - beg_time)

        self._dur_tmr_cumm_n_calls['_gen_gnrc_rltzn'] += 1

        if self._alg_ann_runn_auto_init_temp_search_flag:

            ret = sum(acpts_rjts_all) / len(acpts_rjts_all), temp

        else:
            # _sim_ft set to _sim_ft_best in _update_sim_at_end.
            self._update_ref_at_end()
            self._update_sim_at_end()

            self._rs.label = (
                f'{rltzn_iter:0{len(str(self._sett_misc_n_rltzns))}d}')

            self._rs.iter_ctr = iter_ctr
            self._rs.iters_wo_acpt = iters_wo_acpt
            self._rs.tol = tol
            self._rs.temp = temp
            self._rs.stopp_criteria = np.array(stopp_criteria)
            self._rs.tols = np.array(tols, dtype=np.float64)
            self._rs.obj_vals_all = np.array(obj_vals_all, dtype=np.float64)

            self._rs.acpts_rjts_all = np.array(acpts_rjts_all, dtype=bool)

            self._rs.acpt_rates_all = (
                np.cumsum(self._rs.acpts_rjts_all) /
                np.arange(1, self._rs.acpts_rjts_all.size + 1, dtype=float))

            self._rs.obj_vals_min = np.array(obj_vals_min, dtype=np.float64)

            self._rs.temps = np.array(temps, dtype=np.float64)

            self._rs.acpt_rates_dfrntl = np.array(
                acpt_rates_dfrntl, dtype=np.float64)

            self._rr.ft_cumm_corr = self._get_cumm_ft_corr(
                self._rr.ft, self._rr.ft)

            self._rs.ref_sim_ft_corr = self._get_cumm_ft_corr(
                self._rr.ft, self._rs.ft).astype(np.float64)

            self._rs.sim_sim_ft_corr = self._get_cumm_ft_corr(
                self._rs.ft, self._rs.ft).astype(np.float64)

            self._rs.obj_vals_all_indiv = np.array(
                obj_vals_all_indiv, dtype=np.float64)

            self._rs.cumm_call_durations = self._dur_tmr_cumm_call_times
            self._rs.cumm_n_calls = self._dur_tmr_cumm_n_calls

            # IAAFTSA variable.
            self._rs.mxn_ratio_red_rates = np.array(
                mxn_ratio_red_rates, dtype=np.float64)

            # IAAFTSA variable.
            self._rs.mxn_ratios_probs = np.array(
                mxn_ratios_probs, dtype=np.float64)

            # IAAFTSA variable.
            self._rs.mxn_ratios_margs = np.array(
                mxn_ratios_margs, dtype=np.float64)

            # IAAFTSA variable.
            self._rs.n_levelss = np.array(n_levelss, dtype=np.float64)

            # IAAFTSA variable.
            self._rs.max_shift_exps = np.array(
                max_shift_exps, dtype=np.float64)

            # IAAFTSA variable.
            self._rs.max_shifts = np.array(max_shifts, dtype=np.float64)

            # IAAFTSA variable.
            self._rs.pre_vals_ratios = np.array(
                pre_vals_ratios, dtype=np.float64)

            # IAAFTSA variable.
            self._rs.asymm_n_iterss = np.array(
                asymm_n_iterss, dtype=np.float64)

            # IAAFTSA variable.
            self._rs.prob_centers = np.array(prob_centers, dtype=np.float64)

            self._write_cls_rltzn()

            ret = stopp_criteria

        self._alg_snapshot = None
        return ret

    # @GTGBase._timer_wrap
    # def _run_iaaft(self, opt_vars_cls, plain_iaaft_flag=False):
    #
    #     if plain_iaaft_flag:
    #         data = self._rs.data.copy()
    #         probs = self._rs.probs.copy()
    #
    #         iaaft_n_iters = 10
    #
    #     else:
    #         data = self._rs.init_data.copy()
    #         probs = self._rs.init_probs.copy()
    #
    #         if self._sett_asymm_set_flag:
    #
    #             # Using probs as data works slightly better
    #             # than using data.
    #
    #             if self._sett_asymm_type == 1:
    #                 data = self._asymmetrize_type_1(
    #                     probs, probs, opt_vars_cls)
    #
    #             elif self._sett_asymm_type == 2:
    #                 data = self._asymmetrize_type_2(
    #                     probs, probs, opt_vars_cls)
    #
    #             else:
    #                 raise NotImplementedError
    #
    #             probs = self._get_probs(data, True)
    #
    #             self._rs.probs = probs
    #
    #             self._rs.data = np.empty_like(
    #                 self._data_ref_rltzn_srtd, dtype=np.float64)
    #
    #             for i in range(self._data_ref_n_labels):
    #                 self._rs.data[:, i] = self._data_ref_rltzn_srtd[
    #                     np.argsort(np.argsort(probs[:, i])), i]
    #
    #             if False:
    #                 assert self._rs.data.shape[1] == 1, (
    #                     'self._get_smoothed_array requires 1 column only!')
    #
    #                 self._rs.data = self._get_smoothed_array(
    #                     self._rs.data.ravel(), 2).reshape(-1, 1)
    #
    #             data = self._rs.data
    #
    #             self._rs.ft = np.fft.rfft(data, axis=0)
    #             self._rs.mag_spec = np.abs(self._rs.ft)
    #             self._rs.phs_spec = np.angle(self._rs.ft)
    #
    #             self._update_sim_no_prms()
    #
    #         # Only needed here.
    #         best_ft = self._rs.ft.copy()
    #         best_obj_val = self._get_obj_ftn_val().sum()
    #
    #         iaaft_n_iters = 0  # self._sett_ann_iaaft_n_iters
    #
    #         if False:
    #             ft_data_init = np.fft.rfft(self._rs.init_data, axis=0)
    #             ft_data_asym = best_ft
    #
    #             ft_data_init_corr = self._get_cumm_ft_corr(
    #                 ft_data_init, ft_data_init)[:, 0]
    #
    #             ft_data_asym_corr = self._get_cumm_ft_corr(
    #                 ft_data_asym, ft_data_asym)[:, 0]
    #
    #             ref_periods = (ft_data_init_corr.size * 2) / (
    #                 np.arange(1, ft_data_init_corr.size + 1))
    #
    #             import matplotlib.pyplot as plt
    #
    #             plt.figure(figsize=(10, 7))
    #
    #             plt.semilogx(
    #                 ref_periods,
    #                 ft_data_init_corr,
    #                 alpha=0.75,
    #                 color='r',
    #                 lw=1.5,
    #                 label='init')
    #
    #             plt.semilogx(
    #                 ref_periods,
    #                 ft_data_asym_corr,
    #                 alpha=0.75,
    #                 color='b',
    #                 lw=1.5,
    #                 label='asym')
    #
    #             plt.grid()
    #
    #             plt.ylim(-0.05, +1.05)
    #
    #             plt.gca().set_axisbelow(True)
    #
    #             plt.legend(framealpha=0.7)
    #
    #             plt.ylabel('Cummulative data FT correlation')
    #
    #             plt.xlabel(f'Period (steps)')
    #
    #             plt.xlim(plt.xlim()[::-1])
    #
    #             out_name = f'P:/Downloads/ss__ft_data_init_asym2.png'
    #
    #             plt.savefig(str(out_name), bbox_inches='tight')
    #
    #             plt.close()
    #
    #             raise Exception
    #
    #     order_sdiffs = np.full(iaaft_n_iters, np.nan)
    #
    #     obj_vals = np.full(iaaft_n_iters, np.inf)
    #
    #     auto_spec_flag = True
    #
    #     if data.shape[1] > 1:
    #         cross_spec_flag = True
    #
    #     else:
    #         cross_spec_flag = False
    #
    #     reorder_idxs_old = np.random.randint(0, data.shape[0], size=data.shape)
    #
    #     best_updt_ctr = 0
    #     max_not_best_updt_iters = 2  # Consecutive.
    #
    #     stn_ctr = 0
    #     for i in range(iaaft_n_iters):
    #
    #         if opt_vars_cls.mxn_ratio_margs:
    #             sim_ft_margs = np.fft.rfft(data, axis=0)
    #
    #         if opt_vars_cls.mxn_ratio_probs:
    #             sim_ft_probs = np.fft.rfft(probs, axis=0)
    #
    #         # Marginals auto.
    #         if opt_vars_cls.mxn_ratio_margs and auto_spec_flag:
    #             sim_phs_margs = np.angle(sim_ft_margs)
    #
    #             if self._sett_prsrv_coeffs_set_flag:
    #                 sim_phs_margs[self._rr.prsrv_coeffs_idxs,:] = (
    #                     self._rr.phs_spec[self._rr.prsrv_coeffs_idxs,:])
    #
    #             sim_ft_new = np.empty_like(sim_ft_margs)
    #
    #             sim_ft_new.real[:] = np.cos(sim_phs_margs) * self._rr.mag_spec
    #             sim_ft_new.imag[:] = np.sin(sim_phs_margs) * self._rr.mag_spec
    #
    #             # sim_ft_new[self._rr.prsrv_coeffs_idxs,:] = 0
    #
    #             sim_ft_new[0,:] = 0
    #
    #             sim_ift_margs_auto = np.fft.irfft(sim_ft_new, axis=0)
    #
    #             # reorder_idxs_new = np.empty_like(reorder_idxs_old)
    #             # for k in range(self._data_ref_n_labels):
    #             #     reorder_idxs_new[:, k] = np.argsort(
    #             #         np.argsort(sim_ift_margs_auto[:, k]))
    #             #
    #             # for k in range(self._data_ref_n_labels):
    #             #     sim_ift_margs_auto[:, k] = (
    #             #         self._data_ref_rltzn_srtd[reorder_idxs_new[:, k], k])
    #
    #             # sim_ift_margs_auto -= sim_ift_margs_auto.mean(axis=0)
    #             # sim_ift_margs_auto /= sim_ift_margs_auto.std(axis=0)
    #
    #         else:
    #             sim_ift_margs_auto = 0.0
    #
    #         # Ranks auto.
    #         if opt_vars_cls.mxn_ratio_probs and auto_spec_flag:
    #             sim_phs = np.angle(sim_ft_probs)
    #
    #             if self._sett_prsrv_coeffs_set_flag:
    #                 sim_phs[self._rr.prsrv_coeffs_idxs,:] = (
    #                     self._rr.probs_ft_coeffs_phss[
    #                         self._rr.prsrv_coeffs_idxs,:])
    #
    #             sim_ft_new = np.empty_like(sim_ft_probs)
    #
    #             sim_ft_new.real[:] = (
    #                 np.cos(sim_phs) * self._rr.probs_ft_coeffs_mags)
    #
    #             sim_ft_new.imag[:] = (
    #                 np.sin(sim_phs) * self._rr.probs_ft_coeffs_mags)
    #
    #             # sim_ft_new[self._rr.prsrv_coeffs_idxs,:] = 0
    #
    #             sim_ft_new[0,:] = 0
    #
    #             sim_ift_probs_auto = np.fft.irfft(sim_ft_new, axis=0)
    #
    #             # sim_ift_probs_auto = self._get_probs(sim_ift_probs_auto, True)
    #
    #             # sim_ift_probs_auto -= sim_ift_probs_auto.mean(axis=0)
    #             # sim_ift_probs_auto /= sim_ift_probs_auto.std(axis=0)
    #
    #         else:
    #             sim_ift_probs_auto = 0.0
    #         #==================================================================
    #
    #         # Marginals auto long perods.
    #         if opt_vars_cls.mxn_ratio_margs and auto_spec_flag and False:
    #             sim_phs_margs = np.angle(sim_ft_margs)
    #
    #             if self._sett_prsrv_coeffs_set_flag:
    #                 sim_phs_margs[self._rr.prsrv_coeffs_idxs,:] = (
    #                     self._rr.phs_spec[self._rr.prsrv_coeffs_idxs,:])
    #
    #             sim_ft_new = np.empty_like(sim_ft_margs)
    #
    #             sim_ft_new.real[:] = np.cos(sim_phs_margs) * self._rr.mag_spec
    #             sim_ft_new.imag[:] = np.sin(sim_phs_margs) * self._rr.mag_spec
    #
    #             sim_ft_new[~self._rr.prsrv_coeffs_idxs,:] = 0
    #
    #             sim_ft_new[0,:] = 0
    #
    #             sim_ift_margs_auto_long = np.fft.irfft(sim_ft_new, axis=0)
    #
    #             # reorder_idxs_new = np.empty_like(reorder_idxs_old)
    #             # for k in range(self._data_ref_n_labels):
    #             #     reorder_idxs_new[:, k] = np.argsort(
    #             #         np.argsort(sim_ift_margs_auto_long[:, k]))
    #             #
    #             # for k in range(self._data_ref_n_labels):
    #             #     sim_ift_margs_auto_long[:, k] = (
    #             #         self._data_ref_rltzn_srtd[reorder_idxs_new[:, k], k])
    #
    #             # sim_ift_margs_auto_long -= sim_ift_margs_auto_long.mean(axis=0)
    #             # sim_ift_margs_auto_long /= sim_ift_margs_auto_long.std(axis=0)
    #
    #         else:
    #             sim_ift_margs_auto_long = 0.0
    #
    #         # Ranks auto long periods.
    #         if opt_vars_cls.mxn_ratio_probs and auto_spec_flag and False:
    #             sim_phs = np.angle(sim_ft_probs)
    #
    #             if self._sett_prsrv_coeffs_set_flag:
    #                 sim_phs[self._rr.prsrv_coeffs_idxs,:] = (
    #                     self._rr.probs_ft_coeffs_phss[
    #                         self._rr.prsrv_coeffs_idxs,:])
    #
    #             sim_ft_new = np.empty_like(sim_ft_probs)
    #
    #             sim_ft_new.real[:] = (
    #                 np.cos(sim_phs) * self._rr.probs_ft_coeffs_mags)
    #
    #             sim_ft_new.imag[:] = (
    #                 np.sin(sim_phs) * self._rr.probs_ft_coeffs_mags)
    #
    #             sim_ft_new[~self._rr.prsrv_coeffs_idxs,:] = 0
    #
    #             sim_ft_new[0,:] = 0
    #
    #             sim_ift_probs_auto_long = np.fft.irfft(sim_ft_new, axis=0)
    #
    #             # sim_ift_probs_auto_long = self._get_probs(
    #             #     sim_ift_probs_auto_long, True)
    #
    #             # sim_ift_probs_auto_long -= sim_ift_probs_auto_long.mean(axis=0)
    #             # sim_ift_probs_auto_long /= sim_ift_probs_auto_long.std(axis=0)
    #
    #         else:
    #             sim_ift_probs_auto_long = 0.0
    #         #==================================================================
    #
    #         # Marginals cross.
    #         if opt_vars_cls.mxn_ratio_margs and cross_spec_flag:
    #             sim_mag = self._rr.mag_spec.copy()
    #
    #             sim_phs = (
    #                 np.angle(sim_ft_margs[:, [stn_ctr]]) +
    #                 self._rr.phs_spec -
    #                 self._rr.phs_spec[:, [stn_ctr]])
    #
    #             sim_phs[0,:] = self._rr.phs_spec[0,:]
    #
    #             if self._sett_prsrv_coeffs_set_flag:
    #                 sim_phs[self._rr.prsrv_coeffs_idxs,:] = (
    #                     self._rr.phs_spec[self._rr.prsrv_coeffs_idxs,:])
    #
    #             sim_ft_new = np.empty_like(sim_ft_margs)
    #
    #             sim_ft_new.real[:] = np.cos(sim_phs) * sim_mag
    #             sim_ft_new.imag[:] = np.sin(sim_phs) * sim_mag
    #
    #             sim_ft_new[0,:] = 0
    #
    #             sim_ift_margs_cross = np.fft.irfft(sim_ft_new, axis=0)
    #
    #             reorder_idxs_new = np.empty_like(reorder_idxs_old)
    #             for k in range(self._data_ref_n_labels):
    #                 reorder_idxs_new[:, k] = np.argsort(
    #                     np.argsort(sim_ift_margs_cross[:, k]))
    #
    #             for k in range(self._data_ref_n_labels):
    #                 sim_ift_margs_cross[:, k] = (
    #                     self._data_ref_rltzn_srtd[reorder_idxs_new[:, k], k])
    #
    #             sim_ift_margs_cross -= sim_ift_margs_cross.mean(axis=0)
    #             sim_ift_margs_cross /= sim_ift_margs_cross.std(axis=0)
    #
    #         else:
    #             sim_ift_margs_cross = 0.0
    #
    #         # Ranks cross.
    #         if opt_vars_cls.mxn_ratio_probs and cross_spec_flag:
    #             sim_mag = self._rr.probs_ft_coeffs_mags.copy()
    #
    #             sim_phs = (
    #                 np.angle(sim_ft_probs[:, [stn_ctr]]) +
    #                 self._rr.probs_ft_coeffs_phss -
    #                 self._rr.probs_ft_coeffs_phss[:, [stn_ctr]])
    #
    #             sim_phs[0,:] = self._rr.probs_ft_coeffs_phss[0,:]
    #
    #             if self._sett_prsrv_coeffs_set_flag:
    #                 sim_phs[self._rr.prsrv_coeffs_idxs,:] = (
    #                     self._rr.probs_ft_coeffs_phss[
    #                         self._rr.prsrv_coeffs_idxs,:])
    #
    #             sim_ft_new = np.empty_like(sim_ft_probs)
    #
    #             sim_ft_new.real[:] = np.cos(sim_phs) * sim_mag
    #             sim_ft_new.imag[:] = np.sin(sim_phs) * sim_mag
    #
    #             sim_ft_new[0,:] = 0
    #
    #             sim_ift_probs_cross = np.fft.irfft(sim_ft_new, axis=0)
    #
    #             sim_ift_probs_cross = self._get_probs(
    #                 sim_ift_probs_cross, True)
    #
    #             sim_ift_probs_cross -= sim_ift_probs_cross.mean(axis=0)
    #             sim_ift_probs_cross /= sim_ift_probs_cross.std(axis=0)
    #
    #         else:
    #             sim_ift_probs_cross = 0.0
    #         #==================================================================
    #
    #         # Their sum.
    #         sim_ift = (
    #             (opt_vars_cls.mxn_ratio_margs * sim_ift_margs_auto) +
    #             (opt_vars_cls.mxn_ratio_probs * sim_ift_probs_auto) +
    #             (opt_vars_cls.mxn_ratio_margs * sim_ift_margs_auto_long) +
    #             (opt_vars_cls.mxn_ratio_probs * sim_ift_probs_auto_long) +
    #             (opt_vars_cls.mxn_ratio_margs * sim_ift_margs_cross) +
    #             (opt_vars_cls.mxn_ratio_probs * sim_ift_probs_cross)
    #             )
    #
    #         assert isinstance(sim_ift, np.ndarray), type(sim_ift)
    #
    #         reorder_idxs_new = np.empty_like(reorder_idxs_old)
    #         for k in range(self._data_ref_n_labels):
    #             reorder_idxs_new[:, k] = np.argsort(np.argsort(sim_ift[:, k]))
    #
    #         order_sdiff = 1 - np.corrcoef(
    #             reorder_idxs_old.ravel(), reorder_idxs_new.ravel())[0, 1]
    #
    #         assert order_sdiff >= 0, order_sdiff
    #
    #         order_sdiffs[i] = order_sdiff
    #
    #         reorder_idxs_old = reorder_idxs_new
    #
    #         data = np.empty_like(data)
    #
    #         for k in range(self._data_ref_n_labels):
    #             data[:, k] = self._data_ref_rltzn_srtd[
    #                 reorder_idxs_old[:, k], k]
    #
    #         probs = self._get_probs(data, True)
    #
    #         if order_sdiff <= (1e-3):
    #             # Nothing changed.
    #             break
    #
    #         stn_ctr += 1
    #         if stn_ctr == data.shape[1]:
    #             stn_ctr = 0
    #         #==================================================================
    #
    #         if not plain_iaaft_flag:
    #             self._rs.ft = np.fft.rfft(data, axis=0)
    #             self._rs.mag_spec = np.abs(self._rs.ft)
    #             self._rs.phs_spec = np.angle(self._rs.ft)
    #
    #             self._update_sim_no_prms()
    #
    #             obj_val = self._get_obj_ftn_val().sum()
    #
    #             obj_vals[i] = obj_val
    #
    #             if obj_val < best_obj_val:
    #
    #                 best_obj_val = obj_val
    #
    #                 best_ft = np.fft.rfft(data, axis=0)
    #
    #                 best_updt_ctr = 0
    #
    #             else:
    #                 best_updt_ctr += 1
    #
    #             if best_updt_ctr == max_not_best_updt_iters:
    #                 break
    #         #==================================================================
    #
    #     if plain_iaaft_flag:
    #         best_ft = np.fft.rfft(data, axis=0)
    #
    #     self._rs.ft = best_ft
    #     self._rs.mag_spec = np.abs(self._rs.ft)
    #     self._rs.phs_spec = np.angle(self._rs.ft)
    #     return

    # def _adjust_init_spec(self, opt_vars_cls):
    #
    #     data = self._rs.init_data.copy()
    #     probs = self._rs.init_probs.copy()
    #
    #     data_init = data.copy()
    #
    #     plot_flag = False
    #
    #     if plot_flag:
    #         import matplotlib.pyplot as plt
    #
    #         ft_data_ref_corr = self._get_cumm_ft_corr(
    #             self._rr.data_ft_coeffs, self._rr.data_ft_coeffs)[:, 0]
    #
    #         ft_probs_ref_corr = self._get_cumm_ft_corr(
    #             self._rr.probs_ft_coeffs, self._rr.probs_ft_coeffs)[:, 0]
    #
    #         ref_periods = (ft_data_ref_corr.size * 2) / (
    #             np.arange(1, ft_data_ref_corr.size + 1))
    #
    #         plt.figure(figsize=(13, 7))
    #
    #         plt.semilogx(
    #             ref_periods,
    #             ft_data_ref_corr,
    #             alpha=0.75,
    #             color='r',
    #             lw=1.5,
    #             label='ref_data',
    #             zorder=3)
    #
    #         plt.semilogx(
    #             ref_periods,
    #             ft_probs_ref_corr,
    #             alpha=0.75,
    #             color='r',
    #             lw=1.5,
    #             ls='--',
    #             label='ref_probs',
    #             zorder=3)
    #
    #     best_obj_val = np.inf
    #
    #     best_ft = self._rs.ft.copy()
    #
    #     ft_iter = 0
    #     while True:
    #         ft_data_init = np.fft.rfft(data_init, axis=0)
    #         mag_data_init = np.abs(ft_data_init)
    #         phs_data_init = np.angle(ft_data_init)
    #
    #         ft_data_init_corr = self._get_cumm_ft_corr(
    #             ft_data_init, ft_data_init)[:, 0]
    #
    #         ft_probs_init = np.fft.rfft(probs, axis=0)
    #
    #         ft_probs_init_corr = self._get_cumm_ft_corr(
    #             ft_probs_init, ft_probs_init)[:, 0]
    #
    #         # Using probs as data works slightly better
    #         # than using data.
    #
    #         if self._sett_asymm_type == 1:
    #             data = self._asymmetrize_type_1(
    #                 probs, probs, opt_vars_cls)
    #
    #         elif self._sett_asymm_type == 2:
    #             data = self._asymmetrize_type_2(
    #                 probs, probs, opt_vars_cls)
    #
    #         else:
    #             raise NotImplementedError
    #
    #         probs = self._get_probs(data, True)
    #
    #         data = np.empty_like(
    #             self._data_ref_rltzn_srtd, dtype=np.float64)
    #
    #         for i in range(self._data_ref_n_labels):
    #             data[:, i] = self._data_ref_rltzn_srtd[
    #                 np.argsort(np.argsort(probs[:, i])), i]
    #
    #         ft_data_asym = np.fft.rfft(data, axis=0)
    #
    #         mag_data_asym = np.abs(ft_data_asym)
    #
    #         ft_data_asym_corr = self._get_cumm_ft_corr(
    #             ft_data_asym, ft_data_asym)[:, 0]
    #
    #         ft_probs_asym = np.fft.rfft(probs, axis=0)
    #
    #         mag_probs_asym = np.abs(ft_probs_asym)
    #
    #         ft_probs_asym_corr = self._get_cumm_ft_corr(
    #             ft_probs_asym, ft_probs_asym)[:, 0]
    #
    #         sq_diff = ((self._rr.data_ft_coeffs_mags - mag_data_asym)[1:] /
    #                    self._rr.data_ft_coeffs_mags[1:].sum()) ** 2
    #
    #         sq_diff += ((self._rr.probs_ft_coeffs_mags - mag_probs_asym)[1:] /
    #                    self._rr.probs_ft_coeffs_mags[1:].sum()) ** 2
    #
    #         obj_val = sq_diff.sum()
    #
    #         if obj_val < best_obj_val:
    #             best_obj_val = obj_val
    #
    #             best_ft = ft_data_asym.copy()
    #             # best_ft = ft_data_init.copy()
    #
    #         # print(ft_iter, f'{obj_val:0.3E}')
    #
    #         # mag_data_init = mag_data_init + (self._rr.mag_spec - mag_data_asym)
    #
    #         mag_data_init += (self._rr.data_ft_coeffs_mags - mag_data_asym) * 0.5
    #         mag_data_init += (self._rr.probs_ft_coeffs_mags - mag_probs_asym) * 0.5
    #
    #         if True:
    #             neg_idxs = mag_data_init < 0
    #
    #             mag_data_init[neg_idxs] = 0
    #
    #         ft_data_init = np.empty_like(ft_data_init)
    #
    #         ft_data_init.real[:] = np.cos(phs_data_init) * mag_data_init
    #         ft_data_init.imag[:] = np.sin(phs_data_init) * mag_data_init
    #
    #         data_init = np.fft.irfft(ft_data_init, axis=0)
    #
    #         probs = self._get_probs(data_init, True)
    #
    #         data_init = np.empty_like(
    #             self._data_ref_rltzn_srtd, dtype=np.float64)
    #
    #         for i in range(self._data_ref_n_labels):
    #             data_init[:, i] = self._data_ref_rltzn_srtd[
    #                 np.argsort(np.argsort(probs[:, i])), i]
    #
    #         if plot_flag:
    #             if ft_iter == 0:
    #                 lab_init_data = 'init_data'
    #                 lab_asym_data = 'asym_data'
    #                 lab_init_probs = 'init_probs'
    #                 lab_asym_probs = 'asym_probs'
    #
    #             else:
    #                 lab_init_data = lab_asym_data = None
    #                 lab_init_probs = lab_asym_probs = None
    #
    #             plt.semilogx(
    #                 ref_periods,
    #                 ft_data_init_corr,
    #                 alpha=0.1,
    #                 color='g',
    #                 lw=1.,
    #                 label=lab_init_data,
    #                 zorder=1)
    #
    #             plt.semilogx(
    #                 ref_periods,
    #                 ft_data_asym_corr,
    #                 alpha=0.1,
    #                 color='b',
    #                 lw=1.,
    #                 label=lab_asym_data,
    #                 zorder=2)
    #
    #             plt.semilogx(
    #                 ref_periods,
    #                 ft_probs_init_corr,
    #                 alpha=0.1,
    #                 color='g',
    #                 lw=1.,
    #                 ls='--',
    #                 label=lab_init_probs,
    #                 zorder=1)
    #
    #             plt.semilogx(
    #                 ref_periods,
    #                 ft_probs_asym_corr,
    #                 alpha=0.1,
    #                 color='b',
    #                 lw=1.,
    #                 ls='--',
    #                 label=lab_asym_probs,
    #                 zorder=2)
    #
    #         ft_iter += 1
    #
    #         if ft_iter == 10:
    #             break
    #
    #     if plot_flag:
    #         plt.grid()
    #
    #         plt.ylim(-0.05, +1.05)
    #
    #         plt.gca().set_axisbelow(True)
    #
    #         plt.legend(framealpha=0.7)
    #
    #         plt.title(
    #             f'n_levels: {opt_vars_cls.n_levels}, '
    #             f'max_shift_exp: {opt_vars_cls.max_shift_exp:0.2f}, '
    #             f'max_shift: {opt_vars_cls.max_shift}\n'
    #             f'pre_vals_ratio: {opt_vars_cls.pre_vals_ratio:0.2f}, '
    #             f'asymm_n_iters: {opt_vars_cls.asymm_n_iters}, '
    #             f'prob_center: {opt_vars_cls.prob_center:0.2f}')
    #
    #         plt.ylabel('Cummulative FT correlation')
    #
    #         plt.xlabel(f'Period (steps)')
    #
    #         plt.xlim(plt.xlim()[::-1])
    #
    #         out_name = f'P:/Downloads/ss__ft_data_init_asym4.png'
    #
    #         plt.savefig(str(out_name), bbox_inches='tight')
    #
    #         plt.close()
    #
    #     return best_ft

    # def _adjust_init_spec(self, opt_vars_cls):
    #
    #     data_init = self._rs.init_data.copy()
    #     probs_init = self._rs.init_probs.copy()
    #
    #     best_obj_val = np.inf
    #
    #     best_ft = self._rs.ft.copy()
    #
    #     ft_iter = 0
    #     while True:
    #         ft_data_init = np.fft.rfft(data_init, axis=0)
    #         mag_data_init = np.abs(ft_data_init)
    #         phs_data_init = np.angle(ft_data_init)
    #
    #         ft_probs_init = np.fft.rfft(probs_init, axis=0)
    #         mag_probs_init = np.abs(ft_probs_init)
    #         phs_probs_init = np.angle(ft_probs_init)
    #
    #         # Using probs as data works slightly better
    #         # than using data.
    #
    #         if self._sett_asymm_type == 1:
    #             data_asym = self._asymmetrize_type_1(
    #                 probs_init, probs_init, opt_vars_cls)
    #
    #         elif self._sett_asymm_type == 2:
    #             data_asym = self._asymmetrize_type_2(
    #                 probs_init, probs_init, opt_vars_cls)
    #
    #         else:
    #             raise NotImplementedError
    #
    #         probs_asym = self._get_probs(data_asym, True)
    #
    #         data_asym = np.empty_like(
    #             self._data_ref_rltzn_srtd, dtype=np.float64)
    #
    #         for i in range(self._data_ref_n_labels):
    #             data_asym[:, i] = self._data_ref_rltzn_srtd[
    #                 np.argsort(np.argsort(probs_asym[:, i])), i]
    #
    #         ft_data_asym = np.fft.rfft(data_asym, axis=0)
    #
    #         mag_data_asym = np.abs(ft_data_asym)
    #
    #         ft_probs_asym = np.fft.rfft(probs_asym, axis=0)
    #
    #         mag_probs_asym = np.abs(ft_probs_asym)
    #
    #         sq_diff = ((self._rr.data_ft_coeffs_mags - mag_data_asym)[1:] /
    #                    self._rr.data_ft_coeffs_mags[1:].sum()) ** 2
    #
    #         sq_diff += ((self._rr.probs_ft_coeffs_mags - mag_probs_asym)[1:] /
    #                    self._rr.probs_ft_coeffs_mags[1:].sum()) ** 2
    #
    #         obj_val = sq_diff.sum()
    #
    #         if obj_val < best_obj_val:
    #             best_obj_val = obj_val
    #
    #             best_ft = ft_data_asym.copy()
    #             # best_ft = ft_data_init.copy()
    #
    #         # print(ft_iter, f'{obj_val:0.3E}')
    #
    #         # mag_data_init = mag_data_init + (self._rr.mag_spec - mag_data_asym)
    #
    #         mag_data_init += (self._rr.data_ft_coeffs_mags - mag_data_asym) * 0.5
    #         mag_data_init += (self._rr.probs_ft_coeffs_mags - mag_probs_asym) * 0.5
    #
    #         if True:
    #             neg_idxs = mag_data_init < 0
    #
    #             mag_data_init[neg_idxs] = 0
    #
    #         ft_data_init = np.empty_like(ft_data_init)
    #
    #         ft_data_init.real[:] = np.cos(phs_data_init) * mag_data_init
    #         ft_data_init.imag[:] = np.sin(phs_data_init) * mag_data_init
    #
    #         data_init = np.fft.irfft(ft_data_init, axis=0)
    #
    #         probs_init = self._get_probs(data_init, True)
    #
    #         data_init = np.empty_like(
    #             self._data_ref_rltzn_srtd, dtype=np.float64)
    #
    #         for i in range(self._data_ref_n_labels):
    #             data_init[:, i] = self._data_ref_rltzn_srtd[
    #                 np.argsort(np.argsort(probs_init[:, i])), i]
    #
    #         ft_iter += 1
    #
    #         if ft_iter == 10:
    #             break
    #
    #     return best_ft
