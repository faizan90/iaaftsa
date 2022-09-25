'''
Created on Aug 7, 2022

@author: Faizan3800X-Uni
'''
from time import asctime
from collections import deque
from timeit import default_timer

import numpy as np

from fcopulas import asymmetrize_type_10_ms_cy

from gnrctsgenr import (
    GTGBase,
    GTGAlgRealization,
    )

from gnrctsgenr.misc import print_sl, print_el


class OPTVARS:

    # NOTE: Update IAAFTSAPrepareRltznSim in ba_prep.py as well.

    def __init__(self):

        # For IAAFT.
        self.mxn_ratio_margss = np.array([], dtype=float)
        self.mxn_ratio_probss = np.array([], dtype=float)

        # For asymmetrize.
        self.n_levelss = np.array([], dtype=np.uint32)
        self.max_shift_exps = np.array([], dtype=float)
        self.max_shifts = np.array([], dtype=np.int64)
        self.pre_vals_ratios = np.array([], dtype=float)
        self.asymm_n_iterss = np.array([], dtype=np.uint32)
        self.asymms_rand_errs = np.array([], dtype=float)
        self.prob_centers = np.array([], dtype=float)
        self.pre_val_exps = np.array([], dtype=float)
        self.crt_val_exps = np.array([], dtype=float)
        self.level_thresh_cnsts = np.array([], dtype=np.int64)
        self.level_thresh_slps = np.array([], dtype=float)
        self.rand_err_sclr_cnsts = np.array([], dtype=float)
        self.rand_err_sclr_rels = np.array([], dtype=float)
        self.rand_err_cnst = np.array([], dtype=float)
        self.rand_err_rel = np.array([], dtype=float)

        # After IAAFT is performed. This is needed for later viewing only.
        self.order_sdiff = None
        return

    def get_copy(self):

        opt_vars_cls_new = OPTVARS()

        opt_vars_cls_new.mxn_ratio_probss = (
            self.mxn_ratio_probss.copy())

        opt_vars_cls_new.mxn_ratio_margss = (
            self.mxn_ratio_margss.copy())

        opt_vars_cls_new.n_levelss = self.n_levelss.copy()

        opt_vars_cls_new.max_shift_exps = (
            self.max_shift_exps.copy())

        opt_vars_cls_new.max_shifts = self.max_shifts.copy()

        opt_vars_cls_new.pre_vals_ratios = (
            self.pre_vals_ratios.copy())

        opt_vars_cls_new.asymm_n_iterss = (
            self.asymm_n_iterss.copy())

        opt_vars_cls_new.asymms_rand_errs = (
            self.asymms_rand_errs.copy(order='f'))

        opt_vars_cls_new.prob_centers = (
            self.prob_centers.copy())

        opt_vars_cls_new.pre_val_exps = (
            self.pre_val_exps.copy())

        opt_vars_cls_new.crt_val_exps = (
            self.crt_val_exps.copy())

        opt_vars_cls_new.level_thresh_cnsts = (
            self.level_thresh_cnsts.copy())

        opt_vars_cls_new.level_thresh_slps = (
            self.level_thresh_slps.copy())

        opt_vars_cls_new.rand_err_sclr_cnsts = (
            self.rand_err_sclr_cnsts.copy())

        opt_vars_cls_new.rand_err_sclr_rels = (
            self.rand_err_sclr_rels.copy())

        opt_vars_cls_new.rand_err_cnst = (
            self.rand_err_cnst.copy(order='f'))

        opt_vars_cls_new.rand_err_rel = (
            self.rand_err_rel.copy(order='f'))

        return opt_vars_cls_new


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

            self._init_iaaft_opt_vars(opt_vars_cls_old, False)

            self._init_iaaft(opt_vars_cls_old)

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
            stopp_criteria,
            init_obj_val):

        c1 = self._sett_ann_max_iters >= 10000
        c2 = not (iter_ctr % (0.05 * self._sett_ann_max_iters))
        c3 = all(stopp_criteria)

        if (c1 and c2) or (iter_ctr == 1) or (not c3):
            with self._lock:
                print_sl()

                if c3:
                    print(
                        f'Realization {rltzn_iter} finished {iter_ctr} '
                        f'out of {self._sett_ann_max_iters} iterations '
                        f'on {asctime()}.\n')

                else:
                    print(
                        f'Realization {rltzn_iter} finished on '
                        f'{asctime()} after {iter_ctr} iterations.')

                    print(
                        f'Total objective function value reduced '
                        f'{(init_obj_val / obj_val_min) - 1:.1E} times.\n')

                # right_align_chars
                rac = max([len(lab) for lab in self._alg_cnsts_stp_crit_labs])

                init_obj_str = 'Initial objective function value'
                obj_label_str = 'Current objective function value'
                obj_min_label_str = 'Running minimum objective function value'

                rac = max(rac, len(init_obj_str))
                rac = max(rac, len(obj_label_str))
                rac = max(rac, len(obj_min_label_str))

                print(f'{init_obj_str:>{rac}}: {init_obj_val:.2E}')
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

        assert (
            (0.0 <= opt_vars_cls_old.mxn_ratio_probss).all() and
            (opt_vars_cls_old.mxn_ratio_probss <= 1.0).all()), (
                opt_vars_cls_old.mxn_ratio_probss)

        assert (
            (0.0 <= opt_vars_cls_old.mxn_ratio_margss).all() and
            (opt_vars_cls_old.mxn_ratio_margss <= 1.0).all()), (
                opt_vars_cls_old.mxn_ratio_margss)

        if self._sett_asymm_set_flag:
            assert (
                (self._sett_asymm_n_levels_lbd <=
                 opt_vars_cls_old.n_levelss).all() and
                (opt_vars_cls_old.n_levelss <=
                 self._sett_asymm_n_levels_ubd).all()), (
                    opt_vars_cls_old.n_levelss)

            assert (
                (self._sett_asymm_max_shift_exp_lbd <=
                 opt_vars_cls_old.max_shift_exps).all() and
                (opt_vars_cls_old.max_shift_exps <=
                 self._sett_asymm_max_shift_exp_ubd).all()), (
                        opt_vars_cls_old.max_shift_exps)

            assert (
                (self._sett_asymm_max_shift_lbd <=
                 opt_vars_cls_old.max_shifts).all() and
                (opt_vars_cls_old.max_shifts <=
                 self._sett_asymm_max_shift_ubd).all()), (
                     opt_vars_cls_old.max_shifts)

            assert (
                (self._sett_asymm_pre_vals_ratio_lbd <=
                 opt_vars_cls_old.pre_vals_ratios).all() and
                (opt_vars_cls_old.pre_vals_ratios <=
                 self._sett_asymm_pre_vals_ratio_ubd).all()), (
                        opt_vars_cls_old.pre_vals_ratios)

            assert (
                (self._sett_asymm_n_iters_lbd <=
                 opt_vars_cls_old.asymm_n_iterss).all() and
                (opt_vars_cls_old.asymm_n_iterss <=
                 self._sett_asymm_n_iters_ubd).all()), (
                        opt_vars_cls_old.asymm_n_iterss)

            assert (
                (self._sett_asymm_prob_center_lbd <=
                 opt_vars_cls_old.prob_centers).all() and
                (opt_vars_cls_old.prob_centers <=
                 self._sett_asymm_prob_center_ubd).all()), (
                        opt_vars_cls_old.prob_centers)

            assert (
                (self._sett_asymm_pre_val_exp_lbd <=
                 opt_vars_cls_old.pre_val_exps).all() and
                (opt_vars_cls_old.pre_val_exps <=
                 self._sett_asymm_pre_val_exp_ubd).all()), (
                        opt_vars_cls_old.pre_val_exps)

            assert (
                (self._sett_asymm_crt_val_exp_lbd <=
                 opt_vars_cls_old.crt_val_exps).all() and
                (opt_vars_cls_old.crt_val_exps <=
                 self._sett_asymm_crt_val_exp_ubd).all()), (
                        opt_vars_cls_old.crt_val_exps)

            assert (
                (self._sett_asymm_level_thresh_cnst_lbd <=
                 opt_vars_cls_old.level_thresh_cnsts).all() and
                (opt_vars_cls_old.level_thresh_cnsts <=
                 self._sett_asymm_level_thresh_cnst_ubd).all()), (
                        opt_vars_cls_old.level_thresh_cnsts)

            assert (
                (self._sett_asymm_level_thresh_slp_lbd <=
                 opt_vars_cls_old.level_thresh_slps).all() and
                (opt_vars_cls_old.level_thresh_slps <=
                 self._sett_asymm_level_thresh_slp_ubd).all()), (
                        opt_vars_cls_old.level_thresh_slps)

            assert (
                (self._sett_asymm_rand_err_sclr_cnst_lbd <=
                 opt_vars_cls_old.rand_err_sclr_cnsts).all() and
                (opt_vars_cls_old.rand_err_sclr_cnsts <=
                 self._sett_asymm_rand_err_sclr_cnst_ubd).all()), (
                        opt_vars_cls_old.rand_err_sclr_cnsts)

            assert (
                (self._sett_asymm_rand_err_sclr_rel_lbd <=
                 opt_vars_cls_old.rand_err_sclr_rels).all() and
                (opt_vars_cls_old.rand_err_sclr_rels <=
                 self._sett_asymm_rand_err_sclr_rel_ubd).all()), (
                        opt_vars_cls_old.rand_err_sclr_rels)
        #======================================================================

        opt_vars_cls_new = opt_vars_cls_old.get_copy()

        n_vars_to_choose_from = 1

        if self._sett_asymm_set_flag:
            if self._sett_asymm_type == 2:
                n_vars_to_choose_from += 12

            else:
                raise NotImplementedError

        max_search_atpts = 1000
        search_atpts = 0

        col_idx = np.random.choice(np.arange(self._data_ref_shape[1]))

        var_updt_flag = False

        while not var_updt_flag:

            if search_atpts == max_search_atpts:
                self._rltzn_prm_max_srch_atpts_flag = True
                break

            var_to_updt = np.random.choice(np.arange(n_vars_to_choose_from))

            # Mixing variables.
            if var_to_updt == 0:

                ratio_diff_probs = -0.5 + (1 * np.random.random())

                mxn_ratio_probs = (
                    opt_vars_cls_old.mxn_ratio_probss[col_idx] +
                    (mxn_ratio_red_rate * ratio_diff_probs))

                mxn_ratio_probs = max(0.0, mxn_ratio_probs)
                mxn_ratio_probs = min(1.0, mxn_ratio_probs)

                mxn_ratio_margs = 1.0 - mxn_ratio_probs

                if (opt_vars_cls_old.mxn_ratio_probss[col_idx] ==
                    mxn_ratio_probs):

                    search_atpts += 1
                    continue

                opt_vars_cls_new.mxn_ratio_probss[col_idx] = mxn_ratio_probs
                opt_vars_cls_new.mxn_ratio_margss[col_idx] = mxn_ratio_margs

            # Asymmetrize variables.
            # n_levels.
            elif (var_to_updt == 1) and (
                (self._sett_asymm_n_levels_ubd -
                 self._sett_asymm_n_levels_lbd) > 0):

                if False:
                    n_levels_probs = [
                        0.5 * mxn_ratio_red_rate,
                        1.0 - mxn_ratio_red_rate,
                        0.5 * mxn_ratio_red_rate]

                    n_levels_diff = np.random.choice(
                        [-1, 0, 1], p=n_levels_probs)

                    n_levels = (
                        opt_vars_cls_old.n_levelss[col_idx] + n_levels_diff)

                else:
                    n_levels = opt_vars_cls_old.n_levelss[col_idx] + int(
                        mxn_ratio_red_rate *
                        (self._sett_asymm_n_levels_ubd -
                         self._sett_asymm_n_levels_lbd) * (
                        -0.5 + (1 * np.random.random())))

                n_levels = max(self._sett_asymm_n_levels_lbd, n_levels)
                n_levels = min(self._sett_asymm_n_levels_ubd, n_levels)

                n_levels = int(n_levels)

                if opt_vars_cls_old.n_levelss[col_idx] == n_levels:
                    search_atpts += 1
                    continue

                opt_vars_cls_new.n_levelss[col_idx] = n_levels

            # max_shift_exp.
            elif var_to_updt == 2 and (
                (self._sett_asymm_max_shift_exp_ubd -
                 self._sett_asymm_max_shift_exp_lbd) > 0):

                if False:
                    max_shift_exp_diff = -0.5 + (1 * np.random.random())
                    max_shift_exp_diff *= mxn_ratio_red_rate

                    max_shift_exp = (
                        opt_vars_cls_old.max_shift_exps[col_idx] +
                        max_shift_exp_diff)

                else:
                    max_shift_exp = (
                        opt_vars_cls_old.max_shift_exps[col_idx] + (
                            mxn_ratio_red_rate *
                            (self._sett_asymm_max_shift_exp_ubd -
                             self._sett_asymm_max_shift_exp_lbd) * (
                            -0.5 + (1 * np.random.random()))))

                max_shift_exp = max(
                    self._sett_asymm_max_shift_exp_lbd, max_shift_exp)

                max_shift_exp = min(
                    self._sett_asymm_max_shift_exp_ubd, max_shift_exp)

                if opt_vars_cls_old.max_shift_exps[col_idx] == max_shift_exp:
                    search_atpts += 1
                    continue

                opt_vars_cls_new.max_shift_exps[col_idx] = max_shift_exp

            # max_shift.
            elif var_to_updt == 3 and (
                (self._sett_asymm_max_shift_ubd -
                 self._sett_asymm_max_shift_lbd) > 0):

                if False:
                    max_shift_probs = [
                        0.5 * mxn_ratio_red_rate,
                        1.0 - mxn_ratio_red_rate,
                        0.5 * mxn_ratio_red_rate]

                    max_shift_diff = np.random.choice(
                        [-1, 0, 1], p=max_shift_probs)

                    max_shift = (
                        opt_vars_cls_old.max_shifts[col_idx] + max_shift_diff)

                else:
                    max_shift = opt_vars_cls_old.max_shifts[col_idx] + int(
                        mxn_ratio_red_rate *
                        (self._sett_asymm_max_shift_ubd -
                         self._sett_asymm_max_shift_lbd) * (
                        -0.5 + (1 * np.random.random())))

                max_shift = max(self._sett_asymm_max_shift_lbd, max_shift)
                max_shift = min(self._sett_asymm_max_shift_ubd, max_shift)

                max_shift = int(max_shift)

                if opt_vars_cls_old.max_shifts[col_idx] == max_shift:
                    search_atpts += 1
                    continue

                opt_vars_cls_new.max_shifts[col_idx] = max_shift

            # pre_vals_ratio.
            elif var_to_updt == 4 and (
                (self._sett_asymm_pre_vals_ratio_ubd -
                 self._sett_asymm_pre_vals_ratio_lbd) > 0):

                if False:
                    ratio_diff_pre_vals = -0.5 + (1 * np.random.random())

                    pre_vals_ratio = (
                        (opt_vars_cls_old.pre_vals_ratios[col_idx]) +
                        (mxn_ratio_red_rate * ratio_diff_pre_vals))

                else:
                    pre_vals_ratio = (
                        opt_vars_cls_old.pre_vals_ratios[col_idx] + (
                            mxn_ratio_red_rate *
                            (self._sett_asymm_pre_vals_ratio_ubd -
                             self._sett_asymm_pre_vals_ratio_lbd) * (
                            -0.5 + (1 * np.random.random()))))

                pre_vals_ratio = max(
                    self._sett_asymm_pre_vals_ratio_lbd, pre_vals_ratio)

                pre_vals_ratio = min(
                    self._sett_asymm_pre_vals_ratio_ubd, pre_vals_ratio)

                if (opt_vars_cls_old.pre_vals_ratios[col_idx] ==
                    pre_vals_ratio):

                    search_atpts += 1
                    continue

                opt_vars_cls_new.pre_vals_ratios[col_idx] = pre_vals_ratio

            # asymm_n_iters
            elif var_to_updt == 5 and (
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
                        opt_vars_cls_old.asymm_n_iterss[col_idx] +
                        asymm_n_iters_diff)

                else:
                    asymm_n_iters = (
                        opt_vars_cls_old.asymm_n_iterss[col_idx] + int(
                            mxn_ratio_red_rate *
                            (self._sett_asymm_n_iters_ubd -
                             self._sett_asymm_n_iters_lbd) * (
                            -0.5 + (1 * np.random.random()))))

                asymm_n_iters = max(
                    self._sett_asymm_n_iters_lbd, asymm_n_iters)

                asymm_n_iters = min(
                    self._sett_asymm_n_iters_ubd, asymm_n_iters)

                asymm_n_iters = int(asymm_n_iters)

                if opt_vars_cls_old.asymm_n_iterss[col_idx] == asymm_n_iters:
                    search_atpts += 1
                    continue

                opt_vars_cls_new.asymm_n_iterss[col_idx] = asymm_n_iters

            # prob_center.
            elif var_to_updt == 6 and (
                (self._sett_asymm_prob_center_ubd -
                 self._sett_asymm_prob_center_lbd) > 0):

                if False:
                    diff_pre_vals = -0.5 + (1 * np.random.random())

                    prob_center = opt_vars_cls_old.prob_centers[col_idx] + (
                        mxn_ratio_red_rate * diff_pre_vals)

                else:
                    prob_center = (
                        opt_vars_cls_old.prob_centers[col_idx] + (
                            mxn_ratio_red_rate *
                            (self._sett_asymm_prob_center_ubd -
                             self._sett_asymm_prob_center_lbd) * (
                            -0.5 + (1 * np.random.random()))))

                prob_center = max(
                    self._sett_asymm_prob_center_lbd, prob_center)

                prob_center = min(
                    self._sett_asymm_prob_center_ubd, prob_center)

                if opt_vars_cls_old.prob_centers[col_idx] == prob_center:
                    search_atpts += 1
                    continue

                opt_vars_cls_new.prob_centers[col_idx] = prob_center

            # pre_val_exp.
            elif var_to_updt == 7 and (
                (self._sett_asymm_pre_val_exp_ubd -
                 self._sett_asymm_pre_val_exp_lbd) > 0):

                if False:
                    diff_pre_vals = -0.5 + (1 * np.random.random())

                    pre_val_exp = opt_vars_cls_old.pre_val_exps[col_idx] + (
                        mxn_ratio_red_rate * diff_pre_vals)

                else:
                    pre_val_exp = (
                        opt_vars_cls_old.pre_val_exps[col_idx] + (
                            mxn_ratio_red_rate *
                            (self._sett_asymm_pre_val_exp_ubd -
                             self._sett_asymm_pre_val_exp_lbd) * (
                            -0.5 + (1 * np.random.random()))))

                pre_val_exp = max(
                    self._sett_asymm_pre_val_exp_lbd, pre_val_exp)

                pre_val_exp = min(
                    self._sett_asymm_pre_val_exp_ubd, pre_val_exp)

                if opt_vars_cls_old.pre_val_exps[col_idx] == pre_val_exp:
                    search_atpts += 1
                    continue

                opt_vars_cls_new.pre_val_exps[col_idx] = pre_val_exp

            # crt_val_exp.
            elif var_to_updt == 8 and (
                (self._sett_asymm_crt_val_exp_ubd -
                 self._sett_asymm_crt_val_exp_lbd) > 0):

                if False:
                    diff_pre_vals = -0.5 + (1 * np.random.random())

                    crt_val_exp = opt_vars_cls_old.crt_val_exps[col_idx] + (
                        mxn_ratio_red_rate * diff_pre_vals)

                else:
                    crt_val_exp = (
                        opt_vars_cls_old.crt_val_exps[col_idx] + (
                            mxn_ratio_red_rate *
                            (self._sett_asymm_crt_val_exp_ubd -
                             self._sett_asymm_crt_val_exp_lbd) * (
                            -0.5 + (1 * np.random.random()))))

                crt_val_exp = max(
                    self._sett_asymm_crt_val_exp_lbd, crt_val_exp)

                crt_val_exp = min(
                    self._sett_asymm_crt_val_exp_ubd, crt_val_exp)

                if opt_vars_cls_old.crt_val_exps[col_idx] == crt_val_exp:
                    search_atpts += 1
                    continue

                opt_vars_cls_new.crt_val_exps[col_idx] = crt_val_exp

            # level_thresh_cnst.
            elif (var_to_updt == 9) and (
                (self._sett_asymm_level_thresh_cnst_ubd -
                 self._sett_asymm_level_thresh_cnst_lbd) > 0):

                if False:
                    level_thresh_cnst_probs = [
                        0.5 * mxn_ratio_red_rate,
                        1.0 - mxn_ratio_red_rate,
                        0.5 * mxn_ratio_red_rate]

                    level_thresh_cnst_diff = np.random.choice(
                        [-1, 0, 1], p=level_thresh_cnst_probs)

                    level_thresh_cnst = (
                        opt_vars_cls_old.level_thresh_cnsts[col_idx] +
                        level_thresh_cnst_diff)

                else:
                    level_thresh_cnst = (
                        opt_vars_cls_old.level_thresh_cnsts[col_idx] + int(
                            mxn_ratio_red_rate *
                            (self._sett_asymm_level_thresh_cnst_ubd -
                             self._sett_asymm_level_thresh_cnst_lbd) * (
                            -0.5 + (1 * np.random.random()))))

                level_thresh_cnst = max(
                    self._sett_asymm_level_thresh_cnst_lbd, level_thresh_cnst)

                level_thresh_cnst = min(
                    self._sett_asymm_level_thresh_cnst_ubd, level_thresh_cnst)

                level_thresh_cnst = int(level_thresh_cnst)

                if (opt_vars_cls_old.level_thresh_cnsts[col_idx] ==
                    level_thresh_cnst):

                    search_atpts += 1
                    continue

                opt_vars_cls_new.level_thresh_cnsts[col_idx] = (
                    level_thresh_cnst)

            # level_thresh_slp.
            elif (var_to_updt == 10) and (
                (self._sett_asymm_level_thresh_slp_ubd -
                 self._sett_asymm_level_thresh_slp_lbd) > 0):

                if False:
                    level_thresh_slp_probs = [
                        0.5 * mxn_ratio_red_rate,
                        1.0 - mxn_ratio_red_rate,
                        0.5 * mxn_ratio_red_rate]

                    level_thresh_slp_diff = np.random.choice(
                        [-1, 0, 1], p=level_thresh_slp_probs)

                    level_thresh_slp = (
                        opt_vars_cls_old.level_thresh_slps[col_idx] +
                        level_thresh_slp_diff)

                else:
                    level_thresh_slp = (
                        opt_vars_cls_old.level_thresh_slps[col_idx] + (
                            mxn_ratio_red_rate *
                            (self._sett_asymm_level_thresh_slp_ubd -
                             self._sett_asymm_level_thresh_slp_lbd) * (
                            -0.5 + (1 * np.random.random()))))

                level_thresh_slp = max(
                    self._sett_asymm_level_thresh_slp_lbd, level_thresh_slp)

                level_thresh_slp = min(
                    self._sett_asymm_level_thresh_slp_ubd, level_thresh_slp)

                if (opt_vars_cls_old.level_thresh_slps[col_idx] ==
                    level_thresh_slp):

                    search_atpts += 1
                    continue

                opt_vars_cls_new.level_thresh_slps[col_idx] = (
                    level_thresh_slp)

            # rand_err_sclr_cnst
            elif (var_to_updt == 11) and (
                (self._sett_asymm_rand_err_sclr_cnst_ubd -
                 self._sett_asymm_rand_err_sclr_cnst_lbd) > 0):

                if False:
                    rand_err_sclr_cnst_probs = [
                        0.5 * mxn_ratio_red_rate,
                        1.0 - mxn_ratio_red_rate,
                        0.5 * mxn_ratio_red_rate]

                    rand_err_sclr_cnst_diff = np.random.choice(
                        [-1, 0, 1], p=rand_err_sclr_cnst_probs)

                    rand_err_sclr_cnst = (
                        opt_vars_cls_old.rand_err_sclr_cnsts[col_idx] +
                        rand_err_sclr_cnst_diff)

                else:
                    rand_err_sclr_cnst = (
                        opt_vars_cls_old.rand_err_sclr_cnsts[col_idx] + (
                            mxn_ratio_red_rate *
                            (self._sett_asymm_rand_err_sclr_cnst_ubd -
                             self._sett_asymm_rand_err_sclr_cnst_lbd) * (
                            -0.5 + (1 * np.random.random()))))

                rand_err_sclr_cnst = max(
                    self._sett_asymm_rand_err_sclr_cnst_lbd,
                    rand_err_sclr_cnst)

                rand_err_sclr_cnst = min(
                    self._sett_asymm_rand_err_sclr_cnst_ubd,
                    rand_err_sclr_cnst)

                if (opt_vars_cls_old.rand_err_sclr_cnsts[col_idx] ==
                    rand_err_sclr_cnst):

                    search_atpts += 1
                    continue

                opt_vars_cls_new.rand_err_sclr_cnsts[col_idx] = (
                    rand_err_sclr_cnst)

            # rand_err_sclr_rel
            elif (var_to_updt == 12) and (
                (self._sett_asymm_rand_err_sclr_rel_ubd -
                 self._sett_asymm_rand_err_sclr_rel_lbd) > 0):

                if False:
                    rand_err_sclr_rel_probs = [
                        0.5 * mxn_ratio_red_rate,
                        1.0 - mxn_ratio_red_rate,
                        0.5 * mxn_ratio_red_rate]

                    rand_err_sclr_rel_diff = np.random.choice(
                        [-1, 0, 1], p=rand_err_sclr_rel_probs)

                    rand_err_sclr_rel = (
                        opt_vars_cls_old.rand_err_sclr_rels[col_idx] +
                        rand_err_sclr_rel_diff)

                else:
                    rand_err_sclr_rel = (
                        opt_vars_cls_old.rand_err_sclr_rels[col_idx] + (
                            mxn_ratio_red_rate *
                            (self._sett_asymm_rand_err_sclr_rel_ubd -
                             self._sett_asymm_rand_err_sclr_rel_lbd) * (
                            -0.5 + (1 * np.random.random()))))

                rand_err_sclr_rel = max(
                    self._sett_asymm_rand_err_sclr_rel_lbd, rand_err_sclr_rel)

                rand_err_sclr_rel = min(
                    self._sett_asymm_rand_err_sclr_rel_ubd, rand_err_sclr_rel)

                if (opt_vars_cls_old.rand_err_sclr_rels[col_idx] ==
                    rand_err_sclr_rel):

                    search_atpts += 1
                    continue

                opt_vars_cls_new.rand_err_sclr_rels[col_idx] = (
                    rand_err_sclr_rel)

            else:
                raise NotImplementedError(f'var_to_updt: {var_to_updt}!')

            var_updt_flag = True

            search_atpts += 1
        #======================================================================

        return opt_vars_cls_new

    def _update_sim_no_prms(self):

        # NOTE: Everytime this method is called self._rs.ft must be updated
        # before the call.

        data = np.fft.irfft(self._rs.ft, axis=0)

        probs = self._get_probs(data, True)

        self._rs.data = np.empty_like(
            self._data_ref_rltzn_srtd, dtype=np.float64)

        for i in range(self._data_ref_n_labels):
            self._rs.data[:, i] = self._data_ref_rltzn_srtd[
                np.argsort(np.argsort(probs[:, i])), i]

        self._rs.probs = probs

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
            opt_vars_cls.mxn_ratio_probss = np.full(
                self._data_ref_shape[1], 0.5, dtype=np.float)

            opt_vars_cls.mxn_ratio_margss = (
                1.0 - opt_vars_cls.mxn_ratio_probss[:])

            if self._sett_asymm_set_flag:

                n_levels = int(
                    self._sett_asymm_n_levels_lbd + (
                        (self._sett_asymm_n_levels_ubd -
                         self._sett_asymm_n_levels_lbd) * 0.5))

                max_shift_exp = (
                    self._sett_asymm_max_shift_exp_lbd + (
                        self._sett_asymm_max_shift_exp_ubd -
                        self._sett_asymm_max_shift_exp_lbd) * 0.5)

                max_shift = int(
                    self._sett_asymm_max_shift_lbd + (
                        self._sett_asymm_max_shift_ubd -
                        self._sett_asymm_max_shift_lbd) * 0.5)

                pre_vals_ratio = (
                    self._sett_asymm_pre_vals_ratio_lbd + (
                        self._sett_asymm_pre_vals_ratio_ubd -
                        self._sett_asymm_pre_vals_ratio_lbd) * 0.5)

                asymm_n_iters = int(
                    self._sett_asymm_n_iters_lbd + (
                        self._sett_asymm_n_iters_ubd -
                        self._sett_asymm_n_iters_lbd) * 0.5)

                prob_center = (
                    self._sett_asymm_prob_center_lbd + (
                        self._sett_asymm_prob_center_ubd -
                        self._sett_asymm_prob_center_lbd) * 0.5)

                pre_val_exp = (
                    self._sett_asymm_pre_val_exp_lbd + (
                        self._sett_asymm_pre_val_exp_ubd -
                        self._sett_asymm_pre_val_exp_lbd) * 0.5)

                crt_val_exp = (
                    self._sett_asymm_crt_val_exp_lbd + (
                        self._sett_asymm_crt_val_exp_ubd -
                        self._sett_asymm_crt_val_exp_lbd) * 0.5)

                level_thresh_cnst = (
                    self._sett_asymm_level_thresh_cnst_lbd + (
                        self._sett_asymm_level_thresh_cnst_ubd -
                        self._sett_asymm_level_thresh_cnst_lbd) * 0.5)

                level_thresh_slp = (
                    self._sett_asymm_level_thresh_slp_lbd + (
                        self._sett_asymm_level_thresh_slp_ubd -
                        self._sett_asymm_level_thresh_slp_lbd) * 0.5)

                rand_err_sclr_cnst = (
                    self._sett_asymm_rand_err_sclr_cnst_lbd + (
                        self._sett_asymm_rand_err_sclr_cnst_ubd -
                        self._sett_asymm_rand_err_sclr_cnst_lbd) * 0.5)

                rand_err_sclr_rel = (
                    self._sett_asymm_rand_err_sclr_rel_lbd + (
                        self._sett_asymm_rand_err_sclr_rel_ubd -
                        self._sett_asymm_rand_err_sclr_rel_lbd) * 0.5)

                opt_vars_cls.n_levelss = np.full(
                    self._data_ref_shape[1], n_levels, dtype=np.uint32)

                opt_vars_cls.max_shift_exps = np.full(
                    self._data_ref_shape[1], max_shift_exp, dtype=np.float64)

                opt_vars_cls.max_shifts = np.full(
                    self._data_ref_shape[1], max_shift, dtype=np.int64)

                opt_vars_cls.pre_vals_ratios = np.full(
                    self._data_ref_shape[1], pre_vals_ratio, dtype=np.float64)

                opt_vars_cls.asymm_n_iterss = np.full(
                    self._data_ref_shape[1], asymm_n_iters, dtype=np.uint32)

                opt_vars_cls.prob_centers = np.full(
                    self._data_ref_shape[1], prob_center, dtype=np.float64)

                opt_vars_cls.pre_val_exps = np.full(
                    self._data_ref_shape[1], pre_val_exp, dtype=np.float64)

                opt_vars_cls.crt_val_exps = np.full(
                    self._data_ref_shape[1], crt_val_exp, dtype=np.float64)

                opt_vars_cls.level_thresh_cnsts = np.full(
                    self._data_ref_shape[1], level_thresh_cnst,
                    dtype=np.int64)

                opt_vars_cls.level_thresh_slps = np.full(
                    self._data_ref_shape[1], level_thresh_slp,
                    dtype=np.float64)

                opt_vars_cls.rand_err_sclr_cnsts = np.full(
                    self._data_ref_shape[1], rand_err_sclr_cnst,
                    dtype=np.float64)

                opt_vars_cls.rand_err_sclr_rels = np.full(
                    self._data_ref_shape[1], rand_err_sclr_rel,
                    dtype=np.float64)

        else:
            rands = np.random.random(size=self._data_ref_shape[1])

            opt_vars_cls.mxn_ratio_probss = rands

            opt_vars_cls.mxn_ratio_margss = (
                1.0 - opt_vars_cls.mxn_ratio_probss[:])

            if self._sett_asymm_set_flag:

                rands = np.random.random(size=self._data_ref_shape[1])

                n_levels = (
                    self._sett_asymm_n_levels_lbd + (
                        (1 + self._sett_asymm_n_levels_ubd -
                         self._sett_asymm_n_levels_lbd) * rands))

                rands = np.random.random(size=self._data_ref_shape[1])

                max_shift_exp = (
                    self._sett_asymm_max_shift_exp_lbd + (
                        self._sett_asymm_max_shift_exp_ubd -
                        self._sett_asymm_max_shift_exp_lbd) * rands)

                rands = np.random.random(size=self._data_ref_shape[1])

                max_shift = (
                    self._sett_asymm_max_shift_lbd + (
                        1 + self._sett_asymm_max_shift_ubd -
                        self._sett_asymm_max_shift_lbd) * rands)

                rands = np.random.random(size=self._data_ref_shape[1])

                pre_vals_ratio = (
                    self._sett_asymm_pre_vals_ratio_lbd + (
                        self._sett_asymm_pre_vals_ratio_ubd -
                        self._sett_asymm_pre_vals_ratio_lbd) * rands)

                rands = np.random.random(size=self._data_ref_shape[1])

                asymm_n_iters = (
                    self._sett_asymm_n_iters_lbd + (
                        1 + self._sett_asymm_n_iters_ubd -
                        self._sett_asymm_n_iters_lbd) * rands)

                rands = np.random.random(size=self._data_ref_shape[1])

                prob_center = (
                    self._sett_asymm_prob_center_lbd + (
                        self._sett_asymm_prob_center_ubd -
                        self._sett_asymm_prob_center_lbd) * rands)

                rands = np.random.random(size=self._data_ref_shape[1])

                pre_val_exp = (
                    self._sett_asymm_pre_val_exp_lbd + (
                        self._sett_asymm_pre_val_exp_ubd -
                        self._sett_asymm_pre_val_exp_lbd) * rands)

                rands = np.random.random(size=self._data_ref_shape[1])

                crt_val_exp = (
                    self._sett_asymm_crt_val_exp_lbd + (
                        self._sett_asymm_crt_val_exp_ubd -
                        self._sett_asymm_crt_val_exp_lbd) * rands)

                rands = np.random.random(size=self._data_ref_shape[1])

                level_thresh_cnst = (
                    self._sett_asymm_level_thresh_cnst_lbd + (
                        self._sett_asymm_level_thresh_cnst_ubd -
                        self._sett_asymm_level_thresh_cnst_lbd) * rands)

                rands = np.random.random(size=self._data_ref_shape[1])

                level_thresh_slp = (
                    self._sett_asymm_level_thresh_slp_lbd + (
                        self._sett_asymm_level_thresh_slp_ubd -
                        self._sett_asymm_level_thresh_slp_lbd) * rands)

                rands = np.random.random(size=self._data_ref_shape[1])

                rand_err_sclr_cnst = (
                    self._sett_asymm_rand_err_sclr_cnst_lbd + (
                        self._sett_asymm_rand_err_sclr_cnst_ubd -
                        self._sett_asymm_rand_err_sclr_cnst_lbd) * rands)

                rands = np.random.random(size=self._data_ref_shape[1])

                rand_err_sclr_rel = (
                    self._sett_asymm_rand_err_sclr_rel_lbd + (
                        self._sett_asymm_rand_err_sclr_rel_ubd -
                        self._sett_asymm_rand_err_sclr_rel_lbd) * rands)

                opt_vars_cls.n_levelss = n_levels.astype(np.uint32)
                opt_vars_cls.max_shift_exps = max_shift_exp
                opt_vars_cls.max_shifts = max_shift.astype(np.int64)
                opt_vars_cls.pre_vals_ratios = pre_vals_ratio
                opt_vars_cls.asymm_n_iterss = asymm_n_iters.astype(np.uint32)
                opt_vars_cls.prob_centers = prob_center
                opt_vars_cls.pre_val_exps = pre_val_exp
                opt_vars_cls.crt_val_exps = crt_val_exp
                opt_vars_cls.level_thresh_cnsts = level_thresh_cnst.astype(
                    np.int64)
                opt_vars_cls.level_thresh_slps = level_thresh_slp
                opt_vars_cls.rand_err_sclr_cnsts = rand_err_sclr_cnst
                opt_vars_cls.rand_err_sclr_rels = rand_err_sclr_rel

        if self._sett_asymm_set_flag:

            asymms_rand_errs = np.random.random(self._data_ref_shape)

            data = self._rr.data.copy()

            data[data == 0] = np.nan  # Something like temperature needs this.

            rand_err_cnst_incs = np.nanmin(np.abs(data)).min(axis=0)

            rand_err_cnst = -rand_err_cnst_incs + (
                (2.0 * rand_err_cnst_incs) *
                np.random.random(self._data_ref_shape))

            rand_err_rel = np.random.random(self._data_ref_shape)

            opt_vars_cls.asymms_rand_errs = asymms_rand_errs.copy(order='f')
            opt_vars_cls.rand_err_cnst = rand_err_cnst.copy(order='f')
            opt_vars_cls.rand_err_rel = rand_err_rel.copy(order='f')
        return

    @GTGBase._timer_wrap
    def _asymmetrize_type_2(self, data, probs, opt_vars_cls):

        n_levelss = opt_vars_cls.n_levelss
        max_shift_exps = opt_vars_cls.max_shift_exps
        max_shifts = opt_vars_cls.max_shifts
        pre_vals_ratios = opt_vars_cls.pre_vals_ratios
        asymms_rand_err = opt_vars_cls.asymms_rand_errs
        asymm_n_iterss = opt_vars_cls.asymm_n_iterss
        prob_centers = opt_vars_cls.prob_centers
        pre_val_exps = opt_vars_cls.pre_val_exps
        crt_val_exps = opt_vars_cls.crt_val_exps
        level_thresh_cnsts = opt_vars_cls.level_thresh_cnsts
        level_thresh_slps = opt_vars_cls.level_thresh_slps

        rand_err_sclr_cnsts = opt_vars_cls.rand_err_sclr_cnsts
        rand_err_sclr_rels = opt_vars_cls.rand_err_sclr_rels
        rand_err_cnst = opt_vars_cls.rand_err_cnst
        rand_err_rel = opt_vars_cls.rand_err_rel

        out_data = asymmetrize_type_10_ms_cy(
            data,
            probs,
            n_levelss,
            max_shift_exps,
            max_shifts,
            pre_vals_ratios,
            asymm_n_iterss,
            asymms_rand_err,
            prob_centers,
            pre_val_exps,
            crt_val_exps,
            level_thresh_cnsts,
            level_thresh_slps,
            rand_err_sclr_cnsts,
            rand_err_sclr_rels,
            rand_err_cnst,
            rand_err_rel)

        return out_data

    def _init_iaaft(self, opt_vars_cls):

        self._rs.data_init = self._rs.data.copy(order='f')
        self._rs.probs_init = self._rs.probs.copy(order='f')

        self._rs.ft = np.fft.rfft(self._rs.data_init, axis=0)

        if False:
            self._run_iaaft(opt_vars_cls, True)

        self._update_sim_no_prms()
        return

    @GTGBase._timer_wrap
    def _run_iaaft(self, opt_vars_cls, plain_iaaft_flag=False):

        assert any([self._sett_obj_any_ss_flag, self._sett_obj_any_ms_flag])

        mag_spec_data = self._rr.data_ft_coeffs_mags.copy()
        mag_spec_probs = self._rr.probs_ft_coeffs_mags.copy()

        phs_spec_data = self._rr.data_ft_coeffs_phss.copy()
        phs_spec_probs = self._rr.probs_ft_coeffs_phss.copy()

        iaaft_n_iters = self._sett_ann_iaaft_n_iters

        if not plain_iaaft_flag:
            data = self._rs.data_init.copy(order='f')
            probs = self._rs.probs_init.copy(order='f')

        else:
            data = np.empty_like(self._data_ref_rltzn_srtd)

            for i in range(data.shape[1]):
                rand_idxs = np.argsort(np.argsort(
                    np.random.random(size=data.shape[0])))

                data[:, i] = self._data_ref_rltzn_srtd[rand_idxs, i]

            probs = self._get_probs(data, True)

            # opt_vars_cls.mxn_ratio_margss[:] = 1.0
            # opt_vars_cls.mxn_ratio_probss[:] = 0.0

        reorder_idxs_old = np.random.randint(
            0, self._rs.data.shape[0], size=self._rs.data.shape)

        order_sdiff = np.nan
        order_sdiffs_break_thresh = 1e-3

        readjust_ft_flag = False

        if ((readjust_ft_flag) and
            (self._sett_asymm_set_flag) and
            (not plain_iaaft_flag)):

            ref_pwr = mag_spec_data ** 2
            ref_pwr[0] = 0

            ref_pcorrs = np.fft.irfft(ref_pwr, axis=0)
            ref_pcorrs /= ref_pcorrs[0]

            ref_pwr = mag_spec_probs ** 2
            ref_pwr[0] = 0

            ref_scorrs = np.fft.irfft(ref_pwr, axis=0)
            ref_scorrs /= ref_scorrs[0]

            spec_crctn_cnst = 2.0

            readjust_ft_iters = 2

        else:
            readjust_ft_iters = 1

        for j in range(readjust_ft_iters):

            stn_ctr = 0
            for i in range(iaaft_n_iters * self._rs.shape[1]):

                sim_ift = np.zeros_like(data)

                if True:
                    sim_ft_margs = np.fft.rfft(data, axis=0)
                    sim_ft_probs = np.fft.rfft(probs, axis=0)

                if ((readjust_ft_flag) and
                    (j) and
                    (i == 0) and
                    (self._sett_asymm_set_flag) and
                    (not plain_iaaft_flag)):

                    # Just this produced better copula containment but worse
                    # FT fits and very bad asymmetries.
                    if False:
                        sim_pwr = np.abs(sim_ft_margs) ** 2
                        sim_pwr[0] = 0

                        sim_pcorrs = np.fft.irfft(sim_pwr, axis=0)
                        sim_pcorrs /= sim_pcorrs[0]

                        ref_sim_pcorrs_diff = spec_crctn_cnst * (
                            (ref_pcorrs - sim_pcorrs))

                        wk_pft = np.fft.rfft(
                            ref_sim_pcorrs_diff + sim_pcorrs, axis=0)

                        mag_spec_data = np.abs(wk_pft) ** 0.5

                        # TODO: phs_spec_data update as well?
                        phs_spec_data = (
                            self._rr.data_ft_coeffs_phss -
                            np.angle(sim_ft_margs))
                    #==========================================================

                    # Just this produced better FT fits but worse copula
                    # containment.
                    if True:
                        sim_pwr = np.abs(sim_ft_probs) ** 2
                        sim_pwr[0] = 0

                        sim_scorrs = np.fft.irfft(sim_pwr, axis=0)
                        sim_scorrs /= sim_scorrs[0]

                        ref_sim_scorrs_diff = spec_crctn_cnst * (
                            (ref_scorrs - sim_scorrs))

                        wk_pft = np.fft.rfft(
                            ref_sim_scorrs_diff + sim_scorrs, axis=0)

                        mag_spec_probs = np.abs(wk_pft) ** 0.5

                        # TODO: phs_spec_data update as well?
                        phs_spec_probs = (
                            self._rr.probs_ft_coeffs_phss -
                            np.angle(sim_ft_probs))
                    #==========================================================

                # Marginals auto.
                if self._sett_obj_any_ss_flag:
                    sim_phs_margs = np.angle(sim_ft_margs)

                    if self._sett_prsrv_coeffs_set_flag:
                        sim_phs_margs[self._rr.prsrv_coeffs_idxs,:] = (
                            phs_spec_data[self._rr.prsrv_coeffs_idxs,:])

                    sim_ft_new = np.empty_like(sim_ft_margs)

                    sim_ft_new.real[:] = np.cos(sim_phs_margs) * mag_spec_data
                    sim_ft_new.imag[:] = np.sin(sim_phs_margs) * mag_spec_data

                    sim_ft_new[0,:] = 0

                    sim_ift_margs_auto = np.fft.irfft(sim_ft_new, axis=0)

                    sim_ift += (
                        sim_ift_margs_auto * opt_vars_cls.mxn_ratio_margss)

                else:
                    sim_ift_margs_auto = 0.0

                # Ranks auto.
                if self._sett_obj_any_ss_flag:
                    sim_phs = np.angle(sim_ft_probs)

                    if self._sett_prsrv_coeffs_set_flag:
                        sim_phs[self._rr.prsrv_coeffs_idxs,:] = (
                            phs_spec_probs[self._rr.prsrv_coeffs_idxs,:])

                    sim_ft_new = np.empty_like(sim_ft_probs)

                    sim_ft_new.real[:] = np.cos(sim_phs) * mag_spec_probs

                    sim_ft_new.imag[:] = np.sin(sim_phs) * mag_spec_probs

                    sim_ft_new[0,:] = 0

                    sim_ift_probs_auto = np.fft.irfft(sim_ft_new, axis=0)

                    sim_ift += (
                        sim_ift_probs_auto * opt_vars_cls.mxn_ratio_probss)

                else:
                    sim_ift_probs_auto = 0.0
                #==================================================================

                # Marginals cross.
                if self._sett_obj_any_ms_flag:
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

                    sim_ift += (
                        sim_ift_margs_cross * opt_vars_cls.mxn_ratio_margss)

                else:
                    sim_ift_margs_cross = 0.0

                # Ranks cross.
                if self._sett_obj_any_ms_flag:
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

                    sim_ift += (
                        sim_ift_probs_cross * opt_vars_cls.mxn_ratio_probss)

                else:
                    sim_ift_probs_cross = 0.0
                #==================================================================

                assert isinstance(sim_ift, np.ndarray), type(sim_ift)
                assert np.all(np.isfinite(sim_ift))

                reorder_idxs_new = np.empty_like(reorder_idxs_old)
                for k in range(self._data_ref_n_labels):
                    reorder_idxs_new[:, k] = np.argsort(
                        np.argsort(sim_ift[:, k]))

                order_sdiff = 1 - np.corrcoef(
                    reorder_idxs_old.ravel(), reorder_idxs_new.ravel())[0, 1]

                assert order_sdiff >= 0, order_sdiff

                reorder_idxs_old = reorder_idxs_new

                data = np.empty_like(data, order='f')

                for k in range(self._data_ref_n_labels):
                    data[:, k] = self._data_ref_rltzn_srtd[
                        reorder_idxs_old[:, k], k]

                probs = self._get_probs(data, True)

                if order_sdiff <= order_sdiffs_break_thresh:
                    # Nothing changed.
                    break

                stn_ctr += 1
                if stn_ctr == data.shape[1]:
                    stn_ctr = 0
                #==============================================================

            if self._sett_asymm_set_flag and not plain_iaaft_flag:

                if self._sett_asymm_type == 2:
                    data = self._asymmetrize_type_2(
                        data, probs, opt_vars_cls)

                else:
                    raise NotImplementedError

                assert np.all(np.isfinite(data))

                probs = self._get_probs(data, True)

                data = np.empty_like(
                    self._data_ref_rltzn_srtd, dtype=np.float64, order='f')

                for k in range(self._data_ref_n_labels):
                    data[:, k] = self._data_ref_rltzn_srtd[
                        np.argsort(np.argsort(probs[:, k])), k]

            else:
                break

        self._rs.ft = np.fft.rfft(data, axis=0)

        opt_vars_cls.order_sdiff = order_sdiff
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

        # # IAAFTSA variable.
        # self._rs.data_init = self._rs.data.copy(order='f')
        # self._rs.probs_init = self._rs.probs.copy(order='f')

        # IAAFTSA variable.
        # Depends on the number of mixing ratio variables.
        # For now, two only.
        opt_vars_cls_old = OPTVARS()

        self._init_iaaft_opt_vars(opt_vars_cls_old, False)

        self._init_iaaft(opt_vars_cls_old)

        old_obj_val_indiv = self._get_obj_ftn_val()
        old_obj_val = old_obj_val_indiv.sum()

        init_obj_val = old_obj_val

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

            temps = [[iter_ctr, temp]]

            acpt_rates_dfrntl = [[iter_ctr, acpt_rate]]

            obj_vals_all_indiv = []
            #==================================================================

            # IAAFTSA variables.
            mxn_ratio_red_rates = [[iter_ctr, mxn_ratio_red_rate]]

            mxn_ratios_probs = [
                opt_vars_cls_old.mxn_ratio_probss.copy()]

            mxn_ratios_margs = [
                opt_vars_cls_old.mxn_ratio_margss.copy()]

            n_levelss = [opt_vars_cls_old.n_levelss.copy()]

            max_shift_exps = [opt_vars_cls_old.max_shift_exps.copy()]

            max_shifts = [opt_vars_cls_old.max_shifts.copy()]

            pre_vals_ratios = [opt_vars_cls_old.pre_vals_ratios.copy()]

            asymm_n_iterss = [opt_vars_cls_old.asymm_n_iterss.copy()]

            prob_centers = [opt_vars_cls_old.prob_centers.copy()]

            pre_val_exps = [opt_vars_cls_old.pre_val_exps.copy()]

            crt_val_exps = [opt_vars_cls_old.crt_val_exps.copy()]

            level_thresh_cnsts = [
                opt_vars_cls_old.level_thresh_cnsts.copy()]

            level_thresh_slps = [
                opt_vars_cls_old.level_thresh_slps.copy()]

            rand_err_sclr_cnsts = [opt_vars_cls_old.rand_err_sclr_cnsts.copy()]

            rand_err_sclr_rels = [opt_vars_cls_old.rand_err_sclr_rels.copy()]

            order_sdiffs = [opt_vars_cls_old.order_sdiff]
            #==================================================================

        else:
            pass

        self._rs.ft_best = self._rs.ft.copy()

        opt_vars_cls_best = opt_vars_cls_old.get_copy()

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

            if ((old_new_adj_diff > 0) and
                (new_obj_val_indiv < old_obj_val_indiv).all()):

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
                opt_vars_cls_old = opt_vars_cls_new.get_copy()

                old_obj_val = new_obj_val

                old_obj_val_indiv = new_obj_val_indiv.copy()

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
                #==============================================================

                # IAAFTSA variables.
                mxn_ratios_probs.append(
                    opt_vars_cls_new.mxn_ratio_probss.copy())

                mxn_ratios_margs.append(
                    opt_vars_cls_new.mxn_ratio_margss.copy())

                n_levelss.append(opt_vars_cls_new.n_levelss.copy())

                max_shift_exps.append(opt_vars_cls_new.max_shift_exps.copy())

                max_shifts.append(opt_vars_cls_new.max_shifts.copy())

                pre_vals_ratios.append(opt_vars_cls_new.pre_vals_ratios.copy())

                asymm_n_iterss.append(opt_vars_cls_new.asymm_n_iterss.copy())

                prob_centers.append(opt_vars_cls_new.prob_centers.copy())

                pre_val_exps.append(opt_vars_cls_new.pre_val_exps.copy())

                crt_val_exps.append(opt_vars_cls_new.crt_val_exps.copy())

                level_thresh_cnsts.append(
                    opt_vars_cls_new.level_thresh_cnsts.copy())

                level_thresh_slps.append(
                    opt_vars_cls_new.level_thresh_slps.copy())

                rand_err_sclr_cnsts.append(
                    opt_vars_cls_new.rand_err_sclr_cnsts.copy())

                rand_err_sclr_rels.append(
                    opt_vars_cls_new.rand_err_sclr_rels.copy())

                order_sdiffs.append(opt_vars_cls_new.order_sdiff)
                #==============================================================

                if new_obj_val < obj_val_min:
                    iter_wo_min_updt = 0

                    self._rs.ft_best = self._rs.ft.copy()

                    opt_vars_cls_best = opt_vars_cls_old.get_copy()

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
                    #==========================================================

                    # IAAFTSA variables.
                    mxn_ratio_red_rates.append(
                        [iter_ctr - 1, mxn_ratio_red_rate])

                    mxn_ratio_red_rate = self._get_mxn_ratio_red_rate(
                        iter_ctr, acpt_rate, mxn_ratio_red_rate)

                    mxn_ratio_red_rates.append([iter_ctr, mxn_ratio_red_rate])
                    #==========================================================

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
                        stopp_criteria,
                        init_obj_val)

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
                stopp_criteria,
                init_obj_val)

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
            #==================================================================

            # IAAFTSA variables.
            self._rs.mxn_ratio_red_rates = np.array(
                mxn_ratio_red_rates, dtype=np.float64)

            self._rs.mxn_ratios_probs = np.array(
                mxn_ratios_probs, dtype=np.float64)

            self._rs.mxn_ratios_margs = np.array(
                mxn_ratios_margs, dtype=np.float64)

            self._rs.n_levelss = np.array(n_levelss, dtype=np.uint32)

            self._rs.max_shift_exps = np.array(
                max_shift_exps, dtype=np.float64)

            self._rs.max_shifts = np.array(max_shifts, dtype=np.int64)

            self._rs.pre_vals_ratios = np.array(
                pre_vals_ratios, dtype=np.float64)

            self._rs.asymm_n_iterss = np.array(
                asymm_n_iterss, dtype=np.uint32)

            self._rs.prob_centers = np.array(prob_centers, dtype=np.float64)

            self._rs.pre_val_exps = np.array(pre_val_exps, dtype=np.float64)

            self._rs.crt_val_exps = np.array(crt_val_exps, dtype=np.float64)

            self._rs.level_thresh_cnsts = np.array(
                level_thresh_cnsts, dtype=np.int64)

            self._rs.level_thresh_slps = np.array(
                level_thresh_slps, dtype=np.float64)

            self._rs.rand_err_sclr_cnsts = np.array(
                rand_err_sclr_cnsts, dtype=np.float64)

            self._rs.rand_err_sclr_rels = np.array(
                rand_err_sclr_rels, dtype=np.float64)

            self._rs.order_sdiffs = np.array(order_sdiffs, dtype=np.float64)

            self._rs.asymms_rand_errs = (
                opt_vars_cls_new.asymms_rand_errs.copy(order='f'))

            self._rs.rand_err_cnst = (
                opt_vars_cls_new.rand_err_cnst.copy(order='f'))

            self._rs.rand_err_rel = (
                opt_vars_cls_new.rand_err_rel.copy(order='f'))

            # Best values.
            self._rs.mxn_ratio_margss_best = opt_vars_cls_best.mxn_ratio_margss

            self._rs.mxn_ratio_probss_best = opt_vars_cls_best.mxn_ratio_probss

            self._rs.n_levelss_best = opt_vars_cls_best.n_levelss

            self._rs.max_shift_exps_best = opt_vars_cls_best.max_shift_exps

            self._rs.max_shifts_best = opt_vars_cls_best.max_shifts

            self._rs.pre_vals_ratios_best = opt_vars_cls_best.pre_vals_ratios

            self._rs.asymm_n_iterss_best = opt_vars_cls_best.asymm_n_iterss

            self._rs.prob_centers_best = opt_vars_cls_best.prob_centers

            self._rs.pre_val_exps_best = opt_vars_cls_best.pre_val_exps

            self._rs.crt_val_exps_best = opt_vars_cls_best.crt_val_exps

            self._rs.level_thresh_cnsts_best = (
                opt_vars_cls_best.level_thresh_cnsts)

            self._rs.level_thresh_slps_best = (
                opt_vars_cls_best.level_thresh_slps)

            self._rs.rand_err_sclr_cnsts_best = (
                opt_vars_cls_best.rand_err_sclr_cnsts)

            self._rs.rand_err_sclr_rels_best = (
                opt_vars_cls_best.rand_err_sclr_rels)
            #==================================================================

            self._write_cls_rltzn()

            ret = stopp_criteria

        self._alg_snapshot = None
        return ret
