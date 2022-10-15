'''
Created on Aug 7, 2022

@author: Faizan3800X-Uni
'''
import numpy as np

from gnrctsgenr import (
    GTGPrepareRltznRef,
    GTGPrepareRltznSim,
    GTGPrepareTfms,
    GTGPrepare,
    )

from gnrctsgenr.misc import print_sl, print_el


class IAAFTSAPrepareRltznRef(GTGPrepareRltznRef):

    def __init__(self):

        GTGPrepareRltznRef.__init__(self)

        self.prsrv_phss_idxs_margs = None
        self.prsrv_phss_idxs_ranks = None
        return


class IAAFTSAPrepareRltznSim(GTGPrepareRltznSim):

    def __init__(self):

        GTGPrepareRltznSim.__init__(self)

        # Init random data that is shuffled everytime.
        self.data_init = None
        self.probs_init = None

        # For IAAFT.
        self.mxn_ratio_margss = None
        self.mxn_ratio_probss = None

        # For asymmetrize.
        self.n_levelss = None
        self.max_shift_exps = None
        self.max_shifts = None
        self.pre_vals_ratios = None
        self.asymm_n_iterss = None
        self.asymms_rand_errs = None
        self.prob_centers = None
        self.pre_val_exps = None
        self.crt_val_exps = None
        self.level_thresh_cnsts = None
        self.level_thresh_slps = None
        self.rand_err_sclr_cnsts = None
        self.rand_err_sclr_rels = None
        self.rand_err_cnst = None
        self.rand_err_rel = None
        self.probs_exps = None

        # For IAAFT convergence monitoring.
        self.order_sdiffs = None

        # Final best values.
        self.mxn_ratio_margss_best = None
        self.mxn_ratio_probss_best = None
        self.n_levelss_best = None
        self.max_shift_exps_best = None
        self.max_shifts_best = None
        self.pre_vals_ratios_best = None
        self.asymm_n_iterss_best = None
        self.prob_centers_best = None
        self.pre_val_exps_best = None
        self.crt_val_exps_best = None
        self.level_thresh_cnsts_best = None
        self.level_thresh_slps_best = None
        self.rand_err_sclr_cnsts_best = None
        self.rand_err_sclr_rels_best = None
        self.rand_err_cnst_best = None
        self.rand_err_rel_best = None
        self.probs_exp_best = None
        return


class IAAFTSAPrepareTfms(GTGPrepareTfms):

    def __init__(self):

        GTGPrepareTfms.__init__(self)
        return

    def _get_shuffle_ser_ft(self):

        data = np.empty_like(self._rr.data)

        for i in range(data.shape[1]):
            rand_idxs = np.argsort(np.argsort(
                np.random.random(size=data.shape[0])))

            data[:, i] = self._data_ref_rltzn_srtd[rand_idxs, i]

        ft = np.fft.rfft(data, axis=0)
        return ft


class IAAFTSAPrepare(GTGPrepare):

    def __init__(self):

        GTGPrepare.__init__(self)
        return

    def _get_lock_phss_idxs(self, mags, ref_sers_srtd):

        max_lim_idx = int(
            (1.0 - self._sett_prsrv_phss_auto_alpha) *
            self._sett_prsrv_phss_auto_n_sims)

        assert 0 <= max_lim_idx <= self._sett_prsrv_phss_auto_n_sims, (
            max_lim_idx, self._sett_prsrv_phss_auto_n_sims)

        min_lim_idx = int(
            self._sett_prsrv_phss_auto_alpha *
            self._sett_prsrv_phss_auto_n_sims)

        assert 0 <= min_lim_idx <= self._sett_prsrv_phss_auto_n_sims, (
            min_lim_idx, self._sett_prsrv_phss_auto_n_sims)

        assert max_lim_idx >= min_lim_idx, (min_lim_idx, max_lim_idx)

        # In the case that self._sett_prsrv_phss_auto_alpha is zero.
        if max_lim_idx == self._sett_prsrv_phss_auto_n_sims:
            max_lim_idx -= 1

        lock_phss_idxs = np.zeros(mags.shape, dtype=bool, order='f')

        sim_magss = np.empty(
            (self._sett_prsrv_phss_auto_n_sims, mags.shape[0]))

        for i in range(mags.shape[1]):
            for j in range(self._sett_prsrv_phss_auto_n_sims):

                sim_ser = ref_sers_srtd[:, i].copy()

                np.random.shuffle(sim_ser)

                sim_mags = np.abs(np.fft.rfft(sim_ser))

                sim_magss[j,:] = sim_mags

            sim_magss.sort(axis=0)

            within_lim_idxs = (
                (sim_magss[min_lim_idx,:] <= mags[:, i]) &
                ((sim_magss[max_lim_idx,:] >= mags[:, i])))

            lock_phss_idxs[:, i] = ~within_lim_idxs

        lock_phss_idxs[0,:] = True
        lock_phss_idxs[-1,:] = True
        return lock_phss_idxs

    def _set_prsrv_phss_idxs(self):
        if self._sett_prsrv_phss_auto_set_flag:
            self._rr.prsrv_phss_idxs_margs = self._get_lock_phss_idxs(
                self._rr.data_ft_coeffs_mags,
                self._data_ref_rltzn_srtd)

            self._rr.prsrv_phss_idxs_ranks = self._get_lock_phss_idxs(
                self._rr.probs_ft_coeffs_mags,
                self._rr.probs_srtd)

        else:
            periods = self._rr.probs.shape[0] / (
                np.arange(1, self._rr.ft.shape[0] - 1))

            prsrv_phss_idxs = np.zeros(
                self._rr.ft.shape[0] - 2, dtype=bool)

            if self._sett_prsrv_coeffs_min_prd is not None:
                assert periods.min() <= self._sett_prsrv_coeffs_min_prd, (
                    'Minimum period does not exist in data!')

                assert periods.max() > self._sett_prsrv_coeffs_min_prd, (
                    'Data maximum period greater than or equal to min_period!')

            if self._sett_prsrv_coeffs_max_prd is not None:
                assert periods.max() >= self._sett_prsrv_coeffs_max_prd, (
                    'Maximum period does not exist in data!')

            if self._sett_prsrv_coeffs_beyond_flag:
                if self._sett_prsrv_coeffs_min_prd is not None:
                    prsrv_phss_idxs[
                        periods < self._sett_prsrv_coeffs_min_prd] = True

                if self._sett_prsrv_coeffs_max_prd is not None:
                    prsrv_phss_idxs[
                        periods > self._sett_prsrv_coeffs_max_prd] = True

            if self._sett_prsrv_coeffs_set_flag:
                assert prsrv_phss_idxs.sum(), (
                    'Incorrect min_period or max_period, '
                    'no coefficients selected for IAAFTSA!')

            prsrv_phss_idxs = np.concatenate(
                ([True], prsrv_phss_idxs, [True])).reshape(-1, 1)

            prsrv_phss_idxs = np.concatenate(
                [prsrv_phss_idxs.copy(order='f')] * self._data_ref_shape[1],
                axis=1)

            self.prsrv_phss_idxs_margs = prsrv_phss_idxs.copy(order='f')
            self.prsrv_phss_idxs_ranks = prsrv_phss_idxs.copy(order='f')

        return

    def _gen_ref_aux_data(self):

        self._gen_ref_aux_data_gnrc()

        self._set_prsrv_phss_idxs()

        self._prep_ref_aux_flag = True
        return

    def _gen_sim_aux_data(self):

        assert self._prep_ref_aux_flag, 'Call _gen_ref_aux_data first!'

        if self._data_ref_rltzn.ndim != 2:
            raise NotImplementedError('Implementation for 2D only!')

        self._rs.shape = (1 + (self._data_ref_shape[0] // 2),
            self._data_ref_n_labels)

        ft = self._get_shuffle_ser_ft()

        assert np.all(np.isfinite(ft)), 'Invalid values in ft!'

        data = np.fft.irfft(ft, axis=0)

        assert np.all(np.isfinite(data)), 'Invalid values in data!'

        probs = self._get_probs(data, True)

        self._rs.data = np.empty_like(
            self._data_ref_rltzn_srtd, dtype=np.float64)

        for i in range(self._data_ref_n_labels):
            self._rs.data[:, i] = self._data_ref_rltzn_srtd[
                np.argsort(np.argsort(probs[:, i])), i]

        self._rs.probs = probs

        self._rs.ft = np.fft.rfft(self._rs.data, axis=0)
        self._rs.phs_spec = np.angle(self._rs.ft)
        self._rs.mag_spec = np.abs(self._rs.ft)

        self._rs.data_ft_coeffs = np.fft.rfft(self._rs.data, axis=0)
        self._rs.data_ft_coeffs_mags = np.abs(self._rs.data_ft_coeffs)
        self._rs.data_ft_coeffs_phss = np.angle(self._rs.data_ft_coeffs)

        self._rs.probs_ft_coeffs = np.fft.rfft(self._rs.probs, axis=0)
        self._rs.probs_ft_coeffs_mags = np.abs(self._rs.probs_ft_coeffs)
        self._rs.probs_ft_coeffs_phss = np.angle(self._rs.probs_ft_coeffs)

        self._update_obj_vars('sim')

        self._prep_sim_aux_flag = True
        return

    def verify(self):

        assert self._sett_data_tfm_type == 'data', (
            'self._sett_data_tfm_type can only be \'data\'!')

        if False and self._sett_asymm_set_flag:
            adj_flag = False
            # if self._sett_asymm_n_levels_lbd >= self._data_ref_shape[0]:
            #     self._sett_asymm_n_levels_lbd = self._data_ref_shape[0] - 1
            #     adj_flag = True

            # if self._sett_asymm_n_levels_ubd >= self._data_ref_shape[0]:
            #     self._sett_asymm_n_levels_ubd = self._data_ref_shape[0] - 1
            #     adj_flag = True

            if self._sett_asymm_max_shift_lbd >= self._data_ref_shape[0]:
                self._sett_asymm_max_shift_lbd = self._data_ref_shape[0] - 1
                adj_flag = True

            if self._sett_asymm_max_shift_ubd >= self._data_ref_shape[0]:
                self._sett_asymm_max_shift_ubd = self._data_ref_shape[0] - 1
                adj_flag = True

            if self._sett_asymm_max_shift_lbd <= -self._data_ref_shape[0]:
                self._sett_asymm_max_shift_lbd = -(
                    self._data_ref_shape[0] - 1)

                adj_flag = True

            if self._sett_asymm_max_shift_ubd <= -self._data_ref_shape[0]:
                self._sett_asymm_max_shift_ubd = -(
                    self._data_ref_shape[0] - 1)

                adj_flag = True

            if adj_flag:
                print_sl()

                print(
                    'INFO: Needed to adjust asymmetrize function bounds '
                    'due to series length being shorter!')

                print(
                    f'New bounds for n_levels are: '
                    f'({self._sett_asymm_n_levels_lbd}, '
                    f'{self._sett_asymm_n_levels_ubd})')

                print(
                    f'New bounds for max_shift are: '
                    f'({self._sett_asymm_max_shift_lbd}, '
                    f'{self._sett_asymm_max_shift_ubd})')

                print_el()

        if self._sett_psc_ms_flag:
            assert self._data_ref_shape[1] > 1, (
                'For apply_ms_flag, their should be more than one column '
                'in the input data!')

        GTGPrepare._GTGPrepare__verify(self)
        return

    __verify = verify
