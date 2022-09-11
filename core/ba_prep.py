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

        self.prsrv_coeffs_idxs = None
        self.not_prsrv_coeffs_idxs = None
        return


class IAAFTSAPrepareRltznSim(GTGPrepareRltznSim):

    def __init__(self):

        GTGPrepareRltznSim.__init__(self)

        self.init_data = None
        self.init_probs = None
        return


class IAAFTSAPrepareTfms(GTGPrepareTfms):

    def __init__(self):

        GTGPrepareTfms.__init__(self)
        return

    # def _get_sim_ft_pln(self):
    #
    #     '''
    #     Plain phase randomization.
    #
    #     Not the same as phsann.
    #     '''
    #
    #     ft = np.zeros(self._rs.shape, dtype=np.complex)
    #
    #     mag_spec = self._rr.mag_spec.copy()
    #
    #     rands = np.random.random((self._rs.shape[0], 1))
    #
    #     rands = 1.0 * (-np.pi + (2 * np.pi * rands))
    #
    #     rands[self._rr.prsrv_coeffs_idxs] = 0.0
    #
    #     phs_spec = self._rr.phs_spec[:,:].copy()
    #
    #     phs_spec += rands  # out of bound phs
    #
    #     ft.real[:,:] = mag_spec[:,:] * np.cos(phs_spec)
    #     ft.imag[:,:] = mag_spec[:,:] * np.sin(phs_spec)
    #
    #     # First and last coefficients are not written to anywhere, normally.
    #     ft[+0,:] = self._rr.ft[+0].copy()
    #     ft[-1,:] = self._rr.ft[-1].copy()
    #
    #     return ft

    def _get_shuffle_ser_ft(self):

        data = self._rr.data.copy()

        data_sorted = np.sort(data, axis=0)

        for i in range(data.shape[1]):
            rand_idxs = np.argsort(np.argsort(
                np.random.random(size=data.shape[0])))

            data[:, i] = data_sorted[rand_idxs, i]

        ft = np.fft.rfft(data, axis=0)
        return ft


class IAAFTSAPrepare(GTGPrepare):

    def __init__(self):

        GTGPrepare.__init__(self)
        return

    def _set_prsrv_coeffs_idxs(self):

        periods = self._rr.probs.shape[0] / (
            np.arange(1, self._rr.ft.shape[0] - 1))

        self._rr.prsrv_coeffs_idxs = np.zeros(
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
                self._rr.prsrv_coeffs_idxs[
                    periods < self._sett_prsrv_coeffs_min_prd] = True

            if self._sett_prsrv_coeffs_max_prd is not None:
                self._rr.prsrv_coeffs_idxs[
                    periods > self._sett_prsrv_coeffs_max_prd] = True

        # else:
        #     self._rr.prsrv_coeffs_idxs[
        #         (periods >= self._sett_prsrv_coeffs_min_prd) &
        #         (periods <= self._sett_prsrv_coeffs_max_prd)] = True

        if self._sett_prsrv_coeffs_set_flag:
            assert self._rr.prsrv_coeffs_idxs.sum(), (
                'Incorrect min_period or max_period, '
                'no coefficients selected for IAAFTSA!')

        self._rr.prsrv_coeffs_idxs = np.concatenate(
            ([True], self._rr.prsrv_coeffs_idxs, [True]))

        self._rr.not_prsrv_coeffs_idxs = ~self._rr.prsrv_coeffs_idxs
        return

    def _gen_ref_aux_data(self):

        self._gen_ref_aux_data_gnrc()

        self._set_prsrv_coeffs_idxs()

        # self._rr.coeffs_idxs = np.arange(
        #     1, self._rr.ft.shape[0] - 1)[self._rr.not_prsrv_coeffs_idxs]

        self._prep_ref_aux_flag = True
        return

    def _gen_sim_aux_data(self):

        assert self._prep_ref_aux_flag, 'Call _gen_ref_aux_data first!'

        if self._data_ref_rltzn.ndim != 2:
            raise NotImplementedError('Implementation for 2D only!')

        self._rs.shape = (1 + (self._data_ref_shape[0] // 2),
            self._data_ref_n_labels)

        # ft = self._get_sim_ft_pln()
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

        if self._sett_asymm_set_flag:
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

        GTGPrepare._GTGPrepare__verify(self)
        return

    __verify = verify
