'''
Created on Aug 7, 2022

@author: Faizan3800X-Uni
'''

import numpy as np

from gnrctsgenr import GTGSettings

from gnrctsgenr.misc import print_sl, print_el


class IAAFTSASettings(GTGSettings):

    def __init__(self):

        GTGSettings.__init__(self)

        # This is fixed for IAAFTSA, for now.
        self._sett_data_tfm_type = 'data'

        # IAAFTSA Annealing.
        self._sett_ann_mxn_ratio_red_rate_type = None
        self._sett_ann_mxn_ratio_red_rate = None
        self._sett_ann_min_mxn_ratio_red_rate = None
        self._sett_ann_iaaft_n_iters = None

        # Asymmetrize.
        self._sett_asymm_types = (1, 2)
        self._sett_asymm_type = None
        self._sett_asymm_n_levels_lbd = None
        self._sett_asymm_n_levels_ubd = None
        self._sett_asymm_max_shift_exp_lbd = None
        self._sett_asymm_max_shift_exp_ubd = None
        self._sett_asymm_max_shift_lbd = None
        self._sett_asymm_max_shift_ubd = None
        self._sett_asymm_pre_vals_ratio_lbd = None
        self._sett_asymm_pre_vals_ratio_ubd = None
        self._sett_asymm_n_iters_lbd = None
        self._sett_asymm_n_iters_ubd = None
        self._sett_asymm_prob_center_lbd = 0.0
        self._sett_asymm_prob_center_ubd = 1.0

        # Preserve coefficients.
        self._sett_prsrv_coeffs_min_prd = None
        self._sett_prsrv_coeffs_max_prd = None
        self._sett_prsrv_coeffs_beyond_flag = None

        self._sett_ann_iaaftsa_sa_set_flag = False
        self._sett_asymm_set_flag = False
        self._sett_prsrv_coeffs_set_flag = False
        self._sett_iaaftsa_sett_verify_flag = False
        return

    def set_iaaftsa_sa_settings(
            self,
            mixing_ratio_reduction_rate_type,
            mixing_ratio_reduction_rate,
            mixing_ratio_reduction_rate_min,
            iaaft_n_iterations_max):

        '''
        Simulated annealing variables related to IAAFTSA.

        Parameters
        ----------
        mixing_ratio_reduction_rate_type : integer
            How to limit the amount of change applied to mixing ratios and
            asymmetrize functions.
            A number between 0 and 3.
            0:  No limiting performed.
            1:  A linear reduction with respect to the maximum iterations is
                applied. The more the iteration number the less the change.
            2:  Running mixing_ratio_reduction_rate starts from 1 and tapers
                to the given mixing_ratio_reduction_rate.
            3:  Reduction rate is equal to the mean acceptance rate of
                previous acceptance_rate_iterations.
        mixing_ratio_reduction_rate : float
            If mixing_ratio_reduction_rate_type is 2, then the reduction
            rate is the previous multiplied by mixing_ratio_reduction_rate.
            Should be > 0 and <= 1.
        mixing_ratio_reduction_rate_min : float
            The minimum reduction rate, below which any change is
            considered as zero. Must be >= 0 and < 1.
        iaaft_n_iterations_max : integer
            The maximum number of iterations to perform for the IAAFT
            algorithm, to match the mariginals and probability power spectra.
            Should be more than zero.
        '''

        if self._vb:
            print_sl()

            print('Setting additional IAAFTSA annealing parameters...\n')

        assert isinstance(mixing_ratio_reduction_rate_type, int), (
            'mixing_ratio_reduction_rate_type not an integer!')

        assert 0 <= mixing_ratio_reduction_rate_type <= 3, (
            'Invalid mixing_ratio_reduction_rate_type!')

        if mixing_ratio_reduction_rate_type == 2:

            assert isinstance(mixing_ratio_reduction_rate, float), (
                'mixing_ratio_reduction_rate is not a float!')

            assert 0 < mixing_ratio_reduction_rate <= 1, (
                'Invalid mixing_ratio_reduction_rate!')

        elif mixing_ratio_reduction_rate_type in (0, 1, 3):
            pass

        else:
            raise NotImplementedError(
                'Unknown mixing_ratio_reduction_rate_type!')

        assert isinstance(mixing_ratio_reduction_rate_min, float), (
            'mixing_ratio_reduction_rate_min not a float!')

        assert 0 <= mixing_ratio_reduction_rate_min < 1.0, (
            'Invalid mixing_ratio_reduction_rate_min!')

        assert isinstance(iaaft_n_iterations_max, int), (
            'iaaft_n_iterations_max not an integer!')

        assert iaaft_n_iterations_max > 0, 'Invalid iaaft_n_iterations_max!'

        self._sett_ann_min_mxn_ratio_red_rate = (
            mixing_ratio_reduction_rate_min)

        self._sett_ann_mxn_ratio_red_rate_type = (
            mixing_ratio_reduction_rate_type)

        if mixing_ratio_reduction_rate_type == 2:
            self._sett_ann_mxn_ratio_red_rate = mixing_ratio_reduction_rate

        elif mixing_ratio_reduction_rate_type in (0, 1, 3):
            pass

        else:
            raise NotImplementedError(
                'Unknown mixing_ratio_reduction_rate_type!')

        self._sett_ann_iaaft_n_iters = iaaft_n_iterations_max

        if self._vb:

            print(
                'Mixing ratio reduction rate type:',
                self._sett_ann_mxn_ratio_red_rate_type)

            print(
                'Mixing ratio reduction rate:',
                self._sett_ann_mxn_ratio_red_rate)

            print(
                'Minimum mixing ratio reduction rate:',
                self._sett_ann_min_mxn_ratio_red_rate)

            print(
                'Maximum IAAFT iterations:', self._sett_ann_iaaft_n_iters)

            print_el()

        self._sett_ann_iaaftsa_sa_set_flag = True
        return

    def set_asymmetrize_settings(
            self,
            asymmetrize_type,
            n_levels_bds,
            max_shift_exp_bds,
            max_shift_bds,
            pre_values_ratio_bds,
            asymmetrize_iterations_bds):

        f'''
        Specify the parameter bounds for the asymmetrize function that is
        called on an IAAFTed series. Passing an IAAFTed series through
        this function gives it properties that are different than the
        Gaussian properties.

        Parameters
        ----------
        asymmetrize_type : integer
            The type of the asymmetrize function to use. Can be one of
            {self._sett_asymm_types} only!
        n_levels_bds : list or tuple of two integers
            Bounds for the number of levels to use in the asymmetrize function.
            First value is the lower while the second is the upper bound.
            Both should be greater than 1. Can also be equal.
            A check is performed later to see if the given length of the
            time series can have enough of these levels. If the series is
            too short then the bounds are adjusted to N - 1 levels.
        max_shift_exp_bds : list or tuple of two floats
            To control the nonlinearity of the number of steps that
            can be shifted for a given level. The higher the level, the
            higher the max_shift_exp, the lower the number of steps by which
            the values in that level can be shifted. The shift is constant
            for steps in all levels when it is 1. Both bounds should be
            greater than zero and less than infinity.
        max_shift_bds : list or tuple of two integers
            The maximum number of time steps by which the asymmetrizing
            function can shift values in a given level. Can be positive or
            negative. Values of zero are not allowed. Absolute value should
            be greater than 1.
        pre_values_ratio_bds : list or tuple of two floats
            The ratio of the each value in a level that is mixed
            with a corresponding one. This allows for a smooth transition
            between two values. Both have to be in between zero and one.
        asymmetrize_iterations_bds : list or tuple of two integers
            The number of times a series is passed through the asymmetrizing
            function. The more this number the smoother the series gets.
            Both should be more than zero.
        '''

        if self._vb:
            print_sl()

            print('Setting asymmetrize function settings...')

        assert isinstance(asymmetrize_type, int), (
            'asymmetrize_type not an integer!')

        assert asymmetrize_type in self._sett_asymm_types, (
            'Invalid asymmetrize_type!')

        # n_levels_bds.
        assert isinstance(n_levels_bds, (list, tuple)), (
            'n_levels_bds not a list or a tuple!')

        assert len(n_levels_bds) == 2, (
            'n_levels_bds must have two elements only!')

        assert all([isinstance(n_level, int) for n_level in n_levels_bds]), (
            'All values in n_levels_bds must be integers!')

        assert n_levels_bds[0] > 0, 'Invalid value of n_level lower bounds!'

        assert n_levels_bds[1] >= n_levels_bds[0], (
            'Values in n_levels_bds must be ascending!')

        # max_shift_exp_bds.
        assert isinstance(max_shift_exp_bds, (list, tuple)), (
            'max_shift_exp_bds not a list or a tuple!')

        assert len(max_shift_exp_bds) == 2, (
            'max_shift_exp_bds must have two elements only!')

        assert all([isinstance(max_shift_exp, float)
                    for max_shift_exp in max_shift_exp_bds]), (
            'All values in max_shift_exp_bds must be floats!')

        assert max_shift_exp_bds[0] > 0, (
            'Invalid value of max_shift_exp lower bounds!')

        assert max_shift_exp_bds[1] >= max_shift_exp_bds[0], (
            'Values in n_levels_bds must be ascending!')

        assert max_shift_exp_bds[1] < np.inf, (
            'Invalid value of max_shift_exp upper bounds!')

        # max_shift_bds.
        assert isinstance(max_shift_bds, (list, tuple)), (
            'max_shift_bds not a list or a tuple!')

        assert len(max_shift_bds) == 2, (
            'max_shift_bds must have two elements only!')

        assert all([isinstance(max_shift, int)
                    for max_shift in max_shift_bds]), (
            'All values in max_shift_bds must be integers!')

        # assert max_shift_bds[0] > 1, (
        #     'Invalid value of max_shift lower bounds!')
        #
        # assert max_shift_bds[1] > 1, (
        #     'Invalid value of max_shift upper bounds!')

        assert max_shift_bds[1] >= max_shift_bds[0], (
            'Values in max_shift_bds must be ascending!')

        # pre_values_ratio_bds.
        assert isinstance(pre_values_ratio_bds, (list, tuple)), (
            'pre_values_ratio_bds not a list or a tuple!')

        assert len(pre_values_ratio_bds) == 2, (
            'pre_values_ratio_bds must have two elements only!')

        assert all([isinstance(pre_values_ratio, float)
                    for pre_values_ratio in pre_values_ratio_bds]), (
            'All values in pre_values_ratio_bds must be floats!')

        # assert pre_values_ratio_bds[0] >= 0, (
        #     'Invalid value of pre_values_ratio lower bounds!')

        assert pre_values_ratio_bds[1] >= pre_values_ratio_bds[0], (
            'Values in n_levels_bds must be ascending!')

        # assert pre_values_ratio_bds[1] <= 1, (
        #     'Invalid value of pre_values_ratio upper bounds!')

        # asymmetrize_iterations_bds.
        assert isinstance(asymmetrize_iterations_bds, (list, tuple)), (
            'asymmetrize_iterations_bds not a list or a tuple!')

        assert len(asymmetrize_iterations_bds) == 2, (
            'asymmetrize_iterations_bds must have two elements only!')

        assert all([isinstance(asymmetrize_iterations, int)
                    for asymmetrize_iterations in asymmetrize_iterations_bds
                    ]), (
            'All values in asymmetrize_iterations_bds must be integers!')

        assert asymmetrize_iterations_bds[0] >= 1, (
            'Invalid value of asymmetrize_iterations lower bounds!')

        assert (asymmetrize_iterations_bds[1] >=
                asymmetrize_iterations_bds[0]), (
            'Values in asymmetrize_iterations_bds must be ascending!')

        self._sett_asymm_type = asymmetrize_type

        self._sett_asymm_n_levels_lbd = n_levels_bds[0]
        self._sett_asymm_n_levels_ubd = n_levels_bds[1]

        self._sett_asymm_max_shift_exp_lbd = max_shift_exp_bds[0]
        self._sett_asymm_max_shift_exp_ubd = max_shift_exp_bds[1]

        self._sett_asymm_max_shift_lbd = max_shift_bds[0]
        self._sett_asymm_max_shift_ubd = max_shift_bds[1]

        self._sett_asymm_pre_vals_ratio_lbd = pre_values_ratio_bds[0]
        self._sett_asymm_pre_vals_ratio_ubd = pre_values_ratio_bds[1]

        self._sett_asymm_n_iters_lbd = asymmetrize_iterations_bds[0]
        self._sett_asymm_n_iters_ubd = asymmetrize_iterations_bds[1]

        if self._vb:
            print('Asymmetrize type:', self._sett_asymm_type)

            print(
                'Number of levels\' bounds:',
                self._sett_asymm_n_levels_lbd ,
                self._sett_asymm_n_levels_ubd)

            print(
                'Maximum shift exponent bounds:',
                self._sett_asymm_max_shift_exp_lbd,
                self._sett_asymm_max_shift_exp_ubd)

            print(
                'Maximum shift bounds:',
                self._sett_asymm_max_shift_lbd,
                self._sett_asymm_max_shift_ubd)

            print(
                'Previous value ratio bounds:',
                self._sett_asymm_pre_vals_ratio_lbd,
                self._sett_asymm_pre_vals_ratio_ubd)

            print(
                'Number of asymmetrize calls\' bounds:',
                self._sett_asymm_n_iters_lbd,
                self._sett_asymm_n_iters_ubd)

            print_el()

        self._sett_asymm_set_flag = True
        return

    def set_preserve_coefficients_subset_settings(
            self, min_period, max_period, keep_beyond_flag):

        '''
        Preserve a subset of coefficients in the simulated time series by
        setting them equal to the reference.

        Coefficients having periods less than min_period and greater
        than max period (or vice versa) are always taken from the
        reference data. This is important for the cases such as the
        annual and seasonal cycles.

        Parameters:
        ----------
        min_period : int or None
            Coefficients having periods less (or greater) than min_period are
            overwritten in simulations with the corresponding ones
            from the reference. Should be greater than zero and less than
            max_period. An error is raised, later on, if min_period does
            not exist in the data.
        max_period : int or None
            Coefficients having periods greater (or less) than max_period
            are overwritten in simulations with the corresponding ones
            from the reference. Should be greater than zero and
            greater than min_period. An error is raised, later on, if
            max_period does not exist in the data.
        keep_beyond_flag : bool
            Whether to keep coefficients beyond the min_period and max_period
            or within them. A value of True would keep the periods that
            have periods greater than max_period and less than min_period.
            A value of False would keep the periods whose periods are
            within min_period and max_period (including both).
            If True, then min_period and/or max_period can be None.
            If False, then both min_period and max_period should have valid
            integer values.
        '''

        if self._vb:
            print_sl()

            print(
                'Setting preserve coefficients for IAAFTSA...\n')

        if isinstance(min_period, int):
            assert min_period > 0, 'Invalid min_period!'

        elif min_period is None:
            pass

        else:
            raise AssertionError('min_period can only be None or an int!')

        if isinstance(max_period, int):
            assert max_period > 0, 'Invalid max_period!'

        elif max_period is None:
            pass

        else:
            raise AssertionError('max_period can only be None or an int!')

        if isinstance(min_period, int) and isinstance(max_period, int):
            assert max_period > min_period, (
                'max_period must be greater than min_period!')

        assert isinstance(keep_beyond_flag, bool), (
            'keep_beyond_flag can only be a boolean!')

        if not keep_beyond_flag:
            assert (
                (min_period is not None) and
                (max_period is not None)), (
                    'Both min_period and max_period should be valid '
                    'integers if keep_beyond_flag is False!')

        self._sett_prsrv_coeffs_min_prd = min_period
        self._sett_prsrv_coeffs_max_prd = max_period
        self._sett_prsrv_coeffs_beyond_flag = keep_beyond_flag

        if self._vb:
            print('Minimum period:', self._sett_prsrv_coeffs_min_prd)
            print('Maximum period:', self._sett_prsrv_coeffs_max_prd)
            print('Keep beyond flag:', self._sett_prsrv_coeffs_beyond_flag)

            print_el()

        self._sett_prsrv_coeffs_set_flag = True
        return

    def verify(self):

        assert self._sett_ann_iaaftsa_sa_set_flag, (
            'Call set_iaaft_sa_settings first!')

        GTGSettings._GTGSettings__verify(self)

        self._sett_iaaftsa_sett_verify_flag = True
        return

    __verify = verify
