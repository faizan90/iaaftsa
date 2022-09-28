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
        self._sett_asymm_types = (2,)
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
        self._sett_asymm_prob_center_lbd = None
        self._sett_asymm_prob_center_ubd = None
        self._sett_asymm_pre_val_exp_lbd = None
        self._sett_asymm_pre_val_exp_ubd = None
        self._sett_asymm_crt_val_exp_lbd = None
        self._sett_asymm_crt_val_exp_ubd = None
        self._sett_asymm_level_thresh_cnst_lbd = None
        self._sett_asymm_level_thresh_cnst_ubd = None
        self._sett_asymm_level_thresh_slp_lbd = None
        self._sett_asymm_level_thresh_slp_ubd = None
        self._sett_asymm_rand_err_sclr_cnst_lbd = None
        self._sett_asymm_rand_err_sclr_cnst_ubd = None
        self._sett_asymm_rand_err_sclr_rel_lbd = None
        self._sett_asymm_rand_err_sclr_rel_ubd = None
        self._sett_asymm_probs_exp_lbd = None
        self._sett_asymm_probs_exp_ubd = None

        # Preserve coefficients.
        self._sett_prsrv_coeffs_min_prd = None
        self._sett_prsrv_coeffs_max_prd = None
        self._sett_prsrv_coeffs_beyond_flag = None

        # Flags.
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
            asymmetrize_iterations_bds,
            prob_center_bds,
            pre_val_exp_bds,
            crt_val_exp_bds,
            level_thresh_cnst_bds,
            level_thresh_slp_bds,
            rand_err_sclr_cnst_bds,
            rand_err_sclr_rel_bds,
            probs_exp_bds):

        f'''
        Specify the parameter bounds for the asymmetrize function that is
        called on an IAAFTed series. Passing an IAAFTed series through
        this function gives it properties that are different than that of
        a Gaussian series.

        Parameters
        ----------
        asymmetrize_type : integer
            The type of the asymmetrize function to use. Can be one of
            {self._sett_asymm_types} only!
        n_levels_bds : list or tuple of two integers
            Bounds for the number of levels to use in the asymmetrize function.
            First value is the lower while the second is the upper bound.
            Both should be >= 0. Can also be equal.
        max_shift_exp_bds : list or tuple of two floats
            To control the non-linearity of the number of steps that
            can be shifted for a given level. The higher the level, the
            higher the max_shift_exp, the lower the number of steps by which
            the values in that level can be shifted. Both bounds should be
            >= zero and < infinity.
        max_shift_bds : list or tuple of two integers
            The maximum number of time steps by which the asymmetrizing
            function can shift values in a given level. Can be positive or
            negative. Negative values will reverse the direction of series.
        pre_values_ratio_bds : list or tuple of two floats
            The ratio of the each value in a level that is mixed
            with a corresponding one. This allows for a smooth transition
            between two values.
        asymmetrize_iterations_bds : list or tuple of two integers
            The number of times a series is passed through the asymmetrizing
            function. The more this number the smoother the series gets.
            Both should be >= zero.
        prob_center_bds : list or tuple of two floats
            At which F(x) value to take the lowest level. Should be between
            0 and 1, including.
        pre_val_exp_bds : list or tuple of two floats
            The exponent to which a current value is raised before it is
            multiplied by pre_values_ratio. The mismatch between pre_val_exp
            and crt_val_exp allows for an asymmetric simulation around a
            value. Should be a valid real number equal to or greater than
            zero. The simulation applies the cabs function to take the
            absolute in case resulting value is a complex number. The sign
            is maintained.
        crt_val_exp_bds : list or tuple of two floats
            Same as pre_val_exp_bds but the shifted value is raised to this
            exponent and then muliplied by 1 - pre_values_ratio.
        level_thresh_cnst_bds : list or tuple of two integers
            Selective steps can be only modified given they are within a
            certain limit of levels to the current one. This way values
            that are too far away are not taken into account. This allows for
            asymmetric results. level_thresh_cnst_bds contain the bounds
            on the constant maximum difference that the level at the current
            step and the level that is currently considered can have.
            The equation for this threshold is:
            level_thresh = level_thresh_cnst + (level_thresh_slp * level).
            Can be positive or negative values.
        level_thresh_slp_bds : list or tuple of two floats
            Allows for tapering of the threshold as the level increases.
            This allows for different thresholds based on the current
            level considered. The equation is given in the previous parameter.
        rand_err_sclr_cnst_bds : list or tuple of two floats
            The series returned by the asymmetrizing function are without
            errors i.e. no measurement error. This creates series that have
            lower noise than the reference series. For this purpose two
            errors are added to the final outputs. One is a constant
            random term and the other is a term relative to the magnitude.
            A constant is taken (the non-zero absolute minimum for each
            variable), and it is scaled randomly by multiplying random
            numbers between zero and one and then by randomly taking a number
            between the two values in rand_err_sclr_cnst_bds. All this is
            done to the final asymmetrized series.
        rand_err_sclr_rel_bds : list or tuple of two floats
            Coming from the previous parameter, this term allows for errors
            that are relative to the asymmetrized data at the end. This can
            be seen as a plus minus error in percentage applied to the
            asymmetrized data.
        probs_exp_bds :  list or tuple of two floats
            The level of a given value in the input data is computed based
            on its non-exceedance probability. This probability can be raised
            to a positive power to produce levels that are not uniformly
            distributed between zero and one. Should be >= zero and < Infinity.
        '''

        if self._vb:
            print_sl()

            print('Setting asymmetrize function settings...')
        #======================================================================

        # Type.
        assert isinstance(asymmetrize_type, int), (
            'asymmetrize_type not an integer!')

        assert asymmetrize_type in self._sett_asymm_types, (
            'Invalid asymmetrize_type!')
        #======================================================================

        # n_levels_bds.
        assert isinstance(n_levels_bds, (list, tuple)), (
            'n_levels_bds not a list or a tuple!')

        assert len(n_levels_bds) == 2, (
            'n_levels_bds must have two elements only!')

        assert all([isinstance(n_level, int) for n_level in n_levels_bds]), (
            'All values in n_levels_bds must be integers!')

        assert n_levels_bds[0] >= 0, 'Invalid value of n_level lower bounds!'

        assert n_levels_bds[1] >= n_levels_bds[0], (
            'Values in n_levels_bds must be ascending!')
        #======================================================================

        # max_shift_exp_bds.
        assert isinstance(max_shift_exp_bds, (list, tuple)), (
            'max_shift_exp_bds not a list or a tuple!')

        assert len(max_shift_exp_bds) == 2, (
            'max_shift_exp_bds must have two elements only!')

        assert all([isinstance(max_shift_exp, float)
                    for max_shift_exp in max_shift_exp_bds]), (
            'All values in max_shift_exp_bds must be floats!')

        assert all([np.isfinite(max_shift_exp_bd)
                    for max_shift_exp_bd in max_shift_exp_bds]), (
            'All values in max_shift_exp_bds must be finite!')

        assert max_shift_exp_bds[0] >= 0, (
            'Invalid value of max_shift_exp lower bounds!')

        assert max_shift_exp_bds[1] >= max_shift_exp_bds[0], (
            'Values in max_shift_exp_bds must be ascending!')

        assert max_shift_exp_bds[1] < np.inf, (
            'Invalid value of max_shift_exp upper bounds!')
        #======================================================================

        # max_shift_bds.
        assert isinstance(max_shift_bds, (list, tuple)), (
            'max_shift_bds not a list or a tuple!')

        assert len(max_shift_bds) == 2, (
            'max_shift_bds must have two elements only!')

        assert all([isinstance(max_shift, int)
                    for max_shift in max_shift_bds]), (
            'All values in max_shift_bds must be integers!')

        assert max_shift_bds[1] >= max_shift_bds[0], (
            'Values in max_shift_bds must be ascending!')
        #======================================================================

        # pre_values_ratio_bds.
        assert isinstance(pre_values_ratio_bds, (list, tuple)), (
            'pre_values_ratio_bds not a list or a tuple!')

        assert len(pre_values_ratio_bds) == 2, (
            'pre_values_ratio_bds must have two elements only!')

        assert all([isinstance(pre_values_ratio, float)
                    for pre_values_ratio in pre_values_ratio_bds]), (
            'All values in pre_values_ratio_bds must be floats!')

        assert all([np.isfinite(pre_values_ratio_bd)
                    for pre_values_ratio_bd in pre_values_ratio_bds]), (
            'All values in pre_values_ratio_bds must be finite!')

        assert pre_values_ratio_bds[1] >= pre_values_ratio_bds[0], (
            'Values in pre_values_ratio_bds must be ascending!')
        #======================================================================

        # asymmetrize_iterations_bds.
        assert isinstance(asymmetrize_iterations_bds, (list, tuple)), (
            'asymmetrize_iterations_bds not a list or a tuple!')

        assert len(asymmetrize_iterations_bds) == 2, (
            'asymmetrize_iterations_bds must have two elements only!')

        assert all([isinstance(asymmetrize_iterations, int)
                    for asymmetrize_iterations in asymmetrize_iterations_bds
                    ]), (
            'All values in asymmetrize_iterations_bds must be integers!')

        assert asymmetrize_iterations_bds[0] >= 0, (
            'Invalid value of asymmetrize_iterations lower bounds!')

        assert (asymmetrize_iterations_bds[1] >=
                asymmetrize_iterations_bds[0]), (
            'Values in asymmetrize_iterations_bds must be ascending!')
        #======================================================================

        # prob_center_bds.
        assert isinstance(prob_center_bds, (list, tuple)), (
            'prob_center_bds not a list or a tuple!')

        assert len(prob_center_bds) == 2, (
            'prob_center_bds must have two elements only!')

        assert all([isinstance(prob_center, float)
                    for prob_center in prob_center_bds]), (
            'All values in prob_center_bds must be floats!')

        assert all([np.isfinite(prob_center_bd)
                    for prob_center_bd in prob_center_bds]), (
            'All values in prob_center_bds must be finite!')

        assert prob_center_bds[0] >= 0, (
            'Invalid value of prob_center lower bounds!')

        assert prob_center_bds[1] >= prob_center_bds[0], (
            'Values in prob_center_bds must be ascending!')

        assert prob_center_bds[1] <= 1.0, (
            'Invalid value of prob_center upper bounds!')
        #======================================================================

        # pre_val_exp_bds.
        assert isinstance(pre_val_exp_bds, (list, tuple)), (
            'pre_val_exp_bds not a list or a tuple!')

        assert len(pre_val_exp_bds) == 2, (
            'pre_val_exp_bds must have two elements only!')

        assert all([isinstance(pre_val_exp_bd, float)
                    for pre_val_exp_bd in pre_val_exp_bds]), (
            'All values in pre_val_exp_bds must be floats!')

        assert all([np.isfinite(pre_val_exp_bd)
                    for pre_val_exp_bd in pre_val_exp_bds]), (
            'All values in pre_val_exp_bds must be finite!')

        assert pre_val_exp_bds[0] >= 0, (
            'Invalid value of lower bound in pre_val_exps!')

        assert pre_val_exp_bds[0] <= pre_val_exp_bds[1], (
            'Values in pre_val_exp_bds must be ascending!')
        #======================================================================

        # crt_val_exp_bds.
        assert isinstance(crt_val_exp_bds, (list, tuple)), (
            'crt_val_exp_bds not a list or a tuple!')

        assert len(crt_val_exp_bds) == 2, (
            'crt_val_exp_bds must have two elements only!')

        assert all([isinstance(crt_val_exp_bd, float)
                    for crt_val_exp_bd in crt_val_exp_bds]), (
            'All values in crt_val_exp_bds must be floats!')

        assert all([np.isfinite(crt_val_exp_bd)
                    for crt_val_exp_bd in crt_val_exp_bds]), (
            'All values in crt_val_exp_bds must be finite!')

        assert crt_val_exp_bds[0] >= 0, (
            'Invalid value of lower bound in crt_val_exps!')

        assert crt_val_exp_bds[0] <= crt_val_exp_bds[1], (
            'Values in crt_val_exp_bds must be ascending!')
        #======================================================================

        # level_thresh_cnst_bds.
        assert isinstance(level_thresh_cnst_bds, (list, tuple)), (
            'level_thresh_cnst_bds not a list or a tuple!')

        assert len(level_thresh_cnst_bds) == 2, (
            'level_thresh_cnst_bds must have two elements only!')

        assert all([isinstance(level_thresh_cnst, int)
                    for level_thresh_cnst in level_thresh_cnst_bds]), (
            'All values in level_thresh_cnst_bds must be integers!')

        assert level_thresh_cnst_bds[1] >= level_thresh_cnst_bds[0], (
            'Values in level_thresh_cnst_bds must be ascending!')
        #======================================================================

        # level_thresh_slp_bds.
        assert isinstance(level_thresh_slp_bds, (list, tuple)), (
            'level_thresh_slp_bds not a list or a tuple!')

        assert len(level_thresh_slp_bds) == 2, (
            'level_thresh_slp_bds must have two elements only!')

        assert all([isinstance(level_thresh_slp_bd, float)
                    for level_thresh_slp_bd in level_thresh_slp_bds]), (
            'All values in level_thresh_slp_bds must be floats!')

        assert all([np.isfinite(level_thresh_slp_bd)
                    for level_thresh_slp_bd in level_thresh_slp_bds]), (
            'All values in level_thresh_slp_bds must be finite!')

        assert level_thresh_slp_bds[0] <= level_thresh_slp_bds[1], (
            'Values in level_thresh_slp_bds must be ascending!')
        #======================================================================

        # rand_err_sclr_cnst_bds.
        assert isinstance(rand_err_sclr_cnst_bds, (list, tuple)), (
            'rand_err_sclr_cnst_bds not a list or a tuple!')

        assert len(rand_err_sclr_cnst_bds) == 2, (
            'rand_err_sclr_cnst_bds must have two elements only!')

        assert all([isinstance(rand_err_sclr_cnst_bd, float)
                    for rand_err_sclr_cnst_bd in rand_err_sclr_cnst_bds]), (
            'All values in rand_err_sclr_cnst_bds must be floats!')

        assert all([np.isfinite(rand_err_sclr_cnst_bd)
                    for rand_err_sclr_cnst_bd in rand_err_sclr_cnst_bds]), (
            'All values in rand_err_sclr_cnst_bds must be finite!')

        assert rand_err_sclr_cnst_bds[0] <= rand_err_sclr_cnst_bds[1], (
            'Values in rand_err_sclr_cnst_bds must be ascending!')
        #======================================================================

        # rand_err_sclr_rel_bds.
        assert isinstance(rand_err_sclr_rel_bds, (list, tuple)), (
            'rand_err_sclr_rel_bds not a list or a tuple!')

        assert len(rand_err_sclr_rel_bds) == 2, (
            'rand_err_sclr_rel_bds must have two elements only!')

        assert all([isinstance(rand_err_sclr_rel_bd, float)
                    for rand_err_sclr_rel_bd in rand_err_sclr_rel_bds]), (
            'All values in rand_err_sclr_rel_bds must be floats!')

        assert all([np.isfinite(rand_err_sclr_rel_bd)
                    for rand_err_sclr_rel_bd in rand_err_sclr_rel_bds]), (
            'All values in rand_err_sclr_rel_bds must be finite!')

        assert rand_err_sclr_rel_bds[0] <= rand_err_sclr_rel_bds[1], (
            'Values in rand_err_sclr_rel_bds must be ascending!')
        #======================================================================

        # probs_exp_bds.
        assert isinstance(probs_exp_bds, (list, tuple)), (
            'probs_exp_bds not a list or a tuple!')

        assert len(probs_exp_bds) == 2, (
            'probs_exp_bds must have two elements only!')

        assert all([isinstance(probs_exp_bd, float)
                    for probs_exp_bd in probs_exp_bds]), (
            'All values in probs_exp_bds must be floats!')

        assert all([np.isfinite(probs_exp_bd)
                    for probs_exp_bd in probs_exp_bds]), (
            'All values in probs_exp_bds must be finite!')

        assert probs_exp_bds[0] <= probs_exp_bds[1], (
            'Values in probs_exp_bds must be ascending!')

        assert probs_exp_bds[0] >= 0, (
            'Invalid value of probs_exp lower bounds!')

        assert probs_exp_bds[1] >= probs_exp_bds[0], (
            'Values in probs_exp_bds must be ascending!')
        #======================================================================

        # Set all values.
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

        self._sett_asymm_prob_center_lbd = prob_center_bds[0]
        self._sett_asymm_prob_center_ubd = prob_center_bds[1]

        self._sett_asymm_pre_val_exp_lbd = pre_val_exp_bds[0]
        self._sett_asymm_pre_val_exp_ubd = pre_val_exp_bds[1]

        self._sett_asymm_crt_val_exp_lbd = crt_val_exp_bds[0]
        self._sett_asymm_crt_val_exp_ubd = crt_val_exp_bds[1]

        self._sett_asymm_level_thresh_cnst_lbd = level_thresh_cnst_bds[0]
        self._sett_asymm_level_thresh_cnst_ubd = level_thresh_cnst_bds[1]

        self._sett_asymm_level_thresh_slp_lbd = level_thresh_slp_bds[0]
        self._sett_asymm_level_thresh_slp_ubd = level_thresh_slp_bds[1]

        self._sett_asymm_rand_err_sclr_cnst_lbd = rand_err_sclr_cnst_bds[0]
        self._sett_asymm_rand_err_sclr_cnst_ubd = rand_err_sclr_cnst_bds[1]

        self._sett_asymm_rand_err_sclr_rel_lbd = rand_err_sclr_rel_bds[0]
        self._sett_asymm_rand_err_sclr_rel_ubd = rand_err_sclr_rel_bds[1]

        self._sett_asymm_probs_exp_lbd = probs_exp_bds[0]
        self._sett_asymm_probs_exp_ubd = probs_exp_bds[1]
        #======================================================================

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

            print(
                'Probability center\'s bounds:',
                self._sett_asymm_prob_center_lbd,
                self._sett_asymm_prob_center_ubd)

            print(
                'Previous values exponent\'s bounds:',
                self._sett_asymm_pre_val_exp_lbd,
                self._sett_asymm_pre_val_exp_ubd)

            print(
                'Current values exponent\'s bounds:',
                self._sett_asymm_crt_val_exp_lbd,
                self._sett_asymm_crt_val_exp_ubd)

            print(
                'Level threshold constant\'s bounds:',
                self._sett_asymm_level_thresh_cnst_lbd,
                self._sett_asymm_level_thresh_cnst_ubd)

            print(
                'Level threshold slope\'s bounds:',
                self._sett_asymm_level_thresh_slp_lbd,
                self._sett_asymm_level_thresh_slp_ubd)

            print(
                'Random error scaler constant\'s bounds:',
                self._sett_asymm_rand_err_sclr_cnst_lbd,
                self._sett_asymm_rand_err_sclr_cnst_ubd)

            print(
                'Random error scaler relative\'s bounds:',
                self._sett_asymm_rand_err_sclr_rel_lbd,
                self._sett_asymm_rand_err_sclr_rel_ubd)

            print(
                'Probability exponent\'s bounds:',
                self._sett_asymm_probs_exp_lbd,
                self._sett_asymm_probs_exp_ubd)

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
