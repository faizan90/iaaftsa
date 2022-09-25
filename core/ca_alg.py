'''
Created on Aug 7, 2022

@author: Faizan3800X-Uni
'''

from multiprocessing import Manager, Lock
from pathos.multiprocessing import ProcessPool

import numpy as np

from gnrctsgenr import (
    GTGBase,
    GTGAlgLagNthWts,
    GTGAlgLabelWts,
    GTGAlgAutoObjWts,
    ret_mp_idxs)


class IAAFTSAAlgLagNthWts(GTGAlgLagNthWts):

    def __init__(self):

        GTGAlgLagNthWts.__init__(self)
        return

    @GTGBase._timer_wrap
    def _set_lag_nth_wts(self, opt_vars_cls):

        self._init_lag_nth_wts()

        self._alg_wts_lag_nth_search_flag = True

        for _ in range(self._sett_wts_lags_nths_n_iters):
            opt_vars_cls = self._get_next_iter_vars(1.0, opt_vars_cls)

            self._update_sim(opt_vars_cls, False)

            self._get_obj_ftn_val()

        self._alg_wts_lag_nth_search_flag = False

        self._update_lag_nth_wts()
        return


class IAAFTSAAlgLabelWts(GTGAlgLabelWts):

    def __init__(self):

        GTGAlgLabelWts.__init__(self)
        return

    @GTGBase._timer_wrap
    def _set_label_wts(self, opt_vars_cls):

        self._init_label_wts()

        self._alg_wts_label_search_flag = True

        for _ in range(self._sett_wts_label_n_iters):
            opt_vars_cls = self._get_next_iter_vars(1.0, opt_vars_cls)

            self._update_sim(opt_vars_cls, False)

            self._get_obj_ftn_val()

        self._alg_wts_label_search_flag = False

        self._update_label_wts()
        return


class IAAFTSAAlgAutoObjWts(GTGAlgAutoObjWts):

    def __init__(self):

        GTGAlgAutoObjWts.__init__(self)
        return

    def _set_auto_obj_wts_single(self, args):

        beg_iter, end_iter, opt_vars_cls = args

        for _ in range(beg_iter, end_iter):
            opt_vars_cls = self._get_next_iter_vars(1.0, opt_vars_cls)

            self._update_sim(opt_vars_cls, False)

            self._get_obj_ftn_val()

        return self._alg_wts_obj_raw

    @GTGBase._timer_wrap
    def _set_auto_obj_wts(self, opt_vars_cls):

        self._sett_wts_obj_wts = None
        self._alg_wts_obj_raw = []
        self._alg_wts_obj_search_flag = True
        #======================================================================

        n_cpus = min(self._sett_wts_obj_n_iters, self._sett_misc_n_cpus)

        if n_cpus > 1:

            mp_idxs = ret_mp_idxs(self._sett_wts_obj_n_iters, n_cpus)

            obj_wts_gen = (
                (
                mp_idxs[i],
                mp_idxs[i + 1],
                opt_vars_cls,
                )
                for i in range(mp_idxs.size - 1))

            self._lock = Manager().Lock()

            mp_pool = ProcessPool(n_cpus)
            mp_pool.restart(True)

            ress = list(mp_pool.uimap(
                self._set_auto_obj_wts_single, obj_wts_gen, chunksize=1))

            # Needs to be initialized again here.
            self._alg_wts_obj_raw = []

            for res in ress:
                self._alg_wts_obj_raw.extend(res)

            mp_pool.close()
            mp_pool.join()

            self._lock = None

            mp_pool = None

        else:
            self._lock = Lock()

            self._set_auto_obj_wts_single(
                (0, self._sett_wts_obj_n_iters, opt_vars_cls))

            self._lock = None
        #======================================================================

        self._alg_wts_obj_raw = np.array(
            self._alg_wts_obj_raw, dtype=np.float64)

        assert self._alg_wts_obj_raw.ndim == 2
        assert self._alg_wts_obj_raw.shape[0] > 1

        self._update_obj_wts()

        self._alg_wts_obj_raw = None
        self._alg_wts_obj_search_flag = False
        return
