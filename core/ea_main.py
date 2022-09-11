'''
Created on Aug 7, 2022

@author: Faizan3800X-Uni
'''

from gnrctsgenr import (
    GTGBase,
    GTGData,
    GTGPrepareBase,
    GTGPrepareCDFS,
    GTGPrepareUpdate,
    GTGAlgBase,
    GTGAlgObjective,
    GTGAlgIO,
    GTGAlgTemperature,
    GTGAlgMisc,
    GTGAlgorithm,
    GTGSave,
    )

from .aa_setts import IAAFTSASettings

from .ba_prep import (
    IAAFTSAPrepareRltznRef,
    IAAFTSAPrepareRltznSim,
    IAAFTSAPrepareTfms,
    IAAFTSAPrepare,
    )

from .ca_alg import (
    IAAFTSAAlgLagNthWts,
    IAAFTSAAlgLabelWts,
    IAAFTSAAlgAutoObjWts,
    )

from .da_rltzn import IAAFTSARealization


class IAAFTSAMain(
        GTGBase,
        GTGData,
        IAAFTSASettings,
        GTGPrepareBase,
        IAAFTSAPrepareTfms,
        GTGPrepareCDFS,
        GTGPrepareUpdate,
        IAAFTSAPrepare,
        GTGAlgBase,
        GTGAlgObjective,
        GTGAlgIO,
        IAAFTSAAlgLagNthWts,
        IAAFTSAAlgLabelWts,
        IAAFTSAAlgAutoObjWts,
        IAAFTSARealization,
        GTGAlgTemperature,
        GTGAlgMisc,
        GTGAlgorithm,
        GTGSave):

    def __init__(self, verbose):

        GTGBase.__init__(self, verbose)
        GTGData.__init__(self)
        IAAFTSASettings.__init__(self)

        self._rr = IAAFTSAPrepareRltznRef()  # Reference.
        self._rs = IAAFTSAPrepareRltznSim()  # Simulation.

        GTGPrepareBase.__init__(self)
        IAAFTSAPrepareTfms.__init__(self)
        GTGPrepareCDFS.__init__(self)
        GTGPrepareUpdate.__init__(self)
        IAAFTSAPrepare.__init__(self)
        GTGAlgBase.__init__(self)
        GTGAlgObjective.__init__(self)
        GTGAlgIO.__init__(self)
        IAAFTSAAlgLagNthWts.__init__(self)
        IAAFTSAAlgLabelWts.__init__(self)
        IAAFTSAAlgAutoObjWts.__init__(self)
        IAAFTSARealization.__init__(self)
        GTGAlgTemperature.__init__(self)
        GTGAlgMisc.__init__(self)
        GTGAlgorithm.__init__(self)
        GTGSave.__init__(self)

        self._main_verify_flag = False
        return

    def _write_ref_rltzn_extra(self, *args):

        _ = args

        return

    def _write_sim_rltzn_extra(self, *args):

        _ = args

        # h5_hdl = args[0]
        #
        # main_sim_grp_lab = 'data_sim_rltzns'
        #
        # sim_grp_lab = self._rs.label
        #
        # sim_grp_main = h5_hdl[main_sim_grp_lab]
        #
        # sim_grp = sim_grp_main[sim_grp_lab]

        return

    def verify(self):

        GTGData._GTGData__verify(self)

        IAAFTSASettings._IAAFTSASettings__verify(self)

        IAAFTSAPrepare._IAAFTSAPrepare__verify(self)
        GTGAlgorithm._GTGAlgorithm__verify(self)
        GTGSave._GTGSave__verify(self)

        assert self._save_verify_flag, 'Save in an unverified state!'

        self._main_verify_flag = True
        return
