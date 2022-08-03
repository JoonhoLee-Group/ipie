import numpy as np

from ipie.estimator_base import EstimatorBase
from ipie.utils.io import get_input_value

class EnergyEstimator(EstimatorBase):

    def __init__(
            self,
            comm=None,
            qmc=None,
            system=None,
            ham=None,
            trial=None,
            verbose=False,
            options={}
            ):

        assert system is not None
        assert ham is not None
        assert trial is not None
