import numpy as np

class EstimatorBase(object):

    @property
    @abstractmethod
    def get_shape(self):
        return self.shape

    @property
    @abstractmethod
    def get_group_name(self):
        return self.group_name

    @property
    @abstractmethod
    def ascii_filename(self):
        self.ascii_filename

    @property
    @abstractmethod
    def get_estimator_names(self):
        return self.shape

    @abstractmethod
    def get_data(self):
        return self.data

    @abstractmethod
    def compute_estimator(self, walker_batch, trial_wavefunction):
        pass
