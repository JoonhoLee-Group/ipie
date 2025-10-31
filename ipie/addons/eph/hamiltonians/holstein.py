# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy


class HolsteinModel:
    r"""Class carrying parameters specifying a 1D Holstein chain.

    The Holstein model is described by the Hamiltonian

    .. math::
        \hat{H} = -t \sum_{\langle ij\rangle} \hat{a}_i^\dagger \hat{a}_j
        - g \sqrt{2 w_0 m} \sum_i \hat{a}_i^\dagger \hat{a}_i \hat{X}_i
        + \bigg(\sum_i \frac{m w_0^2}{2} \hat{X}_i^2 + \frac{1}{2m} \hat{P}_i^2
        - \frac{w_0}{2}\bigg),

    where :math:`t` is associated with the electronic hopping, :math:`g` with
    the electron-phonon coupling strength, and :math:``w_0` with the phonon
    frequency.

    Parameters
    ----------
    g : :class:`float`
        Electron-phonon coupling strength
    t : :class:`float`
        Electron hopping parameter
    w0 : :class:`float`
        Phonon frequency
    nsites : :class:`int`
        Length of the 1D Holstein chain
    pbc : :class:``bool`
        Boolean specifying whether periodic boundary conditions should be
        employed.
    """

    def __init__(self, g: float, t: float, w0: float, nsites: int, pbc: bool):
        self.g = g
        self.t = t
        self.w0 = w0
        self.m = 1 / self.w0
        self.nsites = nsites
        self.pbc = pbc
        self.T = None
        self.const = -self.g * numpy.sqrt(2.0 * self.m * self.w0)

    def build(self) -> None:
        """Constructs electronic hopping matrix."""
        self.T = numpy.diag(numpy.ones(self.nsites - 1), 1)
        self.T += numpy.diag(numpy.ones(self.nsites - 1), -1)

        if self.pbc:
            self.T[0, -1] = self.T[-1, 0] = 1.0

        self.T *= -self.t

        self.T = [self.T.copy(), self.T.copy()]

        self.g_tensor = -self.g * numpy.eye(self.nsites)
