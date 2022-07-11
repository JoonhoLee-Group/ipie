import cmath
import copy
import math

import numpy
import scipy.linalg


class HarmonicOscillator(object):
    def __init__(self, m, w, order, shift):
        self.m = m
        self.w = w
        self.order = order
        # self.norm = (self.w / math.pi) ** 0.25 # not necessary but we just include...
        self.norm = 1.0  # not necessary but we just include...
        self.xavg = shift
        # self.eshift = self.xavg**2 * self.w**2 / 2.0

    # -------------------------
    def update_shift(self, shift):  # X : lattice configuration
        self.xavg = shift.copy()

    # -------------------------
    def value(self, X):  # X : lattice configuration
        result = numpy.prod(
            self.norm
            * numpy.exp(-(self.m * self.w / 2.0) * (X - self.xavg) * (X - self.xavg))
        )
        return result

    # -------------------------
    def gradient(self, X):  # grad / value
        # grad = (-self.w * (X-self.xavg)) * self.value(X)
        grad = -self.m * self.w * (X - self.xavg)
        return grad

    # -------------------------
    def laplacian(self, X):  # laplacian / value
        # lap = self.w * self.w * (X-self.xavg) * (X-self.xavg) * self.value(X) - self.w * self.value(X)
        lap = (
            self.m * self.m * self.w * self.w * (X - self.xavg) * (X - self.xavg)
            - self.w * self.m
        )
        return lap

    # -------------------------
    def local_energy(self, X):

        nsites = X.shape[0]

        ke = -0.5 * numpy.sum(self.laplacian(X)) / self.m
        pot = 0.5 * self.m * self.w * self.w * numpy.sum(X * X)

        eloc = ke + pot - 0.5 * self.w * nsites  # No zero-point energy

        return eloc


class HarmonicOscillatorMomentum(object):
    def __init__(self, m, w, order, shift):
        self.m = m
        self.w = w
        self.order = order
        self.norm = 1.0  # not necessary but we just include...
        self.pavg = shift

    # -------------------------
    def value(self, P):  # P : lattice momentum
        result = numpy.prod(
            self.norm
            * numpy.exp(
                -(1.0 / (2.0 * self.m * self.w)) * (P - self.pavg) * (P - self.pavg)
            )
        )
        return result

    # -------------------------
    def gradient(self, P):  # grad / value
        grad = -(1.0 / (self.m * self.w)) * (P - self.pavg)
        return grad

    # -------------------------
    def laplacian(self, P):  # laplacian / value
        lap = (1.0 / (self.m * self.w)) ** 2 * (P - self.pavg) * (P - self.pavg) - (
            1.0 / (self.m * self.w)
        )
        return lap

    # -------------------------
    def local_energy(self, P):

        nsites = P.shape[0]

        ke = (1.0 / (2.0 * self.m)) * numpy.sum(P * P)
        pot = -0.5 * self.m * self.w * self.w * numpy.sum(self.laplacian(P))
        eloc = ke + pot - 0.5 * self.w * nsites  # No zero-point energy

        return eloc
