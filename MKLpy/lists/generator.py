# -*- coding: latin-1 -*-
import random
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

class Generator(object):
    p = 1

    def __init__(self, n=10, base=linear_kernel):
        self.n = n
        self.base = base

    def _next(self, X, T=None):
        raise NotImplementedError('not implemented yet')

    def make_a_list(self, X, T=None):
        self.L = self.base(X,T) if self.base else None
        T = X if T is None else T
        return [self._next(X,T) for i in range(self.n)]


class HPK_generator(Generator):

    def __init__(self, n=10):
        super(self.__class__, self).__init__(n=n)

    def _next(self, X, T):
        p = self.p
        self.p += 1
        return self.L ** p
