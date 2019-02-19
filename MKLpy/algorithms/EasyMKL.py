# -*- coding: latin-1 -*-
"""
@author: Michele Donini
@email: mdonini@math.unipd.it
 
EasyMKL: a scalable multiple kernel learning algorithm
by Fabio Aiolli and Michele Donini
 
Paper @ http://www.math.unipd.it/~mdonini/publications.html
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from .base import MKL
from ..multiclass import OneVsOneMKLClassifier as ovoMKL, OneVsRestMKLClassifier as ovaMKL
from ..utils.exceptions import BinaryProblemError
from .komd import KOMD
from ..lists import HPK_generator
 
from cvxopt import matrix, spdiag, solvers
import numpy as np
 
from MKLpy.arrange import summation
 
 
class EasyMKL(BaseEstimator, ClassifierMixin, MKL):
    ''' EasyMKL is a Multiple Kernel Learning algorithm.
        The parameter lam (lambda) has to be validated from 0 to 1.
 
        For more information:
        EasyMKL: a scalable multiple kernel learning algorithm
            by Fabio Aiolli and Michele Donini
 
        Paper @ http://www.math.unipd.it/~mdonini/publications.html
    '''
    def __init__(self, estimator=KOMD(lam=0.1), lam=0.1, generator=HPK_generator(n=10), multiclass_strategy='ova', max_iter=100, verbose=False):
        super(self.__class__, self).__init__(estimator=estimator, generator=generator, multiclass_strategy=multiclass_strategy, how_to=summation, max_iter=max_iter, verbose=verbose)
        self.lam = lam

        
    def _arrange_kernel(self):
        Y = [
            2 * int(y == self.classes_[1]) -1
            for y in self.Y
        ]
        n_sample = len(self.Y)
        ker_matrix = matrix(
            summation(self.KL)
        )
        YY = spdiag(Y)
         
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = self.max_iter
        sol = solvers.qp(
            2 * (
                (1.0 - self.lam) * YY * ker_matrix * YY 
                +
                spdiag([self.lam] * n_sample)
            ),
            matrix([0.0] * n_sample),
            -spdiag([1.0] * n_sample),
            matrix(
                [0.0] * n_sample,
                (n_sample, 1)
            ),
            matrix(
                [
                    [float(y == +1) for y in Y],
                    [float(y == -1) for y in Y]
                ]
            ).T,
            matrix(
                [[1.0] * 2],
                (2, 1)
            )
        )
        if self.verbose:
            print ('[EasyMKL]')
            print ('optimization finished, #iter = %d' % sol['iterations'])
            print ('status of the solution: %s' % sol['status'])
            print ('objval: %.5f' % sol['primal objective'])

        yg = sol['x'].T * YY
        weights = [
            (yg * matrix(K) * yg.T)[0]
            for K in self.KL
        ]
        norm2 = sum(weights)
        self.weights = np.array([w / norm2 for w in weights])

        self.ker_matrix = summation(self.KL, self.weights)
        return self.ker_matrix

 
    def get_params(self, deep=True):
        # this estimator has parameters:
        return {
            "lam": self.lam,
            "generator": self.generator, "max_iter":self.max_iter,
            "verbose":self.verbose, "multiclass_strategy":self.multiclass_strategy,
            'estimator':self.estimator
        }
