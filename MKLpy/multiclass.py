# -*- coding: utf-8 -*-
"""
.. codeauthor:: Michele Donini <>
.. codeauthor:: Ivano Lauriola <ivanolauriola@gmail.com>



"""

import numpy as np
import sys
from cvxopt import matrix
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn import svm
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import _num_samples
import scipy.sparse as sp
import array
#from utils import HPK_generator
from sklearn.metrics import roc_auc_score, accuracy_score

class OneVsOneMKLClassifier():
    
    def __init__(self, clf1, verbose=False):
        #print 'init ovo'
        self.clf1 = clf1
        self.clf2 = clf1.estimator
        self.verbose = verbose
        
    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter,value)
        return self
    
    def fit(self,K_tr,Y_tr):
        self.K_tr = K_tr
        id_for_train = []
        for l in set(Y_tr):
            id_for_train += [idx for idx,lab in enumerate(Y_tr) if lab==l]
        #id_for_train = [i for l in Y_tr]
        
        #ordering che devo ancora capì
        K_tr = np.array([kk[id_for_train][:,id_for_train] for kk in K_tr])
        
        n_classes = len(np.unique(Y_tr))
        self.classes_ = np.unique(Y_tr)
        self.n_classes = n_classes
        #Ove vs One
        list_of_dichos = []
        for i in range(n_classes):
            for j in range(i+1,n_classes):
                #list_of_dichos.append(((i,),(j,)))
                list_of_dichos.append(((int(self.classes_[i]),),(int(self.classes_[j]),)))
        
        list_of_indices = {}
        list_of_indices_train = {}
        list_of_labels = {}
        list_of_labels_train = {}
        #prendo gli esempi per ogni dicotomia
        for dicho in list_of_dichos:
            list_of_indices[dicho] = [[i for i,l in enumerate(Y_tr) if l in dicho[0]],
                                      [i for i,l in enumerate(Y_tr) if l in dicho[1]]]
            list_of_indices_train[dicho] = [[i for i,l in enumerate(id_for_train) if Y_tr[l] in dicho[0]],
                                            [i for i,l in enumerate(id_for_train) if Y_tr[l] in dicho[1]]]
            
            list_of_labels[dicho] = [1.0]*len(list_of_indices[dicho][0]) + [-1.0]*len(list_of_indices[dicho][1])
            list_of_labels_train[dicho] = [1.0]*len(list_of_indices_train[dicho][0]) + [-1.0]*len(list_of_indices_train[dicho][1])
            
        if self.verbose:
            print ('Learning the models for %d dichotomies' % len(list_of_dichos))
        # LEARNING THE MODELS
        wmodels = {}
        combinations = {}
        functional_form = self.clf1.how_to
        


        for dicho in list_of_dichos:
            ind = list_of_indices_train[dicho][0] + list_of_indices_train[dicho][1]
            cc = self.clf1.__class__(**self.clf1.get_params())
            cc.kernel = 'precomputed'
            #cc = cc.fit(np.array([kk[ind][:,ind]  for kk in K_tr]),
            ker_matrix = cc.arrange_kernel(np.array([kk[ind][:,ind]  for kk in K_tr]),
                       np.array(list_of_labels_train[dicho]))
            wmodels[dicho] = [w / sum(cc.weights) for w in cc.weights]
            combinations[dicho] = ker_matrix#cc.ker_matrix

            del cc
        
        self.ker_matrices = combinations #mi serve solo per alcuni test
        # Train SVM
        if self.verbose:
            print ('SVM training phase...')
        
        svcs = {}
        for dicho in list_of_dichos:
            svcs[dicho] = self.clf2.__class__(**self.clf2.get_params())
            svcs[dicho].kernel = 'precomputed'
            idx = list_of_indices[dicho][0]+list_of_indices[dicho][1]
            k = combinations[dicho]#[idx,:][:,idx]
            svcs[dicho].fit(np.array(k), np.array(list_of_labels[dicho]))
            #print svcs[dicho].n_support_
            #raw_input('b')
            #if self.verbose:
            #    sys.stdout.flush()
                
        #salvo gli oggetti che mi torneranno utili
        self.list_of_dichos = list_of_dichos
        self.svcs = svcs
        self.id_for_train = id_for_train
        self.list_of_indices = list_of_indices
        self.wmodels = wmodels
        self.weights = wmodels
        self.functional_form = functional_form
        return self



    def predict(self,K_te):#, Y_te):
        predicts = {}
        wmodels = self.wmodels
        single_accuracy = {}

        for dicho in self.list_of_dichos:
            idx = self.list_of_indices[dicho][0]+self.list_of_indices[dicho][1]
            w = self.wmodels[dicho]
            k = self.functional_form(np.array([kk[:,idx] for kk in K_te]),w)
            predicts[dicho] = self.svcs[dicho].decision_function(k)

        # Voting   
        #nn = len(Y_te)
        nn = len(K_te[0])
        #points = np.zeros((nn,self.n_classes),dtype=int)
        points = np.zeros((nn,int(np.max(self.classes_))+1),dtype=int)
        #print points.shape
        for dicho in self.list_of_dichos:
            for ir,r in enumerate(predicts[dicho]):
                if r > 0:
                    points[ir,dicho[0][0]] += 1
                else:
                    points[ir,dicho[1][0]] += 1

        y_pred = np.argmax(points,1)
        sys.stdout.flush()
        return y_pred


class OneVsRestMKLClassifier():
    def __init__(self, clf1, verbose=False):
        self.clf1 = clf1
        self.clf2 = clf1.estimator
        self.verbose = verbose

        self.is_fitted = False
        self.classes_ = None

    def fit(self, K_list, Y):
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(Y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        self.n_classes = len(self.classes_)
        self.weights = {}
        self.clfs = {}
        self.ker_matrices = {}

        # learning the models
        for cls_, column in zip(self.classes_, columns):
            # print 'learning model with ',model,' is the positive class'
            # learning the kernel
            cc1 = self.clf1.__class__(**self.clf1.get_params())
            cc1.kernel = 'precomputed'
            ker_matrix = cc1.arrange_kernel(K_list, column)
            self.weights.update({cls_: cc1.weights})

            # fitting the model
            cc2 = self.clf2.__class__(**self.clf2.get_params())
            cc2.kernel = 'precomputed'
            cc2.fit(ker_matrix, column)
            self.clfs.update({cls_: cc2})
            self.ker_matrices.update({cls_:ker_matrix})
        
        print(self.weights)

        self.functional_form = self.clf1.how_to
        self.is_fitted = True
        return self

    def predict(self, K_list):
        if not self.is_fitted:
            raise Exception('Not fitted')
        # predict with binary models
        n_samples = _num_samples(K_list[0])
        if self.label_binarizer_.y_type_ == "multiclass":
            predicts = {
                cls_:
                clf.decision_function(
                    self.functional_form(
                        K_list,
                        self.weights[cls_]
                    )
                )
                for cls_, clf in self.clfs
            }
            scoring = np.zeros((n_samples, self.n_classes))
            for col, cls_ in enumerate(self.classes_):
                scoring[:, col] = predicts[cls_]
            return np.array([self.classes_[np.argmax(sc)] for sc in scoring])
        else:
            indices = array.array('i')
            indptr = array.array('i', [0])
            for cls_, clf in self.clfs.items():
                indices.extend(
                    np.where(
                        np.ravel(
                            clf.decision_function(
                                self.functional_form(
                                    K_list,
                                    self.weights[cls_]
                                )
                            )
                        ) > 0
                    )[0]
                )
                indptr.append(len(indices))
            data = np.ones(len(indices), dtype=int)
            indicator = sp.csc_matrix(
                (data, indices, indptr),
                shape=(n_samples, len(self.clfs))
            )
            return self.label_binarizer_.inverse_transform(indicator)


    def predict_proba(self, K_list):
        Y = np.array(
            [
                clf.predict_proba(
                    self.functional_form(
                        K_list,
                        self.weights[cls_]
                    )
                )[:, 1]
                for cls_, clf in self.clfs.items()
            ]
        ).T
        if len(self.clfs) == 1:
            Y = np.concatenate(((1 - Y), Y), axis=1)
        
        if not self.label_binarizer_.y_type_.startswith('multilabel'):
            # Then, probabilities should be normalized to 1.
            Y /= np.sum(Y, axis=1)[:, np.newaxis]
        return Y
