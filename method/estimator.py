import numpy as np
import sklearn
from scipy.linalg import svd
from sklearn.base import BaseEstimator


class Estimator(BaseEstimator):
    def __init__(self):
        self.n_comp = None
        self.X, self.B, self.A, self.eigen_values, self.obj_list = None, None, None, None, None
        self.alpha = 1e-3
        self.xtx = None
        self.time_list, self.time_a_list, self.time_b_list, self.time_acc_list = [], [], [], []

    def _transform(self):
        return np.dot(self.X, self.B)

    def fit(self, data, label=None):
        # X: (m * n)
        self.X = data
        self.compute()
        return self

    def fit_transform(self, data, label=None):
        self.fit(data)
        return self._transform()

    def transform(self, data=None):
        assert self.B is not None
        return self._transform()

    def compute(self):
        pass

    def get_res(self, obj=False):
        if obj:
            return self.obj_list
        else:
            return [self.B, self.A, self.eigen_values, self.obj_list]

    def f_obj(self, X, B, A):
        f = 0.5 * np.sum(self.compute_residual(X, B, A) ** 2)
        h = self.h_obj(B)
        return f + h

    def h_obj(self, B):
        return self.alpha * np.sum(np.abs(B))

    @staticmethod
    def soft_l1(arr, thresh):
        return np.sign(arr) * np.maximum(np.abs(arr) - thresh, 0)

    @staticmethod
    def svd_method(mat, r=None, if_effect=False):
        if if_effect:
            if r is None:
                r = mat.shape[1]
            return sklearn.utils.extmath.randomized_svd(mat, n_components=r)
        else:
            return svd(mat, full_matrices=False)

    @staticmethod
    def compute_residual(x, b, a):
        return x - x.dot(b).dot(a.T)

    def get_time_list(self, detail=True):
        if detail:
            return self.time_list, self.time_a_list, self.time_b_list, self.time_acc_list
        else:
            return self.time_list
