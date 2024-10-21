from time import time
import numpy as np
import sklearn
from scipy.linalg import svd
from method.estimator import Estimator
from utils.parameters import Params, get_df_step, get_df_method

SVD_E, MAX_ITER, N_TOL = get_df_method()
N_STEP = get_df_step()


class SPCA_PSP(Estimator):
    def __init__(self, n_comp=None, alpha=1e-3, step=N_STEP, svd_e=SVD_E,
                 max_iter=MAX_ITER, tol=N_TOL, init='svd'):
        super().__init__()
        self.alpha, self.alpha_list = alpha, []
        self.svd_e, self.max_iter, self.tol, self.n_comp, self.step = svd_e, max_iter, tol, n_comp, step
        self.X, self.V, self.U, self.xtx, self.eigen_values, self.obj_list = None, None, None, None, None, None
        self.init, self.m, self.r, self.m_sqrt = init, None, None, None

    def transform(self, data=None):
        assert self.V is not None
        return self._transform()

    def _transform(self):
        return np.dot(self.X, self.V)

    @classmethod
    def init_by_params(cls, params=Params()):
        alpha = params.get_alpha()
        svd_e, max_iter, tol, step = params.get_method_setting()
        n_comp = params.get_n_component()
        init = params.get_init()
        return cls(n_comp=n_comp, alpha=alpha, svd_e=svd_e, step=step,
                   max_iter=max_iter, tol=tol, init=init)

    def update_b(self, U, V):
        for i in range(self.m):
            res_i = U.T @ self.X[:, i]  # projection residual
            res_i_norm = np.linalg.norm(res_i)
            if (self.alpha_list[i] * self.m_sqrt) < 2 * res_i_norm:
                v = (1 - ((self.alpha_list[i] * self.m_sqrt) / (2 * res_i_norm))) * res_i
            else:
                v = np.zeros(self.r)
            V[i, :] = v
        return V

    def f_obj(self, X, U, V):
        f = 0.5 * np.sum(self.compute_residual(X, U, V) ** 2)
        h = 0
        for i in range(self.m):
            h += self.alpha_list[i] * self.m_sqrt * np.linalg.norm(V[i, :])
        return f + h

    @staticmethod
    def compute_residual(x, u, v):
        return x - np.dot(u, v.T)

    def compute(self):
        A, B, Dmax = self.init
        U, V = np.dot(self.X, B), A.copy()
        self.m, self.r = V.shape
        self.m_sqrt = np.sqrt(self.m)
        for i in range(self.m):
            self.alpha_list.append(self.alpha / np.linalg.norm(V[i, :]))
        obj = [self.f_obj(self.X, U, V)]
        Z_U, Z_D, Z_V, t, start = None, None, None, time(), time()
        self.time_list.append(0)
        for n_iter in range(1, self.max_iter + 1):
            # Update U
            start = time()
            Z = np.dot(self.X, V)
            if self.svd_e:
                Z_U, Z_D, Z_V = sklearn.utils.extmath.randomized_svd(Z, n_components=self.n_comp)
            else:
                Z_U, Z_D, Z_V = svd(Z, full_matrices=False)
            U = Z_U.dot(Z_V.T)
            self.time_a_list.append(time() - start)
            # Update V
            t = time()
            V = self.update_b(U, V)
            self.time_b_list.append(time() - t)
            # compute obj func
            obj.append(self.f_obj(self.X, U, V))
            self.time_list.append(time() - start)
            if abs(obj[-2] - obj[-1]) / obj[-1] < self.tol:
                break
        self.eigen_values = Z_D / (self.X.shape[0] - 1)
        self.V, self.U, self.obj_list = V, U, obj

    def get_weight_matrix(self):
        return self.V

    def get_res(self, obj=False):
        if obj:
            return self.obj_list
        else:
            raise ValueError("Check")