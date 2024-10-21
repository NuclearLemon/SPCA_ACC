from sklearn.linear_model import ElasticNet
from time import time
import numpy as np
from utils.parameters import get_df_method, get_df_step, Params
from method.estimator import Estimator

SVD_E, MAX_ITER, N_TOL = get_df_method()
N_STEP = get_df_step()


class SPCA_AL(Estimator):
    def __init__(self, n_comp=None, alpha=1e-3, step=N_STEP, svd_e=SVD_E,
                 max_iter=MAX_ITER, tol=N_TOL, init='svd'):
        super().__init__()
        self.alpha = alpha
        self.svd_e, self.max_iter, self.tol, self.n_comp, self.step = svd_e, max_iter, tol, n_comp, step
        self.X, self.B, self.A, self.xtx, self.eigen_values, self.obj_list = None, None, None, None, None, None
        self.init, self.pc_list = init, []

    @classmethod
    def init_by_params(cls, params=Params()):
        alpha = params.get_alpha()
        svd_e, max_iter, tol, step = params.get_method_setting()
        n_comp = params.get_n_component()
        init = params.get_init()
        return cls(n_comp=n_comp, alpha=alpha, svd_e=svd_e, step=step,
                   max_iter=max_iter, tol=tol, init=init)

    def update_b(self, A):
        B = A.copy()
        for i in range(B.shape[1]):
            model = ElasticNet(alpha=self.alpha, l1_ratio=1, fit_intercept=False, max_iter=100)
            model.fit(self.X, self.X @ A[:, i])
            B[:, i] = model.coef_
        return B

    def compute(self):
        A, B, Dmax = self.init
        self.xtx = np.dot(self.X.T, self.X)
        obj, pc_list = [self.f_obj(self.X, B, A)], [B]
        Z_U, Z_D, Z_V = None, None, None
        self.time_list.append(0)
        for n_iter in range(1, self.max_iter + 1):
            # Update A
            start = time()
            Z = np.dot(self.xtx, B)
            Z_U, Z_D, Z_V = self.svd_method(Z, if_effect=self.svd_e)
            A = Z_U.dot(Z_V)
            self.time_a_list.append(time() - start)
            # Update B
            t = time()
            B = self.update_b(A)
            pc_list.append(B)
            self.time_b_list.append(time() - t)
            # compute obj func
            obj.append(self.f_obj(self.X, B, A))
            self.time_list.append(time() - start)
            if abs(obj[-2] - obj[-1]) / obj[-1] < self.tol:
                break
        self.eigen_values = Z_D / (self.X.shape[0] - 1)
        self.B, self.A, self.obj_list = B / np.linalg.norm(B, axis=0), A, obj
        self.pc_list = pc_list

    def get_pc_list(self):
        return self.pc_list
