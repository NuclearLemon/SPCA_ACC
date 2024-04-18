from time import time
import numpy as np
from utils.parameters import get_df_method, get_df_step, Params
from method.estimator import Estimator
from method.l1manifold import manifold_accelerate

SVD_E, MAX_ITER, N_TOL = get_df_method()
N_STEP = get_df_step()


class SPCA_VP(Estimator):

    def __init__(self, n_comp=None, alpha=1e-3, step=N_STEP, newton=False, svd_e=SVD_E,
                 max_iter=MAX_ITER, tol=N_TOL, init='svd', acc=0, acc_step=N_STEP):
        super().__init__()
        self.alpha = alpha
        self.svd_e, self.max_iter, self.tol, self.n_comp, self.step = svd_e, max_iter, tol, n_comp, step
        self.X, self.B, self.A, self.xtx, self.eigen_values, self.obj_list = None, None, None, None, None, None
        self.acc, self.acc_step, self.newton, self.init, self.pc_list = acc, acc_step, newton, init, []

    @classmethod
    def init_by_params(cls, params=Params()):
        alpha = params.get_alpha()
        svd_e, max_iter, tol, step = params.get_method_setting()
        n_comp = params.get_n_component()
        init = params.get_init()
        newton = params.get_newton()
        acc = params.get_accelerate()
        acc_step = params.get_acc_step()
        return cls(n_comp=n_comp, alpha=alpha, svd_e=svd_e, step=step, newton=newton,
                   max_iter=max_iter, tol=tol, init=init, acc=acc, acc_step=acc_step)

    def get_update_b(self, B, A, newton=False):
        if newton:
            return B - A
        else:
            return np.dot(self.xtx, B - A)

    def compute(self):
        A, B, Dmax = self.init
        self.xtx = np.dot(self.X.T, self.X)
        # Set Parameters
        if self.step is None:
            self.step = 1.0 / (Dmax ** 2)
        if self.acc_step is None:
            self.acc_step = self.step
        obj, pc_list = [self.f_obj(self.X, B, A)], [B]
        Z_U, Z_D, Z_V, t, start = None, None, None, time(), time()
        self.time_list.append(0)
        for n_iter in range(1, self.max_iter + 1):
            # Update A: X'XB = UDV' Compute X'XB via SVD of X
            start = time()
            Z = np.dot(self.xtx, B)
            Z_U, Z_D, Z_V = self.svd_method(Z, if_effect=self.svd_e)
            A = Z_U.dot(Z_V)
            self.time_a_list.append(time() - start)
            # Proximal Gradient Descent to Update B
            t = time()
            G = self.get_update_b(B, A, self.newton)
            B = B - self.step * G
            B = self.soft_l1(B, thresh=self.alpha)
            self.time_b_list.append(time() - t)
            if self.acc:
                B = manifold_accelerate(grad=G, param=B, r_alpha=self.acc_step)
                self.time_acc_list.append(time() - t)
            pc_list.append(B)
            # compute obj func
            obj.append(self.f_obj(self.X, B, A))
            self.time_list.append(time() - start)
            if abs(obj[-2] - obj[-1]) / obj[-1] < self.tol:
                break
        self.eigen_values = Z_D / (self.X.shape[0] - 1)
        self.B, self.A, self.obj_list = B, A, obj
        self.pc_list = pc_list

    def get_pc_list(self):
        return self.pc_list

    def f_obj(self, X, B, A):
        f = 0.5 * np.sum(self.compute_residual(X, B, A) ** 2)
        h = self.alpha * np.sum(np.abs(B))
        return f + h
