P_ALPHA = "alpha"
D_ALPHA = 1e-5
SVD_E, N_MAX_ITER, N_TOL, N_STEP = False, 100, 1e-5, 3e-6


def get_df_method():
    return SVD_E, N_MAX_ITER, N_TOL


def get_df_step():
    return N_STEP


class Params:
    def __init__(self, n_comp=None, alpha=D_ALPHA, newton=False, svd_e=SVD_E, max_iter=N_MAX_ITER, tol=N_TOL,
                 step=N_STEP, acc=0, acc_step=N_STEP, init=None):
        self.alpha = alpha
        self.n_comp, self.svd_e, self.max_iter, self.tol = n_comp, svd_e, max_iter, tol
        self.init, self.newton, self.step, self.acc, self.acc_step = init, newton, step, acc, acc_step

    def get_alpha(self):
        return self.alpha

    def get_method_setting(self):
        res = [self.svd_e, self.max_iter, self.tol, self.step]
        return res

    def get_accelerate(self):
        return self.acc

    def get_n_component(self):
        return self.n_comp

    def get_step(self):
        return self.step

    def get_init(self):
        return self.init

    def get_newton(self):
        return self.newton

    def get_acc_step(self):
        return self.acc_step

    def set_max_iter(self, max_iter):
        self.max_iter = max_iter

    def set_accelerate(self, acc):
        self.acc = acc

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_n_comp(self, n_comp):
        self.n_comp = n_comp

    def set_step(self, step):
        self.step = step

    def set_svd_e(self, svd_e):
        self.svd_e = svd_e

    def set_init(self, init_list):
        self.init = init_list

    def set_newton(self, newton):
        self.newton = newton

    def set_acc_step(self, step):
        self.acc_step = step

    def set_tol(self, tol):
        self.tol = tol

    def copy(self):
        return Params(n_comp=self.n_comp, alpha=self.alpha, newton=self.newton, svd_e=self.svd_e,
                      max_iter=self.max_iter, tol=self.tol, step=self.step,
                      acc=self.acc, acc_step=self.acc, init=self.init)
