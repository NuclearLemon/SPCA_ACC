from time import time
import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from method.spca_al import SPCA_AL
from method.spca_psp import SPCA_PSP
from utils.parameters import Params
from method.spca_vp import SPCA_VP

S_SPCA_AL, S_SPCA_PSP, S_SPCA_VP, S_SPCA_N, S_SPCA_ACC = 'SPCA AL', 'SPCA PSP', 'SPCA VP', 'SPCA Newton', 'SPCA ACC'

method_dict = {
    S_SPCA_AL: SPCA_AL,
    S_SPCA_PSP: SPCA_PSP,
    S_SPCA_VP: SPCA_VP,
    S_SPCA_N: SPCA_VP,
    S_SPCA_ACC: SPCA_VP,
}


def get_method_name_list():
    return [S_SPCA_AL, S_SPCA_PSP, S_SPCA_VP, S_SPCA_N, S_SPCA_ACC]


def get_classify_metrics(y_true, y_pred):
    """return cluster metrics"""
    metric_ac = accuracy_score(y_true, y_pred)
    return [metric_ac]


def init_ab(data, n_comp, method='svd', seed=1, svd_e=False):
    print(f'Generating initialization matrix, seed: {seed}, svd efficient: {svd_e}')
    start = time()
    if method not in ['svd', 'rand']:
        raise ValueError('Error Initialization Method')
    if svd_e:
        _, D, Vt = sklearn.utils.extmath.randomized_svd(data, n_components=n_comp, random_state=seed)
    else:
        _, D, Vt = scipy.linalg.svd(data, full_matrices=False, overwrite_a=False)
    A, B = Vt[:n_comp].T, Vt[:n_comp].T
    Dmax = D[0]  # l2 norm
    if method == 'rand':
        rng = np.random.RandomState(seed=seed)
        A, B = rng.random_sample(A.shape), rng.random_sample(B.shape)
    print(f'Init time cost: {time() - start:.2f}s')
    return A, B, Dmax


def classify_proc(data, d_label, p_method, params=Params(), repeat=1):
    """main process of classify"""
    data_trans, obj, time_cost, t, ta, tb, tacc = pre_process_data(data, p_method, params)
    if repeat > 1:
        t_list, ta_list, tb_list, tacc_list = [t], [ta], [tb], [tacc]
        for i in range(repeat - 1):
            data_trans, obj, time_cost, ti, tai, tbi, tacci = pre_process_data(data, p_method, params)
            t_list.append(ti)
            ta_list.append(tai)
            tb_list.append(tbi)
            tacc_list.append(tacci)
        t, ta, tb = np.mean(t_list, axis=0), np.mean(ta_list, axis=0), np.mean(tb_list, axis=0)
        if tacc:
            tacc = np.mean(tacc_list, axis=0)
    classifier = LogisticRegression(max_iter=200)
    data_train, data_test, label_train, label_test = \
        train_test_split(data_trans, d_label, test_size=0.3, random_state=42)
    classifier.fit(data_train, label_train)
    labels_predict = classifier.predict(data_test)
    metrics = get_classify_metrics(label_test, labels_predict)
    return metrics, obj, time_cost, t, ta, tb, tacc


def get_pipe(method, m_name, params=Params()):
    """get total pipeline of process"""
    est = method.init_by_params(params=params)
    est_pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        (m_name, est)]
    )
    return est_pipe


def pre_process_data(data, p_method, params):
    start = time()
    pipe = get_pipe(method=method_dict[p_method], m_name=p_method, params=params)
    data_trans = pipe.fit_transform(data, None)
    time_cost = time() - start
    obj = pipe[-1].get_res(obj=True)
    t, ta, tb, tacc = pipe[-1].get_time_list(detail=True)
    return data_trans, obj, time_cost, t, ta, tb, tacc


def get_method_dict(data, n_comp, init='rand', alpha=1e-3, max_iter=100, tol=1e-4, svd_e=False, seed=1):
    """
    Generate a dictionary of algorithm parameters for the specified settings.

    Args:
        n_comp (int): Number of components to be computed.
        init (str, optional): Initialization method. Defaults to 'rand'.
        alpha (float, optional): Sparse coefficient. Defaults to 1e-3
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-4.
        svd_e (bool, optional): Whether to use effective SVD (for large dataset) in initialization. Defaults to False.

    Returns:
        dict: A dictionary containing the parameter settings for each algorithm.
    """
    step1, step2 = None, 0.5
    """
    If step1 is set to None, the algorithm will automatically determine the step size during runtime based on the eigenvalues of the data matrix.
    When manually specifying step1, it is important to avoid using large step sizes to prevent the SPCA_VP algorithm from diverging. 
    As discussed in our paper, values between 0.1 and 0.5 are generally suitable for step2.
    """
    print(f"Parameter setting: max iter({max_iter}), tol({tol}), svd_e({svd_e}), step({step1}, {step2}), init({init})")
    ab_init = init_ab(data=data, n_comp=n_comp, method=init, seed=seed, svd_e=svd_e)
    p_dict = {
        S_SPCA_AL: Params(alpha=alpha, max_iter=max_iter, init=ab_init, tol=tol, svd_e=svd_e),
        S_SPCA_PSP: Params(alpha=alpha, max_iter=max_iter, init=ab_init, tol=tol, svd_e=svd_e),
        S_SPCA_VP: Params(alpha=alpha, max_iter=max_iter, step=step1, init=ab_init,
                          tol=tol, svd_e=svd_e, newton=False),
        S_SPCA_N: Params(alpha=alpha, max_iter=max_iter, step=step2, newton=True,
                         init=ab_init, tol=tol, svd_e=svd_e),
        S_SPCA_ACC: Params(alpha=alpha, max_iter=max_iter, step=step2, acc_step=step2, newton=True,
                           acc=True, init=ab_init, tol=tol, svd_e=svd_e),
    }
    return p_dict
