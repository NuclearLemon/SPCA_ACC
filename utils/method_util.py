from time import time
import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils.parameters import Params
from method.spca_vp import SPCA_VP

M_SPCA_VP = 'SPCA VP'
S_SPCA_VP, S_SPCA_N, S_SPCA_ACC = M_SPCA_VP, 'SPCA Newton', 'SPCA ACC'


def get_method_name_list():
    return [S_SPCA_VP, S_SPCA_N, S_SPCA_ACC]


def get_classify_metrics(y_true, y_pred):
    """return cluster metrics"""
    metric_ac = accuracy_score(y_true, y_pred)
    return [metric_ac]


def get_classify_metrics_columns():
    return ['AC']


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
    print(f'Initialized (Time cost: {time() - start})s')
    return A, B, Dmax


def classify_proc(data, d_label, p_method, params=Params()):
    """main process of classify"""
    data_trans, obj, time_cost, t, ta, tb, tacc = pre_process_data(data, p_method, params)
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
    est_pipe = Pipeline(steps=[("scaler", StandardScaler()), (m_name, est)])
    return est_pipe


def pre_process_data(data, p_method, params):
    start = time()
    pipe = get_pipe(method=SPCA_VP, m_name=p_method, params=params)
    data_trans = pipe.fit_transform(data, None)
    time_cost = time() - start
    obj = pipe[-1].get_res(obj=True)
    t, ta, tb, tacc = pipe[-1].get_time_list(detail=True)
    return data_trans, obj, time_cost, t, ta, tb, tacc
