import numpy as np


def inner_product(x1, x2):
    """inner product for matrix_res"""
    return np.trace(np.dot(x1.T, x2))


def retraction(x, t):
    """L1 manifold retraction"""
    return x + t


def project(x, g):
    """x is param, g is the gradient, return riemannian gradient"""
    return g * (x != 0)


def manifold_accelerate(grad, param, r_alpha=None):
    """return param processed by riemannian gradient method"""
    r_grad = project(param, grad)
    return retraction(param, -r_alpha * r_grad)
