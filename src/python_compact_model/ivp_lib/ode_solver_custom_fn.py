#!/usr/bin/env python
# coding: utf-8
#
# Copyright (c) 2020-2021 Scipy.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# The following fns are modifications from Scipy
# rk.py and base.py files  Latest commit a534a56 on Aug 18, 2020

"""
Auxiliar functions for solve_ivp problem.

Required to pass dt to the integration function.
"""

import numpy as np


def ode_solver_init_custom(self, fun, t0, y0, t_bound, vectorized,
                           support_complex=False):
    """Replace for OdeSolver __init__ passing h to the wrappers."""
    self.t_old = None
    self.t = t0
    self._fun, self.y = check_arguments_custom(fun, y0, support_complex)
    self.t_bound = t_bound
    self.vectorized = vectorized

    if vectorized:
        def fun_single(t, y, h):
            return self._fun(t, y[:, None], h).ravel()
        fun_vectorized = self._fun
    else:
        fun_single = self._fun

        def fun_vectorized(t, y, h):
            f = np.empty_like(y)
            for i, yi in enumerate(y.T):
                f[:, i] = self._fun(t, yi, h)
            return f

    def fun(t, y, h=1e-10):
        self.nfev += 1
        return self.fun_single(t, y, h)

    self.fun = fun
    self.fun_single = fun_single
    self.fun_vectorized = fun_vectorized

    self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1
    self.n = self.y.size
    self.status = 'running'

    self.nfev = 0
    self.njev = 0
    self.nlu = 0


def check_arguments_custom(fun, y0, support_complex):
    """Helper function for checking arguments common to all solvers."""
    y0 = np.asarray(y0)
    if np.issubdtype(y0.dtype, np.complexfloating):
        if not support_complex:
            raise ValueError("`y0` is complex, but the chosen solver does "
                             "not support integration in a complex domain.")
        dtype = complex
    else:
        dtype = float
    y0 = y0.astype(dtype, copy=False)

    if y0.ndim != 1:
        raise ValueError("`y0` must be 1-dimensional.")

    def fun_wrapped(t, y, dt=1e-10):
        return np.asarray(fun(t, y, dt), dtype=dtype)

    return fun_wrapped, y0


def rk_step_custom(fun, t, y, f, h, A, B, C, K):
    """Perform a single Runge-Kutta step.

    This function computes a prediction of an explicit Runge-Kutta method and
    also estimates the error of a less accurate method.

    Notation for Butcher tableau is as in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
        Current value of the derivative, i.e., ``fun(x, y)``.
    h : float
        Step to use.
    A : ndarray, shape (n_stages, n_stages)
        Coefficients for combining previous RK stages to compute the next
        stage. For explicit methods the coefficients at and above the main
        diagonal are zeros.
    B : ndarray, shape (n_stages,)
        Coefficients for combining RK stages for computing the final
        prediction.
    C : ndarray, shape (n_stages,)
        Coefficients for incrementing time for consecutive RK stages.
        The value for the first stage is always zero.
    K : ndarray, shape (n_stages + 1, n)
        Storage array for putting RK stages here. Stages are stored in rows.
        The last row is a linear combination of the previous rows with
        coefficients

    Returns
    -------
    y_new : ndarray, shape (n,)
        Solution at t + h computed with a higher accuracy.
    f_new : ndarray, shape (n,)
        Derivative ``fun(t + h, y_new)``.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    # print(f'fn: {fun}')

    K[0] = f
    for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
        dy = np.dot(K[:s].T, a[:s]) * h
        K[s] = fun(t + c * h, y + dy, c * h)

    y_new = y + h * np.dot(K[:-1].T, B)
    f_new = fun(t + h, y_new, h)

    K[-1] = f_new

    return y_new, f_new


def rk_step_custom_circular(fun, t, y, f, h, A, B, C, K):
    """Perform a single Runge-Kutta step.

    This function computes a prediction of an explicit Runge-Kutta method and
    also estimates the error of a less accurate method.

    Notation for Butcher tableau is as in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
        Current value of the derivative, i.e., ``fun(x, y)``.
    h : float
        Step to use.
    A : ndarray, shape (n_stages, n_stages)
        Coefficients for combining previous RK stages to compute the next
        stage. For explicit methods the coefficients at and above the main
        diagonal are zeros.
    B : ndarray, shape (n_stages,)
        Coefficients for combining RK stages for computing the final
        prediction.
    C : ndarray, shape (n_stages,)
        Coefficients for incrementing time for consecutive RK stages.
        The value for the first stage is always zero.
    K : ndarray, shape (n_stages + 1, n)
        Storage array for putting RK stages here. Stages are stored in rows.
        The last row is a linear combination of the previous rows with
        coefficients

    Returns
    -------
    y_new : ndarray, shape (n,)
        Solution at t + h computed with a higher accuracy.
    f_new : ndarray, shape (n,)
        Derivative ``fun(t + h, y_new)``.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    # print(f'fn: {fun}')

    def _circular_theta(m):
        # circular operator on theta
        if m[1] > np.pi:
            m[1] = m[1] - np.pi
        elif m[1] < 0:
            m[1] = - m[1]
        return m

    K[0] = f
    # circular operator on theta
    K[0] = _circular_theta(K[0])
    for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
        dy = np.dot(K[:s].T, a[:s]) * h
        K[s] = fun(t + c * h, y + dy, c * h)
        # circular operator on theta
        K[s] = _circular_theta(K[s])

    y_new = y + h * np.dot(K[:-1].T, B)
    f_new = fun(t + h, y_new, h)
    # circular operator on theta
    f_new = _circular_theta(f_new)

    K[-1] = f_new

    return y_new, f_new
