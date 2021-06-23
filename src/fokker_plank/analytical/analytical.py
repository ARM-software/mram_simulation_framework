#!/usr/bin/env python
# coding: utf-8

# Copyright (c) 2021-2021 Arm Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Basic functions solving FPE.

Based on
Tzoufras, M. (2018).
Switching probability of all-perpendicular spin valve nanopillars.
AIP Advances, 8(5).
https://doi.org/10.1063/1.5003832

FPE is expanded into Legendre pilynomials:
ρ(θ, τ) = sum[ r_n (τ) Pn(cos θ)
P_n(x) are the Legendre polynomials solution to eq (3) in [1]

∂ρ/∂τ = sum[ sum [r_n a_{n+k, n} P_{n+k}] ] ->
∂r/∂τ = Ar⇒r(τ) = e^(Aτ)r(0) eq (11) at[1]
A is the pentadiagonal matrix with Legendre coefficients (at generate_A())


Notes:
ρ area over theta should be 1/2pi, following [1]
as it integration should be over phi dimension as well in spherical coord
(i.e. int( 2*pi*f(theta)*sin(theta)d_theta )
"""

import numpy as np
import scipy as scipy
import scipy.special
from scipy.stats import maxwell
from scipy import optimize

import fvm.mtj_fp_fvm as fvm
import python_compact_model.sllgs_solver as sllgs_solver


def sph_area_rho_over_theta(rho, theta, axis=-1):
    """
    Compute the area of rho over theta in an spherical system.

    a) d_line_theta = sin(theta) dtheta
    b) d_solid_angle = sin(theta) dtheta dphi
    c) surface element in a surface of polar angle θ constant
    (a cone with vertex the origin):
    d_s_theta = r sin(theta) dphy dr
    d) surface element in a surface of azimuth φ const (a vertical half-plane):
    d_s_phi = r dr dtheta
    Expects theta between [PI, 0] (comming from z in [-1, 1]).
    """
    return np.abs(np.trapz(rho*np.sin(theta), x=theta))


def generate_A(i, h, delta, j):
    """Compute the pentadiagonal matrix."""
    # print(f'[debug] i: {i}, h: {h}, delta: {delta}')
    field_diff = i - h
    inv_2_delta = 1.0/(2.0*delta)
    A = np.zeros([j+1, j+1], dtype=float)
    # fill matrix
    for n in range(2, j-1):
        n2 = 2*n
        A[n-2, n] = -1.0*n*(n-1)*(n-2)/((n2+1)*(n2-1))
        A[n-1, n] = field_diff*n*(n-1)/(n2+1)
        A[n,   n] = -1.0*n*(n+1)*(inv_2_delta-1/((n2+3)*(n2-1)))
        A[n+1, n] = -1.0*field_diff*(n+1)*(n+2)/(n2+1)
        A[n+2, n] = (n+1)*(n+2)*(n+3)/((n2+1)*(n2+3))
    # initial cases and last cases
    # [0,0] = 0
    A[1, 0] = -1.0*field_diff*2
    A[2, 0] = 2.0
    # [0,1] = 0
    A[1, 1] = -2.0*(inv_2_delta - 1/5)
    A[2, 1] = -1.0*field_diff*2
    A[3, 1] = 2*3*4/(3*5)
    A[j-3, j-1] = -1.0*(j-1)*(j-2)*(j-3)/((2*j-1)*(2*j-3))
    A[j-2, j-1] = field_diff*(j-1)*(j-2)/(2*j-1)
    A[j-1, j-1] = -1.0*(j-1)*j*(inv_2_delta - 1/((2*j+1)*(2*j-3)))
    A[j,  j-1] = -1.0*field_diff*j*(j+1)/(2*j-1)
    A[j-2, j] = -1.0*j*(j-1)*(j-2)/((2*j+1)*(2*j-1))
    A[j-1, j] = field_diff*j*(j-1)/(2*j+1)
    A[j,  j] = -1.0*j*(j+1)*(inv_2_delta - 1/((2*j+3)*(2*j-1)))

    return A


def get_state(tau, i=None, h=None, delta=None, state_0=None, a_matrix=None):
    """
    Compute the magnetization state.

    r(τ) = e^(Aτ)r(0) eq (11) at[1]
    """
    if a_matrix is not None:
        # get state from a known A matrix
        # A matrix can be shared and it takes time to build
        return np.matmul(scipy.linalg.expm(tau*a_matrix), state_0)
    return np.matmul(scipy.linalg.expm(
        tau*generate_A(i, h, delta, state_0.size-1)), state_0)


def untangle_state(m_state,
                   lin_space_z=False,
                   dim_points=2000):
    """Untangle the Legendre series from its coefficients."""
    if lin_space_z:
        z = np.linspace(-1, 1, dim_points)
        theta = np.arccos(z)
        return theta, np.polynomial.legendre.legval(z, m_state)
    theta = np.linspace(np.pi, 0, dim_points)
    return theta, np.polynomial.legendre.legval(np.cos(theta), m_state)


# @np.vectorize
def get_time_to_sw(a_matrix, s_rho_0,
                   rho_0_at_pi,
                   target_sw_prob=0.5,
                   target_tolerance=1e-3,
                   t_max=10,
                   do_manual_sw_prob=False,
                   sn=None):
    """Compute time to switch for a given MTJ for a given A matrix."""
    dim_points = 10000
    max_tau_considered = 1e15
    max_iteration = int(1e2)
    t_min = 0

    # initialize integration matrix
    if not do_manual_sw_prob and sn is None:
        sn = get_sn(s_rho_0.shape[0]-1)

    # ensure switching
    while True:
        new_state = get_state(t_max, a_matrix=a_matrix, state_0=s_rho_0)
        if do_manual_sw_prob:
            theta, data = untangle_state(new_state, dim_points=dim_points)
            sw_prob = compute_sw_prob(data=data,
                                      theta=theta,
                                      rho_0_at_pi=rho_0_at_pi,
                                      normalize=True)
        else:
            sw_prob = 1 - get_analytical_sw_prob(new_state, sn)
        if sw_prob > target_sw_prob:
            break
        if t_max > max_tau_considered:
            print('tmax does not meet requirements')
            return np.inf
        t_max *= 10

    iteration = 0
    while iteration < max_iteration:
        t = (t_min + t_max)/2
        new_state = get_state(t, a_matrix=a_matrix, state_0=s_rho_0)
        theta, data = untangle_state(new_state, dim_points=dim_points)
        if do_manual_sw_prob:
            theta, data = untangle_state(new_state, dim_points=dim_points)
            sw_prob = compute_sw_prob(data=data,
                                      theta=theta,
                                      rho_0_at_pi=rho_0_at_pi,
                                      normalize=True)
        else:
            sw_prob = 1 - get_analytical_sw_prob(new_state, sn)
        if sw_prob < 0:
            print(f'[error] negative sw t: {t}, sw_prob: {sw_prob}')
            return np.inf
        if np.abs(sw_prob - target_sw_prob) < target_tolerance:
            print(f'\t\tfound t: {t}, sw prob: {sw_prob}')
            return t
        if iteration > 1 and iteration % 50 == 1:
            print(f'iteration: {iteration}, t: {t}, sw_prob: {sw_prob}'
                  f' t_max: {t_max}, t_min: {t_min}')
        if sw_prob < target_sw_prob:
            t_min = t
        else:
            t_max = t
        iteration += 1
    # max iterations
    return t


def get_time_to_sw_fitting(c, delta, nu, alpha, h_k,
                           temperature=300,
                           rho_0_at_pi=False):
    """
    Compute time to switch for a given MTJ/current for curve fitting.

    h=0 and only current/delta can vary.
    parameters are: i_c, t_d, delta
    """
    temperature = 300
    L0_max = 150
    lin_space_z = False
    # i
    _, _, s_rho_0 = get_state_rho_0(delta=delta,
                                    do_maxwell=False,
                                    L0=L0_max,
                                    rho_0_at_pi=rho_0_at_pi,
                                    lin_space_z=lin_space_z)
    print(f'nu passed: {nu} alpha passed: {alpha} h_k: {h_k}')
    i_c = delta*(4*alpha*sllgs_solver.c_E*sllgs_solver.c_KB *
                 temperature)/(nu*sllgs_solver.c_hbar)
    t_d = (1+alpha*alpha)/(alpha*sllgs_solver.c_gamma_0*sllgs_solver.c_U0*h_k)
    i = c/i_c
    sw_t = np.zeros(i.shape)
    # alpha = 0.02
    # nu = 0.3
    # temperature = 300
    # delta = nu*llg.c_hbar * i_c / (4*alpha*llg.c_E*llg.c_KB*temperature)
    print(f'delta: {delta} for ic: {i_c}, t_d {t_d}')
    for ii_idx, ii in enumerate(i):
        # share A
        A = generate_A(ii, 0.0, delta, s_rho_0.size-1)
        sw_t[ii_idx] = get_time_to_sw(A, s_rho_0, rho_0_at_pi)

    sw_t *= t_d
    if np.any(sw_t <= 0):
        print('[error] negative sw time')
        return -1

    return sw_t


def basic_error_fn(error_mode, _get_time_to_sw_fitting, currents, _times):
    """Specify basic error fn."""
    if error_mode == 0:
        print('[fitting] Error mode 0: doing abs(err/x)^2')

        def err(p): return np.mean((
            (_get_time_to_sw_fitting(currents, *p)-_times)/_times)**2)
    elif error_mode == 1:
        print('[fitting] Error mode 1: doing abs(err)^2')

        def err(p): return np.mean((
            (_get_time_to_sw_fitting(currents, *p)-_times))**2)
    elif error_mode == 2:
        print('[fitting] Error mode 2: doing abs(err/x)')

        def err(p): return np.mean(np.abs(
            (_get_time_to_sw_fitting(currents, *p)-_times)/_times))
    elif error_mode == 3:
        print('[fitting] Error mode 3: doing abs(err)')

        def err(p): return np.mean(np.abs(
            (_get_time_to_sw_fitting(currents, *p)-_times)))
    return err


def _minimize_error(err, minimize_mode, p0, bounds):
    """
    Perform minimization.

    Mode 0: global using basinhopping
    Mode 1: global using shgo
    Mode 2: global using brute force
    Mode 3: local using minimize
    """
    # choose minimization scheme
    if minimize_mode == 0:
        print('[fitting] Optimization algorithm: Basin Hopping')
        # basinhopping for global minima
        res = optimize.basinhopping(err,
                                    p0,
                                    # stepsize=0.5,
                                    niter=500,
                                    # accept_test=my_bounds,
                                    minimizer_kwargs={'bounds': bounds}
                                    )
        print(f'[res] Mode {minimize_mode}: {res}')
        popt = res.x
        return popt
    elif minimize_mode == 1:
        print('[fitting] Optimization algorithm: SHGO')
        # shgo
        res = optimize.shgo(err,
                            bounds=bounds,
                            iters=200)
        print(f'[res] Mode {minimize_mode}: {res}')
        popt = res.x
        return popt
    elif minimize_mode == 2:
        print('[fitting] Optimization algorithm: Brute')
        # shgo
        res = optimize.brute(err,
                             ranges=bounds,
                             finish=optimize.fmin)
        print(f'[res] Mode {minimize_mode}: {res}')
        # if full_output is set to True
        # res[0] has the params, res[1] the error evaluation
        # otherwise:
        popt = res
        return popt
    elif minimize_mode == 3:
        print('[fitting] Optimization using local algorithm')
        # minimize for local minima
        res = optimize.minimize(err,
                                x0=p0,
                                bounds=bounds,
                                method='L-BFGS-B',
                                options={'eps': 1e-13},
                                # method='dogbox',
                                )
        print(f'[res] Mode {minimize_mode}: {res}')
        popt = res.x
        return popt


def _fit_current_time_points(currents,
                             times,
                             rho_0_at_pi,
                             p0=None,
                             bounds=((0., 0.), (np.inf, np.inf)),
                             do_log=False,
                             minimize_mode=0,
                             error_mode=0):
    """Fit current/time 2d array to i/h params."""
    # generate s_rho_0
    L0_max = 150
    lin_space_z = False
    temperature = 300
    # define internal fn to not pass extra params like rho_0_at_pi

    # delta = 55
    # def _get_time_to_sw_fitting(c, nu, alpha, h_k):
    def _get_time_to_sw_fitting(c, delta, nu, alpha, h_k):
        """
        Compute time to switch for a given MTJ/current for curve fitting.

        h=0 and only current/delta can vary.
        parameters are: delta, nu, alpha, h_k
        """
        print(f'delta: {delta}, nu: {nu} alpha: {alpha} h_k: {h_k}')
        _p = np.array([delta, nu, alpha, h_k])
        if np.any(np.isnan(_p)) or np.any(np.isinf(_p)):
            print('[warning] optimizer passing NaN')
            return np.nan
        # initial state cannot be passed
        _, _, s_rho_0 = get_state_rho_0(delta=delta,
                                        do_maxwell=False,
                                        L0=L0_max,
                                        rho_0_at_pi=rho_0_at_pi,
                                        lin_space_z=lin_space_z)
        # i
        i_c = delta*(4*alpha*sllgs_solver.c_E*sllgs_solver.c_KB *
                     temperature)/(nu*sllgs_solver.c_hbar)
        t_d = (1+alpha*alpha) / \
            (alpha*sllgs_solver.c_gamma_0*sllgs_solver.c_U0*h_k)
        i = c/i_c
        sw_t = np.zeros(i.shape)
        # alpha = 0.02
        # nu = 0.3
        # temperature = 300
        # delta = nu*llg.c_hbar * i_c / (4*alpha*llg.c_E*llg.c_KB*temperature)
        print(f'delta: {delta} for ic: {i_c}, t_d {t_d}')
        for ii_idx, ii in enumerate(i):
            # share A
            A = generate_A(ii, 0.0, delta, s_rho_0.size-1)
            sw_t[ii_idx] = get_time_to_sw(A, s_rho_0, rho_0_at_pi)

        sw_t *= t_d
        if np.any(sw_t <= 0):
            print('[error] negative sw time')
            return -np.inf

        if do_log:
            return np.log(sw_t)
        return sw_t

    _times = times
    if do_log:
        _times = np.log(times)

    # minimize approach
    err = basic_error_fn(error_mode=error_mode,
                         _get_time_to_sw_fitting=_get_time_to_sw_fitting,
                         currents=currents,
                         _times=_times)

    # minimize
    popt = _minimize_error(err, minimize_mode, p0, bounds)

    if do_log:
        return popt, np.exp(_get_time_to_sw_fitting(currents, *popt))
    return popt, _get_time_to_sw_fitting(currents, *popt)


def get_nc(delta):
    """Critical Nc."""
    return np.sqrt((delta/2)+1)-1/2


def get_state_rho_0(delta,
                    L0=150,
                    rho_0_at_pi=False,
                    lin_space_z=False,
                    do_maxwell=False,
                    maxwell_loc=None,
                    maxwell_scale=None,
                    dim_points=2000):
    """Generate the initial rho_0 distribution."""
    if lin_space_z:
        # rho with equidistant z (fitting on z, so best option)
        z = np.linspace(-1, 1, dim_points)
        theta = np.arccos(z)
    else:
        # rho with equidistance theta
        theta = np.linspace(np.pi, 0, dim_points)
        z = np.cos(theta)

    if do_maxwell:
        if maxwell_loc is None or maxwell_scale is None:
            theta_0 = 1/np.sqrt(2*delta)
            maxwell_scale = theta_0
            # theta = np.arccos(z)
            # maxwell_loc = 0  # theta_0
            maxwell_loc = -theta_0
            # maxwell_loc = -theta_0/2
        # fitted
        rho_0 = maxwell.pdf(theta,
                            loc=maxwell_loc,
                            scale=maxwell_scale)
    else:
        sin_theta = np.sin(theta)
        rho_0 = np.exp(-delta*sin_theta*sin_theta)*np.heaviside(np.pi/2-theta,
                                                                0.5)
    # area over theta should be 1/2pi:, following
    # Tzoufras, M. (2018).
    # Switching probability of all-perpendicular spin valve nanopillars.
    # AIP Advances, 8(5).
    # https://doi.org/10.1063/1.5003832
    area = sph_area_rho_over_theta(rho_0, theta)
    rho_0 /= (2*np.pi*area)
    # flip if rho at pi
    if rho_0_at_pi:
        rho_0 = rho_0[::-1]

    # fit legendre coefficients
    print(f'[debug] fitting to delta {delta}')
    s_rho_0 = np.polynomial.legendre.legfit(x=z,
                                            y=rho_0,
                                            deg=L0)

    return z, rho_0, s_rho_0


def get_sn(L0):
    """
    Compute s_n vector from eq (12) in [1].

    s_n = int_0^1 (P_n(x) dx)
    """
    dtype = np.longdouble
    # dtype = np.float
    # print('using double precision and exact factorial2')
    fact_type = np.int
    print('using exact factorial2')

    sn = np.arange(L0+1, dtype=dtype)
    sign = -1.0 * np.ones(sn[3::2].shape[0]).astype(dtype)
    sign[1::2] = 1.0
    # factorials
    nf = np.array([scipy.special.factorial2(f, exact=True)
                   for f in sn[3::2].astype(fact_type)]).astype(dtype)
    nfm1 = np.array([scipy.special.factorial2(f, exact=True)
                     for f in sn[2:-1:2].astype(fact_type)]).astype(dtype)
    # odd numbers
    sn[3::2] = sign * nf / (sn[3::2]*(sn[3::2]+1)*nfm1)
    # even numbers
    sn[2::2] = 0.0
    # n==0 and n==1
    sn[0] = 1.0
    sn[1] = 0.5

    return sn.astype(float)


def get_analytical_sw_prob(state, sn=None):
    """Compute switching probability eq (12) in [1]."""
    if sn is None:
        sn = get_sn(state.shape[0])
    return np.dot(2.0*np.pi*sn, state)


def compute_sw_prob(data, theta, rho_0_at_pi, normalize=False):
    """
    Compute switching probability with area on spherical coordinates.

    data is expected to be [time, theta] or [theta].
    FVM and legendre analytical compute over z.
    Transformation is from int^1_-1 [ f(z)dz ] to
    int^hemisphere [ f(theta) sin(theta) dtheta ].
    area is scalled by 2PI as the integral of rho should also be done
    over unit phi vector between 0-2PI.
    """
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)
        # area = sph_area_rho_over_theta(data, theta, axis=-1)
        # print(f'[debug] area*2pi: {2*np.pi*area}')
    # normalize not needed
    # if normalize:
    #     area = sph_area_rho_over_theta(data, theta, axis=-1)
    #     print(f'[debug] area*2pi: {2*np.pi*area}')
    #     # print('[debug] area*2pi over theta '
    #     #       f'{2*np.pi*np.trapz(data, np.cos(theta))}')
    #     first_theta = theta[0]
    #     if first_theta > np.pi/2:
    #         # theta pi->0
    #         # broadcast op with None
    #         data /= (-area[:, None])
    #     else:
    #         # theta 0-> pi
    #         # broadcast op with None
    #         data /= area[:, None]
    if rho_0_at_pi:
        return 2*np.pi*sph_area_rho_over_theta(data[:, theta < np.pi/2],
                                               theta[theta < np.pi/2])
    return 2*np.pi*sph_area_rho_over_theta(data[:, theta > np.pi/2],
                                           theta[theta > np.pi/2])


def get_sw_prob(s_rho_0, tau, delta, i, h,
                rho_0_at_pi=False,
                compute_fvm=False,
                compute_analytical_sw=False,
                compute_analytical_manual=False,
                dim_points=500,
                lin_space_z=False,
                t_step=1e-3,
                sn=None):
    """
    Evolve FPE and compute switching probability over theta.

    FVM and legendre analytical compute over z.
    Transformation is from int^1_-1 [ f(z)dz ] to
    int^hemisphere [ f(theta) sin(theta) dtheta ].
    """
    # analytical rho computation
    new_state = get_state(tau, i, h, delta, s_rho_0)
    theta, data = untangle_state(new_state, dim_points=5000)
    if compute_analytical_manual:
        manual_prob = compute_sw_prob(data=data,
                                      theta=theta,
                                      rho_0_at_pi=rho_0_at_pi,
                                      normalize=True)
    else:
        manual_prob = -1
    if compute_fvm:
        theta, rho_init = untangle_state(s_rho_0,
                                         dim_points=dim_points,
                                         lin_space_z=lin_space_z)
        rho_init = np.array(rho_init)
        fvm_data = fvm.solve_mtj_fp(rho_init=rho_init,
                                    delta=delta,
                                    i0=i,
                                    h=h,
                                    T=tau,
                                    dim_points=dim_points+1,
                                    t_step=t_step,
                                    do_3d=False,
                                    lin_space_z=lin_space_z)
        theta = np.arccos(fvm_data['z0'])
        data = fvm_data['rho']
        fvm_prob = compute_sw_prob(data=data,
                                   theta=theta,
                                   rho_0_at_pi=rho_0_at_pi,
                                   normalize=True)
    else:
        fvm_prob = -1
    if compute_analytical_sw:
        analytical_prob = 1 - get_analytical_sw_prob(new_state, sn)
    else:
        analytical_prob = -1
    return {'manual_prob': manual_prob,
            'fvm_prob': fvm_prob,
            'analytical_prob': analytical_prob}


def get_sw_continuous_prob(s_rho_0, tau, delta, i, h,
                           rho_0_at_pi=False,
                           compute_fvm=False,
                           compute_analytical_sw=False,
                           compute_analytical_manual=False,
                           dim_points=1000,
                           lin_space_z=False,
                           t_step=1e-3,
                           sn=None):
    """Evolve FPE and compute switching probability over time between 0-tau."""
    # analytical rho computation
    time = np.arange(0, tau, t_step)
    fvm_time = np.arange(0, tau, t_step)
    manual_prob = np.zeros(time.shape[0])
    distribution = np.zeros((time.shape[0], dim_points))
    analytical_prob = np.zeros(time.shape[0])
    # share A
    A = generate_A(i, h, delta, s_rho_0.size-1)
    for t_idx, t in enumerate(time):
        new_state = get_state(t, a_matrix=A, state_0=s_rho_0)
        theta, data = untangle_state(new_state, dim_points=dim_points)
        distribution[t_idx] = data
        if compute_analytical_manual:
            manual_prob[t_idx] = compute_sw_prob(data=data,
                                                 theta=theta,
                                                 rho_0_at_pi=rho_0_at_pi,
                                                 normalize=True)
        else:
            manual_prob[t_idx] = -1
        if compute_analytical_sw:
            analytical_prob[t_idx] = 1 - get_analytical_sw_prob(new_state, sn)
        else:
            analytical_prob[t_idx] = -1
    fvm_theta = theta
    if compute_fvm:
        theta, rho_init = untangle_state(s_rho_0,
                                         dim_points=dim_points,
                                         lin_space_z=lin_space_z)
        rho_init = np.array(rho_init)
        fvm_data = fvm.solve_mtj_fp(rho_init=rho_init,
                                    delta=delta,
                                    i0=i,
                                    h=h,
                                    T=tau,
                                    dim_points=dim_points+1,
                                    t_step=t_step,
                                    do_3d=True,
                                    lin_space_z=lin_space_z)
        fvm_theta = np.arccos(fvm_data['z0'])
        data = fvm_data['rho'].T
        fvm_time = fvm_data['t0']
        fvm_prob = compute_sw_prob(data=data,
                                   theta=fvm_theta,
                                   rho_0_at_pi=rho_0_at_pi,
                                   normalize=True)
    else:
        fvm_prob = -1*np.ones(time.shape[0])
    return {'time': time,
            'theta': theta,
            'manual_prob': manual_prob,
            'pdf': distribution,
            'fvm_time': fvm_time,
            'fvm_theta': fvm_theta,
            'fvm_prob': fvm_prob,
            'analytical_prob': analytical_prob}
