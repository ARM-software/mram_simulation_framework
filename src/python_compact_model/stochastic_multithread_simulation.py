#!/usr/bin/env python
# coding: utf-8
#
# Copyright (c) 2020-2021 Arm Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Multi-threading capabilities example.

Generation of 'total_th_sims' random walks for later comparison against FP.
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
import multiprocessing

import sllgs_solver as sllg_solver


#################################
#################################
# Parameters not passed to the worker,
# change accordingly to your needs
cores = 32
total_th_sims = 10000

c_KB = 1.3806503e-23  # boltzmann's constant [J/K]
diam = 50e-9
t_fl = 1.e-9
temperature = 300
k_i_0 = 1.0e-3
ms = 1.2e6  # magnetisation saturation (A/m)
H_ext = (0., 0., 0.)  # low external magnetic field (A/m) (test anisotropy)
# stt params
eps_prime = 0.0 # [FITTING] constant
p = 0.75
lambda_s = 1.
z0 = 1
initial_m = None

# just for export
t_res = 1e-11
#################################
#################################


def str2bool(b_string):
    """Convert string to bool."""
    b_string = b_string.strip()
    if b_string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b_string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    # error
    print('[ERROR] Boolean value expected.')
    return 'a'


def worker_stt_simplest(idx,
                        solve_normalized,
                        method,
                        I0,
                        t_delay,
                        t_pulse,
                        t_final,
                        H_ext,
                        initial_m,
                        w_fl,
                        l_fl,
                        t_fl,
                        ms,
                        alpha,
                        k_i_0,
                        temperature,
                        p,
                        lambda_s,
                        eps_prime,
                        do_thermal,
                        do_theta_windowing,
                        do_fake_thermal,
                        theta_windowing_factor,
                        d_theta_fake_th,
                        max_step):
    """Parallel worker fn. A simple wrapper."""
    # current
    I0 = -np.sign(z0)*I0

    def test_h_ext(t):
        """H_ext fn."""
        return np.array(H_ext)

    def test_current(t):
        """Current fn."""
        if type(t) is np.ndarray:
            c = np.zeros(t.shape) + I0
            c[t < t_delay] = 0
            c[t > t_delay+t_pulse] = -I0
            return c
        if t < t_delay:
            return 0
        elif t > t_delay+t_pulse:
            return 0
        return I0

    print(f'[info][parallel_solver] doing s: {idx} I0 {I0} alpha {alpha} '
          f'max_step {max_step} tf {t_final}')
    llg_b = sllg_solver.LLG(
        w_fl=diam,
        l_fl=diam,
        t_fl=t_fl,
        ms=ms,
        alpha=alpha,
        k_i_0=k_i_0,
        temperature=temperature,
        stt_mode='stt_oommf_simple',
        p=p,
        lambda_s=lambda_s,
        eps_prime=eps_prime,
        # not include temperature noise,
        # and do not force its effects on the theta_0
        do_thermal=do_thermal,
        do_theta_windowing=do_theta_windowing,
        do_fake_thermal=do_fake_thermal,
        theta_windowing_factor=theta_windowing_factor,
        d_theta_fake_th=d_theta_fake_th,
        m_init=initial_m,
        theta_pl=0.0,                  # [rad] pinned layer theta
        phi_pl=0.0,                    # [rad] pinned layer phi
        h_ext_cart=test_h_ext,
        i_amp_fn=test_current,
        seed=idx+2000)
    llg_sol = llg_b.solve_and_plot(final_t=t_final,
                                   max_step=max_step,
                                   solve_spherical=False,
                                   solve_normalized=solve_normalized,
                                   scipy_ivp=False,
                                   method=method,
                                   plot=False,
                                   plot_simple=False)
    print(f'[info][parallel_solver] done {idx} I0 {I0}'
          f' alpha: {alpha} t_final: {t_final}')

    # do interpolation in the thread
    t_i = np.linspace(0, t_final, int(t_final/t_res))
    theta_i = np.interp(t_i, llg_sol.t, llg_sol.y[1])

    # do interpolation in the thread
    return theta_i


def run_walks(
        suffix,
        solve_normalized,
        method,
        I0,
        alpha,
        max_step,
        t_delay,
        t_pulse=60e-9,
        total_th_sims=10000,
        cores=32,
        plot_walks=True,
):
    """Run walks."""
    t_final = 2*t_delay + t_pulse

    # pool
    pool = multiprocessing.Pool(cores)
    results_a = [pool.apply_async(
        worker_stt_simplest,
        args=(
            idx,
            solve_normalized,
            method,
            I0,
            t_delay,
            t_pulse,
            t_final,
            H_ext,
            initial_m,
            diam,  # w_fl,
            diam,  # l_fl,
            t_fl,
            ms,
            alpha,
            k_i_0,
            temperature,
            p,
            lambda_s,
            eps_prime,
            True,  # do_thermal,
            False,  # do_theta_windowing,
            False,  # do_fake_thermal,
            0.,  # theta_windowing_factor,
            1/13,  # d_theta_fake_th
            max_step,
        )) for idx in range(total_th_sims)]
    print('[info][parallel_solver] all threads launched')

    theta_i_list = [r.get() for r in results_a]
    t_i = np.linspace(0, t_final, int(t_final/t_res))
    print('[info][parallel_solver] all threads computed')

    np.savetxt(
        f'{suffix}_{t_delay}_del_{I0}A_interp_{total_th_sims}_sims.csv',
        np.array(theta_i_list))
    np.savetxt(
        f'{suffix}_{t_delay}_del_{I0}A_time_interp_'
        f'{total_th_sims}_sims.csv',
        t_i)

    if not plot_walks:
        return

    print('[info][parallel_solver] plotting')
    fig, ax = plt.subplots(1, 1, figsize=(6, 2.3))

    for i in np.arange(total_th_sims):
        ax.plot(1e9*t_i, np.cos(theta_i_list[i]), 'gray', alpha=0.5)

    ax.plot(np.nan,
            color='gray', alpha=0.5, label='Stochastic H_th')
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylabel('m_z')
    ax.set_xlabel('time [ns]')
    ax.legend(loc='upper right')
    ax.grid()
    ax.set_title(suffix)
    plt.show()
    plt.savefig(f'{suffix}_{t_delay}_del_{I0}A_time_interp.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--solve-normalized',
        type=str2bool,
        default=False,
        help='use normalized solver')
    parser.add_argument(
        '--method',
        type=str,
        default='heun',
        help='sde solver')
    parser.add_argument(
        '--suffix',
        type=str,
        default='rand_walk',
        help='suffix')
    parser.add_argument(
        '--max-step',
        type=float,
        default=1e-13,
        help='max step [s]')
    parser.add_argument(
        '--I0',
        type=float,
        default=40e-6,
        help='wr current')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.01,
        help='damping')
    parser.add_argument(
        '--t-delay',
        type=float,
        default=3e-9,
        help='delay time')
    parser.add_argument(
        '--t-pulse',
        type=float,
        default=60e-9,
        help='sim time')
    parser.add_argument(
        '--total-th-sims',
        type=int,
        default=int(1e2),
        help='sims')
    parser.add_argument(
        '--cores',
        type=int,
        default=int(32),
        help='cores')
    parser.add_argument(
        '--plot-walks',
        type=str2bool,
        default=False,
        help='Plot distribution')

    args = parser.parse_args()

    # # parse and run
    # FLAGS, unparsed = parser.parse_known_args()
    print('\t[experiment] reproduce results with:')
    for k in args.__dict__:
        print('\t\t', k, ': ', args.__dict__[k])

    run_walks(
        solve_normalized=args.solve_normalized,
        method=args.method,
        suffix=args.suffix,
        max_step=args.max_step,
        I0=args.I0,
        alpha=args.alpha,
        t_delay=args.t_delay,
        t_pulse=args.t_pulse,
        total_th_sims=args.total_th_sims,
        cores=args.cores,
        plot_walks=args.plot_walks
    )

    print('\t[experiment] reproduce results with:')
    for k in args.__dict__:
        print('\t\t', k, ': ', args.__dict__[k])
