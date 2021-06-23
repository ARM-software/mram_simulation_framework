#!/usr/bin/env python
# coding: utf-8
#
# Copyright (c) 2020-2021 Arm Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""s-LLGS basic examples"""

import sllgs_solver as sllg_solver


def current(t):
    """Current waveform."""
    return -100e-6


def do_test(seed_0=100):
    """Simple s-LLGS tests."""
    sllg_solver.eq_info()
    # No thermal, no fake_thermal, solved with scipy_ivp
    llg_a = sllg_solver.LLG(do_fake_thermal=False,
                            do_thermal=False,
                            i_amp_fn=current,
                            seed=seed_0)
    data_a = llg_a.solve_and_plot(15e-9,
                                  scipy_ivp=True,
                                  solve_spherical=True,
                                  solve_normalized=True,
                                  max_step=1e-11,
                                  rtol=1e-4,
                                  atol=1e-9)

    # No thermal, fake_thermal, solved with scipy_ivp
    llg_b = sllg_solver.LLG(do_fake_thermal=True,
                            d_theta_fake_th=1/30,
                            do_thermal=False,
                            i_amp_fn=current,
                            seed=seed_0)
    data_b = llg_b.solve_and_plot(15e-9,
                                  scipy_ivp=True,
                                  solve_spherical=True,
                                  solve_normalized=True,
                                  max_step=1e-11,
                                  rtol=1e-4,
                                  atol=1e-9)

    llg_c = sllg_solver.LLG(do_fake_thermal=False,
                            do_thermal=True,
                            i_amp_fn=current,
                            seed=seed_0)
    data_c = llg_c.solve_and_plot(10e-9,
                                  scipy_ivp=False,
                                  solve_spherical=False,
                                  solve_normalized=True,
                                  max_step=1e-13,
                                  method='stratonovich_heun')
    print(f'data ready to inspect: {data_a}, {data_b}, {data_c}')


do_test()
