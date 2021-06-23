#!/usr/bin/env python
# coding: utf-8
#
# Copyright (c) 2020-2021 Arm Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""s-LLGs importing tool."""

import numpy as np


def process_sllgs_data(data_file, time_file, _t_delay, dim_points):
    """
    Process s-LLGS.

    Note it removes the delayed samples.
    """
    print(f'[debug] reading {data_file}')
    data_tol = np.loadtxt(data_file, delimiter=' ')
    _time_tol = np.loadtxt(time_file, delimiter=' ')
    theta_axis = np.linspace(np.pi, 0, dim_points)

    ####################################################################
    # Process data and doing FPE
    # ~/SharePoint/ProjectEnergyScalableMRAM/docs/essderc_2021/raw_data
    ####################################################################
    data_tol = np.abs(data_tol)
    print(data_tol.shape)
    x = np.zeros(data_tol.shape[0]*data_tol.shape[1])
    y = np.zeros(data_tol.shape[0]*data_tol.shape[1])
    _sllgs_ps_tol = np.zeros(_time_tol.shape[0])
    for i in range(data_tol.shape[0]):
        for j in range(data_tol.shape[1]):
            idx = i*data_tol.shape[1] + j
            x[idx] = _time_tol[j]
            y[idx] = np.abs(data_tol[i, j])
    _s_llgs_bins = [np.max([50, data_tol.shape[1]]), dim_points]
    _sllgs_rho, _, _ = np.histogram2d(x, y,
                                      bins=_s_llgs_bins,
                                      density=True)
    print(_sllgs_rho.shape)
    # normalize and log
    _sllgs_tol_sw_idx = 0
    for h_idx, h in enumerate(_sllgs_rho):
        area = np.abs(np.trapz(h*np.sin(theta_axis), x=theta_axis))
        _sllgs_rho[h_idx] = h/(2*np.pi*area)

    _sllgs_ps_tol = np.zeros(_time_tol.shape[0])
    _sllgs_tol_sw_idx = 0
    for j in range(data_tol.shape[1]):
        _sllgs_ps_tol[j] = float(
            np.sum(data_tol[:, j] > np.pi/2))/float(data_tol.shape[0])
        if _sllgs_tol_sw_idx == 0 and _sllgs_ps_tol[j] >= 0.5:
            _sllgs_tol_sw_idx = j
            print('switch at ', j)

    _sllgs_rho = _sllgs_rho.T
    print(f'H_tol shape: {_sllgs_rho.shape}')
    # H_tol = np.log(H_tol.T)
    print('H computed, sw at {1e9*_time_tol[sllgs_low_tol_switching_idx]}')
    rho_0_time_idx = np.searchsorted(_time_tol, _t_delay)
    print(f'fitting time: {_time_tol[rho_0_time_idx]}')
    # shifting till delay
    _time_tol = _time_tol[rho_0_time_idx:] - _t_delay
    _sllgs_rho = _sllgs_rho[:, rho_0_time_idx:]
    _sllgs_tol_sw_idx -= rho_0_time_idx
    _sllgs_ps_tol = _sllgs_ps_tol[rho_0_time_idx:]
    return _time_tol, _sllgs_rho, _sllgs_tol_sw_idx, _sllgs_ps_tol
