#!/usr/bin/env python
# coding: utf-8
#
# Copyright (c) 2020-2021 Arm Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
s-LLGS Solver for MRAM Magnetization.

Computation of LLGS equation following OOMMF Oxs_SpinXferEvolve
https://math.nist.gov/oommf/doc/userguide20a2/userguide/Standard_Oxs_Ext_Child_Clas.html
Each one of the field components computation
follows the reference defined in the respective method.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate._ivp import rk, base
from scipy.integrate import solve_ivp
from scipy.stats import maxwell

import ivp_lib.ode_solver_custom_fn as custom_ode

# gyromagnetic ratio
c_gamma_0 = 1.76e11  # [rad/s/T]
# https://www.ctcms.nist.gov/~rdm/mumag.org.html
# equivalent to c_U0*2.211e5 (ommited often [rad]) * [m/C] = ([rad]) * [ m/A/s]

# elementary charge, [C]
c_E = 1.602176462e-19
# reduced plank's constant [J s]
c_hbar = 1.054571817e-34
# vacuum permeability, 4 pi / 1e7 [H/m]
c_U0 = 1.25663706212e-6
# boltzmann's constant [J/K]
c_KB = 1.3806503e-23
# Borh magneton [J/T]
c_UB = c_gamma_0 * c_hbar / 2

# custom fitting
loc_theta_0_coeff = 1/3.


def eq_info():
    """Equation info."""
    print('\t* Considered energies:')
    print('\t\ta) Zeeman energy (due to external field)')
    print('\t\tb) Uniaxial anisotropy energy (interfacial, see bellow)')
    print('\t\tc) Thermal energy')
    print('\t\td) Shape anisotropy energy')
    print('\n\t\tZeeman energy, uniaxial anisotropy and thermal energy')
    print('\t\tcontribute to Hext.')

    print('\n\t* Full anisotropy:')
    print('\tWatanabe, K., ')
    print('\tShape anisotropy revisited in single-digit nanometer magnetic')
    print('\ttunnel junctions.')
    print('\tNature Communications, 9(1), 5–10.')
    print('\thttps://doi.org/10.1038/s41467-018-03003-7')

    print('\n\t* Computation of LLGS equation following'
          'OOMMF Oxs_SpinXferEvolve')
    print('\thttps://math.nist.gov/oommf/doc/userguide20a2/userguide/'
          'Standard_Oxs_Ext_Child_Clas.html')
    print('\t with the extra in-plane terms')
    print('\t see OOMMF for the value of the different.')

    print('\n\tLLG equation (with all stt terms) at eq 1.3')
    print('\tSwitching distributions for perpendicular spin-torque devices')
    print('\twithin the macrospin approximation.')
    print('\tIEEE Transactions on Magnetics, 48(12), 4684–4700.')
    print('\thttps://doi.org/10.1109/TMAG.2012.2209122')

    print('\t# (1+alpha^2)dm/dt = #LLG ')
    print('\t#                    # Gilbert damping model: ')
    print('\t#                    -gamma c_U0 m^H_eff')
    print('\t#                    # Gilbert damping model: ')
    print('\t#                    -alpha gamma c_U0 m^m^H_eff')
    print('\t#                    # T_CPP (torque current perp. plane) ')
    print('\t#                    # Slonczewski Model: ')
    print('\t#                    -gamma nu I_ c_hbar / (2 c_E V) m^m^mp')
    print('\t#                    +alpha gamma c_hbar f(I_)/ (2 c_E V) m^m^mp')
    print('\t#                    # T_CIP (torque current in plane) Zhang-Li')
    print('\t#                    +gamma nu f(I_) c_hbar / (2 c_E V) m^mp')
    print('\t#                    +alpha gamma nu I_ c_hbar / (2 c_E V) m^mp')

    print('\n\t-> normalized and grouping by I_ (h_stt) and f(I_) (h_stt_2):')
    print('\t# h_eff = H_eff / (gamma c_U0 Ms)')
    print('\t# Ns = 2 Ms V / (gamma c_hbar)')
    print('\t# I = Is / (c_E gamma c_U0 Ms Ns)')
    print('\t#   = 2 Js / (c_E c_U0 Ms t_fl area)')
    print('\t# (1+alpha^2)dm/dt = -m^h_eff      -alpha*m^m^h_eff')
    print('\t#                    -m^m^h_stt    +alpha*m^h_stt')
    print('\t#                    +m^h_stt_2  +alpha*m^m^h_stt_2')
    print('\n\t-> spherical')
    print('\tAment, S., et al.')
    print('\tSolving the stochastic Landau-Lifshitz-Gilbert-Slonczewski')
    print('\tequation for monodomain nanomagnets :')
    print('\tA survey and analysis of numerical techniques.')
    print('\tRetrieved from http://arxiv.org/abs/1607.04596')
    print('\t# (1+alpha^2)dtheta/dt = [')
    print('\t#       h_eff_phi     + alpha*h_eff_theta')
    print('\t#     + h_stt_1_theta - alpha*h_stt_phi')
    print('\t#     - h_stt_2_phi   - alpha*h_stt_2_theta')
    print('\t# ]')
    print('\t# (1+alpha^2)dphi/dt = [')
    print('\t#     - h_eff_theta   + alpha*h_eff_phi')
    print('\t#     + h_stt_1_phi   + alpha*h_stt_1_theta')
    print('\t#     + h_stt_2_theta - alpha*h_stt_2_phi')
    print('\t# ]/sin(theta)')

    print('\n\t* Noise')
    print('\tPinna, D., et. al.')
    print('\tSpin-transfer torque magnetization reversal in uniaxial')
    print('\tnanomagnets with thermal noise.')
    print('\tJournal of Applied Physics, 114(3), 1–9.')
    print('\thttps://doi.org/10.1063/1.4813488')


def normalize_cart(u):
    """Normalize a cartesian vector u."""
    return u / np.sqrt(np.sum(u**2))


def uxv_cart(u, v):
    """U x V in cartesians."""
    w_x = (u[1]*v[2] - v[1]*u[2])
    w_y = (-u[0]*v[2] + u[2]*v[0])
    w_z = (u[0]*v[1] - u[1]*v[0])
    return np.array([w_x, w_y, w_z])


def cart_from_spherical_fn(sph_vector):
    """
    Convert a vector from spherical to cartesian.

    sph_vector is [idx][ro, theta, phi]
               or [rho, theta, phi]
    """
    if len(sph_vector.shape) != 2:
        sph_vector = np.expand_dims(sph_vector, axis=0)
        expanded = True
    else:
        expanded = False
    xyz_vector = np.zeros(sph_vector.shape)
    xyz_vector[:, 0] = sph_vector[:, 0] * \
        np.sin(sph_vector[:, 1])*np.cos(sph_vector[:, 2])
    xyz_vector[:, 1] = sph_vector[:, 0] * \
        np.sin(sph_vector[:, 1])*np.sin(sph_vector[:, 2])
    xyz_vector[:, 2] = sph_vector[:, 0] * np.cos(sph_vector[:, 1])
    if expanded:
        return xyz_vector[0]
    return xyz_vector


def spherical_from_cart_np(xyz_vector):
    """
    Convert a vector from cart to spherical.

    cart_vector is [idx][x, y, z]
    """
    if len(xyz_vector.shape) != 2:
        xyz_vector = np.expand_dims(xyz_vector, axis=0)
        expanded = True
    else:
        expanded = False
    sph_vector = np.zeros(xyz_vector.shape)
    xy = xyz_vector[:, 0]**2 + xyz_vector[:, 1]**2
    sph_vector[:, 0] = np.sqrt(xy + xyz_vector[:, 2]**2)
    # for elevation angle defined from Z-axis down
    sph_vector[:, 1] = np.arctan2(np.sqrt(xy), xyz_vector[:, 2])
    # for elevation angle defined from XY-plane up
    # sph_vector[:,1] = np.arctan2(xyz_vector[:,2], np.sqrt(xy))
    sph_vector[:, 2] = np.arctan2(xyz_vector[:, 1], xyz_vector[:, 0])
    if expanded:
        return sph_vector[0]
    return sph_vector


def oommf_test(t):
    """Return test on OOMMF."""
    # return np.array([0.0, 0.0, 0.0])
    return np.array([0.0, 0.0, 2e6])


def zero3(t):
    """Return array of zeros."""
    return np.array([0.0, 0.0, 0.0])


def zero(t):
    """Return default function for current."""
    return 0.0


def def_window_power(d_theta, theta, theta_min,  p=20):
    """
    "Windowing fn for theta.

    Requires theta between [0, np.pi], not an open integration.
    """
    def _window(x):
        # power window
        return 1. - np.power(x - 1, p)

    # required if theta_ini bellow theta_min
    if theta > np.pi/2 and d_theta < 0:
        return 1.
    if theta < np.pi/2 and d_theta > 0:
        return 1.
    delta = theta_min/3
    if theta > np.pi/2 and np.pi - theta >= theta_min:
        return _window(-theta + (np.pi - theta_min - delta))
    if theta <= np.pi/2 and theta >= theta_min:
        return _window(theta - theta_min - delta)
    else:
        return 0.


def def_window(d_theta, theta, theta_min):
    """
    "Windowing fn for theta.

    Requires theta between [0, np.pi], not an open integration.
    """
    def _window(x):
        # Tukey window
        if x < theta_min/4:
            return 0.5*(1 - np.cos(2*np.pi*x / (theta_min/2)))
        return 1.

    # required if theta_ini bellow theta_min
    if theta > np.pi/2 and d_theta < 0:
        return 1.
    if theta < np.pi/2 and d_theta > 0:
        return 1.
    if theta > np.pi/2 and np.pi - theta >= theta_min:
        return _window(-theta + (np.pi - theta_min))
    if theta <= np.pi/2 and theta >= theta_min:
        return _window(theta - theta_min)
    else:
        return 0.


class Sol:
    """Solution class."""

    def __init__(self):
        pass


class LLG:
    """Basic s-LLGS class."""

    def __init__(self,
                 hk_mode=1,
                 do_thermal=False,              # simulate thermal noise
                 do_theta_windowing=False,      # force a min theta angle
                                                # (set to theta_init) if
                                                # thermal is not computed,
                                                # to avoid total damping
                 do_fake_thermal=True,          # simulate fake thermal noise
                 d_theta_fake_th=1/15,          # d_theta will have an extra
                                                # term based on this
                 theta_windowing=def_window,    # theta windowing function
                                                # should min theta be forced
                 theta_windowing_factor=2,      # theta_min = factor*theta_0
                 seed=0,
                 i_amp_fn=zero,                 # [A] applied current f(t)
                                                # in P direction
                 temperature=300,               # [K]
                 t_fl=1.0e-9,                   # [m] thickness of free layer
                 t_ox=1.6e-9,                   # [m] thickness of oxide
                 w_fl=50e-9,                    # [m] mtj width
                 l_fl=50e-9,                    # [m] mtj length
                 # init state
                 # order of priority: m_init, angle_init, state
                 # init state priority 0
                 m_init=None,                   # initial m vector state
                                                # [x, y, z],
                 # init state priority 1
                 # only used if m_init is None
                 theta_init=None,               # [rad] None  1/sqrt(2delta(T))
                 phi_init=0.1*np.pi/180,        # [rad]
                 # init state priority 2
                 # only used if m_init and angle_init are None
                 state='P',                     # Initial state, only used if
                                                # theta_init=None

                 # magnetic parameters
                 k_i_0=1e-3,                    # uniaxial anisotropy [J/m^2]
                 alpha=0.005,                   # alpha damping factor
                 ms=0.97e6,                     # magnetization sat. [A/m]
                 # shape anisotropy
                 shape_ani_demag_mode=2,        # 0: no shape anisotropy
                                                # 1: shape_ani_demag_n vector
                                                # 2 and 3:
                                                # two different literature
                                                # implementations
                 shape_ani_demag_n=None,        # Shape anisotropy demag vector
                                                # if None, it is computed
                                                # as a cylinder
                 # exchange energy
                 do_a_exchange=False,           # whether do or not energy
                                                # exchange
                 a_exchange=1e-11,              # Exchange constant [J/m]

                 # stt parameters
                 stt_mode='stt_oommf_simple',   # stt_oommf_full:
                                                #   lamb. & P free per layer
                                                # stt_oommf_simple:
                                                #   single lambda and P
                                                # stt_simplest: just p
                 # stt mode oomf_full
                 p_pl=0.3,                      # polarization factor
                 p_fl=0.25,                      # polarization factor
                 lambda_pl=1.2,                 # [FITTING] parameter
                 lambda_fl=1.2,                 # [FITTING] parameter
                                                # for out of plane torque
                 # stt mode oomf_simple
                 lambda_s=1.2,                  # [FITTING] parameter
                 p=0.35,                        # polarization factor
                 # stt mode simplest
                 nabla_mult=1,                  # epsilon = nabla/2
                 # secondary STT term
                 eps_prime=0.0,                 # [FITTING] constant

                 # vcma paramters
                 do_vcma=False,
                 xi=61e-15,                     # vcma coeff, [J/(V*m)]

                 # TMR and conductance
                 r_p=6e3,                       # RP resistance, [ohm]

                 # pinned layer angle
                 theta_pl=0.0,                  # [rad] pinned layer theta
                 phi_pl=0.0,                    # [rad] pinned layer phi
                 h_ext_cart=zero3               # external field f(t) [A/m]
                 ):
        """Init Method."""
        # parameters

        # random seed
        # deprecated, use RandomState instead
        # np.random.seed(seed)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        print(f'[seed] seed initialized: {seed}')

        self.hk_mode = hk_mode
        print(f'\t[debug] hk_mode: {hk_mode}')

        # thermal
        self.do_thermal = do_thermal
        self.do_theta_windowing = do_theta_windowing
        self.theta_windowing = theta_windowing
        self.theta_windowing_factor = theta_windowing_factor
        self.do_fake_thermal = do_fake_thermal
        self.d_theta_fake_th = d_theta_fake_th
        if not self.do_thermal and self.do_theta_windowing:
            # Convergence issues if not properly initialized
            print('WARNING Convergence issues if not properly initialized')

        # achieved switching time
        self.switching_t = np.inf

        # current
        self.i_amp_fn = i_amp_fn
        self.t_i_idx = 0
        # temperature
        self.temperature = temperature
        # self.temperature_nom = 25         # [Celsius] not used,
        # conductance not implemented
        # dimensions
        self.t_fl = t_fl
        self.t_ox = t_ox
        self.w_fl = w_fl
        self.l_fl = l_fl
        self._area = np.pi/4*self.w_fl*self.l_fl
        self._volume = self._area*self.t_fl

        # magnetic properties
        self.alpha = alpha
        self._U0_gamma_0_alpha = c_U0 * c_gamma_0 / (1 + self.alpha*self.alpha)
        self.ms = ms
        # Full anisotropy:
        # Watanabe, K.,
        # Shape anisotropy revisited in single-digit nanometer magnetic
        # tunnel junctions.
        # Nature Communications, 9(1), 5–10.
        # https://doi.org/10.1038/s41467-018-03003-7
        self.k_i = k_i_0
        # we consider just interfacial term
        self.k_u = self.k_i / self.t_fl
        self._init_shape_anisotropy_n(shape_ani_demag_mode, shape_ani_demag_n)

        self.do_a_exchange = do_a_exchange
        self.a_exchange = a_exchange
        if a_exchange <= 0:
            print('[warning] A energy exchange disabled')
            self.a_exchange = 0.
        # Thermal Stability
        # initial h_k with approximate m_cart = [0, 0, +/-1]
        if m_init is not None:
            z_0 = m_init[2]
        elif theta_init is not None:
            z_0 = np.cos(theta_init)
        else:
            if state == 'P':
                z_0 = 1
            else:
                z_0 = -1
        print(f'z_0: {z_0}')
        h_k = 2 * self.k_i * np.abs(z_0) / (self.t_fl * c_U0 * self.ms)
        if self.hk_mode == 0:
            h_k += -self.ms * \
                (self.shape_ani_demag_n[2] -
                 self.shape_ani_demag_n[0]) * np.abs(z_0)
        else:
            h_k += -self.ms*(self.shape_ani_demag_n[2]) * np.abs(z_0)
        # print(f'fer: hk: {h_k}')
        # self.m_cart_init = [0, 0, z_0]
        # self.theta_0 = np.arccos(z_0)
        # print(f'fer: hkb: {self._get_h_k_eff()}')
        self.thermal_stability = c_U0 * self.ms * h_k * self._volume / (
            2 * c_KB * self.temperature)

        # 'internal parameters'
        # Internal parameters are computed here for efficiency
        # directly from the OOMMF reference in Purdue's manual

        # For the STT term
        # for out of plane torque
        self.eps_prime = eps_prime  # [FITTING] constant
        self._init_stt_constants(stt_mode=stt_mode,
                                 # oommf full mode
                                 lambda_pl=lambda_pl,
                                 lambda_fl=lambda_fl,
                                 p_pl=p_pl,
                                 p_fl=p_fl,
                                 # oommf_simple mode
                                 lambda_s=lambda_s,
                                 p=p,
                                 # simple mode
                                 # p=p, # shared with oommf_simple
                                 nabla_mult=nabla_mult)

        # vcma
        self.do_vcma = do_vcma
        self.xi = xi

        # pinned layer magnetization vector, p
        # used in spin torque transfer
        self.theta_pl = theta_pl
        self.phi_pl = phi_pl

        # thermal noise:
        self._init_thermal_constants()

        # initialize magnetic vectors
        self._init_magnetic_vectors(m_init, theta_init, phi_init, state)

        # external field
        self.h_ext_cart = h_ext_cart

        # conductance
        self.r_p = r_p
        self.tmr = 2*self.p*self.p/(1-self.p*self.p)
        print(f'\t[debug] p: {self.p}')
        print(f'\t[debug] tmr: {self.tmr}')

        # debug ic, taud
        self._get_Ic(debug=True)
        self._get_tau_d(debug=True)

        # time handling
        self.last_t = 0.0
        self.last_dt = 1e-12

    def _init_shape_anisotropy_n(self,
                                 shape_ani_demag_mode,
                                 shape_ani_demag_n):
        """Init shape anisotropy constants."""
        self.shape_ani_demag_n = shape_ani_demag_n
        if shape_ani_demag_mode == 0:
            self.shape_ani_demag_n = np.zeros(3)
        elif shape_ani_demag_mode == 1:
            if np.sum(shape_ani_demag_n) != 1.:
                print('[warning] sum(shape_ani_demag_n) != 1,'
                      f'sum: {np.sum(shape_ani_demag_n)}')
            self.shape_ani_demag_n = shape_ani_demag_n
        elif shape_ani_demag_mode == 2:
            # ferromagnetic cylinder
            # Sato, M., & Ishii, Y. (1989).
            # Simple and approximate expressions of demagnetizing
            # factors of uniformly magnetized rectangular
            # rod and cylinder.
            # Journal of Applied Physics, 66(2), 983–985.
            # https://doi.org/10.1063/1.343481
            r = np.sqrt(self.w_fl*self.l_fl)/2
            nz = 1/(2*self.t_fl/(r*np.sqrt(np.pi)) + 1)
            self.shape_ani_demag_n = np.array([
                (1-nz)/2,
                (1-nz)/2,
                nz])
        elif shape_ani_demag_mode == 3:
            # Zhang, K., et al.
            # Compact Modeling and Analysis of Voltage-Gated
            # Spin-Orbit Torque Magnetic Tunnel Junction.
            # IEEE Access, 8, 50792–50800.
            # https://doi.org/10.1109/ACCESS.2020.2980073
            self.shape_ani_demag_n = np.array([
                np.pi*self.t_fl / (4*np.sqrt(self.w_fl*self.l_fl)),
                np.pi*self.t_fl / (4*np.sqrt(self.w_fl*self.l_fl)),
                1 - 2 * np.pi*self.t_fl/(4*np.sqrt(self.w_fl*self.l_fl))])
        else:
            print('[error] shape_ani_demag_n not supported')
            exit(1)
        self.shape_ani_demag_n = np.array(self.shape_ani_demag_n)
        print(f'\t[debug] shape_ani mode {shape_ani_demag_mode}. '
              f'shape_ani_n: {self.shape_ani_demag_n}')

    def _init_stt_constants(self,
                            stt_mode,
                            # oommf full mode
                            lambda_pl,
                            lambda_fl,
                            p_pl,
                            p_fl,
                            # oommf_simple mode
                            lambda_s,
                            p,
                            # simple mode
                            # p=p, # shared with oommf_simple
                            nabla_mult):
        """
        Initialize STT constants following OOMMF.

        a) OOMMFC, where lambda_pl != lambda_fl
        see full equation at
        https://kelvinxyfong.wordpress.com/research/research-interests
        /oommf-extensions/oommf-extension-xf_thermspinxferevolve/
        b) OOMMFC, where lambda_pl == lambda_fl == lambda
        c) stt_simplest:
        Khvalkovskiy, A. V., et. al.
        Basic principles of STT-MRAM cell operation in memory arrays.
        Journal of Physics D: Applied Physics, 46(7), 074001.
        https://doi.org/10.1088/0022-3727/46/7/074001
        """
        self.stt_mode = stt_mode
        self._stt_scale_mult = 1    # 1 matching OOMMF
        self._stt_scale_fac = self._stt_scale_mult * \
            c_hbar / (c_U0 * c_E * self.ms * self._volume)
        if self.stt_mode == 'stt_oommf_full':
            self.lambda_pl = lambda_pl
            self.lambda_fl = lambda_fl
            _lamb_PL2 = lambda_pl*lambda_pl
            _lamb_FL2 = lambda_fl*lambda_fl
            self._ap = np.sqrt(_lamb_PL2+1)*(_lamb_FL2+1)
            self._am = np.sqrt(_lamb_PL2-1)*(_lamb_FL2-1)
            self._ap2 = (_lamb_PL2+1)*(_lamb_FL2+1)
            self._am2 = (_lamb_PL2-1)*(_lamb_FL2-1)

            self._kp = p_pl*_lamb_PL2 * \
                np.sqrt((_lamb_FL2+1)/(_lamb_PL2+1)) + p_fl * \
                _lamb_FL2*np.sqrt((_lamb_PL2-1)/(_lamb_FL2-1))
            self._km = p_pl*_lamb_PL2 * \
                np.sqrt((_lamb_FL2+1)/(_lamb_PL2+1)) - p_fl * \
                _lamb_FL2*np.sqrt((_lamb_PL2-1)/(_lamb_FL2-1))
        elif self.stt_mode == 'stt_oommf_simple':
            self.lambda_s = lambda_s
            self.p = p
            self._eps_simple_num = p*lambda_s*lambda_s
            self._eps_simple_den0 = lambda_s*lambda_s + 1
            self._eps_simple_den1 = lambda_s*lambda_s - 1
        elif self.stt_mode == 'stt_simplest':
            self._nabla_mult = nabla_mult   # epsilon = nabla/2
            # nabla = _nabla_mult * P/(1+/-P^2)
            # so epsilon = _nabla_mult/2 * P/(1+/-P^2)
            # see model code, different research groups
            # use different factors for SMTJ
            self.p = p
        else:
            print('[ERROR] stt_mode should be '
                  '"stt_oommf_full" | "stt_oommf_simple" | "stt_simplest"'
                  f' Provided: {self.stt_mode}')
            raise Exception

    def _init_thermal_constants(self):
        """Init thermal constants.

        Equation from newest Purdue's paper
        Torunbalci, M. M., et. al (2018).
        Modular Compact Modeling of MTJ Devices.
        IEEE Transactions on Electron Devices, 65(10), 4628–4634.
        https:#doi.org/10.1109/TED.2018.2863538
        and
        De Rose, R., et al.
        A Variation-Aware Timing Modeling Approach for Write Operation
        in Hybrid CMOS/STT-MTJ Circuits.
        IEEE Transactions on Circuits and Systems I: Regular Papers,
        65(3), 1086–1095.
        https:#doi.org/10.1109/TCSI.2017.2762431
        Please,
        note it is different from:
        Ament, S., Rangarajan, N., Parthasarathy, A., & Rakheja, S. (2016).
        Solving the stochastic Landau-Lifshitz-Gilbert-Slonczewski equation
        for monodomain nanomagnets : A survey and analysis
        of numerical techniques. 1–19.
        http://arxiv.org/abs/1607.04596

        We also include the (1+alpha^2) effect from
        Lee, H., Lee, A., Wang, S., Ebrahimi, F., Gupta,
        P., Khalili Amiri, P., & Wang, K. L. (2018).
        Analysis and Compact Modeling of Magnetic Tunnel Junctions
        Utilizing Voltage-Controlled Magnetic Anisotropy.
        IEEE Transactions on Magnetics, 54(4).
        https://doi.org/10.1109/TMAG.2017.2788010

        Also, note that the field generated later is
        sqrt(2 alpha temp Kb / ((1+alpha^2) u0 gamma Ms V dt))
        """
        # Units: A/m -> (note gamma = c_gamma_0*c_U0)
        # sqrt[ (J/K) * K / / ((H/m) * (rad/s/T) * (H/m) * (A/m) * (m^3)) ]
        # = A/m sqrt[s / rad])
        # which is later multiplied by sqrt[1/s] so the final
        # sigma_th = A/m / sqrt(rad)
        th_gamma = c_gamma_0 * c_U0 / (1 + self.alpha*self.alpha)
        self.th_power_noise_std = np.sqrt(
            2. * self.alpha * self.temperature * c_KB /
            (c_U0 * th_gamma * self.ms * self._volume))

        # for the non-normalized field:
        # Units: sqrt(J/((Henry/m)*m*m*m)) = A/m
        self.sigma_th = np.sqrt(
            2. * self.alpha * self.temperature * c_KB /
            (c_U0 * self._volume))
        # for the normalized field:
        # Units: sqrt(J/((Henry/m)*(A/m)*(A/m)*m*m*m)) = 1
        # self.sigma_th = np.sqrt(
        #     2. * self.alpha * self.temperature * c_KB /
        #     (c_U0 * self.ms * self.ms * self._volume))

    def _init_magnetic_vectors(self, m_init, theta_init,
                               phi_init, state):
        """
        Initialize magnetic vectors.

        we could add 1ns of stabilization before injecting current
        or, by adding noise with gaussian/maxwell-boltzmann distribution,
        where the second moment gives
        the thermal average (theta_0 square)

        # Maxwell-Boltzmann distribution info:
        a) Switching distributions for perpendicular spin-torque devices
           within the macrospin approximation.
           IEEE Transactions on Magnetics, 48(12), 4684–4700.
           https://doi.org/10.1109/TMAG.2012.2209122
        b) Khvalkovskiy, A. V., et. al.
           Basic principles of STT-MRAM cell operation in memory arrays.
           Journal of Physics D: Applied Physics, 46(7), 074001.
           https://doi.org/10.1088/0022-3727/46/7/074001

        theta_0 can be given by 1/sqrt(2*delta) (most common aproach)
        a) Sun, J. Z. (2006).
           Spin angular momentum transfer in current-perpendicular
           nanomagnetic junctions. IBM Journal of Research and Development
           https://doi.org/10.1147/rd.501.0081<
           Butler, W. H., et al.
        b) Switching distributions for perpendicular spin-torque devices
           within the macrospin approximation.
           IEEE Transactions on Magnetics, 48(12), 4684–4700.
           https://doi.org/10.1109/TMAG.2012.2209122

        or 1/sqrt(delta)
        c) Khvalkovskiy, A. V., et al.
           Basic principles of STT-MRAM cell operation in memory arrays.
           Journal of Physics D: Applied Physics, 46(7), 074001.
           https://doi.org/10.1088/0022-3727/46/7/074001
        """
        # initial angles under no variability
        print(f'\t[debug] thermal_stability: {self.thermal_stability}')
        self.theta_0 = np.sqrt(1/(2*self.thermal_stability))
        print(f'\t[debug] theta_0: {self.theta_0}, or {np.pi-self.theta_0}')
        print(f'\t[debug] m_init: {m_init}')
        if m_init is not None:
            print(f'\t[debug] m_init given: {m_init}')
            _, self.theta_init, self.phi_init = spherical_from_cart_np(m_init)
            if np.isnan(self.theta_init) or self.theta_init == 0:
                self.theta_init = 0.01
            if np.isnan(self.phi_init):
                self.phi_init = 0.01
        elif theta_init is None:
            # initial free layer angle
            #  mean_angle = maxwell.stats(
            #      moments='m',
            #      loc=-self.theta_0*loc_theta_0_coeff,
            #      scale=self.theta_0,
            #  )
            # _mean = np.sqrt(2/np.pi)*2*self.theta_0 \
            #         - self.theta_0*loc_theta_0_coeff
            _mode = np.sqrt(2)*self.theta_0 - self.theta_0*loc_theta_0_coeff
            if state == 'P':
                self.theta_init = _mode
            elif state == 'AP':
                self.theta_init = np.pi - _mode
            else:
                print(f'[error] invalid state [allowed: P|AP]: {state}')
                return None
            # phi_init
            if phi_init is None:
                self.phi_init = self.rng.uniform(0.0, 2*np.pi)
            else:
                self.phi_init = phi_init
        else:
            self.theta_init = theta_init
            # phi_init
            if phi_init is None:
                self.phi_init = self.rng.uniform(0.0, 2*np.pi)
            else:
                self.phi_init = phi_init

        # for windowing
        self.theta_min = self.theta_windowing_factor * self.theta_0
        # force min theta even when passed initial params
        # in case it is out of boundaries
        if self.do_theta_windowing:
            if (self.theta_init > np.pi/2 and
                    np.pi-self.theta_init < self.theta_min):
                self.theta_init = np.pi-self.theta_min
            elif self.theta_init < self.theta_min:
                self.theta_init = self.theta_min

        # debug
        print(f'\t[debug] initial theta_init : {self.theta_init}')
        print(f'\t[debug] initial phi_init : {self.phi_init}')

        # initial angle variability,
        # f noise is present and an initial angle has not been passed
        # we allow random noise from outside via theta_init
        if self.do_thermal and theta_init is None:
            # get from maxwell distribution
            # generate an object as the np seed might be overrriden
            _maxwell_rv = maxwell(
                loc=-self.theta_0*loc_theta_0_coeff,
                scale=self.theta_0)
            _maxwell_rv.random_state = self.rng
            # np.random.RandomState(seed=self.seed)
            rand_angle = _maxwell_rv.rvs(size=1)[0]
            # while(rand_angle < 0):
            #     rand_angle = maxwell.rvs(
            #         loc=-self.theta_0*loc_theta_0_coeff,
            #         size=1,
            #         scale=self.theta_0)[0]
            # if self.seed % 2 == 1:
            #     rand_angle = -rand_angle
            print(f'\t[debug] rand_angle: {rand_angle}')
            # other distributions, should them be desired
            # rand_inc = np.random.normal(0.,
            #                             0.1/np.sqrt(2*self.thermal_stability),
            #                             1)[0]
            if (self.theta_init > np.pi/2 and
                    np.pi-rand_angle < self.theta_0):
                self.theta_init = np.pi-rand_angle
            elif rand_angle < self.theta_0:
                self.theta_init = rand_angle
            print(f'\t[debug] theta_init after random: {self.theta_init}')

        # init vectors
        self.m_sph_init = np.array([1, self.theta_init, self.phi_init])
        self.m_cart_init = cart_from_spherical_fn(self.m_sph_init)
        self.p_cart = np.array([np.sin(self.theta_pl)*np.cos(self.phi_pl),
                                np.sin(self.theta_pl)*np.sin(self.phi_pl),
                                np.cos(self.theta_pl)])
        print(f'\t[debug] constructor m_cart_init : {self.m_cart_init}')
        print(f'\t[debug] constructor m_shp_init : {self.m_sph_init}')

    def _get_Ic(self, debug=False):
        """Get I critical with no VCMA."""
        eps_0 = self.get_epsilon_stt(np.dot(
            self.m_cart_init, self.p_cart))
        # effective out of plane component
        hk_eff = self._get_h_k_eff()
        # nabla is 2 eps
        ic = 2*self.alpha*c_E*c_U0*hk_eff*self.ms*self._volume/(c_hbar*eps_0*2)
        if debug:
            print(f'\t[debug] eps_0: {eps_0}')
            print(f'\t[debug] hk_eff: {hk_eff}')
            print(f'\t[debug] Ic: {ic}')
        return ic

    def _get_h_k_eff(self, debug=True):
        """Get H_k initial with no VCMA."""
        # effective out of plane component
        m_0 = np.array(self.m_cart_init)
        # h_k_eff = self.get_h_demag(m_0)[2] + self.get_h_una(m_0)[2]
        # if debug:
        #     print(f'\t[debug] H_k: {h_k_eff}')
        if self.hk_mode == 0:
            k_u_eff = self.k_i / self.t_fl - \
                0.5*c_U0 * (
                    self.shape_ani_demag_n[2] -
                    self.shape_ani_demag_n[0])*self.ms*self.ms
        else:
            k_u_eff = self.k_i / self.t_fl - \
                0.5*c_U0 * (self.shape_ani_demag_n[2])*self.ms*self.ms
        h_k_eff = 2 * k_u_eff * m_0[2] / (c_U0 * self.ms)

        # m_0[2] = np.cos(self.theta_0)
        # h_k_eff = self.get_h_demag(m_0)[2] + self.get_h_una(m_0)[2]

        return h_k_eff

    def _get_tau_d(self, debug=False):
        """Get H_k initial with no VCMA."""
        hk_eff = self._get_h_k_eff()
        tau_d = (1 + self.alpha*self.alpha)/(
            self.alpha * c_gamma_0 * c_U0 * hk_eff)
        if debug:
            print(f'\t[debug] hk_eff: {hk_eff}')
            print(f'\t[debug] tau_d: {tau_d}')
        return tau_d

    def get_epsilon_stt(self, mdp):
        """Compute OOMMF epsilon term based on mdp vector.

        a) OOMMFC, where lambda_pl != lambda_fl
        b) OOMMFC, where lambda_pl == lambda_fl == lambda
        c) stt_simplest:
        Khvalkovskiy, A. V., et. al.
        Basic principles of STT-MRAM cell operation in memory arrays.
        Journal of Physics D: Applied Physics, 46(7), 074001.
        https://doi.org/10.1088/0022-3727/46/7/074001
        """
        if self.stt_mode == 'stt_oommf_full':
            if (self.lambda_pl == 1.0) or (self.lambda_fl == 1.0):
                return 0.5 * self.p
            else:
                return self._kp / (self._ap + self._am * mdp) + \
                    self._km / (self._ap - self._am * mdp)
        elif self.stt_mode == 'stt_oommf_simple':
            return self._eps_simple_num / (
                self._eps_simple_den0 + self._eps_simple_den1 * mdp)
        elif self.stt_mode == 'stt_simplest':
            return self._nabla_mult/2*self.p/(1 + self.p * self.p * mdp)
        print('[error] Non-valid stt_mode "stt_oommf_full" | '
              '"stt_oommf_simple" | "stt_simplest"')
        return None

    ###################
    # H_ext components
    ###################

    def get_h_exchange(self):
        """
        Get energy exchange field.

        https://www.iue.tuwien.ac.at/phd/makarov/dissertationch5.html
        """
        # for now return 0, (grad^2 * m)
        if not self.do_a_exchange:
            return 0.
        return 2*self.a_exchange/(c_U0 * self.ms)

    def get_h_th(self, dt):
        """Get thermal var.

        Note that the field generated is a Brownian motion problem.
        Brownian motion can be simulated realizing that
        dW = W_j - W_{j-1} ~ N(0, dt) = sqrt(dt) N(0, 1)
        We return sigma/sqrt(dt) as that term will later be multiplied
        by dt by the solver, and dW = sigma*sqrt(dt).
        """
        if not self.do_thermal or dt == np.inf:
            return np.zeros(3)
        # _rnd = self.rng.normal(0.,
        #                        self.th_power_noise_std / np.sqrt(dt),
        #                        3)
        _rnd = self.rng.normal(0.,
                               1,
                               3)
        _rnd = normalize_cart(_rnd)
        return _rnd * self.th_power_noise_std / np.sqrt(dt)

    def get_h_demag(self, m_cart):
        """Get H_demag field vector due to shape anisotropy in the FL.

        Zhang, K., et. al.
        Compact Modeling and Analysis of Voltage-Gated Spin-Orbit
        Torque Magnetic Tunnel Junction.
        IEEE Access, 8, 50792–50800.
        https://doi.org/10.1109/ACCESS.2020.2980073

        Full anisotropy:
        Watanabe, K.,
        Shape anisotropy revisited in single-digit nanometer magnetic
        tunnel junctions.
        Nature Communications, 9(1), 5–10.
        https://doi.org/10.1038/s41467-018-03003-7
        """
        # given [nx, ny, xz]
        return -self.ms*self.shape_ani_demag_n * m_cart

    def get_h_una(self, m_cart):
        """Get uniaxial anisotropy vector.

        We consider interfacial PMA anisotropy.
        The geometry defines it.
        See Figure 2 at
        Khvalkovskiy, A. V., et. al.
        Basic principles of STT-MRAM cell operation in memory arrays.
        Journal of Physics D: Applied Physics, 46(7), 074001.
        https://doi.org/10.1088/0022-3727/46/7/074001

        Full anisotropy:
        Watanabe, K.,
        Shape anisotropy revisited in single-digit nanometer magnetic
        tunnel junctions.
        Nature Communications, 9(1), 5–10.
        https://doi.org/10.1038/s41467-018-03003-7
        """
        # vector u_anisotropy == unitary z
        # # uniaxial aniotropy constant in J/m^3
        # self.k_u = self.thermal_stability * self.temperature * \
        #     c_KB / (self._volume)
        # print(f'\t[debug] k_u: {self.k_u}')
        # print(f'\t[debug] thermal stability: {self.thermal_stability}')
        # return np.array([0.,
        #                  0.,
        #                  2 * self.k_u * m_cart[2] / (c_U0 * self.ms)
        #                  ])
        return np.array([0.,
                         0.,
                         2 * self.k_i * m_cart[2] /
                         (self.t_fl * c_U0 * self.ms)
                         ])

    def get_h_vcma(self, v_mtj, m_cart):
        """Get VCMA  vector.

        Zhang, K., et. al.
        Compact Modeling and Analysis of Voltage-Gated Spin-Orbit
        Torque Magnetic Tunnel Junction.
        IEEE Access, 8, 50792–50800.
        https://doi.org/10.1109/ACCESS.2020.2980073
        """
        #  note instead of -2 * p_xi...
        #  that is as the voltage v_mtj refers PL/FL voltage,
        #  and not FL/PL voltage as in the paper references

        # Torunbalci, M. et.al.
        # Modular Compact Modeling of MTJ Devices.
        # IEEE Transactions on Electron Devices, 65(10), 4628–4634.
        # https://doi.org/10.1109/TED.2018.2863538
        # where the PL is considered the positive terminal

        # WORKAROUND to validate vcma, see oommf validation
        # return np.array([0.,
        #                  0.,
        #                  2 * self.xi * self.i_amp*self.r_p * m_cart[2] / (
        #                      self.t_fl * self.t_ox * c_U0 * self.ms)
        #                  ])
        if not self.do_vcma:
            return np.zeros(3)
        return np.array([0.,
                         0.,
                         2 * self.xi * v_mtj * m_cart[2] / (
                             self.t_fl * self.t_ox * c_U0 * self.ms)
                         ])

    #######################
    # conductance
    #######################
    def get_r(self, m_cart):
        """Get conductance.

        Julliere, M. (1975).
        Tunneling between ferromagnetic films.
        Physics Letters A, 54(3), 225–226.
        https://doi.org/10.1016/0375-9601(75)90174-7

        Lee, H., et.al.
        Analysis and Compact Modeling of Magnetic Tunnel Junctions Utilizing
        Voltage-Controlled Magnetic Anisotropy.
        IEEE Transactions on Magnetics, 54(4).
        https://doi.org/10.1109/TMAG.2017.2788010
        """
        return self.r_p*(1 + self.tmr/(self.tmr + 2))/(
            1 - self.tmr/(self.tmr + 2)*m_cart[2])

    #######################
    # differential vectors
    #######################

    def get_diff_unit_theta_vector(self, m_sph):
        """Compute dt given infinitesimal displacement from ro, theta, phi."""
        return np.array([np.cos(m_sph[1])*np.cos(m_sph[2]),
                         np.cos(m_sph[1])*np.sin(m_sph[2]),
                         -np.sin(m_sph[1])
                         ])

    def get_diff_unit_phi_vector(self, m_sph):
        """Compute dp given infinitesimal displacement from ro, theta, phi."""
        return np.array([-np.sin(m_sph[2]),
                         np.cos(m_sph[2]),
                         0
                         ])

    def _g_cart(self, m_cart, v_cart):
        """
        Compute g(mt, _v) wienner process.

        g is also commonly called b (under a b nomenclature).
        """
        mxv = np.cross(m_cart, v_cart)
        mxmxv = np.cross(m_cart, mxv)

        return self._U0_gamma_0_alpha * (-mxv + self.alpha*(-mxmxv))

    def _g_sph(self, m_sph, v_cart):
        """
        Compute g(mt, _v) wienner process.

        g is also commonly called b (under a b nomenclature).
        """
        # m_cart = spherical_from_cart_np(m_sph)
        # mxv = np.cross(m_cart, v_cart)
        # mxmxv = np.cross(m_cart, mxv)

        # h_eff_phi (local orthogonal unit vector in the direction
        #            of increasing phi), etc
        # differential terms from cart to spherical
        diff_unit_theta_vector = self.get_diff_unit_theta_vector(m_sph)
        diff_unit_phi_vector = self.get_diff_unit_phi_vector(m_sph)
        d_h_th_theta = np.dot(v_cart, diff_unit_theta_vector)
        d_h_th_phi = np.dot(v_cart, diff_unit_phi_vector)

        # as specified in OOMMF, gamma([(rad)/s/T]) should interface
        # the fields in A/m, so introducing u0
        d_theta = self._U0_gamma_0_alpha * (
            d_h_th_phi + self.alpha*(d_h_th_theta))
        d_phi = self._U0_gamma_0_alpha / np.sin(m_sph[1]) * (
            -d_h_th_theta + self.alpha*d_h_th_phi)

        # dmdt is [dro, dthetadt, dphidt]
        return np.asarray([0, d_theta, d_phi], dtype=np.float64)

    def _f_sph(self, t, m_sph, dt=np.inf):
        """
        Compute dm_sph/dt fn.

        LLG equation (with secondary stt term) at eq 1.3
        Switching distributions for perpendicular spin-torque devices
        within the macrospin approximation.
        IEEE Transactions on Magnetics, 48(12), 4684–4700.
        https://doi.org/10.1109/TMAG.2012.2209122
        see OOMMF for the value of the different terms
        and
        Ament, S., et al.
        Solving the stochastic Landau-Lifshitz-Gilbert-Slonczewski
        equation for monodomain nanomagnets :
        A survey and analysis of numerical techniques.
        1–19. Retrieved from http://arxiv.org/abs/1607.04596
        """
        # check integrity
        if np.isnan(t):
            print('[ERROR] time is NaN')
            raise Exception
        # update variables
        _t = t
        _dt = dt
        # reverse time normalization
        # as i_amp is in natural time [s]
        if self.solve_normalized:
            _t /= self.ms * c_U0 * c_gamma_0
            _dt /= self.ms * c_U0 * c_gamma_0

        # m, Is, mxp and mxpxm vectors
        m_cart = cart_from_spherical_fn(m_sph)

        # Fn defining the current
        i_amp = self.i_amp_fn(_t)
        r = self.get_r(m_cart)
        v_mtj = i_amp * r

        # m*p, mxp and mxpxm
        # note that self.i_amp is signed, and that Is is in Z axis
        mdp = np.dot(m_cart, self.p_cart)
        # not used as the terms are directly computed in spherical coord.
        # mxp = uxv_cart(m_cart, self.p_cart)
        # mxpxm = uxv_cart(mxp, m_cart)

        # add extra additional terms to the effective field:
        # different anisotropy terms, demag, etc
        h_eff_cart = np.array(self.h_ext_cart(_t))
        h_eff_cart += self.get_h_demag(m_cart)
        h_eff_cart += self.get_h_exchange()
        h_eff_cart += self.get_h_una(m_cart)
        h_eff_cart += self.get_h_vcma(v_mtj, m_cart)

        # wienner process
        # similar to what SPICE solvers do
        if self.do_thermal and _dt != np.inf:
            h_eff_cart += self.get_h_th(_dt)

        #  stt component
        epsilon = self.get_epsilon_stt(mdp)
        # beta computation as in OOMMF
        # beta units are [A/m]
        beta = i_amp * self._stt_scale_fac

        # do normalized sLLGS
        if self.solve_normalized:
            beta /= self.ms
            h_eff_cart /= self.ms
            self._U0_gamma_0_alpha = 1

        # for oomffc validation
        # self.ut = self.beta * self.epsilon
        # both eps and eps prime contributions
        # h_stt components in [A/m]
        h_stt_1_cart = beta * (epsilon * self.p_cart)
        h_stt_2_cart = beta * (self.eps_prime * self.p_cart)

        # h_eff_phi (local orthogonal unit vector in the direction
        #            of increasing phi), etc
        # differential terms from cart to spherical
        diff_unit_theta_vector = self.get_diff_unit_theta_vector(m_sph)
        diff_unit_phi_vector = self.get_diff_unit_phi_vector(m_sph)
        d_h_eff_theta = np.dot(h_eff_cart, diff_unit_theta_vector)
        d_h_eff_phi = np.dot(h_eff_cart, diff_unit_phi_vector)
        d_h_stt_1_theta = np.dot(h_stt_1_cart, diff_unit_theta_vector)
        d_h_stt_1_phi = np.dot(h_stt_1_cart, diff_unit_phi_vector)
        d_h_stt_2_theta = np.dot(h_stt_2_cart, diff_unit_theta_vector)
        d_h_stt_2_phi = np.dot(h_stt_2_cart, diff_unit_phi_vector)

        # as specified in OOMMF, gamma([(rad)/s/T]) should interface
        # the fields in A/m, so introducing u0
        # OOMMF like, not normalized
        d_theta = self._U0_gamma_0_alpha * (
            d_h_eff_phi + d_h_stt_1_theta - d_h_stt_2_phi +
            self.alpha*(
                d_h_eff_theta - d_h_stt_1_phi - d_h_stt_2_theta))
        d_phi = self._U0_gamma_0_alpha / np.sin(m_sph[1]) * (
            -d_h_eff_theta + d_h_stt_1_phi + d_h_stt_2_theta +
            self.alpha*(
                d_h_eff_phi + d_h_stt_1_theta - d_h_stt_2_phi))

        # fake theta term
        if (not self.do_thermal) and self.do_fake_thermal:
            # extra term can be seen as a direct contribution to h_phi
            # as d_theta/dt dependence on h_th is
            # ~ h_th_phi + alpha * h_th_theta
            t_th = self._U0_gamma_0_alpha * \
                self.th_power_noise_std / np.sqrt(_dt)
            if self.solve_normalized:
                t_th /= self.ms
            contrib = -self.d_theta_fake_th*np.sign(m_sph[1]-np.pi/2)*t_th
            d_theta += contrib
        # saturate d_theta to emulate min field generated
        # by the thermal noise, avoiding total damping
        if (not self.do_thermal) and self.do_theta_windowing and dt > 0:
            d_theta *= self.theta_windowing(d_theta=d_theta,
                                            theta=m_sph[1],
                                            theta_min=self.theta_min)

        # dmdt is [dro, dthetadt, dphidt]
        return np.asarray([0, d_theta, d_phi], dtype=np.float64)

    def _f_cart(self, t, m_cart, dt=np.inf):
        """
        Compute dm_cart/dt fn.

        Targets the deterministic component for Heun method.
        LLG equation (with secondary stt term) at eq 1.3
        Switching distributions for perpendicular spin-torque devices
        within the macrospin approximation.
        IEEE Transactions on Magnetics, 48(12), 4684–4700.
        https://doi.org/10.1109/TMAG.2012.2209122
        see OOMMF for the value of the different terms
        and
        Ament, S., et al.
        Solving the stochastic Landau-Lifshitz-Gilbert-Slonczewski
        equation for monodomain nanomagnets :
        A survey and analysis of numerical techniques.
        1–19. Retrieved from http://arxiv.org/abs/1607.04596
        """
        # check integrity
        if np.isnan(t):
            print('[ERROR] time is NaN')
            raise Exception
        # update variables
        _t = t
        _dt = dt
        # reverse time normalization
        # as i_amp is in natural time [s]
        if self.solve_normalized:
            _t /= self.ms * c_U0 * c_gamma_0
            _dt /= self.ms * c_U0 * c_gamma_0

        # Fn defining the current
        i_amp = self.i_amp_fn(_t)
        r = self.get_r(m_cart)
        v_mtj = i_amp * r

        # m*p, mxp and mxpxm
        # note that self.i_amp is signed, and that Is is in Z axis
        mdp = np.dot(m_cart, self.p_cart)
        mxp = np.cross(m_cart, self.p_cart)
        mxmxp = np.cross(m_cart, mxp)

        # not used as the terms are directly computed in spherical coord.
        # mxp = uxv_cart(m_cart, self.p_cart)
        # mxpxm = uxv_cart(mxp, m_cart)

        # add extra additional terms to the effective field:
        # different anisotropy terms, demag, etc
        h_eff_cart = np.array(self.h_ext_cart(_t))
        h_eff_cart += self.get_h_demag(m_cart)
        h_eff_cart += self.get_h_exchange()
        h_eff_cart += self.get_h_una(m_cart)
        h_eff_cart += self.get_h_vcma(v_mtj, m_cart)

        # wienner process
        # similar to what SPICE solvers do
        if self.do_thermal and _dt != np.inf:
            h_eff_cart += self.get_h_th(_dt)

        #  stt component
        # epsilon = self.get_epsilon_stt(mdp)
        # beta computation as in OOMMF
        # beta units are [A/m]
        beta = i_amp * self._stt_scale_fac

        # do normalized sLLGS
        if self.solve_normalized:
            beta /= self.ms
            h_eff_cart /= self.ms
            self._U0_gamma_0_alpha = 1

        # for oomffc validation
        # self.ut = self.beta * self.epsilon
        # both eps and eps prime contributions
        # h_stt components in [A/m]
        h_stt_1 = beta * self.get_epsilon_stt(mdp)
        h_stt_2 = beta * self.eps_prime

        # compute cross products
        mxh_eff = np.cross(m_cart, h_eff_cart)
        mxmxh_eff = np.cross(m_cart, mxh_eff)
        mxp_h_stt_1 = h_stt_1 * mxp
        mxp_h_stt_2 = h_stt_2 * mxp
        mxmxp_h_stt_1 = h_stt_1 * mxmxp
        mxmxp_h_stt_2 = h_stt_2 * mxmxp

        return self._U0_gamma_0_alpha * (
            -mxh_eff - mxmxp_h_stt_1 + mxp_h_stt_2
            + self.alpha*(-mxmxh_eff + mxp_h_stt_1 + mxmxp_h_stt_2))

    def solve_ode(self,
                  final_t,
                  scipy_ivp=True,
                  solve_spherical=True,
                  solve_normalized=False,
                  method='RK45',
                  # preferred method to control accuracy:
                  rtol=1e-3,
                  atol=1e-6,
                  # other non-preferred way:
                  max_step=np.inf):
        """
        Integrate a dm_sph/dt fn.

        #######################
        Non stochastic methods:
        #######################
        a) Scipy solve_ivp (RK45, ...) by using "scipy_ivp=True"
        b) Other methods by using "scipy_ivp=True"
        Horley, P., et. al. (2011).
        Numerical Simulations of Nano-Scale Magnetization Dynamics.
        Numerical Simulations of Physical and Engineering Processes.
        https://doi.org/10.5772/23745

        #####################
        SDE methods:
        #####################
        a) Horley, P., et. al. (2011).
        Numerical Simulations of Nano-Scale Magnetization Dynamics.
        Numerical Simulations of Physical and Engineering Processes.
        https://doi.org/10.5772/23745
        b) Ament, S., Rangarajan, N., Parthasarathy, A., & Rakheja, S. (2016).
        Solving the stochastic Landau-Lifshitz-Gilbert-Slonczewski equation
        for monodomain nanomagnets : A survey and analysis
        of numerical techniques. 1–19.
        http://arxiv.org/abs/1607.04596
        #####################
        """
        # solved normalized sLLGS or OOMMF Oxs_SpinXferEvolve
        self.solve_normalized = solve_normalized

        # max_tstep_saved
        if max_step < 1e-12:
            saved_max_step = 1e-12
        else:
            saved_max_step = max_step

        # max_step/tstep checks
        if not scipy_ivp and max_step is None or max_step == np.inf:
            print('[error] Max step should be specified when not using '
                  'scipy_ivp mode')
            return None
        if not scipy_ivp and not solve_spherical and max_step > 1e-13:
            print(f'[warning] max_step {max_step} might '
                  'not be sufficiently small, use <=0.5e-13')
        elif not scipy_ivp and solve_spherical and max_step > 1e-12:
            print(f'[warning] max_step {max_step} might '
                  'not be sufficiently small, use <=1e-12')
        method_info = method
        if solve_spherical:
            _f = self._f_sph
            _g = self._g_sph
            y0 = self.m_sph_init
            method_info += ', spherical coordinates'
        else:
            _f = self._f_cart
            _g = self._g_cart
            y0 = self.m_cart_init
            method_info = ', cartesian coordinates'

        save_every = int(saved_max_step/max_step)
        # normalization
        # do time normalization
        if self.solve_normalized:
            final_t *= self.ms * c_U0 * c_gamma_0
            max_step *= self.ms * c_U0 * c_gamma_0
            saved_max_step *= self.ms * c_U0 * c_gamma_0
            self._U0_gamma_0_alpha = 1.
            method_info += ', normalized time'
        else:
            self._U0_gamma_0_alpha = c_U0 * \
                c_gamma_0 / (1 + self.alpha*self.alpha)
            method_info += ', not time normalized'

        if (self.do_thermal or self.do_fake_thermal) and scipy_ivp:
            print('[info] Calling custom scipy fn for solver...')
            # Application of the monkeypatch to replace rk_step
            # with the custom behavior (expose h to _f)
            base.check_arguments = custom_ode.check_arguments_custom
            base.OdeSolver.__init__ = custom_ode.ode_solver_init_custom
            rk.rk_step = custom_ode.rk_step_custom

        # solve ode using RK from scipy (altered solver if noise present)
        # Default under no Wienner process
        # can be forced to match SPICE like solvers
        if scipy_ivp:
            print(f'[info][solver] solve_ivp. Method: {method_info}')
            print(f'[info] rtol: {rtol} atol: {atol} max_step:{max_step}')
            if self.do_thermal:
                print('[warning][solver] You are using scipy ivp solver.'
                      'For the stochastic simulations, '
                      'the use of a SDE solver is encouraged. '
                      'See "scipy_ivp" parameter.')

            # time normalization already done
            sol = solve_ivp(fun=_f,
                            # lambda t,
                            # y: _f(t, y, a),
                            t_span=[0, final_t],
                            # t_eval=np.linspace(0, final_t, 10000),
                            y0=y0,
                            method=method,
                            # method='RK23'
                            # method='RK45'
                            # method='DOP853'
                            # ...
                            # preferred method to control accuracy:
                            rtol=rtol,
                            atol=atol,
                            # other (non-preferred) way
                            max_step=max_step,
                            )
            # restart variables
            self.t_i_idx = 0
            # y0 is restarted when solve_ivp is called
            # renormalize time
            if self.solve_normalized:
                sol.t /= self.ms * c_U0 * c_gamma_0
            # return solution
            return sol

        print(f'[info][solver] Integration Method: {method_info}')
        print(f'[info][solver] max_step (dt):{max_step}')

        # time normalization already done
        n_t = int(final_t/max_step)
        dt = max_step
        # saved vals
        t_eval = np.linspace(0, final_t, int(final_t/saved_max_step))
        m_eval = np.zeros((t_eval.shape[0], 3))
        # computed_vals
        sqrt_dt = np.sqrt(dt)
        m_eval[0] = y0
        _m = y0
        _m_next = _m
        t = 0
        t_idx = 0
        save_idx = 0

        th_gamma = c_gamma_0 * c_U0 / (1 + self.alpha*self.alpha)
        _sig = np.sqrt(
            2. * self.alpha * self.temperature * c_KB /
            (c_U0 * th_gamma * self.ms * self._volume))
        v_cart = np.array([_sig, _sig, _sig])

        # different options
        while t_idx < n_t:
            _dW = self.rng.normal(loc=0, scale=1, size=(3)) * sqrt_dt
            if method == 'naive_euler':
                _m_next = _m + dt * _f(t, _m)
            elif method == 'heun':
                _dm = _f(t, _m)
                _m_prime = _m + dt * _dm
                if not solve_spherical:
                    _m_prime = normalize_cart(_m_prime)
                _m_next = _m + 0.5*dt*(_dm + _f(t, _m_prime))
            elif method == 'rk45':
                if solve_spherical:
                    _f1 = _f(t, _m)
                    _f2 = _f(t + dt / 2.0, _m + dt * _f1 / 2.0)
                    _f3 = _f(t + dt / 2.0, _m + dt * _f2 / 2.0)
                    _f4 = _f(t + dt, _m + dt * _f3)
                else:
                    _f1 = _f(t, _m)
                    _m1 = normalize_cart(_m + 0.5*dt * _f1)
                    _f2 = _f(t + 0.5*dt, _m1)
                    _m2 = normalize_cart(_m + 0.5*dt * _f2)
                    _f3 = _f(t + 0.5*dt, _m2)
                    _m3 = normalize_cart(_m + dt * _f3)
                    _f4 = _f(t + dt, _m3)
                _m_next = _m + dt * \
                    (_f1 + 2.0 * _f2 + 2.0 * _f3 + _f4) / 6.0
            elif method == 'stratonovich_heun':
                # Stratonovich Heun's
                _dm = _f(t, _m, dt=np.inf)
                _dg = _g(_m, v_cart=v_cart)
                _m_prime = _m + dt * _dm + _dg * _dW
                if not solve_spherical:
                    _m_prime = normalize_cart(_m_prime)
                _m_next = _m + \
                    0.5 * dt * (_dm + _f(t, _m_prime, dt=np.inf)) +\
                    0.5 * _dW * (_dg + _g(_m_prime, v_cart=v_cart))
            elif method == 'stratonovich_rk_weak_2':
                # RK 2 Stratonovich
                _dm = _f(t, _m, dt=np.inf)
                _dg = _g(_m, v_cart=v_cart)
                _m_prime = _m + 2/3*(dt * _dm + _dg * _dW)
                if not solve_spherical:
                    _m_prime = normalize_cart(_m_prime)
                _m_next = _m + dt*(0.25*_dm + 0.75*_f(2/3*dt + t, _m_prime)) \
                    + _dW*(0.25*_dg + 0.75*_g(_m_prime, v_cart=v_cart))
            else:
                print(f'[error] method "{method}" not recogniced '
                      'for the custom solver. '
                      'Use "naive_euler", "heun", "rk45",'
                      '"stratonovich_heun", "stratonovich_rk_weak_2"')
                return None
            # cartesians require normalization
            if not solve_spherical:
                _m_next = normalize_cart(_m_next)
            # update vars
            _m = _m_next
            t += dt
            t_idx += 1
            # save
            if (t_idx % save_every) == 0:
                m_eval[save_idx] = _m
                save_idx += 1

        if not solve_spherical:
            # conversion
            m_eval = spherical_from_cart_np(m_eval)
        # reverse time normalization
        if self.solve_normalized:
            t_eval /= self.ms * c_U0 * c_gamma_0
        sol = Sol()
        sol.t = t_eval
        sol.y = m_eval.T
        return sol

    def state_plotter(self,
                      times,
                      states,
                      h_ext_cart,
                      i_amp,
                      theta_0,
                      title,
                      plot_xy=True,
                      plot_simple=False):
        """Plot magnetic state/evolution."""
        if plot_simple:
            # \t[debug]
            fig = plt.figure(figsize=plt.figaspect(1.))
            xyz = cart_from_spherical_fn(states.T)
            ax_2d = fig.add_subplot(1, 1, 1)
            if plot_xy:
                ax_2d.plot(1e9*times, xyz[:, 0], ':', label='x')
                ax_2d.plot(1e9*times, xyz[:, 1], ':', label='y')
            ax_2d.plot(1e9*times, xyz[:, 2], label='z')
            ax_2d.set_ylabel('m/|m|')
            ax_2d.set_xlabel('time (ns)')
            ax_2d.legend()
            ax_2d.set_ylim([-1, 1])
            ax_2d.grid()
            plt.title(title)
            plt.show()
            return

        # states = states.T
        # states = cart_from_spherical_fn(states)
        # states = spherical_from_cart_np(states)
        # states = states.T

        fig = plt.figure(figsize=plt.figaspect(3.))
        # Plot h_ext_z and i_amp.
        # Recalculating these just to plot, which is a bit of duplication.
        # But avoids touching the solver code and ensures identical
        # timestamps to solver.
        # Not using append() because it's very inefficient in numpy.
        h_ext_pts = np.zeros(len(times))
        i_amp_pts = np.zeros(len(times))
        for i in range(len(times)):
            h_ext_pts[i] = h_ext_cart(times[i])[2]
            i_amp_pts[i] = i_amp(times[i])
        ax_h = fig.add_subplot(2, 2, 1)
        ax_h.plot(1e9*times, h_ext_pts)
        ax_h.set_ylabel('h_ext_z (A/m)')
        ax_h.grid()
        ax_i = fig.add_subplot(2, 2, 2)
        ax_i.plot(1e9*times, 1e6*i_amp_pts)
        ax_i.set_ylabel('i_amp (uA)')
        ax_i.grid()

        ax_theta = fig.add_subplot(2, 2, 3)
        ax_theta.plot(1e9*times, states[1], label='theta')
        ax_theta.plot(1e9*times, np.abs(states[1]), label='abs_theta')
        ax_theta.plot(1e9*times, theta_0*np.ones(times.shape), label='theta_0')
        # non-uniform time axis, integrate and divide
        if times[-1] != times[0]:
            theta_mean = np.trapz(
                np.abs(states[1]), times)/(times[-1] - times[0])
            print(f'\t[debug] mean theta: {theta_mean}')
            ax_theta.plot(1e9*times, theta_mean*np.ones(times.shape),
                          label='theta_mean')
        ax_theta.set_ylabel('theta [rad]')
        ax_theta.legend()
        ax_theta.grid()

        # Data for three-dimensional scattered points
        xyz = cart_from_spherical_fn(states.T)
        ax_2d = fig.add_subplot(2, 2, 4)
        if plot_xy:
            ax_2d.plot(1e9*times, xyz[:, 0], ':', label='x')
            ax_2d.plot(1e9*times, xyz[:, 1], ':', label='y')
        ax_2d.plot(1e9*times, xyz[:, 2], label='z')
        ax_2d.set_ylabel('m/|m|')
        ax_2d.set_xlabel('time (ns)')
        ax_2d.legend()
        ax_2d.set_ylim([-1, 1])
        ax_2d.grid()

        # Now plot polar 3D plot
        fig3d = plt.figure(figsize=plt.figaspect(1.))
        ax_3d = fig3d.add_subplot(1, 1, 1, projection='3d')
        ax_3d.plot3D(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        ax_3d.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                        c=times, cmap='Greens', alpha=0.2)
        ax_3d.set_xlim([-1, 1])
        ax_3d.set_ylim([-1, 1])
        ax_3d.set_zlim([-1, 1])
        ax_3d.set(xlabel='X',
                  ylabel='Y',
                  zlabel='Z')
        plt.title(title)
        plt.show()

    def solve_and_plot(self,
                       final_t=40e-9,
                       scipy_ivp=True,
                       solve_spherical=True,
                       solve_normalized=False,
                       method='RK45',
                       # preferred method to control accuracy:
                       rtol=1e-3,
                       atol=1e-6,
                       # other method (non-preferred)
                       max_step=np.inf,
                       plot=True,
                       plot_simple=True):
        """Solve and plot ODE."""
        t_start = time.time()
        ode = self.solve_ode(final_t,
                             scipy_ivp=scipy_ivp,
                             solve_spherical=solve_spherical,
                             solve_normalized=solve_normalized,
                             method=method,
                             rtol=rtol,
                             atol=atol,
                             max_step=max_step,
                             )
        if ode is None:
            print('[error] an error occurred while computing the ode')
            return
        t_end = time.time()
        print(f'[info] solver took {t_end-t_start} [s]')
        title = f'{final_t*1e9} ns. Solver: {method}, '
        if solve_spherical:
            title += ' spherical coords, '
        else:
            title += ' cartesian coords, '
        if scipy_ivp:
            title += ' Scipy solver, '
        else:
            title += ' custom solver, '
        title += f'max_step: {max_step} s'
        if plot:
            self.state_plotter(title=title,
                               times=ode.t,
                               states=ode.y,
                               h_ext_cart=self.h_ext_cart,
                               i_amp=self.i_amp_fn,
                               theta_0=self.theta_0,
                               plot_simple=plot_simple)
        return ode
