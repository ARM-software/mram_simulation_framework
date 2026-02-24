#!/usr/bin/env python

# Copyright (c) 2020-2021 Arm Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
MTJ Fokker-Plank Finite Volume Method Solver.

Fokker-Plank or advection-diffusion for
MTJ magnetization probability evolution.
"""

import fvm_lib.fvm_classes as fvm
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

np.random.seed(seed=1)


def solve_mtj_fp(
    dim_points=1000,
    rho_init=None,
    delta=60,
    i0=1.5,
    h=0.0,
    t_step=0.001,
    T=10,
    lin_space_z=False,
    discretization="exponential",  # exponential, upwind, central
    do_3d=True,
):
    """Solve FVM for the FPE on a perpendicular symetric MTJ."""
    # cranck-nickolson
    theta = 0.5
    # fully implicit
    # theta = 1.

    # Dirichlet boundary conditions
    # right_value = 1.0
    # left_value = 1.0
    # Neumann boundary conditions
    left_flux = 0.0
    right_flux = 0.0

    if lin_space_z:
        # linear in z
        faces = np.linspace(-1, 1, dim_points)
    else:
        # linear in theta
        faces = np.cos(np.linspace(np.pi, 0, dim_points))
    mesh = fvm.Mesh(faces)

    # drift/advection/convection
    U = (i0 - h - mesh.cells) * (1 - mesh.cells * mesh.cells)
    # diffusion
    D = (1 - mesh.cells * mesh.cells) / (2 * delta)

    # drift/advection/convection
    a = fvm.CellVariable(-U, mesh=mesh)
    # diffusion
    d = fvm.CellVariable(D, mesh=mesh)
    # Source term
    # s[int(np.median(range(mesh.J)))] = 0.0

    # Initial conditions
    if rho_init is None:
        # w_init = np.exp(
        #     -delta*(1-mesh.cells*mesh.cells))*np.heaviside(mesh.cells, 0.5)
        _theta_x = np.arccos(mesh.cells)
        rho_init = np.exp(-delta * np.sin(_theta_x) * np.sin(_theta_x)) * np.heaviside(
            mesh.cells, 0.5
        )
        rho_init /= np.trapezoid(rho_init, x=mesh.cells)
    # w_init = w_init[::-1]
    print(f"\trho_init area: {np.trapezoid(rho_init, x=mesh.cells)}")

    model = fvm.AdvectionDiffusionModel(
        faces, a, d, t_step, discretization=discretization
    )
    # model.set_boundary_conditions(left_value=1., right_value=0.)
    model.set_boundary_conditions(left_flux=left_flux, right_flux=right_flux)
    M = model.coefficient_matrix()
    alpha = model.alpha_matrix()
    beta = model.beta_vector()
    identity = sparse.identity(model.mesh.J)
    print(f"\talpha: [{np.min(alpha.todense())}, {np.max(alpha.todense())}]")
    print(f"\tbeta: [{np.min(beta)}, {np.max(beta)}]")
    print(f"\tidentity: {identity.shape}")

    # Construct linear system from discretised matrices, A.x = d
    A = identity - t_step * theta * alpha * M
    d = (identity + t_step * (1 - theta) * alpha * M) * rho_init + beta

    print(
        "\tPeclet number", np.min(model.peclet_number()), np.max(model.peclet_number())
    )
    print(
        "\tCFL condition", np.min(model.CFL_condition()), np.max(model.CFL_condition())
    )

    rho = rho_init

    t0 = np.linspace(0, T, int(T / t_step) + 1)
    if do_3d:
        rho = np.zeros((t0.shape[0], mesh.cells.shape[0]))
        area = np.zeros(t0.shape[0])
        rho[0] = rho_init
        area[0] = np.trapezoid(rho[0], x=mesh.cells)
        for i in range(1, t0.shape[0]):
            d = (identity + t_step * (1 - theta) * alpha * M) * rho[i - 1] + beta
            rho[i] = linalg.spsolve(A, d)
            # normalization not needed, flux is kept
        # PS/PNS
        ps = np.trapezoid(rho.T[mesh.cells < 0], x=mesh.cells[mesh.cells < 0], axis=0)
        t_sw = t0[np.argmax(ps > 0.5)]
    else:
        rho = np.array(rho_init)
        rho_next = np.array(rho_init)
        t_sw = 0
        for i in range(1, t0.shape[0]):
            d = (identity + t_step * (1 - theta) * alpha * M) * rho + beta
            rho_next = linalg.spsolve(A, d)
            # normalization not needed, flux is kept
            ps = np.trapezoid(
                rho.T[mesh.cells < 0], x=mesh.cells[mesh.cells < 0], axis=0
            )
            if t_sw == 0 and ps > 0.5:
                t_sw = t_step * i
            # update variable by switching
            rho_next, rho = rho, rho_next

    # return
    return {"t_sw": t_sw, "rho": rho.T, "t0": t0, "z0": mesh.cells}


def simple_test():
    """FVM simple test."""
    z_points = 500
    delta = 60
    i0 = 1.5
    h = 0.0
    # time step
    t_step = 0.001
    T = 20

    data = solve_mtj_fp(
        dim_points=z_points, delta=delta, i0=i0, h=h, t_step=t_step, T=T
    )
    rho = data["rho"]
    t0 = data["t0"]
    z0 = data["z0"]
    _theta_x = np.arccos(z0)
    t_mesh0, z_mesh0 = np.meshgrid(t0, z0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # z_mesh0 = np.arccos(z_mesh0)
    ax.plot_surface(z_mesh0, t_mesh0, rho, alpha=0.7, cmap="viridis", edgecolor="k")
    plt.show()

    print("plotting 2d")
    fig = plt.figure()
    ax_0 = fig.add_subplot(211)
    ax_1 = fig.add_subplot(212)

    fixed_list = [0, 1, 2, 4, 8, 14, T]
    for tt_idx, tt in enumerate(fixed_list):
        t0_idx = np.argmax(t0 >= tt)
        if tt == t0[-1]:
            t0_idx = -1
        ax_0.plot(_theta_x, rho[:, t0_idx], label=f"t={tt}")
        ax_1.plot(z0, rho[:, t0_idx], label=f"t={tt}")
    ax_0.legend()
    ax_0.set_yscale("log", base=10)
    ax_1.legend()
    ax_1.set_yscale("log", base=10)
    # ax.set_ylim(bottom=1e-10)
    ax_0.grid()
    ax_1.grid()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # colors
    name = "tab10"
    cmap = cm.get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors  # type: list
    # plot_3d_evolution(p_imp_0, t0, z0, plot_res=1e6,
    #                   title='ro implicit matmult')
    pt = np.trapezoid(rho, z0, axis=0)
    ps = np.trapezoid(y=rho[z0 < 0], x=z0[z0 < 0], axis=0) / pt
    pns = np.trapezoid(y=rho[z0 >= 0], x=z0[z0 >= 0], axis=0) / pt

    ax.plot(t0, ps, "--", color=colors[0], alpha=0.5, label=f"ps i: {i0}")
    ax.plot(t0, pns, label=f"pns i: {i0}")
    ax.set_yscale("log", base=10)
    # ax.set_ylim(bottom=1e-10)
    ax.legend()
    ax.grid()
    plt.show()


if __name__ == "__main__":
    simple_test()
