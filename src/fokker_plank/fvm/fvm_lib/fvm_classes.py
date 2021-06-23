#!/usr/bin/env python
# coding: utf-8


# Copyright (c) 2021. Daniel J. Farrell
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# adapted from Daniel J Farrell notes no FVM
# danieljfarrell.github.io/FVM/overview.html


"""Finite Volume Method Classes."""

import numpy as np
from scipy import sparse
from scipy.sparse import dia_matrix


# Supporting functions
def check_index_within_bounds(i, min_i, max_i):
    """Check that the index (number or an iterable) is within the range."""
    return np.logical_and(i >= min_i, i <= max_i).all()


class Mesh(object):
    """A 1D cell centered mesh defined by faces for the FVM."""

    def __init__(self, faces):
        """Init method."""
        super(Mesh, self).__init__()

        # Check for duplicated points
        if len(faces) != len(set(faces)):
            raise ValueError(
                "The faces array contains duplicated positions."
                "No cell can have zero volume so please"
                " update with unique face positions.")
        self.faces = np.array(faces)
        self.cells = 0.5 * (self.faces[0:-1] + self.faces[1:])
        self.J = len(self.cells)
        self.cell_widths = (self.faces[1:] - self.faces[0:-1])

    def h(self, i):
        """Return the width of the cell at the specified index."""
        return self.cell_widths[i]

    def hm(self, i):
        """Distance between centroids in the backwards direction."""
        if not check_index_within_bounds(i, 1, self.J-1):
            raise ValueError("hm index runs out of bounds")
        return (self.cells[i] - self.cells[i-1])

    def hp(self, i):
        """Distance between centroids in the forward direction."""
        if not check_index_within_bounds(i, 0, self.J-2):
            raise ValueError("hp index runs out of bounds")
        return (self.cells[i+1] - self.cells[i])


class CellVariable(np.ndarray):
    """
    Representation of a variable defined at the cell centers.

    Provides interpolation functions to calculate the value at cell faces.
    """
    # http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    def __new__(cls, input_array, mesh=None):
        """Init like method."""
        # If `input_array` is actually just a constant
        # convert it to an array of len the number of cells.
        try:
            len(input_array)
        except TypeError:
            input_array = input_array*np.ones(len(mesh.cells))

        obj = np.asarray(input_array).view(cls)
        obj.mesh = mesh
        return obj

    def __array_finalize__(self, obj):
        """Build like method."""
        if obj is None:
            return
        self.mesh = getattr(obj, 'mesh', None)
        self.__get_items__ = getattr(obj, '__get_items__', None)

    def m(self, i):
        """
        Linear interpolation of the cell value at the right hand face .

        i.e. along the _m_inus direction.
        """
        return (self.mesh.h(i)/(2*self.mesh.hm(i))*self[i-1]
                + self.mesh.h(i-1)/(2*self.mesh.hm(i))*self[i])

    def p(self, i):
        """
        Linear interpolation of the cell value at the right hand face.

        i.e. along the _p_lus direction.
        """
        return (self.mesh.h(i+1)/(2*self.mesh.hp(i))*self[i]
                + self.mesh.h(i)/(2*self.mesh.hp(i))*self[i+1])


class AdvectionDiffusionModel(object):
    """A model for the advection-diffusion equation."""

    def __init__(self, faces, a, d, k, discretization="central"):
        """Init."""
        super(AdvectionDiffusionModel, self).__init__()

        self.mesh = Mesh(faces)
        self.a = CellVariable(a, mesh=self.mesh)
        self.d = CellVariable(d, mesh=self.mesh)
        self.t_step = k
        self.discretization = discretization

        # Check Peclet number
        mu = self.peclet_number()
        mu_max = np.max(np.abs(mu))
        if mu_max >= 1.5 and mu_max < 2.0:
            print(f'\n\nThe Peclet number is {mu_max},'
                  ' this is getting close to the limit of mod 2.'
                  '\nINCREASE precision on spatial axis')
            # warnings.warn('The Peclet number is %g,'
            #               ' this is getting close to the limit of mod 2.')
        elif mu_max > 2:
            print(f'\n\nThe Peclet number {mu_max} has exceeded the maximum'
                  ' value of mod 2 for the central discretization scheme.'
                  '\nINCREASE precision on spatial axis')
            # warnings.warn(
            #     'The Peclet number %g has exceeded the maximum'
            #     ' value of mod 2 for the central discretization scheme.',
            #     mu_max)

        # Check CFL condition
        CFL = self.CFL_condition()
        CFL_max = np.max(np.abs(CFL))
        if CFL_max > 0.5 and CFL_max < 1.0:
            print(f'\n\n[WARNING] The CFL condition is {CFL_max}',
                  'it is getting close to the upper limit.'
                  '\nINCREASE precision on time axis')
            # warnings.warn('[WARNING] The CFL condition is %g',
            #               'it is getting close to the upper limit.',
            #               CFL_max)
        elif np.max(np.abs(CFL)) > 1:
            print(f'\n\n[WARNING] The CFL condition is {CFL_max},'
                  'and has gone above the upper limit.'
                  '\nINCREASE precision on time axis')
            # warnings.warn('[WARNING] The CFL condition is %g,'
            #               'and has gone above the upper limit.',
            #               CFL_max)

        if discretization == 'exponential':
            self.kappa = (np.exp(mu) + 1)/(np.exp(mu) - 1) - 2/mu
            self.kappa[np.where(mu == 0.0)] = 0
            self.kappa[np.where(np.isposinf(mu))] = 1
            self.kappa[np.where(np.isneginf(mu))] = -1
        elif discretization == 'upwind':
            kappa_neg = np.where(self.a < 0, -1, 0)
            kappa_pos = np.where(self.a > 0, 1, 0)
            self.kappa = kappa_neg + kappa_pos
        elif discretization == 'central':
            self.kappa = np.zeros(self.mesh.J)
        else:
            print('Please set "discretization" to one of the following:'
                  '"upwind", "central" or "exponential"')

        # Artificially modify the diffusion coefficient
        # to introduce adpative discretization
        self.d = self.d + 0.5 * self.a * self.mesh.cell_widths * self.kappa
        print(f'kappa min:{np.min(self.kappa)}, max:{np.max(self.kappa)}')
        # print(f'kappa: {self.kappa}')

    def peclet_number(self):
        """Get Peclet number."""
        return self.a * self.mesh.cell_widths / self.d

    def CFL_condition(self):
        """Get CFL condition."""
        return self.a * self.t_step / self.mesh.cell_widths

    def set_boundary_conditions(self,
                                left_flux=None,
                                right_flux=None,
                                left_value=None,
                                right_value=None):
        """
        Boundary conditions.

        Make sure this function is used
        sensibly otherwise the matrix will be ill posed.
        """
        self.left_flux = left_flux
        self.right_flux = right_flux
        self.left_value = left_value
        self.right_value = right_value

    def _interior_matrix_elements(self, i):
        """Set interior coefficients for matrix equation."""
        def ra(i, a, d, m):
            return 1./m.h(i) * (a.m(i)*m.h(i)/(2*m.hm(i)) + d.m(i)/m.hm(i))

        def rb(i, a, d, m):
            return 1./m.h(i)*(
                a.m(i)*m.h(i-1)/(2*m.hm(i)) -
                a.p(i)*m.h(i+1)/(2*m.hp(i)) - d.m(i)/m.hm(i) - d.p(i)/m.hp(i))

        def rc(i, a, d, m):
            return 1./m.h(i) * (-a.p(i)*m.h(i)/(2*m.hp(i)) + d.p(i)/m.hp(i))

        return (ra(i, self.a, self.d, self.mesh),
                rb(i, self.a, self.d, self.mesh),
                rc(i, self.a, self.d, self.mesh))

    def _neumann_boundary_c_m_e_left(self):
        """Set Left hand side Neumann boundary conditions."""
        def b1(a, d, m):
            return 1./m.h(0) * (-a.p(0)*m.h(1)/(2*m.hp(0)) - d.p(0)/m.hp(0))

        def c1(a, d, m):
            return 1./m.h(0) * (-a.p(0)*m.h(0)/(2*m.hp(0)) + d.p(0)/m.hp(0))

        # Index and element value
        locations = [(0, 0), (0, 1)]
        values = (b1(self.a, self.d, self.mesh),
                  c1(self.a, self.d, self.mesh))
        return tuple([list(x) for x in zip(locations, values)])

    def _neumann_boundary_c_m_e_right(self, matrix=None):
        """Set right hand side Neumann boundary conditions."""
        def aJ(a, d, m):
            return 1./m.h(m.J-1)*(
                a.m(m.J-1) * m.h(m.J-1)/(2*m.hm(m.J-1))
                + d.m(m.J-1)/m.hm(m.J-1))

        def bJ(a, d, m):
            return 1./m.h(m.J-1)*(
                a.m(m.J-1) * m.h(m.J-2)/(2*m.hm(m.J-1))
                - d.m(m.J-1)/m.hm(m.J-1))
        # Index and element value
        J = self.mesh.J

        # Index and element value
        locations = [(J-1, J-2), (J-1, J-1)]
        values = (aJ(self.a, self.d, self.mesh),
                  bJ(self.a, self.d, self.mesh))
        return tuple([list(x) for x in zip(locations, values)])

    def _neumann_boundary_c_v_e_left(self):
        """Index and boundary cond vector elements for Neumann conditions."""
        location = [0]
        value = [self.left_flux/self.mesh.h(0)]
        return tuple([list(x) for x in zip(location, value)])

    def _neumann_boundary_c_v_e_right(self):
        """Index and boundary cond. vector elements for Neumann conditions."""
        location = [self.mesh.J-1]
        value = [-self.right_flux/self.mesh.h(self.mesh.J-1)]
        return tuple([list(x) for x in zip(location, value)])

    def _dirichlet_boundary_c_m_e_left(self):
        """Set left hand side Neumann boundary coefficients for matrix eq."""
        def rb(i, a, d, m):
            return 1./m.h(i)*(
                a.m(i)*m.h(i-1)/(2*m.hm(i))
                - a.p(i)*m.h(i+1)/(2*m.hp(i))
                - d.m(i)/m.hm(i)
                - d.p(i)/m.hp(i))

        def rc(i, a, d, m):
            return 1./m.h(i) * (-a.p(i)*m.h(i)/(2*m.hp(i)) + d.p(i)/m.hp(i))

        # Index and element value
        locations = [(0, 0), (0, 1)]
        # values = ( rb(0, self.a, self.d, self.mesh ),
        #            rc(0, self.a, self.d, self.mesh ) )
        values = (0,
                  1)
        return tuple([list(x) for x in zip(locations, values)])

    def _dirichlet_boundary_c_m_e_right(self):
        """Set right hand side Neumann boundary coefficients for matrix eq."""
        def ra(i, a, d, m):
            return 1./m.h(i) * (a.m(i)*m.h(i)/(2*m.hm(i)) + d.m(i)/m.hm(i))

        def rb(i, a, d, m):
            return 1./m.h(i)*(
                a.m(i)*m.h(i-1)/(2*m.hm(i))
                - a.p(i)*m.h(i+1)/(2*m.hp(i))
                - d.m(i)/m.hm(i)
                - d.p(i)/m.hp(i))
        J = self.mesh.J  # Index and element value

        # Index and element value
        locations = [(J-1, J-2), (J-1, J-1)]
        # values = ( ra(self.J-1, self.a, self.d, self.mesh ),
        #            rb(self.J-1, self.a, self.d, self.mesh ) )
        values = (0,
                  1)
        return tuple([list(x) for x in zip(locations, values)])

    def _dirichlet_boundary_c_v_e_left(self):
        """
        Index and boundary condition vector elements for Dirichlet conditions.

        NB these are always zero, unless BCs are time varying.
        """
        location = [0]
        value = [0]
        return tuple([list(x) for x in zip(location, value)])

    def _dirichlet_boundary_c_v_e_right(self):
        """
        Index and boundary condition vector elements for Dirichlet conditions.

        NB these are always zero, unless BCs are time varying.
        """
        location = [self.mesh.J-1]
        value = [0]
        return tuple([list(x) for x in zip(location, value)])

    def alpha_matrix(self):
        """
        Set alpha matrix.

        The alpha matrix is used to mask boundary conditions values
        for Dirichlet conditions. Otherwise for a fully Neumann (or Robin)
        system it is equal to the identity matrix.
        """
        a1 = 0 if self.left_flux is None else 1
        aJ = 0 if self.left_flux is None else 1
        diagonals = np.ones(self.mesh.J)
        diagonals[0] = a1
        diagonals[-1] = aJ
        return sparse.diags(diagonals, 0)

    def beta_vector(self):
        """Return the Neumann boundary condition vector."""
        b = np.zeros(self.mesh.J)

        if self.left_flux is not None:
            left_bc_elements = self._neumann_boundary_c_v_e_left()

        if self.right_flux is not None:
            right_bc_elements = self._neumann_boundary_c_v_e_right()

        if self.left_value is not None:
            left_bc_elements = self._dirichlet_boundary_c_v_e_left()

        if self.right_value is not None:
            right_bc_elements = self._dirichlet_boundary_c_v_e_right()

        bcs = left_bc_elements + right_bc_elements
        for inx, value in bcs:
            b[inx] = value
        return b

    def coefficient_matrix(self):
        """Return the coeff matrix which appears on the left hand side."""
        J = self.mesh.J
        # k = self.k
        # m = self.mesh
        # a = self.a
        # d = self.d

        # A element which is pushed off the
        # edge of the matrix by the spdiags function
        padding = np.array([0])
        # Yes, its the same. But this element
        # is included in the matrix (semantic difference).
        zero = padding
        # one = np.array([1])    #

        if self.left_flux is not None:
            left_bc_elements = self._neumann_boundary_c_m_e_left()

        if self.right_flux is not None:
            right_bc_elements = self._neumann_boundary_c_m_e_right()

        if self.left_value is not None:
            left_bc_elements = self._dirichlet_boundary_c_m_e_left()

        if self.right_value is not None:
            right_bc_elements = self._dirichlet_boundary_c_m_e_right()

        # Use the functions to layout the matrix Note that the boundary
        # condition elements are set to zero, they are filled in as
        # the next step.
        inx = np.array(list(range(1, J-1)))
        ra, rb, rc = self._interior_matrix_elements(inx)
        #                                 c1
        upper = np.concatenate([padding, zero, rc])

        #                          b1           bJ
        central = np.concatenate([zero, rb, zero])

        #                               aJ
        lower = np.concatenate([ra, zero, padding])

        A = sparse.spdiags([lower, central, upper], [-1, 0, 1], J, J).todok()

        # Apply boundary conditions elements
        bcs = left_bc_elements + right_bc_elements
        for inx, value in bcs:
            print(f'boundary conditions. A[{inx}]={value}')
            A[inx] = value
        return dia_matrix(A)
