#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of casiopeia.
#
# Copyright 2014-2016 Adrian Bürger, Moritz Diehl
#
# casiopeia is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# casiopeia is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with casiopeia. If not, see <http://www.gnu.org/licenses/>.

import numpy as np

from ..interfaces import casadi_interface as ci
from discretization import Discretization

from .. import inputchecks

class ODECollocation(Discretization):

    @property
    def number_of_controls(self):

        return self.number_of_intervals
        

    def __set_collocation_settings(self, number_of_collocation_points, \
        collocation_scheme):

        self.number_of_collocation_points = number_of_collocation_points
        self.collocation_scheme = collocation_scheme

        self.collocation_points = ci.collocation_points( \
            self.number_of_collocation_points, self.collocation_scheme)
        self.collocation_polynomial_degree = len(self.collocation_points) - 1


    def __set_optimization_variables(self):

        self.optimization_variables = {key: ci.dmatrix(0, self.number_of_intervals) \
            for key in ["P", "V", "X", "EPS_U", "U", "Q"]}

        if self.system.nu != 0:

            self.optimization_variables["U"] = ci.mx_sym("U", self.system.nu, \
                self.number_of_intervals)

        if self.system.nx != 0:

            # Attention! Way of ordering has changed! Consider when
            # reapplying collocation and multiple shooting!
            self.optimization_variables["X"] = ci.mx_sym("X", self.system.nx, \
                (self.collocation_polynomial_degree + 1) * \
                self.number_of_intervals + 1)


        if self.system.neps_u != 0:
                
            self.optimization_variables["EPS_U"] = \
                ci.mx_sym("EPS_U", self.system.neps_u, \
                self.number_of_intervals)


        self.optimization_variables["P"] = ci.mx_sym("P", self.system.np)

        self.optimization_variables["V"] = ci.mx_sym("V", self.system.nphi, \
            self.number_of_intervals + 1)

        if self.system.nq != 0:
            self.optimization_variables["Q"] = ci.mx_sym("Q", self.system.nq)
        else:
            self.optimization_variables["Q"] = ci.dmatrix(0, 1)
        

    def __compute_collocation_time_points(self):

        self.__T = np.zeros((self.collocation_polynomial_degree + 1, \
            self.number_of_intervals))

        for k in range(self.number_of_intervals):

            for j in range(self.collocation_polynomial_degree + 1):

                self.__T[j,k] = self.time_points[k] + \
                    (self.time_points[k+1] - self.time_points[k]) * \
                    self.collocation_points[j]


    def __compute_collocation_coefficients(self):
    
        # Coefficients of the collocation equation

        self.__C = np.zeros((self.collocation_polynomial_degree + 1, \
            self.collocation_polynomial_degree + 1))

        # Coefficients of the continuity equation

        self.__D = np.zeros(self.collocation_polynomial_degree + 1)

        # Dimensionless time inside one control interval

        tau = ci.sx_sym("tau")

        # For all collocation points

        for j in range(self.collocation_polynomial_degree + 1):

            # Construct Lagrange polynomials to get the polynomial basis
            # at the collocation point
            
            L = 1
            
            for r in range(self.collocation_polynomial_degree + 1):
            
                if r != j:
            
                    L *= (tau - self.collocation_points[r]) / \
                        (self.collocation_points[j] - \
                            self.collocation_points[r])
    

            lfcn = ci.sx_function("lfcn", [tau],[L])
          
            # Evaluate the polynomial at the final time to get the
            # coefficients of the continuity equation

            self.__D[j] = lfcn([1])

            # Evaluate the time derivative of the polynomial at all 
            # collocation points to get the coefficients of the
            # collocation equation
            
            tfcn = lfcn.tangent()

            for r in range(self.collocation_polynomial_degree + 1):

                self.__C[j,r] = tfcn([self.collocation_points[r]])[0]


    def __initialize_ode_right_hand_side(self):

        self.__ffcn = ci.mx_function("ffcn", \
            [self.system.u, self.system.q, \
            self.system.p, self.system.x, \
            self.system.eps_u], [self.system.f])


    def __compute_collocation_nodes(self):

        h = ci.mx_sym("h", 1)

        u = self.system.u
        q = self.system.q
        p = self.system.p

        x = ci.mx_sym("x", self.system.nx * \
            (self.collocation_polynomial_degree + 1))
        eps_u = self.system.eps_u

        collocation_node = ci.vertcat([ \

            h * self.__ffcn.call([ \

                u, q, p, \

                x[j*self.system.nx : (j+1)*self.system.nx], \
                eps_u])[0] - \

                sum([self.__C[r,j] * x[r*self.system.nx : (r+1)*self.system.nx] \

                    for r in range(self.collocation_polynomial_degree + 1)]) \
                    
                        for j in range(1, self.collocation_polynomial_degree + 1)])


        collocation_node_fcn = ci.mx_function("coleqnfcn", \
            [h, u, q, x, eps_u, p], [collocation_node])
        collocation_node_fcn = collocation_node_fcn.expand()

        X = self.optimization_variables["X"][:, :-1].reshape( \
            (self.system.nx * (self.collocation_polynomial_degree + 1), \
            self.number_of_intervals))

        EPS_U = self.optimization_variables["EPS_U"][:].reshape( \
            (self.system.neps_u, self.number_of_intervals))

        [self.__collocation_nodes] = collocation_node_fcn.map([ \
            np.atleast_2d((self.time_points[1:] - self.time_points[:-1])), \
            self.optimization_variables["U"], self.optimization_variables["Q"], \
             X, EPS_U, self.optimization_variables["P"]])


    def __compute_continuity_nodes(self):

        x = ci.mx_sym("x", self.system.nx * \
            (self.collocation_polynomial_degree + 1))
        x_next = ci.mx_sym("x_next", self.system.nx)

        continuity_node = x_next - sum([self.__D[r] * \
            x[r*self.system.nx : (r+1)*self.system.nx] \
            for r in range(self.collocation_polynomial_degree + 1)])

        continuity_node_fcn = ci.mx_function("continuity_node_fcn", \
            [x_next, x], [continuity_node])
        continuity_node_fcn = continuity_node_fcn.expand()

        X_NEXT = self.optimization_variables["X"][:, \
            (self.collocation_polynomial_degree + 1) :: \
            (self.collocation_polynomial_degree + 1)]

        X = self.optimization_variables["X"][:, :-1].reshape( \
            (self.system.nx * (self.collocation_polynomial_degree + 1), \
            self.number_of_intervals))

        [self.__continuity_nodes] = continuity_node_fcn.map([X_NEXT, X])


    def __set_nlp_equality_constraints(self):

        self.equality_constraints = \
            ci.veccat([self.__collocation_nodes, self.__continuity_nodes])


    def __apply_discretization_method(self):

        self.__set_optimization_variables()

        self.__compute_collocation_time_points()
        self.__compute_collocation_coefficients()

        self.__initialize_ode_right_hand_side()
        self.__compute_collocation_nodes()
        self.__compute_continuity_nodes()
        self.__set_nlp_equality_constraints()


    def __evaluate_measurement_function(self):

        phifcn = ci.mx_function("phifcn", \
            [self.system.u, self.system.q, self.system.x, \
                self.system.eps_u, self.system.p], \
            [self.system.phi])
        phifcn = phifcn.expand()

        # The last control value is silently reused. This should be changed
        # or at least the user should be noticed about that!

        measurement_function_input = [ \

            ci.horzcat([self.optimization_variables["U"], \
                self.optimization_variables["U"][:, -1]]), 
            self.optimization_variables["Q"], \
            
            self.optimization_variables["X"][:, \
                :: (self.collocation_polynomial_degree + 1)],

            ci.horzcat([self.optimization_variables["EPS_U"],
                self.optimization_variables["EPS_U"][:, -1]]),

            self.optimization_variables["P"]
        ]

        [self.measurements] = phifcn.map(measurement_function_input)


    def __discretize(self, number_of_collocation_points, collocation_scheme):

        self.__set_collocation_settings(number_of_collocation_points, \
            collocation_scheme)
        self.__apply_discretization_method()

        self.__evaluate_measurement_function()


    def __init__(self, system, time_points, number_of_collocation_points = 3, \
        collocation_scheme = "radau"):

        super(ODECollocation, self).__init__(system, time_points)

        self.__discretize(number_of_collocation_points, collocation_scheme)
