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

class ODEMultipleShooting(Discretization):

    @property
    def number_of_controls(self):

        return self.number_of_intervals


    def __set_multiple_shooting_settings(self):

        pass


    def __set_optimization_variables(self):

        self.optimization_variables = {key: ci.dmatrix(0, self.number_of_intervals) \
            for key in ["P", "V", "X", "EPS_E", "EPS_U", "U", "Q"]}

        if self.system.nu != 0:

            self.optimization_variables["U"] = ci.mx_sym("U", self.system.nu, \
                self.number_of_intervals)

        if self.system.nx != 0:

            self.optimization_variables["X"] = ci.mx_sym("X", self.system.nx, \
                self.number_of_intervals + 1)
        

        if self.system.neps_u != 0:
                
            self.optimization_variables["EPS_U"] = \
                ci.mx_sym("EPS_U", self.system.neps_u, self.number_of_intervals)


        self.optimization_variables["P"] = ci.mx_sym("P", self.system.np)

        self.optimization_variables["V"] = ci.mx_sym("V", self.system.nphi, \
            self.number_of_intervals + 1)

        if self.system.nq != 0:
            self.optimization_variables["Q"] = ci.mx_sym("Q", self.system.nq)
        else:
            self.optimization_variables["Q"] = ci.dmatrix(0, 1)


    # def __initialize_ode_right_hand_side(self):

    #     self.__ffcn = ci.mx_function("ffcn", \
    #         [self.system.t, self.system.u, self.system.p, self.system.x, \
    #         self.system.eps_e, self.system.eps_u], [self.system.f])


    def __generate_scaled_ode(self):

        t_scale = ci.mx_sym("t_scale", 1)

        self.__ode_scaled = {"x": self.system.x, \
            "p": ci.vertcat([t_scale, \
                self.system.u, self.system.q, \
                self.system.p, self.system.eps_u]), \
            "ode": t_scale * self.system.f}


    def __compute_continuity_constraints(self):

        integrator = ci.Integrator("integrator", "rk", self.__ode_scaled, {"t0": 0, "tf": 1, "expand": True})

        params = ci.vertcat([np.atleast_2d(self.time_points[1:] - self.time_points[:-1]), \
            self.optimization_variables["U"], \
            ci.repmat(self.optimization_variables["Q"], 1, self.number_of_intervals), \
            ci.repmat(self.optimization_variables["P"], 1, self.number_of_intervals), \
            self.optimization_variables["EPS_U"]])

        shooting = integrator.map("shooting", "openmp", self.number_of_intervals)
        X_next = shooting(x0 = self.optimization_variables["X"][:,:-1], \
            p = params)["xf"]

        self.__continuity_constraints = \
            self.optimization_variables["X"][:, 1:] - X_next


    def __set_nlp_equality_constraints(self):

        self.equality_constraints = \
            ci.veccat([self.__continuity_constraints])


    def __apply_discretization_method(self):

        self.__set_optimization_variables()

        self.__generate_scaled_ode()

        self.__generate_scaled_ode()
        self.__compute_continuity_constraints()
        self.__set_nlp_equality_constraints()


    def __evaluate_measurement_function(self):

        phifcn = ci.mx_function("phifcn", \
            [self.system.u, self.system.q, self.system.x, \
                self.system.eps_u, self.system.p], \
            [self.system.phi])
        phifcn = phifcn.expand()

        # The last control value is silently reused. This should be changed
        # or at least the user should be noticed about that!)

        measurement_function_input = [ \

            ci.horzcat([self.optimization_variables["U"], \
                self.optimization_variables["U"][:, -1]]),
            self.optimization_variables["Q"],
            
            self.optimization_variables["X"],
            
            ci.horzcat([self.optimization_variables["EPS_U"],
                self.optimization_variables["EPS_U"][:,-1]]),

            self.optimization_variables["P"],
        ]

        [self.measurements] = phifcn.map(measurement_function_input)


    def __discretize(self):

        self.__set_multiple_shooting_settings()
        self.__apply_discretization_method()

        self.__evaluate_measurement_function()


    def __init__(self, system, time_points):
        
        super(ODEMultipleShooting, self).__init__(system, time_points)

        self.__discretize()
