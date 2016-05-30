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
# but WITHOUT ANY WARRANTY; without even the implied warrantime_points of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with casiopeia. If not, see <http://www.gnu.org/licenses/>.

# Model and data taken from: Bock, Sager et al.: Uebungen zur Numerischen
# Mathematik II, sheet 9, IWR, Heidelberg university, 2006

import pylab as pl
import casadi as ca

import casiopeia as cp

T = pl.linspace(0, 10, 11)

yN = pl.array([[1.0, 0.9978287, 2.366363, 6.448709, 5.225859, 2.617129, \
           1.324945, 1.071534, 1.058930, 3.189685, 6.790586], \

           [1.0, 2.249977, 3.215969, 1.787353, 1.050747, 0.2150848, \
           0.109813, 1.276422, 2.493237, 3.079619, 1.665567]])

# T = T[:2]
# yN = yN[:, :2]

sigma_x1 = 0.1
sigma_x2 = 0.2

x = ca.MX.sym("x", 2)

alpha = 1.0
gamma = 1.0

p = ca.MX.sym("p", 2)

f = ca.vertcat( \
    [-alpha * x[0] + p[0] * x[0] * x[1], 
    gamma * x[1] - p[1] * x[0] * x[1]])

phi = x

system = cp.system.System(x = x, p = p, f = f, phi = phi)

# The weightings for the measurements errors given to casiopeia are calculated
# from the standard deviations of the measurements, so that the least squares
# estimator ist the maximum likelihood estimator for the estimation problem.

wv = pl.zeros((2, yN.shape[1]))
wv[0,:] = (1.0 / sigma_x1**2)
wv[1,:] = (1.0 / sigma_x2**2)

pe_1 = cp.pe.LSq(system = system, time_points = T, xinit = yN, \
    ydata = yN, wv = wv, discretization_method = "collocation")

pe_2 = cp.pe.LSq(system = system, time_points = T, xinit = yN, \
    ydata = yN, wv = wv, discretization_method = "collocation")

# pe_3 = cp.pe.LSq(system = system, time_points = T, xinit = yN, \
#     ydata = yN, wv = wv, discretization_method = "collocation")

# pe_4 = cp.pe.LSq(system = system, time_points = T, xinit = yN, \
#     ydata = yN, wv = wv, discretization_method = "collocation")

# pe_5 = cp.pe.LSq(system = system, time_points = T, xinit = yN, \
#     ydata = yN, wv = wv, discretization_method = "collocation")

# pe_6 = cp.pe.LSq(system = system, time_points = T, xinit = yN, \
#     ydata = yN, wv = wv, discretization_method = "collocation")

# pe_7 = cp.pe.LSq(system = system, time_points = T, xinit = yN, \
#     ydata = yN, wv = wv, discretization_method = "collocation")

mpe = cp.pe.MultiLSq([pe_1, pe_2])

mpe.run_parameter_estimation()
mpe.print_estimation_results()

mpe.compute_covariance_matrix()
mpe.print_estimation_results()

T_sim = pl.linspace(0, 10, 101)
x0 = yN[:,0]

sim = cp.sim.Simulation(system, mpe.estimated_parameters)
sim.run_system_simulation(time_points = T_sim, x0 = x0)

pl.figure()

pl.scatter(T, yN[0,:], color = "b", label = "$x_{1,meas}$")
pl.scatter(T, yN[1,:], color = "r", label = "$x_{2,meas}$")

pl.plot(T_sim, pl.squeeze(sim.simulation_results[0,:]), color="b", label = "$x_{1,sim}$")
pl.plot(T_sim, pl.squeeze(sim.simulation_results[1,:]), color="r", label = "$x_{2,sim}$")

pl.xlabel("$t$")
pl.ylabel("$x_1, x_2$", rotation = 0)
pl.xlim(0.0, 10.0)

pl.legend(loc = "upper left")
pl.grid()

pl.show()
