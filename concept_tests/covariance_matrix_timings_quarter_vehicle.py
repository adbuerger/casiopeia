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

# This example is an adapted version of the system identification example
# included in CasADi, for the original file see:
# https://github.com/casadi/casadi/blob/master/docs/examples/python/sysid.py

import pylab as pl

import casadi as ca
import casiopeia as cp

import os

# System setup

x = ca.MX.sym("x", 4)

u = ca.MX.sym("u", 1)
eps_u = ca.MX.sym("eps_u", 1)

p = ca.MX.sym("p", 3)

k_M = p[0]
c_M = p[1]
c_m = p[2]

M = 250.0
m = 50.0

k_M_true = 4.0
c_M_true = 4.0
c_m_true = 1.6

p_true = [k_M_true, c_M_true, c_m_true]

p_scale = [1e3, 1e4, 1e5]

f = ca.vertcat([ \

        x[1], \
        (p_scale[0] * k_M / m) * (x[3] - x[1]) + (p_scale[1] * c_M / m) * (x[2] - x[0]) - (p_scale[2] * c_m / m) * (x[0] - (u + eps_u)), \
        x[3], \
        -(p_scale[0] * k_M / M) * (x[3] - x[1]) - (p_scale[1] * c_M / M) * (x[2] - x[0]) \

    ])

phi = x

system = cp.system.System( \
    x = x, u = u, p = p, f = f, phi = phi, eps_u = eps_u)


time_covariance_matrix_evlaution = []

time_horizons = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 25.0]#, 50.0, 75.0, 100.0]

for tf in time_horizons:

    T = float(tf)
    dt = 0.01

    N = T / dt

    simulation_true_parameters = cp.sim.Simulation( \
        system = system, pdata = p_true)

    time_points = pl.linspace(0, T, N+1)

    u0 = 0.05
    x0 = pl.zeros(x.shape)

    sigma_u = 0.005
    sigma_y = pl.array([0.01, 0.01, 0.01, 0.01])

    udata = u0 * pl.sin(2 * pl.pi*time_points[:-1])
    udata_noise = udata + sigma_u * pl.randn(*udata.shape)

    simulation_true_parameters.run_system_simulation( \
        x0 = x0, time_points = time_points, udata = udata_noise)

    ydata = simulation_true_parameters.simulation_results.T

    ydata_noise = ydata + sigma_y * pl.randn(*ydata.shape)

    wv = (1.0 / sigma_y**2) * pl.ones(ydata.shape)
    weps_u = (1.0 / sigma_u**2) * pl.ones(udata.shape)

    pe_test = cp.pe.LSq(system = system, \
        time_points = time_points, \
        udata = udata, \
        pinit = [1.0, 1.0, 1.0], \
        ydata = ydata_noise, \
        xinit = ydata_noise, \
        wv = wv, 
        weps_u = weps_u,
        discretization_method = "multiple_shooting")

    pe_test.run_parameter_estimation()
    pe_test.compute_covariance_matrix()

    time_covariance_matrix_evlaution.append( \
        pe_test._duration_covariance_computation)

print time_covariance_matrix_evlaution
