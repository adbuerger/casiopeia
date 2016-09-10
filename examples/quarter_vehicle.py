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

pl.close("all")


# System setup

T = 5.0
N = 50

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
        # (p_scale[0] * k_M / m) * (x[3] - x[1]) + (p_scale[1] * c_M / m) * (x[2] - x[0]) - (p_scale[2] * c_m / m) * (x[0] - (u)), \
        (p_scale[0] * k_M / m) * (x[3] - x[1]) + (p_scale[1] * c_M / m) * (x[2] - x[0]) - (p_scale[2] * c_m / m) * (x[0] - (u + eps_u)), \
        x[3], \
        -(p_scale[0] * k_M / M) * (x[3] - x[1]) - (p_scale[1] * c_M / M) * (x[2] - x[0]) \

    ])

phi = x

system = cp.system.System( \
    # x = x, u = u, p = p, f = f, phi = phi)
    x = x, u = u, p = p, f = f, phi = phi, eps_u = eps_u)


# Generate "measurement" data

time_points = pl.linspace(0, T, N+1)

u0 = 0.05
uinit = u0 * pl.sin(pl.pi*time_points[:-1])

x0 = pl.zeros(x.shape)

simulation_true_parameters = cp.sim.Simulation( \
    system = system, pdata = p_true)

simulation_true_parameters.run_system_simulation( \
    x0 = x0, time_points = time_points, udata = uinit)

ydata = simulation_true_parameters.simulation_results.T
ydata += 0.01 * pl.random(ydata.shape)

uinit_noise = uinit + 0.01 * pl.random(uinit.shape)

# Parameter estimation

pe = cp.pe.LSq(system = system, \
    time_points = time_points, \
    udata = uinit_noise, \
    pinit = [1.0, 1.0, 1.0], \
    ydata = ydata, \
    xinit = ydata, \
    discretization_method = "multiple_shooting")
    # discretization_method = "collocation")

pe.run_parameter_estimation()
pe.print_estimation_results()

pe.compute_covariance_matrix()
pe.print_estimation_results()


# Simulation with estimated parameters

simulation_est_parameters = cp.sim.Simulation( \
    system = system, pdata = pe.estimated_parameters)

simulation_est_parameters.run_system_simulation( \
    x0 = x0, time_points = time_points, udata = uinit)

x_sim = simulation_est_parameters.simulation_results.T

pl.figure()
pl.subplot(4, 1, 1)
pl.plot(time_points, ydata[:,0], label = "x1_meas")
pl.plot(time_points, x_sim[:,0], label = "x1_sim")
pl.legend()
pl.subplot(4, 1, 2)
pl.plot(time_points, ydata[:,1], label = "x2_meas")
pl.plot(time_points, x_sim[:,1], label = "x2_sim")
pl.legend()
pl.subplot(4, 1, 3)
pl.plot(time_points, ydata[:,2], label = "x3_meas")
pl.plot(time_points, x_sim[:,2], label = "x3_sim")
pl.legend()
pl.subplot(4, 1, 4)
pl.plot(time_points, ydata[:,3], label = "x4_meas")
pl.plot(time_points, x_sim[:,3], label = "x4_sim")
pl.legend()
pl.show()


ulim = 0.05
umin = -ulim
umax = +ulim

xlim = [0.1, 0.06, 0.1, 0.22]
xmin = [-lim for lim in xlim]
xmax = [+lim for lim in xlim]

doe = cp.doe.DoE(system = system, time_points = time_points, \
    uinit = uinit, pdata = pe.estimated_parameters, \
    x0 = ydata[0,:], \
    umin = umin, umax = umax, \
    xmin = xmin, xmax = xmax)

# doe.run_experimental_design()

# sim_true = 
