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

from time import time

pl.close("all")

USE_KNOWN_NOISE = False
PARAMETER_ESTIMATION = True
EXPERIMENTAL_DESIGN = False

# System setup

T = 1.0
N = 100

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


if USE_KNOWN_NOISE:

    ydata = pl.loadtxt("quarter_vehicle_ydata_noise.txt", delimiter = ",")
    uinit_noise = pl.loadtxt("quarter_vehicle_uinit_noise.txt", delimiter = ",")

else:

    ydata = (ydata + 0.01 * pl.random(ydata.shape))
    uinit_noise = uinit + 0.01 * pl.random(uinit.shape)


# Parameter estimation

if PARAMETER_ESTIMATION:

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


    # Single shooting for comparison

    dt = T / N

    ffcn = ca.MXFunction("ffcn", \
        ca.daeIn(x = x, p = ca.vertcat([u, eps_u, p])), \
        ca.daeOut(ode = f))

    ffcn = ffcn.expand()

    rk4 = ca.Integrator("rk4", "rk", ffcn, {"t0": 0, "tf": dt, \
        "number_of_finite_elements": 1})

    P = ca.MX.sym("P", 3)
    EPS_U = ca.MX.sym("EPS_U", N)
    X0 = ca.MX.sym("X0", 4)

    V = ca.vertcat([P, EPS_U, X0])

    x_end = X0
    obj = [x_end - ydata[0,:].T]

    for k in range(N):

        x_end = rk4(x0 = x_end, p = ca.vertcat([uinit_noise[k], EPS_U[k], P]))["xf"]
        obj.append(x_end - ydata[k+1, :].T)

    r = ca.vertcat([ca.vertcat(obj), EPS_U])

    nlp = ca.MXFunction("nlp", ca.nlpIn(x = V), ca.nlpOut(f = ca.mul(r.T, r)))

    nlpsolver = ca.NlpSolver("nlpsolver", "ipopt", nlp)

    V0 = ca.vertcat([

            pl.ones(3), \
            pl.zeros(N), \
            ydata[0,:].T

        ])

    sol = nlpsolver(x0 = V0)

    p_est_single_shooting = sol["x"][:3]

    tstart_Sigma_p = time()

    J_s = ca.jacobian(r, V)

    F_s = ca.mul(J_s.T, J_s)

    beta = (ca.mul(r.T, r) / (r.size() - V.size())) 
    Sigma_p_s = beta * ca.solve(F_s, ca.MX.eye(F_s.shape[0]), "csparse")

    beta_fcn = ca.MXFunction("beta_fcn", [V], [beta])
    print beta_fcn([sol["x"]])[0]

    Sigma_p_s_fcn = ca.MXFunction("Sigma_p_s_fcn", \
        [V] , [Sigma_p_s])

    Cov_p = Sigma_p_s_fcn([sol["x"]])[0][:3, :3]

    tend_Sigma_p = time()


    print "\n\n--- Conclusion ---"

    print "\np multiple shooting: " + str(pe.estimated_parameters)
    print "p single shooting: " + str(p_est_single_shooting)

    print "\nDuration Sigma_p computation multiple shooting :" + \
        str(pe._duration_covariance_computation) + " s"
    print "Duration Sigma_p computation single shooting :" + \
        str(tend_Sigma_p - tstart_Sigma_p) +  " s"

    print "\nCovariance matrix multiple shooting: "
    print pe.covariance_matrix[:3, :3]

    print "\nCovariance matrix single shooting: "
    print Cov_p[:3, :3]

    # Simulation with estimated parameters

    simulation_est_multiple_shooting = cp.sim.Simulation( \
        system = system, pdata = pe.estimated_parameters)

    simulation_est_multiple_shooting.run_system_simulation( \
        x0 = x0, time_points = time_points, udata = uinit, print_status = False)

    x_sim_m = simulation_est_multiple_shooting.simulation_results.T


    simulation_est_single_shooting = cp.sim.Simulation( \
        system = system, pdata = p_est_single_shooting)

    simulation_est_single_shooting.run_system_simulation( \
        x0 = x0, time_points = time_points, udata = uinit, print_status = False)

    x_sim_s = simulation_est_single_shooting.simulation_results.T


    pl.figure()
    pl.subplot(5, 1, 1)
    pl.plot(time_points, ydata[:,0], label = "x1_meas")
    pl.plot(time_points, x_sim_m[:,0], "x", label = "x1_sim_m")
    pl.plot(time_points, x_sim_s[:,0], label = "x1_sim_s")
    pl.legend()
    pl.subplot(5, 1, 2)
    pl.plot(time_points, ydata[:,1], label = "x2_meas")
    pl.plot(time_points, x_sim_m[:,1], "x", label = "x2_sim_m")
    pl.plot(time_points, x_sim_s[:,1], label = "x2_sim_s")
    pl.legend()
    pl.subplot(5, 1, 3)
    pl.plot(time_points, ydata[:,2], label = "x3_meas")
    pl.plot(time_points, x_sim_m[:,2], "x", label = "x3_sim_m")
    pl.plot(time_points, x_sim_s[:,2], label = "x3_sim_s")
    pl.legend()
    pl.subplot(5, 1, 4)
    pl.plot(time_points, ydata[:,3], label = "x4_meas")
    pl.plot(time_points, x_sim_m[:,3], "x", label = "x4_sim_m")
    pl.plot(time_points, x_sim_s[:,3], label = "x4_sim_s")
    pl.legend()
    pl.subplot(5, 1, 5)
    pl.plot(time_points[:-1], uinit, label = "uinit")
    pl.plot(time_points[:-1], uinit_noise, label = "uinit_noise")
    pl.legend()
    pl.show()

    p_for_oed = pe.estimated_parameters


# Optimum experimental design

if EXPERIMENTAL_DESIGN:

    p_for_oed = [4.0, 4.0, 1.6]

    ulim = 0.05
    umin = -ulim
    umax = +ulim

    xlim = [0.1, 0.06, 0.1, 0.22]
    xmin = [-lim for lim in xlim]
    xmax = [+lim for lim in xlim]

    doe = cp.doe.DoE(system = system, time_points = time_points, \
        uinit = uinit, pdata = p_for_oed, \
        x0 = ydata[0,:], \
        umin = umin, umax = umax, \
        xmin = xmin, xmax = xmax)

    doe.run_experimental_design()
