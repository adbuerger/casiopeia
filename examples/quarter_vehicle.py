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

import matplotlib.lines as mlines

PARAMETER_ESTIMATION = True
EXPERIMENTAL_DESIGN = False

# System setup

T = 5.0
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

f = ca.vertcat( \

        x[1], \
        (p_scale[0] * k_M / m) * (x[3] - x[1]) + (p_scale[1] * c_M / m) * (x[2] - x[0]) - (p_scale[2] * c_m / m) * (x[0] - (u + eps_u)), \
        x[3], \
        -(p_scale[0] * k_M / M) * (x[3] - x[1]) - (p_scale[1] * c_M / M) * (x[2] - x[0]) \

    )

phi = x


system = cp.system.System( \
    x = x, u = u, p = p, f = f, phi = phi, eps_u = eps_u)

# Generate "measurement" data

time_points = pl.linspace(0, T, N+1)

u0 = 0.05
udata = u0 * pl.sin(2 * pl.pi * time_points[:-1])

simulation_true_parameters = cp.sim.Simulation( \
    system = system, pdata = p_true)

sigma_u = 0.005
sigma_y = pl.array([0.01, 0.01, 0.01, 0.01])

udata_noise = udata + sigma_u * pl.randn(*udata.shape)

x0 = pl.zeros(x.shape)

simulation_true_parameters.run_system_simulation( \
    x0 = x0, time_points = time_points, udata = udata_noise)

ydata = simulation_true_parameters.simulation_results.T

ydata_noise = ydata + sigma_y * pl.randn(*ydata.shape)


wv = (1.0 / sigma_y**2) * pl.ones(ydata.shape)
weps_u = (1.0 / sigma_u**2) * pl.ones(udata.shape)

# Parameter estimation

if PARAMETER_ESTIMATION:

    # Parameter estimation on casiopeia

    pe = cp.pe.LSq(system = system, \
        time_points = time_points, \
        udata = udata, \
        pinit = [1.0, 1.0, 1.0], \
        ydata = ydata_noise, \
        xinit = ydata_noise, \
        wv = wv,
        weps_u = weps_u,
        discretization_method = "multiple_shooting")

    pe.run_parameter_estimation()
    pe.compute_covariance_matrix()

    pe.print_estimation_results()

    # Single shooting for comparison

    dt = T / N

    ffcn = ca.MXFunction("ffcn", \
        ca.daeIn(x = x, p = ca.vertcat([u, eps_u, p])), \
        ca.daeOut(ode = f))

    ffcn = ffcn.expand()

    rk4 = ca.Integrator("cvodes", "rk", ffcn, {"t0": 0, "tf": dt}) #, \
        #"number_of_finite_elements": 1})

    P = ca.MX.sym("P", 3)
    EPS_U = ca.MX.sym("EPS_U", N)
    X0 = ca.MX.sym("X0", 4)

    V = ca.vertcat([P, EPS_U, X0])

    x_end = X0
    obj = [x_end - ydata_noise[0,:].T]

    for k in range(N):

        x_end = rk4(x0 = x_end, p = ca.vertcat([udata[k], EPS_U[k], P]))["xf"]
        obj.append(x_end - ydata_noise[k+1, :].T)

    r = ca.vertcat([ca.vertcat(obj), EPS_U])

    Sigma_y_inv = ca.diag(ca.vec(wv))
    Sigma_u_inv = ca.diag(weps_u)
    Z = ca.DMatrix(pl.zeros((Sigma_y_inv.shape[0], Sigma_u_inv.shape[1])))

    Sigma = ca.blockcat(Sigma_y_inv, Z, Z.T, Sigma_u_inv)

    nlp = ca.MXFunction("nlp", ca.nlpIn(x = V), \
        ca.nlpOut(f = 0.5 * ca.mul([r.T, Sigma, r])))

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

    F_s = ca.mul([J_s.T, Sigma, J_s])

    beta = (ca.mul([r.T, Sigma, r]) / (r.size() - V.size())) 
    Sigma_p_s = beta * ca.solve(F_s, ca.MX.eye(F_s.shape[0]), "csparse")

    beta_fcn = ca.MXFunction("beta_fcn", [V], [beta])
    print beta_fcn([sol["x"]])[0]

    Sigma_p_s_fcn = ca.MXFunction("Sigma_p_s_fcn", \
        [V] , [Sigma_p_s])

    Cov_p = Sigma_p_s_fcn([sol["x"]])[0][:3, :3]

    tend_Sigma_p = time()


    print "\n\n--- Conclusion ---"

    print "p multiple shooting: " + str(pe.estimated_parameters)
    print "p single shooting: " + str(p_est_single_shooting)

    print "\nDuration Sigma_p computation multiple shooting :" + \
        str(pe._duration_covariance_computation) + " s"
    print "Duration Sigma_p computation single shooting :" + \
        str(tend_Sigma_p - tstart_Sigma_p) +  " s"

    print "\nCovariance matrix multiple shooting: "
    print pe.covariance_matrix[:3, :3]

    print "\nCovariance matrix single shooting: "
    print Cov_p[:3, :3]


# Optimum experimental design

if EXPERIMENTAL_DESIGN:

    p_for_oed = [4.14054, 3.96515, 1.64997]

    ulim = 0.05
    umin = -ulim
    umax = +ulim

    xlim = [0.1, 0.4, 0.1, 0.4]
    xmin = [-lim for lim in xlim]
    xmax = [+lim for lim in xlim]

    sigma_y = pl.array([0.01, 0.01, 0.01, 0.01])
    sigma_u = 0.005

    wv = (1.0 / sigma_y**2) * pl.ones(ydata.shape)
    weps_u = (1.0 / sigma_u**2) * pl.ones(uinit.shape)

    doe = cp.doe.DoE(system = system_noise, time_points = time_points, \
        uinit = uinit, pdata = p_for_oed, \
        x0 = ydata[0,:], \
        wv = wv, weps_u = weps_u, \
        umin = umin, umax = umax, \
        xmin = xmin, xmax = xmax)

    doe.run_experimental_design(solver_options = {"ipopt": {"linear_solver": "ma86"}})
