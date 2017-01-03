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

T = 1.0
N = 100

x = ca.MX.sym("x", 4)

u = ca.MX.sym("u", 1)

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
        (p_scale[0] * k_M / m) * (x[3] - x[1]) + (p_scale[1] * c_M / m) * (x[2] - x[0]) - (p_scale[2] * c_m / m) * (x[0] - u), \
        x[3], \
        -(p_scale[0] * k_M / M) * (x[3] - x[1]) - (p_scale[1] * c_M / M) * (x[2] - x[0]) \

    )

phi = x

system = cp.system.System( \
    x = x, u = u, p = p, f = f, phi = phi)


time_points = pl.linspace(0, T, N+1)

u0 = 0.05
x0 = pl.zeros(x.shape)

udata = u0 * pl.sin(2 * pl.pi*time_points[:-1])

simulation_true_parameters = cp.sim.Simulation( \
    system = system, pdata = p_true)

# simulation_true_parameters.run_system_simulation( \
#     x0 = x0, time_points = time_points, udata = udata)

# ydata = simulation_true_parameters.simulation_results.T

sigma_u = 0.001
sigma_y = pl.array([0.01, 0.01, 0.01, 0.01])

repetitions = 100

p_test =[]

for k in range(repetitions):

    udata_noise = udata + sigma_u * pl.randn(*udata.shape)

    simulation_true_parameters.run_system_simulation( \
        x0 = x0, time_points = time_points, udata = udata_noise)

    ydata = simulation_true_parameters.simulation_results.T

    ydata_noise = ydata + sigma_y * pl.randn(*ydata.shape)

    wv = (1.0 / sigma_y**2) * pl.ones(ydata.shape)

    pe_test = cp.pe.LSq(system = system, \
        time_points = time_points, \
        udata = udata, \
        pinit = [1.0, 1.0, 1.0], \
        ydata = ydata_noise, \
        xinit = ydata_noise, \
        wv = wv,
        discretization_method = "multiple_shooting")

    pe_test.run_parameter_estimation()

    p_test.append(pe_test.estimated_parameters)

p_mean = []
p_std = []

for j, e in enumerate(p_true):

    p_mean.append(pl.mean([k[j] for k in p_test]))
    p_std.append(pl.std([k[j] for k in p_test], ddof = 0))

pe_test.compute_covariance_matrix()

# Generate report

print("\np_mean         = " + str(ca.DM(p_mean)))
print("phat_last_exp  = " + str(ca.DM(pe_test.estimated_parameters)))

print("\np_sd           = " + str(ca.DM(p_std)))
print("sd_from_covmat = " + str(ca.diag(ca.sqrt(pe_test.covariance_matrix))))
print("beta           = " + str(pe_test.beta))

print("\ndelta_abs_sd   = " + str(ca.fabs(ca.DM(p_std) - \
    ca.diag(ca.sqrt(pe_test.covariance_matrix)))))
print("delta_rel_sd   = " + str(ca.fabs(ca.DM(p_std) - \
    ca.diag(ca.sqrt(pe_test.covariance_matrix))) / ca.DM(p_std)))


fname = os.path.basename(__file__)[:-3] + ".rst"

report = open(fname, "w")
report.write( \
'''Concept test: covariance matrix computation
===========================================

Simulate system. Then: add gaussian noise N~(0, sigma^2), estimate,
store estimated parameter, repeat.

.. code-block:: python

    y_randn = sim_true.simulation_results + sigma * \n
(np.random.randn(*sim_true.estimated_parameters.shape))

Afterwards, compute standard deviation of estimated parameters, 
and compare to single covariance matrix computation done in PECas.

''')

prob = "ODE, 4 states, 1 control, 3 params, (quarter vehilce)"
report.write(prob)
report.write("\n" + "-" * len(prob) + "\n\n.. code-block:: python")

report.write( \
'''.. code-block:: python

    # ----------------------- casiopeia system definition ---------------------- #

    Starting system definition ...

    The system is a dynamic system defined by a set of explicit ODEs xdot
    which establish the system state x and by an output function phi which
    sets the system measurements:

    xdot = f(t, u, q, x, p, eps_e, eps_u),
    y = phi(t, u, q, x, p).

    Particularly, the system has:
    1 time-varying controls u
    0 time-constant controls q
    3 parameters p
    4 states x
    4 outputs phi

    where xdot is defined by 
    xdot[0] = x[1]
    xdot[1] = (((((1000*p[0])/50)*(x[3]-x[1]))+(((10000*p[1])/50)*(x[2]-x[0])))
        -(((100000*p[2])/50)*(x[0]-(u+eps_u))))
    xdot[2] = x[3]
    xdot[3] = ((-(((1000*p[0])/250)*(x[3]-x[1])))-(((10000*p[1])/250)*
        (x[2]-x[0])))

    and where phi is defined by 
    y[0] = x[0]
    y[1] = x[1]
    y[2] = x[2]
    y[3] = x[3]

''')

report.write("\n**Test results:**\n\n.. code-block:: python")

report.write("\n\n    repetitions    = " + str(repetitions))
# report.write("\n    sigma          = " + str(sigma))

report.write("\n\n    p_true         = " + str(ca.DM(p_true)))
report.write("\n\n    p_mean         = " + str(ca.DM(p_mean)))
report.write("\n    phat_last_exp  = " + \
    str(ca.DM(pe_test.estimated_parameters)))

report.write("\n\n    p_sd           = " + str(ca.DM(p_std)))
report.write("\n    sd_from_covmat = " \
    + str(ca.diag(ca.sqrt(pe_test.covariance_matrix))))
report.write("\n    beta           = " + str(pe_test.beta))

report.write("\n\n    delta_abs_sd   = " + str(ca.fabs(ca.DM(p_std) - \
    ca.diag(ca.sqrt(pe_test.covariance_matrix)))))
report.write("\n    delta_rel_sd   = " + str(ca.fabs(ca.DM(p_std) - \
    ca.diag(ca.sqrt(pe_test.covariance_matrix))) / ca.DM(p_std)) \
    + "\n")

report.close()

try:

    os.system("rst2pdf " + fname)

except:

    print("Generating PDF report failed, is rst2pdf installed correctly?")
