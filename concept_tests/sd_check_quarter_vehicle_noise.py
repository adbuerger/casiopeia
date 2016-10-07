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


time_points = pl.linspace(0, T, N+1)

# u0 = 0.05
x0 = pl.zeros(x.shape)

# udata = u0 * pl.sin(2 * pl.pi*time_points[:-1])

udata = ca.DMatrix([-0.0144505, -0.00970027, -0.0131704, -0.0162461, -0.0194673, -0.0226493, 0.00758111, -0.0377716, 0.00802565, -0.0376995, 0.00770359, -0.00442005, -0.000855387, 0.0022456, 0.00545773, 0.00859222, 0.0117409, 0.0148885, 0.0180565, 0.0212502, 0.0154754, 0.0304801, -0.00435828, 0.0413637, -0.00430136, 0.0415592, -0.00399078, 0.0419632, -0.00324539, 0.00905321, 0.0056727, 0.00271414, -0.000364645, -0.00338611, -0.00643686, -0.00950217, -0.00850269, -0.017266, -0.00206468, -0.025834, 0.0121024, -0.0336159, 0.0120527, -0.0338047, 0.0117481, -0.0342033, 0.0110075, -0.00128918, 0.00209328, 0.00505328, 0.00813338, 0.0111559, 0.0142075, 0.0172735, 0.017385, 0.0247696, -0.00750083, 0.038085, -0.0077066, 0.0380376, -0.00733884, 0.00481979, 0.0348082, -0.010399, 0.0355777, -0.00990138, 0.0360839, -0.00939524, 0.0365825, -0.00891039, 0.0370485, -0.00846729, 0.0374656, -0.00807851, 0.0378246, -0.00775023, 0.0381219, -0.00748364, 0.0383584, -0.00727628, 0.0385379, -0.00712312, 0.0386664, -0.0070176, 0.0387508, -0.00695237, 0.0387987, -0.0069199, 0.0388175, -0.00691297, 0.0388142, -0.00692494, 0.0387951, -0.00694997, 0.0387654, -0.00698312, 0.0387298, -0.00702037, 0.0386917, -0.00705856])

simulation_true_parameters = cp.sim.Simulation( \
    system = system, pdata = p_true)

# simulation_true_parameters.run_system_simulation( \
#     x0 = x0, time_points = time_points, udata = udata)

# ydata = simulation_true_parameters.simulation_results.T

sigma = 0.005

repetitions = 100

p_test =[]

for k in range(repetitions):

    udata_noise = udata + sigma * pl.random(udata.shape)

    simulation_true_parameters.run_system_simulation( \
        x0 = x0, time_points = time_points, udata = udata_noise)

    ydata = simulation_true_parameters.simulation_results.T

    ydata_noise = ydata + sigma * pl.random(ydata.shape)


    pe_test = cp.pe.LSq(system = system, \
        time_points = time_points, \
        udata = udata, \
        pinit = [1.0, 1.0, 1.0], \
        ydata = ydata_noise, \
        xinit = ydata_noise, \
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

print("\np_mean         = " + str(ca.DMatrix(p_mean)))
print("phat_last_exp  = " + str(ca.DMatrix(pe_test.estimated_parameters)))

print("\np_sd           = " + str(ca.DMatrix(p_std)))
print("sd_from_covmat = " + str(ca.diag(ca.sqrt(pe_test.covariance_matrix))))
print("beta           = " + str(pe_test.beta))

print("\ndelta_abs_sd   = " + str(ca.fabs(ca.DMatrix(p_std) - \
    ca.diag(ca.sqrt(pe_test.covariance_matrix)))))
print("delta_rel_sd   = " + str(ca.fabs(ca.DMatrix(p_std) - \
    ca.diag(ca.sqrt(pe_test.covariance_matrix))) / ca.DMatrix(p_std)))


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

prob = "ODE, 4 states, 1 control + noise, 3 params, (quarter vehilce noise)"
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
report.write("\n    sigma          = " + str(sigma))

report.write("\n\n    p_true         = " + str(ca.DMatrix(p_true)))
report.write("\n\n    p_mean         = " + str(ca.DMatrix(p_mean)))
report.write("\n    phat_last_exp  = " + \
    str(ca.DMatrix(pe_test.estimated_parameters)))

report.write("\n\n    p_sd           = " + str(ca.DMatrix(p_std)))
report.write("\n    sd_from_covmat = " \
    + str(ca.diag(ca.sqrt(pe_test.covariance_matrix))))
report.write("\n    beta           = " + str(pe_test.beta))

report.write("\n\n    delta_abs_sd   = " + str(ca.fabs(ca.DMatrix(p_std) - \
    ca.diag(ca.sqrt(pe_test.covariance_matrix)))))
report.write("\n    delta_rel_sd   = " + str(ca.fabs(ca.DMatrix(p_std) - \
    ca.diag(ca.sqrt(pe_test.covariance_matrix))) / ca.DMatrix(p_std)) \
    + "\n")

report.close()

try:

    os.system("rst2pdf " + fname)

except:

    print("Generating PDF report failed, is rst2pdf installed correctly?")
