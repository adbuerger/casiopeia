import casadi as ca
import pylab as pl
import casiopeia as cp

import os

N = 1000
fs = 610.1

p_true = ca.DMatrix([5.625e-6,2.3e-4,1,4.69])
p_guess = ca.DMatrix([5,3,1,5])
scale = ca.vertcat([1e-6,1e-4,1,1])

x = ca.MX.sym("x", 2)
u = ca.MX.sym("u", 1)
p = ca.MX.sym("p", 4)

f = ca.vertcat([
        x[1], \
        (u - scale[3] * p[3] * x[0]**3 - scale[2] * p[2] * x[0] - \
            scale[1] * p[1] * x[1]) / (scale[0] * p[0]), \
    ])

phi = x

system = cp.system.System(x = x, u = u, p = p, f = f, phi = phi)

dt = 1.0 / fs
time_points = pl.linspace(0, N, N+1) * dt

udata = ca.DMatrix(0.1*pl.random(N))

ydata = pl.zeros((x.shape[0], N+1))

sim_true = cp.sim.Simulation(system, (p_true / scale) )

sim_true.run_system_simulation(time_points = time_points, \
    x0 = ydata[:, 0], udata = udata)

# lsqpe_sim = pecas.LSq( \
#     system = odesys, tu = tu, \
#     uN = uN, \
#     pinit = p_guess, \
#     xinit = yN, 
#     # linear_solver = "ma97", \
#     yN = yN, wv = wv)

# lsqpe_sim.run_simulation(x0 = [0.0, 0.0], psim = p_true/scale)

p_test = []

sigma = 0.01
wv = (1. / sigma**2) * pl.ones(ydata.shape)

repetitions = 100

for k in range(repetitions):

    y_randn = sim_true.simulation_results + \
        sigma * (pl.randn(*sim_true.simulation_results.shape))

    pe_test = cp.pe.LSq(system = system, time_points = time_points, \
    udata = udata, pinit = p_guess, \
    xinit = y_randn, ydata = y_randn, wv = wv) #, \
    # linear_solver = "ma97")

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

    y_randn = sim_true.simulation_results + sigma * \
(np.random.randn(*sim_true.estimated_parameters.shape))

Afterwards, compute standard deviation of estimated parameters, 
and compare to single covariance matrix computation done in PECas.

''')

prob = "ODE, 2 states, 1 control, 1 param, (silverbox)"
report.write(prob)
report.write("\n" + "-" * len(prob) + "\n\n.. code-block:: python")

report.write( \
'''.. code-block:: python

    ---------------------- casiopeia system definition -----------------------

    The system is a dynamic system defined by a set of
    explicit ODEs xdot which establish the system state x:
        xdot = f(t, u, x, p, we, wu)
    and by an output function phi which sets the system measurements:
        y = phi(t, x, p).

    Particularly, the system has:
        1 inputs u
        4 parameters p
        2 states x
        2 outputs phi

    Where xdot is defined by: 
    xdot[0] = x[1]
    xdot[1] = ((((u-(p[3]*pow(x[0],3)))-(p[2]*x[0]))- 
        ((0.0001*p[1])*x[1]))/(1e-06*p[0]))

    And where phi is defined by: 
    y[0] = x[0]
    y[1] = x[1]
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
