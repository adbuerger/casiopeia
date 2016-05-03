import casadi as ca
import pylab as pl
import casiopeia as cp

import os

# (Model and data taken from: Diehl, Moritz: Course on System Identification, 
# exercise 7, SYSCOP, IMTEK, University of Freiburg, 2014/2015)

# Defining constant problem parameters: 
#
#     - m: representing the ball of the mass in kg
#     - L: the length of the pendulum bar in meters
#     - g: the gravity constant in m/s^2
#     - psi: the actuation angle of the manuver in radians, which stays
#            constant for this problem

m = 1.0
L = 3.0
g = 9.81
psi = pl.pi / 2.0

# System

x = ca.MX.sym("x", 2)
u = ca.MX.sym("u", 1)
p = ca.MX.sym("p", 1)

f = ca.vertcat([x[1], p[0]/(m*(L**2))*(u-x[0]) - g/L * pl.sin(x[0])])

phi = x

system = cp.system.System(x = x, u = u, p = p, f = f, phi = phi)

data = pl.loadtxt('data_pendulum.txt')
time_points = data[:500, 0]
numeas = data[:500, 1]
wmeas = data[:500, 2]
N = time_points.size
ydata = pl.array([numeas,wmeas])
udata = [psi] * (N-1)

ptrue = [3.0]

sim_true = cp.sim.Simulation(system, ptrue) 

sim_true.run_system_simulation(time_points = time_points, \
    x0 = ydata[:, 0], udata = udata)

# pl.figure()
# pl.plot(time_points, pl.squeeze(sim_true.simulation_results[0,:]))
# pl.plot(time_points, pl.squeeze(sim_true.simulation_results[1,:]))
# pl.show()

p_test = []

sigma = 0.1
wv = (1. / sigma**2) * pl.ones(ydata.shape)


repetitions = 100

for k in range(repetitions):

    y_randn = sim_true.simulation_results + \
        sigma * (pl.randn(*sim_true.simulation_results.shape))

    pe_test = cp.pe.LSq(system = system, time_points = time_points,
        udata = udata, xinit = y_randn, ydata = y_randn, wv = wv, pinit = 1)

    pe_test.run_parameter_estimation()

    p_test.append(pe_test.estimated_parameters)


p_mean = pl.mean(p_test)
p_std = pl.std(p_test, ddof=0)

pe_test.compute_covariance_matrix()
pe_test.print_estimation_results()


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

prob = "ODE, 2 states, 1 control, 1 param, (pendulum)"
report.write(prob)
report.write("\n" + "-" * len(prob) + "\n\n.. code-block:: python")

report.write( \
'''.. code-block:: python

    ------------------------ PECas system information ------------------------

    The system is a dynamic system defined by a set of
    explicit ODEs xdot which establish the system state x:
        xdot = f(t, u, x, p, we, wu)
    and by an output function phi which sets the system measurements:
        y = phi(t, x, p).

    Particularly, the system has:
        1 inputs u
        1 parameters p
        2 states x
        2 outputs phi

    Where xdot is defined by: 
    xdot[0] = x[1]
    xdot[1] = (((p/9)*(u-x[0]))-(3.27*sin(x[0])))

    And where phi is defined by: 
    y[0] = x[0]
    y[1] = x[1]
''')

report.write("\n**Test results:**\n\n.. code-block:: python")

report.write("\n\n    repetitions    = " + str(repetitions))
report.write("\n    sigma          = " + str(sigma))

report.write("\n\n    p_orig         = " + str(ca.DMatrix(ptrue)))
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
