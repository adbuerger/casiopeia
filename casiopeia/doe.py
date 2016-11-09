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

'''The module ``casiopeia.doe`` contains the classes used for optimum
experimental design.'''

import numpy as np
import matplotlib.pyplot as plt
import time
import __main__
import os

from abc import ABCMeta, abstractmethod, abstractproperty

from discretization.nodiscretization import NoDiscretization
from discretization.odecollocation import ODECollocation
from discretization.odemultipleshooting import ODEMultipleShooting

from interfaces import casadi_interface as ci
from matrices import KKTMatrix, FisherMatrix, CovarianceMatrix, \
    setup_a_criterion, setup_d_criterion
from intro import intro
from sim import Simulation

import inputchecks


class DoEProblem(object):

    __metaclass__ = ABCMeta

    @abstractproperty
    def optimized_controls(self):

        r'''
        Abstract method for returning the optimized controls.
        '''


    @property
    def design_results(self):

        try:

            return self._design_results

        except AttributeError:

            raise AttributeError('''
An experimental design has to be executed before the design results
can be accessed, please run run_experimental_design() first.
''')

    @property
    def covariance_matrix_initial(self):

        try:

            return self._covariance_matrix_initial

        except AttributeError:

            self._compute_initial_covariance_matrix()

        return self._covariance_matrix_initial


    @property
    def covariance_matrix_optimized(self):

        try:

            return self._covariance_matrix_optimized

        except AttributeError:

            self._compute_optimized_covariance_matrix()

        return self._covariance_matrix_optimized


    def _set_optimiality_criterion(self, optimality_criterion):

            if str(optimality_criterion).upper() == "A":

                self._optimality_criterion = setup_a_criterion

            elif str(optimality_criterion).upper() == "D":

                self._optimality_criterion = setup_d_criterion

            else:

                raise NotImplementedError('''
Unknown optimality criterion: {0}.
Possible values are "A" and "D".
'''.format(str(discretization_method)))


    def _setup_covariance_matrix(self):

        kkt_matrix = KKTMatrix(self._gauss_newton_lagrangian_hessian, \
            self._constraints, self._cov_matrix_derivative_directions)

        fisher_matrix = FisherMatrix(kkt_matrix.kkt_matrix, \
            self._discretization.system.np)

        self._covariance_matrix = CovarianceMatrix( \
            fisher_matrix.fisher_matrix)


    def _setup_objective(self):

        self._objective_parameters_free = self._optimality_criterion( \
            self._covariance_matrix.covariance_matrix)


    def _setup_nlp(self):

        self._nlp = {"x": self._optimization_variables, \
            "f": self._objective, \
            "g": self._equality_constraints_parameters_applied}



    @abstractmethod
    def __init__(self):

        r'''
        Abstract constructor for experimental design classes.
        '''


    def _plot_confidence_ellipsoids(self, pdata, properties = "initial"):

        if properties == "initial":

            covariance_matrix = {"initial": self.covariance_matrix_initial}

        elif properties == "optimized":

            covariance_matrix = {"optimized": self.covariance_matrix_optimized}

        elif properties == "all":

            covariance_matrix = {"initial" : self._covariance_matrix_initial, \
                "optimized" : self._covariance_matrix_optimized}

        else:

            raise ValueError('''
Input-value not supported, choose either "initial", "final", or "all".
''')

        plotting_directory = "confidence_ellipsoids_" + \
                os.path.basename(__main__.__file__).strip(".py")

        try:
            os.mkdir(plotting_directory)

        except OSError:
            if not os.path.isdir(plotting_directory):

                raise OSError('''
Plotting directory "confidence_ellipsoids_{0}"
does not yet exist, but could not be created.

Do you have write access within your working folder, or is
some file with this name already present within your working folder?
'''.format(plotting_directory))

        xy = np.array([np.cos(np.linspace(0, 2*np.pi, 100)), 
            np.sin(np.linspace(0, 2*np.pi, 100))])

        for p1 in range(pdata.size):

            for p2 in range(pdata.size)[p1+1:]:

                plt.figure()

                for prop, cm in covariance_matrix.iteritems():

                    covariance_matrix_p1p2 = np.array([ \

                            [cm[p1, p1], cm[p1, p2]], \
                            [cm[p2, p1], cm[p2, p2]]

                        ])

                    w, v = np.linalg.eig(covariance_matrix_p1p2)

                    ellipsoid = ci.repmat(np.array([pdata[p1], pdata[p2]]), 1, 100) + \
                        ci.mul([v, ci.diag(w), xy])

                    plt.plot(ellipsoid[0,:].T, ellipsoid[1,:].T, label = \
                        "p_" + str(p1) + ", p_" + str(p2) + " " + prop)
                
                plt.scatter(pdata[p1], pdata[p2], color = "k")    
                plt.legend(loc="upper right")                
                plt.savefig(plotting_directory + "/p_" + str(p1) + \
                    "-p_" + str(p2) + "-" + properties + ".png", bbox_inches='tight')
                plt.close()


    def print_initial_experimental_properties(self):

        r'''
        Print the standard deviations and the covariance matrix for the initial
        experimental setup, i. e. before optimization.
        '''

        print("\nInitial experimental properties:")

        self._print_experimental_properties(self.covariance_matrix_initial)


    def run_experimental_design(self, solver_options = {}):

        r'''
        :param solver_options: options to be passed to the IPOPT solver 
                               (see the CasADi documentation for a list of all
                               possible options)
        :type solver_options: dict

        This functions will pass the experimental design problem
        to the IPOPT solver. The status of IPOPT printed to the 
        console provides information whether the
        optimization finished successfully. The optimized controls
        :math:`\hat{u}` can afterwards be accessed via the class attribute
        ``LSq.optimized_controls``.

        .. note::

            IPOPT finishing successfully does not necessarily
            mean that the optimized controls are useful
            for your purposes, it just means that IPOPT was able to solve the
            given optimization problem.
            A poorly chosen "guess" for the values of the parameters to be
            estimated can lead to very suboptimal controls, an with this,
            to more repetitions in optimization and estimation.

        '''  

        print('\n' + '# ' +  18 * '-' + \
            ' casiopeia optimum experimental design ' + 17 * '-' + ' #')

        self.print_initial_experimental_properties()

        print('''
Starting optimum experimental design using IPOPT, 
this might take some time ...
''')

        self._tstart_optimum_experimental_design = time.time()

        nlpsolver = ci.NlpSolver("solver", "ipopt", self._nlp, \
            options = solver_options)

        self._design_results = \
            nlpsolver(x0 = self._optimization_variables_initials, \
                lbg = 0, ubg = 0, \
                lbx = self._optimization_variables_lower_bounds, \
                ubx = self._optimization_variables_upper_bounds)

        print('''
Optimum experimental design finished. Check IPOPT output for status information.
''')

        self.print_optimized_experimental_properties()

        self._tend_optimum_experimental_design = time.time()
        self._duration_optimum_experimental_design = \
            self._tend_optimum_experimental_design - \
            self._tstart_optimum_experimental_design


    def print_optimized_experimental_properties(self):

        r'''
        Print the standard deviations and the covariance matrix for the
        optimized experimental setup.
        '''

        print("\nFinal experimental properties:")

        self._print_experimental_properties(self.covariance_matrix_optimized)


class DoE(DoEProblem):

    '''The class :class:`casiopeia.doe.DoE` is used to set up
    Design-of-Experiments-problems for systems defined with the
    :class:`casiopeia.system.System` class.

    The aim of the experimental design optimization is to identify a set of
    controls that can be used for the generation of measurement data which
    allows for a better estimation of the unknown parameters of a system.

    To achieve this, an information function on the covariance matrix of the
    estimated parameters is minimized. The values of the estimated parameters,
    though they are mostly an initial
    guess for their values, are not changed during the optimization.

    Optimum experimental design and parameter estimation methods can be used
    interchangeably until a desired accuracy of the parameters has been
    achieved.
    '''

    @property
    def optimized_controls(self):

        return self.design_results["x"][ \
            :(self._discretization.number_of_intervals * \
                self._discretization.system.nu)]


    def _discretize_system(self, system, time_points, discretization_method, \
        **kwargs):

        if system.nx == 0:

            self._discretization = NoDiscretization(system, time_points)

        elif system.nx != 0:

            if discretization_method == "collocation":

                self._discretization = ODECollocation( \
                    system, time_points, **kwargs)

            elif discretization_method == "multiple_shooting":

                self._discretization = ODEMultipleShooting( \
                    system, time_points, **kwargs)

            else:

                raise NotImplementedError('''
Unknown discretization method: {0}.
Possible values are "collocation" and "multiple_shooting".
'''.format(str(discretization_method)))
      

    def _set_parameter_guess(self, pdata):

        self._pdata = inputchecks.check_parameter_data(pdata, \
            self._discretization.system.np)


    def _apply_parameters_to_equality_constraints(self):

        optimization_variables_for_equality_constraints = ci.veccat([ \

                self._discretization.optimization_variables["U"],
                self._discretization.optimization_variables["Q"],
                self._discretization.optimization_variables["X"], 
                self._discretization.optimization_variables["EPS_U"], 
                self._discretization.optimization_variables["P"], 

            ])

        optimization_variables_parameters_applied = ci.veccat([ \

                self._discretization.optimization_variables["U"], 
                self._discretization.optimization_variables["Q"], 
                self._discretization.optimization_variables["X"], 
                ci.mx(*self._discretization.optimization_variables["EPS_U"].shape), 
                self._pdata, 

            ])

        equality_constraints_fcn = ci.mx_function( \
            "equality_constraints_fcn", \
            [optimization_variables_for_equality_constraints], \
            [self._discretization.equality_constraints])

        self._equality_constraints_parameters_applied = \
            equality_constraints_fcn(optimization_variables_parameters_applied)


    def _apply_parameters_to_discretization(self):

        self._apply_parameters_to_equality_constraints()


    def _set_optimization_variables(self):

        self._optimization_variables = ci.veccat([ \

                self._discretization.optimization_variables["U"],
                self._discretization.optimization_variables["Q"],
                self._discretization.optimization_variables["X"],

            ])


    def _set_optimization_variables_initials(self, qinit, x0, uinit):

        self.simulation = Simulation(self._discretization.system, \
            self._pdata, qinit)
        self.simulation.run_system_simulation(x0, \
            self._discretization.time_points, uinit, print_status = False)
        xinit = self.simulation.simulation_results

        repretitions_xinit = \
            self._discretization.optimization_variables["X"][:,:-1].shape[1] / \
                self._discretization.number_of_intervals
        
        Xinit = ci.repmat(xinit[:, :-1], repretitions_xinit, 1)

        Xinit = ci.horzcat([ \

            Xinit.reshape((self._discretization.system.nx, \
                Xinit.numel() / self._discretization.system.nx)),
            xinit[:, -1],

            ])

        uinit = inputchecks.check_controls_data(uinit, \
            self._discretization.system.nu, \
            self._discretization.number_of_intervals)
        Uinit = uinit

        qinit = inputchecks.check_constant_controls_data(qinit, \
            self._discretization.system.nq)
        Qinit = qinit

        self._optimization_variables_initials = ci.veccat([ \

                Uinit,
                Qinit,
                Xinit,

            ])


    def _set_optimization_variables_lower_bounds(self, umin, qmin, xmin, x0):

        umin_user_provided = umin

        umin = inputchecks.check_controls_data(umin, \
            self._discretization.system.nu, 1)

        if umin_user_provided is None:

            umin = -np.inf * np.ones(umin.shape)

        Umin = ci.repmat(umin, 1, \
            self._discretization.optimization_variables["U"].shape[1])


        qmin_user_provided = qmin

        qmin = inputchecks.check_constant_controls_data(qmin, \
            self._discretization.system.nq)

        if qmin_user_provided is None:

            qmin = -np.inf * np.ones(qmin.shape)

        Qmin = qmin


        xmin_user_provided = xmin

        xmin = inputchecks.check_states_data(xmin, \
            self._discretization.system.nx, 0)

        if xmin_user_provided is None:

            xmin = -np.inf * np.ones(xmin.shape)

        Xmin = ci.repmat(xmin, 1, \
            self._discretization.optimization_variables["X"].shape[1])

        Xmin[:,0] = x0


        self._optimization_variables_lower_bounds = ci.veccat([ \

                Umin,
                Qmin,
                Xmin,

            ])


    def _set_optimization_variables_upper_bounds(self, umax, qmax, xmax, x0):

        umax_user_provided = umax

        umax = inputchecks.check_controls_data(umax, \
            self._discretization.system.nu, 1)

        if umax_user_provided is None:

            umax = np.inf * np.ones(umax.shape)

        Umax = ci.repmat(umax, 1, \
            self._discretization.optimization_variables["U"].shape[1])


        qmax_user_provided = qmax

        qmax = inputchecks.check_constant_controls_data(qmax, \
            self._discretization.system.nq)

        if qmax_user_provided is None:

            qmax = -np.inf * np.ones(qmax.shape)

        Qmax = qmax


        xmax_user_provided = xmax

        xmax = inputchecks.check_states_data(xmax, \
            self._discretization.system.nx, 0)

        if xmax_user_provided is None:

            xmax = np.inf * np.ones(xmax.shape)

        Xmax = ci.repmat(xmax, 1, \
            self._discretization.optimization_variables["X"].shape[1])

        Xmax[:,0] = x0


        self._optimization_variables_upper_bounds = ci.veccat([ \

                Umax,
                Xmax,

            ])


    def _set_measurement_data(self):

        # The DOE problem does not depend on actual measurement values,
        # the measurement deviations are only needed to set up the objective;
        # therefore, dummy-values for the measurements can be used
        # (see issue #7 for further information)

        measurement_data = np.zeros((self._discretization.system.nphi, \
            self._discretization.number_of_intervals + 1))

        self._measurement_data_vectorized = ci.vec(measurement_data)


    def _set_weightings(self, wv, weps_u):

        input_error_weightings = \
            inputchecks.check_input_error_weightings(weps_u, \
            self._discretization.system.neps_u, 
            self._discretization.number_of_intervals)

        measurement_weightings = \
            inputchecks.check_measurement_weightings(wv, \
            self._discretization.system.nphi, \
            self._discretization.number_of_intervals + 1)

        self._weightings_vectorized = ci.veccat([ \

            input_error_weightings, 
            measurement_weightings,

            ])


    def _set_measurement_deviations(self):

        self._measurement_deviations = ci.vertcat([ \

                ci.vec(self._discretization.measurements) - \
                self._measurement_data_vectorized + \
                ci.vec(self._discretization.optimization_variables["V"])

            ])


    def _setup_constraints(self):

        self._constraints = ci.vertcat([ \

                self._measurement_deviations,
                self._discretization.equality_constraints,

            ])


    def _set_cov_matrix_derivative_directions(self):

        # These correspond to the optimization variables of the parameter
        # estimation problem; the evaluation of the covariance matrix, though,
        # does not depend on the actual values of V, EPS_E and EPS_U, and with
        # this, the DoE problem does not

        self._cov_matrix_derivative_directions = ci.veccat([ \

                self._discretization.optimization_variables["P"],
                self._discretization.optimization_variables["X"],
                self._discretization.optimization_variables["V"],
                self._discretization.optimization_variables["EPS_U"],

            ])


    def _setup_gauss_newton_lagrangian_hessian(self):

        gauss_newton_lagrangian_hessian_diag = ci.vertcat([ \
            ci.mx(self._cov_matrix_derivative_directions.shape[0] - \
                self._weightings_vectorized.shape[0], 1), \
            self._weightings_vectorized])

        self._gauss_newton_lagrangian_hessian = ci.diag( \
            gauss_newton_lagrangian_hessian_diag)


    def _setup_covariance_matrix_for_evaluation(self):

        covariance_matrix_free_variables = ci.veccat([ \

                self._discretization.optimization_variables["P"],
                self._discretization.optimization_variables["U"],
                self._discretization.optimization_variables["Q"],
                self._discretization.optimization_variables["X"],
                self._discretization.optimization_variables["EPS_U"],

            ])

        self._covariance_matrix_fcn = ci.mx_function("covariance_matrix_fcn", \
            [covariance_matrix_free_variables], \
            [self._covariance_matrix.covariance_matrix])


    def _apply_parameters_to_objective(self):

        # As mentioned above, the objective does not depend on the actual
        # values of V, but on the values of P and EPS_U, while
        # P is fed from pdata, and EPS_U is supposed to be 0

        objective_free_variables = ci.veccat([ \

                self._discretization.optimization_variables["P"],
                self._discretization.optimization_variables["U"],
                self._discretization.optimization_variables["Q"],
                self._discretization.optimization_variables["X"],
                self._discretization.optimization_variables["EPS_U"],

            ])

        objective_free_variables_parameters_applied = ci.veccat([ \

                self._pdata,
                self._discretization.optimization_variables["U"],
                self._discretization.optimization_variables["Q"],
                self._discretization.optimization_variables["X"],
                ci.mx(*self._discretization.optimization_variables["EPS_U"].shape),

            ])

        objective_fcn = ci.mx_function("objective_fcn", \
            [objective_free_variables], [self._objective_parameters_free])

        self._objective = objective_fcn( \
            objective_free_variables_parameters_applied)


    def __init__(self, system, time_points, \
        uinit = None, umin = None, umax = None, \
        qinit = None, qmin = None, qmax = None, \
        pdata = None, x0 = None, \
        xmin = None, xmax = None, \
        wv = None, weps_u = None, \
        discretization_method = "collocation", \
        optimality_criterion = "A", **kwargs):

        r'''
        :raises: AttributeError, NotImplementedError

        :param system: system considered for parameter estimation, specified
                       using the :class:`casiopeia.system.System` class
        :type system: casiopeia.system.System

        :param time_points: time points :math:`t_\text{N} \in \mathbb{R}^\text{N}`
                   used to discretize the continuous time problem. Controls
                   will be applied at the first :math:`N-1` time points,
                   while measurements take place at all :math:`N` time points.
        :type time_points: numpy.ndarray, casadi.DMatrix, list

        :param umin: optional, lower bounds of the time-varying controls
                   :math:`u_\text{min} \in \mathbb{R}^{\text{n}_\text{u}}`;
                   if not values are given, :math:`-\infty` will be used
        :type umin: numpy.ndarray, casadi.DMatrix

        :param umax: optional, upper bounds of the time-vaying controls 
                   :math:`u_\text{max} \in \mathbb{R}^{\text{n}_\text{u}}`;
                   if not values are given, :math:`\infty` will be used
        :type umax: numpy.ndarray, casadi.DMatrix

        :param uinit: optional, initial guess for the values of the time-varying controls
                   :math:`u_\text{N} \in \mathbb{R}^{\text{n}_\text{u} \times \text{N}-1}`
                   that (might) change at the switching time points;
                   if no values are given, 0 will be used; note that a poorly
                   or wrongly chosen initial guess can cause the optimization
                   to fail, and note that the
                   the second dimension of :math:`u_N` is :math:`N-1` and not
                   :math:`N`, since there is no control value applied at the
                   last time point
        :type uinit: numpy.ndarray, casadi.DMatrix

        :param qmin: optional, lower bounds of the time-constant controls
                   :math:`q_\text{min} \in \mathbb{R}^{\text{n}_\text{q}}`;
                   if not values are given, :math:`-\infty` will be used
        :type qmin: numpy.ndarray, casadi.DMatrix

        :param qmax: optional, upper bounds of the time-constant controls
                   :math:`q_\text{max} \in \mathbb{R}^{\text{n}_\text{q}}`;
                   if not values are given, :math:`\infty` will be used
        :type qmax: numpy.ndarray, casadi.DMatrix

        :param qinit: optional, initial guess for the optimal values of the
                   time-constant controls
                   :math:`q_\text{init} \in \mathbb{R}^{\text{n}_\text{q}}`;
                   if not values are given, 0 will be used; note that a poorly
                   or wrongly chosen initial guess can cause the optimization
                   to fail
        :type qinit: numpy.ndarray, casadi.DMatrix

        :param pdata: values of the time-constant parameters 
                      :math:`p \in \mathbb{R}^{\text{n}_\text{p}}`
        :type pdata: numpy.ndarray, casadi.DMatrix

        :param x0: state values :math:`x_0 \in \mathbb{R}^{\text{n}_\text{x}}`
                   at the first time point :math:`t_0`
        :type x0: numpy.ndarray, casadi.DMatrix, list

        :param xmin: optional, lower bounds of the states
                      :math:`x_\text{min} \in \mathbb{R}^{\text{n}_\text{x}}`;
                      if no value is given, :math:`-\infty` will be used
        :type xmin: numpy.ndarray, casadi.DMatrix

        :param xmax: optional, lower bounds of the states
                      :math:`x_\text{max} \in \mathbb{R}^{\text{n}_\text{x}}`;
                      if no value is given, :math:`\infty` will be used
        :type xmax: numpy.ndarray, casadi.DMatrix 

        :param wv: weightings for the measurements
                   :math:`w_\text{v} \in \mathbb{R}^{\text{n}_\text{y} \times \text{N}}`
        :type wv: numpy.ndarray, casadi.DMatrix

        :param weps_u: weightings for the input errors
                   :math:`w_{\epsilon_\text{u}} \in \mathbb{R}^{\text{n}_{\epsilon_\text{u}}}`
                   (only necessary
                   if input errors are used within ``system``)
        :type weps_u: numpy.ndarray, casadi.DMatrix    

        :param discretization_method: optional, the method that shall be used for
                                      discretization of the continuous time
                                      problem w. r. t. the time points given 
                                      in :math:`t_\text{N}`; possible values are
                                      "collocation" (default) and
                                      "multiple_shooting"
        :type discretization_method: str

        :param optimality_criterion: optional, the information function
                                    :math:`I_\text{X}(\cdot)` to be used on the 
                                    covariance matrix, possible values are
                                    `A` (default) and `D`, while

                                    .. math ::

                                        \begin{aligned}
                                          I_\text{A}(\Sigma_\text{p}) & = \frac{1}{n_\text{p}} \text{Tr}(\Sigma_\text{p}),\\
                                          I_\text{D}(\Sigma_\text{p}) & = \begin{vmatrix} \Sigma_\text{p} \end{vmatrix} ^{\frac{1}{n_\text{p}}},
                                        \end{aligned}

                                    for further information see e. g. [#f1]_

        :type optimality_criterion: str

        Depending on the discretization method specified in
        `discretization_method`, the following parameters can be used
        for further specification:

        :param collocation_scheme: optional, scheme used for setting up the
                                   collocation polynomials,
                                   possible values are `radau` (default)
                                   and `legendre`
        :type collocation_scheme: str

        :param number_of_collocation_points: optional, order of collocation
                                             polynomials
                                             :math:`d \in \mathbb{Z}` (default
                                             values is 3)
        :type number_of_collocation_points: int


        :param integrator: optional, integrator to be used with multiple shooting.
                           See the CasADi documentation for a list of
                           all available integrators. As a default, `cvodes`
                           is used.
        :type integrator: str

        :param integrator_options: optional, options to be passed to the CasADi
                                   integrator used with multiple shooting
                                   (see the CasADi documentation for a list of
                                   all possible options)
        :type integrator_options: dict

        You do not need to specify initial guesses for the estimated states,
        since these are obtained with a system simulation using the initial
        states and the provided initial guesses for the controls.

        The resulting optimization problem has the following form:

        .. math::

            \begin{aligned}
                \text{arg}\,\underset{u, q, x}{\text{min}} & & I(\Sigma_{\text{p}}(x, u, q; p)) &\\
                \text{subject to:} & & g(x, u, q; p) & = 0\\
                & & u_\text{min} \leq u_\text{k} & \leq u_\text{max} \hspace{1cm} k = 1, \dots, N-1\\
                & & x_\text{min} \leq x_\text{k}  & \leq x_\text{max} \hspace{1cm} k = 1, \dots, N\\
                & & x_1 \leq x(t_1) & \leq x_1
            \end{aligned}

        where :math:`\Sigma_p = \text{Cov}(p)` and :math:`g(\cdot)` contains the
        discretized system dynamics
        according to the specified discretization method. If the system is
        non-dynamic, it only contains the user-provided equality constraints.

        .. rubric:: References

        .. [#f1] |linkf1|_
        
        .. _linkf1: http://ginger.iwr.uni-heidelberg.de/vplan/images/5/54/Koerkel2002.pdf

        .. |linkf1| replace:: *Körkel, Stefan: Numerische Methoden für Optimale Versuchsplanungsprobleme bei nichtlinearen DAE-Modellen, PhD Thesis, Heidelberg university, 2002, pages 74/75.*

        '''

        intro()

        self._discretize_system( \
            system, time_points, discretization_method, **kwargs)

        self._set_parameter_guess(pdata)

        self._apply_parameters_to_discretization()

        self._set_optimization_variables()

        self._set_optimization_variables_initials(qinit, x0, uinit)

        self._set_optimization_variables_lower_bounds(umin, qmin, xmin, x0)

        self._set_optimization_variables_upper_bounds(umax, qmax, xmax, x0)

        self._set_measurement_data()

        self._set_weightings(wv, weps_u)

        self._set_measurement_deviations()

        self._set_cov_matrix_derivative_directions()

        self._setup_constraints()

        self._setup_gauss_newton_lagrangian_hessian()

        self._setup_covariance_matrix()

        self._setup_covariance_matrix_for_evaluation()

        self._set_optimiality_criterion(optimality_criterion)

        self._setup_objective()

        self._apply_parameters_to_objective()

        self._setup_nlp()


    def _print_experimental_properties(self, covariance_matrix):

        np.set_printoptions(linewidth = 200, \
            formatter={'float': lambda x: format(x, ' 10.8e')})

        print("\nParameters p_i:")

        for k, pk in enumerate(self._pdata):

            print("    p_{0:<3} = {1} +/- {2}".format( \
                 k, pk, ci.sqrt(abs(ci.diag(covariance_matrix)[k]))))

        print("\nCovariance matrix for this setup:")

        print(np.atleast_2d(covariance_matrix))


    def _compute_initial_covariance_matrix(self):

        covariance_matrix_initial_input = ci.veccat([ \

                self._pdata,
                self._optimization_variables_initials,
                np.zeros(self._discretization.optimization_variables["EPS_U"].shape)

            ])


        self._covariance_matrix_initial = self._covariance_matrix_fcn( \
            covariance_matrix_initial_input)



    def _compute_optimized_covariance_matrix(self):

        covariance_matrix_optimized_input = ci.veccat([ \

                self._pdata,
                self.design_results["x"],
                np.zeros(self._discretization.optimization_variables["EPS_U"].shape)

            ])

        self._covariance_matrix_optimized = self._covariance_matrix_fcn( \
            covariance_matrix_optimized_input)


    def plot_confidence_ellipsoids(self, properties = "initial"):

        r'''
        :param properties: Set whether the experimental properties for the
                           initial setup ("initial", default), the optimized setup
                           ("optimized") or for both setups ("all") shall be
                           plotted. In the later case, both ellipsoids for one
                           pair of parameters will be displayed within one plot.
        :type properties: str

        Plot confidence ellipsoids for all parameter pairs. 
        Since the number of plots is possibly big, all plots will be saved
        within a folder *confidence_ellipsoids_scriptname* in you current
        working directory rather than being displayed directly.

        '''

        self._plot_confidence_ellipsoids(pdata = self._pdata, \
            properties = properties)


class MultiDoEProblem(DoEProblem):

    __metaclass__ = ABCMeta

    @property
    def optimized_controls(self):

        starting_position_design_results = 0

        optimized_controls = []

        for doe_setup in self._doe_setups:

            optimized_controls.append(self.design_results["x"][ \
                starting_position_design_results : \
                starting_position_design_results + \
                (doe_setup._discretization.number_of_intervals * \
                    doe_setup._discretization.system.nu)])

            starting_position_design_results += \
                doe_setup._optimization_variables.shape[0]

        return optimized_controls


    def _define_set_of_doe_setups(self, doe_setups):

        inputchecks.check_multi_doe_input(doe_setups)

        self._doe_setups = doe_setups


    def _get_discretization_from_doe_setups(self):

        self._discretization = self._doe_setups[0]._discretization


    def _merge_equaltiy_constraints_parameters_applied(self):

        equality_constraints_parameters_applied = [ \
            doe_setup._equality_constraints_parameters_applied for \
            doe_setup in self._doe_setups]

        self._equality_constraints_parameters_applied = \
            ci.veccat(equality_constraints_parameters_applied)


    def _merge_optimization_variables(self):

        optimization_variables = [ \
            doe_setup._optimization_variables for \
            doe_setup in self._doe_setups]

        self._optimization_variables = ci.vertcat(optimization_variables)


    def _merge_optimization_variables_initials(self):

        optimization_variables_initials = [ \
            doe_setup._optimization_variables_initials for \
            doe_setup in self._doe_setups]

        self._optimization_variables_initials = \
            ci.vertcat(optimization_variables_initials)


    def _merge_optimization_variables_lower_bounds(self):

        optimization_variables_lower_bounds = [ \
            doe_setup._optimization_variables_lower_bounds for \
            doe_setup in self._doe_setups]

        self._optimization_variables_lower_bounds = \
            ci.vertcat(optimization_variables_lower_bounds)


    def _merge_optimization_variables_upper_bounds(self):

        optimization_variables_upper_bounds = [ \
            doe_setup._optimization_variables_upper_bounds for \
            doe_setup in self._doe_setups]

        self._optimization_variables_upper_bounds = \
            ci.vertcat(optimization_variables_upper_bounds)


    def _merge_and_setup_constraints(self):

        constraints = [self._doe_setups[0]._constraints]

        for k, doe_setup in enumerate(self._doe_setups[:-1]):

            # Connect estimation problems by setting p_k - p_k+1 = 0

            constraints.append(doe_setup._discretization.optimization_variables["P"] - \
                self._doe_setups[k+1]._discretization.optimization_variables["P"])

            constraints.append(self._doe_setups[k+1]._constraints)

        self._constraints = ci.vertcat(constraints)


    def _setup_covariance_matrix_for_evaluation(self):

        covariance_matrix_free_variables = []

        for doe_setup in self._doe_setups:

            for key in ["P", "U", "Q", "X"]:

                covariance_matrix_free_variables.append( \
                    doe_setup._discretization.optimization_variables[key])

        covariance_matrix_free_variables = ci.veccat( \
            covariance_matrix_free_variables)

        self._covariance_matrix_fcn = ci.mx_function("covariance_matrix_fcn", \
            [covariance_matrix_free_variables], \
            [self._covariance_matrix.covariance_matrix])


    def _apply_parameters_to_objective(self):

        objective_free_variables = []
        objective_free_variables_parameters_applied = []

        for doe_setup in self._doe_setups:

            objective_free_variables.append( \
                doe_setup._discretization.optimization_variables["P"])
            objective_free_variables_parameters_applied.append( \
                doe_setup._pdata)

            for key in ["U", "Q", "X"]:

                objective_free_variables.append( \
                    doe_setup._discretization.optimization_variables[key])
                objective_free_variables_parameters_applied.append( \
                    doe_setup._discretization.optimization_variables[key])
                
        objective_free_variables = ci.veccat(objective_free_variables)
        objective_free_variables_parameters_applied = \
            ci.veccat(objective_free_variables_parameters_applied)

        objective_fcn = ci.mx_function("objective_fcn", \
            [objective_free_variables], [self._objective_parameters_free])

        self._objective = objective_fcn( \
            objective_free_variables_parameters_applied)


    @abstractmethod
    def __init__(self, doe_setups, optimality_criterion):

        self._define_set_of_doe_setups(doe_setups)

        self._get_discretization_from_doe_setups()

        self._merge_equaltiy_constraints_parameters_applied()

        self._merge_optimization_variables()

        self._merge_optimization_variables_initials()

        self._merge_optimization_variables_lower_bounds()

        self._merge_optimization_variables_upper_bounds()

        self._merge_and_setup_constraints()

        self._set_optimiality_criterion(optimality_criterion)


    def _print_experimental_properties(self, covariance_matrix):

        np.set_printoptions(linewidth = 200, \
            formatter={'float': lambda x: format(x, ' 10.8e')})

        print("\nParameters p_i:")

        for k, pk in enumerate(self._doe_setups[0]._pdata):

            print("    p_{0:<3} = {1} +/- {2}".format( \
                 k, pk, ci.sqrt(abs(ci.diag(covariance_matrix)[k]))))

        print("\nCovariance matrix for this setup:")

        print(np.atleast_2d(covariance_matrix))


    def _compute_initial_covariance_matrix(self):

        covariance_matrix_initial_input = []

        for doe_setup in self._doe_setups:

            covariance_matrix_initial_input.append(doe_setup._pdata)
            covariance_matrix_initial_input.append( \
                doe_setup._optimization_variables_initials)

        covariance_matrix_initial_input = ci.veccat( \
            covariance_matrix_initial_input)


        self._covariance_matrix_initial = self._covariance_matrix_fcn( \
            covariance_matrix_initial_input)


    def _compute_optimized_covariance_matrix(self):

        starting_position_design_results = 0

        covariance_matrix_optimized_input = []

        for doe_setup in self._doe_setups:

            covariance_matrix_optimized_input.append(doe_setup._pdata)

            covariance_matrix_optimized_input.append(self.design_results["x"][ \
                starting_position_design_results : \
                starting_position_design_results + \
                doe_setup._optimization_variables.shape[0]])

            starting_position_design_results += \
                doe_setup._optimization_variables.shape[0]

        covariance_matrix_optimized_input = ci.veccat( \
            covariance_matrix_optimized_input)

        self._covariance_matrix_optimized = self._covariance_matrix_fcn( \
            covariance_matrix_optimized_input)


    def plot_confidence_ellipsoids(self, properties = "initial"):

        r'''
        :param properties: Set whether the experimental properties for the
                           initial setup ("initial", default), the optimized setup
                           ("optimized") or for both setups ("all") shall be
                           plotted. In the later case, both ellipsoids for one
                           pair of parameters will be displayed within one plot.
        :type properties: str

        Plot confidence ellipsoids for all parameter pairs. 
        Since the number of plots is possibly big, all plots will be saved
        within a folder *confidence_ellipsoids_scriptname* in you current
        working directory rather than being displayed directly.

        '''

        self._plot_confidence_ellipsoids(pdata = self._doe_setups[0]._pdata, \
            properties = properties)


class MultiDoESingleKKT(MultiDoEProblem):

    def _merge_cov_matrix_derivative_directions(self):

        cov_matrix_derivative_directions = \
            [doe_setup._cov_matrix_derivative_directions for
            doe_setup in self._doe_setups]

        self._cov_matrix_derivative_directions = \
            ci.vertcat(cov_matrix_derivative_directions)


    def _merge_gauss_newton_lagrangian_hessians(self):

        gauss_newton_lagrangian_hessian = \
            [doe_setup._gauss_newton_lagrangian_hessian for \
                doe_setup in self._doe_setups]

        self._gauss_newton_lagrangian_hessian = \
            ci.diagcat(gauss_newton_lagrangian_hessian)


    def __init__(self, doe_setups = [], optimality_criterion = "A"):

        r'''
        :param doe_setups: list of two or more objects of type :class:`casiopeia.doe.DoE`
        :type doe_setups: list

        '''

        super(MultiDoESingleKKT, self).__init__(doe_setups, \
            optimality_criterion)

        self._merge_cov_matrix_derivative_directions()

        self._merge_gauss_newton_lagrangian_hessians()

        self._setup_covariance_matrix()

        self._setup_covariance_matrix_for_evaluation()

        self._setup_objective()

        self._apply_parameters_to_objective()

        self._setup_nlp()


class MultiDoEMultiKKT(MultiDoEProblem):

    def _setup_fisher_matrix(self):

        fisher_matrices = []

        for doe_setup in self._doe_setups:

            kkt_matrix = KKTMatrix( \
                doe_setup._gauss_newton_lagrangian_hessian, \
                doe_setup._constraints, \
                doe_setup._cov_matrix_derivative_directions)

            fisher_matrix = FisherMatrix(kkt_matrix.kkt_matrix, \
                doe_setup._discretization.system.np)

            fisher_matrices.append(fisher_matrix.fisher_matrix)

        self._fisher_matrix = sum(fisher_matrices)


    def _setup_covariance_matrix(self):

        self._covariance_matrix = CovarianceMatrix(self._fisher_matrix)


    def __init__(self, doe_setups = [], optimality_criterion = "A"):

        r'''
        :param doe_setups: list of two or more objects of type :class:`casiopeia.doe.DoE`
        :type doe_setups: list

        '''

        super(MultiDoEMultiKKT, self).__init__(doe_setups, \
            optimality_criterion)

        self._setup_fisher_matrix()

        self._setup_covariance_matrix()

        self._setup_covariance_matrix_for_evaluation()

        self._setup_objective()

        self._apply_parameters_to_objective()

        self._setup_nlp()


class MultiDoE(MultiDoEMultiKKT):

    '''The class :class:`casiopeia.doe.MultiDoE` is used to construct a single
    experimental design problem from multiple experimental design problems
    defined via two or more objects of type :class:`casiopeia.doe.DoE`.

    This provides the possibility to design multiple experiments within one
    single optimization, so that the several experiments can focus on different
    aspects of the system which in combination then yields more information
    about the complete system.

    Also, this functionality is in particular useful in case an experiment is
    limited to only small variable bounds, small time horizons, highly
    depends on the initialization of the system, or any other case when a single
    experiment might not be enough to capture enough information about a system.

    .. note::

        It is assumed that the system description used for setting up
        the several experimental design problems is the same!
    '''

    pass
