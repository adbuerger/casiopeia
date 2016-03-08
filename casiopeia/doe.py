#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2014-2016 Adrian Bürger
#
# This file is part of casiopeia.
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
import time

from discretization.nodiscretization import NoDiscretization
from discretization.odecollocation import ODECollocation
from discretization.odemultipleshooting import ODEMultipleShooting

from interfaces import casadi_interface as ci
from covariance_matrix import CovarianceMatrix, setup_a_criterion, \
    setup_d_criterion
from intro import intro
from sim import Simulation

import inputchecks

class DoE(object):

    '''The class :class:`casiopeia.pe.DoE` is used to set up
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
    def design_results(self):

        try:

            return self._design_results

        except AttributeError:

            raise AttributeError('''
An experimental design has to be executed before the design results
can be accessed, please run run_experimental_design() first.
''')


    @property
    def optimized_controls(self):

        try:

            return self._design_results["x"][ \
                :(self._discretization.number_of_intervals * \
                    self._discretization.system.nu)]

        except AttributeError:

            raise AttributeError('''
An experimental design has to be executed before the optimized controls
can be accessed, please run run_experimental_design() first.
''')


    def _discretize_system(self, system, time_points, discretization_method, \
        **kwargs):

        if system.nx == 0 and system.nz == 0:

            self._discretization = NoDiscretization(system, time_points)

        elif system.nx != 0 and system.nz == 0:

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

        elif system.nx != 0 and system.nz != 0:

            raise NotImplementedError('''
Support of implicit DAEs is not implemented yet,
but will be in future versions.
''')            


    def _apply_parameters_to_equality_constraints(self, pdata):

        udata = inputchecks.check_parameter_data(pdata, \
            self._discretization.system.np)

        optimization_variables_for_equality_constraints = ci.veccat([ \

                self._discretization.optimization_variables["U"],
                self._discretization.optimization_variables["Q"],
                self._discretization.optimization_variables["X"], 
                self._discretization.optimization_variables["EPS_U"], 
                self._discretization.optimization_variables["EPS_E"], 
                self._discretization.optimization_variables["P"], 

            ])

        optimization_variables_parameters_applied = ci.veccat([ \

                self._discretization.optimization_variables["U"], 
                self._discretization.optimization_variables["Q"], 
                self._discretization.optimization_variables["X"], 
                self._discretization.optimization_variables["EPS_U"], 
                self._discretization.optimization_variables["EPS_E"], 
                pdata, 

            ])

        equality_constraints_fcn = ci.mx_function( \
            "equality_constraints_fcn", \
            [optimization_variables_for_equality_constraints], \
            [self._discretization.equality_constraints])

        [self._equality_constraints_parameters_applied] = \
            equality_constraints_fcn([optimization_variables_parameters_applied])


    def _apply_parameters_to_discretization(self, pdata):

        self._apply_parameters_to_equality_constraints(pdata)


    def _set_optimization_variables(self):

        self._optimization_variables = ci.veccat([ \

                self._discretization.optimization_variables["U"],
                self._discretization.optimization_variables["Q"],
                self._discretization.optimization_variables["X"],

            ])


    def _set_optimization_variables_initials(self, pdata, qinit, x0, uinit):

        self.simulation = Simulation(self._discretization.system, pdata, qinit)
        self.simulation.run_system_simulation(x0, \
            self._discretization.time_points, uinit)
        xinit = self.simulation.simulation_results

        repretitions_xinit = \
            self._discretization.optimization_variables["X"][:,:-1].shape[1] / \
                self._discretization.number_of_intervals
        
        Xinit = ci.repmat(xinit[:, :-1], repretitions_xinit, 1)

        Xinit = ci.horzcat([ \

            Xinit.reshape((self._discretization.system.nx, \
                Xinit.size() / self._discretization.system.nx)),
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

        # The DOE problem does not depend on actual emasurement values,
        # the measurement deviations are only needed to set up the objective;
        # therefore, dummy-values for the measurements can be used
        # (see issue #7 for further information)

        measurement_data = np.zeros((self._discretization.system.nphi, \
            self._discretization.number_of_intervals + 1))

        self._measurement_data_vectorized = ci.vec(measurement_data)


    def _set_weightings(self, wv, weps_e, weps_u):

        measurement_weightings = \
            inputchecks.check_measurement_weightings(wv, \
            self._discretization.system.nphi, \
            self._discretization.number_of_intervals + 1)

        equation_error_weightings = \
            inputchecks.check_equation_error_weightings(weps_e, \
            self._discretization.system.neps_e)

        input_error_weightings = \
            inputchecks.check_input_error_weightings(weps_u, \
            self._discretization.system.neps_u)

        self._weightings_vectorized = ci.veccat([ \

            measurement_weightings,
            equation_error_weightings,
            input_error_weightings, 

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
                self._discretization.optimization_variables["EPS_E"],
                self._discretization.optimization_variables["EPS_U"],

            ])


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


    def _setup_gauss_newton_lagrangian_hessian(self):

        gauss_newton_lagrangian_hessian_diag = ci.vertcat([ \
            ci.mx(self._cov_matrix_derivative_directions.shape[0] - \
                self._weightings_vectorized.shape[0], 1), \
            self._weightings_vectorized])

        self.gauss_newton_lagrangian_hessian = ci.diag( \
            gauss_newton_lagrangian_hessian_diag)


    def _setup_covariance_matrix(self):

        self._cm = CovarianceMatrix(self.gauss_newton_lagrangian_hessian, \
            self._constraints, self._cov_matrix_derivative_directions, \
            self._discretization.system.np)


    def _setup_objective(self):

        self._covariance_matrix_symbolic = \
            self._cm.covariance_matrix_for_evaluation

        self._objective_parameters_free = \
            self._optimality_criterion(self._covariance_matrix_symbolic)


    def _apply_parameters_to_objective(self, pdata):

        # As mentioned above, the objective does not depend on the actual
        # values of V, EPS_E and EPS_U, but on the values of P

        objective_free_variables = ci.veccat([ \

                self._discretization.optimization_variables["P"],
                self._discretization.optimization_variables["U"],
                self._discretization.optimization_variables["Q"],
                self._discretization.optimization_variables["X"],

            ])

        objective_free_variables_parameters_applied = ci.veccat([ \

                pdata,
                self._discretization.optimization_variables["U"],
                self._discretization.optimization_variables["Q"],
                self._discretization.optimization_variables["X"],

            ])

        objective_fcn = ci.mx_function("objective_fcn", \
            [objective_free_variables], [self._objective_parameters_free])

        [self._objective] = objective_fcn( \
            [objective_free_variables_parameters_applied])


    def _setup_nlp(self):

        self._nlp = ci.mx_function("nlp", \
            ci.nlpIn(x = self._optimization_variables), \
            ci.nlpOut(f = self._objective, \
                g = self._equality_constraints_parameters_applied))


    def __init__(self, system, time_points, \
        uinit = None, umin = None, umax = None, \
        qinit = None, qmin = None, qmax = None, \
        pdata = None, x0 = None, \
        xmin = None, xmax = None, \
        wv = None, weps_e = None, weps_u = None, \
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

        :param weps_e: weightings for equation errors
                   :math:`w_{\epsilon_\text{e}} \in \mathbb{R}^{\text{n}_{\epsilon_\text{e}}}`
                   (only necessary 
                   if equation errors are used within ``system``)
        :type weps_e: numpy.ndarray, casadi.DMatrix    

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

        self._apply_parameters_to_discretization(pdata)

        self._set_optimization_variables()

        self._set_optimization_variables_initials(pdata, qinit, x0, uinit)

        self._set_optimization_variables_lower_bounds(umin, qmin, xmin, x0)

        self._set_optimization_variables_upper_bounds(umax, qmax, xmax, x0)

        self._set_measurement_data()

        self._set_weightings(wv, weps_e, weps_u)

        self._set_measurement_deviations()

        self._set_cov_matrix_derivative_directions()

        self._setup_constraints()

        self._setup_gauss_newton_lagrangian_hessian()

        self._setup_covariance_matrix()

        self._set_optimiality_criterion(optimality_criterion)

        self._setup_objective()

        self._apply_parameters_to_objective(pdata)

        self._setup_nlp()


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

        self._tend_optimum_experimental_deisng = time.time()
        self._duration_optimum_experimental_design = \
            self._tend_optimum_experimental_deisng - \
            self._tstart_optimum_experimental_design
