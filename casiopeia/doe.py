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

'''The module ``casiopeia.doe`` contains the class used for optimum experimental
design.'''

import numpy as np
import time

from discretization.nodiscretization import NoDiscretization
from discretization.odecollocation import ODECollocation
from discretization.odemultipleshooting import ODEMultipleShooting

from interfaces import casadi_interface as ci
from covariance_matrix import setup_covariance_matrix, setup_a_criterion, \
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

            return self.__design_results

        except AttributeError:

            raise AttributeError('''
An experimental design has to be executed before the design results
can be accessed, please run run_experimental_design() first.
''')


    @property
    def optimized_controls(self):

        try:

            return self.__design_results["x"][ \
                :(self.__discretization.number_of_intervals * \
                    self.__discretization.system.nu)]

        except AttributeError:

            raise AttributeError('''
An experimental design has to be executed before the optimized controls
can be accessed, please run run_experimental_design() first.
''')


    def __discretize_system(self, system, time_points, discretization_method, \
        **kwargs):

        if system.nx == 0 and system.nz == 0:

            self.__discretization = NoDiscretization(system, time_points)

        elif system.nx != 0 and system.nz == 0:

            if discretization_method == "collocation":

                self.__discretization = ODECollocation( \
                    system, time_points, **kwargs)

            elif discretization_method == "multiple_shooting":

                self.__discretization = ODEMultipleShooting( \
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


    def __apply_parameters_to_equality_constraints(self, pdata):

        udata = inputchecks.check_parameter_data(pdata, \
            self.__discretization.system.np)

        optimization_variables_for_equality_constraints = ci.veccat([ \

                self.__discretization.optimization_variables["U"], 
                self.__discretization.optimization_variables["X"], 
                self.__discretization.optimization_variables["EPS_U"], 
                self.__discretization.optimization_variables["EPS_E"], 
                self.__discretization.optimization_variables["P"], 

            ])

        optimization_variables_parameters_applied = ci.veccat([ \

                self.__discretization.optimization_variables["U"], 
                self.__discretization.optimization_variables["X"], 
                self.__discretization.optimization_variables["EPS_U"], 
                self.__discretization.optimization_variables["EPS_E"], 
                pdata, 

            ])

        equality_constraints_fcn = ci.mx_function( \
            "equality_constraints_fcn", \
            [optimization_variables_for_equality_constraints], \
            [self.__discretization.equality_constraints])

        [self.__equality_constraints_parameters_applied] = \
            equality_constraints_fcn([optimization_variables_parameters_applied])


    def __apply_parameters_to_discretization(self, pdata):

        self.__apply_parameters_to_equality_constraints(pdata)


    def __set_optimization_variables(self):

        self.__optimization_variables = ci.veccat([ \

                self.__discretization.optimization_variables["U"],
                self.__discretization.optimization_variables["X"],

            ])


    def __set_optimization_variables_initials(self, pdata, x0, uinit):

        self.simulation = Simulation(self.__discretization.system, pdata)
        self.simulation.run_system_simulation(x0, \
            self.__discretization.time_points, uinit)
        xinit = self.simulation.simulation_results

        repretitions_xinit = \
            self.__discretization.optimization_variables["X"][:,:-1].shape[1] / \
                self.__discretization.number_of_intervals
        
        Xinit = ci.repmat(xinit[:, :-1], repretitions_xinit, 1)

        Xinit = ci.horzcat([ \

            Xinit.reshape((self.__discretization.system.nx, \
                Xinit.size() / self.__discretization.system.nx)),
            xinit[:, -1],

            ])

        uinit = inputchecks.check_controls_data(uinit, \
            self.__discretization.system.nu, \
            self.__discretization.number_of_intervals)
        Uinit = uinit

        self.__optimization_variables_initials = ci.veccat([ \

                Uinit,
                Xinit,

            ])


    def __set_optimization_variables_lower_bounds(self, umin, xmin, x0):

        umin_user_provided = umin

        umin = inputchecks.check_controls_data(umin, \
            self.__discretization.system.nu, 1)

        if umin_user_provided is None:

            umin = -np.inf * np.ones(umin.shape)

        Umin = ci.repmat(umin, 1, \
            self.__discretization.optimization_variables["U"].shape[1])


        xmin_user_provided = xmin

        xmin = inputchecks.check_states_data(xmin, \
            self.__discretization.system.nx, 0)

        if xmin_user_provided is None:

            xmin = -np.inf * np.ones(xmin.shape)

        Xmin = ci.repmat(xmin, 1, \
            self.__discretization.optimization_variables["X"].shape[1])

        Xmin[:,0] = x0


        self.__optimization_variables_lower_bounds = ci.veccat([ \

                Umin,
                Xmin,

            ])


    def __set_optimization_variables_upper_bounds(self, umax, xmax, x0):

        umax_user_provided = umax

        umax = inputchecks.check_controls_data(umax, \
            self.__discretization.system.nu, 1)

        if umax_user_provided is None:

            umax = np.inf * np.ones(umax.shape)

        Umax = ci.repmat(umax, 1, \
            self.__discretization.optimization_variables["U"].shape[1])


        xmax_user_provided = xmax

        xmax = inputchecks.check_states_data(xmax, \
            self.__discretization.system.nx, 0)

        if xmax_user_provided is None:

            xmax = np.inf * np.ones(xmax.shape)

        Xmax = ci.repmat(xmax, 1, \
            self.__discretization.optimization_variables["X"].shape[1])

        Xmax[:,0] = x0


        self.__optimization_variables_upper_bounds = ci.veccat([ \

                Umax,
                Xmax,

            ])


    def __set_measurement_data(self):

        measurement_data = inputchecks.check_measurement_data( \
            self.simulation.simulation_results, \
            self.__discretization.system.nphi, \
            self.__discretization.number_of_intervals + 1)
        self.__measurement_data_vectorized = ci.vec(measurement_data)


    def __set_weightings(self, wv, weps_e, weps_u):

        measurement_weightings = \
            inputchecks.check_measurement_weightings(wv, \
            self.__discretization.system.nphi, \
            self.__discretization.number_of_intervals + 1)

        equation_error_weightings = \
            inputchecks.check_equation_error_weightings(weps_e, \
            self.__discretization.system.neps_e)

        input_error_weightings = \
            inputchecks.check_input_error_weightings(weps_u, \
            self.__discretization.system.neps_u)

        self.__weightings_vectorized = ci.veccat([ \

            measurement_weightings,
            equation_error_weightings,
            input_error_weightings, 

            ])


    def __set_measurement_deviations(self):

        self.__measurement_deviations = ci.vertcat([ \

                ci.vec(self.__discretization.measurements) - \
                self.__measurement_data_vectorized + \
                ci.vec(self.__discretization.optimization_variables["V"])

            ])


    def __setup_constraints(self):

        self.__constraints = ci.vertcat([ \

                self.__measurement_deviations,
                self.__discretization.equality_constraints,

            ])


    def __set_cov_matrix_derivative_directions(self):

        # These correspond to the optimization variables of the parameter
        # estimation problem; the evaluation of the covariance matrix, though,
        # does not depend on the actual values of V, EPS_E and EPS_U, and with
        # this, the DoE problem does not

        self.__cov_matrix_derivative_directions = ci.veccat([ \

                self.__discretization.optimization_variables["P"],
                self.__discretization.optimization_variables["X"],
                self.__discretization.optimization_variables["V"],
                self.__discretization.optimization_variables["EPS_E"],
                self.__discretization.optimization_variables["EPS_U"],

            ])

    def __set_optimiality_criterion(self, optimality_criterion):

            if str(optimality_criterion).upper() == "A":

                self.__optimality_criterion = setup_a_criterion

            elif str(optimality_criterion).upper() == "D":

                self.__optimality_criterion = setup_d_criterion

            else:

                raise NotImplementedError('''
Unknown optimality criterion: {0}.
Possible values are "A" and "D".
'''.format(str(discretization_method)))


    def __setup_objective(self):

        self.__covariance_matrix_symbolic = setup_covariance_matrix( \
                self.__cov_matrix_derivative_directions, \
                self.__weightings_vectorized, \
                self.__constraints, self.__discretization.system.np)

        self.__objective_parameters_free = \
            self.__optimality_criterion(self.__covariance_matrix_symbolic)


    def __apply_parameters_to_objective(self, pdata):

        # As mentioned above, the objective does not depend on the actual
        # values of V, EPS_E and EPS_U, but on the values of P

        objective_free_variables = ci.veccat([ \

                self.__discretization.optimization_variables["P"],
                self.__discretization.optimization_variables["U"],
                self.__discretization.optimization_variables["X"],

            ])

        objective_free_variables_parameters_applied = ci.veccat([ \

                pdata,
                self.__discretization.optimization_variables["U"],
                self.__discretization.optimization_variables["X"],

            ])

        objective_fcn = ci.mx_function("objective_fcn", \
            [objective_free_variables], [self.__objective_parameters_free])

        [self.__objective] = objective_fcn( \
            [objective_free_variables_parameters_applied])


    def __setup_nlp(self):

        self.__nlp = ci.mx_function("nlp", \
            ci.nlpIn(x = self.__optimization_variables), \
            ci.nlpOut(f = self.__objective, \
                g = self.__equality_constraints_parameters_applied))


    def __init__(self, system, time_points, \
        uinit = None, umin = None, umax = None, \
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

        :param time_points: time points :math:`t_N \in \mathbb{R}^{N}`
                   used to discretize the continuous time problem. Controls
                   will be applied at the first :math:`N-1` time points,
                   while measurements take place at all :math:`N` time points.
        :type time_points: numpy.ndarray, casadi.DMatrix, list

        :param uinit: optional, initial guess for the optimal values of the
                   controls at the switching time
                   points :math:`u_{init} \in \mathbb{R}^{n_u \times N-1}`;
                   if not values are given, 0 will be used; note that a poorly
                   or wrongly chosen initial guess can cause the optimization
                   to fail, and note that the
                   the second dimension of :math:`u_N` is :math:`N-1` and not
                   :math:`N`, since there is no control value applied at the
                   last time point
        :type uinit: numpy.ndarray, casadi.DMatrix

        :param umin: optional, lower bounds of the
                   controls :math:`u_{min} \in \mathbb{R}^{n_u \times N-1}`;
                   if not values are given, :math:`-\infty` will be used
        :type umin: numpy.ndarray, casadi.DMatrix

        :param umax: optional, upper bounds of the
                   controls :math:`u_max \in \mathbb{R}^{n_u \times N-1}`;
                   if not values are given, :math:`\infty` will be used
        :type umax: numpy.ndarray, casadi.DMatrix

        :param pdata: values of the time-constant parameters 
                      :math:`p \in \mathbb{R}^{n_p}`
        :type pdata: numpy.ndarray, casadi.DMatrix

        :param x0: state values :math:`x_0 \in \mathbb{R}^{n_x}`
                   at the first time point :math:`t_0`
        :type x0: numpy.ndarray, casadi.DMatrix, list

        :param xmin: optional, lower bounds of the states
                      :math:`x_{min} \in \mathbb{R}^{n_x \times N}`;
                      if no value is given, :math:`-\infty` will be used
        :type xmin: numpy.ndarray, casadi.DMatrix

        :param xmax: optional, lower bounds of the states
                      :math:`x_{max} \in \mathbb{R}^{n_x \times N}`;
                      if no value is given, :math:`\infty` will be used
        :type xmax: numpy.ndarray, casadi.DMatrix 

        :param wv: weightings for the measurements
                   :math:`w_v \in \mathbb{R}^{n_y \times N}`
        :type wv: numpy.ndarray, casadi.DMatrix    

        :param weps_e: weightings for equation errors
                   :math:`w_{\epsilon_e} \in \mathbb{R}^{n_{\epsilon_e}}`
                   (only necessary 
                   if equation errors are used within ``system``)
        :type weps_e: numpy.ndarray, casadi.DMatrix    

        :param weps_u: weightings for the input errors
                   :math:`w_{\epsilon_u} \in \mathbb{R}^{n_{\epsilon_u}}`
                   (only necessary
                   if input errors are used within ``system``)
        :type weps_u: numpy.ndarray, casadi.DMatrix    

        :param discretization_method: optional, the method that shall be used for
                                      discretization of the continuous time
                                      problem w. r. t. the time points given 
                                      in :math:`t_N`; possible values are
                                      "collocation" (default) and
                                      "multiple_shooting"
        :type discretization_method: str

        :param optimality_criterion: optional, the information function
                                    :math:`I_X(\cdot)` to be used on the 
                                    covariance matrix, possible values are
                                    `A` (default) and `D`, while

                                    .. math ::

                                        \begin{aligned}
                                          I_A(\Sigma_p) & = \frac{1}{n_p} \text{Tr}(\Sigma_p),\\
                                          I_D(\Sigma_p) & = \begin{vmatrix} \Sigma_p \end{vmatrix} ^{\frac{1}{n_p}},
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
                \text{arg}\,\underset{u, x}{\text{min}} & & I(\Sigma_{p}(x, u)) &\\
                \text{subject to:} & & \text{Cov}(p) & = \Sigma_p\\
                & & g(p, x, v, \epsilon_e, \epsilon_u) & = 0\\
                & & u_{min} \leq u_k  & \leq u_{max} \hspace{1cm} k = 1, \dots, N-1\\
                & & x_{min} \leq x_k  & \leq x_{max} \hspace{1cm} k = 1, \dots, N\\
                & & x_1 \leq x(t_1) & \leq x_1
            \end{aligned}

        while :math:`g(\cdot)` contains the discretized system dynamics
        according to the specified discretization method. If the system is
        non-dynamic, it only contains the user-provided equality constraints.

        .. rubric:: References

        .. [#f1] *Körkel, Stefan: Numerische Methoden für Optimale Versuchsplanungsprobleme bei nichtlinearen DAE-Modellen, PhD Thesis, Heidelberg university, 2002, pages 74/75.*

        '''

        intro()

        self.__discretize_system( \
            system, time_points, discretization_method, **kwargs)

        self.__apply_parameters_to_discretization(pdata)

        self.__set_optimization_variables()

        self.__set_optimization_variables_initials(pdata, x0, uinit)

        self.__set_optimization_variables_lower_bounds(umin, xmin, x0)

        self.__set_optimization_variables_upper_bounds(umax, xmax, x0)

        self.__set_measurement_data()

        self.__set_weightings(wv, weps_e, weps_u)

        self.__set_measurement_deviations()

        self.__set_cov_matrix_derivative_directions()

        self.__setup_constraints()

        self.__set_optimiality_criterion(optimality_criterion)

        self.__setup_objective()

        self.__apply_parameters_to_objective(pdata)

        self.__setup_nlp()


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

        self.__tstart_optimum_experimental_design = time.time()

        nlpsolver = ci.NlpSolver("solver", "ipopt", self.__nlp, \
            options = solver_options)

        self.__design_results = \
            nlpsolver(x0 = self.__optimization_variables_initials, \
                lbg = 0, ubg = 0, \
                lbx = self.__optimization_variables_lower_bounds, \
                ubx = self.__optimization_variables_upper_bounds)

        print('''
Optimum experimental design finished. Check IPOPT output for status information.
''')

        self.__tend_optimum_experimental_deisng = time.time()
        self.__duration_optimum_experimental_design = \
            self.__tend_optimum_experimental_deisng - \
            self.__tstart_optimum_experimental_design
