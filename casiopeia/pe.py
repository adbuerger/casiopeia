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

'''The module ``casiopeia.pe`` contains the classes for parameter estimation.
For now, only least squares parameter estimation problems are covered.'''

import numpy as np
import time

from discretization.nodiscretization import NoDiscretization
from discretization.odecollocation import ODECollocation
from discretization.odemultipleshooting import ODEMultipleShooting

from interfaces import casadi_interface as ci
from covariance_matrix import setup_covariance_matrix, setup_beta
from intro import intro

import inputchecks

class LSq(object):

    '''The class :class:`casiopeia.pe.LSq` is used to set up least
    squares parameter estimation problems for systems defined with the
    :class:`casiopeia.system.System`
    class, using a given set of user-provided control 
    data, measurement data and different kinds of weightings.'''

    @property
    def estimation_results(self):

        try:

            return self.__estimation_results

        except AttributeError:

            raise AttributeError('''
A parameter estimation has to be executed before the estimation results
can be accessed, please run run_parameter_estimation() first.
''')


    @property
    def estimated_parameters(self):

        try:

            return self.__estimation_results["x"][ \
                :self.__discretization.system.np]

        except AttributeError:

            raise AttributeError('''
A parameter estimation has to be executed before the estimated parameters
can be accessed, please run run_parameter_estimation() first.
''')


    @property
    def covariance_matrix(self):

        try:

            return self.__covariance_matrix

        except AttributeError:

            raise AttributeError('''
Covariance matrix for the estimated parameters not yet computed.
Run compute_covariance_matrix() to do so.
''')


    @property
    def standard_deviations(self):

        try:

            return ci.sqrt([abs(var) for var \
                in ci.diag(self.covariance_matrix)])

        except AttributeError:

            raise AttributeError('''
Standard deviations for the estimated parameters not yet computed.
Run compute_covariance_matrix() to do so.
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


    def __apply_controls_to_equality_constraints(self, udata):

        udata = inputchecks.check_controls_data(udata, \
            self.__discretization.system.nu, \
            self.__discretization.number_of_controls)

        optimization_variables_for_equality_constraints = ci.veccat([ \

                self.__discretization.optimization_variables["U"], 
                self.__discretization.optimization_variables["X"], 
                self.__discretization.optimization_variables["EPS_U"], 
                self.__discretization.optimization_variables["EPS_E"], 
                self.__discretization.optimization_variables["P"], 

            ])

        optimization_variables_controls_applied = ci.veccat([ \

                udata, 
                self.__discretization.optimization_variables["X"], 
                self.__discretization.optimization_variables["EPS_U"], 
                self.__discretization.optimization_variables["EPS_E"], 
                self.__discretization.optimization_variables["P"], 

            ])

        equality_constraints_fcn = ci.mx_function( \
            "equality_constraints_fcn", \
            [optimization_variables_for_equality_constraints], \
            [self.__discretization.equality_constraints])

        [self.__equality_constraints_controls_applied] = \
            equality_constraints_fcn([optimization_variables_controls_applied])


    def __apply_controls_to_measurements(self, udata):

        udata = inputchecks.check_controls_data(udata, \
            self.__discretization.system.nu, \
            self.__discretization.number_of_controls)

        optimization_variables_for_measurements = ci.veccat([ \

                self.__discretization.optimization_variables["U"], 
                self.__discretization.optimization_variables["X"], 
                self.__discretization.optimization_variables["EPS_U"], 
                self.__discretization.optimization_variables["P"], 

            ])

        optimization_variables_controls_applied = ci.veccat([ \

                udata, 
                self.__discretization.optimization_variables["X"], 
                self.__discretization.optimization_variables["EPS_U"], 
                self.__discretization.optimization_variables["P"], 

            ])

        measurements_fcn = ci.mx_function( \
            "measurements_fcn", \
            [optimization_variables_for_measurements], \
            [self.__discretization.measurements])

        [self.__measurements_controls_applied] = \
            measurements_fcn([optimization_variables_controls_applied])


    def __apply_controls_to_discretization(self, udata):

        self.__apply_controls_to_equality_constraints(udata)
        self.__apply_controls_to_measurements(udata)


    def __set_optimization_variables(self):

        self.__optimization_variables = ci.veccat([ \

                self.__discretization.optimization_variables["P"],
                self.__discretization.optimization_variables["X"],
                self.__discretization.optimization_variables["V"],
                self.__discretization.optimization_variables["EPS_E"],
                self.__discretization.optimization_variables["EPS_U"],

            ])


    def __set_optimization_variables_initials(self, pinit, xinit):

        xinit = inputchecks.check_states_data(xinit, \
            self.__discretization.system.nx, \
            self.__discretization.number_of_intervals)
        repretitions_xinit = \
            self.__discretization.optimization_variables["X"][:,:-1].shape[1] / \
                self.__discretization.number_of_intervals
        
        Xinit = ci.repmat(xinit[:, :-1], repretitions_xinit, 1)

        Xinit = ci.horzcat([ \

            Xinit.reshape((self.__discretization.system.nx, \
                Xinit.size() / self.__discretization.system.nx)),
            xinit[:, -1],

            ])

        pinit = inputchecks.check_parameter_data(pinit, \
            self.__discretization.system.np)
        Pinit = pinit

        Vinit = np.zeros(self.__discretization.optimization_variables["V"].shape)
        EPS_Einit = np.zeros( \
            self.__discretization.optimization_variables["EPS_E"].shape)
        EPS_Uinit = np.zeros( \
            self.__discretization.optimization_variables["EPS_U"].shape)

        self.__optimization_variables_initials = ci.veccat([ \

                Pinit,
                Xinit,
                Vinit,
                EPS_Einit,
                EPS_Uinit,

            ])


    def __set_measurement_data(self, ydata):

        measurement_data = inputchecks.check_measurement_data(ydata, \
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

                ci.vec(self.__measurements_controls_applied) - \
                self.__measurement_data_vectorized + \
                ci.vec(self.__discretization.optimization_variables["V"])

            ])


    def __setup_residuals(self):

        self.__residuals = ci.sqrt(self.__weightings_vectorized) * \
            ci.veccat([ \

                self.__discretization.optimization_variables["V"],
                self.__discretization.optimization_variables["EPS_E"],
                self.__discretization.optimization_variables["EPS_U"],

            ])


    def __setup_constraints(self):

        self.__constraints = ci.vertcat([ \

                self.__measurement_deviations,
                self.__equality_constraints_controls_applied,

            ])


    def __setup_objective(self):

        self.__objective =  0.5 * ci.mul([self.__residuals.T, self.__residuals])


    def __setup_nlp(self):

        self.__nlp = ci.mx_function("nlp", \
            ci.nlpIn(x = self.__optimization_variables), \
            ci.nlpOut(f = self.__objective, g = self.__constraints))


    def __init__(self, system, time_points, \
        udata = None, ydata = None, \
        pinit = None, xinit = None, \
        wv = None, weps_e = None, weps_u = None, \
        discretization_method = "collocation", **kwargs):

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

        :param udata: optional, values for the controls at the switching time
                   points :math:`u_N \in \mathbb{R}^{n_u \times N-1}`;
                   if not values are given, 0 will be used; note that the
                   the second dimension of :math:`u_N` is :math:`N-1` and not
                   :math:`N`, since there is no control value applied at the
                   last time point
        :type udata: numpy.ndarray, casadi.DMatrix

        :param ydata: values for the measurements at the switching time points
                   :math:`u_y \in \mathbb{R}^{n_y \times N}`
        :type ydata: numpy.ndarray, casadi.DMatrix    

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

        :param pinit: optional, initial guess for the values of the
                      parameters that will be estimated
                      :math:`p_{init} \in \mathbb{R}^{n_p}`; if no
                      value is given, 0 will be used; note that a poorly or
                      wrongly chosen initial guess can cause the estimation
                      to fail
        :type pinit: numpy.ndarray, casadi.DMatrix

        :param xinit: optional, initial guess for the values of the
                      states that will be estimated
                      :math:`x_{init} \in \mathbb{R}^{n_x \times N}`;
                      if no value is given, 0 will be used; note that a poorly
                      or wrongly chosen initial guess can cause the estimation
                      to fail
        :type xinit: numpy.ndarray, casadi.DMatrix

        :param discretization_method: optional, the method that shall be used for
                                      discretization of the continuous time
                                      problem w. r. t. the time points given 
                                      in :math:`t_N`; possible values are
                                      "collocation" (default) and
                                      "multiple_shooting"
        :type discretization_method: str

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

        The resulting parameter estimation problem has the following form:

        .. math::

            \begin{aligned}
                \text{arg}\,\underset{p, x, v, \epsilon_e, \epsilon_u}{\text{min}} & & \frac{1}{2} \| R(w, v, \epsilon_e, \epsilon_u) \|_2^2 &\\
                \text{subject to:} & & \begin{pmatrix} {w_{v}}^T & {w_{\epsilon_{e}}}^T & {w_{\epsilon_{u}}}^T \end{pmatrix}^{^\mathbb{1}/_\mathbb{2}} \begin{pmatrix} {v} \\ {\epsilon_e} \\ {\epsilon_u} \end{pmatrix} & = R \\
                & & v_{l} + y_{l} - \phi(t_{l}, u_{l}, x_{l}, p) & = 0 \hspace{1cm} l = 1, \dots, N\\
                & & g(p, x, v, \epsilon_e, \epsilon_u) & = 0
            \end{aligned}

        while :math:`g(\cdot)` contains the discretized system dynamics
        according to the specified discretization method. If the system is
        non-dynamic, it only contains the user-provided equality constraints.

        '''

        intro()

        self.__discretize_system( \
            system, time_points, discretization_method, **kwargs)

        self.__apply_controls_to_discretization(udata)

        self.__set_optimization_variables()

        self.__set_optimization_variables_initials(pinit, xinit)

        self.__set_measurement_data(ydata)

        self.__set_weightings(wv, weps_e, weps_u)

        self.__set_measurement_deviations()

        self.__setup_residuals()

        self.__setup_constraints()

        self.__setup_objective()

        self.__setup_nlp()


    def run_parameter_estimation(self, solver_options = {}):

        r'''
        :param solver_options: options to be passed to the IPOPT solver 
                               (see the CasADi documentation for a list of all
                               possible options)
        :type solver_options: dict

        This functions will pass the least squares parameter estimation
        problem to the IPOPT solver. The status of IPOPT printed to the 
        console provides information whether the
        optimization finished successfully. The estimated parameters
        :math:`\hat{p}` can afterwards be accessed via the class attribute
        ``LSq.estimated_parameters``.

        .. note::

            IPOPT finishing successfully does not necessarily
            mean that the estimation results for the unknown parameters are useful
            for your purposes, it just means that IPOPT was able to solve the given
            optimization problem.
            You have in any case to verify your results, e. g. by simulation using
            the casiopeia :class:`casiopeia.sim.Simulation` class!

        '''  

        print('\n' + '# ' +  14 * '-' + \
            ' casiopeia least squares parameter estimation ' + 14 * '-' + ' #')

        print('''
Starting least squares parameter estimation using IPOPT, 
this might take some time ...
''')

        self.__tstart_estimation = time.time()

        nlpsolver = ci.NlpSolver("solver", "ipopt", self.__nlp, \
            options = solver_options)

        self.__estimation_results = \
            nlpsolver(x0 = self.__optimization_variables_initials, \
                lbg = 0, ubg = 0)

        self.__tend_estimation = time.time()
        self.__duration_estimation = self.__tend_estimation - \
            self.__tstart_estimation

        print('''
Parameter estimation finished. Check IPOPT output for status information.
''')


    def print_estimation_results(self):

        r'''
        :raises: AttributeError

        This function displays the results of the parameter estimation
        computations. It can not be used before function
        :func:`run_parameter_estimation()` has been used. The results
        displayed by the function contain:
        
          - the values of the estimated parameters :math:`\hat{p}`
            and their corresponding standard deviations
            :math:`\sigma_{\hat{p}},`
            (the values of the standard deviations are presented
            only if the covariance matrix had already been computed),
          - the values of the covariance matrix
            :math:`\Sigma_{\hat{p}}` for the
            estimated parameters (if it had already been computed), and
          - the durations of the estimation and (if already executed)
            of the covariance matrix computation.
        '''

        np.set_printoptions(linewidth = 200, \
            formatter={'float': lambda x: format(x, ' 10.8e')})

        try:

            print('\n' + '# ' + 17 * '-' + \
                ' casiopeia parameter estimation results ' + 17 * '-' + ' #')
             
            print("\nEstimated parameters p_i:")

            for k, pk in enumerate(self.estimated_parameters):
            
                try:

                    print("    p_{0:<3} = {1} +/- {2}".format( \
                         k, pk[0], self.standard_deviations[k]))

                except AttributeError:

                    print("    p_{0:<3} = {1}".format(\
                        k, pk[0]))

            print("\nCovariance matrix for the estimated parameters:")

            try:

                print(np.atleast_2d(self.covariance_matrix))

            except AttributeError:

                print( \
'''    Covariance matrix for the estimated parameters not yet computed.
    Run compute_covariance_matrix() to do so.''')

            
            print("\nDuration of the estimation" + 23 * "." + \
                ": {0:10.8e} s".format(self.__duration_estimation))

            try:

                print("Duration of the covariance matrix computation...." + \
                    ": {0:10.8e} s".format( \
                        self.__duration_covariance_computation))

            except AttributeError:

                pass

        except AttributeError:

            raise AttributeError('''
You must execute at least run_parameter_estimation() to obtain results,
and compute_covariance_matrix() before all results can be displayed.
''')   

        finally:

            np.set_printoptions()


    def compute_covariance_matrix(self):

        r'''
        This function computes the covariance matrix for the estimated
        parameters from the inverse of the KKT matrix for the parameter
        estimation problem, which allows for statements on the quality of
        the values of the estimated parameters.

        For efficiency, only the inverse of the relevant part of the matrix
        is computed using the Schur complement.

        The values of the covariance matrix :math:`\Sigma_{\hat{p}}` can afterwards
        be accessed via the class attribute ``LSq.covariance_matrix``, and the
        contained standard deviations :math:`\sigma_{\hat{p}}` for the
        estimated parameters directly via 
        ``LSq.standard_deviations``.
        '''

        print('\n' + '# ' + 17 * '-' + \
            ' casiopeia covariance matrix computation ' + 16 * '-' + ' #')

        print('''
Computing the covariance matrix for the estimated parameters,
this might take some time ...''')

        self.__tstart_covariance_computation = time.time()

        self.__covariance_matrix_symbolic = setup_covariance_matrix( \
                self.__optimization_variables, self.__weightings_vectorized, \
                self.__constraints, self.__discretization.system.np)

        self.__beta_symbolic = setup_beta(self.__residuals, \
            self.__measurement_data_vectorized, \
            self.__constraints, self.__optimization_variables)

        covariance_matrix_fcn = ci.mx_function("covariance_matrix_fcn", \
            [self.__optimization_variables], \
            [self.__covariance_matrix_symbolic])

        beta_fcn = ci.mx_function("beta_fcn", \
            [self.__optimization_variables], \
            [self.__beta_symbolic])

        self.__beta = beta_fcn([self.estimation_results["x"]])[0]

        self.__covariance_matrix = self.__beta * \
            covariance_matrix_fcn([self.estimation_results["x"]])[0]

        self.__tend_covariance_computation = time.time()
        self.__duration_covariance_computation = \
            self.__tend_covariance_computation - \
            self.__tstart_covariance_computation

        print("Covariance matrix computation finished.")
