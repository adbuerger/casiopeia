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

'''The module ``casiopeia.pe`` contains the classes for parameter estimation.
For now, only least squares parameter estimation problems are covered.'''

import numpy as np
import time

from abc import ABCMeta, abstractmethod, abstractproperty

from discretization.nodiscretization import NoDiscretization
from discretization.odecollocation import ODECollocation
from discretization.odemultipleshooting import ODEMultipleShooting

from interfaces import casadi_interface as ci
from matrices import KKTMatrix, FisherMatrix, CovarianceMatrix, \
    DirectFactorizationCovarianceMatrix, \
    setup_covariance_matrix_scaling_factor_beta
from intro import intro

import inputchecks

class PEProblem(object):

    __metaclass__ = ABCMeta

    @abstractproperty
    def gauss_newton_lagrangian_hessian(self):

        r'''
        Abstract method for returning the Hessian of the Gauss Newton 
        Langrangian.
        '''


    @property
    def estimation_results(self):

        try:

            return self._estimation_results

        except AttributeError:

            raise AttributeError('''
A parameter estimation has to be executed before the estimation results
can be accessed, please run run_parameter_estimation() first.
''')


    @property
    def estimated_parameters(self):

        try:

            return self._estimation_results["x"][ \
                :self._discretization.system.np]

        except AttributeError:

            raise AttributeError('''
A parameter estimation has to be executed before the estimated parameters
can be accessed, please run run_parameter_estimation() first.
''')


    @property
    def beta(self):

        try:

            return self._beta

        except AttributeError:

            raise AttributeError('''
Beta-factor for the parameter estimation not yet computed.
Run compute_covariance_matrix() to do so.
''')


    @property
    def covariance_matrix(self):

        try:

            return self._covariance_matrix

        except AttributeError:

            raise AttributeError('''
Covariance matrix for the estimated parameters not yet computed.
Run compute_covariance_matrix() to do so.
''')


    @property
    def standard_deviations(self):

        try:

            variances = []

            for k in range(ci.diag(self.covariance_matrix).numel()):

                variances.append(abs(ci.diag(self.covariance_matrix)[k]))

            standard_deviations = ci.sqrt(variances)

            return standard_deviations

            # return ci.sqrt([abs(var) for var \
            #     in ci.diag(self.covariance_matrix)])

        except AttributeError:

            raise AttributeError('''
Standard deviations for the estimated parameters not yet computed.
Run compute_covariance_matrix() to do so.
''')


    def _setup_objective(self):

        self._objective =  0.5 * ci.mul([self._residuals.T, self._residuals])


    def _setup_nlp(self):

        self._nlp = {"x": self._optimization_variables, \
            "f": self._objective, "g": self._constraints}


    @abstractmethod
    def __init__(self):

        r'''
        Abstract constructor for parameter estimation classes.
        '''


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

        self._tstart_estimation = time.time()

        nlpsolver = ci.NlpSolver("solver", "ipopt", self._nlp, \
            options = solver_options)

        self._estimation_results = \
            nlpsolver(x0 = self._optimization_variables_initials, \
                lbg = 0, ubg = 0)

        self._tend_estimation = time.time()
        self._duration_estimation = self._tend_estimation - \
            self._tstart_estimation

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
            :math:`\sigma_{\hat{\text{p}}},`
            (the values of the standard deviations are presented
            only if the covariance matrix had already been computed),
          - the values of the covariance matrix
            :math:`\Sigma_{\hat{\text{p}}}` for the
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

            # for k, pk in enumerate(self.estimated_parameters):
            for k in range(self.estimated_parameters.numel()):
            
                try:

                    print("    p_{0:<3} = {1} +/- {2}".format( \
                        k, self.estimated_parameters[k], \
                        self.standard_deviations[k]))

                except AttributeError:

                    print("    p_{0:<3} = {1}".format(\
                        k, self.estimated_parameters[k]))

            print("\nCovariance matrix for the estimated parameters:")

            try:

                print(np.atleast_2d(self.covariance_matrix))

            except AttributeError:

                print( \
'''    Covariance matrix for the estimated parameters not yet computed.
    Run compute_covariance_matrix() to do so.''')

            
            print("\nDuration of the estimation" + 23 * "." + \
                ": {0:10.8e} s".format(self._duration_estimation))

            try:

                print("Duration of the covariance matrix computation...." + \
                    ": {0:10.8e} s".format( \
                        self._duration_covariance_computation))

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
        the values of the estimated parameters [#f1]_ [#f2]_.

        For efficiency, only the inverse of the relevant part of the matrix
        is computed [#f3]_.

        The values of the covariance matrix :math:`\Sigma_{\hat{\text{p}}}` can afterwards
        be accessed via the class attribute ``LSq.covariance_matrix``, and the
        contained standard deviations :math:`\sigma_{\hat{\text{p}}}` for the
        estimated parameters directly via 
        ``LSq.standard_deviations``.

        .. rubric:: References

        .. [#f1] |linkf1|_

        .. _linkf1:  https://www.researchgate.net/publication/228407918_Computing_Covariance_Matrices_for_Constrained_Nonlinear_Large_Scale_Parameter_Estimation_Problems_Using_Krylov_Subspace_Methods

        .. |linkf1| replace:: *Kostina, Ekaterina and Kostyukova, Olga: Computing Covariance Matrices for Constrained Nonlinear Large Scale Parameter Estimation Problems Using Krylov Subspace Methods, 2012.*

        .. [#f2] |linkf2|_

        .. _linkf2: http://www.am.uni-erlangen.de/home/spp1253/wiki/images/b/b3/Freising10_19_-_Kostina_-_Towards_Optimum.pdf

        .. |linkf2| replace:: *Kostina, Ekaterina and Kriwet, Gregor: Towards Optimum Experimental Design for Partial Differential Equations, SPP 1253 annual conference 2010, slides 12/13.*

        .. [#f3] *Walter, Eric and Prozanto, Luc: Identification of Parametric Models from Experimental Data, Springer, 1997, pages 288/289.*

        '''

        print('\n' + '# ' + 17 * '-' + \
            ' casiopeia covariance matrix computation ' + 16 * '-' + ' #')

        print('''
Computing the covariance matrix for the estimated parameters,
this might take some time ...''')

        self._tstart_covariance_computation = time.time()

        kkt_matrix = KKTMatrix(self.gauss_newton_lagrangian_hessian, \
            self._constraints, self._optimization_variables)

        fisher_matrix = FisherMatrix(kkt_matrix.kkt_matrix, \
            self._discretization.system.np)

        self._covariance_matrix = CovarianceMatrix(fisher_matrix.fisher_matrix)

        # self._covariance_matrix = DirectFactorizationCovarianceMatrix( \
        #     kkt_matrix.kkt_matrix, self._discretization.system.np)

        beta = setup_covariance_matrix_scaling_factor_beta( \
            self._constraints, self._optimization_variables, self._residuals)

        beta_fcn = ci.mx_function("beta_fcn", \
            [self._optimization_variables], [beta])

        self._beta = beta_fcn(self.estimation_results["x"])

        self._covariance_matrix_scaled = self._beta * \
            self._covariance_matrix.covariance_matrix

        covariance_matrix_fcn = ci.mx_function("covariance_matrix_fcn", \
            [self._optimization_variables], \
            [self._covariance_matrix_scaled])

        self._covariance_matrix = \
            covariance_matrix_fcn(self.estimation_results["x"])

        self._tend_covariance_computation = time.time()
        self._duration_covariance_computation = \
            self._tend_covariance_computation - \
            self._tstart_covariance_computation

        print("Covariance matrix computation finished.")


class LSq(PEProblem):

    '''The class :class:`casiopeia.pe.LSq` is used to set up least
    squares parameter estimation problems for systems defined with the
    :class:`casiopeia.system.System`
    class, using a given set of user-provided control 
    data, measurement data and different kinds of weightings.'''

    @property
    def gauss_newton_lagrangian_hessian(self):

        gauss_newton_lagrangian_hessian_diag = ci.vertcat([ \
            ci.mx(self._optimization_variables.shape[0] - \
                self._weightings_vectorized.shape[0], 1), \
            self._weightings_vectorized])

        gauss_newton_lagrangian_hessian = ci.diag( \
            gauss_newton_lagrangian_hessian_diag)

        return gauss_newton_lagrangian_hessian


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


    def _apply_controls_to_equality_constraints(self, udata, qdata):

        udata = inputchecks.check_controls_data(udata, \
            self._discretization.system.nu, \
            self._discretization.number_of_controls)
        qdata = inputchecks.check_constant_controls_data(qdata, \
            self._discretization.system.nq)

        optimization_variables_for_equality_constraints = ci.veccat([ \

                self._discretization.optimization_variables["U"],
                self._discretization.optimization_variables["Q"],
                self._discretization.optimization_variables["X"], 
                self._discretization.optimization_variables["EPS_U"], 
                self._discretization.optimization_variables["P"], 

            ])

        optimization_variables_controls_applied = ci.veccat([ \

                udata,
                qdata,
                self._discretization.optimization_variables["X"], 
                self._discretization.optimization_variables["EPS_U"], 
                self._discretization.optimization_variables["P"], 

            ])

        equality_constraints_fcn = ci.mx_function( \
            "equality_constraints_fcn", \
            [optimization_variables_for_equality_constraints], \
            [self._discretization.equality_constraints])

        self._equality_constraints_controls_applied = \
            equality_constraints_fcn(optimization_variables_controls_applied)


    def _apply_controls_to_measurements(self, udata, qdata):

        udata = inputchecks.check_controls_data(udata, \
            self._discretization.system.nu, \
            self._discretization.number_of_controls)
        qdata = inputchecks.check_constant_controls_data(qdata, \
            self._discretization.system.nq)

        optimization_variables_for_measurements = ci.veccat([ \

                self._discretization.optimization_variables["U"], 
                self._discretization.optimization_variables["Q"], 
                self._discretization.optimization_variables["X"], 
                self._discretization.optimization_variables["EPS_U"], 
                self._discretization.optimization_variables["P"], 

            ])

        optimization_variables_controls_applied = ci.veccat([ \

                udata, 
                qdata, 
                self._discretization.optimization_variables["X"], 
                self._discretization.optimization_variables["EPS_U"], 
                self._discretization.optimization_variables["P"], 

            ])

        measurements_fcn = ci.mx_function( \
            "measurements_fcn", \
            [optimization_variables_for_measurements], \
            [self._discretization.measurements])

        self._measurements_controls_applied = \
            measurements_fcn(optimization_variables_controls_applied)


    def _apply_controls_to_discretization(self, udata, qdata):

        self._apply_controls_to_equality_constraints(udata, qdata)
        self._apply_controls_to_measurements(udata, qdata)


    def _set_optimization_variables(self):

        self._optimization_variables = ci.veccat([ \

                self._discretization.optimization_variables["P"],
                self._discretization.optimization_variables["X"],
                self._discretization.optimization_variables["V"],
                self._discretization.optimization_variables["EPS_U"],

            ])


    def _set_optimization_variables_initials(self, pinit, xinit):

        xinit = inputchecks.check_states_data(xinit, \
            self._discretization.system.nx, \
            self._discretization.number_of_intervals)
        repretitions_xinit = \
            self._discretization.optimization_variables["X"][:,:-1].shape[1] / \
                self._discretization.number_of_intervals
        
        Xinit = ci.repmat(xinit[:, :-1], repretitions_xinit, 1)

        Xinit = ci.horzcat([ \

            Xinit.reshape((self._discretization.system.nx, \
                Xinit.numel() / self._discretization.system.nx)),
            xinit[:, -1],

            ])

        pinit = inputchecks.check_parameter_data(pinit, \
            self._discretization.system.np)
        Pinit = pinit

        Vinit = np.zeros(self._discretization.optimization_variables["V"].shape)
        EPS_Uinit = np.zeros( \
            self._discretization.optimization_variables["EPS_U"].shape)

        self._optimization_variables_initials = ci.veccat([ \

                Pinit,
                Xinit,
                Vinit,
                EPS_Uinit,

            ])


    def _set_measurement_data(self, ydata):

        measurement_data = inputchecks.check_measurement_data(ydata, \
            self._discretization.system.nphi, \
            self._discretization.number_of_intervals + 1)
        self._measurement_data_vectorized = ci.vec(measurement_data)


    def _set_weightings(self, wv, weps_u):

        input_error_weightings = \
            inputchecks.check_input_error_weightings(weps_u, \
            self._discretization.system.neps_u, \
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

                ci.vec(self._measurements_controls_applied) - \
                self._measurement_data_vectorized + \
                ci.vec(self._discretization.optimization_variables["V"])

            ])


    def _setup_residuals(self):

        self._residuals = ci.sqrt(self._weightings_vectorized) * \
            ci.veccat([ \

                self._discretization.optimization_variables["V"],
                self._discretization.optimization_variables["EPS_U"],

            ])


    def _setup_constraints(self):

        self._constraints = ci.vertcat([ \

                self._measurement_deviations,
                self._equality_constraints_controls_applied,

            ])


    def __init__(self, system, time_points, \
        udata = None, qdata = None,\
        ydata = None, \
        pinit = None, xinit = None, \
        wv = None, weps_u = None, \
        discretization_method = "collocation", **kwargs):

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

        :param udata: optional, values for the time-varying controls 
                   :math:`u_\text{N} \in \mathbb{R}^{\text{n}_\text{u} \times \text{N}-1}`
                   that can change at the switching time points;
                   if no values are given, 0 will be used; note that the
                   the second dimension of :math:`u_\text{N}` is :math:`N-1` and not
                   :math:`N`, since there is no control value applied at the
                   last time point
        :type udata: numpy.ndarray, casadi.DMatrix

        :param qdata: optional, values for the time-constant controls
                   :math:`q_\text{N} \in \mathbb{R}^{\text{n}_\text{q}}`;
                   if not values are given, 0 will be used
        :type qdata: numpy.ndarray, casadi.DMatrix

        :param ydata: values for the measurements at the switching time points
                   :math:`y_\text{N} \in \mathbb{R}^{\text{n}_\text{y} \times \text{N}}`
        :type ydata: numpy.ndarray, casadi.DMatrix    

        :param wv: weightings for the measurements
                   :math:`w_\text{v} \in \mathbb{R}^{\text{n}_\text{y} \times \text{N}}`
        :type wv: numpy.ndarray, casadi.DMatrix    

        :param weps_u: weightings for the input errors
                   :math:`w_{\epsilon_\text{u}} \in \mathbb{R}^{\text{n}_{\epsilon_\text{u}}}`
                   (only necessary
                   if input errors are used within ``system``)
        :type weps_u: numpy.ndarray, casadi.DMatrix    

        :param pinit: optional, initial guess for the values of the
                      parameters that will be estimated
                      :math:`p_\text{init} \in \mathbb{R}^{\text{n}_\text{p}}`; if no
                      value is given, 0 will be used; note that a poorly or
                      wrongly chosen initial guess can cause the estimation
                      to fail
        :type pinit: numpy.ndarray, casadi.DMatrix

        :param xinit: optional, initial guess for the values of the
                      states that will be estimated
                      :math:`x_\text{init} \in \mathbb{R}^{\text{n}_\text{x} \times \text{N}}`;
                      if no value is given, 0 will be used; note that a poorly
                      or wrongly chosen initial guess can cause the estimation
                      to fail
        :type xinit: numpy.ndarray, casadi.DMatrix

        :param discretization_method: optional, the method that shall be used for
                                      discretization of the continuous time
                                      problem w. r. t. the time points given 
                                      in :math:`t_\text{N}`; possible values are
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
                \text{arg}\,\underset{p, x, v, \epsilon_\text{u}}{\text{min}} & & \frac{1}{2} \| R(\cdot) \|_2^2 &\\
                \text{subject to:} & & v_\text{k} + y_\text{k} - \phi(x_\text{k}, p; u_\text{k}, q) & = 0 \hspace{1cm} k = 1, \dots, N\\
                & & g(x, p, \epsilon_\text{u}; u, q) & = 0 \\
                \text{with:} & & \begin{pmatrix} {w_\text{v}}^T & {w_{\epsilon_\text{u}}}^T \end{pmatrix}^{^\mathbb{1}/_\mathbb{2}} \begin{pmatrix} {v} \\ {\epsilon_\text{u}} \end{pmatrix} & = R \\
            \end{aligned}

        while :math:`g(\cdot)` contains the discretized system dynamics
        according to the specified discretization method. If the system is
        non-dynamic, it only contains the user-provided equality constraints.

        '''

        intro()

        self._discretize_system( \
            system, time_points, discretization_method, **kwargs)

        self._apply_controls_to_discretization(udata, qdata)

        self._set_optimization_variables()

        self._set_optimization_variables_initials(pinit, xinit)

        self._set_measurement_data(ydata)

        self._set_weightings(wv, weps_u)

        self._set_measurement_deviations()

        self._setup_residuals()

        self._setup_constraints()

        self._setup_objective()

        self._setup_nlp()


class MultiLSq(PEProblem):

    '''The class :class:`casiopeia.pe.MultiLSq` is used to construct a single
    least squares parameter estimation problem from multiple least squares
    parameter estimation problems defined via two or more objects of type
    :class:`casiopeia.pe.LSq`.

    In this way, the results of multiple independent experimental setups
    can be used for parameter estimation.

    .. note::

        It is assumed that the system description used for setting up
        the several parameter estimation problems is the same.
    '''

    @property
    def gauss_newton_lagrangian_hessian(self):

        gauss_newton_lagrangian_hessians = \
            [pe_setup.gauss_newton_lagrangian_hessian \
                for pe_setup in self._pe_setups]

        gauss_newton_lagrangian_hessian = \
            ci.diagcat(gauss_newton_lagrangian_hessians)

        return gauss_newton_lagrangian_hessian


    def _define_set_of_pe_setups(self, pe_setups):

        inputchecks.check_multi_lsq_input(pe_setups)

        self._pe_setups = pe_setups


    def _get_discretization_from_pe_setups(self):

        self._discretization = self._pe_setups[0]._discretization


    def _merge_optimization_variables(self):

        optimization_variables = [ \
            pe_setup._optimization_variables for \
            pe_setup in self._pe_setups]

        self._optimization_variables = ci.vertcat(optimization_variables)


    def _merge_optimization_variables_initials(self):

        optimization_variables_initials = [ \
            pe_setup._optimization_variables_initials for \
            pe_setup in self._pe_setups]

        self._optimization_variables_initials = \
            ci.vertcat(optimization_variables_initials)


    def _merge_measurement_data(self):

        measurement_data = [pe_setup._measurement_data_vectorized for \
            pe_setup in self._pe_setups]

        self._measurement_data_vectorized = ci.vertcat(measurement_data)


    def _merge_weightings(self):

        weightings = [pe_setup._weightings_vectorized for \
            pe_setup in self._pe_setups]

        self._weightings_vectorized = ci.vertcat(weightings)


    def _merge_residuals(self):

        residuals = [pe_setup._residuals for \
            pe_setup in self._pe_setups]

        self._residuals = ci.vertcat(residuals)


    def _merge_and_setup_constraints(self):

        constraints = [self._pe_setups[0]._constraints]

        for k, pe_setup in enumerate(self._pe_setups[:-1]):

            # Connect estimation problems by setting p_k - p_k+1 = 0

            constraints.append(pe_setup._discretization.optimization_variables["P"] - \
                self._pe_setups[k+1]._discretization.optimization_variables["P"])

            constraints.append(self._pe_setups[k+1]._constraints)

        self._constraints = ci.vertcat(constraints)


    def __init__(self, pe_setups = []):

        r'''
        :param pe_setups: list of two or more objects of type :class:`casiopeia.pe.LSq`
        :type pe_setups: list

        '''

        self._define_set_of_pe_setups(pe_setups)

        self._get_discretization_from_pe_setups()

        self._merge_optimization_variables()

        self._merge_optimization_variables_initials()

        self._merge_measurement_data()

        self._merge_weightings()

        self._merge_residuals()

        self._merge_and_setup_constraints()

        self._setup_objective()

        self._setup_nlp()
