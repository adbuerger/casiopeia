.. Copyright 2014-2016 Adrian Bürger
..
.. This file is part of casiopeia.
..
.. casiopeia is free software: you can redistribute it and/or modify
.. it under the terms of the GNU Lesser General Public License as published by
.. the Free Software Foundation, either version 3 of the License, or
.. (at your option) any later version.
..
.. casiopeia is distributed in the hope that it will be useful,
.. but WITHOUT ANY WARRANTY; without even the implied warranty of
.. MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
.. GNU Lesser General Public License for more details.
..
.. You should have received a copy of the GNU Lesser General Public License
.. along with casiopeia. If not, see <http://www.gnu.org/licenses/>.


Defining a system
=================

Since casiopeia uses CasADi, the user first has to define the considered system using CasADi symbolic variables (of type MX). Afterwards, the symbolic variables which define states, controls, parameters, etc. of the system can be brought into connection by creating a :class:`casiopeia.System` object.

.. automodule:: casiopeia.system
    :members:

This system object can now be used within the casiopeia simulation, parameter estimation and optimum experimental design classes.
