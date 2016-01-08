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

.. casiopeia documentation master file, created by
.. sphinx-quickstart on Mon Dec  8 09:36:29 2014.
.. You can adapt this file completely to your liking, but it should at least
.. contain the root `toctree` directive.

casiopeia
=========

Casadi Interface for Optimum experimental design and Parameter Estimation and Identification Applications
---------------------------------------------------------------------------------------------------------

`casiopeia <https://github.com/adbuerger/casiopeia>`_ holds a user-friendly environment for optimum experimental design and parameter estimation and identification applications. It does so by providing Python classes that can be initialized with the problem specifications, while the computations can then easily be performed using the available class functions.

casiopeia makes use of the optimization framework
`CasADi <http://casadi.org>`_ to solve parameter estimation problems.

.. note:: casiopeia is still in it's testing state, and does not yet contain all the features it will provide in future versions. Therefore, you should check for updates on a regular basis.

In the following sections, you will receive the information necessary to obtain,
install and use casiopeia. If you encounter any problems using this software, please feel free
to submit your errors with a description of how it occurred to adrian.buerger@hs-karlsruhe.de.

**New and experimental:** `try casiopeia live in your browser <https://ec2-52-29-32-46.eu-central-1.compute.amazonaws.com:8888/8cacf81e-3d6e-488d-972a-679cb5befefb>`_ [#f1]_






Contents
--------

.. toctree::
   :maxdepth: 2
   :numbered:

   install
   system
   sim
   pe
   doe
   samples

.. [#f1] This service is at the moment limited to one user at a time, due to restricted resources. If your computations do no start immediately, there's probably another user testing casiopeia at the moment. 

.. Indices and tables
.. ------------------

.. * :ref:`genindex`
.. * :ref:`search`

.. * :ref:`modindex`
