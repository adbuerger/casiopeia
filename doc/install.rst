.. This file is part of casiopeia.
..
.. Copyright 2014-2016 Adrian Bürger, Moritz Diehl
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

Get and install casiopeia
=========================

The following instructions have been tested on Ubuntu 18.04. If you are planning to install casiopeia on Linux systems different from Ubuntu 18.04, these commands need to be adapted accordingly.

Installation on Ubuntu 18.04
----------------------------

**casiopeia is available only for Python 2.7.** Also, the version of CasADi required by casiopeia requires the installation of ``libgfortran3``. You might need root priviliges to run the following commands:

.. code:: bash

    apt install libgfortran3 

.. note:: Some functionalitites of CasADi might required other additional open-source libraries to be installed. For a list of possibly required libraries and installation instructions, see `the corresponding section in the CasADi installation guide <https://github.com/casadi/casadi/wiki/linuxplugins>`_. If something goes wrong with executing CasADi and/or casiopeia, missing one or more of these libraries might be the reason.

Depending on your setup, you might need root privileges to install the following packages via pip:

.. code:: bash

    pip install numpy==1.8 casadi==3.1.0

casiopeia can then also be obtained via pip by running the command

.. code:: bash

    pip install casiopeia

or by cloning the corresponding repository and installing casiopeia by

.. code:: bash
   
   git clone https://github.com/adbuerger/casiopeia.git
   cd casiopeia
   python setup.py install

To run the examples shipped with casiopeia, ``pylab`` is required, which can be made available by installing the following packages in addition to ``numpy``:

.. code:: bash

    pip install scipy==0.13 matplotlib==2.2.4


Recommendations
---------------

To speed up computations in casiopeia, it is recommended to install `HSL for IPOPT <http://www.hsl.rl.ac.uk/ipopt/>`_. On how to install the solvers and for further information, see the page `Obtaining HSL <https://github.com/casadi/casadi/wiki/Obtaining-HSL>`_ in the CasADi wiki.
