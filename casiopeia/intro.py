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

import os

def intro():

    try:

        os.environ["CASIOPEIA_INTRO_SHOWN"]

    except:

        os.environ["CASIOPEIA_INTRO_SHOWN"] = "1"

        print('\n' + 78 * '#')
        print('#' + 76 * ' ' + '#')
        print('#' + 32 * ' ' + 'casiopeia 0.1' + 31 * ' ' + '#')
        print('#' + 76 * ' ' + '#')
        print('#' + 12 * ' ' + \
            'Adrian Buerger 2014-2016, Jesus Lago Garcia 2014-2015' + \
            11 * ' ' + '#')
        print('#' + 19 * ' ' + \
            ' SYSCOP, IMTEK, University of Freiburg ' + 18 * ' ' + '#')
        print('#' + 76 * ' ' + '#')
        print(78 * '#')