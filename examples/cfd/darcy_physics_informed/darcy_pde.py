# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from modulus.sym.eq.pde import PDE
from sympy import Symbol, Function


class Darcy(PDE):
    """Darcy PDE using Modulus Sym"""

    name = "Darcy"

    def __init__(self):

        # time
        x, y = Symbol("x"), Symbol("y")

        # make input variables
        input_variables = {"x": x, "y": y}

        # make sol function
        u = Function("sol")(*input_variables)
        k = Function("K")(*input_variables)
        f = 1.0

        # set equation
        self.equations = {}
        self.equations["darcy"] = (
            f
            + k.diff(x) * u.diff(x)
            + k * u.diff(x).diff(x)
            + k.diff(y) * u.diff(y)
            + k * u.diff(y).diff(y)
        )
