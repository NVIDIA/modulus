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
