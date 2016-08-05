import numpy as np


def issym(x):
    return 'sympy.' in str(type(x))


class Multipol:
    coeffs = 0
    monomials = 0

    def __init__(self, coeffs, monomials):
        self.coeffs = coeffs
        self.monomials = monomials

    @staticmethod
    def multipol(*args):
        if (type(args[0]) is int or type(
                args[0]) is list or type(
            args[0]) is np.ndarray) and \
                (type(args[1]) is int or type(args[1]) is list or type(
                    args[1]) is np.ndarray):
            coeffs = args[0]
            monomials = args[1]
            if not coeffs or coeffs.size == 0:
                coeffs = 0
            if not monomials or monomials.size == 0:
                monomials = 0
            if monomials.size != coeffs.size:
                assert False
            s = Multipol(coeffs, monomials)

        else:
            assert False

        return s


print(__name__)
