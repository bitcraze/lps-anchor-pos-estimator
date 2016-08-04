import numpy as np
import scipy.linalg


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
        if len(args) == 0:
            s = Multipol(0, 0)

        elif len(args) == 1:
            coeffs = args[0]

            if type(coeffs) is Multipol:
                s = coeffs

            elif type(coeffs) is int:
                s = Multipol(coeffs, 0)

            elif type(coeffs) is list or type(coeffs) is np.ndarray:
                for i in range(len(args), 0, -1):
                    s[i] = Multipol(coeffs[i])

                s = np.reshape(s, coeffs.shape)

            elif issym(coeffs):
                assert False
                # S = coeffs
                # varibs =
                # nvar = len(varibs)

                # print("Subsituting [")
                # for i in range(nvar-2):
                # 	print(chr(varibs[i]))

                # print('] --> [' + chr(varibs(nvar)))

                # for i in range(nvar-2):
                # 	print("x%u]\n")

        elif len(args) == 2 and (type(args[0]) is int or type(args[0]) is list or type(args[0]) is np.ndarray) and \
                (type(args[1]) is int or type(args[1]) is list or type(args[1]) is np.ndarray):
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

if __name__ == '__main__':
    a = Multipol.multipol(42)
