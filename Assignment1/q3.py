from matrices import Matrix
from finite_diff import SuccessiveOverRelaxer, relaxation


def q3a():
    omega = 0.5
    epsilon = 0.00001
    relaxer = SuccessiveOverRelaxer(omega)
    n = 100
    phi = Matrix.empty(100, 100)
    result = relaxation(phi, epsilon, relaxer)


if __name__ == '__main__':
    q3a()