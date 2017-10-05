import copy

from matrices import Matrix
from finite_diff import SuccessiveOverRelaxer, OuterConductorBoundary, QuarterInnerConductorBoundary, \
    CoaxialCableMeshConstructor, JacobiRelaxer, IterativeRelaxer


def q3a():
    omega = 0.5
    epsilon = 0.00001
    h = 0.02
    phi = CoaxialCableMeshConstructor().construct_mesh(h)
    print(phi)
    relaxer = JacobiRelaxer()
    iter_relaxer = IterativeRelaxer(relaxer, epsilon, phi, h)
    iter_relaxer.relaxation()
    print(iter_relaxer.phi)


if __name__ == '__main__':
    q3a()
