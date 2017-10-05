from __future__ import division

import copy

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from matrices import Matrix
from finite_diff import SuccessiveOverRelaxer, OuterConductorBoundary, QuarterInnerConductorBoundary, \
    CoaxialCableMeshConstructor, JacobiRelaxer, IterativeRelaxer, successive_over_relaxation, jacobi_relaxation

epsilon = 0.00001
x = 0.06
y = 0.04

NUM_H_ITERATIONS = 2


def q3b():
    print('=== Question 3(a) ===')
    h = 0.02
    phi = CoaxialCableMeshConstructor().construct_mesh(h)

    min_num_iterations = float('inf')
    best_omega = float('inf')

    omegas = []
    num_iterations = []

    for omega_diff in range(10):
        omega = 1 + omega_diff / 10
        print('Omega: {}'.format(omega))
        iter_relaxer = successive_over_relaxation(omega, epsilon, phi, h)
        # print(iter_relaxer.phi)
        print('Num iterations: {}'.format(iter_relaxer.num_iterations))
        print('Potential at ({}, {}): {:.3f} V'.format(x, y, iter_relaxer.get_potential(0.06, 0.04)))
        if iter_relaxer.num_iterations < min_num_iterations:
            best_omega = omega
        min_num_iterations = min(min_num_iterations, iter_relaxer.num_iterations)

        omegas.append(omega)
        num_iterations.append(iter_relaxer.num_iterations)

    print('Best number of iterations: {}'.format(min_num_iterations))
    print('Best omega: {}'.format(best_omega))

    f = plt.figure()
    x_range = omegas
    y_range = num_iterations
    plt.plot(x_range, y_range, 'o-', label='Number of iterations')
    plt.xlabel('Omega')
    plt.ylabel('Number of Iterations')
    plt.grid(True)
    f.savefig('plots/q3b.pdf', bbox_inches='tight')

    return best_omega


def q3c(omega):
    print('=== Question 3(c) ===')
    h = 0.04
    h_values = []
    potential_values = []
    for i in range(NUM_H_ITERATIONS):
        h = h / 2
        print('h: {}'.format(h))
        phi = CoaxialCableMeshConstructor().construct_mesh(h)
        iter_relaxer = successive_over_relaxation(omega, epsilon, phi, h)
        potential = iter_relaxer.get_potential(x, y)

        h_values.append(1 / h)
        potential_values.append(potential)

    f = plt.figure()
    x_range = h_values
    y_range = potential_values
    plt.plot(x_range, y_range, 'o-', label='Potential at (0.06, 0.04)')
    plt.xlabel('1 / h')
    plt.ylabel('Potential at (0.06, 0.04)')
    plt.grid(True)
    f.savefig('plots/q3c.pdf', bbox_inches='tight')


def q3d():
    print('=== Question 3(d) ===')
    h = 0.04
    h_values = []
    potential_values = []
    for i in range(NUM_H_ITERATIONS):
        h = h / 2
        print('h: {}'.format(h))
        phi = CoaxialCableMeshConstructor().construct_mesh(h)
        iter_relaxer = jacobi_relaxation(epsilon, phi, h)
        potential = iter_relaxer.get_potential(x, y)

        h_values.append(1 / h)
        potential_values.append(potential)

    f = plt.figure()
    x_range = h_values
    y_range = potential_values
    plt.plot(x_range, y_range, 'o-', label='Potential at (0.06, 0.04)')
    plt.xlabel('1 / h')
    plt.ylabel('Potential at (0.06, 0.04)')
    plt.grid(True)
    f.savefig('plots/q3d.pdf', bbox_inches='tight')


if __name__ == '__main__':
    best_omega = q3b()
    q3c(best_omega)
    q3d()  # TODO: Exploit symmetry of grid
