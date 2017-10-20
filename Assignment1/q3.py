from __future__ import division

import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import sympy as sp

from csv_saver import save_rows_to_csv
from finite_diff import successive_over_relaxation, jacobi_relaxation, \
    non_uniform_jacobi

EPSILON = 0.00001
X_QUERY = 0.06
Y_QUERY = 0.04
NUM_H_ITERATIONS = 6


def q3():
    o = q3b()
    h_values, potential_values, iterations_values = q3c(o)
    _, potential_values_jacobi, iterations_values_jacobi = q3d()
    plot_sor_jacobi(h_values, potential_values, potential_values_jacobi, iterations_values, iterations_values_jacobi)
    q3e()


def q3b():
    """
    Question 3(b): With h = 0.02, explore the effect of varying omega.

    :return: the best omega value found for SOR
    """
    print('\n=== Question 3(b) ===')
    h = 0.02
    min_num_iterations = float('inf')
    best_omega = float('inf')

    omegas = []
    num_iterations = []
    potentials = []

    for omega_diff in range(10):
        omega = 1 + omega_diff / 10
        print('Omega: {}'.format(omega))
        iter_relaxer = successive_over_relaxation(omega, EPSILON, h)
        print('Quarter grid: {}'.format(iter_relaxer.phi.mirror_horizontal()))
        print('Num iterations: {}'.format(iter_relaxer.num_iterations))
        potential = iter_relaxer.get_potential(X_QUERY, Y_QUERY)
        print('Potential at ({}, {}): {:.3f} V'.format(X_QUERY, Y_QUERY, potential))
        if iter_relaxer.num_iterations < min_num_iterations:
            best_omega = omega
        min_num_iterations = min(min_num_iterations, iter_relaxer.num_iterations)

        omegas.append(omega)
        num_iterations.append(iter_relaxer.num_iterations)
        potentials.append('{:.3f}'.format(potential))

    print('Best number of iterations: {}'.format(min_num_iterations))
    print('Best omega: {}'.format(best_omega))

    f = plt.figure()
    x_range = omegas
    y_range = num_iterations
    plt.plot(x_range, y_range, 'o-', label='Number of iterations')
    plt.xlabel('$\omega$')
    plt.ylabel('Number of Iterations')
    plt.grid(True)
    f.savefig('report/plots/q3b.pdf', bbox_inches='tight')

    save_rows_to_csv('report/csv/q3b_potential.csv', zip(omegas, potentials), header=('Omega', 'Potential (V)'))
    save_rows_to_csv('report/csv/q3b_iterations.csv', zip(omegas, num_iterations), header=('Omega', 'Iterations'))

    return best_omega


def q3c(omega):
    """
    Question 3(c): With an appropriate value of w, chosen from the above experiment, explore the effect of decreasing
    h on the potential.

    :param omega: the omega value to be used by SOR
    :return: the h values, potential values and number of iterations
    """
    print('\n=== Question 3(c): SOR ===')
    h = 0.04
    h_values = []
    potential_values = []
    iterations_values = []
    for i in range(NUM_H_ITERATIONS):
        h = h / 2
        print('h: {}'.format(h))
        print('1/h: {}'.format(1 / h))
        iter_relaxer = successive_over_relaxation(omega, EPSILON, h)
        # print(phi.mirror_horizontal())
        potential = iter_relaxer.get_potential(X_QUERY, Y_QUERY)
        num_iterations = iter_relaxer.num_iterations

        print('Num iterations: {}'.format(num_iterations))
        print('Potential at ({}, {}): {:.3f} V'.format(X_QUERY, Y_QUERY, potential))

        h_values.append(1 / h)
        potential_values.append('{:.3f}'.format(potential))
        iterations_values.append(num_iterations)

    f = plt.figure()
    x_range = h_values
    y_range = potential_values
    plt.plot(x_range, y_range, 'o-', label='Data points')

    plt.xlabel('1 / h')
    plt.ylabel('Potential at [0.06, 0.04] (V)')
    plt.grid(True)
    f.savefig('report/plots/q3c_potential.pdf', bbox_inches='tight')

    f = plt.figure()
    x_range = h_values
    y_range = iterations_values

    x_new = np.linspace(x_range[0], x_range[-1], num=len(x_range) * 10)
    polynomial_coeffs = poly.polyfit(x_range, y_range, deg=3)
    polynomial_fit = poly.polyval(x_new, polynomial_coeffs)
    N = sp.symbols("1/h")
    poly_label = sum(sp.S("{:.5f}".format(v)) * N ** i for i, v in enumerate(polynomial_coeffs))
    equation = '${}$'.format(sp.printing.latex(poly_label))
    plt.plot(x_new, polynomial_fit, '{}-'.format('C0'), label=equation)

    plt.plot(x_range, y_range, 'o', label='Data points')
    plt.xlabel('1 / h')
    plt.ylabel('Number of Iterations')
    plt.grid(True)
    plt.legend(fontsize='small')

    f.savefig('report/plots/q3c_iterations.pdf', bbox_inches='tight')

    save_rows_to_csv('report/csv/q3c_potential.csv', zip(h_values, potential_values), header=('1/h', 'Potential (V)'))
    save_rows_to_csv('report/csv/q3c_iterations.csv', zip(h_values, iterations_values), header=('1/h', 'Iterations'))

    return h_values, potential_values, iterations_values


def q3d():
    """
    Question 3(d): Use the Jacobi method to solve this problem for the same values of h used in part (c).

    :return: the h values, potential values and number of iterations
    """
    print('\n=== Question 3(d): Jacobi ===')
    h = 0.04
    h_values = []
    potential_values = []
    iterations_values = []
    for i in range(NUM_H_ITERATIONS):
        h = h / 2
        print('h: {}'.format(h))
        iter_relaxer = jacobi_relaxation(EPSILON, h)
        potential = iter_relaxer.get_potential(X_QUERY, Y_QUERY)
        num_iterations = iter_relaxer.num_iterations

        print('Num iterations: {}'.format(num_iterations))
        print('Potential at ({}, {}): {:.3f} V'.format(X_QUERY, Y_QUERY, potential))

        h_values.append(1 / h)
        potential_values.append('{:.3f}'.format(potential))
        iterations_values.append(num_iterations)

    f = plt.figure()
    x_range = h_values
    y_range = potential_values
    plt.plot(x_range, y_range, 'C1o-', label='Data points')
    plt.xlabel('1 / h')
    plt.ylabel('Potential at [0.06, 0.04] (V)')
    plt.grid(True)
    f.savefig('report/plots/q3d_potential.pdf', bbox_inches='tight')

    f = plt.figure()
    x_range = h_values
    y_range = iterations_values
    plt.plot(x_range, y_range, 'C1o', label='Data points')
    plt.xlabel('1 / h')
    plt.ylabel('Number of Iterations')

    x_new = np.linspace(x_range[0], x_range[-1], num=len(x_range) * 10)
    polynomial_coeffs = poly.polyfit(x_range, y_range, deg=4)
    polynomial_fit = poly.polyval(x_new, polynomial_coeffs)
    N = sp.symbols("1/h")
    poly_label = sum(sp.S("{:.5f}".format(v if i < 3 else -v)) * N ** i for i, v in enumerate(polynomial_coeffs))
    equation = '${}$'.format(sp.printing.latex(poly_label))
    plt.plot(x_new, polynomial_fit, '{}-'.format('C1'), label=equation)

    plt.grid(True)
    plt.legend(fontsize='small')

    f.savefig('report/plots/q3d_iterations.pdf', bbox_inches='tight')

    save_rows_to_csv('report/csv/q3d_potential.csv', zip(h_values, potential_values), header=('1/h', 'Potential (V)'))
    save_rows_to_csv('report/csv/q3d_iterations.csv', zip(h_values, iterations_values), header=('1/h', 'Iterations'))

    return h_values, potential_values, iterations_values


def q3e():
    """
    Question 3(e): Modify the program you wrote in part (a) to use the five-point difference formula derived in class
    for non-uniform node spacing.
    """
    print('\n=== Question 3(e): Non-Uniform Node Spacing ===')

    print('Jacobi (for reference)')
    iter_relaxer = jacobi_relaxation(EPSILON, 0.01)
    print('Quarter grid: {}'.format(iter_relaxer.phi.mirror_horizontal()))
    print('Num iterations: {}'.format(iter_relaxer.num_iterations))
    jacobi_potential = iter_relaxer.get_potential(X_QUERY, Y_QUERY)
    print('Potential at ({}, {}): {:.3f} V'.format(X_QUERY, Y_QUERY, jacobi_potential))

    print('Uniform Mesh (same as Jacobi)')
    x_values = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11]
    y_values = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11]
    iter_relaxer = non_uniform_jacobi(EPSILON, x_values, y_values)
    print('Quarter grid: {}'.format(iter_relaxer.phi.mirror_horizontal()))
    print('Num iterations: {}'.format(iter_relaxer.num_iterations))
    uniform_potential = iter_relaxer.get_potential(X_QUERY, Y_QUERY)
    print('Potential at ({}, {}): {:.3f} V'.format(X_QUERY, Y_QUERY, uniform_potential))
    print('Jacobi potential: {} V, same as uniform potential: {} V'.format(jacobi_potential, uniform_potential))

    print('Non-Uniform (clustered around (0.06, 0.04))')
    x_values = [0.00, 0.01, 0.02, 0.03, 0.05, 0.055, 0.06, 0.065, 0.07, 0.09, 0.1, 0.11]
    y_values = [0.00, 0.01, 0.03, 0.035, 0.04, 0.045, 0.05, 0.07, 0.08, 0.09, 0.1, 0.11]
    iter_relaxer = non_uniform_jacobi(EPSILON, x_values, y_values)
    print('Quarter grid: {}'.format(iter_relaxer.phi.mirror_horizontal()))
    print('Num iterations: {}'.format(iter_relaxer.num_iterations))
    potential = iter_relaxer.get_potential(X_QUERY, Y_QUERY)
    print('Potential at ({}, {}): {:.3f} V'.format(X_QUERY, Y_QUERY, potential))

    print('Non-Uniform (more clustered around (0.06, 0.04))')
    x_values = [0.00, 0.01, 0.02, 0.03, 0.055, 0.059, 0.06, 0.061, 0.065, 0.09, 0.1, 0.11]
    y_values = [0.00, 0.01, 0.035, 0.039, 0.04, 0.041, 0.045, 0.07, 0.08, 0.09, 0.1, 0.11]
    iter_relaxer = non_uniform_jacobi(EPSILON, x_values, y_values)
    print('Quarter grid: {}'.format(iter_relaxer.phi.mirror_horizontal()))
    print('Num iterations: {}'.format(iter_relaxer.num_iterations))
    potential = iter_relaxer.get_potential(X_QUERY, Y_QUERY)
    print('Potential at ({}, {}): {:.3f} V'.format(X_QUERY, Y_QUERY, potential))

    print('Non-Uniform (clustered near outer conductor)')
    x_values = [0.00, 0.020, 0.032, 0.044, 0.055, 0.06, 0.074, 0.082, 0.089, 0.096, 0.1, 0.14]
    y_values = [0.00, 0.020, 0.032, 0.04, 0.055, 0.065, 0.074, 0.082, 0.089, 0.096, 0.1, 0.14]
    iter_relaxer = non_uniform_jacobi(EPSILON, x_values, y_values)
    print('Quarter grid: {}'.format(iter_relaxer.phi.mirror_horizontal()))
    print('Num iterations: {}'.format(iter_relaxer.num_iterations))
    potential = iter_relaxer.get_potential(X_QUERY, Y_QUERY)
    print('Potential at ({}, {}): {:.3f} V'.format(X_QUERY, Y_QUERY, potential))

    plot_mesh(x_values, y_values)


def plot_mesh(x_values, y_values):
    f = plt.figure()
    ax = f.gca()
    ax.set_aspect('equal', adjustable='box')
    x_range = []
    y_range = []
    for x in x_values[:-1]:
        for y in y_values[:-1]:
            x_range.append(x)
            y_range.append(y)
    plt.plot(x_range, y_range, 'o', label='Mesh points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    f.savefig('report/plots/q3e.pdf', bbox_inches='tight')


def plot_sor_jacobi(h_values, potential_values, potential_values_jacobi, iterations_values, iterations_values_jacobi):
    f = plt.figure()
    plt.plot(h_values, potential_values, 'o-', label='SOR')
    plt.plot(h_values, potential_values_jacobi, 'o-', label='Jacobi')
    plt.xlabel('1 / h')
    plt.ylabel('Potential at [0.06, 0.04] (V)')
    plt.grid(True)
    plt.legend()
    f.savefig('report/plots/q3d_potential_comparison.pdf', bbox_inches='tight')

    f = plt.figure()
    plt.plot(h_values, iterations_values, 'o-', label='SOR')
    plt.plot(h_values, iterations_values_jacobi, 'o-', label='Jacobi')
    plt.xlabel('1 / h')
    plt.ylabel('Number of Iterations')
    plt.grid(True)
    plt.legend()
    f.savefig('report/plots/q3d_iterations_comparison.pdf', bbox_inches='tight')


if __name__ == '__main__':
    t = time.time()
    q3()
    print('Total runtime: {} s'.format(time.time() - t))
