from __future__ import division

import csv

import matplotlib.pyplot as plt
import time

import numpy.polynomial.polynomial as poly

import numpy as np
import sympy as sp

from finite_diff import CoaxialCableMeshConstructor, successive_over_relaxation, jacobi_relaxation

epsilon = 0.00001
x = 0.06
y = 0.04

NUM_H_ITERATIONS = 6


def q3b():
    print('=== Question 3(b) ===')
    h = 0.02
    min_num_iterations = float('inf')
    best_omega = float('inf')

    omegas = []
    num_iterations = []
    potentials = []

    for omega_diff in range(10):
        omega = 1 + omega_diff / 10
        print('Omega: {}'.format(omega))
        phi = CoaxialCableMeshConstructor().construct_symmetric_mesh(h)
        iter_relaxer = successive_over_relaxation(omega, epsilon, phi, h)
        print('Quarter grid: {}'.format(phi.mirror_horizontal()))
        # print(iter_relaxer.phi)
        print('Num iterations: {}'.format(iter_relaxer.num_iterations))
        potential = iter_relaxer.get_potential(x, y)
        print('Potential at ({}, {}): {:.3f} V'.format(x, y, potential))
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
    print('=== Question 3(c): SOR ===')
    h = 0.04
    h_values = []
    potential_values = []
    iterations_values = []
    for i in range(NUM_H_ITERATIONS):
        h = h / 2
        print('h: {}'.format(h))
        print('1/h: {}'.format(1 / h))
        phi = CoaxialCableMeshConstructor().construct_symmetric_mesh(h)
        iter_relaxer = successive_over_relaxation(omega, epsilon, phi, h)
        # print(phi.mirror_horizontal())
        potential = iter_relaxer.get_potential(x, y)
        num_iterations = iter_relaxer.num_iterations

        print('Num iterations: {}'.format(num_iterations))
        print('Potential at ({}, {}): {:.3f} V'.format(x, y, potential))

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
    print('=== Question 3(d): Jacobi ===')
    h = 0.04
    h_values = []
    potential_values = []
    iterations_values = []
    for i in range(NUM_H_ITERATIONS):
        h = h / 2
        print('h: {}'.format(h))
        phi = CoaxialCableMeshConstructor().construct_symmetric_mesh(h)
        iter_relaxer = jacobi_relaxation(epsilon, phi, h)
        potential = iter_relaxer.get_potential(x, y)
        num_iterations = iter_relaxer.num_iterations

        print('Num iterations: {}'.format(num_iterations))
        print('Potential at ({}, {}): {:.3f} V'.format(x, y, potential))

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


def save_rows_to_csv(filename, rows, header=None):
    with open(filename, "wb") as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def q3():
    o = q3b()
    h_values, potential_values, iterations_values = q3c(o)
    _, potential_values_jacobi, iterations_values_jacobi = q3d()
    plot_sor_jacobi(h_values, potential_values, potential_values_jacobi, iterations_values, iterations_values_jacobi)


if __name__ == '__main__':
    t = time.time()
    q3()
    print('Total runtime: {}'.format(time.time() - t))
