import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import sympy as sp
from matplotlib.ticker import MaxNLocator

from csv_saver import save_rows_to_csv
from linear_networks import find_mesh_resistance


def q2():
    """
    Question 2
    """
    runtimes1, choleski_runtimes1 = q2ab()
    pts, runtimes2, choleski_runtimes2 = q2c()
    plot_runtimes(runtimes1, runtimes2)
    plot_runtimes(choleski_runtimes1, choleski_runtimes2, True)
    q2d(pts)


def q2ab():
    """
    Question 2(a): Using the program you developed in question 1, find the resistance, R, between the node at the
    bottom left corner of the mesh and the node at the top right corner of the mesh, for N = 2, 3, ..., 10.

    Question 2(b):Are the timings you observe for your practical implementation consistent with this?

    :return: the timings for finding the mesh resistance for N = 2, 3 ... 10
    """
    print('\n=== Question 2(a)(b) ===')
    _, runtimes, choleski_runtimes = find_mesh_resistances(banded=False)
    save_rows_to_csv('report/csv/q2b.csv', zip(runtimes.keys(), runtimes.values()), header=('N', 'Runtime (s)'))
    save_rows_to_csv('report/csv/q2b_choleski.csv', zip(choleski_runtimes.keys(), choleski_runtimes.values()),
                     header=('N', 'Runtime (ms)'))
    return runtimes, choleski_runtimes


def q2c():
    """
    Question 2(c): Modify your program to exploit the sparse nature of the matrices to save computation time.

    :return: the mesh resistances and the timings for N = 2, 3 ... 10
    """
    print('\n=== Question 2(c) ===')
    resistances, runtimes, choleski_runtimes = find_mesh_resistances(banded=True)
    save_rows_to_csv('report/csv/q2c.csv', zip(runtimes.keys(), runtimes.values()), header=('N', 'Runtime (s)'))
    save_rows_to_csv('report/csv/q2c_choleski.csv', zip(choleski_runtimes.keys(), choleski_runtimes.values()),
                     header=('N', 'Runtime (ms)'))
    return resistances, runtimes, choleski_runtimes


def q2d(resistances):
    """
    Question 2(d): Plot a graph of R versus N. Find a function R(N) that fits the curve reasonably well and is
    asymptotically correct as N tends to infinity, as far as you can tell.

    :param resistances: a dictionary of resistance values for each N value
    """
    print('\n=== Question 2(d) ===')
    f = plt.figure()
    ax = f.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_range = [float(x) for x in resistances.keys()]
    y_range = [float(y) for y in resistances.values()]
    plt.plot(x_range, y_range, 'o', label='Data points')

    x_new = np.linspace(x_range[0], x_range[-1], num=len(x_range) * 10)
    coeffs = poly.polyfit(np.log(x_range), y_range, deg=1)
    polynomial_fit = poly.polyval(np.log(x_new), coeffs)
    plt.plot(x_new, polynomial_fit, '{}-'.format('C0'), label='${:.2f}\log(N) + {:.2f}$'.format(coeffs[1], coeffs[0]))

    plt.xlabel('N')
    plt.ylabel('R ($\Omega$)')
    plt.grid(True)
    plt.legend()
    f.savefig('report/plots/q2d.pdf', bbox_inches='tight')
    save_rows_to_csv('report/csv/q2a.csv', zip(resistances.keys(), resistances.values()), header=('N', 'R (Omega)'))


def find_mesh_resistances(banded):
    branch_resistance = 1000
    points = {}
    total_runtimes = {}
    choleski_runtimes = {}
    for n in range(2, 11):
        start_time = time.time()
        half_bandwidth = 2 * n + 1 if banded else None
        equivalent_resistance, choleski_runtime = find_mesh_resistance(n, branch_resistance, half_bandwidth=half_bandwidth)
        print('Equivalent resistance for {}x{} mesh: {:.2f} Ohms.'.format(n, 2 * n, equivalent_resistance))
        points[n] = '{:.3f}'.format(equivalent_resistance)
        runtime = time.time() - start_time
        total_runtimes[n] = '{:.3f}'.format(runtime)
        print('Choleski runtime: {} ms'.format(choleski_runtime))
        choleski_runtimes[n] = '{:.3f}'.format(choleski_runtime)
        print('Runtime: {} s.'.format(runtime))
    plot_runtime(total_runtimes, banded)
    plot_runtime(choleski_runtimes, banded, True)
    return points, total_runtimes, choleski_runtimes


def plot_runtime(points, banded=False, choleski=False):
    f = plt.figure()
    ax = f.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_range = [float(x) for x in points.keys()]
    y_range = [float(y) for y in points.values()]
    plt.plot(x_range, y_range, '{}o'.format('C1' if banded else 'C0'), label='Data points')

    x_new = np.linspace(x_range[0], x_range[-1], num=len(x_range) * 10)
    degree = 6 if not choleski else (4 if banded else 5)
    polynomial_coeffs = poly.polyfit(x_range, y_range, degree)
    polynomial_fit = poly.polyval(x_new, polynomial_coeffs)
    N = sp.symbols("N")
    poly_label = sum(sp.S("{:.4f}".format(v)) * N ** i for i, v in enumerate(polynomial_coeffs))
    equation = '${}$'.format(sp.printing.latex(poly_label))
    plt.plot(x_new, polynomial_fit, '{}-'.format('C1' if banded else 'C0'), label=equation)

    plt.xlabel('N')
    plt.ylabel('Runtime ({})'.format('ms' if choleski else 's'))
    plt.grid(True)
    plt.legend(fontsize='x-small')
    f.savefig('report/plots/q2{}{}.pdf'.format('c' if banded else 'b', '_choleski' if choleski else ''),
              bbox_inches='tight')


def plot_runtimes(points1, points2, choleski=False):
    f = plt.figure()
    ax = f.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_range = points1.keys()
    y_range = points1.values()
    y_banded_range = points2.values()
    plt.plot(x_range, y_range, 'o-', label='Non-banded elimination')
    plt.plot(x_range, y_banded_range, 'o-', label='Banded elimination')
    plt.xlabel('N')
    plt.ylabel('Runtime ({})'.format('ms' if choleski else 's'))
    plt.grid(True)
    plt.legend()
    f.savefig('report/plots/q2bc{}.pdf'.format('_choleski' if choleski else ''), bbox_inches='tight')


if __name__ == '__main__':
    q2()
