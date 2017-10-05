import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from linear_networks import find_mesh_resistance


def find_mesh_resistances(banded=False):
    branch_resistance = 1000
    points = {}
    for n in range(2, 11):
        start_time = time.time()
        half_bandwidth = 2 * n + 1 if banded else None
        equivalent_resistance = find_mesh_resistance(n, branch_resistance, half_bandwidth=half_bandwidth)
        print('Equivalent resistance for {}x{} mesh: {:.2f} Ohms.'.format(n, 2 * n, equivalent_resistance))
        points[n] = equivalent_resistance
        print('Runtime: {} s.'.format(time.time() - start_time))
    return points


def q2ab():
    print('=== Question 2(a)(b) ===')
    return find_mesh_resistances(banded=False)


def q2c():
    print('=== Question 2(c) ===')
    return find_mesh_resistances(banded=True)


def q2d(points):
    print('=== Question 2(d) ===')
    f = plt.figure()
    ax = f.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_range = points.keys()
    y_range = points.values()
    plt.plot(x_range, y_range, 'o-', label='Resistance')
    plt.xlabel('Mesh Size, N')
    plt.ylabel('Resistance, R (Ohms)')
    plt.grid(True)
    # plt.legend()
    # plt.show()
    f.savefig('plots/q2d.pdf', bbox_inches='tight')


if __name__ == '__main__':
    q2ab()
    pts = q2c()
    q2d(pts)
