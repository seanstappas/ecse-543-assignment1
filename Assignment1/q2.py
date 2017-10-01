import time

from linear_networks import find_mesh_resistance


def q2ab():
    print('=== Question 2(a)(b) ===')
    resistance = 1000
    for n in range(2, 11):
        start_time = time.time()
        equivalent_resistance = find_mesh_resistance(n, resistance)
        print('Equivalent resistance for {}x{} mesh: {:.2f} Ohms.'.format(n, 2*n, equivalent_resistance))
        print('Runtime: {} s.'.format(time.time() - start_time))


def q2c():
    print('=== Question 2(c) ===')
    resistance = 1000
    for n in range(2, 11):
        start_time = time.time()
        half_bandwidth = 2*n + 1
        equivalent_resistance = find_mesh_resistance(n, resistance, half_bandwidth=half_bandwidth)
        print('Equivalent resistance for {}x{} mesh: {:.2f} Ohms.'.format(n, 2*n, equivalent_resistance))
        print('Runtime: {} s.'.format(time.time() - start_time))


if __name__ == '__main__':
    q2ab()
    q2c()
