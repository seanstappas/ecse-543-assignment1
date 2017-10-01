from linear_networks import create_incidence_matrix_from_mesh


def q1a():
    for n in range(2, 11):
        x = create_incidence_matrix_from_mesh(n, 2 * n)
        print(x)


if __name__ == '__main__':
    q1a()
