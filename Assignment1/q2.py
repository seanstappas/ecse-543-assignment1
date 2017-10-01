from linear_networks import create_finite_difference_mesh


def q1a():
    x = create_finite_difference_mesh(2, 4)
    print(x)


if __name__ == '__main__':
    q1a()
