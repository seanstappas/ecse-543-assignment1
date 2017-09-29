from matrix_utilities import matrix_multiply, empty_matrix
from choleski import is_positive_definite, choleski

matrix_2 = [
    [1, 0],
    [0, 2]
]

matrix_3 = [
    [3, 0, 0],
    [0, 2, 0],
    [0, 0, 1]
]

matrix_4 = [
    [5, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 2, 0],
    [0, 0, 0, 7]
]
matrices = [matrix_2, matrix_3, matrix_4]

x_2 = [
    [8],
    [3]
]

x_3 = [
    [9],
    [4],
    [3]
]

x_4 = [
    [5],
    [4],
    [1],
    [9]
]


def q1b():
    for count, A in enumerate(matrices):
        n = count + 2
        print('n={} matrix is positive-definite: {}'.format(n, is_positive_definite(A)))


def q1c():
    xs = [x_2, x_3, x_4]
    for count, (x, A) in enumerate(zip(xs, matrices)):
        n = count + 2
        b = matrix_multiply(A, x)
        print('A: {}'.format(A))
        print('x: {}'.format(x))
        print('b: {}'.format(b))

        x_result = empty_matrix(n, 1)
        choleski(A, x_result, b)
        print('x_result: {}'.format(x_result))  # TODO: Assert equal here (to number of sig figs)


if __name__ == '__main__':
    # q1b()
    q1c()
