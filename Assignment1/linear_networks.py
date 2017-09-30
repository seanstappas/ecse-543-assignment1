from choleski import choleski_solve


def solve_linear_network(A, Y, J, E):
    A_trans = A.transpose()
    p1 = A * Y
    A_new = p1 * A_trans
    b = A * (J - Y * E)
    return choleski_solve(A_new, b)
