import math


def choleski(A, x, b):
    if elimination(A, b):
        back_substitution(A, x, b)


def elimination(A, b):
    n = len(A)
    for j in range(n):
        if A[j][j] <= 0:
            return False  # Error flag: A not positive definite
        A[j][j] = math.sqrt(A[j][j])
        b[j][0] = b[j][0] / A[j][j]
        for i in range(j + 1, n):
            A[i][j] = A[i][j] / A[j][j]
            b[i][0] = b[i][0] - A[i][j] * b[j][0]
            for k in range(j + 1, i + 1):
                A[i][k] = A[i][k] - A[i][j] * A[k][j]
    return True


def back_substitution(L, x, y):
    n = len(L)
    for i in range(n):
        prev_sum = 0
        for j in range(i + 1, n):
            prev_sum += L[j][i] * x[j][0]
        x[i][0] = (y[i][0] - prev_sum) / L[i][i]


def is_positive_definite(A):
    n = len(A)
    for j in range(n):
        if A[j][j] <= 0:
            return False  # Error flag: A not positive definite
        A[j][j] = math.sqrt(A[j][j])
        for i in range(j + 1, n):
            A[i][j] = A[i][j] / A[j][j]
            for k in range(j + 1, i + 1):
                A[i][k] = A[i][k] - A[i][j] * A[k][j]
    return True
