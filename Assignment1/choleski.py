import math


def choleski(A, x, b):
    elimination(A, b)
    back_substitution(A, x, b)


def elimination(A, b):
    n = len(A)
    for j in range(n):
        if A[j][j] <= 0:
            return -1  # Error flag: A not positive definite
        A[j][j] = math.sqrt(A[j][j])
        b[j] = b[j] / A[j][j]
        for i in range(j + 1, n):
            A[i][j] = A[i][j] / A[j][j]
            b[i] = b[i] - A[i][j] * b[j]
            for k in range(j + 1, i + 1):
                A[i][k] = A[i][k] - A[i][j] * A[k][j]


def back_substitution(L, x, y):
    n = len(L)
    for i in range(n):
        prev_sum = 0
        for j in range(i + 1, n):
            prev_sum += L[j][i] * x[j]
        x[i] = (y[i] - prev_sum) / L[i][i]