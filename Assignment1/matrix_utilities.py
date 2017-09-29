def matrix_multiply(A, B):
    """
    Multiplies the given matrices.

    :param A: the first matrix to multiply
    :param B: the second matrix to multiply
    :return: the multiplication of the matrices
    """
    m = len(A[0])
    if m != len(B):
        return False
    n = len(A)
    p = len(B[0])

    # Inspired from https://en.wikipedia.org/wiki/Matrix_multiplication
    product = empty_matrix(n, p)
    for i in range(n):
        for j in range(p):
            row_sum = 0
            for k in range(m):
                row_sum += A[i][k] * B[k][j]
            product[i][j] = row_sum
    return product


def empty_matrix(rows, cols):
    """
    Returns an empty matrix (filled with zeroes) with the specified number of columns and rows.

    :param rows: number of rows
    :param cols: number of columns
    :return: the empty matrix
    """
    return [[0 for row in range(cols)] for col in range(rows)]
