import csv
from ast import literal_eval


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
    return [[0 for col in range(cols)] for row in range(rows)]


def transpose(matrix):
    """
    Returns the transpose of the given matrix.

    :param matrix: the matrix to take the transpose of
    :return: the transpose of the matrix
    """
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[row][col] for row in range(rows)] for col in range(cols)]


def column_vector(*values):
    """
    Returns a column vector (matrix) with the specified values.

    :param values: the values, one for each row of the column vector
    :return: the column vector
    """
    return [[value] for value in values]


def csv_to_matrix(filename):
    with open('network_branches.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        matrix = []
        for row_number, row in enumerate(reader):
            matrix.append([literal_eval(val) for val in row])
        return matrix
