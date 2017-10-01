from __future__ import division
from __future__ import print_function

import copy
import csv
from ast import literal_eval

import math


class Matrix:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        # return self.data.__str__()
        string = ''
        for row in self.data:
            string += '\n'
            for val in row:
                string += '{: >4}'.format(val)
        return string

    def __add__(self, other):
        A = self.data
        B = other.data
        if len(A) != len(B) or len(A[0]) != len(B[0]):
            raise ValueError('Incompatible matrix sizes for addition. Matrix A is {}x{}, but matrix B is {}x{}.'
                             .format(len(A), len(A[0]), len(B), len(B[0])))
        rows = len(A)
        cols = len(A[0])

        return Matrix([[A[row][col] + B[row][col] for col in range(cols)] for row in range(rows)])

    def __sub__(self, other):
        A = self.data
        B = other.data
        if len(A) != len(B) or len(A[0]) != len(B[0]):
            raise ValueError('Incompatible matrix sizes for subtraction. Matrix A is {}x{}, but matrix B is {}x{}.'
                             .format(len(A), len(A[0]), len(B), len(B[0])))
        rows = len(A)
        cols = len(A[0])

        return Matrix([[A[row][col] - B[row][col] for col in range(cols)] for row in range(rows)])

    def __mul__(self, other):
        A = self.data
        B = other.data
        m = len(A[0])
        n = len(A)
        p = len(B[0])
        if m != len(B):
            raise ValueError('Incompatible matrix sizes for multiplication. Matrix A is {}x{}, but matrix B is {}x{}.'
                             .format(n, m, len(B), p))

        # Inspired from https://en.wikipedia.org/wiki/Matrix_multiplication
        product = Matrix.empty(n, p)
        for i in range(n):
            for j in range(p):
                row_sum = 0
                for k in range(m):
                    row_sum += A[i][k] * B[k][j]
                product[i][j] = row_sum
        return product

    def is_positive_definite(self):
        A = copy.deepcopy(self.data)
        n = len(A)
        for j in range(n):
            if A[j][j] <= 0:
                print(A)
                return False
            A[j][j] = math.sqrt(A[j][j])
            for i in range(j + 1, n):
                A[i][j] = A[i][j] / A[j][j]
                for k in range(j + 1, i + 1):
                    A[i][k] = A[i][k] - A[i][j] * A[k][j]
        return True

    def transpose(self):
        rows = len(self.data)
        cols = len(self.data[0])
        return Matrix([[self.data[row][col] for row in range(rows)] for col in range(cols)])

    @staticmethod
    def multiply(*matrices):
        n = len(matrices[0])
        product = Matrix.identity(n)
        for matrix in matrices:
            product = product * matrix
        return product

    @staticmethod
    def empty(rows, cols):
        """
        Returns an empty matrix (filled with zeroes) with the specified number of columns and rows.

        :param rows: number of rows
        :param cols: number of columns
        :return: the empty matrix
        """
        return Matrix([[0 for col in range(cols)] for row in range(rows)])

    @staticmethod
    def identity(n):
        return Matrix([[1 if row == col else 0 for col in range(n)] for row in range(n)])

    @staticmethod
    def diagonal(values):
        n = len(values)
        return Matrix([[values[row] if row == col else 0 for col in range(n)] for row in range(n)])

    @staticmethod
    def column_vector(values):
        """
        Transforms a row vector into a column vector.

        :param values: the values, one for each row of the column vector
        :return: the column vector
        """
        return Matrix([[value] for value in values])

    @staticmethod
    def csv_to_matrix(filename):
        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file)
            data = []
            for row_number, row in enumerate(reader):
                data.append([literal_eval(val) for val in row])
            return Matrix(data)
