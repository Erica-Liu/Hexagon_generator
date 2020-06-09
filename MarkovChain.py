from sympy import *
import numpy as np

M = Matrix([[0, 0.2, 0.5, 1],
            [0, 0.4, 0.25, 0],
            [1, 0.4, 0, 0],
            [0, 0, 0.25, 0]])

print("Matrix : {} ".format(M))

# Use sympy.diagonalize() method
P, D = M.diagonalize()
#p = np.array(P).astype(np.float64)
#d = np.array(D).astype(np.float64)
p_inf = float("inf")

n = symbols('n')
for i in range(4):
    a = D[i,i]
    print(a)

"""
expr = pow(a,n);

print("Expression : {}".format(expr))

# Use sympy.limit() method
limit_expr = limit(expr, n, p_inf)

print("Limit of the expression tends to 0 : {}".format(limit_expr))
"""

print("Diagonal of a matrix : {}".format(D))
print("P of a matrix : {}".format(P))

D_n = Matrix([[1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]])

D_n_1 = D ** 100
P_inv = P**-1
print("Inverse P of a matrix : {}".format(P_inv))
Dis_1 = P * D_n_1

print("distribution of a matrix : {}".format(Dis_1))
Dis_2 = Dis_1 * P_inv

print("distribution of a matrix : {}".format(Dis_2))
print(simplify(Dis_2))