import numpy as np
from scipy import sparse

vector_row = np.array([1,2,3])
print(vector_row)

vector_col = np.array([[1],[2],[3]])
print(vector_col)

matrix = np.array([[1,2],[2,3],[3,4]])
print(matrix)

vetor = np.array([1,2,3,4,5,6])
print(vetor[2])
print(vetor[2:])
matrix = np.array([[1,2,3],[1,3,5],[5,6,7]])
print(matrix[1,1])
print(matrix[:1,1:])


print(matrix.shape)
print(matrix.size)
print(matrix.ndim)
print(100+matrix)

######

print(np.max(matrix, axis=0))
print(np.mean(matrix))

#####
matrix = np.array([[1,2,3], [4,5,6], [7,8,9],[1,3,6]])
print(matrix)
print(matrix.T)
print(matrix.reshape(2,6))
print(matrix.reshape(1,-1))
print(np.linalg.matrix_rank(matrix))

#####
matrix = np.array([[1,2,3],[1,3,5],[5,6,7]])
print(np.linalg.det(matrix))
print(np.diagonal(matrix))


matrix_a = ([[1,4,3],[423,534,43],[2,4,10]])
matrix_b = ([[3,5,1],[4,32,534],[34,43,1]])

print(np.add(matrix_a, matrix_b))
print(np.subtract(matrix_a, matrix_b))


print(np.dot(matrix_a, matrix_b))

print(np.linalg.inv(matrix_a))