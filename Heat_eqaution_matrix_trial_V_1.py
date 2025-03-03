# Heat eqaution matrix trial V_1
# flores 

import numpy as np 
import scipy 
from scipy.linalg import  solve

l = 10
gamma = -2
mat = (1-2*gamma)

# format is A = np.[]array 
# B =np.[]array 
# X= solve(A,B)
# or x = np.linalg.solve(A,b)
u = np.zeros((l,1))
u.fill(10)

print(u)





# Define parameters
rows, cols = l, l  # Number of rows and columns
x, y = gamma, mat        # Define the repeating values

# Initialize a zero matrix
matrix = np.zeros((rows, cols), dtype=int)

# Define the base pattern
pattern = np.array([x, y, x])  # x, y, x pattern


# Populate the matrix with the shifting pattern
for i in range(1,rows): # runs when 1 is removed 
    start_index = i % cols  # Calculate shift position per row
    end_index = start_index + len(pattern)

    if end_index <= cols:
        # Insert pattern normally if it fits within the row
        matrix[i, start_index:end_index] = pattern
    else:
        # Handle wrapping when the pattern exceeds row length
        wrap_point = cols - start_index
        matrix[i, start_index:] = pattern[:wrap_point]  # Fill till row end
        matrix[i, :len(pattern) - wrap_point] = pattern[wrap_point:]  # Wrap to start
def set_bound(matric):
    rows =len(matrix)
    cols =len(matrix[0])
    for j in range (l):
        matrix[0][j]=0
        matrix[rows-1][j] = 0
    
    # Set first and last columns to 0 (including corners again, but that's fine)
    for i in range(rows):
        matrix[i][0] = 0
        matrix[i][cols-1] = 0
    matrix[0][0]=gamma
    matrix[l-1][l-1]=gamma
    
    return matrix




matrix =set_bound(matrix)

print (matrix)

x = solve(matrix,u)
print(x)
print('hi')

