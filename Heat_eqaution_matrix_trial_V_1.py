# Heat eqaution matrix trial V_1
# flores 

import numpy as np 
import matplotlib.pyplot as plt # plottig 
import matplotlib.animation as animation # animation for plot
from matplotlib.animation import FuncAnimation 
import scipy 
from scipy.linalg import  solve

grid_len = 10
gamma = -2
mat = (1-2*gamma)
temp =100
n_steps= 10

x_low = 0 
x_high = 1
t_array = np.zeros((grid_len,1))

t_array[0][0]=temp
t_array[grid_len-1][0]=temp
t_current = t_array 
print(t_array)
lin = np.linspace(x_low, x_high, grid_len) 


# Define parameters
rows, cols = grid_len, grid_len  # Number of rows and columns
x, y = gamma, mat        # Define the repeating values

# Initialize a zero matrix
matrix = np.zeros((rows, cols), dtype=int)

# Define the base pattern
pattern = np.array([x, y, x])  # x, y, x pattern

for i in range (0,grid_len-1):
    matrix [i,i+1] = gamma
for i in range (1,grid_len):
     matrix[i,i-1] = gamma

for i in range (0,grid_len):
     matrix[i,i] = 1-2*gamma

def set_bound(matric):
    rows =len(matrix)
    cols =len(matrix[0])
    for j in range (grid_len):
        matrix[0][j]=0
        matrix[rows-1][j] = 0
   
    # Set first and last columns to 0 (including corners again, but that's fine)
    for i in range(rows):
        matrix[i][0] = 0
        matrix[i][cols-grid_len] = 0
    matrix[0][0]=gamma
    matrix[grid_len-1][grid_len-1]=gamma
    
    return matrix

#matrix =set_bound(matrix)

print (matrix)

#x = solve(matrix,t_current)
for i in range (n_steps):
    t_forward = solve (matrix,t_current)
    t_current[1:grid_len-1,0] = t_forward[1:grid_len-1,0]
    #print('x=x', t_forward[1:grid_len-1,0])  
    #print (t_current)




plt.plot(lin,t_forward)
plt.show()

