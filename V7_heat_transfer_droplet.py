# Jose Flores
# 1/21/2025
# currnet date 3/27/25
import math
import numpy as np # array operations
import matplotlib.pyplot as plt # plottig 
import matplotlib.animation as animation # animation for plot
from matplotlib.animation import FuncAnimation 
import scipy 
from scipy.linalg import  solve


print('2D equation solver')
def explicit():
    plate_length = (int(input('Enter a derired plate_length(defaut=150): ') or 150)) 
    n = plate_length+1 
    x = np.linspace(0,plate_length,n)
    y = np.linspace(0,plate_length,n)
    max_iter_time = int(input('Enter a desired iter time(default iter time=100): ') or 100)
    Droplet_r = 10
    delta_x = plate_length/(n-1) # chnage with respect to x 
    rho1 = 2710 # g/cm^3 aluminum
    rho2 = 1.293 # glcm^3 air
    cp1 = .89 *1000 # J/g celcius * 1000 for kg  
    cp2 = 1.012 * 1000 # J/g celcius * 100 for kg 
    k_min = .062 # W/(mK) air
    k_max = 237 # W/(mK) aluminum 
    gamma=.25
    delta_t= gamma*(min(rho1,rho2)*min(cp1,cp2)/(k_max*delta_x**2))
    i_0 = int(n/2)
    j_0 = int(n/2)
    cond= np.empty((n,n))
    cappa = np.empty((n,n))
    cp_ar = np.empty((n,n))
    rho_ar = np.empty((n,n))

    #initalize solution: The grid of grid(k,i,j)
    grid = np.empty((max_iter_time, n, n))
    # inital conditions everywhere inside the grid
    grid_initial = 0.0
    # boundary conditions (fixed temperature)
    grid_top = 0.0
    grid_left = 0.0
    grid_bottom = 0.0
    grid_right = 0.0

    # set the inital conditions
    grid.fill(grid_initial)

    for i in range(0,n,1):
        for j in range(0,n, 1):
            if (math.sqrt((x[i]-x[i_0])**2+(y[j]-y[j_0])**2)<Droplet_r):
                cond[i][j] = k_max
                cp_ar[i][j] = cp1
                rho_ar[i][j] = rho1
                grid[0][i][j]=300
            else:
                cond[i][j] = k_min
                cp_ar[i][j] = cp2
                rho_ar[i][j] =rho2
    # set boundary conditions
    grid[:, (n-1):, :] = grid_top
    grid[:,:, :1] = grid_left
    grid[:,:1, 1:] = grid_bottom
    grid[:,:,(n-1):] = grid_right

    def calculate(grid):    
        for k in range(0,max_iter_time-1,1):
            for i in range(1,n-1,1):    
                for j in range(1,n-1, 1):
                    grid[k+1,i,j] = (cond[i][j] * ((1/delta_x**2)*delta_t/(rho_ar[i][j]*cp_ar[i][j])))*  (grid[k][i+1][j] + grid[k][i-1][j] + grid[k][i][j+1] + grid[k][i][j-1] - 4*grid[k][i][j]) + grid[k][i][j]
                        
        return grid

    def plotheatmap(grid_k, k):
        # clear the current plot figure
        plt.clf()
        plt.title(f"Temperatiure at t = {k*delta_t:.3f} unit time")
        plt.xlabel("x")
        plt.ylabel('y')

        # this is to plot grid_k(grid at time-setp k)
        plt.pcolormesh(grid_k, cmap=plt.cm.jet, vmin=0, vmax=100)
        plt.colorbar()

        return plt

    # do the calculation here
    grid = calculate(grid)

    def animate(k):
        plotheatmap(grid[k],k)

    anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=max_iter_time, repeat = False)
    anim.save("heat_equation_solutionV9.gif")
    print("Done!")


def implicit():
      
    grid_len = 1 # (int(input('Enter a derired plate_length(defaut=150): ') or 150)) 
   
    temp = int(input('Set a desired boundary temp: ') or 100)
    n_steps= int (input('Enter a desired step time: ') or 100)
    rho1 = 2710 # g/cm^3 aluminum 
    rho2 = 1.293 # g/cm^3 air 
    cp1 = .89 *1000 # J/g celcius * 1000 for kg  
    cp2 = 1.012 * 1000 # J/g celcuis * 1000 for kg 
    k_min = .062 # W/(mK) air
    k_max = 237 # W/(mK) aluminum 
    delta_x = grid_len/(n_steps-1) # chnage with respect to x 
    
   
    DCFL = .25 # stability condtion 
    delta_t = DCFL * (delta_x**2*rho1*cp1)/(k_max*(temp-0)) 
    gamma = k_max/(rho1*cp1)*(delta_t/delta_x**2)   # conductivity(K)/(rho*cp) * delta_t/delta-x^2 
    zeta = gamma
    mat = (1-2*zeta)
    #d_r = int(input('set a radius for the droplet': ))

    
    

    x_low = 0 
    x_high = 1
    t_array = np.zeros((n_steps,1))
    t_array[0][0]=temp
    t_array[n_steps-1][0]=temp
    t_current = t_array 
    
    print(t_array)
    lin = np.linspace(x_low, x_high, n_steps) 

    # Define parameters
    rows, cols = n_steps, n_steps  # Number of rows and columns
    x, y = zeta, mat        # Define the repeating values

    # Initialize a zero matrix
    matrix = np.zeros((rows, cols))

    # Define the base pattern
   # pattern = np.array([x, y, x])  # x, y, x pattern

    for i in range (0,n_steps-1):
        matrix [i,i+1] = -zeta
    for i in range (1,n_steps):
        matrix[i,i-1] =- zeta

    for i in range (0,n_steps):
        matrix[i,i] = (1+2*zeta)

    

    print (matrix)

    #x = solve(matrix,t_current)
    for i in range (10000):
       
        t_forward = solve (matrix,t_current)
        t_current[1:n_steps-1,0] =   t_forward[1:n_steps-1,0]
        #print('x=x', t_forward[1:grid_len-1,0])  
        #print (t_current)

    plt.plot(lin,t_forward)
    plt.show()
    
    return     

solver = int(input('Select 1 for explicit or any other # for implicit: '))

if (solver==1):
    explicit()
else :
    implicit()

