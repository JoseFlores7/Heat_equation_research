# Jose Flores
# 1/21/2025
# currnet date 2/10/15
# Stoffel checked this

import math
import numpy as np # array operations
import matplotlib.pyplot as plt # plottig 
import matplotlib.animation as animation # animation for plot
from matplotlib.animation import FuncAnimation 

print('2D equation solver')

length_x = int(input('Enter plate length x: '  ))
length_y = int(input('Enter plate length y: '  ))
nx = int (input('Enter iter time for x ( default enter 100):' ))
ny = int (input('Enter iter time for y ( default enter 100):' ))
x = np.linspace(0,length_x,nx)
y = np.linspace(0,length_y,ny)



r = int(input('Enter a radius for droplet: ' ))

delta_x = length_x/(nx-1)
delta_y = length_y/(ny-1)

if (delta_x != delta_y):
    print ('hi')
else:
    print ('hello')

rho1 = 2710 # g/cm^3
rho2 = 1.293
cp1 = .89 *1000 # J/g celcius * 1000 for kg  
cp2 = 1.012 * 1000

k_min = .062 # W/(mK) air
k_max = 237 # W/(mK) aluminum 
gamma=.25
delta_t= gamma*(min(rho1,rho2)*min(cp1,cp2)/(k_max*delta_x**2))




i_0 = length_x/2
j_0 = length_y/2


cond= np.empty((length_x,length_y))
cappa = np.empty((length_x,length_y))
cp_ar = np.empty((length_x,length_y))
rho_ar = np.empty((length_x,length_y))
           

#initalize solution: The grid of u(k,i,j)
u = np.empty((nx,length_x, length_y))

# inital conditions everywher inside the grid
u_initial = 0.0

# boundary conditions (fixed temperature)
u_top = 0.0
u_left = 0.0
u_bottom = 0.0
u_right = 0.0



# set the inital conditions
u.fill(u_initial)

for i in range(0,length_x,delta_x):
    for j in range(0,length_y, delta_y):
        if (math.sqrt((i-i_0)**2+(j-j_0)**2)<r):
            cond[i][j] = k_max
            cp_ar[i][j] = cp1
            rho_ar[i][j] = rho1
            u[0][i][j]=300
        else:
            cond[i][j] = k_min
            cp_ar[i][j] = cp2
            rho_ar[i][j] =rho2
# set boundary conditions
u[:, (plate_length-1):, :] = u_top
u[:,:, :1] = u_left
u[:,:1, 1:] = u_bottom
u[:,:,(plate_length-1):] = u_right



def calculate(u):    
    for k in range(0,max_iter_time-1,1):
        for i in range(1,length_x-1,delta_x):    
            for j in range(1,length_y-1, delta_y):
                u[k+1,i,j] = (cond[i][j] * ((1/delta_x**2)*delta_t/(rho_ar[i][j]*cp_ar[i][j])))*  (u[k][i+1][j] + u[k][i-1][j] + u[k][i][j+1] + u[k][i][j-1] - 4*u[k][i][j]) + u[k][i][j]
                
                
               
               
             
                
    return u

def plotheatmap(u_k, k):
    # clear the current plot figure
    plt.clf()
    plt.title(f"Temperatiure at t = {k*delta_t:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel('y')

    # this is to plot u_k(u at time-setp k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.colorbar()

    return plt

# do the calculation here
u = calculate(u)

def animate(k):
    plotheatmap(u[k],k)

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=nx, repeat = False)

anim.save("heat_equation_solutionV6.gif")

print("Done!")


