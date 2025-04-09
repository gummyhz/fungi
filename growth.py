# data managemement
import numpy as np 
import pandas as pd 

# plotting
import matplotlib as mpl 
import matplotlib.pyplot as plt 

# math methods
from scipy.integrate import solve_ivp 
from scipy.ndimage import convolve

# Fisher Kolmogorov model
# maybe numpy or scipy functions would be more efficient or accurate
# here I will do everything by hand for clarity 

### Parameters:
# t: time, for compliance with solve_ivp call
# u: vector of population density at each 
# D: diffusion constant
# r: growth rate
# K: carrying capacity
def fk(t, u, D, r, K):
    N = len(u)
    u_t = np.zeros(N) # du/dt

    for i in range(0, N):
        # using the (second order) central difference approximation
        # \[ f``(x) ~ \frac{f(x+h) - 2f(x) + f(x-h)}{h^2} \] 
        # here, h will be 1
        if (i == N - 1):
            u_xx = - 2*u[i] + u[i-1]
        elif (i == 0):
            u_xx = u[i+1] - 2*u[i]
        else:
            u_xx = u[i+1] - 2*u[i] + u[i-1]

        u_t[i] = (D * u_xx) + (r * u[i] * (1 - (u[i]/K)))
    return u_t

def fk_polar():
    N = len(u)
    # https://www.math.ucdavis.edu/~saito/courses/21C.w11/polar-lap.pdf
    # Polar Laplacian: \[ u_{xx} + u_{yy} = u_{rr} + \frac{1}{r}u_r + \frac{1}{r^2}u_{\theta\theta\]

# 2-D Fisher Kolmorogov using my direct interpretation of the laplacian formula
def fk_2d(u, dt, D, r, K):
    nx = len(u)
    ny = len(u[0])
    u_t = np.zeros((nx, ny))
    u_new = u

    # using the (second order) central difference approximation
    for i in range(0,nx):
        for j in range(0,ny):
            # find u_xx 
            if (i == nx - 1):
                u_xx = - 2*u[i][j] + u[i-1][j]
            elif (i == 0):
                u_xx = u[i+1][j] - 2*u[i][j]
            else:
                u_xx = u[i+1][j] - 2*u[i][j] + u[i-1][j]

            # find u_yy
            if (j == ny - 1):
                u_yy = - 2*u[i][j] + u[i][j-1]
            elif (j == 0):
                u_yy = u[i][j+1] - 2*u[i][j]
            else:
                u_yy = u[i][j+1] - 2*u[i][j] + u[i][j-1]

            u_t[i][j] = (D * (u_xx + u_yy)) + (r * u[i][j] * (1 - (u[i][j]/K)))
            u_new[i][j] = u[i][j] + dt*u_t[i][j]
    
    return u_new

# 2-D Fisher Kolmorogov using a Laplacian Kernel to approximate the second derivative
def fk_2d_conv(u, dt, dx, dy, D, r, K):
    u_new = u.copy()

    # using the Laplacian kernel approximation
    laplacian_kernel = np.array([[0,  1,  0],
                                 [1, -4,  1],
                                 [0,  1,  0]])
    u_lap = convolve(u, laplacian_kernel, mode='constant') / dx**2
    
    u_t = D * u_lap + r * u * (1 - u/K)
    u_new = u + dt*u_t
    return u_new

# Simulate Fisher Kolmorogov Model
def fk_sim_1d():
    # Define parameters
    D = 0.1      # Diffusion constant
    r = 1        # Growth rate
    K = 1        # Carrying capacity (1 because it is density being measured)
    T = 20       # Number of time points to evaulate at
    N = 100      # Number of points

    # Initial conditions
    u0 = np.zeros(N)
    u0[int(N/2 - N/10):int(N/2 + N/10)] = 1

    # Time vector
    t_span = (0, T)
    t_eval = range(0,20,3)

    # Solve the equation using solve_ivp
    sol = solve_ivp(fk, t_span, u0, args=(D, r, K), dense_output=True, t_eval=t_eval, method='RK45')

    x = range(0,100)
    # Plot the solution
    for i in range(len(sol.t)):
        plt.plot(x, sol.sol(sol.t[i]), label=f't={sol.t[i]:.2f}')
    
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('Fisher-Kolmogorov Equation Solution')
    plt.legend()
    plt.grid(True) # add grid lines
    plt.show()


# Simulate Fisher Kolmorogov Model
def fk_sim_2d():
    # Define parameters
    D = 0.1    # Diffusion constant
    r = .9       # Growth rate
    K = .8       # Carrying capacity (1 because it is density being measured)
    
    T = 10 # total duration
    dt = .01 # time step
    nt = int(T/dt)     # Number of time points to evaulate at
    
    xmax = 10
    ymax = 10
    dx = 0.1
    dy = 0.1 # spatial step lengths
    nx = int(xmax/dx)    # Number of points in x
    ny = int(ymax/dy)    # Number of points in y

    # Initial conditions
    u = np.zeros((nx,ny))
    u[int(nx/2-nx/4):int(nx/2+nx/4), int(ny/2-ny/4):int(ny/2+ny/4)] = 0.9 # set seed


    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(u, extent=[0, xmax, 0, ymax], vmin=0, vmax=1, origin='lower', cmap='plasma')
    fig.colorbar(im)
 
    for n in range(nt):
        # comment out either line to choose which method to use to update u
        #u = fk_2d(u, dt, D, r, K)
        u = fk_2d_conv(u, dt, dx, dy, D, r, K)
    
        if n % 10 == 0:
            im.set_data(u)
            ax.set_title(f'Time: {n*dt:.2f}')
            fig.canvas.draw_idle()
            plt.pause(0.01)

    plt.ioff()
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('Fisher-Kolmogorov Equation Solution')
    plt.show()

fk_sim_2d()