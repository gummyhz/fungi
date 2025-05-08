# data managemement
import numpy as np 
import pandas as pd 

# plotting
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from matplotlib.axis import Axis 
import imageio
import os

# math methods
from scipy.integrate import solve_ivp 
from scipy.ndimage import convolve

# I am using u here for tip density, and K for carrying capacity
# despite using variables which follow my references in my paper.
# Initially I wanted my code to reflect more general Fisher-Kolmogorov 
# formulas. I regret this, but at this point should finish the
# writing before refactoring the code.


# 1-D Fisher Kolmogorov 
def fk_1d(t, u, D, r, K):
    ### Parameters:
    # t: time, for compliance with solve_ivp call
    # u: population density at each spatial step
    # D: diffusion constant
    # r: growth rate
    # K: carrying capacity

    N = len(u)
    u_t = np.zeros(N) # du/dt, initialized to 0s

    for i in range(0, N):
        # using the (second order) central difference approximation, with h=1
        # \[ f``(x) ~ \frac{f(x+h) - 2f(x) + f(x-h)}{h^2} \] 
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
    # Polar Laplacian: \[ u_{xx} + u_{yy} = u_{rr} + \frac{1}{r}u_r + \frac{1}{r^2}u_{\theta\theta\]

# 2-D Fisher Kolmogorov using my direct interpretation of the laplacian formula
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

# 2-D Fisher Kolmogorov using a Laplacian Kernel to approximate the second derivative
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
    D = 0.0008     # Diffusion constant, mm/h
    r = 1          # Growth rate
    K = 0.3     # Carrying capacity, tips/mm
    T = 150          # Number of time points to evaulate at
    N = 100         # Number of points

    # Initialize vector of tip densities at each position
    u0 = np.zeros(N)
    # x0 = int(N/2)
    u0[int(N/2 - 5):int(N/2 + 5)] = K
    # for i in range(N):
    #     u0[i] = K*np.exp(-((u0[i]-x0)**2) / (2 * 3**2))

    # Time vector
    t_span = (0, T)
    t_eval = range(0,T,25)

    # Solve the equation using solve_ivp
    sol = solve_ivp(fk_1d, t_span, u0, args=(D, r, K), dense_output=True, t_eval=t_eval, method='RK45')

    x = range(0,N)
    
    plt.xlabel('x (mm)')
    plt.ylabel('n(x,t)')
    plt.title('Fisher-Kolmogorov Equation Solution')

    # Plot the solution
    for i in range(len(sol.t)):
        plt.plot(x, sol.sol(sol.t[i]), label=f't={sol.t[i]:.2f}')
    
    plt.legend()
    plt.grid(True) # add grid lines
    plt.show()


# Simulate Fisher Kolmorogov Model
def fk_sim_2d():
    # define parameters
    D = 0.0008 # Diffusion constant
    r = 1  # Growth rate
    K = 0.3  # Carrying capacity
    
    T = 10          # Total duration
    dt = .01        # Time step
    nt = int(T/dt)  # Number of time points to evaulate at
    
    xmax = 10           # Max y value (min=0)
    ymax = 10           # Max y value (min=0)
    dx = 0.1            # Spatial step length in x
    dy = 0.1            # Spatial step length in y
    nx = int(xmax/dx)   # Number of points in x
    ny = int(ymax/dy)   # Number of points in y

    # Make a temporary directory for storing frames for GIF
    frames = []
    os.makedirs("frames", exist_ok=True)

    # Vector of tip densities at each position
    u = np.zeros((nx,ny))
    # Initialize to small gaussian at center
    x, y = np.linspace(0, xmax, nx), np.linspace(0, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    x0, y0 = int(xmax/2), int(ymax/2)
    sigma_u = 0.5 
    u = K*np.exp(-((X-x0)**2 + (Y-y0)**2) / (2 * sigma_u**2))
    
    # Set up plot
    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(u, extent=[0, xmax, 0, ymax], vmin=0, vmax=K, origin='lower', cmap='plasma')
    fig.colorbar(im).set_label('Tip Density', labelpad=10)
 
    # Plot at each time step
    for n in range(nt):
        # comment out either line to choose which method to use to update u
        # u = fk_2d(u, dt, D, r, K) # direct implementation of Laplacian formula
        u = fk_2d_conv(u, dt, dx, dy, D, r, K) # Convolution of Laplacian kernel
    
        if n % 10 == 0:
            im.set_data(u)
            ax.set_title(f'Time: {n*dt:.2f}')
            fig.canvas.draw_idle()
            plt.pause(0.01)

            # Save frame for GIF
            filename = f"frames/frame_{n:04d}.png"
            plt.savefig(filename)
            frames.append(filename)

    # additional plot settings
    plt.ioff()
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title('Fisher-Kolmogorov Equation Solution')
    plt.show()

    # Create GIF
    with imageio.get_writer("simulation.gif", mode="I", duration=0.1) as writer:
        for f in frames:
            image = imageio.imread(f)
            writer.append_data(image)

    # Delete frames
    for f in frames:
        os.remove(f)
    
def fk_2d_withFilament():
    # units milimeters
    r = 0.9         # growth rate
    alpha = 0.039   # branching rate
    beta = 0.022    # anastomosis rate 
    c = .280        # speed of traveling wavefront
    D = 0.0008      # diffusion constant 
    v = .240        # velocity of tips
    K = .3          # tip saturation density
    
    T = 10          # Total duration
    dt = .01        # Time step
    nt = int(T/dt)  # Number of time points to evaulate at
    
    xmax = 10           # Max y value (min=0)
    ymax = 10           # Max y value (min=0)
    dx = 0.1            # Spatial step length in x
    dy = 0.1            # Spatial step length in y
    nx = int(xmax/dx)   # Number of points in x
    ny = int(ymax/dy)   # Number of points in y

    # intialize u and rho
    x, y = np.linspace(0, xmax, nx), np.linspace(0, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    x0, y0 = 5, 5
    sigma_u = 0.5 
    # tip density - starting with a small gaussian
    u = K*np.exp(-((X-x0)**2 + (Y-y0)**2) / (2 * sigma_u**2))
    # filament density - starting with 0
    rho = np.zeros((nx,ny))

    # set up plots
    plt.ion()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    ax[0].grid(False)
    ax[1].grid(False)
    imu = ax[0].imshow(u, extent=[0, xmax, 0, ymax], vmin=0, vmax=K, origin='lower', cmap='plasma')
    imrho = ax[1].imshow(u, extent=[0, xmax, 0, ymax], vmin=0, vmax=K, origin='lower', cmap='plasma')
    fig.colorbar(imu).set_label('Tip Density', labelpad=10)
    fig.colorbar(imrho).set_label('Filament Density', labelpad=10)
    ax[0].set_xlabel('x (mm)')
    ax[1].set_xlabel('x (mm)')
    ax[0].set_ylabel('y (mm)')
    ax[1].set_ylabel('y (mm)')
 
    # plot at each time step
    for n in range(nt+1):
        rho_t = v*u
        laplacian_kernel = np.array([[0,  1,  0],
                                 [1, -4,  1],
                                 [0,  1,  0]])
        u_lap = convolve(u, laplacian_kernel, mode='constant') / dx**2
        u_t = D * u_lap + r * u * (1 - u/K)

        u = u + dt*u_t
        rho = rho + dt*rho_t
        
        if n % 10 == 0: # plot at time step (every 10 steps)
            imu.set_data(u)
            imrho.set_data(u)
            ax[0].set_title(f'Tips Density at Time: {n*dt:.2f}h')
            ax[1].set_title(f'Filament Density at Time: {n*dt:.2f}h')
            fig.canvas.draw_idle()
            plt.pause(0.01)

    # plot
    plt.ioff()
    plt.show()


# END FUNCITON DEFINITIONS ------------------------
# fk_sim_1d()
# fk_sim_2d()
fk_2d_withFilament()