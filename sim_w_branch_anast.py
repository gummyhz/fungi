# data managemement
import numpy as np 
import pandas as pd 

# plotting
import matplotlib as mpl 
import matplotlib.pyplot as plt 

# math methods
from scipy.integrate import solve_ivp 
from scipy.ndimage import convolve    


# units milimeters
alpha = 0.039   # branching rate
beta = 0.022    # anastomosis rate 
c = .280        # speed of traveling wavefront
D = 0.0008      # diffusion constant 
v = .240        # velocity of tips
K = .3          # tip saturation density

T = 10          # Total duration
dt = .01        # Time step
nt = int(T/dt)  # Number of time points to evaulate at

rmax = 5           # Max r value
dr = 0.1            # Spatial step length in r
nr = int(rmax/dr)   # Number of points in r
ntheta = 100        # Number of points in theta
dtheta = 2 * np.pi / ntheta # Spatial step length in theta

r, theta = np.linspace(0,rmax, nr), np.linspace(0,2*np.pi,ntheta)
R, THETA = np.meshgrid(r,theta, indexing='ij')

u = np.zeros((nr,ntheta))
rho = np.zeros((nr,ntheta))

u[:, :] = np.exp(-R**2)  # smooth Gaussian seed
rho[:, :] = np.exp(-R**2)

# set up plot
plt.ioff()
figs, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5), subplot_kw={'projection': 'polar'})
axes[0].grid(False)
axes[1].grid(False)
axes[0].tick_params(axis='y', colors='white') 
axes[1].tick_params(axis='y', colors='white') 

imu = axes[0].pcolormesh(THETA, R, u, cmap='plasma',  vmin=0, vmax=K)
imrho = axes[1].pcolormesh(THETA, R, rho, cmap='plasma',  vmin=0, vmax=1.5)
figs.colorbar(imu).set_label('Tip Density', labelpad=10)
figs.colorbar(imrho).set_label('Filament Density', labelpad=10)

plt.tight_layout()


# plot at each time step
for n in range(nt+1):
    R_nonzero = R + 1e-8  # to avoid divide-by-zero

    u_r = np.gradient(u, dr, axis=0)
    u_theta = np.gradient(u, dtheta, axis=1)
    J = v * u - D * u_r

    div_r = (1 / R_nonzero) * np.gradient(R*J, dr, axis=0)
    div_theta = (1 / R_nonzero**2) * np.gradient(D * u_theta, dtheta, axis=1)
    divergence = div_r + div_theta

    u_t = (alpha*u) - (beta*u*rho) - divergence
    rho_t = v*u

    u = u + u_t*dt
    rho = rho + rho_t*dt
    u = np.clip(u, 0, K)


    if n % 10 == 0: # plot every 10 time steps

        axes[0].collections.clear()
        axes[1].collections.clear()
        
        axes[0].pcolormesh(THETA, R, u, shading='auto', cmap='plasma', vmin=0, vmax=K)
        axes[1].pcolormesh(THETA, R, rho, shading='auto', cmap='plasma', vmin=0, vmax=1.5)
        
        axes[0].set_title(f'Tip Density at t={n*dt:.1f}h')
        axes[1].set_title(f'Filament Density at t={n*dt:.1f}h')

        figs.canvas.draw_idle()
        plt.pause(0.01)

plt.ioff()
plt.show()


## code to swap in contourf instead of pcolormesh

# axes[0].contourf(THETA, R, u, 100, cmap='plasma')
# axes[1].contourf(THETA, R, rho, 100, cmap='plasma')

# axes[0].cla()
# axes[1].cla()

# axes[0].set_title(f'u at t={n*dt:.1f}')
# axes[1].set_title(f'rho at t={n*dt:.1f}')

# axes[0].contourf(THETA, R, u, levels=100, cmap='plasma', vmin=0, vmax=K)
# axes[1].contourf(THETA, R, rho, levels=100, cmap='plasma', vmin=0, vmax=K)