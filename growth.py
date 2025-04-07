import numpy as np 
import pandas as pd 

from scipy.integrate import solve_ivp 

import matplotlib as mpl 
import matplotlib.pyplot as plt 


# Fisher Kolmogorov model
# maybe numpy or scipy functions would be more efficient or accurate
# here I will do everything by hand for clarity 

### Parameters:
# u: vector of population density at each 
# D: diffusion constant
# r: growth rate
# K: carrying capacity
def fk(t, u, D, r, K):
    N = len(u)
    u_t = np.zeros(N) # du/dt

    for i in range(0, N-1):
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

# Simulate Fisher Kolmorogov Model
def fk_sim():
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

fk_sim()