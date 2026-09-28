"""
Author: Rakesh Therala
Date: 2025-04-10
Description: This script implements the Direct Quadrature Method of Moments (DQMoM) for solving population balance equations.
---------------------------------------------------------------------------------------------
This script performs:
1. Definition of Gamma distribution (PDF)
2. Calculation of raw moments from the distribution
3. Application of Wheeler's Algorithm to compute intial abscissas and weights 
    for the Quadrature Method of Moments (QMoM)
4. Definition of moment ODEs based on a logistic growth model
5. Integration of the moment ODEs using the DQMoM method
6. Validation of the DQMoM results by comparing with analytical moments
7. Plotting the DQMoM distribution at different time steps
8. Saving the plots as PNG files
9. Printing the final moment comparison between DQMoM and analytical results
10. The script is designed to be run as a standalone program.
---------------------------------------------------------------------------------------------
This script may require modifications for specific use cases.
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import time

def gamma_function(z):
    """
    Computes the Gamma function Γ(z) using numerical integration.

    Parameters:
    z (float): Argument to the Gamma function (shape parameter)

    Returns:
    float: Value of the Gamma function for given z.
    """
    def integrand(t):
        return t**(z - 1) * np.exp(-t)

    result, _ = quad(integrand, 0, np.inf) # Integrate from 0 to infinity
    return result


# Gamma Probability Density Function (PDF)

def gamma_pdf(x):
    """
    Gamma Probability Density Function.

    Parameters:
    x (float or array): Input value(s) to evaluate the PDF.

    Returns:
    float or array: PDF evaluated at x.
    """
    z = 2         # Shape parameter
    theta = 1     # Scale parameter

    gamma_z = gamma_function(z)
    pdf = ((x**(z - 1)) * np.exp(-x / theta)) / (gamma_z * theta**z)
    return pdf

def moments(degree,z,theta):
    """
    Calculation moment of a given degree for the Gamma distribution
    """
    def integrand(x):
        return (x**degree)*((x**(z-1))*np.exp(-x/theta))/(gamma_function(z)*theta**z)
    
    moment,_ = quad(integrand, 0, np.inf)  
    return moment

# Wheeler's Algorithm for Moment Inversion

def wheelers_algorithm(moments, N):
    """
    Wheeler's Algorithm to compute abscissas (x) and weights (w)
    for the Quadrature Method of Moments (QMOM).

    Parameters:
    moments (array): Array of raw moments of length 2*N.
    N (int): Number of quadrature points (nodes).

    Returns:
    x (array): Abscissas (Nodes).
    w (array): Weights.
    """

    if len(moments) != 2*N:
        raise ValueError(f"Number of moments provided = {len(moments)}, but required = {2*N}")

    nu = moments.copy()         

    a = np.zeros(N)
    b = np.zeros(N)
    sig = np.zeros((N+1, 2*N+1))

    # Initialize sigma table with moments
    for i in range(1, 2*N+1):
        sig[1, i] = nu[i-1]

    # Initial coefficients
    a[0] = nu[1]/nu[0]  # First moment divided by the zeroth moment
    b[0] = 0.0

    # Recurrence relation
    for k in range(2, N+1):
        for l in range(k-1, 2*N-k+2):
            sig[k, l] = sig[k-1, l+1] - a[k-2]*sig[k-1, l]-b[k-2]*sig[k-2, l]
            if abs(sig[k, l]) < 1e-16:
                sig[k, l] = 1e-16  # Numerical stability fix

        a[k-1] = sig[k, k+1]/sig[k, k]-sig[k-1, k] /sig[k-1, k-1]
        b[k-1] = sig[k, k]/sig[k-1, k-1]

    # Jacobi Matrix
    J = np.diag(a)+np.diag(np.sqrt(b[1:]), 1)+np.diag(np.sqrt(b[1:]), -1)

    # Eigenvalues = Abscissas
    # Eigenvectors give weights
    eigenvalues, eigenvectors = np.linalg.eigh(J)

    x = eigenvalues                         # Abscissas
    w = eigenvectors[0, :]**2*moments[0]  # Weights

    return x, w

def analytical_moments_function(t, degree):
    """
    Computes the analytical moments of a gamma distribution at time t.

    Parameters:
    t (float): Time at which to evaluate the moments.  
    degree (int): Degree of the moment to compute.

    returns:
    float: Analytical moment of the gamma distribution at time t.
    
    """
    z = 2  # Shape parameter of the gamma distribution
    theta = 1  # Scale parameter of the gamma distribution
    M = moments(degree,z,theta)
    M0 = moments(0,z,theta)

    a = 0.5 # Growth rate
    K = 2   # Carrying capacity of the system
    
    f = (M*K) / ((K-M0)*np.exp(-a*t) + M0) # Analytical moment function

    return f

def moment_ode(y0,a, K):

    """
    Definition of moment ODEs, can be changed according to the mathematical model (Population balance equation),
    Next, formulate the moment equations from the mathematical model
    The number of moments ODEs depend upon the number of quadrature nodes (N) considered,
    If N, then, 2N-1 number of ODEs required!!

    In this, Logistic growth equation is considered and the moment equations are formulated accordingly.

    This function is not required in DQMoM implementation, but it is defined here for reference.
    The DQMoM implementation is done in the function DQMoM_linSystem.

    """
    # Unpack the moments from the initial conditions
    M0, M1, M2, M3,M4, M5 = y0  
    
    # ODEs for the moments
    dM0_dt = a*M0*(1-M0/K)
    dM1_dt = a*M1*(1-M0/K)
    dM2_dt = a*M2*(1-M0/K) 
    dM3_dt = a*M3*(1-M0/K)
    dM4_dt = a*M4*(1-M0/K)
    dM5_dt = a*M5*(1-M0/K) 
    
    return [dM0_dt, dM1_dt, dM2_dt, dM3_dt,dM4_dt, dM5_dt]

def moments_validation(x,w):
    """
    Validation by calculating moments from the quadrature nodes and weights.

    parameters: x - array of quadrature nodes
                w - array of quadrature weights
    returns: M - array of moments calculated from the nodes and weights

    """
    num_moments = 2*len(x)  # Number of moments to calculate
    M = np.zeros(num_moments)

    for i in range(num_moments):
            M[i] = np.sum(w*x**i)  # Calculate moment using quadrature nodes and weights
    return M

def quadrature_moment(nodes, weights, degree):
    """
    Computes the moment from the quadrature nodes and weights obtained from wheeler's algorithm.
    
    Parameters:
    nodes (array): Quadrature abscissas.
    weights (array): Quadrature weights.
    degree (int): Degree of the moment.
    
    Returns:
    float: Reconstructed moment.
    """
    return np.sum(weights * nodes**degree)

def plot_dqmom_distribution(x, w, t, x_range=(0, 20)):
    """
    Plot QMOM nodes/weights at time t.

    args:
    x (array): Abscissas (nodes).  
    w (array): Weights.
    t (float): Time at which to plot the distribution.
    x_range (tuple): Range for the x-axis of the plot.
    
    return: Saves the plot as a PNG file.
    """
    # Sort nodes/weights for clearer visualization
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    w_sorted = w[sorted_idx] # Normalize weights

    x_axis = np.linspace(x_range[0], x_range[1], 1000) # Range for PDF
    gammapdf = gamma_pdf(x_axis)

    plt.figure(figsize=(10, 6))
    plt.stem(x_sorted, w_sorted, linefmt='r-', markerfmt='ro', basefmt=' ')
    # Add text labels above each stem
    for xi, wi in zip(x_sorted, w_sorted):
        plt.text(xi, wi + 0.01, f"{wi:.2f}", ha='center', va='bottom', fontsize=9, color='black')
    
    plt.plot(x_axis, gammapdf, label='Gamma distribution')
    plt.xlabel("x")
    plt.ylabel("Weights")
    plt.title(f"DQMoM Approximation at t = {t:.2f} sec")
    plt.grid(True, alpha=0.3)
    plt.legend()
    filename = f"dqmom_t_{t:.2f}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

def DQMoM_linSystem(weights, alphas, source_terms):
    """
    Solve the 2N×2N DQMoM linear system in the primary variables:
    for the moment equations:
        d/dt m_k = S_k
        where S_k are the source terms.
    i = 0,1,...,N-1
    k = 0,1,...,2N-1

    Parameters:
    weights (array): Quadrature weights.
    alphas (array): Quadrature nodes*Quadrature weights.
    source_terms (array): Source terms for the moment equations.

    returns:
    tuple: Solution of the linear system w_k,alpha_k
    """
    N = len(weights)
    nodes = alphas / weights            # r_i
    A = np.zeros((2*N, 2*N))

    for k in range(2*N):
        for i in range(N):
            A[k,i]   = (1-k)*nodes[i]**k
            if k > 0:
                A[k, N+i] = k*nodes[i]**(k-1)

    sol = np.linalg.solve(A, source_terms)
    return sol[:N], sol[N:]

def DQMoM(N, t, w0, alpha0, a, K):

    dt         = t[1]-t[0]
    time_steps = len(t)

    weights_evolution      = np.zeros((time_steps, N))
    alphas_evolution       = np.zeros((time_steps, N))
    nodes_evolution        = np.zeros((time_steps, N))
    moments_evolution      = np.zeros((time_steps, 2*N))

    weights_evolution[0]     = w0
    alphas_evolution[0] = alpha0
    nodes_evolution[0]  = alpha0 / w0
    for k in range(2*N):
        moments_evolution[0,k] = np.sum(w0*nodes_evolution[0]**k)

    for i in range(1, time_steps):
        w_prev     = weights_evolution[i-1]
        alpha_prev = alphas_evolution[i-1]
        r_prev     = alpha_prev/w_prev

        S = np.zeros(2*N)
        m0 = np.sum(w_prev)
        factor = a * (1 - m0/K)

        for k in range(2*N):
            mk = np.sum(w_prev*r_prev**k)
            S[k] = factor*mk
        
        dw_dt, dalpha_dt = DQMoM_linSystem(w_prev, alpha_prev, S)

        w_new     = w_prev+dt * dw_dt
        alpha_new = alpha_prev+dt*dalpha_dt
        r_new     = alpha_new/w_new

        weights_evolution[i] = w_new
        alphas_evolution[i]  = alpha_new
        nodes_evolution[i]   = r_new
        for k in range(2*N):
            moments_evolution[i,k] = np.sum(w_new* r_new**k)

    return weights_evolution, alphas_evolution, nodes_evolution, moments_evolution

if __name__ == "__main__":

    # Parameters of logistic growth
    a = 0.5  # Growth rate
    K = 2    # Carrying capacity

    # Parameters of the initial gamma distribution
    z = 2
    theta = 1

    N = 3               # Number of quadrature points
    T_max = 50          # Simulation time
    t = np.linspace(0, T_max, 1000)

    # Initial moments from Gamma distribution
    y0 = np.array([moments(i, z, theta) for i in range(2*N)])

    # Analytical moments (pre-computed)
    y_analytical = np.vstack([analytical_moments_function(t, k) for k in range(2*N)]).T

    ############# DQMoM Implementation ######################
    # Initialize quadrature nodes & weights using Wheeler's algorithm
    initial_nodes, initial_weights = wheelers_algorithm(y0, N)
    initial_alphas = initial_weights * initial_nodes

    start_time = time.time()
    
    # Run DQMoM solver
    weights_evolution, alphas_evolution, r_evolution, moments_evolution = DQMoM(
                                    N, t, initial_weights, initial_alphas, a, K)
   
    solver_time = time.time() - start_time
    ###########################################################
    
    time_steps = len(t)

    # Print node/weight evolution and moment comparisons
    for idx in range(1, time_steps, time_steps // 5):
        current_time = t[idx]
        current_nodes = r_evolution[idx]
        current_weights = weights_evolution[idx]

        print(f"\nTime t = {current_time:.2f} sec:")
        print("  Nodes (d_i):", current_nodes)
        print("  Weights (w_i):", current_weights)
        print("---------------------------------------------------")

        # Compare moments at current time
        print("Comparison of moments:")
        header = f"{'Degree':>6}   {'Analytical':>15}   {'Quadrature':>15}"
        print(header)
        print("-" * len(header))

        for degree in range(2 * N):
            m_analytic = analytical_moments_function(current_time, degree)
            m_quadrature = np.sum(current_weights * current_nodes**degree)
            print(f"{degree:6d}   {m_analytic:15.6e}   {m_quadrature:15.6e}")

        plot_dqmom_distribution(current_nodes, current_weights, current_time)

    # Final time comparison
    print("\nFinal comparison at t = {:.2f} sec:".format(t[-1]))
    final_nodes = r_evolution[-1]
    final_weights = weights_evolution[-1]

    header = f"{'Degree':>6}   {'Analytical':>15}   {'Quadrature':>15}"
    print(header)
    print("-" * len(header))

    for degree in range(2 * N):
        m_analytic = analytical_moments_function(t[-1], degree)
        m_quadrature = np.sum(final_weights * final_nodes**degree)
        print(f"{degree:6d}   {m_analytic:15.6e}   {m_quadrature:15.6e}")

    print(f"The solver time: {solver_time:.2f} seconds")