# Particle Transport: Moment Methods for Population Balance Equations

Python implementations of the Quadrature Method of Moments (QMoM) and the Direct
Quadrature Method of Moments (DQMoM) for population balance equations, applied
to a logistic growth benchmark with an analytical solution.

## The Problem

A population balance equation tracks the evolution of a number density
function n(x, t), where x is an internal coordinate such as particle size.
Instead of discretising the distribution, moment methods track a small set of
its moments

```
m_k(t) = integral of x^k n(x, t) dx,   k = 0, 1, ..., 2N-1
```

and close the system by approximating the distribution as a sum of N weighted
Dirac deltas (the quadrature approximation). This reduces a PDE in the
internal coordinate to a small ODE system in time.

## The Two Methods

| Method | Script | How it works |
|---|---|---|
| QMoM | `qmom.py` | Integrates the moment ODEs, then inverts the moments into nodes and weights at every time step using Wheeler's algorithm (Jacobi matrix eigenvalue problem) |
| DQMoM | `dqmom.py` | Evolves the weights and the products alpha_i = w_i * x_i directly, by solving a 2N x 2N linear system per step. No moment inversion during the evolution |

The logistic growth benchmark starts from a Gamma distribution
(shape 2, scale 1), so every analytical moment is known in closed form and both
methods can be validated point by point in time.

## Validation

Each script prints a table comparing analytical and quadrature moments at
several time steps and at the final time, and saves plots of the quadrature
nodes and weights against the initial Gamma density. Solver runtime is
reported for a rough cost comparison between the two methods.

## Usage

```bash
pip install numpy scipy matplotlib

python qmom.py     # QMoM with Wheeler inversion at every step
python dqmom.py    # DQMoM with direct evolution of nodes and weights
```

Default settings: N = 3 quadrature nodes (6 moments), logistic growth rate
a = 0.5, carrying capacity K = 2, t in [0, 50] s with 1000 explicit Euler steps.
All parameters are defined at the top of the `__main__` block.

Output: moment comparison tables in the console and PNG plots of the
quadrature approximation at sampled time steps.

## Notes

- Time integration is explicit Euler. The interesting numerical content here is
  the moment closure and inversion, not the time stepper, so the simple
  integrator is deliberate.
- The moment ODE source terms are written for logistic growth. Other growth,
  breakage, or aggregation kernels can be dropped into the source term
  functions without changing the method machinery.
- Wheeler's algorithm uses a small floor on the sigma table entries for
  numerical stability near degenerate moment sets.

## References

- McGraw, R. (1997). Description of Aerosol Dynamics by the Quadrature Method
  of Moments. Aerosol Science and Technology, 27(2).
- Marchisio, D. L., Fox, R. O. (2005). Solution of Population Balance Equations
  Using the Direct Quadrature Method of Moments. Journal of Aerosol Science,
  36(1).
- Wheeler, J. C. (1974). Modified Moments and Gaussian Quadratures. Rocky
  Mountain Journal of Mathematics, 4(2).