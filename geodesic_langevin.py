import jax
import jax.numpy as np
import numpy
import nonlinear_solver
from functools import partial

@jax.jit
def cotangency_lhs(jac_constraint, inverse_mass):
    A = np.zeros((jac_constraint.shape[0] + jac_constraint.shape[1], jac_constraint.shape[0] + jac_constraint.shape[1]))
    A = A.at[:jac_constraint.shape[1], :jac_constraint.shape[1]].set(np.identity(jac_constraint.shape[1]))
    A = A.at[:jac_constraint.shape[1], jac_constraint.shape[1]:].set(-jac_constraint.T)
    A = A.at[jac_constraint.shape[1]:, :jac_constraint.shape[1]].set(jac_constraint@inverse_mass)
    return A

@partial(jax.jit, static_argnums=(3, 4))
def rattle_kick(position, momentum, dt, potential, constraint, inverse_mass=None):

    if inverse_mass is None:
        inverse_mass = np.identity(momentum.size)

    jac_constraint = jax.jacfwd(constraint)(position)

    A = cotangency_lhs(jac_constraint, inverse_mass)
    b = np.pad(momentum - dt * jax.grad(potential)(position), (0, jac_constraint.shape[0]))
    x = np.linalg.solve(A, b)

    momentum_new = x[:momentum.size]
    lagrange_multiplier_new = x[momentum.size:]

    return position, momentum_new, lagrange_multiplier_new

@partial(jax.jit, static_argnums=(4, 5, 7, 8))
def rattle_drift(position, momentum, lagrange_multiplier, dt, potential, constraint, inverse_mass=None, max_newton_iter=20, tol=1e-9):
    
    if inverse_mass is None:
        inverse_mass = np.identity(momentum.size)

    def drift_residual(x):
        
        position_new = x[:position.size]
        lagrange_multiplier_new = x[position.size:]
        jac_constraint = jax.jacfwd(constraint)(position_new)
        momentum_new = momentum + lagrange_multiplier_new@jac_constraint
        return np.concatenate([position_new - (position + dt * inverse_mass@momentum_new), constraint(position_new)])

    x = np.concatenate([position, lagrange_multiplier])
    x, success = nonlinear_solver.newton(x, drift_residual, max_iter=max_newton_iter)

    position_new = x[:position.size]
    lagrange_multiplier_new = x[position.size:]
    jac_constraint = jax.jacfwd(constraint)(position_new)
    momentum_new = momentum + lagrange_multiplier_new@jac_constraint

    A = cotangency_lhs(jac_constraint, inverse_mass)
    b = np.pad(momentum_new, (0, jac_constraint.shape[0]))
    x = np.linalg.solve(A, b)

    momentum_new = x[:momentum.size]

    return position_new, momentum_new, lagrange_multiplier_new, success

@partial(jax.jit, static_argnums=(5, 6))
def rattle_noise(position, momentum, dt, friction, prng_key, potential, constraint, inverse_mass=None, temperature=1):
    
    if inverse_mass is None:
        inverse_mass = np.identity(momentum.size)

    drag = np.exp(-friction * dt)
    noise_scale = np.sqrt(temperature * (1 - drag**2))

    jac_constraint = jax.jacfwd(constraint)(position)
    
    A = cotangency_lhs(jac_constraint, inverse_mass)
    
    key, subkey = jax.random.split(prng_key)
    W = jax.random.normal(key, momentum.shape)
    L = np.linalg.cholesky(inverse_mass)
    W = noise_scale * jax.scipy.linalg.solve_triangular(L, W, lower=True)

    b = np.pad(drag * momentum + W, (0, jac_constraint.shape[0]))
    x = np.linalg.solve(A, b)

    momentum_new = x[:momentum.size]
    lagrange_multiplier_new = x[momentum.size:]

    return position, momentum_new, lagrange_multiplier_new, key
