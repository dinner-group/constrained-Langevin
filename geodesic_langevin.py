import jax
import jax.numpy as np
import numpy
import nonlinear_solver
from functools import partial
jax.config.update("jax_enable_x64", True)

@jax.jit
def cotangency_lhs(jac_constraint, inverse_mass):
    A = np.zeros((jac_constraint.shape[0] + jac_constraint.shape[1], jac_constraint.shape[0] + jac_constraint.shape[1]))
    A = A.at[:jac_constraint.shape[1], :jac_constraint.shape[1]].set(np.identity(jac_constraint.shape[1]))
    A = A.at[:jac_constraint.shape[1], jac_constraint.shape[1]:].set(-jac_constraint.T)
    A = A.at[jac_constraint.shape[1]:, :jac_constraint.shape[1]].set(jac_constraint@inverse_mass)
    return A

@partial(jax.jit, static_argnums=(3, 4))
def rattle_kick(position, momentum, dt, potential, constraint, inverse_mass=None, energy=None, force=None, jac_constraint_qr=None):

    if inverse_mass is None:
        inverse_mass = np.identity(momentum.size)

    if energy is None:
        energy = potential(position)

    if force is None:
        force = jax.grad(potential)(position)

    if jac_constraint_qr is None:
        jac_constraint = jax.jacfwd(constraint)(position)
        jac_constraint_qr = jax.scipy.linalg.qr(jac_constraint.T, mode="economic")

    momentum_new = momentum - dt * force
    lagrange_multiplier_new = jac_constraint_qr[0].T@momentum_new
    momentum_new = momentum_new - jac_constraint_qr[0]@lagrange_multiplier_new

    return position, momentum_new, lagrange_multiplier_new, energy, force, jac_constraint_qr

@partial(jax.jit, static_argnums=(4, 5, 7, 8))
def rattle_drift(position, momentum, lagrange_multiplier, dt, potential, constraint, inverse_mass=None, max_newton_iter=20, tol=1e-9):
    
    if inverse_mass is None:
        inverse_mass = np.identity(momentum.size)

    position_new = position + dt * inverse_mass@momentum
    position_new, success = nonlinear_solver.gauss_newton(position_new, constraint)
    jac_constraint = jax.jacfwd(constraint)(position_new)
    momentum_new = (position_new - position) / dt

    jac_constraint_qr = jax.scipy.linalg.qr(jac_constraint.T, mode="economic")
    lagrange_multiplier_new = jac_constraint_qr[0].T@momentum_new
    momentum_new = momentum_new - jac_constraint_qr[0]@lagrange_multiplier_new

    return position_new, momentum_new, lagrange_multiplier_new, jac_constraint_qr, success

@partial(jax.jit, static_argnums=(5, 6))
def rattle_noise(position, momentum, dt, friction, prng_key, potential, constraint, inverse_mass=None, jac_constraint_qr=None, temperature=1):
    
    if inverse_mass is None:
        inverse_mass = np.identity(momentum.size)

    if jac_constraint_qr is None:
        jac_constraint = jax.jacfwd(constraint)(position)
        jac_constraint_qr = jax.scipy.linalg.qr(jac_constraint.T, mode="economic")

    drag = np.exp(-friction * dt)
    noise_scale = np.sqrt(temperature * (1 - drag**2))
    
    key, subkey = jax.random.split(prng_key)
    W = jax.random.normal(key, momentum.shape)
    L = np.linalg.cholesky(inverse_mass)
    W = noise_scale * jax.scipy.linalg.solve_triangular(L, W, lower=True)

    momentum_new = drag * momentum + W
    lagrange_multiplier_new = jac_constraint_qr[0].T@momentum_new
    momentum_new = momentum_new - jac_constraint_qr[0]@lagrange_multiplier_new

    return position, momentum_new, lagrange_multiplier_new, key

@partial(jax.jit, static_argnums=(5, 6, 8, 9, 14, 15))
def gBAOAB(position, momentum, lagrange_multiplier, dt, friction, n_steps, thin, prng_key, potential, constraint, inverse_mass=None, energy=None, force=None, temperature=1, max_newton_iter=20, tol=1e-9):
    
    if energy is None:
        energy = potential(position)

    if force is None:
        force = jax.grad(potential)(position)

    if inverse_mass is None:
        inverse_mass = np.identity(momentum.size)

    jac_constraint = jax.jacfwd(constraint)(position)
    jac_constraint_qr = jax.scipy.linalg.qr(jac_constraint.T, mode="economic")

    out = np.full((n_steps // thin, position.size + momentum.size + lagrange_multiplier.size + 1 + force.size), np.nan)

    def cond(carry):
        i, position, momentum, lagrange_multiplier, energy, force, jac_constraint_qr, out, success, prng_key = carry
        return (i < n_steps) & success & (energy < 2e3)

    def loop_body(carry):
        
        i, position, momentum, lagrange_multiplier, energy, force, jac_constraint_qr, out, success, prng_key = carry

        position, momentum, _, energy, force, jac_constraint_qr,= rattle_kick(position, momentum, dt / 2, potential, constraint, inverse_mass, energy, force, jac_constraint_qr)
        position, momentum, lagrange_multiplier, jac_constraint_qr, success =  rattle_drift(position, momentum, lagrange_multiplier, dt / 2, potential, constraint, inverse_mass, max_newton_iter, tol)
        position, momentum, _, prng_key = rattle_noise(position, momentum, dt, friction, prng_key, potential, constraint, inverse_mass, jac_constraint_qr, temperature)
        position, momentum, lagrange_multiplier, jac_constraint_qr, success =  rattle_drift(position, momentum, lagrange_multiplier, dt / 2, potential, constraint, inverse_mass, max_newton_iter, tol)
        position, momentum, _, energy, force, jac_constraint_qr = rattle_kick(position, momentum, dt / 2, potential, constraint, inverse_mass, jac_constraint_qr=jac_constraint_qr)
        
        out_step = np.concatenate([position, momentum, lagrange_multiplier, np.array([energy]), force])
        out = jax.lax.dynamic_update_slice(out, np.array([out_step]), (i // thin, 0))
        return (i + 1, position, momentum, lagrange_multiplier, energy, force, jac_constraint_qr, out, success, prng_key)

    init = (0, position, momentum, lagrange_multiplier, energy, force, jac_constraint_qr, out, True, prng_key)
    i, position, momentum, lagrange_multiplier, energy, force, jac_constraint_qr, out, success, prng_key = jax.lax.while_loop(cond, loop_body, init)
    
    return out, prng_key
