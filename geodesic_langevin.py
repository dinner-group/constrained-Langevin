import jax
import jax.numpy as np
import numpy
import nonlinear_solver
import util
from functools import partial
jax.config.update("jax_enable_x64", True)

@jax.jit
def cotangency_proj(Jcons, inverse_mass):

    if isinstance(Jcons, util.BVPJac):

        if inverse_mass is None:
            _, R = np.linalg.qr(Jcons.todense().T)
        else:
            _, R = np.linalg.qr((np.sqrt(inverse_mass) * Jcons.todense()).T)
    else:

        if inverse_mass is None:
            _, R = np.linalg.qr(Jcons.T)
        elif len(inverse_mass.shape) == 1:
            _, R = np.linalg.qr((np.sqrt(inverse_mass) * Jcons).T)
        else:
            R = jax.scipy.linalg.cholesky(Jcons@inverse_mass@Jcons.T)

    return Jcons, R

@jax.jit
def velocity(momentum, inverse_mass):

    if inverse_mass is None:
        v = momentum
    elif len(inverse_mass.shape) == 1:
        v = inverse_mass * momentum
    else:
        v = inverse_mass@momentum

    return v

@partial(jax.jit, static_argnums=(3, 4, 5))
def rattle_kick(position, momentum, dt, potential, constraint, jac_constraint=None, inverse_mass=None, energy=None, force=None, proj=None):

    if energy is None:
        energy = potential(position)

    if force is None:
        force = jax.grad(potential)(position)

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)

    if proj is None:
        Jcons = jac_constraint(position)
        proj = cotangency_proj(Jcons, inverse_mass)

    momentum_new = momentum - dt * force
    lagrange_multiplier_new = jax.scipy.linalg.cho_solve((proj[1], False), proj[0]@velocity(momentum_new, inverse_mass))
    momentum_new = momentum_new - proj[0].T@lagrange_multiplier_new

    return position, momentum_new, lagrange_multiplier_new, energy, force, proj

@partial(jax.jit, static_argnums=(4, 5, 6, 9, 10, 11))
def rattle_drift(position, momentum, lagrange_multiplier, dt, potential, constraint, jac_constraint=None, inverse_mass=None, proj=None, nlsol=nonlinear_solver.newton_rattle, max_newton_iter=20, tol=1e-9):

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)
    
    if proj is None:
        Jcons = jac_constraint(position)
        proj = cotangency_proj(Jcons, inverse_mass)

    if isinstance(Jcons, util.BVPJac):
        if inverse_mass is None:
            jac_prevM = proj[0]
        else:
            jac_prevM = proj[0].right_multiply_diag(inverse_mass)
    else:
        if inverse_mass is None:
            jac_prevM = proj[0]
        elif len(inverse_mass.shape) == 1:
            jac_prevM = proj[0] * inverse_mass
        else:
            jac_prevM = proj[0]@inverse_mass

    position_new = position + dt * velocity(momentum, inverse_mass)
    position_new, success = nlsol(position_new, constraint, jac_prevM)
    Jcons = jac_constraint(position_new)
    momentum_new = (position_new - position) / dt

    proj = cotangency_proj(Jcons, inverse_mass)
    lagrange_multiplier_new = jax.scipy.linalg.cho_solve((proj[1], False), proj[0]@velocity(momentum_new, inverse_mass))
    momentum_new = momentum_new - proj[0].T@lagrange_multiplier_new

    return position_new, momentum_new, lagrange_multiplier_new, proj, success

@partial(jax.jit, static_argnums=(5, 6, 7))
def rattle_noise(position, momentum, dt, friction, prng_key, potential, constraint, jac_constraint=None, inverse_mass=None, proj=None, temperature=1):

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)
    
    if proj is None:
        Jcons = jac_constraint(position)
        proj = cotangency_proj(Jcons, inverse_mass)

    drag = np.exp(-friction * dt)
    noise_scale = np.sqrt(temperature * (1 - drag**2))
    
    key, subkey = jax.random.split(prng_key)
    W = jax.random.normal(key, momentum.shape)

    if inverse_mass is None:
        W = noise_scale * W
    elif len(inverse_mass.shape) == 1:
        W = noise_scale * W / np.sqrt(inverse_mass)
    else:
        R = jax.scipy.linalg.cholesky(inverse_mass)
        W = noise_scale * jax.scipy.linalg.solve_triangular(R, W, lower=False)

    momentum_new = drag * momentum + W
    lagrange_multiplier_new = jax.scipy.linalg.cho_solve((proj[1], False), proj[0]@velocity(momentum_new, inverse_mass))
    momentum_new = momentum_new - proj[0].T@lagrange_multiplier_new

    return position, momentum_new, lagrange_multiplier_new, key

@partial(jax.jit, static_argnums=(5, 6, 8, 9, 10, 15, 16, 17))
def gBAOAB(position, momentum, lagrange_multiplier, dt, friction, n_steps, thin, prng_key, potential, constraint, jac_constraint=None, inverse_mass=None, energy=None, force=None, nlsol=nonlinear_solver.newton_rattle, temperature=1, max_newton_iter=20, tol=1e-9):

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)
    
    if energy is None:
        energy = potential(position)

    if force is None:
        force = jax.grad(potential)(position)

    Jcons = jac_constraint(position)
    proj = cotangency_proj(Jcons, inverse_mass)

    out = np.full((n_steps // thin, position.size + momentum.size + lagrange_multiplier.size + 1 + force.size), np.nan)

    def cond(carry):
        i, position, momentum, lagrange_multiplier, energy, force, proj, out, success, prng_key = carry
        return (i < n_steps) & success & (energy < 2e3)

    def loop_body(carry):
        
        i, position, momentum, lagrange_multiplier, energy, force, proj, out, success, prng_key = carry

        position, momentum, _, energy, force, proj,= rattle_kick(position, momentum, dt / 2, potential, constraint, jac_constraint, inverse_mass, energy, force, proj)
        position, momentum, lagrange_multiplier, proj, success =  rattle_drift(position, momentum, lagrange_multiplier, dt / 2, potential, constraint, jac_constraint, inverse_mass, proj, nlsol, max_newton_iter, tol)
        position, momentum, _, prng_key = rattle_noise(position, momentum, dt, friction, prng_key, potential, constraint, jac_constraint, inverse_mass, proj, temperature)
        position, momentum, lagrange_multiplier, proj, success =  rattle_drift(position, momentum, lagrange_multiplier, dt / 2, potential, constraint, jac_constraint, inverse_mass, proj, nlsol, max_newton_iter, tol)
        position, momentum, _, energy, force, proj = rattle_kick(position, momentum, dt / 2, potential, constraint, jac_constraint, inverse_mass, proj=proj)
        
        out_step = np.concatenate([position, momentum, lagrange_multiplier, np.array([energy]), force])
        out = jax.lax.dynamic_update_slice(out, np.array([out_step]), (i // thin, 0))
        return (i + 1, position, momentum, lagrange_multiplier, energy, force, proj, out, success, prng_key)

    init = (0, position, momentum, lagrange_multiplier, energy, force, proj, out, True, prng_key)
    i, position, momentum, lagrange_multiplier, energy, force, proj, out, success, prng_key = jax.lax.while_loop(cond, loop_body, init)
    
    return out, prng_key
