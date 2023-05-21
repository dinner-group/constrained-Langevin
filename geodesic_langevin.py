import jax
import jax.numpy as np
import numpy
import nonlinear_solver
import linear_solver
import util
from functools import partial
jax.config.update("jax_enable_x64", True)

@jax.jit
def cotangency_proj(Jcons, inverse_mass):

    if isinstance(Jcons, util.BVPJac):

        if inverse_mass is None:
            QR = np.linalg.qr(Jcons.todense().T)
        else:
            _, R = np.linalg.qr((np.sqrt(inverse_mass) * Jcons.todense()).T)
    else:

        if inverse_mass is None:
            QR = np.linalg.qr(Jcons.T)
        elif len(inverse_mass.shape) == 1:
            QR = np.linalg.qr((np.sqrt(inverse_mass) * Jcons).T)
        else:
            sqrtMinv = jax.scipy.linalg.cholesky(inverse_mass)
            QR = np.linalg.qr((sqrtMinv@Jcons).T)

    return Jcons, QR

@jax.jit
def velocity(momentum, inverse_mass):

    if inverse_mass is None:
        v = momentum
    elif len(inverse_mass.shape) == 1:
        v = inverse_mass * momentum
    else:
        v = inverse_mass@momentum

    return v

@jax.jit
def vjp(J, v):

    if isinstance(J, util.BVPJac):
        return J.left_multiply(v)
    else:
        return v@J

@jax.jit
def jvp(J, v):

    if isinstance(J, util.BVPJac):
        return J.right_multiply(v)
    else:
        return J@v

@partial(jax.jit, static_argnums=(3, 4, 5, 11))
def rattle_kick(position, momentum, dt, potential, constraint, jac_constraint=None, inverse_mass=None, energy=None, force=None, J_and_factor=None, args=(), linsol=linear_solver.qr_lstsq_rattle):

    if energy is None:
        energy = potential(position, *args)

    if force is None:
        force = jax.jacfwd(potential)(position, *args)

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)

    if J_and_factor is None:
        Jcons = jac_constraint(position, *args)
        J_and_factor = cotangency_proj(Jcons, inverse_mass)

    momentum_new = momentum - dt * force
    momentum_new, lagrange_multiplier_new, J_and_factor = linsol(Jcons, momentum_new, J_and_factor, inverse_mass)

    return position, momentum_new, lagrange_multiplier_new, energy, force, J_and_factor, args

@partial(jax.jit, static_argnums=(4, 5, 6, 10, 11, 12, 13))
def rattle_drift(position, momentum, lagrange_multiplier, dt, potential, constraint, jac_constraint=None, inverse_mass=None, J_and_factor=None, args=(), nlsol=nonlinear_solver.newton_rattle, linsol=linear_solver.qr_lstsq_rattle, max_newton_iter=20, tol=1e-9):

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)
    
    if J_and_factor is None:
        Jcons = jac_constraint(position, *args)
        J_and_factor = cotangency_proj(Jcons, inverse_mass)

    position_new = position + dt * velocity(momentum, inverse_mass)
    position_new, args, success = nlsol(position_new, constraint, J_and_factor[0], jac_constraint, inverse_mass, max_newton_iter, tol, args=args)
    Jcons = jac_constraint(position_new, *args)
    velocity_new = (position_new - position) / dt

    if inverse_mass is None:
        momentum_new = velocity_new
    elif len(inverse_mass.shape) == 1:
        momentum_new = velocity_new / inverse_mass
    else:
        R = jax.scipy.linalg.cholesky(inverse_mass)
        momentum_new = jax.scipy.linalg.cho_solve((R, False), velocity_new)

    momentum_new, lagrange_multiplier_new, J_and_factor = linsol(Jcons, momentum_new, None, inverse_mass)

    return position_new, momentum_new, lagrange_multiplier_new, J_and_factor, args, success

@partial(jax.jit, static_argnums=(5, 6, 7, 11))
def rattle_noise(position, momentum, dt, friction, prng_key, potential, constraint, jac_constraint=None, inverse_mass=None, J_and_factor=None, args=(), linsol=linear_solver.qr_lstsq_rattle, temperature=1):

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)
    
    if J_and_factor is None:
        Jcons = jac_constraint(position, *args)
        J_and_factor = cotangency_proj(Jcons, inverse_mass)

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
    mometnum_new, lagrange_multiplier_new, J_and_factor = linsol(J_cons, momentum_new, J_and_factor, inverse_mass)

    return position, momentum_new, lagrange_multiplier_new, key, args

@partial(jax.jit, static_argnums=(5, 6, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22))
def gBAOAB(position, momentum, lagrange_multiplier, dt, friction, n_steps, thin, prng_key, potential, constraint, jac_constraint=None, inverse_mass=None, energy=None, force=None, args=(), temperature=1, A=rattle_drift, B=rattle_kick, O=rattle_noise, nlsol=nonlinear_solver.newton_rattle, linsol=linear_solver.qr_lstsq_rattle, max_newton_iter=20, tol=1e-9):

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)
    
    if energy is None:
        energy = potential(position, *args)

    if force is None:
        force = jax.jacfwd(potential)(position, *args)

    Jcons = jac_constraint(position, *args)
    J_and_factor = cotangency_proj(Jcons, inverse_mass)
    args_size = 0

    if len(args) > 0:
        args_flatten = np.concatenate(tuple(np.ravel(x) for x in args))
        args_size = args_flatten.size

    out = np.full((n_steps // thin, position.size + momentum.size + lagrange_multiplier.size + 1 + force.size + args_size), np.nan)
    key_out = np.zeros((n_steps // thin, 2), dtype=np.uint32)

    def cond(carry):
        i, position, momentum, lagrange_multiplier, energy, force, J_and_factor, args, out, success, prng_key, key_out = carry
        return (i < n_steps) & success & (energy < 2e3)

    def loop_body(carry):
        
        i, position, momentum, lagrange_multiplier, energy, force, J_and_factor, args, out, success, prng_key, key_out = carry

        position, momentum, _, energy, force, J_and_factor, args = B(position, momentum, dt / 2, potential, constraint, jac_constraint, inverse_mass, energy, force, J_and_factor, args)
        position, momentum, lagrange_multiplier, J_and_factor, args, success = A(position, momentum, lagrange_multiplier, dt / 2, potential, constraint, jac_constraint, inverse_mass, J_and_factor, args, nlsol, max_newton_iter, tol)
        position, momentum, _, prng_key, args = O(position, momentum, dt, friction, prng_key, potential, constraint, jac_constraint, inverse_mass, J_and_factor, args, temperature)
        position, momentum, lagrange_multiplier, J_and_factor, args, success = A(position, momentum, lagrange_multiplier, dt / 2, potential, constraint, jac_constraint, inverse_mass, J_and_factor, args, nlsol, max_newton_iter, tol)
        position, momentum, _, energy, force, J_and_factor, args = B(position, momentum, dt / 2, potential, constraint, jac_constraint, inverse_mass, J_and_factor=J_and_factor, args=args)
        
        out_step = np.concatenate([position, momentum, lagrange_multiplier, np.array([energy]), force])

        if(args_size > 0):
            args_flatten = np.concatenate(tuple(np.ravel(x) for x in args))
            out_step = np.concatenate([out_step, args_flatten])
        
        out = jax.lax.dynamic_update_slice(out, np.expand_dims(out_step, 0), (i // thin, 0))
        key_out = jax.lax.dynamic_update_slice(key_out, np.expand_dims(prng_key, 0), (i // thin, 0))
        return (i + 1, position, momentum, lagrange_multiplier, energy, force, J_and_factor, args, out, success, prng_key, key_out)

    init = (0, position, momentum, lagrange_multiplier, energy, force, J_and_factor, args, out, True, prng_key, key_out)
    i, position, momentum, lagrange_multiplier, energy, force, J_and_factor, args, out, success, prng_key, key_out = jax.lax.while_loop(cond, loop_body, init)
    
    return out, key_out
