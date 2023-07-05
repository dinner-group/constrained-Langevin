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
def kinetic(momentum, inverse_mass):
    
    if inverse_mass is None:
        return momentum@momentum / 2
    elif len(inverse_mass.shape) == 1:
        return momentum@(inverse_mass * momentum) / 2
    else:
        return momentum@inverse_mass@momentum / 2

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
    else:
        Jcons = J_and_factor[0]

    momentum_new = momentum - dt * force
    momentum_new, lagrange_multiplier_new, J_and_factor = linsol(Jcons, momentum_new, J_and_factor, inverse_mass)

    return position, momentum_new, lagrange_multiplier_new, energy, force, J_and_factor, args

@partial(jax.jit, static_argnums=(4, 5, 6, 10, 11, 12, 13))
def rattle_drift(position, momentum, lagrange_multiplier, dt, potential, constraint, jac_constraint=None, inverse_mass=None, J_and_factor=None, args=(), nlsol=nonlinear_solver.newton_rattle, linsol=linear_solver.qr_lstsq_rattle, max_newton_iter=20, tol=1e-9):

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)

    if J_and_factor is None:
        Jcons = jac_constraint(position, *args)
    else:
        Jcons = J_and_factor[0]
    
    position_new = position + dt * velocity(momentum, inverse_mass)
    position_new, args, success = nlsol(position_new, constraint, Jcons, jac_constraint, inverse_mass, max_newton_iter, tol, args=args)
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

@partial(jax.jit, static_argnums=(4, 5, 6, 10, 11, 12, 13))
def rattle_drift_bvp_mm(position, momentum, lagrange_multiplier, dt, potential, constraint, jac_constraint=None, inverse_mass=None, J_and_factor=None, 
                        constraint_args=(), nlsol=nonlinear_solver.newton_rattle, linsol=linear_solver.qr_lstsq_rattle, max_newton_iter=20, tol=1e-9):
    
    mesh_points = constraint_args[1]
    ode_model = constraint_args[0]
    y = position[ode_model.n_par:-1]
    y = y.reshape((ode_model.n_dim, y.size // ode_model.n_dim), order="F")
    yp = momentum[ode_model.n_par:-1]
    yp = yp.reshape((ode_model.n_dim, yp.size // ode_model.n_dim), order="F")
    mesh_new, mesh_density = util.recompute_mesh(y, mesh_points, util.gauss_points)
    y_new = util.recompute_node_y(y, mesh_points, mesh_new, util.gauss_points)
    yp_new = util.recompute_node_y(yp, mesh_points, mesh_new, util.gauss_points)
    position = position.at[ode_model.n_par:-1].set(y_new.ravel(order="F"))
    momentum = momentum.at[ode_model.n_par:-1].set(yp_new.ravel(order="F"))
    constraint_args = list(constraint_args)
    constraint_args[1] = mesh_new
    constraint_args = tuple(constraint_args)
    return rattle_drift(position, momentum, lagrange_multiplier, dt, potential, constraint, jac_constraint, inverse_mass, J_and_factor, 
                             constraint_args, nlsol, linsol, max_newton_iter, tol)

@partial(jax.jit, static_argnums=(5, 6, 7, 11))
def rattle_noise(position, momentum, dt, friction, prng_key, potential, constraint, jac_constraint=None, inverse_mass=None, J_and_factor=None, args=(), linsol=linear_solver.qr_lstsq_rattle, temperature=1):

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)
    
    if J_and_factor is None:
        Jcons = jac_constraint(position, *args)
    else:
        Jcons = J_and_factor[0]

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
    momentum_new, lagrange_multiplier_new, J_and_factor = linsol(Jcons, momentum_new, J_and_factor, inverse_mass)

    return position, momentum_new, lagrange_multiplier_new, key, args

@partial(jax.jit, static_argnums=(5, 6, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23, 24))
def gBAOAB(position, momentum, lagrange_multiplier, dt, friction, n_steps, thin, prng_key, potential, constraint, jac_constraint=None, inverse_mass=None, energy=None, force=None, args=(), temperature=1, A=rattle_drift, B=rattle_kick, O=rattle_noise, nlsol=nonlinear_solver.newton_rattle, linsol=linear_solver.qr_lstsq_rattle, max_newton_iter=20, tol=1e-9, metropolize=False, non_reversible_check=False):

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)
    
    if energy is None:
        energy = potential(position, *args)

    if force is None:
        force = jax.jacfwd(potential)(position, *args)

    Jcons = jac_constraint(position, *args)
    _, _, J_and_factor = linsol(Jcons, momentum, inverse_mass=inverse_mass)
    args_size = 0

    if len(args) > 0:
        args_flatten = np.concatenate(tuple(x.ravel() for x in args))
        args_size = args_flatten.size

    out = np.full((n_steps // thin, position.size + momentum.size + lagrange_multiplier.size + 1 + force.size + args_size), np.nan)
    key_out = np.zeros((n_steps // thin, 2), dtype=np.uint32)

    def cond(carry):
        i, position, momentum, lagrange_multiplier, energy, force, J_and_factor, args, out, success, prng_key, key_out = carry
        return (i < n_steps) & (energy < 2e3)

    def loop_body(carry):
        
        i, position_0, momentum_0, lagrange_multiplier_0, energy_0, force_0, J_and_factor_0, args_0, out, success, prng_key, key_out = carry
        accept = not metropolize
        success = True

        position, momentum, _, energy, force, J_and_factor, args = B(position_0, momentum_0, dt / 2, potential, constraint, jac_constraint, inverse_mass, energy_0, force_0, J_and_factor_0, args_0, linsol)
        position, momentum, lagrange_multiplier, J_and_factor, args, success_step = A(position, momentum, lagrange_multiplier_0, dt / 2, potential, constraint, jac_constraint, inverse_mass, J_and_factor, args, nlsol, linsol, max_newton_iter, tol)
        success = success & success_step

        if non_reversible_check:
            position_1 = position
            position_rev, _, _, _, _, success_step = A(position, -momentum, lagrange_multiplier_0, dt / 2, potential, constraint, jac_constraint, inverse_mass, J_and_factor, args, nlsol, linsol, max_newton_iter, tol)
            success = success & success_step & np.all(np.abs(position_rev - position_0) < tol)

        position, momentum, _, prng_key, args = O(position, momentum, dt, friction, prng_key, potential, constraint, jac_constraint, inverse_mass, J_and_factor, args, linsol, temperature)
        position, momentum, lagrange_multiplier, J_and_factor, args, success_step = A(position, momentum, lagrange_multiplier, dt / 2, potential, constraint, jac_constraint, inverse_mass, J_and_factor, args, nlsol, linsol, max_newton_iter, tol)
        success = success & success_step

        if non_reversible_check:
            position_rev, _, _, _, _, success_step = A(position, -momentum, lagrange_multiplier, dt / 2, potential, constraint, jac_constraint, inverse_mass, J_and_factor, args, nlsol, linsol, max_newton_iter, tol)
            success = success & success_step & np.all(np.abs(position_rev - position_1) < tol)

        position, momentum, _, energy, force, J_and_factor, args = B(position, momentum, dt / 2, potential, constraint, jac_constraint, inverse_mass, J_and_factor=J_and_factor, args=args, linsol=linsol)

        if metropolize:
            prng_key, subkey = jax.random.split(prng_key)
            u = jax.random.uniform(subkey)
            H_0 = kinetic(momentum_0, inverse_mass) + energy_0
            H = kinetic(momentum, inverse_mass) + energy
            accept = (H_0 - H > np.log(u))

        accept = accept & success

        def on_accept():
            return position, momentum, lagrange_multiplier, energy, force, J_and_factor, args
        def on_reject():
            return position_0, -momentum_0, lagrange_multiplier_0, energy_0, force_0, J_and_factor_0, args_0
        position, momentum, lagrange_multiplier, energy, force, J_and_factor, args = jax.lax.cond(accept, on_accept, on_reject)

        out_step = np.concatenate([position, momentum, lagrange_multiplier, np.array([energy]), force])

        if(args_size > 0):
            args_flatten = np.concatenate(tuple(x.ravel() for x in args))
            out_step = np.concatenate([out_step, args_flatten])
        
        out = jax.lax.dynamic_update_slice(out, np.expand_dims(out_step, 0), (i // thin, 0))
        key_out = jax.lax.dynamic_update_slice(key_out, np.expand_dims(prng_key, 0), (i // thin, 0))
        return (i + 1, position, momentum, lagrange_multiplier, energy, force, J_and_factor, args, out, success, prng_key, key_out)

    init = (0, position, momentum, lagrange_multiplier, energy, force, J_and_factor, args, out, True, prng_key, key_out)
    i, position, momentum, lagrange_multiplier, energy, force, J_and_factor, args, out, success, prng_key, key_out = jax.lax.while_loop(cond, loop_body, init)
    
    return out, key_out
