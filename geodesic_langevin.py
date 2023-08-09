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
        force = jax.jacrev(potential)(position, *args)

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)

    if J_and_factor is None:
        Jcons = jac_constraint(position, *args)
    else:
        Jcons = J_and_factor[0]

    force, lagrange_multiplier_new, J_and_factor = linsol(Jcons, force, J_and_factor, inverse_mass)
    momentum_new = momentum - dt * force

    return position, momentum_new, lagrange_multiplier_new, energy, force, J_and_factor, args

@partial(jax.jit, static_argnums=(4, 5, 6, 10, 11, 12, 13, 14))
def rattle_drift(position, momentum, lagrange_multiplier, dt, potential, constraint, jac_constraint=None, inverse_mass=None, J_and_factor=None, args=(), nlsol=nonlinear_solver.newton_rattle, linsol=linear_solver.qr_lstsq_rattle, max_newton_iter=20, tol=1e-9, reversibility_tol=None):

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)

    if J_and_factor is None:
        Jcons = jac_constraint(position, *args)
    else:
        Jcons = J_and_factor[0]
    
    position_new = position + dt * velocity(momentum, inverse_mass)
    position_new, args_new, success, _ = nlsol(position_new, constraint, jac_prev=Jcons, jac=jac_constraint, inverse_mass=inverse_mass, max_iter=max_newton_iter, tol=tol, J_and_factor=J_and_factor, args=args)
    Jcons = jac_constraint(position_new, *args_new)
    velocity_new = (position_new - position) / dt

    if inverse_mass is None:
        momentum_new = velocity_new
    elif len(inverse_mass.shape) == 1:
        momentum_new = velocity_new / inverse_mass
    else:
        R = jax.scipy.linalg.cholesky(inverse_mass)
        momentum_new = jax.scipy.linalg.cho_solve((R, False), velocity_new)

    momentum_new, lagrange_multiplier_new, J_and_factor_new = linsol(Jcons, momentum_new, None, inverse_mass)

    if reversibility_tol is not None:
        position_rev, momentum_rev, lagrange_multiplier_rev, _, args_rev, success_rev = rattle_drift(position_new, -momentum_new, lagrange_multiplier_new, dt, potential, constraint, jac_constraint, inverse_mass, J_and_factor_new, 
                                                                                                     args_new, nlsol, linsol, max_newton_iter, tol, reversibility_tol=None)
        #jax.debug.print("{}", np.max(np.abs(position - position_rev)))
        success = success & success_rev & np.all(np.abs(position - position_rev) < reversibility_tol)

    return position_new, momentum_new, lagrange_multiplier_new, J_and_factor_new, args_new, success

@partial(jax.jit, static_argnums=(4, 5, 6, 10, 11, 12, 13, 14))
def rattle_drift_bvp_mm(position, momentum, lagrange_multiplier, dt, potential, constraint, jac_constraint=None, inverse_mass=None, J_and_factor=None,
                        args=(), nlsol=nonlinear_solver.newton_rattle, linsol=linear_solver.qr_lstsq_rattle, max_newton_iter=100, tol=1e-9, reversibility_tol=None):
    
    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)
    
    if J_and_factor is None:
        Jcons = jac_constraint(position, *args)
    else:
        Jcons = J_and_factor[0]
        
    ode_model = args[0]
    position_new = position + dt * velocity(momentum, inverse_mass)
    position_new, args_new, success, n_iter = nlsol(position_new, constraint, Jcons, jac_constraint, inverse_mass, max_newton_iter, J_and_factor=J_and_factor, args=args)
    
    def cond(carry):
        position_0, position_1, mesh_0, mesh_1, args, success, n_iter = carry
        return (np.any(np.abs(position_0 - position_1) > tol) | np.any(np.abs(mesh_0 - mesh_1) > tol)) & success & (n_iter < max_newton_iter)
    
    def loop_body(carry):
        
        position_0, position_1, mesh_0, mesh_1, args_new, success, n_iter = carry
        y = position_1[ode_model.n_par:-1]
        y = y.reshape((ode_model.n_dim, y.size // ode_model.n_dim), order="F")
        mesh_2 = util.recompute_mesh(y, mesh_1)[0]
        args_new = list(args_new)
        args_new[1] = mesh_2
        args_new = tuple(args_new)
        position_2, args_new, success_inner, n_iter_inner = nlsol(position_1, constraint, Jcons, jac_constraint, inverse_mass, max_newton_iter, J_and_factor=J_and_factor, args=args_new)
        return position_1, position_2, mesh_1, mesh_2, args_new, success & success_inner, n_iter + n_iter_inner
    
    init = (position, position_new, args[1], args[1], args, True, n_iter)
    position_0, position_new, mesh_0, mesh_new, args_new, success, n_iter = jax.lax.while_loop(cond, loop_body, init)
    velocity_new = (position_new - position) / dt
    
    if inverse_mass is None:
        momentum_new = velocity_new
    elif len(inverse_mass.shape) == 1:
        momentum_new = velocity_new / inverse_mass
    else:
        R = jax.scipy.linalg.cholesky(inverse_mass)
        momentum_new = jax.scipy.linalg.cho_solve((R, False), velocity_new)
    
    Jcons = jac_constraint(position_new, *args_new)
    momentum_new, lagrange_multiplier_new, J_and_factor_new = linsol(Jcons, momentum_new, None, inverse_mass)
    
    if reversibility_tol is not None:
        position_rev, momentum_rev, lagrange_multiplier_rev, _, args_rev, success_rev = rattle_drift_bvp_mm(position_new, -momentum_new, lagrange_multiplier_new, dt, potential, constraint, jac_constraint,
                                                                                                           inverse_mass, J_and_factor_new, args_new, nlsol, linsol, max_newton_iter, reversibility_tol=None)
        success = success & success_rev & np.all(np.abs(position - position_rev) < reversibility_tol) & np.all(np.abs(args[1] - args_rev[1]) < reversibility_tol)
        
    return position_new, momentum_new, lagrange_multiplier_new, J_and_factor_new, args_new, success & np.all(position_0 - position_new < tol) & np.all(mesh_0 - mesh_new < tol)


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

    W, lagrange_multiplier_new, J_and_factor = linsol(Jcons, W, J_and_factor, inverse_mass)
    momentum_new = drag * momentum + W

    return position, momentum_new, lagrange_multiplier_new, key, args

@partial(jax.jit, static_argnums=(5, 6, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23, 24))
def gBAOAB(position, momentum, lagrange_multiplier, dt, friction, n_steps, thin, prng_key, potential, constraint, jac_constraint=None, inverse_mass=None, energy=None, force=None, args=(), temperature=1, A=rattle_drift, B=rattle_kick, O=rattle_noise, nlsol=nonlinear_solver.newton_rattle, linsol=linear_solver.qr_lstsq_rattle, max_newton_iter=20, tol=1e-9, metropolize=False, reversibility_tol=None):

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)
    
    if energy is None:
        energy = potential(position, *args)

    if force is None:
        force = jax.jacrev(potential)(position, *args)

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
        position, momentum, lagrange_multiplier, J_and_factor, args, success_step = A(position, momentum, lagrange_multiplier_0, dt / 2, potential, constraint, jac_constraint, inverse_mass, J_and_factor, args, nlsol, linsol, max_newton_iter, tol, reversibility_tol)
        success = success & success_step
        position, momentum, _, prng_key, args = O(position, momentum, dt, friction, prng_key, potential, constraint, jac_constraint, inverse_mass, J_and_factor, args, linsol, temperature)
        position, momentum, lagrange_multiplier, J_and_factor, args, success_step = A(position, momentum, lagrange_multiplier, dt / 2, potential, constraint, jac_constraint, inverse_mass, J_and_factor, args, nlsol, linsol, max_newton_iter, tol, reversibility_tol)
        success = success & success_step
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

@partial(jax.jit, static_argnums=(5, 6, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23, 24))
def gOBABO(position, momentum, lagrange_multiplier, dt, friction, n_steps, thin, prng_key, potential, constraint, jac_constraint=None, inverse_mass=None, energy=None, force=None, args=(), temperature=1, A=rattle_drift, B=rattle_kick, O=rattle_noise, nlsol=nonlinear_solver.newton_rattle, linsol=linear_solver.qr_lstsq_rattle, max_newton_iter=20, tol=1e-9, metropolize=False, reversibility_tol=None):

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)
    
    if energy is None:
        energy = potential(position, *args)

    if force is None:
        force = jax.jacrev(potential)(position, *args)

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

        position, momentum, _, prng_key, args = O(position_0, momentum_0, dt / 2, friction, prng_key, potential, constraint, jac_constraint, inverse_mass, J_and_factor_0, args_0, linsol, temperature)
        position, momentum, _, energy, force, J_and_factor, args = B(position, momentum, dt / 2, potential, constraint, jac_constraint, inverse_mass, energy_0, force_0, J_and_factor_0, args, linsol)
        position, momentum, lagrange_multiplier, J_and_factor, args, success_step = A(position, momentum, lagrange_multiplier_0, dt, potential, constraint, jac_constraint, inverse_mass, J_and_factor, args, nlsol, linsol, max_newton_iter, tol, reversibility_tol)
        success = success & success_step
        position, momentum, _, energy, force, J_and_factor, args = B(position, momentum, dt / 2, potential, constraint, jac_constraint, inverse_mass, J_and_factor=J_and_factor, args=args, linsol=linsol)
        position, momentum, _, prng_key, args = O(position, momentum, dt / 2, friction, prng_key, potential, constraint, jac_constraint, inverse_mass, J_and_factor, args, linsol, temperature)

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
