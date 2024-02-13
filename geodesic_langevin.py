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
def kinetic(momentum, inverse_mass=None, *args, **kwargs):
    
    if inverse_mass is None:
        return momentum@momentum / 2
    elif len(inverse_mass.shape) == 1:
        return momentum@(inverse_mass * momentum) / 2
    else:
        return momentum@inverse_mass@momentum / 2

@partial(jax.jit, static_argnames=("potential", "constraint", "jac_constraint", "linsol"))
def rattle_kick(position, momentum, energy=None, force=None, dt=None, potential=None, constraint=None, jac_constraint=None, J_and_factor=None, linsol=linear_solver.qr_ortho_proj, *args, **kwargs):

    if energy is None:
        energy = potential(position, *args, **kwargs)

    if force is None:
        force = jax.jacrev(potential)(position, *args, **kwargs)

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)

    if J_and_factor is None:
        Jcons = jac_constraint(position, *args, **kwargs)
    else:
        Jcons = J_and_factor[0]

    force, lagrange_multiplier_new, J_and_factor = linsol(Jcons, force, J_and_factor)
    momentum_new = momentum - dt * force

    return position, momentum_new, lagrange_multiplier_new, energy, force, J_and_factor

@partial(jax.jit, static_argnames=("potential", "constraint", "jac_constraint", "nlsol", "linsol"))
def rattle_drift(position, momentum, lagrange_multiplier, dt, potential, constraint, jac_constraint=None, J_and_factor=None, linsol=linear_solver.qr_ortho_proj, nlsol=nonlinear_solver.newton_rattle, max_newton_iter=20, constraint_tol=1e-9, reversibility_tol=None, *args, inverse_mass=None, **kwargs):

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)

    if J_and_factor is None:
        Jcons = jac_constraint(position, *args, **kwargs)
    else:
        Jcons = J_and_factor[0]
    
    position_new = position + dt * velocity(momentum, inverse_mass)
    position_new, success, n_iter = nlsol(position_new, constraint, jac_prev=Jcons, jac=jac_constraint, max_iter=max_newton_iter, tol=constraint_tol, J_and_factor=J_and_factor, *args, **kwargs)
    Jcons = jac_constraint(position_new, *args, **kwargs)
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
        position_rev, momentum_rev, lagrange_multiplier_rev, _, success_rev, n_iter_rev = rattle_drift(position_new, -momentum_new, lagrange_multiplier_new, dt, potential, constraint, jac_constraint, J_and_factor_new, 
                                                                                           linsol, nlsol, max_newton_iter, constraint_tol, reversibility_tol=None, *args, inverse_mass=inverse_mass, **kwargs)
        success = success & success_rev & np.all(np.abs(position - position_rev) < reversibility_tol)

    return position_new, momentum_new, lagrange_multiplier_new, J_and_factor_new, success, n_iter

@partial(jax.jit, static_argnums=(4, 5, 6, 10, 11, 12, 13, 14))
def rattle_drift_bvp_mm(position, momentum, lagrange_multiplier, dt, potential, constraint, jac_constraint=None, inverse_mass=None, J_and_factor=None,
                        args=(), nlsol=nonlinear_solver.newton_rattle, linsol=linear_solver.qr_ortho_proj, max_newton_iter=100, constraint_tol=1e-9, reversibility_tol=None):
    
    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)
    
    if J_and_factor is None:
        Jcons = jac_constraint(position, *args)
    else:
        Jcons = J_and_factor[0]
        
    ode_model = args[0]
    position_new = position + dt * velocity(momentum, inverse_mass)
    position_new, success, n_iter = nlsol(position_new, constraint, Jcons, jac_constraint, inverse_mass, max_newton_iter, J_and_factor=J_and_factor, args=args)
    
    def cond(carry):
        position_0, position_1, mesh_0, mesh_1, args, success, n_iter = carry
        return (np.any(np.abs(position_0 - position_1) > constraint_tol) | np.any(np.abs(mesh_0 - mesh_1) > constraint_tol)) & success & (n_iter < max_newton_iter)
    
    def loop_body(carry):
        
        position_0, position_1, mesh_0, mesh_1, args_new, success, n_iter = carry
        y = position_1[ode_model.n_par:-1]
        y = y.reshape((ode_model.n_dim, y.size // ode_model.n_dim), order="F")
        mesh_2 = util.recompute_mesh(y, mesh_1)[0]
        args_new = list(args_new)
        args_new[1] = mesh_2
        args_new = tuple(args_new)
        position_2, success_inner, n_iter_inner = nlsol(position_1, constraint, Jcons, jac_constraint, inverse_mass, max_newton_iter, J_and_factor=J_and_factor, args=args_new)
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
        
    return position_new, momentum_new, lagrange_multiplier_new, J_and_factor_new, args_new, success & np.all(position_0 - position_new < constraint_tol) & np.all(mesh_0 - mesh_new < constraint_tol)

@partial(jax.jit, static_argnames=("constraint", "jac_constraint", "linsol"))
def rattle_noise(position, momentum, prng_key, dt, friction, constraint, jac_constraint=None, J_and_factor=None, linsol=linear_solver.qr_ortho_proj, *args, temperature=1, inverse_mass=None, **kwargs):

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)
    
    if J_and_factor is None:
        Jcons = jac_constraint(position, *args, **kwargs)
    else:
        Jcons = J_and_factor[0]

    drag = np.exp(-friction * dt)
    noise_scale = np.sqrt(temperature * (1 - drag**2))
    
    prng_key, subkey = jax.random.split(prng_key)
    W = jax.random.normal(subkey, momentum.shape)

    if inverse_mass is None:
        W = noise_scale * W
    elif len(inverse_mass.shape) == 1:
        W = noise_scale * W / np.sqrt(inverse_mass)
    else:
        R = jax.scipy.linalg.cholesky(inverse_mass)
        W = noise_scale * jax.scipy.linalg.solve_triangular(R, W, lower=False)

    W, lagrange_multiplier_new, J_and_factor = linsol(Jcons, W, J_and_factor, inverse_mass)
    momentum_new = drag * momentum + W

    return position, momentum_new, lagrange_multiplier_new, J_and_factor, prng_key

@jax.jit
def position_update(q, p, dt, *args, sqrtHinv=None, **kwargs):

    if sqrtHinv is None:
        q = q + dt * p
    else:
        q = q + dt * sqrtHinv@p
    return q

@partial(jax.jit, static_argnames=("potential",))
def momentum_update(q, p, energy=None, force=None, dt=1e-2, potential=lambda q:0., *args, sqrtHinv=None, **kwargs):
        
    if energy is None:
        energy = potential(q, *args, **kwargs)
    if force is None:
        force = -jax.jacrev(potential)(q, *args, **kwargs)
    if sqrtHinv is None:
        p = p + dt * force
    else:
        p = p + dt * sqrtHinv.T@force
    return p, energy, force

@jax.jit
def momentum_noise(p, dt, prng_key, friction=0., temperature=1, *args, **kwargs):
    
    a = np.exp(-friction * dt)
    prng_key, subkey = jax.random.split(prng_key)
    return a * p + np.sqrt(temperature * (1 - a**2)) * jax.random.normal(subkey, p.shape), prng_key

@partial(jax.jit, static_argnames=("potential", "metropolize"))
def BAOAB(q, p, energy=None, force=None, prng_key=jax.random.PRNGKey(0), dt=1e-2, friction=0., temperature=1, potential=lambda q:0., *args, metropolize=False, **kwargs):

    p_new, energy, force = momentum_update(q, p, energy, force, dt / 2, potential, *args, **kwargs)
    q_new = position_update(q, p_new, dt / 2, *args, **kwargs)
    p_new, prng_key = momentum_noise(p_new, dt, prng_key, friction, temperature, *args, **kwargs)
    q_new = position_update(q_new, p_new, dt / 2, *args, **kwargs)
    p_new, energy_new, force_new = momentum_update(q_new, p_new, dt=dt / 2, potential=potential, *args, **kwargs)
    accept = True

    if metropolize:
        prng_key, subkey = jax.random.split(prng_key)
        u = jax.random.uniform(subkey)
        H_0 = p@p / 2 + energy
        H = p_new@p_new / 2 + energy_new
        accept = (H_0 - H > np.log(u))

    vars_to_save = jax.lax.cond(accept, lambda:(q_new, p_new, energy_new, force_new, prng_key), lambda:(q, -p, energy, force, prng_key))

    return vars_to_save, (), accept

@partial(jax.jit, static_argnames=("potential", "constraint", "jac_constraint", "nlsol", "linsol", "metropolize"))
def gBAOAB(position, momentum, lagrange_multiplier, energy=None, force=None, prng_key=None, J_and_factor=None, dt=None, friction=1, potential=None, constraint=None, jac_constraint=None, 
             linsol=linear_solver.qr_ortho_proj, nlsol=nonlinear_solver.newton_rattle, max_newton_iter=20, constraint_tol=1e-9, *args, metropolize=False, reversibility_tol=None, inverse_mass=None, **kwargs):

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)
    if energy is None:
        energy = potential(position, *args, **kwargs)
    if force is None:
        force = jax.jacrev(potential)(position, *args, **kwargs)
    if J_and_factor is None:
        Jcons = jac_constraint(position, *args, **kwargs)
        _, _, J_and_factor = linsol(Jcons, momentum, inverse_mass=inverse_mass)

    prng_key = prng_key.view(jax.random.PRNGKey(0).dtype)
    accept = True
    position_new, momentum_new, lagrange_multiplier_new, energy_new, force_new, J_and_factor_new\
        = rattle_kick(position, momentum, energy, force, dt / 2, potential, constraint, jac_constraint, J_and_factor_new, linsol, *args, **kwargs)
    position_new, momentum_new, lagrange_multiplier_new, J_and_factor_new, success, n_iter\
        = rattle_drift(position_new, momentum_new, dt / 2, potential, constraint, jac_constraint, J_and_factor_new, linsol, nlsol, max_newton_iter, constraint_tol, reversibility_tol, *args, **kwargs)
    position_new, momentum_new, lagrange_multiplier_new, J_and_factor_new, prng_key = rattle_noise(position, momentum, prng_key, dt, friction, constraint, jac_constraint, J_and_factor, linsol, *args, **kwargs)
    position_new, momentum_new, lagrange_multiplier_new, J_and_factor_new, success, n_iter\
        = rattle_drift(position_new, momentum_new, dt / 2, potential, constraint, jac_constraint, J_and_factor_new, linsol, nlsol, max_newton_iter, constraint_tol, reversibility_tol, *args, **kwargs)
    position_new, momentum_new, lagrange_multiplier_new, energy_new, force_new, J_and_factor_new\
        = rattle_kick(position_new, momentum_new, None, None, dt / 2, potential, constraint, jac_constraint, J_and_factor_new, linsol, *args, **kwargs)

    if metropolize:
        prng_key, subkey = jax.random.split(prng_key)
        u = jax.random.uniform(subkey)
        H_0 = kinetic(momentum, inverse_mass) + energy
        H = kinetic(momentum_new, inverse_mass) + energy_new
        accept = (H_0 - H > np.log(u))

    accept = accept & success
    vars_to_save, vars_to_discard = jax.lax.cond(accept, 
                                                 lambda:((position_new, momentum_new, lagrange_multiplier_new, energy_new, force_new, prng_key.view(np.float64)), (J_and_factor_new,)), 
                                                 lambda:((position, -momentum, lagrange_multiplier, energy, force, prng_key.view(np.float64)), (J_and_factor,)))

    return vars_to_save, vars_to_discard, accept

@partial(jax.jit, static_argnames=("potential", "constraint", "jac_constraint", "nlsol", "linsol", "metropolize"))
def gOBABO(position, momentum, lagrange_multiplier, energy=None, force=None, prng_key=None, J_and_factor=None, dt=None, friction=1, potential=None, constraint=None, jac_constraint=None, 
             linsol=linear_solver.qr_ortho_proj, nlsol=nonlinear_solver.newton_rattle, max_newton_iter=20, constraint_tol=1e-9, *args, metropolize=False, reversibility_tol=None, inverse_mass=None, **kwargs):

    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)
    if energy is None:
        energy = potential(position, *args, **kwargs)
    if force is None:
        force = jax.jacrev(potential)(position, *args, **kwargs)
    if J_and_factor is None:
        Jcons = jac_constraint(position, *args, **kwargs)
        _, _, J_and_factor = linsol(Jcons, momentum, inverse_mass=inverse_mass)

    prng_key = prng_key.view(jax.random.PRNGKey(0).dtype)
    position_new, momentum_new, lagrange_multiplier_new, J_and_factor_new, prng_key = rattle_noise(position, momentum, prng_key, dt / 2, friction, constraint, jac_constraint, J_and_factor, linsol, *args, **kwargs)
    position_new, momentum_new, lagrange_multiplier_new, energy_new, force_new, J_and_factor_new\
        = rattle_kick(position_new, momentum_new, energy, force, dt / 2, potential, constraint, jac_constraint, J_and_factor_new, linsol, *args, **kwargs)
    position_new, momentum_new, lagrange_multiplier_new, J_and_factor_new, success, n_iter\
        = rattle_drift(position_new, momentum_new, lagrange_multiplier_new, dt, potential, constraint, jac_constraint, J_and_factor_new, linsol, nlsol, max_newton_iter, constraint_tol, reversibility_tol, *args, **kwargs)
    position_new, momentum_new, lagrange_multiplier_new, energy_new, force_new, J_and_factor_new\
        = rattle_kick(position_new, momentum_new, None, None, dt / 2, potential, constraint, jac_constraint, J_and_factor_new, linsol, *args, **kwargs)
    position_new, momentum_new, lagrange_multiplier_new, J_and_factor_new, prng_key = rattle_noise(position_new, momentum_new, prng_key, dt / 2, friction, constraint, jac_constraint, J_and_factor_new, linsol, *args, **kwargs)
    accept = success

    if metropolize:
        prng_key, subkey = jax.random.split(prng_key)
        u = jax.random.uniform(subkey)
        H_0 = kinetic(momentum, inverse_mass) + energy
        H = kinetic(momentum_new, inverse_mass) + energy_new
        accept = (H_0 - H > np.log(u))

    accept = accept & success
    vars_to_save, vars_to_discard = jax.lax.cond(accept, 
                                                 lambda:((position_new, momentum_new, lagrange_multiplier_new, energy_new, force_new, prng_key.view(np.float64)), (J_and_factor_new,)), 
                                                 lambda:((position, -momentum, lagrange_multiplier, energy, force, prng_key.view(np.float64)), (J_and_factor,)))

    return vars_to_save, vars_to_discard, accept

@partial(jax.jit, static_argnames=("potential", "constraint", "jac_constraint", "nlsol", "linsol", "metropolize"))
def gEuler_Maruyama(position, momentum, lagrange_multiplier, energy=None, force=None, prng_key=None, J_and_factor=None, dt=None, potential=None, constraint=None, jac_constraint=None, linsol=linear_solver.qr_ortho_proj, 
                      nlsol=nonlinear_solver.newton_rattle, max_newton_iter=20, constraint_tol=1e-9, *args, metropolize=False, reversibility_tol=None, inverse_mass=None, **kwargs):
    
    if jac_constraint is None:
        jac_constraint = jax.jacfwd(constraint)
    if energy is None:
        energy = potential(position, *args, **kwargs)
    if force is None:
        force = jax.jacrev(potential)(position, *args, **kwargs)
    if J_and_factor is None:
        Jcons = jac_constraint(position, *args, **kwargs)
        _, _, J_and_factor = linsol(Jcons, momentum, inverse_mass=inverse_mass)

    prng_key = prng_key.view(jax.random.PRNGKey(0).dtype)
    prng_key, subkey = jax.random.split(prng_key)
    momentum_new = jax.random.normal(subkey, momentum.shape) - dt * force
    momentum_new, lagrange_multiplier_new, J_and_factor_new = linsol(J_and_factor[0], momentum_new, J_and_factor)
    position_new, momentum_new, lagrange_multiplier_new, J_and_factor_new, success, n_iter\
        = rattle_drift(position, momentum_new, lagrange_multiplier_new, dt, potential, constraint, jac_constraint, J_and_factor_new, linsol, nlsol, max_newton_iter, constraint_tol, reversibility_tol, *args, **kwargs)
    force_new = jax.jacrev(potential)(position_new, *args, **kwargs)
    energy_new = potential(position_new, *args, **kwargs)
    accept = success

    if metropolize:
        prng_key, subkey = jax.random.split(prng_key)
        u = jax.random.uniform(subkey)
        H_0 = kinetic(momentum, inverse_mass) + energy
        H = kinetic(momentum_new, inverse_mass) + energy_new
        accept = (H_0 - H > np.log(u))

    accept = accept & success
    vars_to_save, vars_to_discard = jax.lax.cond(accept, 
                                                 lambda:((position_new, momentum_new, lagrange_multiplier_new, energy_new, force_new, prng_key.view(np.float64)), (J_and_factor_new,)), 
                                                 lambda:((position, -momentum, lagrange_multiplier, energy, force, prng_key.view(np.float64)), (J_and_factor,)))

    return vars_to_save, vars_to_discard, accept


@partial(jax.jit, static_argnames=("n_steps", "thin", "potential", "constraint", "jac_constraint", "stepsize_monitor", "stepper", "nlsol", "linsol", "metropolize", "print_acceptance", "n_mesh_intervals", "n_smooth", "phase_condition"))
def sample(dynamic_vars, dt, n_steps, potential, stepper, *args, thin=1, print_acceptance=False, **kwargs):

    def loop_body(i, carry):
        vars_to_save, vars_to_discard, n_accept, out = carry
        vars_to_save, vars_to_discard, accept = stepper(*vars_to_save, *vars_to_discard, dt=dt, potential=potential, *args, **kwargs)
        vars_to_save_flatten = np.concatenate([x.ravel() for x in jax.tree_util.tree_flatten(vars_to_save)[0]])
        out = jax.lax.dynamic_update_slice(out, np.expand_dims(vars_to_save_flatten, 0), (i // thin, 0))
        return vars_to_save, vars_to_discard, n_accept + accept, out

    vars_to_save, vars_to_discard, accept = stepper(*dynamic_vars, dt=dt, potential=potential, *args, **kwargs)
    vars_to_save_flatten = np.concatenate([x.ravel() for x in jax.tree_util.tree_flatten(vars_to_save)[0]])
    out = np.full((n_steps // thin, vars_to_save_flatten.size), np.nan)
    out = out.at[0].set(vars_to_save_flatten)
    n_accept = 0 + accept
    init_val = (vars_to_save, vars_to_discard, n_accept, out)
    n_accept, out = jax.lax.fori_loop(1, n_steps, loop_body, init_val)[-2:]

    if print_acceptance:
        jax.debug.print("Acceptance: {}", n_accept / n_steps)

    return out
