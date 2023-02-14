from mpi4py import MPI
import numpy
import diffrax
import jax
import jax.numpy as np
import mpi4jax
import scipy.sparse
from model import KaiODE
from collocation import colloc
import continuation
from functools import partial
import os
import time
jax.config.update("jax_enable_x64", True)

path = os.path.dirname(__file__)
K = np.array(numpy.loadtxt(path + "/K.txt", dtype=numpy.int64))
S = np.array(numpy.loadtxt(path + "/S.txt", dtype=numpy.int64))

@jax.jit
def E_floquet(y0, period, log_rc, a0=0.6, c0=3.5, floquet_multiplier_threshold=0.8):

    E = 0

    M = continuation.compute_monodromy_1(y0, period, np.exp(log_rc), a0, c0)
    floquet_multipliers = np.linalg.eigvals(M)
    abs_multipliers = np.abs(floquet_multipliers)
    order = np.argsort(np.abs(abs_multipliers - 1))
    E += np.where(abs_multipliers[order][1:] > floquet_multiplier_threshold, 100 * (abs_multipliers[order][1:] - floquet_multiplier_threshold)**2, 0).sum()

    return E

@jax.jit
def grad_floquet(y0, period, log_rc, a0=0.6, c0=3.5, floquet_multiplier_threshold=0.8):
    return jax.jacfwd(E_floquet, 2)(y0, period, log_rc, a0, c0, floquet_multiplier_threshold)

def termination_condition(colloc_solver):
    return colloc_solver.args[0][-1] < 0

@jax.jit
def E_arclength(y, min_arclength=0.3):

    arclength = np.linalg.norm(y[:, 1:] - y[:, :-1], axis=0).sum()
    return np.where(arclength > min_arclength, 0, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4)

@jax.jit 
def grad_arclength(y, min_arclength=0.3):
    return jax.jacrev(E_arclength)(y, min_arclength)

@jax.jit
def E_bounds(position, bounds):
    
    E = 0
    E += np.where(position < bounds[:, 0], 100 * (position - bounds[:, 0])**2, 0).sum()
    E += np.where(position > bounds[:, 1], 100 * (position - bounds[:, 1])**2, 0).sum()

    return E

@jax.jit
def grad_bounds(position, bounds):
    return jax.jacrev(E_bounds)(position, bounds)

@jax.jit
def smooth_max(x, beta=2):
    return np.sum(x * np.exp(beta * x)) / np.exp(beta * x).sum()

grad_smooth_max = jax.jit(jax.jacfwd(smooth_max))

def compute_energy_and_force_kai(position, momentum, colloc_solver, bounds, floquet_multiplier_threshold=0.8):

    y_prev = colloc_solver.y
    p_prev = colloc_solver.p
    args_prev = colloc_solver.args

    E = 0
    F = np.zeros(position.shape)

    rc_direction = position - np.log(colloc_solver.args[5].reaction_consts)
    natural_direction = np.zeros(colloc_solver.n).at[-1].set(1)
    tangent_direction = colloc_solver.jac_LU.solve(numpy.asanyarray(natural_direction))
    tangent_direction = tangent_direction / tangent_direction[-1]

    colloc_solver.p = colloc_solver.p.at[-1].set(0)

    y_guess = colloc_solver.y + tangent_direction[:-colloc_solver.n_par].reshape(colloc_solver.y.shape, order="F")
    p_guess = colloc_solver.p + tangent_direction[-colloc_solver.n_par:]
    continuation.update_args(colloc_solver, natural_direction, colloc_solver.y, colloc_solver.p, y_guess, p_guess, rc_direction)
    colloc_solver.y = y_guess
    colloc_solver.p = p_guess

    colloc_solver.solve(tol=1e-5)

    y_cont = np.array([colloc_solver.y])
    p_cont = np.array([colloc_solver.p])

    if not colloc_solver.success:

        colloc_solver.y = colloc_solver.args[1].reshape(colloc_solver.y.shape, order="F")
        colloc_solver.p = colloc_solver.args[2]
        continuation.update_args(colloc_solver, natural_direction, colloc_solver.y, colloc_solver.p, colloc_solver.y, colloc_solver.p, rc_direction)
        y_cont, p_cont = continuation.cont(colloc_solver, 1, -1e-1, step_size=1, termination_condition=None, min_step_size=1e-5, tol=1e-5)

    if p_cont[-1, 1] > 1:
        try:
            y_cont, p_cont = continuation.cont(colloc_solver, p_cont[-1, 1] + 1e-1, 1, step_size=1 - p_cont[-1, 1], max_step_size=np.abs(1 - p_cont[-1, 1]), termination_condition=None, min_step_size=1e-5, tol=1e-5)
        except RuntimeError:
            solver.success = False

    if p_cont[-1, 1] != 1 or not colloc_solver.success:
        colloc_solver.args = args_prev
        colloc_solver.args[-2].reaction_consts = np.exp(position)
        colloc_solver.y = y_prev
        colloc_solver.p = p_prev.at[-1].set(0)
        return np.inf, F

    J = colloc_solver.jac()
    J = J[:-colloc_solver.n_par + 1, :-colloc_solver.n_par + 1]
    J_LU = scipy.sparse.linalg.splu(J) 
    J_rc = np.zeros((colloc_solver.n - 1, position.size))

    for i in range(position.size):

        rc_direction = np.zeros(position.shape).at[i].set(1)
        args = list(colloc_solver.args)
        args[-1] = rc_direction
        colloc_solver.args = tuple(args)
        dr_drc = colloc_solver.jacp(y_cont[-1].ravel(order="F"), p_cont[-1])[:-colloc_solver.n_par + 1, 1]
        J_rc = J_rc.at[:, i].set(J_LU.solve(-numpy.asanyarray(dr_drc)))

    E += 300 * (p_cont[-1, 0] - 1)**2
    F -= 600 * (p_cont[-1, 0] - 1) * J_rc[colloc_solver.y.size, :]

    E += E_arclength(y_cont[-1])
    F -= grad_arclength(y_cont[-1]).ravel(order="F")@J_rc[:colloc_solver.y.size, :]

    E += E_bounds(position, bounds)
    F -= grad_bounds(position, bounds)

    colloc_solver.args[5].reaction_consts = np.exp(position)

    return E, F

def compute_energy_and_force_ml(position, momentum, colloc_solver, bounds):

    y_prev = colloc_solver.y
    p_prev = colloc_solver.p
    args_prev = colloc_solver.args

    E = 0
    F = np.zeros(position.shape)

    rc_direction = position - colloc_solver.args[5].par
    natural_direction = np.zeros(colloc_solver.n).at[-1].set(1)
    tangent_direction = colloc_solver.jac_LU.solve(numpy.asanyarray(natural_direction))
    tangent_direction = tangent_direction / tangent_direction[-1]

    colloc_solver.p = colloc_solver.p.at[-1].set(0)

    y_guess = colloc_solver.y + tangent_direction[:-colloc_solver.n_par].reshape(colloc_solver.y.shape, order="F")
    p_guess = colloc_solver.p + tangent_direction[-colloc_solver.n_par:]
    continuation.update_args(colloc_solver, natural_direction, colloc_solver.y, colloc_solver.p, y_guess, p_guess, rc_direction)
    colloc_solver.y = y_guess
    colloc_solver.p = p_guess

    try:
        colloc_solver.solve(tol=1e-5)

        y_cont = np.array([colloc_solver.y])
        p_cont = np.array([colloc_solver.p])

        if not colloc_solver.success:

            colloc_solver.y = colloc_solver.args[1].reshape(colloc_solver.y.shape, order="F")
            colloc_solver.p = colloc_solver.args[2]
            continuation.update_args(colloc_solver, natural_direction, colloc_solver.y, colloc_solver.p, colloc_solver.y, colloc_solver.p, rc_direction)
            y_cont, p_cont = continuation.cont(colloc_solver, 1, -1e-1, step_size=1, termination_condition=None, min_step_size=1e-5, tol=1e-5)

        if p_cont[-1, 1] > 1:
            y_cont, p_cont = continuation.cont(colloc_solver, p_cont[-1, 1] + 1e-1, 1, step_size=1 - p_cont[-1, 1], max_step_size=np.abs(1 - p_cont[-1, 1]), termination_condition=None, min_step_size=1e-5, tol=1e-5

    except RuntimeError:
        colloc_solver.success = False

    if p_cont[-1, 1] != 1 or not colloc_solver.success:
        colloc_solver.args = args_prev
        colloc_solver.args[-2].par = position
        colloc_solver.y = y_prev
        colloc_solver.p = p_prev.at[-1].set(0)
        return np.inf, F

    J = colloc_solver.jac()
    J = J[:-colloc_solver.n_par + 1, :-colloc_solver.n_par + 1]
    J_LU = scipy.sparse.linalg.splu(J) 
    J_rc = np.zeros((colloc_solver.n - 1, position.size))

    for i in range(position.size):

        rc_direction = np.zeros(position.shape).at[i].set(1)
        args = list(colloc_solver.args)
        args[-1] = rc_direction
        colloc_solver.args = tuple(args)
        dr_drc = colloc_solver.jacp(y_cont[-1].ravel(order="F"), p_cont[-1])[:-colloc_solver.n_par + 1, 1]
        J_rc = J_rc.at[:, i].set(J_LU.solve(-numpy.asanyarray(dr_drc)))

    if p_cont[-1, 0] > 400:
        E += 300 * (p_cont[-1, 0] - 500)**2
        F -= 600 * (p_cont[-1, 0] - 500) * J_rc[colloc_solver.y.size, :]

    if p_cont[-1, 0] < 100:
        E += 300 * (p_cont[-1, 0] - 100)**2
        F -= 600 * (p_cont[-1, 0] - 100) * J_rc[colloc_solver.y.size, :]

    E += E_arclength(y_cont[-1], min_arclength=30)
    F -= grad_arclength(y_cont[-1], min_arclength=30).ravel(order="F")@J_rc[:colloc_solver.y.size, :]

    E += 1e-2 * (smooth_max(y_cont[-1][0]) - 35)**2
    F -= 2e-2 * (smooth_max(y_cont[-1][0]) - 35) * grad_smooth_max(y_cont[-1].ravel(order="F"))@J_rc[:colloc_solver.y.size, :]

    E += 1e-2 * (smooth_max(-y_cont[-1][0]) - 50)**2
    F -= 2e-2 * (smooth_max(-y_cont[-1][0]) - 50) * grad_smooth_max(y_cont[-1].ravel(order="F"))@J_rc[:colloc_solver.y.size, :]

    #E += E_floquet(y_cont[-1, :, 0], p_cont[-1, 0], position, colloc_solver.args[5].a0, colloc_solver.args[5].c0, floquet_multiplier_threshold)
    #F -= grad_floquet(y_cont[-1, :, 0], p_cont[-1, 0], position, colloc_solver.args[5].a0, colloc_solver.args[5].c0, floquet_multiplier_threshold)

    E += E_bounds(position, bounds)
    F -= grad_bounds(position, bounds)

    colloc_solver.args[5].par = position

    return E, F

@jax.jit
def wcov(x, dat, scale, cov_weight=1):
    
    weight = np.exp(-scale * np.sum((dat - x)**2, axis=1) / 2)
    # weight = scale / (np.pi * (1 + (scale * np.linalg.norm(dat - x, axis=1))**2))
    weight /= weight.sum()
    wmean = np.sum(weight.reshape((weight.size, 1)) * dat, axis=0)
    dat_centered = dat - wmean
    
    return np.identity(x.size) + cov_weight * (weight * dat_centered.T)@dat_centered 

@jax.jit
def B_wcov(x, dat, scale, cov_weight=1):
    return np.linalg.cholesky(wcov(x, dat, scale, cov_weight))

@jax.jit
def divBT(x, dat, scale, cov_weight=1):
    
    dB = jax.jacfwd(B_wcov)(x, dat, scale, cov_weight)
    
    def loop_body(carry, _):
        i = carry
        return i + 1, dB[:, :, i].T[:, i]
    
    return jax.lax.scan(loop_body, init=0, xs=None, length=x.size)[1].sum(axis=0)

@partial(jax.jit, static_argnums=(6))
def implicit_position_update(position, momentum, dt, wcov_dat, wcov_scale, wcov_weight, maxiter=10, tol=1e-9):
        
    position0 = position
        
    def resid(position):
        B = B_wcov(position, wcov_dat, wcov_scale, wcov_weight)
        return position0 + (dt / 2) * B@momentum - position
    
    def loop_body(carry):
        
        i, position, r = carry
        d = np.linalg.solve(jax.jacfwd(resid)(position), -r)
        position = position + d
        r = resid(position)
        
        return i + 1, position, r
    
    def cond(carry):
        
        i, position, r = carry
        err = np.max(np.abs(r / (1 + position)))
        
        return (err > tol) & (i < maxiter)
    
    i, position, r = jax.lax.while_loop(cond, loop_body, (0, position, resid(position)))
    err = np.max(np.abs(r / (1 + position)))
    
    return np.where(err < tol, position, np.full(position.shape, np.nan))

def baoab_precondition(position, momentum, F_prev, dt, friction, wcov_dat, wcov_scale, wcov_weight, prng_key, energy_function, energy_function_args):
    
    prng_key, subkey = jax.random.split(prng_key)
    
    W = jax.random.normal(subkey, shape=momentum.shape)
    c1 = np.exp(-dt * friction / 2)
    c2 = np.sqrt(1 - c1**2)
    
    B0 = B_wcov(position, wcov_dat, wcov_scale, wcov_weight)
    momentum = momentum + (dt / 2) * B0.T@F_prev
    position = implicit_position_update(position, momentum, dt, wcov_dat, wcov_scale, wcov_weight)
    momentum = c1 * momentum + (c1 + 1) * (dt / 2) * divBT(position, wcov_dat, wcov_scale, wcov_weight) + c2 * W
    
    B1 = B_wcov(position, wcov_dat, wcov_scale, wcov_weight)
    position = position + (dt / 2) * B1@momentum

    B2 = B_wcov(position, wcov_dat, wcov_scale, wcov_weight)
    E, F = energy_function(position, momentum, *energy_function_args)
    momentum = momentum + (dt / 2) * B2.T@F
    
    return position, momentum, E, F, prng_key

def obabo(position, momentum, F_prev, dt, friction, prng_key, energy_function, energy_function_args):

    prng_key, subkey = jax.random.split(prng_key)

    W = jax.random.normal(subkey, shape=momentum.shape)
    c1 = np.exp(-dt * friction / 2)
    c2 = np.sqrt(1 - c1**2)

    momentum = c1 * momentum + c2 * W + (dt / 2) * F_prev
    position = position + dt * momentum

    E, F = energy_function(position, momentum, *energy_function_args)

    key, subkey = jax.random.split(prng_key)

    W = jax.random.normal(subkey, shape=momentum.shape)
    momentum = c1 * (momentum + (dt / 2) * F) + c2 * W

    return position, momentum, E, F, prng_key

def generate_langevin_trajectory(position, L, dt, friction, prng_key, stepper, energy_function, colloc_solver, bounds, E_prev=None, F_prev=None):

    prng_key, subkey = jax.random.split(prng_key)

    position_out = numpy.full((L, *position.shape), np.nan)
    momentum_out = numpy.full((L, *position.shape), np.nan)
    E_out = numpy.full(L, np.inf)
    F_out = numpy.full((L, *position.shape), np.nan)
    y_out = numpy.full((L, colloc_solver.y.size), np.nan)
    p_out = numpy.full((L, colloc_solver.p.size), np.nan)

    momentum = jax.random.normal(subkey, position.shape)

    if F_prev is None or E_prev is None:
        E, F = energy_function(position, momentum, (colloc_solver, bounds))
    else:
        E, F = E_prev, F_prev

    H0 = E + np.linalg.norm(momentum)**2 / 2
    
    for i in range(L):

        position, momentum, E, F, prng_key = stepper(position, momentum, F, dt, friction, prng_key, energy_function, (colloc_solver, bounds))
        position_out[i] = position
        momentum_out[i] = momentum
        E_out[i] = E
        F_out[i] = F
        y_out[i] = colloc_solver.y.ravel(order="F")
        p_out[i] = colloc_solver.p

        if E > 2e3:
            break

    H1 = E_out + np.linalg.norm(momentum_out, axis=1)**2 / 2

    prng_key, subkey = jax.random.split(prng_key)
    u = jax.random.uniform(subkey, shape=E_out.shape)
    accept = np.log(u) < -(H1 - H0)

    return position_out, momentum_out, E_out, F_out, y_out, p_out, accept, prng_key

def generate_langevin_trajectory_precondition(position, L, dt, friction, wcov_dat, wcov_scale, wcov_weight, prng_key, energy_function, colloc_solver, bounds, E_prev=None, F_prev=None, thin=1, metropolize=True):

    prng_key, subkey = jax.random.split(prng_key)

    y_prev = colloc_solver.y
    p_prev = colloc_solver.p
    args_prev = colloc_solver.args

    position_prev = position
    momentum_prev = jax.random.normal(subkey, position.shape)
    position_out = numpy.full((L // thin, *position.shape), np.nan)
    momentum_out = numpy.full((L // thin, *position.shape), np.nan)
    E_out = numpy.full(L // thin, np.inf)
    F_out = numpy.full((L // thin, *position.shape), np.nan)
    y_out = numpy.full((L // thin, colloc_solver.y.size), np.nan)
    p_out = numpy.full((L // thin, colloc_solver.p.size), np.nan)

    momentum = momentum_prev 

    if F_prev is None or E_prev is None:
        E_prev, F_prev = energy_function(position, momentum, colloc_solver, bounds)

    E, F = E_prev, F_prev

    H0 = E + np.linalg.norm(momentum)**2 / 2
    j = 0

    for i in range(L):

        position, momentum, E, F, prng_key = baoab_precondition(position, momentum, F, dt, friction, wcov_dat, wcov_scale, wcov_weight, prng_key, energy_function, (colloc_solver, bounds))

        if i % thin == thin - 1:
            position_out[j] = position
            momentum_out[j] = momentum
            E_out[j] = E
            F_out[j] = F
            y_out[j] = colloc_solver.y.ravel(order="F")
            p_out[j] = colloc_solver.p
            j += 1

        if E > 2e3:
            break

    H1 = E_out + np.linalg.norm(momentum_out, axis=1)**2 / 2

    prng_key, subkey = jax.random.split(prng_key)
    u = jax.random.uniform(subkey, shape=E_out.shape)
    accept = np.log(u) < -(H1 - H0)
    fail = np.isinf(E_out)

    if not metropolize:
        accept = np.isfinite(E_out)

    accept = accept.reshape((accept.size, 1))
    position_out = np.where(accept, position_out, position_prev)
    momentum_out = np.where(accept, momentum_out, momentum_prev)
    E_out = np.where(accept.ravel(), E_out, E_prev)
    F_out = np.where(accept, F_out, F_prev)
    y_out = np.where(accept, y_out, y_prev.ravel(order="F"))
    p_out = np.where(accept, p_out, p_prev)

    if not accept[-1]:
        colloc_solver.args = args_prev
        colloc_solver.args[-2].reaction_consts = np.exp(position)
        colloc_solver.y = y_prev
        colloc_solver.p = p_prev.at[-1].set(0)

    return position_out, momentum_out, E_out, F_out, y_out, p_out, accept.ravel(), fail, prng_key

def random_walk_metropolis_precondition(position, L, dt, friction, wcov_dat, wcov_scale, wcov_weight, prng_key, energy_function, colloc_solver, bounds, E_prev=None, F_prev=None, thin=1, metropolize=True):

    position_out = numpy.full((L // thin, *position.shape), np.nan)
    momentum_out = numpy.full((L // thin, *position.shape), np.nan)
    E_out = numpy.full(L // thin, np.inf)
    F_out = numpy.full((L // thin, *position.shape), np.nan)
    y_out = numpy.full((L // thin, colloc_solver.y.size), np.nan)
    p_out = numpy.full((L // thin, colloc_solver.p.size), np.nan)
    accept = numpy.zeros(L // thin, np.bool_)
    fail = numpy.zeros(L // thin, np.bool_)

    momentum = np.zeros_like(position)

    if F_prev is None or E_prev is None:
        E, F = energy_function(position, momentum, colloc_solver, bounds)
    else:
        E, F = E_prev, F_prev

    j = 0

    for i in range(L):

        prng_key, subkey = jax.random.split(prng_key)
        W = jax.random.normal(subkey, shape=momentum.shape)
        B0 = B_wcov(position, wcov_dat, wcov_scale, wcov_weight)
        momentum = (dt / 2) * B0.T@W
        position_propose = position + momentum
        E_propose, F_propose = energy_function(position, momentum, colloc_solver, bounds)

        prng_key, subkey = jax.random.split(prng_key)
        u = jax.random.uniform(subkey)
        accept[j] = np.log(u) < -(E_propose - E)
        fail[j] = np.isinf(E)

        if accept[j]:
            position = position_propose
            E = E_propose
            F = F_propose

        if i % thin == thin - 1:
            position_out[j] = position
            momentum_out[j] = momentum
            E_out[j] = E
            F_out[j] = F
            y_out[j] = colloc_solver.y.ravel(order="F")
            p_out[j] = colloc_solver.p
            j += 1

    return position_out, momentum_out, E_out, F_out, y_out, p_out, accept, fail, prng_key

def sample_mpi(odesystem, energy_function, f_bvp, bc_bvp, position, y0, period0, bounds, trajectory_length, comm, dt=1e-3, friction=1e-1, maxiter=1000, floquet_multiplier_threshold=8e-1, seed=None, thin=1, metropolize=True, dynamics=generate_langevin_trajectory_precondition):

    if seed is None:
        seed = time.time_ns()

    prng_key = jax.random.PRNGKey(seed)
    n_dim = odesystem.n_dim - odesystem.n_conserve
    n_walkers = position.shape[0]

    save_length = trajectory_length // thin + (trajectory_length % thin != 0)
    out = np.zeros((save_length * maxiter + 1, ((n_walkers + 1) // 2) * 2, position.shape[1] + 1 + position.shape[1] + y0[0].size + 1))

    solver = [None for _ in range(n_walkers)]
    odes = [None for _ in range(n_walkers)]

    for i in range(2):

        for j in range(i * (n_walkers // 2) + comm.Get_rank(), i * (n_walkers // 2) + n_walkers // 2, comm.Get_size()):

            odes[j] = odesystem(position[j], log_rc=True)
            p0 = np.array([period0[j], 0])
            solver_args = (np.zeros(y0[j].size + p0.size).at[-1].set(1), y0[j].ravel(order="F"), p0, y0[j].ravel(order="F"), p0, odes[j], np.zeros(position.shape[1]))
            solver[j] = colloc(f_bvp, bc_bvp, y0[j].reshape((n_dim, y0[j].size // n_dim), order="F"), p0, solver_args)
            solver[j]._superLU()

            E, F = energy_function(position[j], np.zeros(position.shape[1]), solver[j], bounds)

            out = out.at[0, j, position.shape[1]:].set(np.concatenate([np.array([E]), F, y0[j].ravel(order="F"), np.array([period0[j]])]))

    out = out.at[0, :, :position.shape[1]].set(position)

    accepted = np.zeros(n_walkers)
    rejected = np.zeros(n_walkers)
    failed = np.zeros(n_walkers)

    for i in range(maxiter):

        for j in range(2):

            for k in range(j * (n_walkers // 2) + comm.Get_rank(), (j + 1) * (n_walkers // 2), comm.Get_size()):

                prng_key, subkey = jax.random.split(prng_key)
                args_prev = solver[k].args
                E_prev = out[i * save_length, k, position.shape[1]]
                F_prev = out[i * save_length, k, position.shape[1] + 1:2 * position.shape[1] + 1]
                j_other = np.logical_not(j).astype(np.int64)
                wcov_dat = out[i * save_length, j_other * (n_walkers // 2):(j_other + 1) * (n_walkers // 2), :position.shape[1]]

                pos_traj, mom_new, E_traj, F_traj, y_traj, p_traj, accept, fail, prng_key = dynamics(
                    out[i * save_length, k, :position.shape[1]], save_length * thin,
                    dt, friction, wcov_dat, wcov_scale=2e-1, wcov_weight=1,
                    prng_key=prng_key, energy_function=energy_function, 
                    colloc_solver=solver[k], bounds=bounds, E_prev=E_prev, F_prev=F_prev, thin=thin, metropolize=metropolize)

                accepted = accepted.at[k].add(accept.sum())
                rejected = rejected.at[k].add(np.logical_and(np.logical_not(accept), np.logical_not(fail)).sum())
                failed = failed.at[k].add(fail.sum())
                
                out_traj = np.concatenate([pos_traj, E_traj.reshape((E_traj.size, 1)), F_traj, y_traj, p_traj[:, :-1]], axis=1)
                out = out.at[i * save_length + 1:(i + 1) * save_length + 1, k].set(out_traj)

                pos_partial = np.copy(out[i * save_length + 1:(i + 1) * save_length + 1, j * (n_walkers // 2):(j + 1) * (n_walkers // 2)])
                #allwalkers, _ = mpi4jax.allreduce(x=pos_partial, op=MPI.SUM, comm=comm)
                allwalkers = comm.allreduce(pos_partial, op=MPI.SUM)
                out = out.at[i * save_length + 1:(i + 1) * save_length + 1, j * (n_walkers // 2):(j + 1) * (n_walkers // 2)].set(allwalkers)

                print("Iteration:%d Walker:%d Accepted:%d Rejected:%d Failed:%d"%(i, k, accepted[k], rejected[k], failed[k]), flush=True)

    accepted = comm.allreduce(accepted, op=MPI.SUM)
    rejected = comm.allreduce(failed, op=MPI.SUM)
    failed = comm.allreduce(failed, op=MPI.SUM)

    return out[1:], accepted, rejected, failed

def sample(odesystem, energy_function, position, y0, period0, bounds, trajectory_length, dt=1e-3, friction=1e-1, maxiter=1000, floquet_multiplier_threshold=8e-1, seed=None, thin=1, metropolize=True):

    if seed is None:
        seed = time.time_ns()

    prng_key = jax.random.PRNGKey(seed)

    odes = odesystem(np.exp(position))
    n_dim = odes.n_dim - odes.n_conserve
    p0 = np.array([period0, 0])
    solver_args = (np.zeros(y0.size + p0.size).at[-1].set(1), y0.ravel(order="F"), p0, y0.ravel(order="F"), p0, odes, np.zeros(position.size))
    solver = colloc(continuation.f_rc, continuation.bc_rc, y0.reshape((n_dim, y0.size // n_dim), order="F"), p0, solver_args)
    solver._superLU()

    out = numpy.empty((trajectory_length * maxiter + 1, position.size + 1 + position.size + y0.size + np.size(period0)))
    E, F = energy_function(position, np.zeros(position.size), solver, bounds)

    out[0, :position.size] = position
    out[0, position.size] = E
    out[0, position.size + 1:2 * position.size + 1] = F
    out[0, 2 * position.size + 1:2 * position.size + 1 + y0.size] = y0.ravel(order="F")
    out[0, 2 * position.size + 1 + y0.size:2 * position.size + 1 + y0.size + np.size(period0)] = p0[0]

    y0 = y0.reshape(solver.y.shape, order="F")
    accepted, rejected, failed = 0, 0, 0

    for i in range(maxiter):

        prng_key, subkey = jax.random.split(prng_key)
        args_prev = solver.args
        pos_traj, mom_new, E_traj, F_traj, y_traj, p_traj, accept, prng_key = generate_langevin_trajectory(position, trajectory_length, dt, friction, prng_key, stepper=obabo, energy_function=energy_function, 
                                                                                            colloc_solver=solver, bounds=bounds, E_prev=E, F_prev=F)

        if not metropolize:
            accept = np.isfinite(E_traj)

        accept = accept.reshape((accept.size, 1))

        accepted += accept.sum()
        failed += np.isinf(E_traj).sum()
        rejected += np.logical_and(np.logical_not(accept.ravel()), np.isfinite(E_traj)).sum()

        out[i * trajectory_length + 1:(i + 1) * trajectory_length + 1, :position.size]\
        = np.where(accept, pos_traj, out[i * trajectory_length, :position.size])

        out[i * trajectory_length + 1:(i + 1) * trajectory_length + 1, position.size]\
        = np.where(accept.ravel(), E_traj, out[i * trajectory_length, position.size])

        out[i * trajectory_length + 1:(i + 1) * trajectory_length + 1, position.size + 1:2 * position.size + 1]\
        = np.where(accept, F_traj, out[i * trajectory_length, position.size + 1:2 * position.size + 1])

        out[i * trajectory_length + 1:(i + 1) * trajectory_length + 1, 2 * position.size + 1:2 * position.size + 1 + y0.size]\
        = np.where(accept, y_traj, out[i * trajectory_length, 2 * position.size + 1:2 * position.size + 1 + y0.size])

        out[i * trajectory_length + 1:(i + 1) * trajectory_length + 1, 2 * position.size + 1 + y0.size:2 * position.size + 1 + y0.size + np.size(period0)]\
        = np.where(accept, p_traj[:, :1], out[i * trajectory_length, 2 * position.size + 1 + y0.size: 2 * position.size + 1 + y0.size + np.size(period0)])

        if accept[-1]:
            position = pos_traj[-1]
            y0 = solver.y
            p0 = solver.p.at[-1].set(0)
            E = E_traj[-1]
            F = F_traj[-1]

        else:
            solver.args = args_prev
            solver.args[-2].reaction_consts = np.exp(position)
            solver.y = y0
            solver.p = p0

        print("Iteration:%d Accepted:%d Rejected:%d Failed:%d"%(i, accepted, rejected, failed), flush=True)

    return out[1::thin], accepted, rejected, failed
