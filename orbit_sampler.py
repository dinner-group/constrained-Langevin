import numpy
import diffrax
import jax
import jax.numpy as np
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
def grad_arclength(y):
    return jax.jacrev(E_arclength)(y)

def compute_energy_and_force(position, momentum, colloc_solver, bounds, floquet_multiplier_threshold=0.8):

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
        y_cont, p_cont = continuation.cont(colloc_solver, p_cont[-1, 1] + 1e-1, 1, step_size=1 - p_cont[-1, 1], max_step_size=np.abs(1 - p_cont[-1, 1]), termination_condition=None, min_step_size=1e-5, tol=1e-5)

    if p_cont[-1, 1] != 1:
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

    #E += E_floquet(y_cont[-1, :, 0], p_cont[-1, 0], position, colloc_solver.args[5].a0, colloc_solver.args[5].c0, floquet_multiplier_threshold)
    #F -= grad_floquet(y_cont[-1, :, 0], p_cont[-1, 0], position, colloc_solver.args[5].a0, colloc_solver.args[5].c0, floquet_multiplier_threshold)

    for i in range(bounds.shape[0]):

        if position[i] < bounds[i, 0]:
            E += (bounds[i, 0] - position[i])**2
            F -= F.at[i].add(2 * (position[i] - bounds[i, 0]))
        elif position[i] > bounds[i, 1]:
            E += (bounds[i, 1] - position[i])**2
            F -= F.at[i].add(2 * (position[i] - bounds[i, 1]))

    colloc_solver.args[5].reaction_consts = np.exp(position)

    return E, F

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
        E, F = energy_function(position, momentum, *energy_function_args)
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

def sample(position, y0, period0, bounds, langevin_trajectory_length, dt=1e-3, friction=1e-1, maxiter=1000, floquet_multiplier_threshold=8e-1, seed=None, thin=1, metropolize=True):

    if seed is None:
        seed = time.time_ns()

    prng_key = jax.random.PRNGKey(seed)

    kaiabc = KaiODE(np.exp(position))
    n_dim = kaiabc.n_dim - kaiabc.n_conserve
    p0 = np.array([period0, 0])
    solver_args = (np.zeros(y0.size + p0.size).at[-1].set(1), y0.ravel(order="F"), p0, y0.ravel(order="F"), p0, kaiabc, np.zeros(position.size))
    solver = colloc(continuation.f_rc, continuation.fp_rc, y0.reshape((n_dim, y0.size // n_dim), order="F"), p0, solver_args)
    solver._superLU()

    out = numpy.empty((langevin_trajectory_length * maxiter + 1, position.size + 1 + position.size + y0.size + np.size(period0)))
    E, F = compute_energy_and_force(position, np.zeros(position.size), solver, bounds)

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
        pos_traj, mom_new, E_traj, F_traj, y_traj, p_traj, accept, prng_key = generate_langevin_trajectory(position, langevin_trajectory_length, dt, friction, prng_key, stepper=obabo, energy_function=compute_energy_and_force, 
                                                                                            colloc_solver=solver, bounds=bounds, E_prev=E, F_prev=F)

        if not metropolize:
            accept = np.isfinite(E_traj)

        accept = accept.reshape((accept.size, 1))

        accepted += accept.sum()
        failed += np.isinf(E_traj).sum()
        rejected += np.logical_and(np.logical_not(accept.ravel()), np.isfinite(E_traj)).sum()

        out[i * langevin_trajectory_length + 1:(i + 1) * langevin_trajectory_length + 1, :position.size]\
        = np.where(accept, pos_traj, out[i * langevin_trajectory_length, :position.size])

        out[i * langevin_trajectory_length + 1:(i + 1) * langevin_trajectory_length + 1, position.size]\
        = np.where(accept.ravel(), E_traj, out[i * langevin_trajectory_length, position.size])

        out[i * langevin_trajectory_length + 1:(i + 1) * langevin_trajectory_length + 1, position.size + 1:2 * position.size + 1]\
        = np.where(accept, F_traj, out[i * langevin_trajectory_length, position.size + 1:2 * position.size + 1])

        out[i * langevin_trajectory_length + 1:(i + 1) * langevin_trajectory_length + 1, 2 * position.size + 1:2 * position.size + 1 + y0.size]\
        = np.where(accept, y_traj, out[i * langevin_trajectory_length, 2 * position.size + 1:2 * position.size + 1 + y0.size])

        out[i * langevin_trajectory_length + 1:(i + 1) * langevin_trajectory_length + 1, 2 * position.size + 1 + y0.size:2 * position.size + 1 + y0.size + np.size(period0)]\
        = np.where(accept, p_traj[:, :1], out[i * langevin_trajectory_length, 2 * position.size + 1 + y0.size: 2 * position.size + 1 + y0.size + np.size(period0)])

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

    return out[1::thin]
