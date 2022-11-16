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
    E += np.where(abs_multipliers > floquet_multiplier_threshold, 100 * (abs_multipliers - floquet_multiplier_threshold)**2, 0).sum()

    return E

@jax.jit
def grad_floquet(y0, period, log_rc, a0=0.6, c0=3.5, floquet_multiplier_threshold=0.8):
    return jax.jacfwd(E_floquet, 2)(y0, period, log_rc, a0, c0, floquet_multiplier_threshold)

def termination_condition(colloc_solver):
    return colloc_solver.args[0][-1] < 0

@jax.jit
def E_arclength(y):
    return 1 / np.linalg.norm(y[1:] - y[:-1], axis=0).sum()**2

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

    colloc_solver.solve()

    y_cont = np.array([colloc_solver.y])
    p_cont = np.array([colloc_solver.p])

    if not colloc_solver.success:

        colloc_solver.y = colloc_solver.args[1].reshape(colloc_solver.y.shape, order="F")
        colloc_solver.p = colloc_solver.args[2]
        y_cont, p_cont = continuation.cont(colloc_solver, 1, 0, step_size=1, termination_condition=termination_condition, min_step_size=1e-5, tol=1e-5)

    if p_cont[-1, 1] > 1:
        y_cont, p_cont = continuation.cont(colloc_solver, 1.1, 1, step_size=1 - p_cont[-1, 1], max_step_size=np.abs(1 - p_cont[-1, 1]), termination_condition=termination_condition, min_step_size=1e-5, tol=1e-5)

    if p_cont[-1, 1] != 1:
        return np.inf, np.zeros(position.shape)

    J = colloc_solver.jac()
    J = J[:-colloc_solver.n_par + 1, :-colloc_solver.n_par + 1]
    J_LU = scipy.sparse.linalg.splu(J) 
    J_rc = np.zeros((colloc_solver.n - 1, position.size))

    for i in range(position.size):

        rc_direction = np.zeros(position.shape).at[i].set(1)
        args = list(colloc_solver.args)
        args[-1] = rc_direction
        colloc_solver.args = tuple(args)
        dr_drc = colloc_solver.jacp(colloc_solver.resid, argnums=1)(y_cont[-1].ravel(order="F"), p_cont[-1])[:-colloc_solver.n_par + 1, 1]
        J_rc = J_rc.at[i, :].set(J_LU.solve(numpy.asanyarray(dr_drc)))

    E += 300 * (p_cont[-1, 0] - 1)**2
    F -= 600 * J_rc[-colloc_solver.n_par, :]

    E += E_arclength(y_cont[-1])
    F -= grad_arclength(y_cont[-1]).ravel(order="F")@J_rc[:-colloc_solver.n_par, :]

    E += E_floquet(y_cont[-1, :, 0], p_cont[-1, 0], position, colloc_solver.args[5].a0, colloc_solver.args[5].c0, floquet_multiplier_threshold)
    F -= grad_floquet(y_cont[-1, :, 0], p_cont[-1, 0], position, colloc_solver.args[5].a0, colloc_solver.args[5].c0, floquet_multiplier_threshold)

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

    prng_key, subkey = jax.random.split(key)

    W = jax.random.normal(subkey, shape=momentum.shape)
    c1 = np.exp(-dt * friction / 2)
    momentum = c1 * momentum + c1 * W0 + (dt / 2) * F_prev
    position = position + dt * momentum
    
    E, F = energy_function(position, momentum, *energy_function_args)

    key, subkey = jax.random.split(prng_key)

    W = jax.random.normal(subkey, shape=momentum.shape)
    momentum = np.exp(-dt * friction / 2) * (momentum + (dt / 2) * force) + c1 * W

    return position, momentum, E, F, prng_key

def generate_langevin_trajectory(position, L, dt, friction, prng_key, stepper, energy_function, energy_function_args, F_prev=None, E_prev=None):

    prng_key, subkey = jax.random.split(prng_key)

    position_out = np.full((L, *position.shape), np.nan)
    momentum_out = np.full((L, *position.shape), np.nan)
    E_out = np.full(L, np.inf)
    F_out = np.full(L, np.nan)

    momentum = jax.random.normal(subkey, position.shape)

    if F_prev is None or E_prev is None:
        E, F = energy_function(position, momentum, *energy_function_args)
    else:
        E, F = E_prev, F_prev

    H0 = E + np.linalg.norm(momentum)**2 / 2
    
    for i in range(L):

        position, momentum, E, F, prng_key = stepper(position, momentum, F, dt, friction, prng_key, energy_function, energy_function_args)
        position_out = position_out.at[i].set(position)
        momentum_out = momentum_out.at[i].set(momentum)
        E_out = E_out.at[i].set(E)
        F_out = F_out.at[i].set(F)

        if E > 2e3:
            break

    H1 = E + np.linalg.norm(momentum)**2 / 2

    prng_key, subkey = jax.random.split(prng_key)
    u = jax.random.uniform(subkey)
    accept = np.log(u) > -(H1 - H0)

    return position_out, momentum_out, E_out, F_out, accept, prng_key

def sample(y0, period0, reaction_consts_0, bounds, ds=1e-2, dt=1e-1, maxiter=1000, floquet_multiplier_threshold=7e-1, seed=None):

    a0 = 0.6
    a1 = 1.2
    if seed is None:
        seed = time.time_ns()
        
    key = jax.random.PRNGKey(seed)

    max_amplitude_species = np.argmax(np.max(y0, axis=1) - np.min(y0, axis=1))

    model = KaiODE(reaction_consts_0)
    p0_a = np.array([period0, a0])

    initial_continuation_direction = np.zeros(y0.size + np.size(period0) + 1).at[-1].set(1)

    solvera = colloc(f_a, fp_a, y0, p0_a, args=(initial_continuation_direction, y0.ravel(order="F"), p0_a, ds, model, max_amplitude_species))
    y_acont, p_acont = cont(solvera, a1, step_size=ds)

    y_out = [[y0, y_acont[-1]]]
    period_out = [[period0, p_acont[-1, 0]]]
    dperiod_out = []
    reaction_consts_out = [reaction_consts_0]

    rc_direction = np.zeros_like(reaction_consts_0)

    colloc_solver = colloc(f_rc, fp_rc, y0, np.array([period0, 0.]), args=(initial_continuation_direction, y0.ravel(order="F"), p0_rc, ds, model, reaction_consts_0, a0, rc_direction, max_amplitude_species))
    solver2 = colloc(f_rc, fp_rc, y_acont[-1], np.array([p_acont[-1, 0], 0.]), args=(intial_continuation_direction, y0.ravel(order="F"), p0_rc, ds, model, reaction_consts_0, p_acont[-1, 1], rc_direction, max_amplitude_species))
    LL = compute_LL(colloc_solver, solver2, bounds)

    #_, dp = compute_sensitivity_boundary(y0[:, 0], y0[:, -1], p0, reaction_consts_0, max_amplitude_species, M=M)
    #dp = dp * reaction_consts_0
    #dperiod_out.append(dp)
    
    i = 0

    accepted = 0
    rejected = 0
    failed = 0

    while i < maxiter:
        
        i += 1

        print("iteration %d"%(i), flush=True)
        print("Log likelihood %.5f"%(LL), flush=True)
        print("accepted:%d rejected:%d failed:%d"%(accepted, rejected, failed), flush=True)
        print("period: %.5f, %.5f"%(period_out[-1][0], period_out[-1][1]), flush=True)
        
        key, subkey = jax.random.split(key)
        randn = jax.random.normal(subkey, shape=(reaction_consts_0.shape[0],))
        step = randn * dt
        
        reaction_consts_propose = np.exp(np.log(reaction_consts_out[-1]) + step)
        max_amplitude_species = np.argmax(np.max(y_out[-1][0], axis=1) - np.min(y_out[-1][0], axis=1))
   
        colloc_solver.success = False
        colloc_solver.y = y_out[-1][0]
        colloc_solver.p = colloc_solver.p.at[0].set(period_out[-1][0])
        colloc_solver.p = colloc_solver.p.at[1].set(0)
        colloc_solver.args = (initial_continuation_direction, y_out[-1][0], period_out[-1][0], colloc_solver.args[3], model, reaction_consts_out[-1], a0, step, max_amplitude_species)
        cont(colloc_solver, p_stop=1, step_size=ds)

        solver2.success = False
        solver2.y = y_out[-1][1]
        solver2.p = solver2.p.at[0].set(period_out[-1][1])
        solver2.args = (initial_continuation_direction, y_out[-1][1], period_out[-1][1], solver2.args[3], model, reaction_consts_out[-1], p_acont[-1, 1], step, max_amplitude_species)
        cont(solver2, p_stop=1, step_size=ds)
        
        if not (colloc_solver.success and solver2.success):
            y_out.append(y_out[-1])
            period_out.append(period_out[-1])
            reaction_consts_out.append(reaction_consts_out[-1])
            failed += 1
            continue
            
        LL_propose = compute_LL(colloc_solver, solver2)
        proposal_factor_r = 0
        proposal_factor_f = 0

        acceptance_ratio = LL_propose - LL + proposal_factor_r - proposal_factor_f
        
        key, subkey = jax.random.split(key)
        accept = np.log(jax.random.uniform(subkey)) < acceptance_ratio
        
        if accept:
            
            y_out.append([colloc_solver.y, solver2.y])
            period_out.append([colloc_solver.p[0], solver2.p[0]])
            reaction_consts_out.append(reaction_consts_propose)
            LL = LL_propose
            accepted += 1

        else:

            y_out.append(y_out[-1])
            period_out.append(period_out[-1])
            reaction_consts_out.append(reaction_consts_out[-1])
            rejected += 1

    return np.array(y_out), np.array(period_out), np.array(dperiod_out), np.array(reaction_consts_out)
