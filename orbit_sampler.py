import numpy
import diffrax
import jax
import jax.numpy as np
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
def LL_monodromy(y0, period, reaction_consts, a0=0.6, c0=3.5, floquet_multiplier_threshold=0.8):

    LL = 0

    M = continuation.compute_monodromy_1(y0, period, reaction_consts, a0, c0)
    floquet_multipliers, evecs = np.linalg.eig(M)
    abs_multipliers = np.abs(floquet_multipliers)
    LL += np.where(abs_multipliers > floquet_multiplier_threshold, 100 * (abs_multipliers - floquet_multipliers_threshold)**2, 0).sum()

def compute_LL(solver1, solver2, bounds, floquet_multiplier_threshold=0.8):

    LL = 0
    LL -= 300 * (solver1.p[0] - 1)**2
    LL -= 100 * (solver2.p[0] - 1)**2
    LL -= 1 / np.linalg.norm(solver1.y[:, 1:] - solver1.y[:, :-1], axis=0).sum()**2
    LL -= 1 / np.linalg.norm(solver2.y[:, 1:] - solver2.y[:, :-1], axis=0).sum()**2
    LL += LL_monodromy(solver1.y[:, 0], solver1.p[0], solver1.args[5].a0, solver1.args[5].c0, floquet_multiplier_threshold)
    LL += LL_monodromy(solver2.y[:, 0], solver2.p[0], solver2.args[5].a0, solver2.args[5].c0, floquet_multiplier_threshold)

    for i in range(bounds.shape[0]):

        if solver1.args[0].reaction_consts[i] < bounds[i, 0]:
            LL -= (np.log(bounds[i, 0]) - np.log(solver1.args[0].reaction_consts[i]))**2
        elif solver1.args[0].reaction_consts[i] > bounds[i, 1]:
            LL -= (np.log(bounds[i, 1]) - np.log(solver1.args[0].reaction_consts[i]))**2
    
    return LL

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

    solver1 = colloc(f_rc, fp_rc, y0, np.array([period0, 0.]), args=(initial_continuation_direction, y0.ravel(order="F"), p0_rc, ds, model, reaction_consts_0, a0, rc_direction, max_amplitude_species))
    solver2 = colloc(f_rc, fp_rc, y_acont[-1], np.array([p_acont[-1, 0], 0.]), args=(intial_continuation_direction, y0.ravel(order="F"), p0_rc, ds, model, reaction_consts_0, p_acont[-1, 1], rc_direction, max_amplitude_species))
    LL = compute_LL(solver1, solver2, bounds)

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
   
        solver1.success = False
        solver1.y = y_out[-1][0]
        solver1.p = solver1.p.at[0].set(period_out[-1][0])
        solver1.p = solver1.p.at[1].set(0)
        solver1.args = (initial_continuation_direction, y_out[-1][0], period_out[-1][0], solver1.args[3], model, reaction_consts_out[-1], a0, step, max_amplitude_species)
        cont(solver1, p_stop=1, step_size=ds)

        solver2.success = False
        solver2.y = y_out[-1][1]
        solver2.p = solver2.p.at[0].set(period_out[-1][1])
        solver2.args = (initial_continuation_direction, y_out[-1][1], period_out[-1][1], solver2.args[3], model, reaction_consts_out[-1], p_acont[-1, 1], step, max_amplitude_species)
        cont(solver2, p_stop=1, step_size=ds)
        
        if not (solver1.success and solver2.success):
            y_out.append(y_out[-1])
            period_out.append(period_out[-1])
            reaction_consts_out.append(reaction_consts_out[-1])
            failed += 1
            continue
            
        LL_propose = compute_LL(solver1, solver2)
        proposal_factor_r = 0
        proposal_factor_f = 0

        acceptance_ratio = LL_propose - LL + proposal_factor_r - proposal_factor_f
        
        key, subkey = jax.random.split(key)
        accept = np.log(jax.random.uniform(subkey)) < acceptance_ratio
        
        if accept:
            
            y_out.append([solver1.y, solver2.y])
            period_out.append([solver1.p[0], solver2.p[0]])
            reaction_consts_out.append(reaction_consts_propose)
            LL = LL_propose
            accepted += 1

        else:

            y_out.append(y_out[-1])
            period_out.append(period_out[-1])
            reaction_consts_out.append(reaction_consts_out[-1])
            rejected += 1

    return np.array(y_out), np.array(period_out), np.array(dperiod_out), np.array(reaction_consts_out)
