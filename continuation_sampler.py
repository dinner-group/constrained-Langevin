import numpy
import scipy.integrate
import jax
import jax.numpy as np
from model import *
from collocation import *
import time
from functools import partial
import matplotlib.pyplot as plt
import os
jax.config.update("jax_enable_x64", True)

path = os.path.dirname(__file__)
K = np.array(numpy.loadtxt(path + "/K.txt", dtype=numpy.int64))
S = np.array(numpy.loadtxt(path + "/S.txt", dtype=numpy.int64))
orbits = np.array([np.ravel(np.load("candidate_orbit_%d.npy"%(i))) for i in range(41)])

@jax.jit
def f_M(t, y, model):
    
    ydot = np.zeros_like(y)
    ydot = ydot.at[:KaiODE.n_dim - KaiODE.n_conserve].set(model.f_red(t, y[:KaiODE.n_dim - KaiODE.n_conserve]))
    ydot = ydot.at[KaiODE.n_dim - KaiODE.n_conserve:].set((model.jac_red(t, y[:KaiODE.n_dim - KaiODE.n_conserve])\
                                                          @np.reshape(y[KaiODE.n_dim - KaiODE.n_conserve:], 
                                                                      (KaiODE.n_dim - KaiODE.n_conserve, ydot.shape[0] // (KaiODE.n_dim - KaiODE.n_conserve) - 1), order="F")).ravel(order="F"))
    
    return ydot

jac_M = jax.jit(jax.jacfwd(f_M, argnums=1))

@jax.jit
def f_sens_fwd(t, y, model):
    
    ydot = np.zeros_like(y)
    Jy = model.jac_red(t, y[:KaiODE.n_dim - KaiODE.n_conserve], model.reaction_consts)
    Jp = jax.jacfwd(model.f_red, argnums=2)(t, y[:KaiODE.n_dim - KaiODE.n_conserve], model.reaction_consts)
    
    ydot = ydot.at[:KaiODE.n_dim - KaiODE.n_conserve].set(model.f_red(t, y[:KaiODE.n_dim - KaiODE.n_conserve], model.reaction_consts))
    ydot = ydot.at[KaiODE.n_dim - KaiODE.n_conserve:].set(np.ravel(Jy@y[KaiODE.n_dim - KaiODE.n_conserve:].reshape((KaiODE.n_dim - KaiODE.n_conserve, 50), order="F") + Jp, order="F"))
    
    return ydot

jac_sens = jax.jit(jax.jacfwd(f_sens_fwd, argnums=1))

def compute_monodromy(y0, period, reaction_consts, a0=0.6, c0=3.5, ATPfrac=1.0):
    
    model = KaiODE(reaction_consts, a0=a0, c0=c0, ATPfrac=ATPfrac)
    yM_0 = np.concatenate([y0, np.identity(y0.shape[0]).ravel(order="F")])
    traj = scipy.integrate.solve_ivp(fun=f_M, jac=jac_M, t_span=(0, period), y0=yM_0, method="BDF", args=(model,), atol=1e-6, rtol=1e-4)
    
    return traj

def compute_sensitivity(y0, period, reaction_consts, a0=0.6, c0=3.5, ATPfrac=1.0):

    model = KaiODE(reaction_consts, a0=a0, c0=c0, ATPfrac=ATPfrac)
    ys_0 = np.concatenate([y0, np.zeros(y0.shape[0] * reaction_consts.shape[0])])
    traj = scipy.integrate.solve_ivp(fun=f_sens_fwd, jac=jac_sens, t_span=(0, period), y0=ys_0, method="BDF", args=(model,), atol=1e-6, rtol=1e-4)
    
    return traj

def compute_sensitivity_boundary(ya, yb, period, reaction_consts, max_amplitude_species, a0=0.6, c0=3.5, ATPfrac=1.0, M=None, sens=None):

    model = KaiODE(reaction_consts, a0=a0, c0=c0, ATPfrac=ATPfrac)
                   
    if M is None:
        M = compute_monodromy(ya, period, reaction_consts, a0, c0, ATPfrac).y[KaiODE.n_dim - KaiODE.n_conserve:, -1].reshape((KaiODE.n_dim - KaiODE.n_conserve, KaiODE.n_dim - KaiODE.n_conserve), order="F")
        
    if sens is None:
        sens = compute_sensitivity(ya, period, reaction_consts, a0, c0, ATPfrac).y[KaiODE.n_dim - KaiODE.n_conserve:, -1].reshape((KaiODE.n_dim - KaiODE.n_conserve, reaction_consts.shape[0]), order="F")
        
    J = np.zeros((M.shape[0] + 1, M.shape[0] + 1))
    J = J.at[:M.shape[0], :M.shape[0]].set(M)
    J = J.at[np.diag_indices(M.shape[0])].add(-1)
    J = J.at[-1, :KaiODE.n_dim - KaiODE.n_conserve].set(model.jac_red(0, ya)[max_amplitude_species, :])
    J = J.at[:KaiODE.n_dim - KaiODE.n_conserve, -1].set(model.f_red(0, yb))
    r = np.vstack([sens,
                  jax.jacfwd(model.f_red, argnums=2)(0, ya, model.reaction_consts)[max_amplitude_species, :]])
    
    sol = np.linalg.solve(J, -r)
    
    return sol[:KaiODE.n_dim - KaiODE.n_conserve, :], sol[KaiODE.n_dim - KaiODE.n_conserve:, :]

def sample(y0, p0, reaction_consts_0, step_size=1e-1, maxiter=1000, seed=None):

    floquet_multiplier_threshold = 7e-1
    if seed is None:
        seed = time.time_ns()
        
    key = jax.random.PRNGKey(seed)
    
    y_out = [y0]
    p_out = [p0]
    dp_out = []
    reaction_consts_out = [reaction_consts_0]
    M = compute_monodromy(y0[:, 0], p0, reaction_consts_0).y[KaiODE.n_dim - KaiODE.n_conserve:, -1].reshape((KaiODE.n_dim - KaiODE.n_conserve, KaiODE.n_dim - KaiODE.n_conserve), order="F")
    floquet_multipliers = np.linalg.eigvals(M)
    large_multipliers = floquet_multipliers[np.abs(floquet_multipliers) > floquet_multiplier_threshold]
    
    if large_multipliers.shape[0] > 1:
        LL = -100 * (floquet_multiplier_threshold - np.max(np.abs(large_multipliers)[np.argsort(np.abs(large_multipliers) - 1)][1:]))**2
    else:
        LL = 0
    
    LL -= 100 * (p0 - 1)**2
    
    max_amplitude_species = np.argmax(np.max(y_out[-1], axis=1) - np.min(y_out[-1], axis=1))
    _, dp = compute_sensitivity_boundary(y0[:, 0], y0[:, -1], p0, reaction_consts_0, max_amplitude_species, M=M)
    dp = dp * reaction_consts_0
    dp_out.append(dp)
    
    i = 0

    while i < maxiter:
        
        i += 1
        print("iteration %d"%(i))
        
        key, subkey = jax.random.split(key)
        u, s, vh = np.linalg.svd(dp_out[-1])
        randn = jax.random.normal(subkey, shape=(reaction_consts_0.shape[0],))
        s_full = np.zeros_like(reaction_consts_0).at[:s.shape[0]].set(s)
        sigma = step_size / (1 + s_full)
        step = (vh.T * (randn * sigma)).sum(axis=1)
        proposal_factor_f = -np.sum(randn**2) - np.pi * np.prod(sigma)
        
        reaction_consts_propose = np.exp(np.log(reaction_consts_out[-1]) + step)
        model = KaiODE(reaction_consts_propose)
        max_amplitude_species = np.argmax(np.max(y_out[-1], axis=1) - np.min(y_out[-1], axis=1))
        
        @jax.jit
        def phase_condition(t, y, p):
            return np.array([model.f_red(t[0], y[:KaiODE.n_dim - KaiODE.n_conserve])[max_amplitude_species]])
        
        @jax.jit
        def f(t, y, p):
            return p[0] * model.f_red(t, y)
        
        solver = colloc(f, phase_condition, y_out[-1], np.array([p_out[-1]]))
        solver.solve()
        
        if not solver.success:
            y_out.append(y_out[-1])
            p_out.append(p_out[-1])
            reaction_consts_out.append(reaction_consts_out[-1])
            dp_out.append(dp_out[-1])
            print("Accept: %s"%(False))
            continue
            
        M_propose = compute_monodromy(solver.y[:, 0], solver.p[0], reaction_consts_propose).y[KaiODE.n_dim - KaiODE.n_conserve:, -1].reshape(
            (KaiODE.n_dim - KaiODE.n_conserve, KaiODE.n_dim - KaiODE.n_conserve), order="F")
        floquet_multipliers_propose = np.linalg.eigvals(M_propose)
        large_multipliers_propose = floquet_multipliers_propose[np.abs(floquet_multipliers_propose) > floquet_multiplier_threshold]
        
        if large_multipliers_propose.shape[0] > 1:
            LL_propose = -100 * (floquet_multiplier_threshold - np.max(np.abs(large_multipliers_propose)[np.argsort(np.abs(large_multipliers_propose) - 1)][1:]))**2
        else:
            LL_propose = 0

        LL_propose -= 100 * (solver.p[0] - 1)**2 
            
        _, dp_propose = compute_sensitivity_boundary(solver.y[:, 0], solver.y[:, -1], solver.p[0], reaction_consts_propose, max_amplitude_species, M=M_propose)
        u, s, vh = np.linalg.svd(dp_propose)
        s_full = np.zeros_like(reaction_consts_0).at[:s.shape[0]].set(s)
        sigma = step_size / (1 + s_full)
        proposal_factor_r = -np.sum((-vh@step / sigma)**2) - np.pi * np.prod(sigma)
        
        acceptance_ratio = LL_propose - LL + proposal_factor_r - proposal_factor_f
        
        key, subkey = jax.random.split(key)
        accept = np.log(jax.random.uniform(subkey)) < acceptance_ratio
        print("Accept: %s"%(accept))
        print("likelihood: %.5f"%(LL), "proposal likelihood: %.5f"%(LL_propose), "forward proposal factor: %.5f"%(proposal_factor_f), "reverse proposal factor: %.5f"%(proposal_factor_r))
        
        if accept:
            
            y_out.append(solver.y)
            p_out.append(solver.p[0])
            reaction_consts_out.append(reaction_consts_propose)
            dp_out.append(dp_propose)
            LL = LL_propose
                    
        print("period: %.5f"%(p_out[-1]), flush=True)

    return np.array(y_out), np.array(p_out), np.array(dp_out), np.array(reaction_consts_out)
