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

# def compute_monodromy(y0, period, reaction_consts, a0=0.6, c0=3.5, ATPfrac=1.0):
    
#     model = KaiODE(reaction_consts, a0=a0, c0=c0, ATPfrac=ATPfrac)
#     yM_0 = np.concatenate([y0, np.identity(y0.shape[0]).ravel(order="F")])
#     traj = scipy.integrate.solve_ivp(fun=f_M, jac=jac_M, t_span=(0, period), y0=yM_0, method="BDF", args=(model,), atol=1e-6, rtol=1e-4)
    
#     return traj

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

@jax.jit
def elim1(solver, J=None):

    if J == None:
        J = solver._jac(solver.y.ravel(order="F"), solver.p)
    
    interval_width = solver.n_colloc_point * solver.n_dim
    block_size = (interval_width + solver.n_dim + solver.n_par) * interval_width
    bc_block_size = (2 * solver.n_dim**2  + solver.n_par * solver.n_dim)
    par_eq = J.data[-solver.n * solver.n_par:].reshape((solver.n_par, solver.n))
    
    def loop_body(carry, _):

        i, data, par_eq = carry
        block_start = i * block_size
        block = jax.lax.dynamic_slice(data, (block_start,), (block_size,)).reshape((interval_width, interval_width + solver.n_dim + 1))
        
        lu, _, p = jax.lax.linalg.lu(block[:, solver.n_dim:-solver.n_dim - solver.n_par])
        L = np.identity(lu.shape[0]).at[:, :lu.shape[1]].add(np.tril(lu, k=-1))
        U = np.triu(lu)[:interval_width - solver.n_dim]
        block = jax.scipy.linalg.solve_triangular(L, block[p], lower=True)
        data = jax.lax.dynamic_update_slice(data, block.ravel(), (block_start,))
        
        block_par_eq = jax.lax.dynamic_slice(par_eq, (0, i * interval_width), (solver.n_par, interval_width + solver.n_dim))
        elim_coeff = jax.scipy.linalg.solve_triangular(U.T, block_par_eq[:, solver.n_dim:-solver.n_dim].T, lower=True)
        par_eq = jax.lax.dynamic_update_slice(par_eq, block_par_eq - elim_coeff.T@block[:interval_width - solver.n_dim, :solver.n_par], (0, i * interval_width))
        par_eq = jax.lax.dynamic_update_slice(par_eq, par_eq[:, -solver.n_par]\
                                              - elim_coeff.T@block[:interval_width - solver.n_dim, solver.n_par:], (0, solver.n_coeff))
        
        return (i + 1, data, par_eq), elim_coeff
    
    _, data, par_eq = jax.lax.scan(f=loop_body, init=(0, J.data, par_eq), xs=None, length=solver.n_mesh_point)[0]
    data = data.at[-solver.n_par * solver.n:].set(par_eq.ravel())
    
    J.data = data
    return J

@jax.jit
def elim2(solver, J):

    interval_width = solver.n_colloc_point * solver.n_dim
    block_size = (interval_width + solver.n_dim + solver.n_par) * interval_width
    offset = (interval_width + solver.n_dim + solver.n_par) * solver.n_dim
    ind_base = np.mgrid[interval_width - solver.n_dim:interval_width, :solver.n_dim].reshape((2, solver.n_dim * solver.n_dim)).T
    par_eq = J.data[-solver.n * solver.n_par:].reshape((solver.n_par, solver.n))
    
    block = J.data[block_size - offset:block_size].reshape((solver.n_dim, interval_width + solver.n_dim + solver.n_par))
    lu, _, p = jax.lax.linalg.lu(block[:, -solver.n_dim - solver.n_par:-solver.n_par])
    L = np.tril(lu, k=-1).at[np.diag_indices(lu.shape[0])].set(1)
    U = np.triu(lu)
    block = jax.scipy.linalg.solve_triangular(L, block[p], lower=True)
    J.data = J.data.at[block_size - offset:block_size].set(block.ravel())
    block_left = block[:, :solver.n_dim]
    block_right = block[:, -solver.n_par:]
    
    def loop_body(carry, _):
        
        i, data, U_prev, block_left, block_right, par_eq = carry
        block_start = i * block_size
        
        block_par_eq = jax.lax.dynamic_slice(par_eq, (0, i * interval_width + solver.n_dim), (solver.n_par, solver.n_dim))
        elim_coeff = jax.scipy.linalg.solve_triangular(U_prev.T, block_par_eq.T, lower=True)
        par_eq = jax.lax.dynamic_update_slice(par_eq, block_par_eq - elim_coeff.T@U_prev, (0, i * interval_width + solver.n_dim))
        par_eq = jax.lax.dynamic_update_slice(par_eq, jax.lax.dynamic_slice(par_eq, (0, 0), (solver.n_par, solver.n_dim))\
                                              - elim_coeff.T@block_left, (0, 0))
        par_eq = par_eq.at[:, -solver.n_par:].add(-elim_coeff.T@block_right)
        
        block = jax.lax.dynamic_slice(data, (block_start + block_size - offset,), (offset,)).reshape((solver.n_dim, interval_width + solver.n_dim + solver.n_par))
        elim_coeff = jax.scipy.linalg.solve_triangular(U_prev.T, block[:, :solver.n_dim].T, lower=True)
        block = block.at[:, :solver.n_dim].add(-elim_coeff.T@U_prev)
        block = block.at[:, -solver.n_par:].add(-elim_coeff.T@block_right)
        block_left = -elim_coeff.T@block_left
        
        lu, _, p = jax.lax.linalg.lu(block[:, -solver.n_dim - solver.n_par:-solver.n_par])
        L = np.tril(lu, k=-1).at[np.diag_indices(lu.shape[0])].set(1)
        U = np.triu(lu)
        block = jax.scipy.linalg.solve_triangular(L, block[p], lower=True)
        data = jax.lax.dynamic_update_slice(data, block.ravel(), (block_start + block_size - offset,))
        block_left = jax.scipy.linalg.solve_triangular(L, block_left[p], lower=True)
        block_right = block[:, -solver.n_par:]
        
        return (i + 1, data, U, block_left, block_right, par_eq), (block_left, ind_base.at[:, 0].add(i * interval_width))
        
    carry, mat = jax.lax.scan(f=loop_body, init=(1, J.data, U, block_left, block_right, par_eq), xs=None, length=solver.n_mesh_point - 2)
    i, data, U_prev, block_left, block_right, par_eq = carry
    block_start = i * block_size
    
    block_par_eq = jax.lax.dynamic_slice(par_eq, (0, i * interval_width + solver.n_dim), (solver.n_par, solver.n_dim))
    elim_coeff = jax.scipy.linalg.solve_triangular(U_prev.T, block_par_eq.T, lower=True)
    par_eq = jax.lax.dynamic_update_slice(par_eq, block_par_eq - elim_coeff.T@U_prev, (0, i * interval_width + solver.n_dim))
    par_eq = jax.lax.dynamic_update_slice(par_eq, jax.lax.dynamic_slice(par_eq, (0, 0), (solver.n_par, solver.n_dim))\
                                          - elim_coeff.T@block_left, (0, 0))
    par_eq = par_eq.at[:, -solver.n_par:].add(-elim_coeff.T@block_right)
    
    block = jax.lax.dynamic_slice(data, (block_start + block_size - offset,), (offset,)).reshape((solver.n_dim, interval_width + solver.n_dim + solver.n_par))
    elim_coeff = jax.scipy.linalg.solve_triangular(U_prev.T, block[:, :solver.n_dim].T, lower=True)
    block = block.at[:, :solver.n_dim].add(-elim_coeff.T@U_prev)
    block = block.at[:, -solver.n_par:].add(-elim_coeff.T@block_right)
    block_left = -elim_coeff.T@block_left
    data = jax.lax.dynamic_update_slice(data, block.ravel(), (block_start + block_size - offset,))
    
    data = data.at[-solver.n_par * solver.n:].set(par_eq.ravel())
    data = np.concatenate((data, mat[0].ravel(), block_left.ravel()))
    indices = np.vstack((J.indices, np.vstack(mat[1]), ind_base.at[:, 0].add(i * interval_width)))
    
    return jax.experimental.sparse.BCOO((data, indices), shape=(solver.n, solver.n)).sort_indices()

@jax.jit
def compute_monodromy(solver):
    
    J = solver._jac()
    J = elim1(solver, J)
    J = elim2(solver, J)
    
    interval_width = solver.n_colloc_point * solver.n_dim
    block_size = (interval_width + solver.n_dim + solver.n_par) * interval_width
    i = block_size * (solver.n_mesh_point - 1) + solver.n_dim**2 * (solver.n_mesh_point - 2)\
      + (block_size - solver.n_dim * (interval_width + solver.n_dim + solver.n_par))
    
    rows_A = J.data[i:i + solver.n_dim**2 + solver.n_dim*(interval_width + solver.n_dim + solver.n_par)].reshape((solver.n_dim, interval_width + 2 * solver.n_dim + solver.n_par))
    A0 = rows_A[:, :solver.n_dim]
    A1 = rows_A[:, -solver.n_dim - solver.n_par:-solver.n_par]
    
    return np.linalg.solve(-A1, A0)

def compute_LL(solver, model, floquet_multiplier_threshold=0.8):

    LL = 0
    LL -= 300 * (solver.p[0] - 1)**2
    LL -= 1 / (100 * np.linalg.norm(model.f_red(0, solver.y[:, -1], reaction_consts=solver.args[0]))**2)

    M = compute_monodromy(solver)
    floquet_multipliers, evecs = np.linalg.eig(M)
    large_multipliers = floquet_multipliers[np.abs(floquet_multipliers) > floquet_multiplier_threshold]
    large_evecs = evecs[:, np.abs(floquet_multipliers) > floquet_multiplier_threshold]
    abs_multipliers = np.abs(large_multipliers)
    sorted_indices = np.argsort(abs_multipliers)

    if abs_multipliers[sorted_indices[-1]] > 1:
        LL -= 100 * (floquet_multiplier_threshold - abs_multipliers[sorted_indices[-1]])**2
    elif abs_multipliers.shape[0] > 1:
        
        mask = np.max(np.min(solver.y[:, 0] / np.abs(large_evecs).T, axis=1) * np.abs(large_evecs), axis=0) > 1e-4

        if np.sum(mask) > 1:

            masked = abs_multipliers[mask]
            masked = np.sort(masked)
            LL -= 100 * (floquet_multiplier_threshold - masked[-2])**2

    return LL

def continuation(solver, p_stop, step_size=1e-2, min_step_size=1e-3, maxiter=1000):

    i = 0
    y_out = [solver.y]
    p_out = [solver.p]
    direction_rhs = numpy.zeros(solver.y.size + solver.p.size)
    direction_rhs[-1] = 1

    solver._superLU()
    direction = solver.jac_LU.solve(direction_rhs)

    solver.y = solver.y + direction[:solver.n_coeff].reshape((solver.n_dim, solver.n_coeff // solver.n_dim), order="F") * step_size
    solver.p = solver.p + direction[solver.n_coeff:] * step_size

    args = list(solver.args)
    args[0] = direction
    args[1] = solver.y.ravel(order="F")
    args[2] = solver.p
    solver.args = tuple(args)

    while p_out[-1][-1] < p_stop and i < maxiter:

        i += 1
        solver.solve()

        if solver.success:
            y_out.append(solver.y)
            p_out.append(solver.p)

            if solver.n_iter == 1:
                step_size *= 2

            direction = solver.jac_LU.solve(direction_rhs)
            
            solver.y = solver.y + direction[:solver.n_coeff].reshape((solver.n_dim, solver.n_coeff // solver.n_dim), order="F") * step_size
            solver.p = solver.p + direction[solver.n_coeff:] * step_size

            args = list(solver.args)
            args[0] = direction
            args[1] = solver.y.ravel(order="F")
            args[2] = solver.p
            solver.args = tuple(args)

        elif step_size > min_step_size:
            step_size /= 2
        else:
            raise RuntimeError("Continuation step size decreased below threshold of %s"%(min_step_size))

    if i >= maxiter and p_out[-1][-1] < p_stop:
        raise RuntimeError("Continuation iterations exceeded %d"%(maxiter))

    return np.array(y_out), np.array(p_out)


def sample(y0, p0, reaction_consts_0, step_size=1e-1, maxiter=1000, seed=None):

    floquet_multiplier_threshold = 7e-1
    if seed is None:
        seed = time.time_ns()
        
    key = jax.random.PRNGKey(seed)
    
    y_out = [y0]
    p_out = [p0]
    dp_out = []
    reaction_consts_out = [reaction_consts_0]
    
    max_amplitude_species = np.argmax(np.max(y_out[-1], axis=1) - np.min(y_out[-1], axis=1))
    model = KaiODE(reaction_consts_0)

    @jax.jit
    def phase_condition(t, y, p, reaction_consts, a0, max_amplitude_species):
        return np.array([model.f_red(t[0], y[:KaiODE.n_dim - KaiODE.n_conserve], reaction_consts, a0)[max_amplitude_species]])
    
    @jax.jit
    def f(t, y, p, reaction_consts, a0, max_amplitude_species):
        return p[0] * model.f_red(t, y, reaction_consts, a0)


    solver = colloc(f, phase_condition, y_out[-1], np.array([p_out[-1]]), args=(model.reaction_consts, model.a0, max_amplitude_species))
    LL = compute_LL(solver, model)

    #_, dp = compute_sensitivity_boundary(y0[:, 0], y0[:, -1], p0, reaction_consts_0, max_amplitude_species, M=M)
    #dp = dp * reaction_consts_0
    #dp_out.append(dp)
    
    i = 0

    accepted = 0
    rejected = 0
    failed = 0

    while i < maxiter:
        
        i += 1

        print("iteration %d"%(i), flush=True)
        print("Log likelihood %.5f"%(LL), flush=True)
        print("accepted:%d rejected:%d failed:%d"%(accepted, rejected, failed), flush=True)
        print("period: %.5f"%(p_out[-1]), flush=True)
        
        key, subkey = jax.random.split(key)
        #u, s, vh = np.linalg.svd(dp_out[-1])
        randn = jax.random.normal(subkey, shape=(reaction_consts_0.shape[0],))
        #s_full = np.zeros_like(reaction_consts_0).at[:s.shape[0]].set(s)
        #sigma = step_size / (1 + s_full)
        #step = (vh.T * (randn * sigma)).sum(axis=1)
        #proposal_factor_f = -np.sum(randn**2) - np.pi * np.prod(sigma)
        step = randn * step_size
        
        reaction_consts_propose = np.exp(np.log(reaction_consts_out[-1]) + step)
        max_amplitude_species = np.argmax(np.max(y_out[-1], axis=1) - np.min(y_out[-1], axis=1))
   
        solver.success = False
        solver.y = y_out[-1]
        solver.p = solver.p.at[0].set(p_out[-1])
        solver.args = (reaction_consts_propose, model.a0, max_amplitude_species)
        solver.solve(atol=1e-9)
        
        if not solver.success:
            y_out.append(y_out[-1])
            p_out.append(p_out[-1])
            reaction_consts_out.append(reaction_consts_out[-1])
            failed += 1
            #dp_out.append(dp_out[-1])
            #print("Accept: %s"%(False))
            continue
            
        LL_propose = compute_LL(solver, model)
            
        #_, dp_propose = compute_sensitivity_boundary(solver.y[:, 0], solver.y[:, -1], solver.p[0], reaction_consts_propose, max_amplitude_species, M=M_propose)
        #u, s, vh = np.linalg.svd(dp_propose)
        #s_full = np.zeros_like(reaction_consts_0).at[:s.shape[0]].set(s)
        #sigma = step_size / (1 + s_full)
        #proposal_factor_r = -np.sum((-vh@step / sigma)**2) - np.pi * np.prod(sigma)
        
        proposal_factor_r = 0
        proposal_factor_f = 0

        acceptance_ratio = LL_propose - LL + proposal_factor_r - proposal_factor_f
        
        key, subkey = jax.random.split(key)
        accept = np.log(jax.random.uniform(subkey)) < acceptance_ratio
        #print("Accept: %s"%(accept))
        #print("likelihood: %.5f"%(LL), "proposal likelihood: %.5f"%(LL_propose), "forward proposal factor: %.5f"%(proposal_factor_f), "reverse proposal factor: %.5f"%(proposal_factor_r))
        
        if accept:
            
            y_out.append(solver.y)
            p_out.append(solver.p[0])
            reaction_consts_out.append(reaction_consts_propose)
            #dp_out.append(dp_propose)
            LL = LL_propose
            accepted += 1

        else:

            y_out.append(y_out[-1])
            p_out.append(p_out[-1])
            reaction_consts_out.append(reaction_consts_out[-1])
            rejected += 1

    return np.array(y_out), np.array(p_out), np.array(dp_out), np.array(reaction_consts_out)
