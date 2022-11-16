import numpy
import jax
import jax.numpy as np
import diffrax
from model import *
from collocation import *
import time
from functools import partial
import os
jax.config.update("jax_enable_x64", True)

path = os.path.dirname(__file__)
K = np.array(numpy.loadtxt(path + "/K.txt", dtype=numpy.int64))
S = np.array(numpy.loadtxt(path + "/S.txt", dtype=numpy.int64))

@jax.jit
def elim1(solver, J=None):

    if J is None:
        J = solver._jac(solver.y.ravel(order="F"), solver.p)
    
    interval_width = solver.n_colloc_point * solver.n_dim
    block_length = interval_width + solver.n_dim + solver.n_par
    block_size = block_length * interval_width
    bc_block_size = (2 * solver.n_dim**2  + solver.n_par * solver.n_dim)
    par_eq = J.data[-solver.n * solver.n_par:].reshape((solver.n_par, solver.n))
    
    def loop_body(carry, _):

        i, data, par_eq = carry
        block_start = i * block_size
        block = jax.lax.dynamic_slice(data, (block_start,), (block_size,)).reshape((interval_width, block_length))
        
        lu, _, p = jax.lax.linalg.lu(block[:, solver.n_dim:-solver.n_dim - solver.n_par])
        L = np.identity(lu.shape[0]).at[:, :lu.shape[1]].add(np.tril(lu, k=-1))
        U = np.triu(lu)[:interval_width - solver.n_dim]
        block = block.at[:, :solver.n_dim].set(jax.scipy.linalg.solve_triangular(L, block[p, :solver.n_dim], lower=True))
        block = block.at[:, solver.n_dim:interval_width].set(np.triu(lu))
        block = block.at[:, interval_width:].set(jax.scipy.linalg.solve_triangular(L, block[p, interval_width:], lower=True))
        
        data = jax.lax.dynamic_update_slice(data, block.ravel(), (block_start,))
        
        block_par_eq = jax.lax.dynamic_slice(par_eq, (0, i * interval_width), (solver.n_par, interval_width + solver.n_dim))
        elim_coeff = jax.scipy.linalg.solve_triangular(U.T, block_par_eq[:, solver.n_dim:-solver.n_dim].T, lower=True)
        par_eq = jax.lax.dynamic_update_slice(par_eq, block_par_eq[:, :solver.n_dim] - elim_coeff.T@block[:interval_width - solver.n_dim, :solver.n_dim], (0, i * interval_width))
        par_eq = jax.lax.dynamic_update_slice(par_eq, np.zeros((solver.n_par, interval_width - solver.n_dim)), (0, i * interval_width + solver.n_dim))
        par_eq = jax.lax.dynamic_update_slice(par_eq, block_par_eq[:, -solver.n_dim:]\
                                              - elim_coeff.T@block[:interval_width - solver.n_dim, -solver.n_par - solver.n_dim:- solver.n_par], (0, (i + 1) * interval_width))
        par_eq = jax.lax.dynamic_update_slice(par_eq, par_eq[:, -solver.n_par]\
                                              - elim_coeff.T@block[:interval_width - solver.n_dim, -solver.n_par:], (0, solver.n_coeff))
        
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

@jax.jit
def compute_monodromy_1(y0, period, reaction_consts, a0=0.6, c0=3.5):

    def rhs(t, y, args):
        model = KaiODE(args[:-2])
        return model.f_red(t, y, args[:model.reaction_consts.size], *args[-2:])
        
    term = diffrax.ODETerm(rhs)
    integrator = diffrax.Kvaerno4()
    stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)
        
    return jax.jacfwd(diffrax.diffeqsolve, argnums=5)(term, integrator, 0, solver.p[0], None, solver.y[:, 0], args=np.concatenate([reaction_consts, np.array([a0, c0])]), 
                                                      stepsize_controller=stepsize_controller, max_steps=10000).ys[0]

@partial(jax.jit, static_argnums=(2, 3))
def normalize_direction(v, t, n_dim, n_points):
    norm = newton_cotes_6(t, np.linalg.norm(v[:n_dim * n_points].reshape((n_dim, n_points), order="F"), axis=0)**2)
    norm = np.sqrt(norm + np.sum(v[n_dim * n_points:]**2))
    return v / norm

def update_args(solver, direction=None, y_prev=None, p_prev=None, y_guess=None, p_guess=None, rc_direction=None):
    args = list(solver.args)

    if direction is not None:
        args[0] = direction

    if y_prev is not None:
        args[1] = y_prev.ravel(order="F")

    if p_prev is not None:
        args[2] = p_prev
    
    if y_guess is not None:
        args[3] = y_guess.ravel(order="F")

    if p_guess is not None:
        args[4] = p_guess

    if rc_direction is not None:
        args[6] = rc_direction

    solver.args = tuple(args)

def gauss_newton(solver, maxiter=10, tol=1e-6):
    
    i = 0
    r, scale = solver._resid_and_scale()
    solver.err = np.max(np.abs(r / (1 + scale)))
    direction_rhs = numpy.asanyarray(np.zeros(solver.n).at[-1].set(1))

    while i < maxiter and solver.err > tol:

        i += 1
        r, scale = solver._newton_step()
        direction = solver.jac_LU.solve(direction_rhs)
        direction = normalize_direction(direction, solver.t, solver.n_dim, solver.n_coeff // solver.n_dim)
        update_args(solver, direction, solver.args[1], solver.args[2], solver.y, solver.p)

    solver.n_iter = i
    solver.success = solver.err <= tol

def cont(solver, p_max, p_min=None, step_size=1e-2, min_step_size=1e-4, max_step_size=1, maxiter=1000, termination_condition=None, tol=1e-6):

    i = 0
    y_out = [solver.y]
    p_out = [solver.p]
    direction_rhs = numpy.zeros(solver.y.size + solver.p.size)
    direction_rhs[-1] = 1
        
    def step(solver, direction, step_size):
        solver.y = solver.y + direction[:solver.n_coeff].reshape((solver.n_dim, solver.n_coeff // solver.n_dim), order="F") * step_size
        solver.p = solver.p + direction[solver.n_coeff:] * step_size
    
    solver._superLU()
    direction = solver.jac_LU.solve(direction_rhs)
    direction = normalize_direction(direction, solver.t, solver.n_dim, solver.n_coeff // solver.n_dim)
    step(solver, direction, step_size)
    args_prev = solver.args
    update_args(solver, direction, y_out[-1], p_out[-1], solver.y, solver.p)
    
    while p_out[-1][-1] < p_max  and p_out[-1][-1] > p_min and i < maxiter and (termination_condition is None or not termination_condition(solver)):
        
        i += 1
        
        try:
            gauss_newton(solver, tol=tol)
        except RuntimeError:
            solver.success = False
                    
        if solver.success:
                    
            y_out.append(solver.y)
            p_out.append(solver.p)

            if solver.n_iter <= 4:
                step_size = np.sign(step_size) * np.minimum(np.abs(step_size) * 1.3, max_step_size)

            direction = solver.jac_LU.solve(direction_rhs)
            direction = normalize_direction(direction, solver.t, solver.n_dim, solver.n_coeff // solver.n_dim)
            step(solver, direction, step_size)
            args_prev = solver.args
            update_args(solver, direction, y_out[-1], p_out[-1], solver.y, solver.p)

        elif np.abs(step_size) > min_step_size:
            step_size /= 2            
            solver.y = y_out[-1] + direction[:solver.n_coeff].reshape((solver.n_dim, solver.n_coeff // solver.n_dim), order="F") * step_size
            solver.p = p_out[-1] + direction[solver.n_coeff:] * step_size
            args_prev = solver.args
            update_args(solver, direction, solver.args[1], solver.args[2], solver.y, solver.p)
            
        else:
            warnings.warn("Continuation step size decreased below threshold of %s"%(min_step_size))
            break

    if p_out[-1][-1] > p_max:
        
        dp = p_max - p_out[-1][-1]
        direction = solver.args[0] / solver.args[0][-1]
        solver.y = y_out[-1] + direction[:-solver.n_par].reshape(solver.y.shape, order="F") * dp
        solver.p = p_out[-1] + direction[-solver.n_par:] * dp
        update_args(solver, np.zeros(solver.n).at[-1].set(1), y_out[-1], p_out[-1], solver.y, solver.p)
        args_prev = solver.args

        solver.damped_newton(tol=tol)
        
        if solver.success:
            y_out.append(solver.y)
            p_out.append(solver.p)
            
    # if i >= maxiter and p_out[-1][-1] < p_stop:
    #     raise RuntimeError("Continuation iterations exceeded %d"%(maxiter))

    solver.y = y_out[-1]
    solver.p = p_out[-1]
    solver.args = args_prev

    return np.array(y_out), np.array(p_out)

@jax.jit
def newton_cotes_6(t, y):
    
    weights = 5 * np.array([19, 75, 50, 50, 75, 19]) / 288

    def loop_body(carry, _):

        i = carry
        ti = yi = jax.lax.dynamic_slice(t, (5 * i,), (6,))
        yi = jax.lax.dynamic_slice(y, (5 * i,), (6,))

        return i + 1, np.sum((ti[-1] - ti[0]) / 5 * weights * yi)

    return jax.lax.scan(loop_body, init=0, xs=None, length=t.size // 5)[1].sum()

@jax.jit
def f_a(t, y, p, continuation_direction, y_prev, p_prev, y_guess, p_guess, model):
    return p[0] * model.f_red(t, y, a0=p[1])

@jax.jit
def fp_a(t, y, p, continuation_direction, y_prev, p_prev, y_guess, p_guess, model):
    
    n_dim = model.n_dim - model.n_conserve
    n_points = y.size // (model.n_dim - model.n_conserve)
    continuation_direction = normalize_direction(continuation_direction, t, n_dim, n_points)

    def loop_body(carry, _):
        i = carry
        return i + 1, f_a(t, y_prev.reshape((n_dim, n_points), order="F")[:, i], p_prev, continuation_direction, y_prev, p_prev, y_guess, p_guess, model)

    ydot = jax.lax.scan(loop_body, init=0, xs=None, length=n_points)[1].T

    return np.array([newton_cotes_6(t, np.sum(y.reshape((n_dim, n_points), order="F") * ydot, axis=0)),
        newton_cotes_6(t, np.sum((y - y_guess).reshape((n_dim, n_points), order="F") * continuation_direction[:y.size].reshape((n_dim, n_points), order="F"), axis=0))\
        + (p - p_guess)@continuation_direction[y.size:]])

@jax.jit
def f_rc(t, y, p, continuation_direction, y_prev, p_prev, y_guess, p_guess, model, rc_direction):
    reaction_consts = np.exp(p[1] * rc_direction) * model.reaction_consts
    return p[0] * model.f_red(t, y, reaction_consts=reaction_consts)

@jax.jit
def fp_rc(t, y, p, continuation_direction, y_prev, p_prev, y_guess, p_guess, model, rc_direction):
    
    reaction_consts = np.exp(p[1] * rc_direction) * model.reaction_consts
    n_dim = model.n_dim - model.n_conserve
    n_points = y.size // (model.n_dim - model.n_conserve)
    continuation_direction = normalize_direction(continuation_direction, t, n_dim, n_points)

    def loop_body(carry, _):
        i = carry
        return i + 1, f_rc(t, y_prev.reshape((n_dim, n_points), order="F")[:, i], p_prev, continuation_direction, y_prev, p_prev, y_guess, p_guess, model, rc_direction)

    ydot = jax.lax.scan(loop_body, init=0, xs=None, length=n_points)[1].T    

    return np.array([newton_cotes_6(t, np.sum(y.reshape((n_dim, n_points), order="F") * ydot, axis=0)),
                     newton_cotes_6(t, np.sum((y - y_guess).reshape((n_dim, n_points), order="F") * continuation_direction[:y.size].reshape((n_dim, n_points), order="F"), axis=0))\
                    + (p - p_guess)@continuation_direction[y.size:]])
