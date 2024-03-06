import numpy
import jax
import jax.numpy as np
import model
import time
import geodesic_langevin as lgvn
from functools import partial
import util
import nonlinear_solver
import linear_solver
import defining_systems
import scipy
import scipy.integrate
import argparse
jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument("-process", type=int, default=0)
parser.add_argument("-iter", type=int, required=True)
parser.add_argument("-n_dim", type=int, default=3)
parser.add_argument("-n_steps", type=int, default=1000000)
parser.add_argument("-thin", type=int, default=1)
argp = parser.parse_args()

data = np.load("repressilator_data.npy")
t_eval = data[:, 0]
data_period_average = data[:, 1]
n_mesh_intervals = 60
colloc_points_unshifted = util.gauss_points_4

@partial(jax.jit, static_argnames=("n_mesh_intervals"))
def repressilator_log_bvp_mm_potential(q, ode_model, *args, n_mesh_intervals=60):
    
    E = 0

    n_points = (n_mesh_intervals * util.gauss_points_4.size + 1)
    k = q[:ode_model.n_par]
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape(ode_model.n_dim, n_points, order="F")
    arclength = np.linalg.norm(y[:, 1:] - y[:, :-1], axis=0).sum()
    min_arclength = 0.3

    E += 100 * np.where(k[-ode_model.n_dim:] < 0, k[-ode_model.n_dim:]**2, 0.).sum()
    E += 100 * np.where(k[-ode_model.n_dim:] > 10, (k[-ode_model.n_dim:] - 10)**2, 0.).sum()
    E += 100 * np.where(k[:-ode_model.n_dim] > 5, (k[:-ode_model.n_dim] - 5)**2, 0,).sum()
    E += 100 * np.where(k[:-ode_model.n_dim] < -5, (k[:-ode_model.n_dim] + 5)**2, 0,).sum()
    E += np.where(arclength < min_arclength, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4, 0)

    return E

@partial(jax.jit, static_argnames=("n_mesh_intervals"))
def repressilator_log_bvp_mm_inference_potential(q, ode_model, *args, n_mesh_intervals=60):

    std = 5e-2
    n_points = (n_mesh_intervals * util.gauss_points_4.size + 1)
    k = q[:ode_model.n_par]
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape(ode_model.n_dim, n_points, order="F")
    mesh_points = q[ode_model.n_par + n_points * ode_model.n_dim:ode_model.n_par + n_points * ode_model.n_dim + n_mesh_intervals - 2]
    mesh_points = np.pad(mesh_points, (1, 1), constant_values=(0, 1))
    period = q[-1]
    period_ref = t_eval[-1]
    t = t_eval / t_eval[-1]
    y_interp = util.interpolate(y, mesh_points, t)
    E = repressilator_log_bvp_mm_potential(q, ode_model, *args, n_mesh_intervals=n_mesh_intervals)
    E += jax.scipy.integrate.trapezoid((data_period_average - np.exp(y_interp[0]))**2 / (2 * std**2), x=t)
    E += (period - period_ref)**2  / (2 * std**2)
    return E

@partial(jax.jit, static_argnames=("n_mesh_intervals"))
def repressilator_log_bvp_mm_inference_potential_1(q, ode_model, colloc_points_unshifted=util.gauss_points_4, *args, n_mesh_intervals=60, **kwargs):

    std = 5e-2
    n_points = (n_mesh_intervals * colloc_points_unshifted.size + 1)
    k = q[:ode_model.n_par]
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape(ode_model.n_dim, n_points, order="F")
    mesh_points = np.pad(np.cumsum(q[-n_mesh_intervals:]), (1, 0))
    period = mesh_points[-1]
    mesh_points = mesh_points / period
    period_ref = t_eval[-1]
    t = t_eval / t_eval[-1]
    y_interp = util.interpolate(y, mesh_points, t)
    E = repressilator_log_bvp_mm_potential(q, ode_model, *args, n_mesh_intervals)
    E += jax.scipy.integrate.trapezoid((data_period_average - np.exp(y_interp[0]))**2 / (2 * std**2), x=t)
    E += (period - period_ref)**2  / (2 * std**2)
    return E

@jax.jit
def repressilator_log_bvp_inference_potential(q, ode_model, mesh_points=np.linspace(0, 1, 61)):

    std = 5e-2
    n_mesh_intervals = mesh_points.size - 1
    n_points = (n_mesh_intervals * util.gauss_points_4.size + 1)
    k = q[:ode_model.n_par]
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape(ode_model.n_dim, n_points, order="F")
    period = q[-1]
    period_ref = t_eval[-1]
    t = t_eval / t_eval[-1]
    y_interp = util.interpolate(y, mesh_points, t)
    E = defining_systems.repressilator_log_bvp_potential(q, ode_model, mesh_points)
    E += jax.scipy.integrate.trapezoid((data_period_average - np.exp(y_interp[0]))**2 / (2 * std**2), x=t)
    E += (period - period_ref)**2  / (2 * std**2)
    return E

n_dim = argp.n_dim
dt = 1e-1
friction = 1e-1

n_points = n_mesh_intervals * colloc_points_unshifted.size + 1

rp = model.Repressilator_log_n(n_dim=n_dim)
x = np.load("repressilator_%d_lc_infer_%d_%d.npy"%(n_dim, argp.iter - 1, argp.process))[-1]
q0 = x[:rp.n_par + rp.n_dim * n_points + n_mesh_intervals]
p0 = x[q0.size:2 * q0.size]
prng_key = x[-1:].view(np.uint32)

potential = partial(repressilator_log_bvp_mm_inference_potential_1, n_mesh_intervals=n_mesh_intervals)
resid = partial(defining_systems.bvp_mm_colloc_resid, n_mesh_intervals=n_mesh_intervals)
jac = partial(defining_systems.bvp_mm_colloc_jac, n_mesh_intervals=n_mesh_intervals)
n_constraints = resid(q0, ode_model=rp, colloc_points_unshifted=colloc_points_unshifted).shape[0]
l0 = x[2 * q0.size:2 * q0.size + n_constraints]

n_steps = argp.n_steps
thin = argp.thin

traj_rp_lc = lgvn.sample((q0, p0, l0, None, None, prng_key), dt=1e-1, n_steps=n_steps, thin=thin, friction=1e-1,
                                 potential=potential, stepper=lgvn.gOBABO, constraint=resid, jac_constraint=jac, linsol=linear_solver.lq_ortho_proj_bvp, 
                                 nlsol=nonlinear_solver.quasi_newton_bvp_symm_broyden, max_newton_iter=100, metropolize=True, reversibility_tol=1e-6, ode_model=rp, 
                                 colloc_points_unshifted=util.gauss_points_4, print_acceptance=True)

np.save("repressilator_%d_lc_infer_%d_%d.npy"%(n_dim, argp.iter, argp.process), traj_rp_lc)
