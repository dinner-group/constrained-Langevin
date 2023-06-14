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
jax.config.update("jax_enable_x64", True)

data = np.load("repressilator_data.npy")
t_eval = data[:, 0]
data_period_average = data[:, 1]

@jax.jit
def repressilator_log_bvp_inference_potential(q, ode_model, mesh_points=np.linspace(0, 1, 61)):

    std = 5e-2
    n_mesh_intervals = mesh_points.size - 1
    n_points = (n_mesh_intervals * util.gauss_points.size + 1)
    k = q[:ode_model.n_par]
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape(ode_model.n_dim, n_points, order="F")
    period = q[-1]
    period_ref = t_eval[-1]
    t = t_eval / t_eval[-1]
    y_interp = util.interpolate(y, mesh_points, t)
    E = defining_systems.repressilator_log_bvp_potential(q, ode_model, mesh_points)
    E += np.trapz((data_period_average - np.exp(y_interp[0]))**2 / (2 * std**2), x=t)
    E += (period - period_ref)**2  / (2 * std**2)
    return E

i = 1
n_dim = 7
dt = 1e-2
prng_key = np.load("repressilator_%d_lc_infer_key%d.npy"%(n_dim, i - 1))[-1]
friction = 1e-1

n_mesh_intervals = 60
n_points = n_mesh_intervals * util.gauss_points.size + 1

rp = model.Repressilator_log_n(n_dim=n_dim)
x = np.load("repressilator_%d_lc_infer%d.npy"%(n_dim, i - 1))[-1]
q0 = x[:rp.n_par + rp.n_dim.n_dim * n_points + 1]
p0 = x[q0.size:2 * q0.size] 
mesh_points = x[-n_mesh_intervals - 1:]
args = (rp, mesh_points)

potential = repressilator_log_bvp_inference_potential
resid = defining_systems.periodic_bvp_colloc_resid
jac = defining_systems.periodic_bvp_colloc_jac
n_constraints = jac(q0, *args).shape[0]
l0 = x[2 * q0.size:2 * q0.size + n_constraints]

n_steps = 100000
thin = 100

traj_rp_lc, key_lc = lgvn.gBAOAB(q0, p0, l0, dt, friction, n_steps, thin, prng_key, potential, resid, jac,
                                 A=lgvn.rattle_drift_bvp_mm, nlsol=nonlinear_solver.quasi_newton_bvp_symm_1, linsol=linear_solver.qr_lstsq_rattle_bvp,
                                 max_newton_iter=20, tol=1e-8, args=args)

np.save("repressilator_%d_lc_infer%d.npy"%(n_dim, i), traj_rp_lc)
np.save("repressilator_%d_lc_infer_key%d.npy"%(n_dim, i), key_lc)
