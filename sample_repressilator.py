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


i = 8
dt = 1e-2
prng_key = np.load("repressilator_lc_key%d.npy"%(i - 1))[-1]
friction = 1e-1

n_mesh_intervals = 60
n_points = n_mesh_intervals * util.gauss_points_4.size + 1

x = np.load("repressilator_lc%d.npy"%(i - 1))[-1]
q0 = x[:model.Repressilator_log.n_par + model.Repressilator_log.n_dim * n_points + 1]
p0 = x[q0.size:2 * q0.size] 
mesh_points = x[-n_mesh_intervals - 1:]
args = (model.Repressilator_log(q0[:model.Repressilator_log.n_par]), mesh_points)

potential = defining_systems.repressilator_log_bvp_potential
resid = defining_systems.periodic_bvp_colloc_resid
jac = defining_systems.periodic_bvp_colloc_jac
n_constraints = jac(q0, *args).shape[0]
l0 = x[2 * q0.size:2 * q0.size + n_constraints]

n_steps = 1000000
thin = 100

traj_rp_lc, key_lc = lgvn.gBAOAB(q0, p0, l0, dt, friction, n_steps, thin, prng_key, potential, resid, jac,
                                 A=lgvn.rattle_drift_bvp_mm, nlsol=nonlinear_solver.quasi_newton_bvp_symm_broyden, linsol=linear_solver.lq_ortho_proj_bvp,
                                 max_newton_iter=20, constraint_tol=1e-8, args=args)

np.save("repressilator_lc%d.npy"%(i), traj_rp_lc)
np.save("repressilator_lc_key%d.npy"%(i), key_lc)
