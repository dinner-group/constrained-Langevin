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
import os
jax.config.update("jax_enable_x64", True)

dt = 1e-1
n_steps = 500000
thin = 10
friction = 1e-2
seed = time.time_ns()
prng_key = jax.random.PRNGKey(seed)

traj0 = np.load("morris_lecar_hb_0.npy")
bounds = np.load("morris_lecar_bounds_nondim.npy")

prng_key, subkey = jax.random.split(prng_key)
index = jax.random.randint(subkey, (), 0, traj0.shape[0])
ml = model.Morris_Lecar_nondim(np.zeros(model.Morris_Lecar_nondim.n_par))
q0 = traj0[index, :ml.n_par + 2 * ml.n_dim + 1]
q0 = nonlinear_solver.gauss_newton(q0, defining_systems.fully_extended_hopf_2n_log, args=(ml,))[0]
prng_key, subkey = jax.random.split(prng_key)
p0 = jax.random.normal(prng_key, q0.shape)
J = jax.jacfwd(defining_systems.fully_extended_hopf_2n_log)(q0, ml)
p0 = linear_solver.lq_ortho_proj(J, p0)[0]
l0 = np.zeros(J.shape[0])
traj, _ = lgvn.gOBABO(q0, p0, l0, dt, friction, n_steps, thin, prng_key, defining_systems.morris_lecar_hb_potential, defining_systems.fully_extended_hopf_2n_log, max_newton_iter=100, constraint_tol=1e-9,
                      nlsol=nonlinear_solver.quasi_newton_rattle_symm_1, metropolize=True, reversibility_tol=np.inf, args=(ml, bounds))

i_fail = np.searchsorted(np.cumsum((traj[1:, q0.size - 1] - traj[:-1, q0.size - 1])[::-1]), 1)
np.save("morris_lecar_hb_%d.npy"%(seed), traj[:-i_fail])
