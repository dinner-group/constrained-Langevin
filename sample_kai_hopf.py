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

i = 3
dt = 7e-2
prng_key = np.load("kaiabc_hb_key%d.npy"%(i - 1))[-1]
friction = 1e-2

x = np.load("kaiabc_hb%d.npy"%(i - 1))[-1]
q0 = x[:model.KaiABC_nondim.n_par + 3 * (model.KaiABC_nondim.n_dim + model.KaiABC_nondim.conservation_law.shape[0]) + 1]
p0 = x[q0.size:2 * q0.size] 

potential = defining_systems.kai_sna_hb_potential
resid = defining_systems.fully_extended_hopf_kai_dae
n_constraints = resid(q0).size
l0 = x[2 * q0.size:2 * q0.size + n_constraints]

n_steps = 10000000
thin = 100

traj, key = lgvn.gOBABO(q0, p0, l0, dt, friction, n_steps, thin, prng_key, potential, resid, nlsol=nonlinear_solver.quasi_newton_rattle_symm_1, max_newton_iter=100, constraint_tol=1e-9, metropolize=True, reversibility_tol=1e-9)

np.save("kaiabc_hb%d.npy"%(i), traj)
np.save("kaiabc_hb_key%d.npy"%(i), key)
