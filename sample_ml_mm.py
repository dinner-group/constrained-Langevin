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
parser.add_argument("-process", type=int, required=True)
parser.add_argument("-iter", type=int, required=True)
argp = parser.parse_args()

@partial(jax.jit, static_argnames=("n_mesh_intervals",))
def morris_lecar_mm_bvp_potential_multi_eqn_shared_k(q, ode_models, colloc_points_unshifted=(util.gauss_points, util.gauss_points), bounds=None, bounds_membrane_voltage=None, n_mesh_intervals=(60, 60)):

    E = 0
    k = q[:ode_models[0].n_par]
    start = ode_models[0].n_par
    min_arclength = 0.3

    for i in range(len(ode_models)):

        n_points = (n_mesh_intervals[i] * colloc_points_unshifted[i].size + 1)
        stop = start + n_points * ode_models[i].n_dim
        y = q[start:stop].reshape(ode_models[i].n_dim, n_points, order="F")
        start = stop + n_mesh_intervals[i]
        arclength = np.linalg.norm(y[:, 1:] - y[:, :-1], axis=0).sum()
        
        E += np.where(arclength < min_arclength, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4, 0)

        if bounds_membrane_voltage is not None:
            E += 100 * (util.smooth_max(y[0]) - bounds_membrane_voltage[1])**2
            E += 100 * (-util.smooth_max(-y[0]) - bounds_membrane_voltage[0])**2

        if bounds is not None:
            E += np.where(k < bounds[:, 0], 100 * (k - bounds[:, 0])**2, 0).sum()
            E += np.where(k > bounds[:, 1], 100 * (k - bounds[:, 1])**2, 0).sum()

        y_smooth = util.weighted_average_periodic_smoothing(y[:, :-1].T)
        E += 10 * np.sum((y_smooth - y[:, :-1].T)**2)

    index = ode_models[0].n_par + (n_mesh_intervals[0] * colloc_points_unshifted[0].size + 1) * ode_models[0].n_dim  + n_mesh_intervals[0] - 1
    period1 = q[index]
    index = index + (n_mesh_intervals[1] * colloc_points_unshifted[1].size + 1) * ode_models[1].n_dim + n_mesh_intervals[1]
    period2 = q[index]
    E += 100 * (period2 / period1 - 2)**2 / 2
    E += np.where(k[6] > k[8], 100 * (k[6] - k[8])**2, 0)

    return E

n_mesh_intervals = 60
colloc_points_unshifted = util.gauss_points

dt = 1e-2
prng_key = np.load("ml_lc_key_%d_%d.npy"%(argp.iter - 1, argp.process))[-1]
friction = 1e-2

n_points = n_mesh_intervals * colloc_points_unshifted.size + 1
x = np.load("ml_lc_%d_%d.npy"%(argp.iter - 1, argp.process))[-1]
#bounds = np.load("morris_lecar_bounds_nondim.npy")
#bounds_membrane_voltage = np.array([-35/84, 50/84])

n_steps = 400000
thin = 100

#ml = model.Morris_Lecar_nondim(par=np.zeros(model.Morris_Lecar_nondim.n_par))
#q0 = x[:ml.n_par + ml.n_dim * n_points + 1 + n_mesh_intervals - 1]
#p0 = x[q0.size:2 * q0.size] 
#args = (ml, colloc_points_unshifted, bounds, bounds_membrane_voltage)
#potential = lambda *args:defining_systems.morris_lecar_mm_bvp_potential(*args, n_mesh_intervals=n_mesh_intervals)
#resid = lambda *args:defining_systems.periodic_bvp_mm_colloc_resid(*args, n_mesh_intervals=n_mesh_intervals)
#jac = lambda *args:defining_systems.periodic_bvp_mm_colloc_jac(*args, n_mesh_intervals=n_mesh_intervals)
#n_constraints = resid(q0, *args).size
#l0 = x[2 * q0.size:2 * q0.size + n_constraints]
#traj_ml_lc, key_lc = lgvn.gOBABO(q0, p0, l0, dt, friction, n_steps, thin, prng_key, potential, resid, jac, nlsol=nonlinear_solver.quasi_newton_bvp_symm_broyden, linsol=linear_solver.qr_lstsq_rattle_bvp,
#                                  max_newton_iter=100, constraint_tol=1e-9, args=args, metropolize=True, reversibility_tol=1e-6)

ml1 = model.Morris_Lecar_nondim(par=np.zeros(model.Morris_Lecar_nondim.n_par))
ml2 = model.Morris_Lecar_nondim(ml1.par, par_scale=np.ones_like(ml1.par).at[3].multiply(1/3))
q0 = x[:ml1.n_par + 2 * (ml1.n_dim * n_points + n_mesh_intervals)]
p0 = x[q0.size:2 * q0.size]
args = ((ml1, ml2), (colloc_points_unshifted, colloc_points_unshifted))
potential = lambda *args:morris_lecar_mm_bvp_potential_multi_eqn_shared_k(*args, n_mesh_intervals=(n_mesh_intervals, n_mesh_intervals))
resid = lambda *args:defining_systems.periodic_bvp_mm_colloc_resid_multi_eqn_shared_k(*args, n_mesh_intervals=(n_mesh_intervals, n_mesh_intervals))
jac = lambda *args:defining_systems.periodic_bvp_mm_colloc_jac_multi_eqn_shared_k(*args, n_mesh_intervals=(n_mesh_intervals, n_mesh_intervals))
n_constraints = resid(q0, *args).size
l0 = x[2 * q0.size:2 * q0.size + n_constraints]
traj_ml_lc, key_lc = lgvn.gOBABO(q0, p0, l0, dt, friction, n_steps, thin, prng_key, potential, resid, jac, nlsol=nonlinear_solver.quasi_newton_bvp_multi_eqn_shared_k_symm_broyden, 
                                 linsol=linear_solver.qr_lstsq_rattle_bvp_multi_eqn_shared_k, max_newton_iter=100, constraint_tol=1e-9, args=args, metropolize=True, reversibility_tol=1e-6)

np.save("ml_lc_%d_%d.npy"%(argp.iter, argp.process), traj_ml_lc)
np.save("ml_lc_key_%d_%d.npy"%(argp.iter, argp.process), key_lc)
