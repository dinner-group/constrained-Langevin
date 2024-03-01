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
def morris_lecar_mm_bvp_potential_multi_eqn_shared_k(q, ode_models, colloc_points_unshifted=(util.gauss_points_4, util.gauss_points_4), n_mesh_intervals=(60, 60), *args, **kwargs):

    E = 0
    k = q[:ode_models[0].n_par]
    start = ode_models[0].n_par
    min_arclength = 0.3
    voltage_bound_upper = 2
    voltage_bound_lower = -1
    ind_V = np.array([2, 6, 8, 12])
    ind_Vscale = np.array([7, 9, 10, 13, 14])

    for i in range(len(ode_models)):

        n_points = (n_mesh_intervals[i] * colloc_points_unshifted[i].size + 1)
        stop = start + n_points * ode_models[i].n_dim
        y = q[start:stop].reshape(ode_models[i].n_dim, n_points, order="F")
        start = stop + n_mesh_intervals[i]
        arclength = np.linalg.norm(y[:, 1:] - y[:, :-1], axis=0).sum()

        E += np.where(arclength < min_arclength, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4, 0)
        
        vmax = util.smooth_max(y[0])
        E += np.where(vmax > voltage_bound_upper, 100 * (vmax - voltage_bound_upper)**2, 0)
        vmin = -util.smooth_max(-y[0])
        E += np.where(vmin < voltage_bound_lower, 100 * (vmin - voltage_bound_lower)**2, 0)


        y_smooth = util.weighted_average_periodic_smoothing(y[:, :-1].T)
        E += 10 * np.sum((y_smooth - y[:, :-1].T)**2)

    index = ode_models[0].n_par + (n_mesh_intervals[0] * colloc_points_unshifted[0].size + 1) * ode_models[0].n_dim + n_mesh_intervals[0]
    interval_widths_1 = q[index - n_mesh_intervals[0]:index]
    period1 = interval_widths_1.sum()
    index = index + (n_mesh_intervals[1] * colloc_points_unshifted[1].size + 1) * ode_models[1].n_dim + n_mesh_intervals[1]
    interval_widths_2 = q[index - n_mesh_intervals[1]:index]
    period2 = interval_widths_2.sum()
    E += 100 * (period2 / period1 - 2)**2 / 2
    E += np.where(k[ind_V] < voltage_bound_lower, 100 * (k[ind_V] - voltage_bound_lower)**2, 0).sum()
    E += np.where(k[ind_V] > voltage_bound_upper, 100 * (k[ind_V] - voltage_bound_upper)**2, 0).sum()
    E += np.where(np.log(k[ind_Vscale]**2) < -10, 100 * (np.log(k[ind_Vscale]**2) + 10)**2, 0).sum()
    E += np.where(k[6] > k[8], 1000 * (k[6] - k[8])**2, 0)
    E += np.where(k[5] > k[4], 1000 * ((k[4] - k[5]) / k[4])**2, 0)

    return E

n_mesh_intervals = 120
colloc_points_unshifted = util.gauss_points_4

dt = 5e-3
friction = 1e-1

n_points = n_mesh_intervals * colloc_points_unshifted.size + 1
x = np.load("ml_lc_%d_%d.npy"%(argp.iter - 1, argp.process))[-1]
#bounds = np.load("morris_lecar_bounds_nondim.npy")
#bounds_membrane_voltage = np.array([-35/84, 50/84])

n_steps = 100000
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
#traj_ml_lc, key_lc = lgvn.gOBABO(q0, p0, l0, dt, friction, n_steps, thin, prng_key, potential, resid, jac, nlsol=nonlinear_solver.quasi_newton_bvp_symm_broyden, linsol=linear_solver.qr_ortho_proj_bvp,
#                                  max_newton_iter=100, constraint_tol=1e-9, args=args, metropolize=True, reversibility_tol=1e-6)

ml1 = model.Morris_Lecar_nondim(par=np.zeros(model.Morris_Lecar_nondim.n_par))
ml2 = model.Morris_Lecar_nondim(ml1.par, par_scale=np.ones_like(ml1.par).at[3].multiply(1/3))
q0 = x[:ml1.n_par + 2 * (ml1.n_dim * n_points + n_mesh_intervals)]
p0 = x[q0.size:2 * q0.size]
potential = lambda *args, **kwargs:morris_lecar_mm_bvp_potential_multi_eqn_shared_k(*args, n_mesh_intervals=(n_mesh_intervals, n_mesh_intervals), **kwargs)
resid = lambda *args, **kwargs:defining_systems.bvp_mm_colloc_resid_multi_shared_k(*args, n_mesh_intervals=(n_mesh_intervals, n_mesh_intervals), n_smooth=6, **kwargs)
jac = lambda *args, **kwargs:defining_systems.bvp_mm_colloc_jac_multi_shared_k(*args, n_mesh_intervals=(n_mesh_intervals, n_mesh_intervals), n_smooth=6, **kwargs)
n_constraints = resid(q0, ode_models=(ml1, ml2)).size
l0 = x[2 * q0.size:2 * q0.size + n_constraints]
linsol = linear_solver.qr_ortho_proj_bvp_multi_shared_k_1
#nlsol = nonlinear_solver.gauss_newton_bvp_multi_shared_k_1
prng_key = x[-1:].view(np.uint32)

@partial(jax.jit, static_argnames=("max_iter", "tol", "resid", "jac"))
def nlsol(x, resid, jac_prev, jac, max_iter, tol=1e-9, *args, **kwargs):
    q, success, n_iter = nonlinear_solver.quasi_newton_bvp_multi_shared_k_symm_broyden_1(x, resid, jac_prev, jac, max_iter, tol=1e-9, *args, **kwargs)
    q, success, n_iter = jax.lax.cond(success, lambda:(q, success, n_iter), 
                                      lambda:nonlinear_solver.gauss_newton_bvp_multi_shared_k_1(x, resid, jac_prev, jac, max_iter=20, tol=1e-9, *args, **kwargs))
    return q, success, n_iter

traj_ml_lc = lgvn.sample((q0, p0, l0, None, None, prng_key), dt, n_steps, thin=thin, potential=potential, stepper=lgvn.gEuler_Maruyama, constraint=resid, jac_constraint=jac,
                         linsol=linsol, nlsol=nlsol, max_newton_iter=100, constraint_tol=1e-9, reversibility_tol=np.inf, metropolize=False, ode_models=(ml1, ml2), print_acceptance=True)

np.save("ml_lc_%d_%d.npy"%(argp.iter, argp.process), traj_ml_lc)
