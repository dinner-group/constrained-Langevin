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

obs_mean = np.load("kai_data.npy")
n_mesh_intervals = 240

@jax.jit
def kai_bvp_potential_mm(q, ode_model, colloc_points_unshifted=util.gauss_points):
    
    #n_mesh_intervals = mesh_points.size - 1
    n_points = n_mesh_intervals * colloc_points_unshifted.size + 1
    k = q[:ode_model.n_par]
    y = q[kai.n_par:ode_model.n_par + ode_model.n_dim * n_points].reshape((ode_model.n_dim, n_points), order="F")
    mesh_points = np.pad(q[ode_model.n_par + ode_model.n_dim * n_points:ode_model.n_par + ode_model.n_dim * n_points + n_mesh_intervals - 1], (1, 1), constant_values=(0, 1))
    E = np.where(np.abs(k) > 7, 100 * (np.abs(k) - 7)**2 / 2, 0).sum()
    yfull = np.vstack([1 - ode_model.conservation_law[0, 1:-1]@y, y, ode_model.a0 - ode_model.conservation_law[1, 1:-1]@y])
    y_mean = yfull.mean(axis=1)
    
    pU_logsum = np.log(y_mean[:2].sum())
    E += np.where(pU_logsum < -2, 10 * (pU_logsum + 2)**2 / 2, 0)
    pT_logsum = np.log(y_mean[2:4].sum())
    E += np.where(pT_logsum < -2, 10 * (pT_logsum + 2)**2 / 2, 0)
    pD_logsum = np.log(y_mean[np.array([4, 5, 8, 9, 12, 13])].sum())
    E += np.where(pD_logsum < -2, 10 * (pD_logsum + 2)**2 / 2, 0)
    pS_logsum = np.log(y_mean[np.array([6, 7, 10, 11, 14, 15])].sum())
    E += np.where(pS_logsum < -2, 10 * (pS_logsum + 2)**2 / 2, 0)
    
    min_arclength = 0.3
    arclength = np.linalg.norm(y[:, 1:] - y[:, :-1], axis=0).sum()
    E += np.where(arclength < min_arclength, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4, 0)
    
    std = 1.5e-1 / np.sqrt(7)
    t = np.linspace(0, 1, obs_mean.size)
    y_interp = util.interpolate(y, mesh_points, t)
    kaiB = y_interp[7:].sum(axis=0)
    kaiB = kaiB - kaiB.mean()
    kaiB = kaiB / np.std(kaiB)
    E += np.trapz((kaiB - obs_mean)**2 / (2 * std**2), x=t)
    
    return E

@partial(jax.jit, static_argnames=("n_mesh_intervals",))
def kai_dae_log_bvp_potential_mm(q, ode_model, colloc_points_unshifted=util.gauss_points, n_mesh_intervals=60):
    
    n_points = n_mesh_intervals * colloc_points_unshifted.size + 1
    k = q[:ode_model.n_par]
    y = q[kai.n_par:ode_model.n_par + ode_model.n_dim * n_points].reshape((ode_model.n_dim, n_points), order="F")
    mesh_points = q[ode_model.n_par + ode_model.n_dim * n_points:ode_model.n_par + ode_model.n_dim * n_points + n_mesh_intervals - 1]
    mesh_points = np.pad(mesh_points, (1, 1), constant_values=(0, 1))
    E = np.where(np.abs(k) > 7, 100 * (np.abs(k) - 7)**2 / 2, 0).sum()
    y_mean = np.exp(y).mean(axis=1)
    
    pU_logsum = np.log(y_mean[:2].sum())
    E += np.where(pU_logsum < -2, 10 * (pU_logsum + 2)**2 / 2, 0)
    pT_logsum = np.log(y_mean[2:4].sum())
    E += np.where(pT_logsum < -2, 10 * (pT_logsum + 2)**2 / 2, 0)
    pD_logsum = np.log(y_mean[np.array([4, 5, 8, 9, 12, 13])].sum())
    E += np.where(pD_logsum < -2, 10 * (pD_logsum + 2)**2 / 2, 0)
    pS_logsum = np.log(y_mean[np.array([6, 7, 10, 11, 14, 15])].sum())
    E += np.where(pS_logsum < -2, 10 * (pS_logsum + 2)**2 / 2, 0)
    
    min_arclength = 0.3
    arclength = np.linalg.norm(y[:, 1:] - y[:, :-1], axis=0).sum()
    E += np.where(arclength < min_arclength, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4, 0)
    
    std = 1.5e-1
    t = np.linspace(0, 1, obs_mean.size)
    y_interp = util.interpolate(y, mesh_points, t)
    kaiB = np.exp(y_interp[7:-1]).sum(axis=0)
    kaiB = kaiB - kaiB.mean()
    kaiB = kaiB / np.std(kaiB)
    E += np.trapz((kaiB - obs_mean)**2 / (2 * std**2), x=t)
    
    return E

@jax.jit
def kai_bvp_potential(q, ode_model, mesh_points):
    
    n_mesh_intervals = mesh_points.size - 1
    n_points = n_mesh_intervals * util.gauss_points.size + 1
    k = q[:ode_model.n_par]
    y = q[kai.n_par:ode_model.n_par + ode_model.n_dim * n_points].reshape((ode_model.n_dim, n_points), order="F")
    E = np.where(np.abs(k) > 7, 100 * (np.abs(k) - 7)**2 / 2, 0).sum()
    yfull = np.vstack([1 - ode_model.conservation_law[0, 1:-1]@y, y, ode_model.a0 - ode_model.conservation_law[1, 1:-1]@y])
    y_mean = yfull.mean(axis=1)
    
    pU_logsum = np.log(y_mean[:2].sum())
    E += np.where(pU_logsum < -2, 10 * (pU_logsum + 2)**2 / 2, 0)
    pT_logsum = np.log(y_mean[2:4].sum())
    E += np.where(pT_logsum < -2, 10 * (pT_logsum + 2)**2 / 2, 0)
    pD_logsum = np.log(y_mean[np.array([4, 5, 8, 9, 12, 13])].sum())
    E += np.where(pD_logsum < -2, 10 * (pD_logsum + 2)**2 / 2, 0)
    pS_logsum = np.log(y_mean[np.array([6, 7, 10, 11, 14, 15])].sum())
    E += np.where(pS_logsum < -2, 10 * (pS_logsum + 2)**2 / 2, 0)
    
    min_arclength = 0.3
    arclength = np.linalg.norm(y[:, 1:] - y[:, :-1], axis=0).sum()
    E += np.where(arclength < min_arclength, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4, 0)
    
    std = 1.5e-1 / np.sqrt(7)
    t = np.linspace(0, 1, obs_mean.size)
    y_interp = util.interpolate(y, mesh_points, t)
    kaiB = y_interp[7:].sum(axis=0)
    kaiB = kaiB - kaiB.mean()
    kaiB = kaiB / np.std(kaiB)
    E += np.trapz((kaiB - obs_mean)**2 / (2 * std**2), x=t)
    
    return E

dt = 1e-2
prng_key = np.load("kai_lc_key_%d_%d.npy"%(argp.iter - 1, argp.process))[-1]
friction = 1e-2

n_points = n_mesh_intervals * util.gauss_points.size + 1
x = np.load("kai_lc_%d_%d.npy"%(argp.iter - 1, argp.process))[-1]

n_steps = 1000
thin = 10


#kai = model.KaiABC_nondim(par=np.zeros(model.KaiABC_nondim.n_par))
#q0 = x[:kai.n_par + kai.n_dim * n_points + 1]
#p0 = x[q0.size:2 * q0.size] 
#mesh_points = x[-n_mesh_intervals - 1:]
#args = (kai, mesh_points)
#potential = kai_bvp_potential
#resid = defining_systems.periodic_bvp_colloc_resid
#jac = defining_systems.periodic_bvp_colloc_jac
#n_constraints = resid(q0, *args).size
#l0 = x[2 * q0.size:2 * q0.size + n_constraints]
#traj_kai_lc, key_lc = lgvn.gOBABO(q0, p0, l0, dt, friction, n_steps, thin, prng_key, potential, resid, jac,
#                                 A=lgvn.rattle_drift_bvp_mm, nlsol=nonlinear_solver.quasi_newton_bvp_symm_broyden, linsol=linear_solver.qr_lstsq_rattle_bvp,
#                                 max_newton_iter=100, tol=1e-9, args=args, metropolize=True, reversibility_tol=1e-6)

#traj_kai_lc, key_lc = lgvn.gOBABO(q0, p0, l0, dt, friction, n_steps, thin, prng_key, potential, resid, jac, nlsol=nonlinear_solver.quasi_newton_rattle_symm_broyden, 
#                                  max_newton_iter=100, tol=1e-9, args=args, metropolize=True, reversibility_tol=1e-6)

#kai = model.KaiABC_nondim(par=np.zeros(model.KaiABC_nondim.n_par))
#q0 = x[:kai.n_par + kai.n_dim * n_points + 1 + n_mesh_intervals - 1]
#p0 = x[q0.size:2 * q0.size] 
#args = (kai,)
#potential = kai_bvp_potential_mm
#resid = defining_systems.periodic_bvp_mm_colloc_resid
#jac = defining_systems.periodic_bvp_mm_colloc_jac
#n_constraints = resid(q0, *args).size
#l0 = x[2 * q0.size:2 * q0.size + n_constraints]
#traj_kai_lc, key_lc = lgvn.gOBABO(q0, p0, l0, dt, friction, n_steps, thin, prng_key, potential, resid, jac, nlsol=nonlinear_solver.quasi_newton_bvp_symm_broyden, linsol=linear_solver.qr_lstsq_rattle_bvp,
#                                  max_newton_iter=100, tol=1e-9, args=args, metropolize=True, reversibility_tol=1e-6)

#kai = model.KaiABC_DAE_log_nondim(par=np.zeros(model.KaiABC_nondim.n_par))
#q0 = x[:kai.n_par + kai.n_dim * n_points + 1 + n_mesh_intervals - 1]
#p0 = x[q0.size:2 * q0.size] 
#args = (kai,)
#potential = kai_dae_log_bvp_potential_mm
#resid = defining_systems.periodic_bvp_mm_colloc_resid
#jac = defining_systems.periodic_bvp_mm_colloc_jac
#n_constraints = resid(q0, *args).size
#l0 = x[2 * q0.size:2 * q0.size + n_constraints]
#traj_kai_lc, key_lc = lgvn.gOBABO(q0, p0, l0, dt, friction, n_steps, thin, prng_key, potential, resid, jac, nlsol=nonlinear_solver.quasi_newton_bvp_symm_broyden, linsol=linear_solver.qr_lstsq_rattle_bvp,
#                                  max_newton_iter=100, tol=1e-9, args=args, metropolize=True, reversibility_tol=1e-6)

kai = model.KaiABC_DAE_log_nondim(par=np.zeros(model.KaiABC_nondim.n_par))
q0 = x[:kai.n_par + kai.n_dim * n_points + 1 + n_mesh_intervals - 1]
p0 = x[q0.size:2 * q0.size] 
args = (kai, util.midpoint)
potential = lambda *args:kai_dae_log_bvp_potential_mm(*args, n_mesh_intervals=n_mesh_intervals)
resid = lambda *args:defining_systems.periodic_bvp_mm_colloc_resid(*args, n_mesh_intervals=n_mesh_intervals)
jac = lambda *args:defining_systems.periodic_bvp_mm_colloc_jac(*args, n_mesh_intervals=n_mesh_intervals)
n_constraints = resid(q0, *args).size
l0 = x[2 * q0.size:2 * q0.size + n_constraints]
traj_kai_lc, key_lc = lgvn.gOBABO(q0, p0, l0, dt, friction, n_steps, thin, prng_key, potential, resid, jac, nlsol=nonlinear_solver.quasi_newton_bvp_symm_broyden, linsol=linear_solver.qr_lstsq_rattle_bvp,
                                  max_newton_iter=100, tol=1e-10, args=args, metropolize=True, reversibility_tol=1e-6)


np.save("kai_lc_%d_%d.npy"%(argp.iter, argp.process), traj_kai_lc)
np.save("kai_lc_key_%d_%d.npy"%(argp.iter, argp.process), key_lc)
