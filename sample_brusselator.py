import numpy
import jax
import jax.numpy as np
import model
import time
import geodesic_langevin as lgvn
from functools import partial
import util
import nonlinear_solver
import defining_systems
import scipy
import scipy.integrate
jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnums=(4, 5, 6, 10, 11, 12))
def rattle_drift_brusselator_bvp_mm(position, momentum, lagrange_multiplier, dt, potential, constraint, jac_constraint=None, inverse_mass=None, proj=None, 
                                    constraint_args=(), nlsol=nonlinear_solver.newton_rattle, max_newton_iter=20, constraint_tol=1e-9):
    
    mesh_points = constraint_args[0]
    y = position[model.Brusselator.n_par:-1]
    y = y.reshape((model.Brusselator.n_dim, y.size // model.Brusselator.n_dim), order="F")
    yp = momentum[model.Brusselator.n_par:-1]
    yp = yp.reshape((model.Brusselator.n_dim, yp.size // model.Brusselator.n_dim), order="F")
    mesh_new, mesh_density = util.recompute_mesh(y, mesh_points, util.gauss_points)
    y_new = util.recompute_node_y(y, mesh_points, mesh_new, util.gauss_points)
    yp_new = util.recompute_node_y(yp, mesh_points, mesh_new, util.gauss_points)
    position = position.at[model.Brusselator.n_par:-1].set(y_new.ravel(order="F"))
    momentum = momentum.at[model.Brusselator.n_par:-1].set(yp_new.ravel(order="F"))
    constraint_args = list(constraint_args)
    constraint_args[0] = mesh_new
    constraint_args = tuple(constraint_args)
    return lgvn.rattle_drift(position, momentum, lagrange_multiplier, dt, potential, constraint, jac_constraint, inverse_mass, None, 
                             constraint_args, nlsol, max_newton_iter, constraint_tol)

i = 2
dt = 1e-2
prng_key = np.load("brusselator_lc_key%d.npy"%(i - 1))[-1]
#prng_key = jax.random.PRNGKey(0)
friction = 1e-1

n_mesh_intervals = 60
#mesh_points = np.linspace(0, 1, n_mesh_intervals + 1)
#mesh_points = np.load("brusselator_lc_mesh%d.npy"%i)
#n_mesh_intervals = mesh_points.size - 1
n_points = n_mesh_intervals * util.gauss_points.size + 1
fourier_basis_size = 100

x = np.load("brusselator_lc%d.npy"%(i - 1))[-1]
#x = np.load("brusselator_lc%d.npy"%(i - 1))[-1, :2 * (model.Brusselator.n_dim * n_points + model.Brusselator.n_par + 1)]
#x = np.load("brusselator_lc%d.npy"%(i - 1))[-1, :2 * (model.Brusselator.n_dim * (2 * fourier_basis_size - 1) + model.Brusselator.n_par + 1)]
q0 = x[:model.Brusselator.n_par + model.Brusselator.n_dim * n_points + 1]
p0 = x[q0.size:2 * q0.size]
mesh_points = x[-n_mesh_intervals - 1:]
args = (mesh_points,)

br_lc_U = defining_systems.brusselator_log_bvp_potential
br_lc_f = defining_systems.brusselator_log_bvp
br_lc_J = defining_systems.brusselator_log_bvp_jac
#br_lc_U = defining_systems.brusselator_bvp_fourier_potential
#br_lc_f = defining_systems.brusselator_bvp_fourier

q1, p1, l1, energy, force, _, args = lgvn.rattle_kick(q0, p0, dt / 2, br_lc_U, br_lc_f, br_lc_J, args=args)
n_nan = 0
n_steps = 1000000
thin = 100
out = []
i_out = []
mesh_out = []
n_success = n_steps
n_mesh_recompute = 2000
key_out = []

#for i in range(max(n_steps // n_mesh_recompute, 1)):
#
#    traj_br_lc, key_lc = lgvn.gBAOAB(q1, p1, l1, dt, friction, n_mesh_recompute, thin, prng_key, br_lc_U, br_lc_f, br_lc_J, energy=energy, force=force, nlsol=nonlinear_solver.newton_bvp_dense, max_newton_iter=20, constraint_tol=1e-8)
#
#    out.append(traj_br_lc)
#    mesh_out.append(mesh_points)
#    key_out.append(key_lc)
#
#    if np.any(np.isnan(traj_br_lc)):
#        break
#
#    print(i, flush=True)
#
#    q1 = traj_br_lc[i, :q0.size]
#    p1 = traj_br_lc[i, q0.size:2 * q0.size]
#    prng_key = key_lc[-1]
#
#    q1, p1, mesh_points = brusselator_recompute_mesh(q1, p1, mesh_points)
#
#    br_lc_U = jax.jit(lambda q:defining_systems.brusselator_bvp_potential(q, mesh_points))
#    br_lc_f = jax.jit(lambda q:defining_systems.brusselator_bvp(q, mesh_points))
#    br_lc_J = jax.jit(lambda q:defining_systems.brusselator_bvp_jac(q, mesh_points))
#
#    q1, p1, l1, energy, force, _ = lgvn.rattle_kick(q1, p1, dt / 2, br_lc_U, br_lc_f, br_lc_J)

#while n_steps > 0 and n_success >= 10:
#
#    traj_br_lc, key_lc = lgvn.gBAOAB(q1, p1, l1, dt, friction, n_steps, thin, prng_key, br_lc_U, br_lc_f, br_lc_J, energy=energy, force=force, nlsol=nonlinear_solver.newton_bvp_dense, max_newton_iter=20, constraint_tol=1e-8)
#    n_nan = np.isnan(traj_br_lc[:, 0]).sum()
#    n_success = np.isfinite(traj_br_lc[:, 0]).sum()
#    i = int((n_steps // thin - n_nan) * 0.9)
#
#    if n_nan > 0:
#        n_steps = n_steps - i * thin
#        out.append(traj_br_lc[:i + 1])
#    else:
#        n_steps = 0
#        out.append(traj_br_lc)
#
#    i_out.append(i + 1)
#    mesh_out.append(mesh_points)
#
#    q1 = traj_br_lc[i, :q0.size]
#    p1 = traj_br_lc[i, q0.size:2 * q0.size]
#    prng_key = key_lc[i]
#
#    q1, p1, mesh_points = brusselator_recompute_mesh(q1, p1, mesh_points)
#
#    print(n_steps, n_nan, traj_br_lc.shape, flush=True)
#
#    br_lc_U = jax.jit(lambda q:defining_systems.brusselator_bvp_potential(q, mesh_points))
#    br_lc_f = jax.jit(lambda q:defining_systems.brusselator_bvp(q, mesh_points))
#    br_lc_J = jax.jit(lambda q:defining_systems.brusselator_bvp_jac(q, mesh_points))
#
#    q1, p1, l1, energy, force, _ = lgvn.rattle_kick(q1, p1, dt / 2, br_lc_U, br_lc_f, br_lc_J)

traj_br_lc, key_lc = lgvn.gBAOAB(q1, p1, l1, dt, friction, n_steps, thin, prng_key, br_lc_U, br_lc_f, br_lc_J, energy=energy, force=force, A=rattle_drift_brusselator_bvp_mm, 
                                 nlsol=nonlinear_solver.quasi_newton_bvp_dense, max_newton_iter=50, constraint_tol=1e-8, args=args)
#np.save("brusselator_lc.npy", traj_br_lc)
#np.save("brusselator_lc_key.npy", key_lc)

np.save("brusselator_lc%d.npy"%(i), traj_br_lc)
#np.save("brusselator_lc_mesh.npy", np.vstack(mesh_out))
#np.save("brusselator_lc_mesh_index.npy", np.cumsum(np.array(i_out)))
np.save("brusselator_lc_key%d.npy"%(i), key_lc)
