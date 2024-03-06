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

@partial(jax.jit, static_argnums=(4, 5, 6, 10, 11, 12, 13))
def rattle_drift_ml_bvp_mm(position, momentum, lagrange_multiplier, dt, potential, constraint, jac_constraint=None, inverse_mass=None, J_and_factor=None, 
                                    constraint_args=(), nlsol=nonlinear_solver.newton_rattle, linsol=linear_solver.lq_ortho_proj, max_newton_iter=20, constraint_tol=1e-9):
    
    mesh_points = constraint_args[0]
    y = position[model.Morris_Lecar.n_par:-1]
    y = y.reshape((model.Morris_Lecar.n_dim, y.size // model.Morris_Lecar.n_dim), order="F")
    yp = momentum[model.Morris_Lecar.n_par:-1]
    yp = yp.reshape((model.Morris_Lecar.n_dim, yp.size // model.Morris_Lecar.n_dim), order="F")
    mesh_new, mesh_density = util.recompute_mesh(y, mesh_points, util.gauss_points_4)
    y_new = util.recompute_node_y(y, mesh_points, mesh_new, util.gauss_points_4)
    yp_new = util.recompute_node_y(yp, mesh_points, mesh_new, util.gauss_points_4)
    position = position.at[model.Morris_Lecar.n_par:-1].set(y_new.ravel(order="F"))
    momentum = momentum.at[model.Morris_Lecar.n_par:-1].set(yp_new.ravel(order="F"))
    constraint_args = list(constraint_args)
    constraint_args[0] = mesh_new
    constraint_args = tuple(constraint_args)
    return lgvn.rattle_drift(position, momentum, lagrange_multiplier, dt, potential, constraint, jac_constraint, inverse_mass, J_and_factor, 
                             constraint_args, nlsol, linsol, max_newton_iter, constraint_tol)

dt = 1e-2
#prng_key = np.load("ml_lc_key%d.npy"%(i - 1))[-1]
prng_key = jax.random.PRNGKey(time.time_ns())
prng_key, subkey = jax.random.split(prng_key)
friction = 1
Minv = None

n_mesh_intervals = 60
n_points = n_mesh_intervals * util.gauss_points_4.size + 1
#fourier_basis_size = 100

ref = []
i = 1

while os.path.isfile("ml_lc%d.npy"%(i)):
    ref.append(np.load("ml_lc%d.npy"%(i))[::10])
    i += 1

ref = np.vstack(ref)
ref_index = jax.random.randint(subkey, (1,), 0, ref.shape[0])

#x = np.load("ml_lc%d.npy"%(i - 1))[-1]
x = ref[ref_index][0]
q0 = x[:model.Morris_Lecar.n_par + model.Morris_Lecar.n_dim * n_points + 1]
#p0 = x[q0.size:2 * q0.size]
p0 = jax.random.normal(prng_key, q0.shape)
mesh_points = x[-n_mesh_intervals - 1 - 2 * model.Morris_Lecar.n_par:-2 * model.Morris_Lecar.n_par]
bounds = x[-2 * model.Morris_Lecar.n_par:].reshape((model.Morris_Lecar.n_par, 2), order="C")
args = (mesh_points, bounds)

U = defining_systems.morris_lecar_bvp_potential
f = defining_systems.morris_lecar_bvp
J = defining_systems.morris_lecar_bvp_jac

q1, p1, l1, energy, force, _, args = lgvn.rattle_kick(q0, p0, dt / 2, U, f, J, args=args, linsol=linear_solver.lq_ortho_proj_bvp)
n_steps = 1000000
thin = 100
out = []
key_out = []

while n_steps > 0:
    
    traj_br_lc, key_lc = lgvn.gBAOAB(q1, p1, l1, dt, friction, n_steps, thin, prng_key, U, f, J,\
                                     energy=energy, force=force, A=rattle_drift_ml_bvp_mm, nlsol=nonlinear_solver.quasi_newton_bvp_symm_broyden,\
                                     linsol=linear_solver.lq_ortho_proj_bvp, max_newton_iter=100, constraint_tol=1e-8, args=(mesh_points, bounds), inverse_mass=Minv)
    
    n_success = np.isfinite(traj_br_lc[:, 0]).sum() - 10

    if n_success <= 0:
        break

    traj_br_lc = traj_br_lc[:n_success]
    key_lc = key_lc[:n_success]
    out.append(traj_br_lc)
    key_out.append(key_lc)
    prng_key, subkey = jax.random.split(key_lc[-1])
    q1 = traj_br_lc[-1, :q1.size]
    prng_key, subkey = jax.random.split(prng_key)
    p1 = jax.random.normal(prng_key, p1.shape)
    mesh_points = traj_br_lc[-1, -n_mesh_intervals - 1 - 2 * model.Morris_Lecar.n_par:-2 * model.Morris_Lecar.n_par]
    q1, p1, l1, energy, force, _, args = lgvn.rattle_kick(q1, p1, dt / 2, U, f, J, args=(mesh_points, bounds), linsol=linear_solver.lq_ortho_proj_bvp)
    n_steps = int(n_steps - n_success * thin)

    print("%d successful steps"%(n_success), flush=True)

np.save("ml_lc%d.npy"%(i), np.vstack(out))
np.save("ml_lc_key%d.npy"%(i), np.vstack(key_out))
