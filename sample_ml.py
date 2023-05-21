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
def rattle_drift_ml_bvp_mm(position, momentum, lagrange_multiplier, dt, potential, constraint, jac_constraint=None, inverse_mass=None, proj=None, 
                                    constraint_args=(), nlsol=nonlinear_solver.newton_rattle, max_newton_iter=20, tol=1e-9):
    
    mesh_points = constraint_args[0]
    y = position[model.Morris_Lecar.n_par:-1]
    y = y.reshape((model.Morris_Lecar.n_dim, y.size // model.Morris_Lecar.n_dim), order="F")
    yp = momentum[model.Morris_Lecar.n_par:-1]
    yp = yp.reshape((model.Morris_Lecar.n_dim, yp.size // model.Morris_Lecar.n_dim), order="F")
    mesh_new, mesh_density = util.recompute_mesh(y, mesh_points, util.gauss_points)
    y_new = util.interpolate(y, mesh_points, mesh_new, util.gauss_points)
    yp_new = util.interpolate(yp, mesh_points, mesh_new, util.gauss_points)
    position = position.at[model.Morris_Lecar.n_par:-1].set(y_new.ravel(order="F"))
    momentum = momentum.at[model.Morris_Lecar.n_par:-1].set(yp_new.ravel(order="F"))
    constraint_args = list(constraint_args)
    constraint_args[0] = mesh_new
    constraint_args = tuple(constraint_args)
    return lgvn.rattle_drift(position, momentum, lagrange_multiplier, dt, potential, constraint, jac_constraint, inverse_mass, None, 
                             constraint_args, nlsol, max_newton_iter, tol)

i = 1
dt = 1e-2
prng_key = np.load("ml_lc_key%d.npy"%(i - 1))[-1]
friction = 1

n_mesh_intervals = 100
n_points = n_mesh_intervals * util.gauss_points.size + 1
fourier_basis_size = 100

x = np.load("ml_lc%d.npy"%(i - 1))[-1]
q0 = x[:model.Morris_Lecar.n_par + model.Morris_Lecar.n_dim * n_points + 1]
p0 = x[q0.size:2 * q0.size]
mesh_points = x[-n_mesh_intervals - 1 - 2 * model.Morris_Lecar.n_par:-2 * model.Morris_Lecar.n_par]
bounds = x[-2 * model.Morris_Lecar.n_par:].reshape((model.Morris_Lecar.n_par, 2), order="C")
args = (mesh_points, bounds)

U = defining_systems.morris_lecar_bvp_potential
f = defining_systems.morris_lecar_bvp
J = defining_systems.morris_lecar_bvp_jac

q1, p1, l1, energy, force, _, args = lgvn.rattle_kick(q0, p0, dt / 2, U, f, J, args=args)
n_nan = 0
n_steps = 100000
thin = 100
out = []
i_out = []
mesh_out = []
key_out = []

traj_br_lc, key_lc = lgvn.gBAOAB(q1, p1, l1, dt, friction, n_steps, thin, prng_key, U, f, J, energy=energy, force=force, A=rattle_drift_ml_bvp_mm, 
                                 nlsol=nonlinear_solver.quasi_newton_bvp_dense, max_newton_iter=100, tol=1e-8, args=args)

np.save("ml_lc%d.npy"%(i), traj_br_lc)
np.save("ml_lc_key%d.npy"%(i), key_lc)
