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
parser.add_argument("-process", type=int, default=0)
argp=parser.parse_args()

@jax.jit
def f4(t, y, k, a0=1/7):
    
    ydot = np.zeros_like(y)
    ydot = ydot.at[0].set(-a0 * np.exp(k[0]) + np.exp(k[0] + y[1]) + np.exp(k[1] - y[0] + y[1]) + np.exp(-y[0] + y[2]) + np.exp(k[0] + y[4]))
    ydot = ydot.at[1].set(-np.exp(k[1]) - np.exp(k[2]) - np.exp(k[0] + y[0]) + a0 * np.exp(k[0] + y[0] - y[1]) - np.exp(k[0] + y[0] - y[1] + y[4]))
    ydot = ydot.at[2].set(-1 - np.exp(k[4]) + np.exp(k[2] + y[1] - y[2]) + np.exp(k[3] - y[2] + y[3]))
    ydot = ydot.at[3].set(-1 - np.exp(k[3]) - a0 * np.exp(k[5]) + np.exp(k[5] + y[1]) - np.exp(y[0] - y[3]) - np.exp(y[1] - y[3]) - np.exp(y[2] - y[3]) + np.exp(k[4] + y[2] - y[3])\
                          + np.exp(-y[3]) + np.exp(k[5] + y[4]) - np.exp(-y[3] + y[4]) + np.exp(k[6] - y[3] + y[4]))
    ydot = ydot.at[4].set(-np.exp(k[6]) - np.exp(k[7]) - np.exp(k[5] + y[3]) + a0 * np.exp(k[5] + y[3] - y[4]) - np.exp(k[5] + y[1] + y[3] - y[4]))
    return ydot

@jax.jit
def f_multi(q):
    k = q[:8]
    y1 = q[8:13]
    y2 = q[13:18]
    return np.concatenate([f4(0., y1, k), f4(0., y2, k)])

@jax.jit
def potential(q):
    k = q[:8]
    y1 = q[8:13]
    y2 = q[13:18]
    min_distance = 1
    distance_sq = np.sum((y1 - y2)**2)
    E_attract = min_distance**2 / (2 * distance_sq)
    E_repel = E_attract**2
    E = np.where(distance_sq < min_distance**2, E_repel - E_attract + 1/4, 0)
    E += np.where(q < -9, (q + 9)**2 / 2, 0).sum()
    return E

i = argp.process
dt = 1e-2
prng_key = jax.random.PRNGKey(i)
friction = 1e-2

n_steps = 20000000
thin = 100

x = np.load("kai_bistable_%d.npy"%(i))
q0 = x[-1, :18]
p0 = x[-1, 18:36]
l0 = np.zeros_like(f_multi(q0))

traj, key = lgvn.gOBABO(q0, p0, l0, dt, friction, n_steps, thin, prng_key, potential, f_multi, max_newton_iter=100, tol=1e-9, nlsol=nonlinear_solver.quasi_newton_rattle_symm_broyden, 
                        linsol=linear_solver.lq_ortho_proj, metropolize=True, reversibility_tol=1e-6)

np.save("kai_bistable_%d.npy"%(i), traj)
np.save("kai_bistable_key_%d.npy"%(i), key)
