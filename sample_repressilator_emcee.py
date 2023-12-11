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
import emcee
import multiprocessing
import argparse
jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument("-iter", type=int, required=True)
parser.add_argument("-n_dim", type=int, default=3)
argp = parser.parse_args()

numpy.random.seed(4)
data = np.load("repressilator_data.npy")
t_eval = data[:, 0]
data_period_average = data[:, 1]
data_period_average = np.concatenate([data_period_average, data_period_average])

def log_probability(k, rp_dim):
    
    rp = model.Repressilator_log_n(k, rp_dim)
    std = 5e-2
    t_eval = np.linspace(25.05, 30, 100)
    E = 0
    E -= 100 * np.where(k[-rp_dim:] < 0, k[-rp_dim:]**2, 0.).sum()
    E -= 100 * np.where(k[-rp_dim:] > 10, (k[-rp_dim:] - 10)**2, 0.).sum()
    E -= 100 * np.where(k[:-rp_dim] > 5, (k[:-rp_dim] - 5)**2, 0.).sum()
    E -= 100 * np.where(k[:-rp_dim] < -5, (k[:-rp_dim] + 5)**2, 0.).sum()

    try:
        traj = scipy.integrate.solve_ivp(rp.f, jac=rp.jac, y0=np.log(np.ones(rp_dim).at[-2].add(0.2)), t_span=(0, 30), t_eval=t_eval, args=(k,), method="LSODA", rtol=1e-6)
    except ValueError:
        return -np.inf

    if len(traj.t) == 0 or not traj.success:
        return -np.inf

    E -= np.trapz((data_period_average - np.exp(traj.y[0]))**2 / (2 * std**2), x=t_eval) / 2

    if np.isnan(E):
        return -np.inf

    return E

rp_dim = argp.n_dim
x = np.load("repressilator_%d_lc_emcee%d.npy"%(rp_dim, argp.iter - 1))[-1]
n_walkers, n_dim = x.shape
#n_steps = 2000000 // n_walkers
#thin = n_steps // 10000
n_steps = 50000
thin = 100

with multiprocessing.Pool() as pool:
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool=pool, args=(rp_dim,))
    sampler.run_mcmc(x, n_steps)
    np.save("repressilator_%d_lc_emcee%d.npy"%(rp_dim, argp.iter), sampler.get_chain()[::thin])
    np.save("repressilator_%d_lc_emcee_LL%d.npy"%(rp_dim, argp.iter), sampler.get_log_prob()[::thin])
    print(sampler.acceptance_fraction)
