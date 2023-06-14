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
jax.config.update("jax_enable_x64", True)

numpy.random.seed(4)
data = np.load("repressilator_data.npy")
t_eval = data[:, 0]
data_period_average = data[:, 1]
data_period_average = np.concatenate([data_period_average, data_period_average])

def log_probability(k):
    
    rp = model.Repressilator_log(k)
    std = 5e-2
    t_eval = np.linspace(25.05, 30, 100)
    E = 0
    E -= 100 * np.where(k[-3:] < 0, k[-3:]**2, 0.).sum()
    E -= 100 * np.where(k[-3:] > 5, (k[-3:] - 5)**2, 0.).sum()
    E -= 100 * np.where(k[:-3] > 5, (k[:-3] - 5)**2, 0.).sum()
    E -= 100 * np.where(k[:-3] < -5, (k[:-3] + 5)**2, 0.).sum()

    try:
        traj = scipy.integrate.solve_ivp(rp.f, jac=rp.jac, y0=np.log(np.array([1, 1.2, 1])), t_span=(0, 30), t_eval=t_eval, args=(k,), method="LSODA", rtol=1e-6)
    except ValueError:
        return -np.inf

    if len(traj.t) == 0:
        return -np.inf

    E -= np.trapz((data_period_average - np.exp(traj.y[0]))**2 / (2 * std**2), x=t_eval) / 2

    if np.isnan(E):
        return -np.inf

    return E

i = 5
x = np.load("repressilator_lc_emcee%d.npy"%(i - 1))[-1]
n_walkers, n_dim = x.shape
#n_steps = 2000000 // n_walkers
#thin = n_steps // 10000
n_steps = 20000
thin = 1

with multiprocessing.Pool() as pool:
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool=pool)
    sampler.run_mcmc(x, n_steps, thin_by=thin)
    np.save("repressilator_lc_emcee%d.npy"%(i), sampler.get_chain())
    np.save("repressilator_lc_emcee_LL%d.npy"%(i), sampler.get_log_prob())
    print(sampler.acceptance_fraction)
