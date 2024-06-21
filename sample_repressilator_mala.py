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
import multiprocess
import schwimmbad
import argparse
import diffrax
jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument("-process", type=int, default=0)
parser.add_argument("-iter", type=int, required=True)
parser.add_argument("-n_dim", type=int, default=3)
parser.add_argument("-n_steps", type=int, default=100000)
parser.add_argument("-thin", type=int, default=1)
argp = parser.parse_args()

numpy.random.seed(4)
data = np.load("repressilator_data_2.npy")
data_period_average = data
# data_period_average = np.concatenate([data_period_average, data_period_average])

@partial(jax.jit, static_argnames=("rp_dim",))
def repressilator_log_mala_potential(k, rp_dim):

    rp = model.Repressilator_log_n(k, rp_dim)
    std = 5e-2
    t_eval = np.linspace(25.05, 30, 100)
    E = 0
    E += 100 * np.where(k[-rp_dim:] < 0, k[-rp_dim:]**2, 0.).sum()
    E += 100 * np.where(k[-rp_dim:] > 10, (k[-rp_dim:] - 10)**2, 0.).sum()
    E += 100 * np.where(k[:-rp_dim] > 5, (k[:-rp_dim] - 5)**2, 0.).sum()
    E += 100 * np.where(k[:-rp_dim] < -5, (k[:-rp_dim] + 5)**2, 0.).sum()
    dat = data_period_average

    term = diffrax.ODETerm(rp.f)
    solver = diffrax.Kvaerno4()
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)
    out = diffrax.diffeqsolve(term, solver, 0, t_eval[-1], None, np.log(np.ones(rp_dim).at[-2].add(0.2)),
                              stepsize_controller=stepsize_controller, max_steps=8192, throw=False, args=k, saveat=diffrax.SaveAt(ts=t_eval))

    arclength = np.linalg.norm(out.ys[1:] - out.ys[:-1], axis=1).sum()
    min_arclength = 0.3
    E += np.where(arclength < min_arclength, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4, 0) / 2

    success = np.all(out.result._value == 0)
    E += 2 * jax.scipy.integrate.trapezoid((dat - np.exp(out.ys[:, 0]))**2 / (2 * std**2), x=t_eval / (t_eval[-1] - t_eval[0]))
    E = np.where(np.isfinite(E) & success, E, -np.inf)

    return E


@jax.jit
def repressilator_log_emcee_potential_multiple_shooting(q, ode_model):

    k = q[:ode_model.n_par]
    y = q[ode_model.n_par:]
    y = y.reshape((y.size // ode_model.n_dim, ode_model.n_dim))
    std = 5e-2
    
    E = 0
    E -= 100 * np.where(k[-ode_model.n_dim:] < 0, k[-ode_model.n_dim:]**2, 0.).sum()
    E -= 100 * np.where(k[-ode_model.n_dim:] > 10, (k[-ode_model.n_dim:] - 10)**2, 0.).sum()
    E -= 100 * np.where(k[:-ode_model.n_dim] > 5, (k[:-ode_model.n_dim] - 5)**2, 0.).sum()
    E -= 100 * np.where(k[:-ode_model.n_dim] < -5, (k[:-ode_model.n_dim] + 5)**2, 0.).sum()
    
    arclength = np.linalg.norm(y[1:] - y[:-1], axis=1).sum()
    min_arclength = 0.3
    E += np.where(arclength < min_arclength, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4, 0)
    
    term = diffrax.ODETerm(ode_model.f)
    solver = diffrax.Kvaerno4()
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)
    out = jax.vmap(lambda t1, y0:diffrax.diffeqsolve(term, solver, 0., t1, None, y0, stepsize_controller=stepsize_controller, max_steps=8192, throw=False, args=k))(t_eval[1:] - t_eval[:-1], y[:-1])
    success = np.all(out.result._value == 0)
    E -= jax.scipy.integrate.trapezoid((data_period_average - np.exp(y[:, 0]))**2 / (2 * std**2), x=t_eval / t_eval[-1])
    E -= jax.scipy.integrate.trapezoid(np.linalg.norm(np.squeeze(out.ys) - y[1:], axis=1) / (2 * std**2), x=t_eval[1:] / t_eval[-1])
    E = np.where(success, E, -np.inf)

    return E

rp_dim = argp.n_dim
thin = argp.thin
dt = 1e-2
friction = 100

rp = model.Repressilator_log_n(n_dim=rp_dim)
x = np.load("repressilator_%d_lc_mala_%d_%d.npy"%(rp_dim, argp.iter - 1, argp.process))[-1]
q0 = x[:rp.n_par]
p0 = x[rp.n_par:2 * rp.n_par]
prng_key =x[-1:].view(np.uint32)

potential = partial(repressilator_log_mala_potential, rp_dim=rp_dim)

traj_rp_mala = lgvn.sample((q0, p0, None, None), dt=dt, friction=friction, n_steps=argp.n_steps, potential=potential, thin=thin, stepper=lgvn.BAOAB, metropolize=True, print_acceptance=True)

np.save("repressilator_%d_lc_mala_%d_%d.npy"%(rp_dim, argp.iter, argp.process), traj_rp_mala)
