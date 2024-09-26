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
import diffrax
jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument("-process", type=int, required=True)
parser.add_argument("-iter", type=int, required=True)
argp = parser.parse_args()

#obs_mean = np.load("kai_data.npy")
obs_kai_ab = np.load("kai_ab_data.npy")
obs_kai_phos = np.load("kai_phos_data.npy")
obs_kaiac_autophos = np.asarray(numpy.loadtxt("KaiAC_autophos.txt"))
obs_kaic_autodephos = numpy.loadtxt("KaiC_autodephos.txt")
n_mesh_intervals = 60
colloc_points_unshifted = util.gauss_points_4

@jax.jit
def dephos(t, y0, k):
    k = np.exp(k)
    l0 = k[0] - k[2] - k[3]
    l1 = k[1] - k[2] - k[3]
    l2 = k[2] + k[3]
    return np.array([np.exp(-k[0] * t) * (y0[0] + y0[2] * k[2] * (np.exp(l0 * t) - 1) / l0),
                     np.exp(-k[1] * t) * (y0[1] + y0[2] * k[3] * (np.exp(l1 * t) - 1) / l1),
                     y0[2] * np.exp(-l2 * t)])

@partial(jax.jit, static_argnames=("n_mesh_intervals",))
def kai_bvp_potential_mm(q, ode_model, colloc_points_unshifted=util.gauss_points_4, n_mesh_intervals=60):
    
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
    
    t = np.linspace(0, 1, obs_kai_ab.shape[0])
    y_interp = util.interpolate(y, mesh_points, t, colloc_points_unshifted)

    std = 2 / 9 / np.sqrt(5)
    B_bound = y_interp[7:].sum(axis=0)
    B_bound = B_bound - B_bound.mean()
    B_bound = B_bound / np.std(B_bound)
    E += jax.scipy.integrate.trapezoid((B_bound - obs_kai_ab[:, 1])**2 / (2 * std**2), x=t)
    
    std = 2 / 3 / np.sqrt(5)
    A_bound = ode_model.conservation_law[1, 1:-1]@y_interp
    A_bound = A_bound - A_bound.mean()
    A_bound = A_bound / np.std(A_bound)
    E += jax.scipy.integrate.trapezoid((A_bound - obs_kai_ab[:, 0])**2 / (2 * std**2), x=t)

    t = np.linspace(0, 1, obs_kai_phos.shape[0])
    y_interp = util.interpolate(y, mesh_points, t, colloc_points_unshifted)

    phos = y_interp[1:].sum(axis=0)
    E += jax.scipy.integrate.trapezoid(100 * (phos - obs_kai_phos) ** 2 / 2, x=t)

    return E

@partial(jax.jit, static_argnames=("n_mesh_intervals",))
def kai_bvp_potential_mm_multi(q, ode_models, colloc_points_unshifted=(util.gauss_points_4, util.gauss_points_4), n_mesh_intervals=(60, 60)):
   
    k = q[:ode_models[0].n_par]
    start = ode_models[0].n_par
    E = np.where(np.abs(k) > 7, 100 * (np.abs(k) - 7)**2 / 2, 0).sum()

    for i in range(len(ode_models)):

        n_points = n_mesh_intervals[i] * colloc_points_unshifted[i].size + 1
        stop = start + ode_models[i].n_dim * n_points
        y = q[start:stop].reshape((ode_models[i].n_dim, n_points), order="F")
        start = stop
        stop = start + n_mesh_intervals[i] - 1
        mesh_points = np.pad(q[start:stop], (1, 1), constant_values=(0, 1))
        start = stop + 1
        yfull = np.vstack([1 - ode_models[i].conservation_law[0, 1:-1]@y, y, ode_models[i].a0 - ode_models[i].conservation_law[1, 1:-1]@y])
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
    
    t_ab = np.linspace(0, 1, obs_kai_ab.shape[0])
    n_points = n_mesh_intervals[0] * colloc_points_unshifted[0].size + 1
    start = ode_models[0].n_par
    stop = start + ode_models[0].n_dim * n_points
    y = q[start:stop].reshape((ode_models[i].n_dim, n_points), order="F")
    start = stop
    stop = start + n_mesh_intervals[0] - 1
    mesh_points = np.pad(q[start:stop], (1, 1), constant_values=(0, 1))
    y_interp = util.interpolate(y, mesh_points, t_ab, colloc_points_unshifted[0])
    start = stop
    period06 = q[start]

    std = 2 / 9 / np.sqrt(5)
    B_bound = y_interp[7:].sum(axis=0)
    B_mean = B_bound.mean()
    B_bound = B_bound - B_mean
    B_scale = np.std(B_bound)
    B_bound = B_bound / B_scale
    E += jax.scipy.integrate.trapezoid((B_bound - obs_kai_ab[:, 1])**2 / (2 * std**2), x=t_ab)
    
    std = 2 / 3 / np.sqrt(5)
    A_bound = ode_models[0].conservation_law[1, 1:-1]@y_interp
    A_bound = A_bound - A_bound.mean()
    A_bound = A_bound / np.std(A_bound)
    E += jax.scipy.integrate.trapezoid((A_bound - obs_kai_ab[:, 0])**2 / (2 * std**2), x=t_ab)

    t_phos = np.linspace(0, 1, obs_kai_phos.shape[0])
    y_interp = util.interpolate(y, mesh_points, t_phos, colloc_points_unshifted[0])
    pT = y_interp[1:3].sum(axis=0)
    pD = y_interp[np.array([3, 4, 7, 8, 11, 12])].sum(axis=0)
    pS = y_interp[np.array([5, 6, 9, 10, 13, 14])].sum(axis=0)
    E += jax.scipy.integrate.trapezoid(3000 * (pT - obs_kai_phos[:, 0]) ** 2 / 2, x=t_phos)
    E += jax.scipy.integrate.trapezoid(3000 * (pS - obs_kai_phos[:, 1]) ** 2 / 2, x=t_phos)
    E += jax.scipy.integrate.trapezoid(3000 * (pD - obs_kai_phos[:, 2]) ** 2 / 2, x=t_phos)

    n_points = n_mesh_intervals[1] * colloc_points_unshifted[1].size + 1
    start = start + 1
    stop = start + ode_models[1].n_dim * n_points
    y = q[start:stop].reshape((ode_models[1].n_dim, n_points), order="F")
    start = stop
    stop = start + n_mesh_intervals[1] - 1
    mesh_points = np.pad(q[start:stop], (1, 1), constant_values=(0, 1))
    y_interp = util.interpolate(y, mesh_points, t_ab, colloc_points_unshifted[1])

    std = 2 / 9 /np.sqrt(5)
    B_bound = y_interp[7:].sum(axis=0)
    B_bound = B_bound - B_mean
    B_bound = B_bound / B_scale
    E += jax.scipy.integrate.trapezoid((B_bound - obs_kai_ab[:, 1])**2 / (2 * std**2), x=t_ab)

    start = stop
    period18 = q[start]

    E += 100 * (period06 / period18 - 1.1)**2 / 2

    return E

@partial(jax.jit, static_argnames=("n_mesh_intervals"))
def kai_bvp_potential_mm_4cond(q, ode_models, colloc_points_unshifted=None, quadrature_weights=None, n_mesh_intervals=None):
   
    if colloc_points_unshifted is None:
        colloc_points_unshifted = tuple(util.gauss_points_4 for _ in range(len(ode_models)))
    if quadrature_weights is None:
        quadrature_weights = tuple(util.gauss_weights_4 for _ in range(len(ode_models)))
    if n_mesh_intervals is None:
        n_mesh_intervals = tuple(60 for _ in range(len(ode_models)))

    k = q[:ode_models[0].n_par]
    start = ode_models[0].n_par
    E = np.where(np.abs(k) > 7, 100 * (np.abs(k) - 7)**2 / 2, 0).sum()
    #E += np.where(np.abs(k[3] - k[1]) > 1, 4 * (np.abs(k[3] - k[1]) - 1)**2, 0)
    #E += np.where(np.abs(k[2] - k[0]) > 1, 4 * (np.abs(k[2] - k[0]) - 1)**2, 0)
    
    for i in range(len(ode_models)):

        n_points = n_mesh_intervals[i] * colloc_points_unshifted[i].size + 1
        stop = start + ode_models[i].n_dim * n_points
        y = q[start:stop].reshape((ode_models[i].n_dim, n_points), order="F")
        start = stop
        stop = start + n_mesh_intervals[i]
        interval_widths = q[start:stop]
        mesh_points = np.pad(q[start:stop].cumsum(), (1, 0))
        start = stop
        yfull = np.vstack([1 - ode_models[i].conservation_law[0, 1:-1]@y, y, ode_models[i].a0 - ode_models[i].conservation_law[1, 1:-1]@y])
        y_mean = jax.scipy.integrate.trapezoid(yfull, x=util.fill_mesh(mesh_points, colloc_points_unshifted=colloc_points_unshifted[i]) / mesh_points[-1])
        reaction_fluxes = np.exp((ode_models[i].K.T@np.log(np.maximum(yfull, 1e-9))).T.at[:, 1:].add(k))
        mesh_full = util.fill_mesh(mesh_points)
        atp_consumption = jax.scipy.integrate.trapezoid(reaction_fluxes[:, ode_models[i].ind_ATP].sum(axis=1), x=mesh_full)
        E += atp_consumption**2 / 10000

        pU_logsum = np.log(y_mean[:2].sum())
        E += np.where(pU_logsum < -2, 10 * (pU_logsum + 2)**2 / 2, 0)
        pT_logsum = np.log(y_mean[2:4].sum())
        E += np.where(pT_logsum < -2, 10 * (pT_logsum + 2)**2 / 2, 0)
        E += np.where(pT_logsum > -1, 20 * (pT_logsum + 1)**2 / 2, 0)
        pD_logsum = np.log(y_mean[np.array([4, 5, 8, 9, 12, 13])].sum())
        E += np.where(pD_logsum < -2, 10 * (pD_logsum + 2)**2 / 2, 0)
        E += np.where(pD_logsum > -1, 20 * (pD_logsum + 1)**2 / 2, 0)
        pS_logsum = np.log(y_mean[np.array([6, 7, 10, 11, 14, 15])].sum())
        E += np.where(pS_logsum < -2, 10 * (pS_logsum + 2)**2 / 2, 0)
        E += np.where(pS_logsum > -1, 20 * (pS_logsum + 1)**2 / 2, 0)
            
        min_arclength = 0.3
        arclength = np.linalg.norm(y[:, 1:] - y[:, :-1], axis=0).sum()
        E += np.where(arclength < min_arclength, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4, 0)

        ynorm, yderiv2 = util.curvature_poly(y, colloc_points_unshifted[i], quadrature_weights[i])
        max_curvature = 0
        y_curvature = yderiv2.sum() / ynorm.sum()
        E += np.where(y_curvature > max_curvature, 100 * (y_curvature - max_curvature)**2, 0)
        
        if(i == 0):
            term = diffrax.ODETerm(ode_models[0].f)
            solver = diffrax.Kvaerno4()
            stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)
            sol = diffrax.diffeqsolve(term, solver, t0=0, t1=mesh_points[-1], dt0=None, y0=np.full(ode_models[0].n_dim, 1e-9), stepsize_controller=stepsize_controller, args=k, throw=False)
            E += 1e4 * np.linalg.norm(sol.ys[-1] - y.T, axis=1).min()**2

        J_floquet = defining_systems.bvp_floquet_jac(np.zeros(y.size), y, k, interval_widths, ode_models[i], colloc_points_unshifted[i])
        J_floquet_LQ = J_floquet.lq_factor_1(method="householder")
        floquet_basis = J_floquet_LQ.Q_multiply(np.zeros((y.size, ode_models[i].n_dim)).at[-ode_models[i].n_dim:].set(np.identity(ode_models[i].n_dim)), permute=False)
        floquet_multipliers = np.linalg.eigvals(np.linalg.solve(floquet_basis[:ode_models[i].n_dim], floquet_basis[-ode_models[i].n_dim:]))
        max_multiplier = np.max(np.abs(floquet_multipliers))
        E += np.where(max_multiplier > 1.01, 100 * (max_multiplier - 1.01)**2, 0)
                
    t_ab = np.linspace(0, 1, obs_kai_ab.shape[0])
    n_points = n_mesh_intervals[0] * colloc_points_unshifted[0].size + 1
    start = ode_models[0].n_par
    stop = start + ode_models[0].n_dim * n_points
    y = q[start:stop].reshape((ode_models[i].n_dim, n_points), order="F")
    start = stop
    stop = start + n_mesh_intervals[0]
    mesh_points = np.pad(q[start:stop].cumsum(), (1, 0))
    period06 = mesh_points[-1]
    y_interp = util.interpolate(y, mesh_points, t_ab * period06, colloc_points_unshifted[0])
    start = stop

    std = 2 / 9 / np.sqrt(5)
    B_bound = y_interp[7:].sum(axis=0)
    B_mean = B_bound.mean()
    B_bound = B_bound - B_mean
    B_scale = np.std(B_bound)
    B_bound = B_bound / B_scale
    E += jax.scipy.integrate.trapezoid((B_bound - obs_kai_ab[:, 1])**2 / (2 * std**2), x=t_ab)
    
    std = 2 / 3 / np.sqrt(5)
    A_bound = ode_models[0].conservation_law[1, 1:-1]@y_interp
    A_bound = A_bound - A_bound.mean()
    A_bound = A_bound / np.std(A_bound)
    E += jax.scipy.integrate.trapezoid((A_bound - obs_kai_ab[:, 0])**2 / (2 * std**2), x=t_ab)

    t_phos = np.linspace(0, 1, obs_kai_phos.shape[0])
    y_interp = util.interpolate(y, mesh_points, t_phos * period06, colloc_points_unshifted[0])
    pT = y_interp[1:3].sum(axis=0)
    pD = y_interp[np.array([3, 4, 7, 8, 11, 12])].sum(axis=0)
    pS = y_interp[np.array([5, 6, 9, 10, 13, 14])].sum(axis=0)
    E += jax.scipy.integrate.trapezoid(3000 * (pT - obs_kai_phos[:, 0])** 2 / 2, x=t_phos)
    E += jax.scipy.integrate.trapezoid(3000 * (pS - obs_kai_phos[:, 1])** 2 / 2, x=t_phos)
    E += jax.scipy.integrate.trapezoid(3000 * (pD - obs_kai_phos[:, 2])** 2 / 2, x=t_phos)

    t_autophos = obs_kaiac_autophos[:, 0]
    kai_ac = model.KaiAC_nondim(k[ode_models[0].ind_C2_rxns[1:] - 1], a0=ode_models[0].a0)
    term = diffrax.ODETerm(kai_ac.f)
    solver = diffrax.Kvaerno4()
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)
    saveat = diffrax.SaveAt(ts=obs_kaiac_autophos[:, 0] * period06)
    sol = diffrax.diffeqsolve(term, solver, t0=0, t1=t_autophos[-1] * period06, dt0=None, y0=np.full(kai_ac.n_dim, 1e-9), saveat=saveat,
                              stepsize_controller=stepsize_controller, args=k[ode_models[0].ind_C2_rxns[1:] - 1], throw=False)
    pT = sol.ys[:, 1:3].sum(axis=1)
    pD = sol.ys[:, 3:5].sum(axis=1)
    pS = sol.ys[:, 5:7].sum(axis=1)
    E += jax.scipy.integrate.trapezoid(3000 * (pT - obs_kaiac_autophos[:, 1])** 2 / 2, x=t_autophos)
    E += jax.scipy.integrate.trapezoid(3000 * (pS - obs_kaiac_autophos[:, 2])** 2 / 2, x=t_autophos)
    E += jax.scipy.integrate.trapezoid(3000 * (pD - obs_kaiac_autophos[:, 3])** 2 / 2, x=t_autophos)
    
    k_autodephos = k[ode_models[0].ind_C2_rxns[np.array([1, 2, 6, 13])] - 1]
    traj_autodephos = dephos(obs_kaic_autodephos[:, 0] * period06, obs_kaic_autodephos[0, 1:], k_autodephos)
    E += jax.scipy.integrate.trapezoid(3000 * (traj_autodephos[0] - obs_kaic_autodephos[:, 1])** 2 / 2, x=obs_kaic_autodephos[:, 0])
    E += jax.scipy.integrate.trapezoid(3000 * (traj_autodephos[1] - obs_kaic_autodephos[:, 2])** 2 / 2, x=obs_kaic_autodephos[:, 0])
    E += jax.scipy.integrate.trapezoid(3000 * (traj_autodephos[2] - obs_kaic_autodephos[:, 3])** 2 / 2, x=obs_kaic_autodephos[:, 0])
    
    n_points = n_mesh_intervals[1] * colloc_points_unshifted[1].size + 1
    stop = start + ode_models[1].n_dim * n_points
    y = q[start:stop].reshape((ode_models[1].n_dim, n_points), order="F")
    start = stop
    stop = start + n_mesh_intervals[1]
    mesh_points = np.pad(q[start:stop].cumsum(), (1, 0))
    period12 = mesh_points[-1]
    y_interp = util.interpolate(y, mesh_points, t_ab * period12, colloc_points_unshifted[1])

    std = 2 / 9 / np.sqrt(5)
    B_bound = y_interp[7:].sum(axis=0)
    B_bound = B_bound - B_mean
    B_bound = B_bound / B_scale
    E += jax.scipy.integrate.trapezoid((B_bound - obs_kai_ab[:, 1])**2 / (2 * std**2), x=t_ab)
    E += 100 * (period06 / period12 - 1)**2 / 2
    
    start = stop
    n_points = n_mesh_intervals[2] * colloc_points_unshifted[2].size + 1
    stop = start + ode_models[2].n_dim * n_points
    y = q[start:stop].reshape((ode_models[2].n_dim, n_points), order="F")
    start = stop
    stop = start + n_mesh_intervals[2]
    mesh_points = np.pad(q[start:stop].cumsum(), (1, 0))
    period18 = mesh_points[-1]
    y_interp = util.interpolate(y, mesh_points, t_ab * period18, colloc_points_unshifted[2])

    std = 2 / 9 / np.sqrt(5)
    B_bound = y_interp[7:].sum(axis=0)
    B_bound = B_bound - B_mean
    B_bound = B_bound / B_scale
    E += jax.scipy.integrate.trapezoid((B_bound - obs_kai_ab[:, 1])**2 / (2 * std**2), x=t_ab)
    E += 100 * (period06 / period18 - 1.1)**2 / 2
    
    start = stop
    n_points = n_mesh_intervals[3] * colloc_points_unshifted[3].size + 1
    stop = start + ode_models[3].n_dim * n_points
    y = q[start:stop].reshape((ode_models[3].n_dim, n_points), order="F")
    start = stop
    stop = start + n_mesh_intervals[3]
    mesh_points = np.pad(q[start:stop].cumsum(), (1, 0))
    period_ATPfrac025 = mesh_points[-1]
    y_interp = util.interpolate(y, mesh_points, t_phos * period_ATPfrac025, colloc_points_unshifted[3])
    pT = y_interp[1:3].sum(axis=0)
    pD = y_interp[np.array([3, 4, 7, 8, 11, 12])].sum(axis=0)
    pS = y_interp[np.array([5, 6, 9, 10, 13, 14])].sum(axis=0)
    E += jax.scipy.integrate.trapezoid(3000 * (pT - obs_kai_phos[:, 3])** 2 / 2, x=t_phos)
    E += jax.scipy.integrate.trapezoid(3000 * (pS - obs_kai_phos[:, 4])** 2 / 2, x=t_phos)
    E += jax.scipy.integrate.trapezoid(3000 * (pD - obs_kai_phos[:, 5])** 2 / 2, x=t_phos)
    E += 100 * (period06 / period_ATPfrac025 - 1)**2 / 2
    
    return E

@partial(jax.jit, static_argnames=("n_mesh_intervals"))
def kai_bvp_potential_mm_4cond_1(q, ode_models, colloc_points_unshifted=None, quadrature_weights=None, n_mesh_intervals=None):
   
    if colloc_points_unshifted is None:
        colloc_points_unshifted = tuple(util.gauss_points_4 for _ in range(len(ode_models)))
    if quadrature_weights is None:
        quadrature_weights = tuple(util.gauss_weights_4 for _ in range(len(ode_models)))
    if n_mesh_intervals is None:
        n_mesh_intervals = tuple(60 for _ in range(len(ode_models)))

    k = q[:ode_models[0].n_par]
    start = ode_models[0].n_par
    E = np.where(np.abs(k) > 7, 100 * (np.abs(k) - 7)**2 / 2, 0).sum()
    #E += np.where(np.abs(k[3] - k[1]) > 1, 4 * (np.abs(k[3] - k[1]) - 1)**2, 0)
    #E += np.where(np.abs(k[2] - k[0]) > 1, 4 * (np.abs(k[2] - k[0]) - 1)**2, 0)
    
    for i in range(len(ode_models)):

        n_points = n_mesh_intervals[i] * colloc_points_unshifted[i].size + 1
        stop = start + ode_models[i].n_dim * n_points
        y = q[start:stop].reshape((ode_models[i].n_dim, n_points), order="F")
        start = stop
        stop = start + n_mesh_intervals[i]
        interval_widths = q[start:stop]
        mesh_points = np.pad(q[start:stop].cumsum(), (1, 0))
        start = stop
        yfull = np.vstack([1 - ode_models[i].conservation_law[0, 1:-1]@y, y, ode_models[i].a0 - ode_models[i].conservation_law[1, 1:-1]@y])
        y_mean = jax.scipy.integrate.trapezoid(yfull, x=util.fill_mesh(mesh_points, colloc_points_unshifted=colloc_points_unshifted[i]) / mesh_points[-1])
        reaction_fluxes = np.exp((ode_models[i].K.T@np.log(np.maximum(yfull, 1e-9))).T.at[:, 1:].add(k))
        mesh_full = util.fill_mesh(mesh_points)
        atp_consumption = jax.scipy.integrate.trapezoid(reaction_fluxes[:, ode_models[i].ind_ATP].sum(axis=1), x=mesh_full)
        E += atp_consumption**2 / 10000

        pU_logsum = np.log(y_mean[:2].sum())
        E += np.where(pU_logsum < -2, 10 * (pU_logsum + 2)**2 / 2, 0)
        pT_logsum = np.log(y_mean[2:4].sum())
        E += np.where(pT_logsum < -2, 10 * (pT_logsum + 2)**2 / 2, 0)
        E += np.where(pT_logsum > -1, 20 * (pT_logsum + 1)**2 / 2, 0)
        pD_logsum = np.log(y_mean[np.array([4, 5, 8, 9, 12, 13])].sum())
        E += np.where(pD_logsum < -2, 10 * (pD_logsum + 2)**2 / 2, 0)
        E += np.where(pD_logsum > -1, 20 * (pD_logsum + 1)**2 / 2, 0)
        pS_logsum = np.log(y_mean[np.array([6, 7, 10, 11, 14, 15])].sum())
        E += np.where(pS_logsum < -2, 10 * (pS_logsum + 2)**2 / 2, 0)
        E += np.where(pS_logsum > -1, 20 * (pS_logsum + 1)**2 / 2, 0)
            
        min_arclength = 0.3
        arclength = np.linalg.norm(y[:, 1:] - y[:, :-1], axis=0).sum()
        E += np.where(arclength < min_arclength, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4, 0)

        ynorm, yderiv2 = util.curvature_poly(y, colloc_points_unshifted[i], quadrature_weights[i])
        max_curvature = 0
        y_curvature = yderiv2.sum() / ynorm.sum()
        E += np.where(y_curvature > max_curvature, 100 * (y_curvature - max_curvature)**2, 0)
        
        #if(i == 0):
        #    term = diffrax.ODETerm(ode_models[0].f)
        #    solver = diffrax.Kvaerno4()
        #    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)
        #    sol = diffrax.diffeqsolve(term, solver, t0=0, t1=mesh_points[-1], dt0=None, y0=np.full(ode_models[0].n_dim, 1e-9), stepsize_controller=stepsize_controller, args=k, throw=False)
        #    E += 1e4 * np.linalg.norm(sol.ys[-1] - y.T, axis=1).min()**2

        # J_floquet = defining_systems.bvp_floquet_jac(np.zeros(y.size), y, k, interval_widths, ode_models[i], colloc_points_unshifted[i])
        # J_floquet_LQ = J_floquet.lq_factor_1(method="householder")
        # floquet_basis = J_floquet_LQ.Q_multiply(np.zeros((y.size, ode_models[i].n_dim)).at[-ode_models[i].n_dim:].set(np.identity(ode_models[i].n_dim)), permute=False)
        # floquet_multipliers = np.linalg.eigvals(np.linalg.solve(floquet_basis[:ode_models[i].n_dim], floquet_basis[-ode_models[i].n_dim:]))
        # max_multiplier = np.max(np.abs(floquet_multipliers))
        # E += np.where(max_multiplier > 1.01, 100 * (max_multiplier - 1.01)**2, 0)
                
    t_ab = np.linspace(0, 1, obs_kai_ab.shape[0])
    n_points = n_mesh_intervals[0] * colloc_points_unshifted[0].size + 1
    start = ode_models[0].n_par
    stop = start + ode_models[0].n_dim * n_points
    y = q[start:stop].reshape((ode_models[i].n_dim, n_points), order="F")
    start = stop
    stop = start + n_mesh_intervals[0]
    mesh_points = np.pad(q[start:stop].cumsum(), (1, 0))
    period06 = mesh_points[-1]
    y_interp = util.interpolate(y, mesh_points, t_ab * period06, colloc_points_unshifted[0])
    start = stop

    std = 2 / 9 / np.sqrt(5)
    B_bound = y_interp[7:].sum(axis=0)
    B_mean = B_bound.mean()
    B_bound = B_bound - B_mean
    B_scale = np.std(B_bound)
    B_bound = B_bound / B_scale
    E += jax.scipy.integrate.trapezoid((B_bound - obs_kai_ab[:, 1])**2 / (2 * std**2), x=t_ab)
    
    std = 2 / 3 / np.sqrt(5)
    A_bound = ode_models[0].conservation_law[1, 1:-1]@y_interp
    A_bound = A_bound - A_bound.mean()
    A_bound = A_bound / np.std(A_bound)
    E += jax.scipy.integrate.trapezoid((A_bound - obs_kai_ab[:, 0])**2 / (2 * std**2), x=t_ab)

    t_phos = np.linspace(0, 1, obs_kai_phos.shape[0])
    y_interp = util.interpolate(y, mesh_points, t_phos * period06, colloc_points_unshifted[0])
    pT = y_interp[1:3].sum(axis=0)
    pD = y_interp[np.array([3, 4, 7, 8, 11, 12])].sum(axis=0)
    pS = y_interp[np.array([5, 6, 9, 10, 13, 14])].sum(axis=0)
    E += jax.scipy.integrate.trapezoid(3000 * (pT - obs_kai_phos[:, 0])** 2 / 2, x=t_phos)
    E += jax.scipy.integrate.trapezoid(3000 * (pS - obs_kai_phos[:, 1])** 2 / 2, x=t_phos)
    E += jax.scipy.integrate.trapezoid(3000 * (pD - obs_kai_phos[:, 2])** 2 / 2, x=t_phos)

    #t_autophos = obs_kaiac_autophos[:, 0]
    #kai_ac = model.KaiAC_nondim(k[ode_models[0].ind_C2_rxns[1:] - 1], a0=ode_models[0].a0)
    #term = diffrax.ODETerm(kai_ac.f)
    #solver = diffrax.Kvaerno4()
    #stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)
    #saveat = diffrax.SaveAt(ts=obs_kaiac_autophos[:, 0] * period06)
    #sol = diffrax.diffeqsolve(term, solver, t0=0, t1=t_autophos[-1] * period06, dt0=None, y0=np.full(kai_ac.n_dim, 1e-9), saveat=saveat,
    #                          stepsize_controller=stepsize_controller, args=k[ode_models[0].ind_C2_rxns[1:] - 1], throw=False)
    #pT = sol.ys[:, 1:3].sum(axis=1)
    #pD = sol.ys[:, 3:5].sum(axis=1)
    #pS = sol.ys[:, 5:7].sum(axis=1)
    #E += jax.scipy.integrate.trapezoid(3000 * (pT - obs_kaiac_autophos[:, 1])** 2 / 2, x=t_autophos)
    #E += jax.scipy.integrate.trapezoid(3000 * (pS - obs_kaiac_autophos[:, 2])** 2 / 2, x=t_autophos)
    #E += jax.scipy.integrate.trapezoid(3000 * (pD - obs_kaiac_autophos[:, 3])** 2 / 2, x=t_autophos)
    
    k_autodephos = k[ode_models[0].ind_C2_rxns[np.array([1, 2, 6, 13])] - 1]
    traj_autodephos = dephos(obs_kaic_autodephos[:, 0] * period06, obs_kaic_autodephos[0, 1:], k_autodephos)
    E += jax.scipy.integrate.trapezoid(3000 * (traj_autodephos[0] - obs_kaic_autodephos[:, 1])** 2 / 2, x=obs_kaic_autodephos[:, 0])
    E += jax.scipy.integrate.trapezoid(3000 * (traj_autodephos[1] - obs_kaic_autodephos[:, 2])** 2 / 2, x=obs_kaic_autodephos[:, 0])
    E += jax.scipy.integrate.trapezoid(3000 * (traj_autodephos[2] - obs_kaic_autodephos[:, 3])** 2 / 2, x=obs_kaic_autodephos[:, 0])
    
    n_points = n_mesh_intervals[1] * colloc_points_unshifted[1].size + 1
    stop = start + ode_models[1].n_dim * n_points
    y = q[start:stop].reshape((ode_models[1].n_dim, n_points), order="F")
    start = stop
    stop = start + n_mesh_intervals[1]
    mesh_points = np.pad(q[start:stop].cumsum(), (1, 0))
    period12 = mesh_points[-1]
    y_interp = util.interpolate(y, mesh_points, t_ab * period12, colloc_points_unshifted[1])

    std = 2 / 9 / np.sqrt(5)
    B_bound = y_interp[7:].sum(axis=0)
    B_bound = B_bound - B_mean
    B_bound = B_bound / B_scale
    E += jax.scipy.integrate.trapezoid((B_bound - obs_kai_ab[:, 1])**2 / (2 * std**2), x=t_ab)
    E += 100 * (period06 / period12 - 1)**2 / 2
    
    start = stop
    n_points = n_mesh_intervals[2] * colloc_points_unshifted[2].size + 1
    stop = start + ode_models[2].n_dim * n_points
    y = q[start:stop].reshape((ode_models[2].n_dim, n_points), order="F")
    start = stop
    stop = start + n_mesh_intervals[2]
    mesh_points = np.pad(q[start:stop].cumsum(), (1, 0))
    period18 = mesh_points[-1]
    y_interp = util.interpolate(y, mesh_points, t_ab * period18, colloc_points_unshifted[2])

    std = 2 / 9 / np.sqrt(5)
    B_bound = y_interp[7:].sum(axis=0)
    B_bound = B_bound - B_mean
    B_bound = B_bound / B_scale
    E += jax.scipy.integrate.trapezoid((B_bound - obs_kai_ab[:, 1])**2 / (2 * std**2), x=t_ab)
    E += 100 * (period06 / period18 - 1.1)**2 / 2
    
    start = stop
    n_points = n_mesh_intervals[3] * colloc_points_unshifted[3].size + 1
    stop = start + ode_models[3].n_dim * n_points
    y = q[start:stop].reshape((ode_models[3].n_dim, n_points), order="F")
    start = stop
    stop = start + n_mesh_intervals[3]
    mesh_points = np.pad(q[start:stop].cumsum(), (1, 0))
    period_ATPfrac025 = mesh_points[-1]
    y_interp = util.interpolate(y, mesh_points, t_phos * period_ATPfrac025, colloc_points_unshifted[3])
    pT = y_interp[1:3].sum(axis=0)
    pD = y_interp[np.array([3, 4, 7, 8, 11, 12])].sum(axis=0)
    pS = y_interp[np.array([5, 6, 9, 10, 13, 14])].sum(axis=0)
    E += jax.scipy.integrate.trapezoid(3000 * (pT - obs_kai_phos[:, 3])** 2 / 2, x=t_phos)
    E += jax.scipy.integrate.trapezoid(3000 * (pS - obs_kai_phos[:, 4])** 2 / 2, x=t_phos)
    E += jax.scipy.integrate.trapezoid(3000 * (pD - obs_kai_phos[:, 5])** 2 / 2, x=t_phos)
    E += 100 * (period06 / period_ATPfrac025 - 1)**2 / 2
    
    return E

@partial(jax.jit, static_argnames=("n_mesh_intervals",))
def kai_dae_log_bvp_potential_mm(q, ode_model, colloc_points_unshifted=util.gauss_points_4, n_mesh_intervals=60):
    
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
    y_interp = util.interpolate(y, mesh_points, t, colloc_points_unshifted)
    kaiB = np.exp(y_interp[8:-1]).sum(axis=0)
    kaiB = kaiB - kaiB.mean()
    kaiB = kaiB / np.std(kaiB)
    E += jax.scipy.integrate.trapezoid((kaiB - obs_mean)**2 / (2 * std**2), x=t)
    
    y_smooth = util.weighted_average_periodic_smoothing(y[:, :-1].T)
    E += 10 * np.sum((y_smooth - y[:, :-1].T)**2)

    return E

@jax.jit
def kai_bvp_potential(q, ode_model, mesh_points):
    
    n_mesh_intervals = mesh_points.size - 1
    n_points = n_mesh_intervals * util.gauss_points_4.size + 1
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
    y_interp = util.interpolate(y, mesh_points, t, colloc_points_unshifted)
    kaiB = y_interp[7:].sum(axis=0)
    kaiB = kaiB - kaiB.mean()
    kaiB = kaiB / np.std(kaiB)
    E += jax.scipy.integrate.trapezoid((kaiB - obs_mean)**2 / (2 * std**2), x=t)
    
    return E

dt = 1e-2
friction = 1e-1

n_points = n_mesh_intervals * colloc_points_unshifted.size + 1
x = np.load("kai_lc_%d_%d.npy"%(argp.iter - 1, argp.process))[-1]

n_steps = 10000
thin = 100

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
#                                 A=lgvn.rattle_drift_bvp_mm, nlsol=nonlinear_solver.quasi_newton_bvp_symm_broyden, linsol=linear_solver.lq_ortho_proj_bvp,
#                                 max_newton_iter=100, constraint_tol=1e-9, args=args, metropolize=True, reversibility_tol=1e-6)

#traj_kai_lc, key_lc = lgvn.gOBABO(q0, p0, l0, dt, friction, n_steps, thin, prng_key, potential, resid, jac, nlsol=nonlinear_solver.quasi_newton_rattle_symm_broyden, 
#                                  max_newton_iter=100, constraint_tol=1e-9, args=args, metropolize=True, reversibility_tol=1e-6)

#kai = model.KaiABC_nondim(par=np.zeros(model.KaiABC_nondim.n_par))
#q0 = x[:kai.n_par + kai.n_dim * n_points + 1 + n_mesh_intervals - 1]
#p0 = x[q0.size:2 * q0.size] 
#args = (kai, colloc_points_unshifted)
#potential = lambda *args:kai_bvp_potential_mm(*args, n_mesh_intervals=n_mesh_intervals)
#resid = lambda *args:defining_systems.periodic_bvp_mm_colloc_resid(*args, n_mesh_intervals=n_mesh_intervals)
#jac = lambda *args:defining_systems.periodic_bvp_mm_colloc_jac(*args, n_mesh_intervals=n_mesh_intervals)
#n_constraints = resid(q0, *args).size
#l0 = x[2 * q0.size:2 * q0.size + n_constraints]
#traj_kai_lc, key_lc = lgvn.gOBABO(q0, p0, l0, dt, friction, n_steps, thin, prng_key, potential, resid, jac, nlsol=nonlinear_solver.quasi_newton_bvp_symm_broyden, linsol=linear_solver.lq_ortho_proj_bvp,
#                                  max_newton_iter=100, constraint_tol=1e-9, args=args, metropolize=True, reversibility_tol=1e-6)

#kai = model.KaiABC_DAE_log_nondim(par=np.zeros(model.KaiABC_nondim.n_par))
#q0 = x[:kai.n_par + kai.n_dim * n_points + 1 + n_mesh_intervals - 1]
#p0 = x[q0.size:2 * q0.size] 
#args = (kai, util.gauss_points_4)
#potential = kai_dae_log_bvp_potential_mm
#resid = lambda *args:defining_systems.periodic_bvp_mm_colloc_resid(*args, n_mesh_intervals=n_mesh_intervals)
#jac = lambda *args:defining_systems.periodic_bvp_mm_colloc_jac(*args, n_mesh_intervals=n_mesh_intervals)
#n_constraints = resid(q0, *args).size
#l0 = x[2 * q0.size:2 * q0.size + n_constraints]
#traj_kai_lc, key_lc = lgvn.gOBABO(q0, p0, l0, dt, friction, n_steps, thin, prng_key, potential, resid, jac, nlsol=nonlinear_solver.quasi_newton_bvp_symm_broyden, linsol=linear_solver.lq_ortho_proj_bvp,
#                                  max_newton_iter=100, constraint_tol=1e-9, args=args, metropolize=True, reversibility_tol=1e-6)

#kai = model.KaiABC_DAE_log_nondim(par=np.zeros(model.KaiABC_nondim.n_par))
#q0 = x[:kai.n_par + kai.n_dim * n_points + 1 + n_mesh_intervals - 1]
#p0 = x[q0.size:2 * q0.size] 
#args = (kai, util.midpoint)
#potential = lambda *args:kai_dae_log_bvp_potential_mm(*args, n_mesh_intervals=n_mesh_intervals)
#resid = lambda *args:defining_systems.periodic_bvp_mm_colloc_resid(*args, n_mesh_intervals=n_mesh_intervals)
#jac = lambda *args:defining_systems.periodic_bvp_mm_colloc_jac(*args, n_mesh_intervals=n_mesh_intervals)
#n_constraints = resid(q0, *args).size
#l0 = x[2 * q0.size:2 * q0.size + n_constraints]
#traj_kai_lc, key_lc = lgvn.gOBABO(q0, p0, l0, dt, friction, n_steps, thin, prng_key, potential, resid, jac, nlsol=nonlinear_solver.quasi_newton_bvp_symm_broyden, linsol=linear_solver.lq_ortho_proj_bvp,
#                                  max_newton_iter=100, constraint_tol=1e-9, args=args, metropolize=True, reversibility_tol=1e-6)

kai06 = model.KaiABC_nondim(par=np.zeros(model.KaiABC_nondim.n_par), a0=6/35)
kai12 = model.KaiABC_nondim(par=np.zeros(model.KaiABC_nondim.n_par), a0=12/35)
kai18 = model.KaiABC_nondim(par=np.zeros(model.KaiABC_nondim.n_par), a0=18/35)
kai_ATPfrac025 = model.KaiABC_nondim(par=np.zeros(model.KaiABC_nondim.n_par), ATPfrac=0.25)
prng_key = x[-1:]
q0 = x[:kai06.n_par + 4 * (kai06.n_dim * n_points + n_mesh_intervals)]
ode_models = (kai06, kai12, kai18, kai_ATPfrac025)
p0 = x[q0.size:2 * q0.size] 
colloc_points_unshifted = tuple(colloc_points_unshifted for _ in range(4))
n_mesh_intervals = tuple(60 for _ in range(4))
potential = partial(kai_bvp_potential_mm_4cond, n_mesh_intervals=n_mesh_intervals)
potential_1 = partial(kai_bvp_potential_mm_4cond_1, n_mesh_intervals=n_mesh_intervals)
resid = partial(defining_systems.bvp_mm_colloc_resid_multi_shared_k, n_mesh_intervals=n_mesh_intervals)
jac = partial(defining_systems.bvp_mm_colloc_jac_multi_shared_k, n_mesh_intervals=n_mesh_intervals)
nlsol = nonlinear_solver.quasi_newton_bvp_multi_shared_k_symm_broyden_1
linsol = linear_solver.lq_ortho_proj_bvp_multi_shared_k_1
n_constraints = resid(q0, ode_models=ode_models).size
l0 = x[2 * q0.size:2 * q0.size + n_constraints]
traj_kai_lc = lgvn.sample((q0, p0, l0, None, None, prng_key), dt, n_steps, friction=friction, thin=thin, potential=potential, stepper=lgvn.gOBABO, constraint=resid, jac_constraint=jac, nlsol=nlsol, linsol=linsol, max_newton_iter=100, constraint_tol=1e-8, metropolize=True, reversibility_tol=1e-6, ode_models=ode_models, colloc_points_unshifted=colloc_points_unshifted, print_acceptance=True, force=jax.jacrev(potential_1))

np.save("kai_lc_%d_%d.npy"%(argp.iter, argp.process), traj_kai_lc)
