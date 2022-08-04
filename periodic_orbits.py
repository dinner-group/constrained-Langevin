from sampler import *
from model import *
import scipy
import scipy.integrate
import scipy.signal
import warnings

def find_limit_cycle(rates, y0, tau0=None, perturb=True, period_rtol=1e-5, n_periods_lower=16, n_periods_upper=20):
    """Perform long time integration and hope that trajectory is attracted to a limit cycle.
    Returns point on the limit cycle and estimated period if successful."""

    model = KaiODE(rates)
   
    if perturb:

        mask = y0 > 5e-2
        if np.sum(mask * (cC > 0)) >= 2:
            mask = mask * (cC > 0)
        elif np.sum(mask * (cA > 0)) >= 2:
            mask = mask * (cA > 0)
        else:
            warnings.warn("Invalid concentrations")
            return np.concatenate([np.full_like(y0, np.nan), np.array([np.nan])])

        ind = np.arange(mask.shape[0])[mask][:2]
        y0 = y0.at[ind].add(np.array([-1e-2, 1e-2]))

    if tau0 is None:

        J_fp = model.jac_red(0, y0[1:-1])
        evals = np.linalg.eigvals(J_fp)
        evals = evals[np.argsort(evals.real)]
        evals = evals[np.argsort(evals.imag, kind="stable")]
        if evals[-1].imag > 0:
            tau0 = 2 * np.pi / np.abs(evals[-1].imag)
        else:
            tau0 = 100
    try:
        int0 = scipy.integrate.solve_ivp(model.f, jac=model.jac, t_span=(0, n_periods_upper * tau0), y0=y0, t_eval=np.linspace(n_periods_lower * tau0, n_periods_upper * tau0, int(1 / period_rtol)), method="LSODA")
    except:
        warnings.warn("Integration failed")
        return np.concatenate([np.full_like(y0, np.nan), np.array([np.nan])])

    if not int0.success:
        warnings.warn("Integration failed")
        return np.concatenate([np.full_like(y0, np.nan), np.array([np.nan])])
    
    if np.linalg.norm(model.f(0, int0.y[:, -1])) < 1e-9:
        warnings.warn("Converged to fixed point")
        return np.concatenate([int0.y[:, -1], np.array([np.inf])])
   
    a_whiten = (int0.y[-1, :] - int0.y[-1, :].mean()) / np.std(int0.y[-1, :])
    autocorr_a = scipy.signal.correlate(a_whiten, a_whiten) / int0.t.shape[0]
    autocorr_peaks = scipy.signal.find_peaks(autocorr_a, prominence=0.3)

    if autocorr_peaks[0].shape[0] < 5:
        warnings.warn("Failed to converge")
        return np.concatenate([np.full_like(y0, np.nan), np.array([np.nan])])

    tau1 = np.median(autocorr_peaks[0][1:] - autocorr_peaks[0][:-1]) * (int0.t[-1] - int0.t[-2])

    return np.concatenate([int0.y[:, -1], np.array([tau1])])

@jax.jit
def f_M(t, y, model, tau):

    return tau * np.concatenate([model.f_red(t, y[:15]), np.ravel(model.jac_red(t, y[:15])@y[15:].reshape((15, 15), order="F"), order="F")])

jac_M = jax.jit(jax.jacfwd(f_M, argnums=1))

def single_shooting(rates, y0, tau0, tol=1e-9, maxiter=10):

    model = KaiODE(rates)
    porbit0 = scipy.integrate.solve_ivp(model.f_red, jac=model.jac_red, t_span=(0, tau0), y0=y0, t_eval=np.linspace(0, tau0, 1000), method="LSODA", atol=1e-9, rtol=1e-6)

    if not porbit0.success:

        return np.concatenate([y0, np.array([tau0])])

    phase_species = np.argmax(np.max(porbit.y, axis=1) - np.min(porbit.y, axis=1))
    porbit1 = scipy.integrate.solve_ivp(model.f_red, jac=model.jac_red, t_span=(0, tau0), y0=porbit0.y[:, np.argmax(porbit0.y[phase_species, :])], method="LSODA", atol=1e-9, rtol=1e-6)

    if not porbit1.success:

        return np.concatenate([y0, np.array([tau0])])

    err = np.linalg.norm(np.concatenate([porbit1.y[:, -1] - porbit1[:, 0], np.array([model.f_red(porbit1.y[phase_species, 0])])]))
    i = 0
    tau = tau0
    ya = porbit1.y[:, 0]
    yb = porbit1.y[:, -1]
    err0 = np.linalg.norm(np.concatenate([yb - ya, np.array(model.f_red(0, ya)[phase_species])]))

    while err > tol and i < maxiter:

        M0 = numpy.zeros((ya.shape[0], ya.shape[0] + 1))
        M0[:ya.shape[0]] = ya
        M0[:, 1:][np.diag_indices(ya.shape[0])] = 1
        M0 = np.ravel(M0, order="F")
        int_M = scipy.integrate.solve_ivp(f_M, jac=jac_M, t_span=(0, 1), y0=M0, args=(tau,), method="LSODA", atol=1e-9, rtol=1e-6)

        J = numpy.zeros((ya.shape[0] + 1, ya.shape[0] + 1))
        J[:ya.shape[0], :ya.shape[0]] = int_M.y[15:, -1].reshape((15, 15), order="F")
        J[np.diag_indices(ya.shape[0])] -= 1
        J[:ya.shape[0], -1] = model.f_red(0, yb)
        J[-1, :ya.shape[0]] = model.jac_red(0, ya)[phase_species, :]

        resid = np.concatenate([yb - ya, np.array([model.f_red(0, ya)[phase_species]])])
        err = np.linalg.norm(resid)

        dx = np.linalg.solve(J, -resid)
        ya += dx[:ya.shape[0]]
        tau += dx[-1]

    if i >= maxiter and err > err0:

        return np.concatenate([y0, np.array([tau0])])

    return np.concatenate([ya, np.array([tau])])
