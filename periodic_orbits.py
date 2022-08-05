from sampler import *
from model import *
import scipy
import scipy.integrate
import scipy.signal
import warnings

def find_limit_cycle(rates, y0, tau0=None, perturb=True, period_rtol=1e-5, n_periods_lower=12, n_periods_upper=20):
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
        int0 = scipy.integrate.solve_ivp(model.f, jac=model.jac, t_span=(0, n_periods_upper * tau0), y0=y0, t_eval=np.linspace(n_periods_lower * tau0, n_periods_upper * tau0, int(1 / period_rtol)), method="BDF", atol=1e-9, rtol=1e-6)
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
def f_M(t, y, tau, model):

    return tau * np.concatenate([model.f_red(t, y[:15]), (model.jac_red(t, y[:15])@y[15:].reshape((15, 15), order="F")).ravel(order="F")])

jac_M = jax.jit(jax.jacfwd(f_M, argnums=1))

def newton_picard(M, ya, yb, tau, phase_species, model, eps=1e-6):
    
    M_evals, M_evecs = np.linalg.eig(M)
    V = M_evecs[:, M_evals > eps]
    Vp, _ = np.linalg.qr(V)
    S = Vp.T@M@Vp
    r = yb - ya
    dq = r - Vp@(Vp.T@r)
    tmp = (M@dq + r)
    dq = tmp - Vp@(Vp.T@tmp)
    c = model.jac_red(0, ya)[phase_species, :]
    
    J = np.zeros((S.shape[0] + 1, S.shape[1] + 1))
    J = J.at[:-1, :-1].set((S - np.identity(S.shape[0])))
    J = J.at[:-1, -1].set(model.f_red(0, yb)@Vp)
    J = J.at[-1, :-1].set(c@Vp)
    
    resid = np.concatenate([-Vp.T@(r + M@dq), np.array([-model.f_red(0, ya)[phase_species] - c@dq])]).real
    e = np.linalg.solve(J, resid)
    
    return np.concatenate([ya + Vp@e[:-1] + dq, np.array([tau + e[-1]])]).real

def newton(M, ya, yb, tau, phase_species, model):

    J = np.zeros((M.shape[0] + 1, M.shape[0] + 1))
    J = J.at[:-1, :-1].set(M)
    J = J.at[np.diag_indices(M.shape[0])].set(J[np.diag_indices(M.shape[0])] - 1)
    J = J.at[:-1, -1].set(model.f_red(0, yb))
    J = J.at[-1, :-1].set(model.jac_red(0, ya)[phase_species, :])

    resid = np.concatenate([yb - ya, np.array([model.f_red(0, ya)[phase_species]])])
    dx = np.linalg.solve(J, -resid)

    return np.concatenate([ya + dx[:-1], np.array([tau + dx[-1]])])

def single_shooting(rates, y0, tau0, tol=1e-9, maxiter=10):

    model = KaiODE(rates)
    porbit0 = scipy.integrate.solve_ivp(model.f_red, jac=model.jac_red, t_span=(0, tau0), y0=y0, t_eval=np.linspace(0, tau0, 10000), method="BDF", atol=1e-9, rtol=1e-6)

    if not porbit0.success:

        return np.concatenate([y0, np.array([tau0])])

    phase_species = np.argmax(np.max(porbit0.y, axis=1) - np.min(porbit0.y, axis=1))
    i_phase = np.argmin(porbit0.y[phase_species, :])
    porbit1 = scipy.integrate.solve_ivp(model.f_red, jac=model.jac_red, t_span=(0, tau0), y0=porbit0.y[:, i_phase], method="BDF", atol=1e-9, rtol=1e-6)

    if not porbit1.success:

        return np.concatenate([y0, np.array([tau0])])

    i = 0
    tau = tau0
    ya = porbit1.y[:, 0]
    yb = porbit1.y[:, -1]
    err = np.linalg.norm(np.concatenate([yb - ya, np.array([model.f_red(0, ya)[phase_species]])]))
    err0 = err

    while err > tol and i < maxiter:

        M0 = np.zeros((ya.shape[0], ya.shape[0] + 1))
        M0 = M0.at[:, 0].set(ya)
        M0 = M0.at[:, 1:].set(np.identity(ya.shape[0]))
        M0 = np.ravel(M0, order="F")

        int_M = scipy.integrate.solve_ivp(f_M, jac=jac_M, t_span=(0, 1), y0=M0, args=(tau, model), method="BDF", atol=1e-9, rtol=1e-6)

        M = int_M.y[15:, -1].reshape((15, 15), order="F")
        yb = int_M.y[:15, -1]

        resid = np.concatenate([yb - ya, np.array([model.f_red(0, ya)[phase_species]])])
        err = np.linalg.norm(resid)
        x = newton(M, ya, yb, tau, phase_species, model)

        ya = x[:15]
        tau = x[-1]
        i += 1

    if i >= maxiter and err > err0:

        return np.concatenate([y0, np.array([tau0])])

    return np.concatenate([ya, np.array([tau])])
