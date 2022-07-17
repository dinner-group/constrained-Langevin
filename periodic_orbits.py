from sampler import *
from model import *
import scipy
import scipy.integrate
import scipy.signal
import warnings

def find_limit_cycle(q, y_atol=1e-3, period_rtol=1e-4):
    """Perform long time integration and hope that trajectory is attracted to a limit cycle.
    Returns point on the limit cycle and estimated period if successful."""

    model = KaiODE(np.exp(log_rates(q)))
    y0 = np.exp(q[50:])
    mask = y0 > 5e-2

    if np.sum(mask * (cC > 0)) >= 2:
        mask = mask * (cC > 0)
    elif np.sum(mask * (cA > 0)) >= 2:
        mask = mask * (cA > 0)
    else:
        warnings.warn("Concentrations too low")
        return np.full_like(y0, np.nan), np.nan

    ind = np.arange(mask.shape[0])[mask][:2]
    y0 = y0.at[ind].add(np.array([-1e-2, 1e-2]))
    
    J_fp = model.jac_red(0, y0[1:-1])
    evals = np.linalg.eigvals(J_fp)
    evals = evals[np.argsort(evals.real)]
    evals = evals[np.argsort(evals.imag, kind="stable")]
    tau0 = 100

    if evals[-1].imag > 0:
        tau0 = 2 * np.pi / np.abs(evals[-1].imag)
    
    try:
        int0 = scipy.integrate.solve_ivp(model.f, jac=model.jac, t_span=(0, 10 * tau0), y0=y0, t_eval=np.linspace(6 * tau0, 10 * tau0, int(1 / period_rtol)), method="LSODA")
    except:
        warnings.warn("Integration failed")
        return np.full_like(y0, np.nan), np.nan
    
    if np.linalg.norm(model.f(0, int0.y[:, -1])) < 1e-9:
        warnings.warn("Converged to fixed point")
        return int0.y[:, -1], np.inf
   
    a_whiten = (int0.y[-1, :] - int0.y[-1, :].mean()) / np.std(int0.y[-1, :])
    autocorr_a = scipy.signal.correlate(a_whiten, a_whiten) / int0.t.shape[0]
    autocorr_peaks = scipy.signal.find_peaks(autocorr_a, prominence=0.3)

    if autocorr_peaks[0].shape < 5:
        warnings.warn("Failed to converge")
        return np.full_like(y0, np.nan), np.nan

    tau1 = np.median(autocorr_peaks[0][1:] - autocorr_peaks[0][:-1]) * (int0.t[-1] - int0.t[-2])

    return int0.y[:, -1], tau1
