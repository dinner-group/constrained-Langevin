from periodic_orbits import *
from mpi4py import MPI
import scipy.signal
import argparse
import numpy

@jax.jit
def f_arclength(t, y, model):

    ydot = np.zeros_like(y)
    ydot = ydot.at[:17].set(model.f(t, y))
    ydot = ydot.at[17].set(np.linalg.norm(ydot[:17]))
    return ydot

jac_arclength = jax.jit(jax.jacfwd(f_arclength, argnums = 1))

def arclength(rates, y0):

    if np.any(np.isnan(rates)) or np.any(np.isnan(y0)):
        return np.nan

    model = KaiODE(rates)
    result = scipy.integrate.solve_ivp(f_arclength, jac=jac_arclength, t_span=(0, 1), y0=y0, args=(model,), t_eval=np.linspace(0, 1, 1000), method="LSODA", atol=1e-9, rtol=1e-6)

    if result.success:
        ydot = jax.lax.scan(lambda _, y:(_, f_arclength(0, y, model), 0, result.y.T)
        speed = ydot[-1, :]
        speed_peaks = scipy.signal.find_peaks(numpy.array(speed), width=0, height=float(np.quantile(speed, 0.99) / 4))
        return result.y[-1, -1], (result.t[1] - result.t[0]) * speed_peaks[1]["widths"].sum()
    else:
        return np.nan, np.nan

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, required=True)
parser.add_argument("-o", type=str, required=True)
args = parser.parse_args()

indata = np.load(args.i)
out = numpy.zeros((indata.shape[0], 7))
partial = numpy.zeros_like(out)

comm = MPI.COMM_WORLD
ntasks = indata.shape[0]
i = comm.Get_rank()

while i < ntasks:

    print(i)

    KaiA1en2 = find_limit_cycle(indata[i, :50], indata[i, 50:].at[-1].add(1e-2), tau0=1, perturb=False)
    KaiA1en1 = find_limit_cycle(indata[i, :50], indata[i, 50:].at[-1].add(1e-1), tau0=1, perturb=False)
    KaiA1e0 = find_limit_cycle(indata[i, :50], indata[i, 50:].at[-1].add(1), tau0=1, perturb=False)

    partial[i, 0] = KaiA1en2[-1]
    partial[i, 1] = KaiA1en1[-1]
    partial[i, 2] = KaiA1e0[-1]

    partial[i, np.array([3, 6])] = arclength(indata[i, :50], indata[i, 50:])
    partial[i, 4] = arclength(indata[i, :50], KaiA1en2[:-1])[0]
    partial[i, 5] = arclength(indata[i, :50], KaiA1en1[:-1])[0]

    i += comm.Get_size()

comm.Allreduce([partial, MPI.DOUBLE], [out, MPI.DOUBLE], op=MPI.SUM)

if comm.Get_rank() == 0:
    np.save(args.o, out)
