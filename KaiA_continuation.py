from periodic_orbits import *
from mpi4py import MPI
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
    result = scipy.integrate.solve_ivp(f_arclength, jac=jac_arclength, t_span=(0, 1), y0=y0, args=(model,), method="LSODA", atol=1e-9, rtol=1e-6)

    if result.success:
       return result.y[-1, -1]
    else:
        return np.nan

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, required=True)
parser.add_argument("-o", type=str, required=True)
args = parser.parse_args()

indata = np.load(args.i)
out = numpy.zeros((indata.shape[0], 5))
partial = numpy.zeros_like(out)

comm = MPI.COMM_WORLD
ntasks = indata.shape[0]
i = comm.Get_rank()

while i < ntasks:

    print(i)

    KaiA1en2 = find_limit_cycle(indata[i, :50], indata[i, 50:].at[-1].add(1e-2), tau0=1)
    KaiA1en1 = find_limit_cycle(indata[i, :50], indata[i, 50:].at[-1].add(1e-1), tau0=1)

    partial[i, 0] = KaiA1en2[-1]
    partial[i, 1] = KaiA1en1[-1]

    try:
        partial[i, 2] = arclength(indata[i, :50], indata[i, 50:])
    except:
        partial[i, 2] = np.nan

    try:
        partial[i, 3] = arclength(indata[i, :50], KaiA1en2[:-1])
    except:
        partial[i, 3] = np.nan

    try:
        partial[i, 4] = arclength(indata[i, :50], KaiA1en1[:-1])
    except:
        partial[i, 4] = np.nan

    i += comm.Get_size()

comm.Allreduce([partial, MPI.DOUBLE], [out, MPI.DOUBLE], op=MPI.SUM)

if comm.Get_rank() == 0:
    np.save(args.o, out)
