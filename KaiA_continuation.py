from periodic_orbits import *
from mpi4py import MPI
import argparse
import numpy

@jax.jit
def f_arclength(t, y, model):

    ydot = np.zeros_like(y)
    ydot = ydot.at[:17].set(model.f(t, y))
    ydot = ydot.at[17].set(np.linalg.norm(ydot[:17]))

jac_arclength = jax.jit(jax.jacfwd(f_arclength))

def arclength(rates, y0):

    model = KaiODE(rates)
    return scipy.integrate.solve_ivp(f_arclength, jac=jac_arclength, t_span=(0, 1), method="LSODA", atol=1e-9, rtol=1e-6).y[-1, -1]

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, required=True)
parser.add_argument("-o", type=str, required=True)
args = parser.parse_args()

indata = np.load(args.i)
outdata = np.zeros((indata.shape[0], 3))
partial = np.zeros_like(outdata)

i = comm.Get_rank()

while i < indata.shape[0]:

    partial[i, 0] = find_limit_cycle(indata[i, :50], indata[i, 50:].at[-1].add(1e-2), tau0=1)[-1]
    partial[i, 1] = find_limit_cycle(indata[i, :50], indata[i, 50:].at[-1].add(1e-1), tau0=1)[-1]

    try:
        partial[i, 2] = arclength(indata[i, :50], indata[i, 50:])
    except:
        partial[i, 2] = np.nan

    i += comm.Get_size()

comm.Allreduce([partial, MPI.DOUBLE], [out, MPI.DOUBLE], op=MPI.SUM)
