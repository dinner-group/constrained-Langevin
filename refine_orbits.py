from periodic_orbits import *
from mpi4py import MPI
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, required=True)
parser.add_argument("-start", type=int, default=0)
parser.add_argument("-stop", type=int, default=-1)
parser.add_argument("-stride", type=int, default=1);
parser.add_argument("-o", type=str, required=True)
args = parser.parse_args()

traj = np.load(args.i)[args.start:args.stop:args.stride]

if args.stop == -1:
    args.stop = traj.shape[0]

out = numpy.zeros(((args.stop - args.start) // args.stride, 67))
partial = numpy.zeros_like(out)

comm = MPI.COMM_WORLD
i = comm.Get_rank()

while i < (args.stop - args.start) // args.stride:

    lc0 = find_limit_cycle(np.exp(log_rates(traj[i, :])), np.exp(traj[i, 50:]))
    tau1 = lc0[-1]
    lc1 = single_shooting(np.exp(log_rates(traj[i, :])), lc0[1:-2], tau1)
    partial[i, :50] = lc1[-1] * np.exp(log_rates(traj[i, :]))
    partial[i, 51:-1] = lc1[:-1]
    partial[i, 50] = KaiODE.cC@lc0[:-1] - KaiODE.cC[1:-1]@lc1[:-1]
    partial[i, -1] = KaiODE.cA@lc0[:-1] - KaiODE.cA[1:-1]@lc1[:-1]
    i += comm.Get_size()

comm.Allreduce([partial, MPI.DOUBLE], [out, MPI.DOUBLE], op=MPI.SUM)

if comm.Get_rank() == 0:

    np.save(args.o, out)
