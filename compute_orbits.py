from periodic_orbits import *
from mpi4py import MPI
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, required=True)
parser.add_argument("-start", type=int)
parser.add_argument("-stop", type=int)
parser.add_argument("-stride", type=int, default=1);
parser.add_argument("-o", type=str, required=True)
args = parser.parse_args()

traj = np.load(args.i)[args.start:args.stop:args.stride, :]

if args.start is None:
    args.start = 0
if args.stop is None:
    args.stop = traj.shape[0]

out = numpy.zeros(((args.stop - args.start) // args.stride, 67))
partial = numpy.zeros_like(out)

comm = MPI.COMM_WORLD
ntasks = partial.shape[0] // comm.Get_size()
segment_size = (args.stop - args.start) // comm.Get_size()
i = comm.Get_rank() * ntasks

while i < (comm.Get_rank() + 1) * ntasks:

    j = segment_size *  comm.Get_rank() + (i - comm.Get_rank() * ntasks) * args.stride
    lc = find_limit_cycle(np.exp(log_rates(traj[j, :])), np.exp(q[50:]))
    partial[i, :50] = lc[-1] * np.exp(log_rates(traj[j, :]))
    partial[i, 50:] = lc[:-1]
    i += 1

comm.Allreduce([partial, MPI.DOUBLE], [out, MPI.DOUBLE], op=MPI.SUM)

if comm.Get_rank() == 0:

    np.save(args.o, out)
