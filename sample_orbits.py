from orbit_sampler import sample
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, required=True)
parser.add_argument("-o", type=str, required=True)
parser.add_argument("-n", type=int, required=True)
parser.add_argument("-L", type=int, required=True)
parser.add_argument("-ref", type=str, required=True)
args = parser.parse_args()

n_mesh_point = 60
reference = np.load(args.ref)
bounds = np.vstack([reference[:KaiODE.n_react] - 4.6, reference[:KaiODE.n_react] + 4.6]).T

init = np.load(args.i)
position = init[-1, :KaiODE.n_react]
y = init[-1, KaiODE.n_react:KaiODE.n_react + (n_mesh_point * colloc.n_colloc_point) * (KaiODE.n_dim - KaiODE.n_conserve)]
period = init[-1, KaiODE.n_react + (n_mesh_point * colloc.n_colloc_point) * (KaiODE.n_dim - KaiODE.n_conserve) + 1]

result = sample(position, y, period, bounds, args.L)
np.save(args.o, result)
