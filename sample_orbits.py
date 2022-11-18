from orbit_sampler import *
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
bounds = np.vstack([reference[:model.KaiODE.n_react] - 4.6, reference[:model.KaiODE.n_react] + 4.6]).T

init = np.load(args.i)
position = init[-1, :model.KaiODE.n_react]
y = init[-1, model.KaiODE.n_react:model.KaiODE.n_react + (n_mesh_point * collocation.colloc.n_colloc_point) * (model.KaiODE.n_dim - model.KaiODE.n_conserve)]
period = init[-1, model.KaiODE.n_react + (n_mesh_point * collocation.colloc.n_colloc_point) * (model.KaiODE.n_dim - model.KaiODE.n_conserve) + 1]

result = sample_orbits(position, y, period, bounds, args.L)
np.save(args.o, result)
