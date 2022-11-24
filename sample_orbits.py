from orbit_sampler import sample
from model import KaiODE
from collocation import colloc
import jax
import jax.numpy as np
import argparse
import numpy
jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, required=True)
parser.add_argument("-o", type=str, required=True)
parser.add_argument("-n", type=int, required=True)
parser.add_argument("-L", type=int, required=True)
parser.add_argument("-ref", type=str, required=True)
parser.add_argument("-seed", type=int)
parser.add_argument("-thin", type=int, default=1)
parser.add_argument("-met", type=int, default=1)
parser.add_argument("-dt", type=float, default=5e-4)
parser.add_argument("-fric", type=float, default=1e-1)
args = parser.parse_args()

n_mesh_point = 60
reference = np.load(args.ref)
bounds = np.vstack([reference[-1, :KaiODE.n_react] - 4.6, reference[-1, :KaiODE.n_react] + 4.6]).T

init = np.load(args.i)
position = init[-1, :KaiODE.n_react]

y_size = (n_mesh_point * colloc.n_colloc_point + 1) * (KaiODE.n_dim - KaiODE.n_conserve)
y = init[-1,  2 * KaiODE.n_react + 1:2 * KaiODE.n_react + 1 + y_size]
y = y.reshape((KaiODE.n_dim - KaiODE.n_conserve), y.size // (KaiODE.n_dim - KaiODE.n_conserve), order="F")

period = init[-1, 2 * KaiODE.n_react + 1 + y_size + 1]

result = sample(position, y, period, bounds, langevin_trajectory_length=args.L, dt=args.dt, friction=args.fric, maxiter=args.n, seed=args.seed, thin=args.thin, metropolize=args.met)
np.save(args.o, result)
