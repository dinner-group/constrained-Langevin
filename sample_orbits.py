from orbit_sampler import sample
import model
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
parser.add_argument("-rst", type=str)
parser.add_argument("-seed", type=int)
parser.add_argument("-thin", type=int, default=1)
parser.add_argument("-met", type=int, default=1)
parser.add_argument("-dt", type=float, default=5e-4)
parser.add_argument("-fric", type=float, default=1e-1)
parser.add_argument("-model", type=str, default="KaiODE")
parser.add_argument("-lbound", type=float, default=-4.6)
parser.add_argument("-ubound", type=float, default=4.6)
args = parser.parse_args()

models = {"KaiODE":model.KaiODE, "Brusselator":model.Brusselator}

if args.rst is None:
    args.rst = args.i

n_mesh_point = 60
reference = np.load(args.ref)
bounds = np.vstack([reference[-1, :models[args.model].n_react] + args.lbound, reference[-1, :models[args.model].KaiODE.n_react] + args.ubound]).T

init = np.load(args.i)
position = init[-1, :models[args.model].n_react]

y_size = (n_mesh_point * colloc.n_colloc_point + 1) * (models[args.model].n_dim - models[args.model].n_conserve)
y = init[-1,  2 * models[args.model].n_react + 1:2 * models[args.model].n_react + 1 + y_size]
y = y.reshape((models[args.model].n_dim - models[args.model].n_conserve), y.size // (models[args.model].n_dim - models[args.model].n_conserve), order="F")

period = init[-1, 2 * models[args.model].n_react + 1 + y_size + 1]

result, accepted, rejected, failed = sample(models[args.model], position, y, period, bounds, langevin_trajectory_length=args.L, dt=args.dt, friction=args.fric, maxiter=args.n, seed=args.seed, thin=args.thin, metropolize=args.met)

if failed / (accepted + rejected + failed) > 0.9:
    restart = np.load(args.i)
    index = numpy.random.randint(restart.shape[0])
    result[-1] = restart[index]

np.save(args.o, result)
