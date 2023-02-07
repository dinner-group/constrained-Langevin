import orbit_sampler
import model
from collocation import colloc
import jax
import jax.numpy as np
from mpi4py import MPI
import mpi4jax
import argparse
import numpy
jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, required=True)
parser.add_argument("-o", type=str, required=True)
parser.add_argument("-n", type=int, required=True)
parser.add_argument("-L", type=int, required=True)
parser.add_argument("-ref", type=str, required=True)
parser.add_argument("-dyn", type=str, default="preconditioned_langevin")
parser.add_argument("-rst", type=str)
parser.add_argument("-seed", type=int)
parser.add_argument("-thin", type=int, default=1)
parser.add_argument("-met", type=int, default=1)
parser.add_argument("-dt", type=float, default=5e-4)
parser.add_argument("-fric", type=float, default=1e-1)
parser.add_argument("-model", type=str, default="KaiODE")
parser.add_argument("-lbound", type=float, default=-np.inf)
parser.add_argument("-ubound", type=float, default=np.inf)
parser.add_argument("-bounds", type=str)
parser.add_argument("-nmesh", type=int, default=60)
args = parser.parse_args()

models = {"KaiODE":(model.KaiODE, orbit_sampler.compute_energy_and_force_kai), "Brusselator":model.Brusselator, "Morris_Lecar":(model.Morris_Lecar, orbit_sampler.compute_energy_and_force_ml)}
dynamics = {"preconditioned_langevin":orbit_sampler.generate_langevin_trajectory_precondition, "preconditioned_random_walk":orbit_sampler.random_walk_metropolis_precondition}

comm = MPI.COMM_WORLD

if args.rst is None:
    args.rst = args.i

n_mesh_point = args.nmesh
reference = np.load(args.ref)

if args.bounds is not None:
    bounds = np.load(args.bounds)
else:
    bounds = np.vstack([reference[-1, :models[args.model][0].n_par] + args.lbound, reference[-1, :models[args.model][0].n_par] + args.ubound]).T

init = np.load(args.i)
position = init[-1, :, :models[args.model][0].n_par]

y_size = (n_mesh_point * colloc.n_colloc_point + 1) * (models[args.model][0].n_dim - models[args.model][0].n_conserve)
y = init[-1, :, 2 * models[args.model][0].n_par + 1:2 * models[args.model][0].n_par + 1 + y_size]

period = init[-1, :, 2 * models[args.model][0].n_par + 1 + y_size + 1]

result, accepted, rejected, failed = orbit_sampler.sample_mpi(*models[args.model], position, y, period, bounds, trajectory_length=args.L, comm=comm, dt=args.dt, friction=args.fric, maxiter=args.n, seed=args.seed, thin=args.thin, metropolize=args.met, dynamics=dynamics[args.dyn])

if comm.Get_rank() == 0:

    for i in range(failed.size):

        if failed[i] / (accepted[i] + rejected[i] + failed[i]) > 0.9:
            restart = np.load(args.i)
            r1 = numpy.random.randint(restart.shape[0])
            r2 = numpy.random.randint(restart.shape[1])
            result = result.at[-1, i].set(restart[r1, r2])

    np.save(args.o, result)
