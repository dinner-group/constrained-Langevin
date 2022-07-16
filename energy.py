from sampler import *

traj = np.load("sna_run2.npy")
E = jax.vmap(potential, 0, 0)(traj)
np.save("sna_run2_energy.npy", E)
