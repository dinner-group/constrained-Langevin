from sna import *

run_id = 11
traj = np.load("sna_run%d.npy"%(run_id))
E = jax.vmap(potential, 0, 0)(traj)
np.save("sna_run%d_energy.npy"%(run_id), E)
