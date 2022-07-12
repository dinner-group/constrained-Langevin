from sampler import *

nsteps = 1000000
out = sample(q0, nsteps, key)
_, accept, h_arr, _, _ = out[0]
traj = out[1]
print("acceptance ratio: %s"%(accept.sum() / nsteps))
np.save("run1.npy", traj)
