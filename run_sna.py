from sampler import *

q0 = np.array(numpy.loadtxt(path + "/q0.txt"))
#q0 = q0.at[np.arange(50)[np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=bool)]].add(-5)
nsteps = 1000000
if __name__=="__main__":
    out = sample(q0, nsteps, key)
    _, accept, h_arr, _, _ = out[0]
    traj = out[1]
    print("acceptance ratio: %s"%(accept.sum() / nsteps))
    np.save("sna_run2.npy", traj)
