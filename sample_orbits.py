from continuation_sampler import *

n_mesh_point = 10
orbits = np.array([np.ravel(np.load("candidate_orbit_%d.npy"%(i))) for i in range(41)])
model = KaiODE(orbits[29, :50])
orbit0 = scipy.integrate.solve_ivp(model.f_red, jac=model.jac_red, t_span=(0, 1), t_eval=np.linspace(0, 1, n_mesh_point * colloc.n_colloc_point + 1), y0=orbits[29, 51:-1], method="BDF", atol=1e-9, rtol=1e-6)
result = sample(orbit0.y, 1.0, orbits[29, :50])

np.save("continue_29_y.npy", result[0])
np.save("continue_29_period.npy", result[1])
np.save("continue_29_period_grad.npy", result[2])
np.save("continue_29_reaction_consts.npy", result[3])
