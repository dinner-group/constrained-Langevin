from continuation_sampler import *

n_mesh_point = 50
model_index = 32
orbits = np.array([np.ravel(np.load("candidate_orbit_%d.npy"%(i))) for i in range(41)])
model = KaiODE(orbits[model_index, :50])
orbit0 = scipy.integrate.solve_ivp(model.f_red, jac=model.jac_red, t_span=(0, 1), t_eval=np.linspace(0, 1, n_mesh_point * colloc.n_colloc_point + 1), y0=orbits[model_index, 51:-1], method="BDF", atol=1e-9)
print("sampling orbit %d with %d mesh points"%(model_index, n_mesh_point))
result = sample(orbit0.y, 1.0, orbits[model_index, :50], maxiter=10000, step_size=1e-3)

np.save("continue_%d_y.npy"%(model_index), result[0])
np.save("continue_%d_period.npy"%(model_index), result[1])
np.save("continue_%d_period_grad.npy"%(model_index), result[2])
np.save("continue_%d_reaction_consts.npy"%(model_index), result[3])
