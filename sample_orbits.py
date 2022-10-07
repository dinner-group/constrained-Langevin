from continuation import *

n_mesh_point = 50
model_index = 29
#orbits = np.array([np.ravel(np.load("candidate_orbit_%d.npy"%(i))) for i in range(41)])
orbits = np.load("continue_1_%d_y.npy"%(model_index))
reaction_consts = np.load("continue_1_%d_reaction_consts.npy"%(model_index))
period = np.load("continue_1_%d_period.npy"%(model_index))
model = KaiODE(reaction_consts[-1])
#orbit0 = scipy.integrate.solve_ivp(model.f_red, jac=model.jac_red, t_span=(0, 1), t_eval=np.linspace(0, 1, n_mesh_point * colloc.n_colloc_point + 1), y0=orbits[model_index, 51:-1], method="BDF", atol=1e-9)

bounds = np.array([[k / 100, k * 100] for k in reaction_consts])

print("sampling orbit %d with %d mesh points"%(model_index, n_mesh_point))
i = 0
result = sample(orbits[i, 0, :, :], period[i, 0], reaction_consts[i], bounds, maxiter=10000, step_size=1e-2)

prefix="continue_2"

np.save("%s_%d_y.npy"%(prefix, model_index), result[0])
np.save("%s_%d_period.npy"%(prefix, model_index), result[1])
np.save("%s_%d_period_grad.npy"%(prefix, model_index), result[2])
np.save("%s_%d_reaction_consts.npy"%(prefix, model_index), result[3])
