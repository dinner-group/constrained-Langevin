import numpy
import jax
import jax.numpy as np
import warnings
from functools import partial
import util
import os
import gurobipy as gurobi
import tpt_excitability as tpt
import model
import scipy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-process", type=int, required=True)
argp = parser.parse_args()

path = os.path.dirname(__file__)
params = np.load(path + "/cs2d_par_1.npy")
fps_all = np.load(path + "/cs2d_fp_1.npy")

def compute_max_deviation_all_fps(par, fps, mesh_shape=(150, 150)):

    cs2d = model.Cubic_System_2d(par)
    stable = np.all(jax.vmap(np.linalg.eigvals)(jax.vmap(cs2d.jac, (None, 0, None))(0., fps, par)) < 0, axis=1)
    unstable = np.any(jax.vmap(np.linalg.eigvals)(jax.vmap(cs2d.jac, (None, 0, None))(0., fps, par)) > 0, axis=1)
    ind_stable = np.arange(stable.size)[stable]
    fps_stable = fps[stable]
    width = 2 / (np.mean(np.abs(np.pad(par, (1, 0), constant_values=1).reshape((2, 9))[:, np.array([-4, -1])])) / np.max(np.abs(np.pad(par, (1, 0), constant_values=1).reshape((2, 9))[:, 2:-4])))
    mesh_coords = (np.nanmin(fps[:, 0], axis=0) - width, np.nanmax(fps[:, 0], axis=0) + width, np.nanmin(fps[:, 1], axis=0) - width, np.nanmax(fps[:, 1], axis=0) + width)
    diffusion_constant = 1.
    G, xy, mesh_spacing = tpt.discretize_generator(par, diffusion_constant, mesh_coords, mesh_shape)
    G = scipy.sparse.coo_matrix((G.data, tuple(G.indices.T))).tocsc()
    dist = np.linalg.norm(np.array(xy).reshape((2, xy[0].shape[0] * xy[0].shape[0])).T - np.expand_dims(fps_stable, 1), axis=2)
    mask = dist < np.minimum(*mesh_spacing)

    out = np.full((stable.size, 2), np.nan)

    for i in range(fps_stable.shape[0]):
        if np.any(mask[i]):
            try:
                mask1 = np.ones(mask.shape[0], dtype=np.bool_).at[i].set(False)@mask
                committor = tpt.compute_committor(G, mask1, mask[i])
                maxdist = tpt.compute_mean_excursion_size_q_noabsorb(G, dist[i], mask1, mask[i], np.minimum(*mesh_shape), committor)
                out = out.at[ind_stable[i], 0].set(np.maximum(0, np.max((maxdist - committor * dist[i]) / (10 * np.minimum(*mesh_spacing) + dist[i]))))
                if stable.sum() > 0 and unstable.sum() > 0:
                    out = out.at[ind_stable[i], 1].set(np.min(np.linalg.norm(fps_stable - np.expand_dims(fps[unstable], 1), axis=2)) / np.max(maxdist - committor * dist[i]))
            except(RuntimeError):
                continue

    return out

out = []
#job_size = params.shape[0] // 50
job_size = 40
start = argp.process * job_size

for i in range(start, start + job_size):
    out.append(compute_max_deviation_all_fps(params[i].ravel()[1:], fps_all[i]))

np.save("cs2d_excursion_size_%d.npy"%(argp.process), np.array(out))
