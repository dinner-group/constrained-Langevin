import numpy
import jax
import jax.numpy as np
import warnings
from functools import partial
import util
import os
import gurobipy as gurobi
import model
import scipy
# import sparseqr
jax.config.update("jax_enable_x64", True)
env = gurobi.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

def compute_mean_excursion_size(generator, d, mask):
    
    ind = np.arange(generator.shape[0])[mask]
    generator = generator.tocoo().tocsr()
    for i in ind:
        generator.data[generator.indptr[i]:generator.indptr[i + 1]] *= mask[generator.indices[generator.indptr[i]:generator.indptr[i + 1]]]
    opt = gurobi.Model("mes", env=env)
    x = opt.addMVar(shape=d.shape)
    opt.addConstr(x >= numpy.array(d), "lower_bound")
    opt.addConstr(generator@x <= numpy.zeros(generator.shape[0]), "generator_equation")
    opt.setObjective(numpy.ones(generator.shape[1])@x, gurobi.GRB.MINIMIZE)
    opt.optimize()
    
    if opt.status == 2:
        return np.array(opt.x)
    else:
        return np.full(d.shape, np.nan)
    
def compute_mean_excursion_size_q(generator, d, mask, n_points):
    
    qs = []
    for c in np.linspace(0, d.max(), n_points):
        mask_r = d > c
        qs.append(compute_committor(generator, mask, mask_r))
    qs = np.array(qs)
    maxdist = jax.scipy.integrate.trapezoid(qs, axis=0, x=np.linspace(0, d.max(), n_points))
    return maxdist

def compute_mean_excursion_size_q_noabsorb(generator, d, mask_A, mask_B, n_points, q0):
    
    qs = []
    if q0 is None:
        q0 = compute_committor(generator, mask_A, mask_B)
    for c in np.linspace(0, d.max(), n_points):
        mask_r = d > c
        qs.append(compute_committor(generator, mask_B + (mask_A * ~mask_r), mask_r, q0))
    qs = np.array(qs)
    maxdist = jax.scipy.integrate.trapezoid(qs, axis=0, x=np.linspace(0, d.max(), n_points))
    return maxdist
    
@jax.jit
def int_f_x(x0, x1, y, par, **kwargs):
    K = model.Cubic_System_2d.K.at[0].add(1)
    par = np.pad(par, (1, 0), constant_values=1)
    par = par.reshape((2, par.size // 2))
    par = par.at[1].multiply(1 / K[0])
    v0 = np.array([x0, y])
    a0 = par@np.prod(v0**K.T, axis=1)
    v1 = np.array([x1, y])
    a1 = par@np.prod(v1**K.T, axis=1)
    return a1[1] - a0[1]
    
@jax.jit
def int_f_y(y0, y1, x, par, **kwargs):
    K = model.Cubic_System_2d.K.at[1].add(1)
    par = np.pad(par, (1, 0), constant_values=(1))
    par = par.reshape((2, par.size // 2))
    par = par.at[0].multiply(1 / K[1])
    v0 = np.array([x, y0])
    a0 = par@np.prod(v0**K.T, axis=1)
    v1 = np.array([x, y1])
    a1 = par@np.prod(v1**K.T, axis=1)
    return a1[0] - a0[0]
    
@partial(jax.jit, static_argnames=("mesh_shape", "int_f_x", "int_f_y"))
def discretize_generator(drift_pars, diffusion_constant, mesh_coords, mesh_shape, int_f_x=int_f_x, int_f_y=int_f_y, **kwargs):
    
    mesh_x0, mesh_x1, mesh_y0, mesh_y1 = mesh_coords
    mesh_spacing = ((mesh_x1 - mesh_x0) / mesh_shape[0], (mesh_y1 - mesh_y0) / mesh_shape[1])
    xy = np.meshgrid(np.linspace(mesh_x0 + mesh_spacing[0] / 2, mesh_x1 - mesh_spacing[0] / 2, mesh_shape[0]), 
                     np.linspace(mesh_y1 + mesh_spacing[1] / 2, mesh_y0 - mesh_spacing[1] / 2, mesh_shape[1]))
    k = 0

    def loop_inner(carry, _):
        
        i, j, k = carry        
        fx = int_f_y(mesh_y1 - (j + 1) * mesh_spacing[1], mesh_y1 - j * mesh_spacing[1], mesh_x0 + (i + 1) * mesh_spacing[0], drift_pars, **kwargs)
        fy = int_f_x(mesh_x0 + i * mesh_spacing[0], mesh_x0 + (i + 1) * mesh_spacing[0], mesh_y1 - (j + 1) * mesh_spacing[1], drift_pars, **kwargs)
        
        ind = np.array([[k, k],
                        [k, k + 1],
                        [k, k + mesh_shape[0]],
                        [k + 1, k],
                        [k + 1, k + 1],
                        [k + mesh_shape[0], k],
                        [k + mesh_shape[0], k + mesh_shape[0]]])
        
        dat = np.zeros(ind.shape[0])
        
        #upwind drift
        dat = dat.at[0].add(np.where(fx > 0, -fx, 0))
        dat = dat.at[1].add(np.where(fx > 0, fx, 0))        
        dat = dat.at[0].add(np.where(fy < 0, fy, 0))
        dat = dat.at[2].add(np.where(fy < 0, -fy, 0))
        dat = dat.at[3].add(np.where(fx < 0, -fx, 0))
        dat = dat.at[4].add(np.where(fx < 0, fx, 0))
        dat = dat.at[5].add(np.where(fy > 0, fy, 0))
        dat = dat.at[6].add(np.where(fy > 0, -fy, 0))
        
        #diffusion
        dat = dat.at[0].add(-diffusion_constant * mesh_spacing[1] / mesh_spacing[0])
        dat = dat.at[1].add(diffusion_constant * mesh_spacing[1] / mesh_spacing[0])
        dat = dat.at[0].add(-diffusion_constant * mesh_spacing[0] / mesh_spacing[1])
        dat = dat.at[2].add(diffusion_constant * mesh_spacing[0] / mesh_spacing[1])
        dat = dat.at[3].add(diffusion_constant * mesh_spacing[1] / mesh_spacing[0])
        dat = dat.at[4].add(-diffusion_constant * mesh_spacing[1] / mesh_spacing[0])
        dat = dat.at[5].add(diffusion_constant * mesh_spacing[0] / mesh_spacing[1])
        dat = dat.at[6].add(-diffusion_constant * mesh_spacing[0] / mesh_spacing[1])
        
        return (i + 1, j, k + 1), (dat, ind)
    
    def loop_outer(carry, _):
        
        j, k = carry
        carry, out = jax.lax.scan(loop_inner, (0, j, k), None, length=mesh_shape[1] - 1)
        k = carry[2]    
        fy = int_f_x(mesh_x0 + mesh_shape[0] * mesh_spacing[0], mesh_x0 + (mesh_shape[0] + 1) * mesh_spacing[0], mesh_y1 - (j + 1) * mesh_spacing[1], drift_pars, **kwargs)
        dat, ind = out
        
        ind = np.vstack([np.vstack(ind), np.array([[k, k],
                                        [k, k + mesh_shape[0]],
                                        [k + mesh_shape[0], k],
                                        [k + mesh_shape[0], k + mesh_shape[0]]])])
        
        dat = np.pad(np.concatenate(dat), (0, 4))    
        dat = dat.at[-4].add(np.where(fy < 0, fy, 0))
        dat = dat.at[-3].add(np.where(fy < 0, -fy, 0))
        dat = dat.at[-2].add(np.where(fy > 0, fy, 0))
        dat = dat.at[-1].add(np.where(fy > 0, -fy, 0))
                
        return (j + 1, k + 1), (dat, ind)

    def loop2(carry, _):
        
        i, k = carry
        fx = int_f_y(mesh_y1 - mesh_shape[1] * mesh_spacing[1], mesh_y1 - (mesh_shape[1] - 1) * mesh_spacing[1], mesh_x0 + (i + 1) * mesh_spacing[0], drift_pars, **kwargs)
        
        ind = np.array([[k, k],
                        [k, k + 1],
                        [k + 1, k],
                        [k + 1, k + 1]])
        
        dat = np.zeros(ind.shape[0]) 
        dat = dat.at[0].add(np.where(fx > 0, -fx, 0))
        dat = dat.at[1].add(np.where(fx > 0, fx, 0))
        dat = dat.at[2].add(np.where(fx < 0, -fx, 0))
        dat = dat.at[3].add(np.where(fx < 0, fx, 0))
                
        dat = dat.at[0].add(-diffusion_constant * mesh_spacing[1] / mesh_spacing[0])
        dat = dat.at[1].add(diffusion_constant * mesh_spacing[1] / mesh_spacing[0])
        dat = dat.at[2].add(diffusion_constant * mesh_spacing[1] / mesh_spacing[0])
        dat = dat.at[3].add(-diffusion_constant * mesh_spacing[1] / mesh_spacing[0])
                
        return (i + 1, k + 1), (dat, ind)
    
    carry, out = jax.lax.scan(loop_outer, (0, 0), None, length=mesh_shape[0] - 1)
    dat, ind = out
    dat = np.concatenate(dat)
    ind = np.vstack(ind)
    carry, out = jax.lax.scan(loop2, (0, carry[1]), None, length=mesh_shape[1] - 1)
    dat = np.concatenate([dat, np.concatenate(out[0])])
    ind = np.vstack([ind, np.vstack(out[1])])
    
    J = jax.experimental.sparse.BCOO((dat, ind), shape=(mesh_shape[0] * mesh_shape[1], mesh_shape[0] * mesh_shape[1]))
    
    return J, xy, mesh_spacing

def compute_committor(generator, mask_A, mask_B, bc=None):
    if bc is None:
        bc = np.ones(generator.shape[0])
    q0 = scipy.sparse.linalg.spsolve(generator[~(mask_A + mask_B)][:, ~(mask_A + mask_B)], -generator[~(mask_A + mask_B)][:, mask_B]@bc[mask_B])
    q = bc
    q = q.at[~(mask_A + mask_B)].set(q0)
    q = q.at[mask_A].set(0)
    return q

def compute_committor_return(generator, mask_A, mask_B):
    generator_1 = generator.tocoo().tocsc()
    ind = np.arange(generator_1.shape[0])[mask_B]
    for i in ind:
        generator_1.data[generator_1.indptr[i]:generator_1.indptr[i + 1]] *= mask_B[generator_1.indices[generator_1.indptr[i]:generator_1.indptr[i + 1]]]
    q0 = scipy.sparse.linalg.spsolve(generator_1[~(mask_A)][:, ~(mask_A)], -(generator[~(mask_A)][:, mask_B].sum(axis=1) - generator_1[~(mask_A)][:, mask_B].sum(axis=1)))
    q = np.ones(generator.shape[0])
    q = q.at[~mask_A].set(q0)
    q = q.at[mask_A].set(0)
    return q

def compute_mfpt(generator, mask_B):
    mfpt0 = scipy.sparse.linalg.spsolve(generator[~mask_B][:, ~mask_B], -np.ones(generator.shape[1] - mask_B.sum()))
    mfpt = np.zeros(generator.shape[0])
    mfpt = mfpt.at[~mask_B].set(mfpt0)
    return mfpt

def compute_mfrt(generator, mask_B):
    generator_1 = generator.tocoo().tocsc()
    ind = np.arange(generator_1.shape[0])[mask_B]
    for i in ind:
        generator_1.data[generator_1.indptr[i]:generator_1.indptr[i + 1]] *= mask_B[generator_1.indices[generator_1.indptr[i]:generator_1.indptr[i + 1]]]
    lq = scipy.sparse.linalg.splu(generator_1)
    mfrt = lq.solve(-numpy.ones(generator_1.shape[0]))
    var_frt = 2 * lq.solve(-mfrt) - mfrt - mfrt**2
    return mfrt, np.sqrt(var_frt)
