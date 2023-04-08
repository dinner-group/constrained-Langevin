import jax
import jax.numpy as np
import numpy
import model
import util
from functools import partial
jax.config.update("jax_enable_x64", True)

@jax.jit
def quadratic_roots_potential(position):
    E = 0
    E += np.where(np.abs(position[0]) > 10, 100 * (np.abs(position[0]) - 10)**2, 0)
    E += np.where(np.abs(position[1]) > 10, 100 * (np.abs(position[1]) - 10)**2, 0)
    return E

@jax.jit
def quadratic_roots(position):
    return np.array([position[2]**2 + position[0] * position[2] + position[1]])

@jax.jit
def quadratic_double_root(position):
    return np.array([position[2]**2 + position[0] * position[2] + position[1],
                     jax.jacfwd(quadratic_roots)(position)[0, 2]])

@jax.jit
def brusselator_rhs(q):
    br = model.Brusselator(np.exp(q[:model.Brusselator.n_par]))
    return br.f(0., np.exp(q[model.Brusselator.n_par:model.Brusselator.n_par + model.Brusselator.n_dim]))

@jax.jit
def brusselator_hb_3n(q):
    
    br = model.Brusselator(np.exp(q[:model.Brusselator.n_par]))
    
    k = np.exp(q[:model.Brusselator.n_par])
    y = np.exp(q[model.Brusselator.n_par:model.Brusselator.n_par + model.Brusselator.n_dim])
    evec_real = q[model.Brusselator.n_par + model.Brusselator.n_dim:model.Brusselator.n_par + 2 * model.Brusselator.n_dim]
    evec_imag = q[model.Brusselator.n_par + 2 * model.Brusselator.n_dim:model.Brusselator.n_par + 3 * model.Brusselator.n_dim]
    eval_imag = q[model.Brusselator.n_par + 3 * model.Brusselator.n_dim]
    
    f = br.f(0., y , k)
    jac = br.jac(0., y, k)
    evec_abs = evec_real**2 + evec_imag**2
    index = evec_abs.argmax()
    
    return np.concatenate([f,
                           jac@evec_real + eval_imag * evec_imag,
                           jac@evec_imag - eval_imag * evec_real,
                           np.array([evec_abs.sum() - 1]),
                           np.array([evec_imag[index]])])

@jax.jit
def brusselator_hb_n(q):
    
    br = model.Brusselator(np.exp(q[:model.Brusselator.n_par]))
    
    k = np.exp(q[:model.Brusselator.n_par])
    y = np.exp(q[model.Brusselator.n_par:model.Brusselator.n_par + model.Brusselator.n_dim])
    eval_imag = q[model.Brusselator.n_par + model.Brusselator.n_dim]
    
    f = br.f(0., y , k)
    jac = br.jac(0., y, k)
    
    M = np.zeros((2 * jac.shape[0] + 2, 2 * jac.shape[1] + 2))
    M = M.at[:jac.shape[0], :jac.shape[1]].set(jac)
    M = M.at[:jac.shape[0], jac.shape[1]:2 * jac.shape[1]].set(np.identity(jac.shape[0]) * eval_imag)
    M = M.at[0, 2 * jac.shape[1]].set(1)
    M = M.at[jac.shape[0]:2 * jac.shape[0], :jac.shape[1]].set(-np.identity(jac.shape[0]) * eval_imag)
    M = M.at[jac.shape[0]:2 * jac.shape[0], jac.shape[1]:2 * jac.shape[1]].set(jac)
    M = M.at[jac.shape[0], 2 * jac.shape[1] + 1].set(1)
    M = M.at[2 * jac.shape[0], 0].set(1)
    M = M.at[2 * jac.shape[0] + 1, jac.shape[1]].set(1)
    
    x = np.linalg.solve(M, np.zeros(2 * jac.shape[1] + 2).at[2 * jac.shape[0]].set(1))
    h = x[-2:]
    
    return np.concatenate([f, h])

@jax.jit
def brusselator_potential(q):
    return 100 * np.where(np.abs(q[:model.Brusselator.n_par]) > np.log(100), (np.abs(q[:model.Brusselator.n_par]) - np.log(100))**2, 0).sum()

gauss_points = np.array([-np.sqrt(3 / 7 + (2 / 7) * np.sqrt(6 / 5)), -np.sqrt(3 / 7 - (2 / 7) * np.sqrt(6 / 5)), np.sqrt(3 / 7 - (2 / 7) * np.sqrt(6 / 5)), np.sqrt(3 / 7 + (2 / 7) * np.sqrt(6 / 5))])
gauss_weights = np.array([18 + np.sqrt(30), 18 - np.sqrt(30), 18 - np.sqrt(30), 18 + np.sqrt(30)]) / 36
n_mesh_intervals = 60
mesh_points = np.linspace(0, 1, n_mesh_intervals + 1)

@jax.jit
def brusselator_bvp_interval(y, k, period, colloc_points, node_points):
    
    br = model.Brusselator(k)
    dd = util.divided_difference(node_points, y)
    poly_interval = lambda t:util.newton_polynomial(t, node_points, y, dd)
    poly = jax.vmap(poly_interval)(colloc_points)
    poly_deriv = jax.vmap(jax.jacfwd(poly_interval))(colloc_points)
    return np.ravel(poly_deriv - jax.vmap(lambda yy:period * br.f(0., yy, k))(poly), order="C")

@jax.jit
def brusselator_bvp(q):
    
    k = np.exp(q[:model.Brusselator.n_par])
    n_points = (n_mesh_intervals * gauss_points.size + 1)
    y = q[model.Brusselator.n_par:model.Brusselator.n_par + n_points * model.Brusselator.n_dim].reshape(model.Brusselator.n_dim, n_points, order="F")
    period = q[model.Brusselator.n_par + n_points * model.Brusselator.n_dim + 1]
    
    def loop_body(i, _):
        node_points = np.linspace(mesh_points[i], mesh_points[i + 1], gauss_points.size + 1)
        colloc_points = mesh_points[i] + gauss_points * (mesh_points[i + 1] - mesh_points[i])
        y_i = jax.lax.dynamic_slice(y, (0, i * gauss_points.size), (model.Brusselator.n_dim, gauss_points.size + 1))
        r_i = brusselator_bvp_interval(y_i, k, period, colloc_points, node_points)
        return i + 1, r_i
    
    colloc_eqs = jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1].ravel(order="C")
    return np.concatenate([colloc_eqs, y[:, -1] - y[:, 0]])

@jax.jit
def brusselator_bvp_jac(q):
    
    k = np.exp(q[:model.Brusselator.n_par])
    n_points = (n_mesh_intervals * gauss_points.size + 1)
    y = q[model.Brusselator.n_par:model.Brusselator.n_par + n_points * model.Brusselator.n_dim].reshape(model.Brusselator.n_dim, n_points, order="F")
    period = q[model.Brusselator.n_par + n_points * model.Brusselator.n_dim + 1]
    
    def loop_body(i, _):
        node_points = np.linspace(mesh_points[i], mesh_points[i + 1], gauss_points.size + 1)
        colloc_points = mesh_points[i] + gauss_points * (mesh_points[i + 1] - mesh_points[i])
        y_i = jax.lax.dynamic_slice(y, (0, i * gauss_points.size), (model.Brusselator.n_dim, gauss_points.size + 1))
        Jy_i = jax.jacfwd(brusselator_bvp_interval, argnums=0)(y_i, k, period, colloc_points, node_points)\
                .reshape((gauss_points.size * model.Brusselator.n_dim, (gauss_points.size + 1) * model.Brusselator.n_dim), order="F")
        Jk_i = jax.jacfwd(lambda x:brusselator_bvp_interval(y_i, np.exp(x), period, colloc_points, node_points))(q[:model.Brusselator.n_par])
        Jw_i = jax.jacfwd(brusselator_bvp_interval, argnums=2)(y_i, k, period, colloc_points, node_points)
        return i + 1, (Jy_i, np.hstack([Jk_i, Jw_i.reshape([Jw_i.size, 1])]))
    
    J = util.BVPJac(*jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1], model.Brusselator.n_dim, model.Brusselator.n_par, n_mesh_intervals)
    return J

@jax.jit
def brusselator_bvp_potential(q):
    
    E = 0
    
    n_points = (n_mesh_intervals * gauss_points.size + 1)
    y = q[model.Brusselator.n_par:model.Brusselator.n_par + n_points * model.Brusselator.n_dim].reshape(model.Brusselator.n_dim, n_points, order="F")
    period = q[model.Brusselator.n_par + n_points * model.Brusselator.n_dim + 1]
    arclength = np.linalg.norm(y[:, 1:] - y[:, :-1], axis=0).sum()
    min_arclength = 0.3
    
    E += 100 * np.where(np.abs(q[:model.Brusselator.n_par]) > np.log(100), (np.abs(q[:model.Brusselator.n_par]) - np.log(100))**2, 0).sum()
    E += np.where(period > 12, 1 * (period - 12)**2, 0)
    E += np.where(arclength < min_arclength, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4, 0)
    
    return E
