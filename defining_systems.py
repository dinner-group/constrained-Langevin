import jax
import jax.numpy as np
import numpy
import model
import util
from functools import partial
jax.config.update("jax_enable_x64", True)

@jax.jit
def fixed_point(q, ode_model, *args):

    k = q[:ode_model.n_par]
    y = q[ode_model.n_par:ode_model.n_par + ode_model.n_dim]
    return ode_model.f(0., y, k)

@jax.jit
def fully_extended_hopf(q, ode_model, *args):
    
    k = q[:ode_model.n_par]
    y = q[ode_model.n_par:ode_model.n_par + ode_model.n_dim]
    evec_real = q[ode_model.n_par + ode_model.n_dim:ode_model.n_par + 2 * ode_model.n_dim]
    evec_imag = q[ode_model.n_par + 2 * ode_model.n_dim:ode_model.n_par + 3 * ode_model.n_dim]
    eval_imag = q[ode_model.n_par + 3 * ode_model.n_dim]
    
    ode_rhs, jvp_real = jax.jvp(ode_model.f, (0., y, k), (0., evec_real, np.zeros_like(k)))
    ode_rhs, jvp_imag = jax.jvp(ode_model.f, (0., y, k), (0., evec_imag, np.zeros_like(k)))
    evec_abs = evec_real**2 + evec_imag**2
    
    return np.concatenate([ode_rhs,
                           jvp_real + eval_imag * evec_imag,
                           jvp_imag - eval_imag * evec_real,
                           np.array([evec_abs.sum() - 1]),
                           np.array([evec_imag[0]])])

@jax.jit
def fully_extended_hopf_log(q, ode_model, *args):
    
    eval_imag = np.exp(q[ode_model.n_par + 3 * ode_model.n_dim])
    q = q.at[ode_model.n_par + 3 * ode_model.n_dim].set(eval_imag)
    return fully_extended_hopf(q, ode_model, *args)

@jax.jit
def fully_extended_hopf_2n(q, ode_model, *args):
    
    k = q[:ode_model.n_par]
    y = q[ode_model.n_par:ode_model.n_par + ode_model.n_dim]
    eigvec = q[ode_model.n_par + ode_model.n_dim:ode_model.n_par + 2 * ode_model.n_dim] 
    eigval = q[ode_model.n_par + 2 * ode_model.n_dim] 
    ode_rhs, r = jax.jvp(ode_model.f, (0., y, k), (0., eigvec, np.zeros_like(k)))
    _, r = jax.jvp(ode_model.f, (0., y, k), (0., r, np.zeros_like(k)))
    
    return np.concatenate([ode_rhs,
                           r - eigval * eigvec,
                           np.ravel(eigvec@eigvec) - 1
                          ])

@jax.jit
def fully_extended_hopf_2n_log(q, ode_model, *args):
    
    eigval = -np.exp(q[ode_model.n_par + 2 * ode_model.n_dim])
    q = q.at[ode_model.n_par + 2 * ode_model.n_dim].set(eigval)
    return fully_extended_hopf_2n(q, ode_model, *args)

@jax.jit
def bvp_colloc_resid_interval(y, k, integration_time, interval_endpoints, ode_model, colloc_points_unshifted=util.gauss_points_4, quadrature_weights=util.gauss_weights_4, dd=None):

    colloc_points = interval_endpoints[0] + (1 + colloc_points_unshifted) * (interval_endpoints[1] - interval_endpoints[0]) / 2
    node_points = np.linspace(*interval_endpoints, colloc_points.size + 1)
    if dd is None:
        dd = util.divided_difference(node_points, y)
    poly = jax.vmap(util.newton_polynomial, (0, None, None, None))(colloc_points, node_points, y, dd)
    poly_deriv = jax.vmap(jax.jacfwd(util.newton_polynomial), (0, None, None, None))(colloc_points, node_points, y, dd)
    f_interval = jax.vmap(ode_model.f, (None, 0, None))(0., poly, k)
    f_interval = np.where(ode_model.not_algebraic, integration_time * f_interval, f_interval)
    return np.ravel(ode_model.not_algebraic * poly_deriv - f_interval, order="C")

@jax.jit
def bvp_colloc_resid_interval_hermite3(y, k, period, interval_endpoints, ode_model):

    colloc_points_unshifted = util.midpoint
    interval_width = interval_endpoints[1] - interval_endpoints[0]
    colloc_points = interval_endpoints[0] + (1 + colloc_points_unshifted) * interval_width / 2
    node_points = np.linspace(*interval_endpoints, colloc_points_unshifted.size + 1)
    ydot = jax.vmap(lambda yy:ode_model.f(0., yy, k))(y.T).T
    
    def poly_interval(x):

        interval_width = interval_endpoints[1] - interval_endpoints[0]
        t = (x - interval_endpoints[0]) / interval_width
        return (2 * t**3 - 3 * t**2 + 1) * y[:, 0] \
                + (t**3 - 2 * t**2 + t) * interval_width * ydot[:, 0]\
                + (-2 * t**3 + 3 * t**2) * y[:, 1]\
                + (t**3 - t**2) * interval_width * ydot[:, 1]

    poly = jax.vmap(poly_interval)(colloc_points)
    poly_deriv = jax.vmap(jax.jacfwd(poly_interval))(colloc_points)
    f_interval = jax.vmap(lambda yy:ode_model.f(0., yy, k))(poly)
    f_interval = np.where(ode_model.not_algebraic, period * f_interval, f_interval)
    return np.ravel(ode_model.not_algebraic * poly_deriv - f_interval, order="C")

@partial(jax.jit, static_argnames=("n_smooth",))
def bvp_mm_mesh_resid(y, k, mesh_points, ode_model, colloc_points_unshifted=util.gauss_points_4, quadrature_weights=util.gauss_weights_4, n_smooth=4):

    n_mesh_intervals = mesh_points.size - 1
    n_points = n_mesh_intervals * colloc_points_unshifted.size + 1
    y = y.reshape((ode_model.n_dim, n_points), order="F")
    _, mesh_density = util.recompute_mesh(y, mesh_points, colloc_points_unshifted, n_smooth)
    mesh_mass_interval = (mesh_points[1:] - mesh_points[:-1]) * (mesh_density[1:] + mesh_density[:-1]) / (2 * mesh_points[-1]) 
    mesh_mass = mesh_mass_interval.sum()
    return mesh_mass_interval[1:] - mesh_mass_interval[:-1]

@jax.jit
def bvp_mm_mesh_resid_curvature_interval(y_i, k, interval_width, ode_model, colloc_points_unshifted=util.gauss_points_4, quadrature_weights=util.gauss_weights_4, dd=None, *args, **kwargs):

    node_points = np.linspace(0, 1, colloc_points_unshifted.size + 1)
    if dd is None:
        dd = util.divided_difference(node_points, y_i)
    poly = jax.vmap(util.newton_polynomial, (0, None, None, None))(colloc_points_unshifted, node_points, y_i, dd)
    d2y = jax.vmap(lambda yy:jax.jvp(ode_model.f, (0., yy, k), (0., yy, np.zeros(ode_model.n_par)))[1])(poly)
    return interval_width * quadrature_weights@(1 + np.sum(d2y**2, axis=1))**(1/4)

@jax.jit
def bvp_mm_mesh_resid_curvature(y, k, interval_widths, ode_model, colloc_points_unshifted=util.gauss_points_4, quadrature_weights=util.gauss_weights_4, dd=None, *args, **kwargs):

    n_mesh_intervals = interval_widths.size
    n_points = n_mesh_intervals * colloc_points_unshifted.size + 1
    y = y.reshape((ode_model.n_dim, n_points), order="F")

    def loop_body(i, _):
        y_i = jax.lax.dynamic_slice(y, (0, i * colloc_points_unshifted.size), (ode_model.n_dim, colloc_points_unshifted.size + 1))
        return i + 1, bvp_mm_mesh_resid_curvature_interval(y_i, k, interval_widths[i], ode_model, colloc_points_unshifted, quadrature_weights, dd)

    mesh_mass_interval = jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1]
    return mesh_mass_interval[1:] - mesh_mass_interval[:-1]

@partial(jax.jit, static_argnames=("n_mesh_intervals",))
def periodic_boundary_condition(q, ode_model, n_mesh_intervals, colloc_points_unshifted, *args, **kwargs):
    n_points = n_mesh_intervals * colloc_points_unshifted.size + 1
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape((ode_model.n_dim, n_points), order="F")
    return y[:, -1] - y[:, 0]

@partial(jax.jit, static_argnames=("n_mesh_intervals", "n_smooth", "boundary_condition", "mesh_resid"))
def bvp_mm_colloc_resid(q, ode_model, colloc_points_unshifted=util.gauss_points_4, quadrature_weights=util.gauss_weights_4, boundary_condition=periodic_boundary_condition, mesh_resid=None, *args, n_mesh_intervals=60, n_smooth=4, **kwargs):

    k = q[:ode_model.n_par]
    n_points = n_mesh_intervals * colloc_points_unshifted.size + 1
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape((ode_model.n_dim, n_points), order="F")
    interval_widths = q[ode_model.n_par + n_points * ode_model.n_dim:ode_model.n_par + n_points * ode_model.n_dim + n_mesh_intervals]
    mesh_points = np.cumsum(interval_widths)
    mesh_points = np.pad(mesh_points, (1, 0))

    def loop_body(i, _):
        y_i = jax.lax.dynamic_slice(y, (0, i * colloc_points_unshifted.size), (ode_model.n_dim, colloc_points_unshifted.size + 1))
        interval_endpoints = np.array([0., 1])
        node_points = np.linspace(*interval_endpoints, colloc_points_unshifted.size + 1)
        dd = util.divided_difference(node_points, y_i)
        r_i = bvp_colloc_resid_interval(y_i, k, interval_widths[i], interval_endpoints, ode_model, colloc_points_unshifted, quadrature_weights)
        if mesh_resid is None:
            mesh_mass_interval = bvp_mm_mesh_resid_curvature_interval(y_i, k, interval_widths[i], ode_model, colloc_points_unshifted, quadrature_weights, dd) 
            return i + 1, (r_i, mesh_mass_interval)
        else:
            return i + 1, r_i

    if mesh_resid is None:
        colloc_eqs, mesh_mass = jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1]
        mesh_eqs = mesh_mass[1:] - mesh_mass[:-1]
    else:
        colloc_eqs = jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1]
        mesh_eqs = mesh_resid(y, mesh_points, ode_model, colloc_points_unshifted, n_smooth)

    resid = np.concatenate([colloc_eqs.ravel(order="C"), boundary_condition(q, ode_model, n_mesh_intervals, colloc_points_unshifted, *args, **kwargs), mesh_eqs])

    return resid

@partial(jax.jit, static_argnames=("n_mesh_intervals", "n_smooth", "boundary_condition", "mesh_resid"))
def bvp_mm_colloc_jac(q, ode_model, colloc_points_unshifted=util.gauss_points_4, quadrature_weights=util.gauss_weights_4, boundary_condition=periodic_boundary_condition, mesh_resid=None, *args, n_mesh_intervals=60, n_smooth=4, **kwargs):

    k = q[:ode_model.n_par]
    n_points = n_mesh_intervals * colloc_points_unshifted.size + 1
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape((ode_model.n_dim, n_points), order="F")
    interval_widths = q[ode_model.n_par + n_points * ode_model.n_dim:ode_model.n_par + n_points * ode_model.n_dim + n_mesh_intervals]
    mesh_points = np.cumsum(interval_widths)
    mesh_points = np.pad(mesh_points, (1, 0))

    def loop_body(i, _):
        y_i = jax.lax.dynamic_slice(y, (0, i * colloc_points_unshifted.size), (ode_model.n_dim, colloc_points_unshifted.size + 1))
        Jy_i = jax.jacfwd(bvp_colloc_resid_interval, argnums=0)(y_i, k, interval_widths[i], np.array([0., 1]), ode_model, colloc_points_unshifted, quadrature_weights)
        Jk_i = jax.jacfwd(bvp_colloc_resid_interval, argnums=1)(y_i, k, interval_widths[i], np.array([0., 1]), ode_model, colloc_points_unshifted, quadrature_weights)
        Jm_i = jax.jacfwd(bvp_colloc_resid_interval, argnums=2)(y_i, k, interval_widths[i], np.array([0., 1]), ode_model, colloc_points_unshifted, quadrature_weights)
        Jy_i = Jy_i.reshape((colloc_points_unshifted.size * ode_model.n_dim, (colloc_points_unshifted.size + 1) * ode_model.n_dim), order="F")

        return i + 1, (np.hstack([Jy_i[:, :ode_model.n_dim], np.expand_dims(Jm_i, 1), Jy_i[:, ode_model.n_dim:]]), Jk_i)
    
    if mesh_resid is None:
        mesh_resid = bvp_mm_mesh_resid_curvature

    Jmesh_y = jax.jacrev(mesh_resid, argnums=0)(y.ravel(order="F"), k, interval_widths, ode_model, colloc_points_unshifted, quadrature_weights, n_smooth=n_smooth)
    Jmesh_k = jax.jacrev(mesh_resid, argnums=1)(y.ravel(order="F"), k, interval_widths, ode_model, colloc_points_unshifted, quadrature_weights, n_smooth=n_smooth)
    Jmesh_m = jax.jacrev(mesh_resid, argnums=2)(y.ravel(order="F"), k, interval_widths, ode_model, colloc_points_unshifted, quadrature_weights, n_smooth=n_smooth)
    Jmesh = np.hstack([Jmesh_y, Jmesh_m])
    Jmesh = util.BVPMMJac_1.permute_col(Jmesh, ode_model.n_dim, n_mesh_intervals, colloc_points_unshifted)
    Jbc = jax.jacrev(boundary_condition)(q, ode_model, n_mesh_intervals, colloc_points_unshifted, *args, **kwargs)
    Jy, Jk = jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1]
    Jk = np.vstack(Jk)
    Jk = np.vstack([Jk, Jbc[:, :ode_model.n_par]])
    Jk = np.vstack([Jk, Jmesh_k])
    if len(Jbc.shape) == 1:
        Jbc = np.expand_dims(Jbc, 0)
    Jbc_y = np.vstack([util.BVPMMJac_1.permute_col(Jbc[:, ode_model.n_par:], ode_model.n_dim, n_mesh_intervals, colloc_points_unshifted), Jmesh])
    J = util.BVPMMJac_1(Jy, Jk, Jbc_y, ode_model.n_dim, ode_model.n_par, colloc_points_unshifted=colloc_points_unshifted)

    return J

@partial(jax.jit, static_argnames=("n_mesh_intervals", "n_smooth"))
def bvp_mm_colloc_resid_multi_shared_k(q, ode_models, colloc_points_unshifted=None, *args, n_mesh_intervals=None, n_smooth=4, **kwargs):

    if colloc_points_unshifted is None:
        colloc_points_unshifted = tuple(util.gauss_points_4 for _ in ode_models)

    if n_mesh_intervals is None:
        n_mesh_intervals = tuple(60 for _ in ode_models)

    y_indices = [0 for _ in range(len(ode_models) + 1)]
    y_indices[0] = ode_models[0].n_par

    for i in range(len(ode_models)):
        y_indices[i + 1] = y_indices[i] + ode_models[i].n_dim * (n_mesh_intervals[i] * colloc_points_unshifted[i].size + 1) + n_mesh_intervals[i]

    return np.concatenate([bvp_mm_colloc_resid(np.concatenate([q[:ode_models[0].n_par], q[y_indices[i]:y_indices[i + 1]]]), ode_models[i], colloc_points_unshifted[i], *args, 
                           n_mesh_intervals=n_mesh_intervals[i], n_smooth=n_smooth, **kwargs) for i in range(len(ode_models))])

@partial(jax.jit, static_argnames=("n_mesh_intervals", "n_smooth"))
def bvp_mm_colloc_jac_multi_shared_k(q, ode_models, colloc_points_unshifted=None, *args, n_mesh_intervals=None, n_smooth=4, **kwargs):

    if colloc_points_unshifted is None:
        colloc_points_unshifted = tuple(util.gauss_points_4 for _ in ode_models)

    if n_mesh_intervals is None:
        n_mesh_intervals = tuple(60 for _ in ode_models)

    y_indices = [0 for _ in range(len(ode_models) + 1)]
    y_indices[0] = ode_models[0].n_par

    for i in range(len(ode_models)):
        y_indices[i + 1] = y_indices[i] + ode_models[i].n_dim * (n_mesh_intervals[i] * colloc_points_unshifted[i].size + 1) + n_mesh_intervals[i]

    return tuple(bvp_mm_colloc_jac(np.concatenate([q[:ode_models[0].n_par], q[y_indices[i]:y_indices[i + 1]]]), ode_models[i], colloc_points_unshifted[i], *args, n_mesh_intervals=n_mesh_intervals[i], n_smooth=n_smooth) 
            for i in range(len(ode_models)))

@jax.jit
def bvp_floquet_resid_interval(v, y, k, interval_width, ode_model, colloc_points_unshifted=util.gauss_points_4):    
    
    colloc_points = (1 + colloc_points_unshifted) / 2
    node_points = np.linspace(0, 1, colloc_points_unshifted.size + 1)
    dd_y = util.divided_difference(node_points, y)
    dd_v = util.divided_difference(node_points, v)
    poly_y = jax.vmap(util.newton_polynomial, (0, None, None, None))(colloc_points, node_points, y, dd_y)
    poly_v = jax.vmap(util.newton_polynomial, (0, None, None, None))(colloc_points, node_points, v, dd_v)
    poly_v_deriv = jax.vmap(jax.jacfwd(util.newton_polynomial), (0, None, None, None))(colloc_points, node_points, v, dd_v)
    f_interval = jax.vmap(lambda yy, vv:jax.jvp(ode_model.f, (0., yy, k), (0., vv, np.zeros_like(k)))[1])(poly_y, poly_v)
    return np.ravel(poly_v_deriv - interval_width * f_interval)

@jax.jit
def bvp_floquet_resid(v, y, k, interval_widths, ode_model, colloc_points_unshifted=util.gauss_points_4, *args, **kwargs):
    
    n_points = interval_widths.size * colloc_points_unshifted.size + 1
    y = y.reshape((ode_model.n_dim, n_points), order="F")
    v = v.reshape(y.shape, order="F")
         
    def loop_body(i, _):
        
        y_i = jax.lax.dynamic_slice(y, (0, i * colloc_points_unshifted.size), (ode_model.n_dim, colloc_points_unshifted.size + 1))
        v_i = jax.lax.dynamic_slice(v, (0, i * colloc_points_unshifted.size), (ode_model.n_dim, colloc_points_unshifted.size + 1))
        return i + 1, bvp_floquet_resid_interval(v_i, y_i, k, interval_widths[i], ode_model, colloc_points_unshifted)
        
    colloc_eqs = jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1].ravel()
    return colloc_eqs

@partial(jax.jit, static_argnames=("n_mesh_intervals",))
def bvp_floquet_jac(v, y, k, interval_widths, ode_model, colloc_points_unshifted=util.gauss_points_4, *args, **kwargs):

    n_points = interval_widths.size * colloc_points_unshifted.size + 1
    y = y.reshape((ode_model.n_dim, n_points), order="F")
    v = v.reshape(y.shape, order="F")
    
    def loop_body(i, _):
        
        y_i = jax.lax.dynamic_slice(y, (0, i * colloc_points_unshifted.size), (ode_model.n_dim, colloc_points_unshifted.size + 1))
        v_i = jax.lax.dynamic_slice(v, (0, i * colloc_points_unshifted.size), (ode_model.n_dim, colloc_points_unshifted.size + 1))
        return i + 1, jax.jacfwd(bvp_floquet_resid_interval)(v_i, y_i, k, interval_widths[i], ode_model, colloc_points_unshifted).reshape((colloc_points_unshifted.size * ode_model.n_dim, (colloc_points_unshifted.size + 1) * ode_model.n_dim), order="F")
    
    jac = jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1]
    return util.BVPJac(jac, np.zeros((jac.shape[0] * jac.shape[1], 0)), ode_model.n_dim, 0, np.zeros((0, ode_model.n_dim)), np.zeros((0, ode_model.n_dim)))
