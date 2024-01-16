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
def bvp_colloc_resid_interval(y, k, integration_time, interval_endpoints, ode_model, colloc_points_unshifted=util.gauss_points):

    colloc_points = interval_endpoints[0] + (1 + colloc_points_unshifted) * (interval_endpoints[1] - interval_endpoints[0]) / 2
    node_points = np.linspace(*interval_endpoints, colloc_points.size + 1)
    dd = util.divided_difference(node_points, y)
    poly_interval = lambda t:util.newton_polynomial(t, node_points, y, dd)
    poly = jax.vmap(poly_interval)(colloc_points)
    poly_deriv = jax.vmap(jax.jacfwd(poly_interval))(colloc_points)
    f_interval = jax.vmap(lambda yy:ode_model.f(0., yy, k))(poly)
    f_interval = np.where(ode_model.not_algebraic, integration_time * f_interval, f_interval)
    return np.ravel(ode_model.not_algebraic * poly_deriv - f_interval, order="C")

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
def periodic_bvp_colloc_resid(q, ode_model, mesh_points=np.linspace(0, 1, 61), colloc_points_unshifted=util.gauss_points, *args):

    n_mesh_intervals = mesh_points.size - 1
    k = q[:ode_model.n_par]
    n_points = (n_mesh_intervals * colloc_points_unshifted.size + 1)
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape((ode_model.n_dim, n_points), order="F")
    period = q[ode_model.n_par + n_points * ode_model.n_dim]

    def loop_body(i, _):
        interval_endpoints = jax.lax.dynamic_slice(mesh_points, (i,), (2,))
        y_i = jax.lax.dynamic_slice(y, (0, i * colloc_points_unshifted.size), (ode_model.n_dim, colloc_points_unshifted.size + 1))
        r_i = bvp_colloc_resid_interval(y_i, k, period, interval_endpoints, ode_model, colloc_points_unshifted)
        return i + 1, r_i

    colloc_eqs = jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1].ravel(order="C")
    return np.concatenate([colloc_eqs, y[:, -1] - y[:, 0]])

@jax.jit
def periodic_bvp_colloc_jac(q, ode_model, mesh_points=np.linspace(0, 1, 61), colloc_points_unshifted=util.gauss_points, *args):

    n_mesh_intervals = mesh_points.size - 1
    k = q[:ode_model.n_par]
    n_points = (n_mesh_intervals * colloc_points_unshifted.size + 1)
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape((ode_model.n_dim, n_points), order="F")
    period = q[ode_model.n_par + n_points * ode_model.n_dim]

    def loop_body(i, _):
        interval_endpoints = jax.lax.dynamic_slice(mesh_points, (i,), (2,))
        y_i = jax.lax.dynamic_slice(y, (0, i * colloc_points_unshifted.size), (ode_model.n_dim, colloc_points_unshifted.size + 1))
        Jy_i = jax.jacfwd(bvp_colloc_resid_interval, argnums=0)(y_i, k, period, interval_endpoints, ode_model, colloc_points_unshifted)
        Jk_i = jax.jacfwd(bvp_colloc_resid_interval, argnums=1)(y_i, k, period, interval_endpoints, ode_model, colloc_points_unshifted)
        Jw_i = jax.jacfwd(bvp_colloc_resid_interval, argnums=2)(y_i, k, period, interval_endpoints, ode_model, colloc_points_unshifted)
        Jy_i = Jy_i.reshape((colloc_points_unshifted.size * ode_model.n_dim, (colloc_points_unshifted.size + 1) * ode_model.n_dim), order="F")
        return i + 1, (Jy_i, np.hstack([Jk_i, Jw_i.reshape([Jw_i.size, 1])]))

    J = util.BVPJac(*jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1], ode_model.n_dim, ode_model.n_par)
    return J

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
def periodic_bvp_mm_mesh_resid_hermite3(y, k, mesh_points, ode_model, n_smooth=4):

    colloc_points_unshifted = util.midpoint
    n_mesh_intervals = mesh_points.size + 1
    n_points = n_mesh_intervals * colloc_points_unshifted.size + 1
    y = y.reshape((ode_model.n_dim, n_points), order="F")
    mesh_points = np.pad(mesh_points, (1, 1), constant_values=(0, 1))
    midpoints = (mesh_points[1:] + mesh_points[:-1]) / 2
    interval_width = mesh_points[1:] - mesh_points[:-1]
    ydot = jax.vmap(lambda yy:ode_model.f(0., yy, k))(y.T).T
    deriv = 2 * y[:, :-1] + interval_width * ydot[:, :-1] - 2 * y[:, 1:] + interval_width * ydot[:, 1:]
    deriv = np.pad((deriv[:, 1:] - deriv[:, :-1]) / (midpoints[1:] - midpoints[:-1]), ((0, 0), (1, 1)), mode="edge")
    a = np.maximum(1, jax.scipy.integrate.trapezoid(np.sum(deriv**2, axis=0)**(1 / (1 + 2 * (colloc_points_unshifted.size + 3))), x=mesh_points)**(1 + 2 * (colloc_points_unshifted.size + 3)))
    mesh_density = (1 + np.sum(deriv**2, axis=0) / a)**(1 / (1 + 2 * (colloc_points_unshifted.size + 3)))
    mesh_density = util.weighted_average_smoothing(mesh_density, n_smooth)
    mesh_mass_interval = interval_width * (mesh_density[1:] + mesh_density[:-1]) / 2
    mesh_mass = mesh_mass_interval.sum()
    return n_mesh_intervals * mesh_mass_interval[:-1] - mesh_mass

@partial(jax.jit, static_argnames=("n_mesh_intervals", "n_smooth"))
def periodic_bvp_mm_colloc_resid_hermite3(q, ode_model, *args, n_mesh_intervals=60, n_smooth=4):
   
    colloc_points_unshifted = util.midpoint
    k = q[:ode_model.n_par]
    n_points = n_mesh_intervals * colloc_points_unshifted.size + 1
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape((ode_model.n_dim, n_points), order="F")
    mesh_points = q[ode_model.n_par + n_points * ode_model.n_dim:ode_model.n_par + n_points * ode_model.n_dim + n_mesh_intervals - 1]
    period = q[ode_model.n_par + n_points * ode_model.n_dim + n_mesh_intervals - 1]
    mesh_eqs = periodic_bvp_mm_mesh_resid_hermite3(y, k, mesh_points, ode_model, n_smooth=4)
    mesh_points = np.pad(mesh_points, (1, 1), constant_values=(0, 1))

    def loop_body(i, _):
        interval_endpoints = jax.lax.dynamic_slice(mesh_points, (i,), (2,))
        y_i = jax.lax.dynamic_slice(y, (0, i * colloc_points_unshifted.size), (ode_model.n_dim, colloc_points_unshifted.size + 1))
        r_i = bvp_colloc_resid_interval_hermite3(y_i, k, period, interval_endpoints, ode_model)
        return i + 1, r_i

    colloc_eqs = jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1].ravel(order="C")
    return np.concatenate([colloc_eqs, y[:, -1] - y[:, 0], mesh_eqs])

@partial(jax.jit, static_argnames=("n_smooth",))
def bvp_mm_mesh_resid(y, mesh_points, ode_model, colloc_points_unshifted=util.gauss_points, n_smooth=4):

    n_mesh_intervals = mesh_points.size - 1
    n_points = n_mesh_intervals * colloc_points_unshifted.size + 1
    y = y.reshape((ode_model.n_dim, n_points), order="F")
    _, mesh_density = util.recompute_mesh(y, mesh_points, colloc_points_unshifted, n_smooth)
    mesh_mass_interval = (mesh_points[1:] - mesh_points[:-1]) * (mesh_density[1:] + mesh_density[:-1]) / 2
    mesh_mass = mesh_mass_interval.sum()
    return n_mesh_intervals * mesh_mass_interval[:-1] - mesh_mass

@partial(jax.jit, static_argnames=("n_mesh_intervals", "n_smooth"))
def periodic_bvp_mm_colloc_resid(q, ode_model, colloc_points_unshifted=util.gauss_points, *args, n_mesh_intervals=60, n_smooth=4):

    k = q[:ode_model.n_par]
    n_points = n_mesh_intervals * colloc_points_unshifted.size + 1
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape((ode_model.n_dim, n_points), order="F")
    mesh_points = q[ode_model.n_par + n_points * ode_model.n_dim:ode_model.n_par + n_points * ode_model.n_dim + n_mesh_intervals - 1]
    mesh_points = np.pad(mesh_points, (1, 1), constant_values=(0, 1))
    period = q[ode_model.n_par + n_points * ode_model.n_dim + n_mesh_intervals - 1]
    mesh_eqs = bvp_mm_mesh_resid(y, mesh_points, ode_model, colloc_points_unshifted, n_smooth)

    def loop_body(i, _):
        interval_endpoints = jax.lax.dynamic_slice(mesh_points, (i,), (2,))
        y_i = jax.lax.dynamic_slice(y, (0, i * colloc_points_unshifted.size), (ode_model.n_dim, colloc_points_unshifted.size + 1))
        r_i = bvp_colloc_resid_interval(y_i, k, period, interval_endpoints, ode_model, colloc_points_unshifted)
        return i + 1, r_i

    colloc_eqs = jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1].ravel(order="C")
    return np.concatenate([colloc_eqs, y[:, -1] - y[:, 0], mesh_eqs])

@partial(jax.jit, static_argnames=("n_mesh_intervals", "n_smooth"))
def periodic_bvp_mm_colloc_jac(q, ode_model, colloc_points_unshifted=util.gauss_points, *args, n_mesh_intervals=60, n_smooth=4):

    k = q[:ode_model.n_par]
    n_points = n_mesh_intervals * colloc_points_unshifted.size + 1
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape((ode_model.n_dim, n_points), order="F")
    mesh_points = q[ode_model.n_par + n_points * ode_model.n_dim:ode_model.n_par + n_points * ode_model.n_dim + n_mesh_intervals - 1]
    mesh_points = np.pad(mesh_points, (1, 1), constant_values=(0, 1))
    period = q[ode_model.n_par + n_points * ode_model.n_dim + n_mesh_intervals - 1]

    def loop_body(i, _):
        interval_endpoints = jax.lax.dynamic_slice(mesh_points, (i,), (2,))
        y_i = jax.lax.dynamic_slice(y, (0, i * colloc_points_unshifted.size), (ode_model.n_dim, colloc_points_unshifted.size + 1))
        Jy_i = jax.jacfwd(bvp_colloc_resid_interval, argnums=0)(y_i, k, period, interval_endpoints, ode_model, colloc_points_unshifted)
        Jk_i = jax.jacfwd(bvp_colloc_resid_interval, argnums=1)(y_i, k, period, interval_endpoints, ode_model, colloc_points_unshifted)
        Jw_i = jax.jacfwd(bvp_colloc_resid_interval, argnums=2)(y_i, k, period, interval_endpoints, ode_model, colloc_points_unshifted)
        Jm_i = jax.jacfwd(bvp_colloc_resid_interval, argnums=3)(y_i, k, period, interval_endpoints, ode_model, colloc_points_unshifted)
        Jy_i = Jy_i.reshape((colloc_points_unshifted.size * ode_model.n_dim, (colloc_points_unshifted.size + 1) * ode_model.n_dim), order="F")

        return i + 1, (np.hstack([Jy_i[:, :ode_model.n_dim], Jm_i[:, :1], Jy_i[:, ode_model.n_dim:], Jm_i[:, 1:2]]), np.hstack([Jk_i, Jw_i.reshape([Jw_i.size, 1])]))

    Jmesh_y = jax.jacrev(bvp_mm_mesh_resid, argnums=0)(y.ravel(order="F"), mesh_points, ode_model, colloc_points_unshifted, n_smooth)
    Jmesh_m = jax.jacrev(bvp_mm_mesh_resid, argnums=1)(y.ravel(order="F"), mesh_points, ode_model, colloc_points_unshifted, n_smooth)[:, 1:-1]
    Jmesh = np.hstack([Jmesh_y, Jmesh_m])
    Jmesh = util.permute_q_mesh(Jmesh, ode_model.n_dim, n_mesh_intervals, colloc_points_unshifted)
    Jy, Jk = jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1]
    Jk = np.vstack(Jk)

    J = util.BVPMMJac(Jy, Jk, Jmesh, ode_model.n_dim, ode_model.n_par, colloc_points_unshifted=colloc_points_unshifted)

    return J

@partial(jax.jit, static_argnames=("n_mesh_intervals", "n_smooth", "phase_condition"))
def periodic_bvp_mm_colloc_resid_1(q, ode_model, colloc_points_unshifted=util.gauss_points, phase_condition=None, *args, n_mesh_intervals=60, n_smooth=4, **kwargs):

    k = q[:ode_model.n_par]
    n_points = n_mesh_intervals * colloc_points_unshifted.size + 1
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape((ode_model.n_dim, n_points), order="F")
    interval_widths = q[ode_model.n_par + n_points * ode_model.n_dim:ode_model.n_par + n_points * ode_model.n_dim + n_mesh_intervals]
    mesh_points = np.cumsum(interval_widths)
    mesh_points = np.pad(mesh_points, (1, 0))
    mesh_eqs = bvp_mm_mesh_resid(y, mesh_points, ode_model, colloc_points_unshifted, n_smooth)

    def loop_body(i, _):
        y_i = jax.lax.dynamic_slice(y, (0, i * colloc_points_unshifted.size), (ode_model.n_dim, colloc_points_unshifted.size + 1))
        r_i = bvp_colloc_resid_interval(y_i, k, interval_widths[i], np.array([0., 1]), ode_model, colloc_points_unshifted)
        return i + 1, r_i

    colloc_eqs = jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1].ravel(order="C")
    resid = np.concatenate([colloc_eqs, y[:, -1] - y[:, 0], mesh_eqs])

    if phase_condition is not None:
        resid = np.concatenate([resid, np.ravel(phase_condition(q, ode_model=ode_model, *args, **kwargs))])

    return resid

@partial(jax.jit, static_argnames=("n_mesh_intervals", "n_smooth", "phase_condition"))
def periodic_bvp_mm_colloc_jac_1(q, ode_model, colloc_points_unshifted=util.gauss_points, phase_condition=None, *args, n_mesh_intervals=60, n_smooth=4, **kwargs):

    k = q[:ode_model.n_par]
    n_points = n_mesh_intervals * colloc_points_unshifted.size + 1
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape((ode_model.n_dim, n_points), order="F")
    interval_widths = q[ode_model.n_par + n_points * ode_model.n_dim:ode_model.n_par + n_points * ode_model.n_dim + n_mesh_intervals]
    mesh_points = np.cumsum(interval_widths)
    mesh_points = np.pad(mesh_points, (1, 0))

    def loop_body(i, _):
        y_i = jax.lax.dynamic_slice(y, (0, i * colloc_points_unshifted.size), (ode_model.n_dim, colloc_points_unshifted.size + 1))
        Jy_i = jax.jacfwd(bvp_colloc_resid_interval, argnums=0)(y_i, k, interval_widths[i], np.array([0., 1]), ode_model, colloc_points_unshifted)
        Jk_i = jax.jacfwd(bvp_colloc_resid_interval, argnums=1)(y_i, k, interval_widths[i], np.array([0., 1]), ode_model, colloc_points_unshifted)
        Jm_i = jax.jacfwd(bvp_colloc_resid_interval, argnums=2)(y_i, k, interval_widths[i], np.array([0., 1]), ode_model, colloc_points_unshifted)
        Jy_i = Jy_i.reshape((colloc_points_unshifted.size * ode_model.n_dim, (colloc_points_unshifted.size + 1) * ode_model.n_dim), order="F")

        return i + 1, (np.hstack([Jy_i[:, :ode_model.n_dim], np.expand_dims(Jm_i, 1), Jy_i[:, ode_model.n_dim:]]), Jk_i)

    def mm_mesh_resid_widths(y, interval_widths, ode_model, colloc_points_unshifted):
        mesh_points = np.cumsum(interval_widths)
        mesh_points = np.pad(mesh_points, (1, 0))
        return bvp_mm_mesh_resid(y, mesh_points, ode_model, colloc_points_unshifted, n_smooth)

    Jmesh_y = jax.jacrev(bvp_mm_mesh_resid, argnums=0)(y.ravel(order="F"), mesh_points, ode_model, colloc_points_unshifted, n_smooth)
    Jmesh_m = jax.jacrev(mm_mesh_resid_widths, argnums=1)(y.ravel(order="F"), interval_widths, ode_model, colloc_points_unshifted)
    Jmesh = np.hstack([Jmesh_y, Jmesh_m])
    Jmesh = util.permute_q_mesh_1(Jmesh, ode_model.n_dim, n_mesh_intervals, colloc_points_unshifted)
    Jbc = np.pad(Jmesh, ((ode_model.n_dim, 0), (0, 0)))
    Jbc = Jbc.at[:ode_model.n_dim, :ode_model.n_dim].set(-np.identity(ode_model.n_dim))
    Jbc = Jbc.at[:ode_model.n_dim, -ode_model.n_dim:].set(np.identity(ode_model.n_dim))
    Jy, Jk = jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1]
    Jk = np.vstack(Jk)
    Jk = np.pad(Jk, ((0, Jy.shape[0] * (Jy.shape[1] + 1) - 1 + ode_model.n_dim - Jk.shape[0]), (0, 0)))

    if phase_condition is not None:
        Jpc = jax.jacrev(phase_condition)(q, ode_model=ode_model, *args, **kwargs)
        if len(Jpc.shape) == 1:
            Jpc = np.expand_dims(Jpc, 0)
        Jk = np.vstack([Jk, Jpc[:, :ode_model.n_par]])
        Jbc = np.vstack([Jbc, util.permute_q_mesh_1(Jpc[:, ode_model.n_par:], ode_model.n_dim, n_mesh_intervals, colloc_points_unshifted)])

    J = util.BVPMMJac_1(Jy, Jk, Jbc, ode_model.n_dim, ode_model.n_par, colloc_points_unshifted=colloc_points_unshifted)

    return J

@partial(jax.jit, static_argnames=("n_mesh_intervals", "n_smooth"))
def periodic_bvp_mm_colloc_resid_multi_eqn_shared_k(q, ode_models, colloc_points_unshifted=None, *args, n_mesh_intervals=None, n_smooth=4):

    if colloc_points_unshifted is None:
        colloc_points_unshifted = (util.gauss_points for _ in ode_models)

    if n_mesh_intervals is None:
        n_mesh_intervals = (60 for _ in ode_models)

    y_indices = [0 for _ in range(len(ode_models) + 1)]
    y_indices[0] = ode_models[0].n_par

    for i in range(len(ode_models)):
        y_indices[i + 1] = y_indices[i] + ode_models[i].n_dim * (n_mesh_intervals[i] * colloc_points_unshifted[i].size + 1) + n_mesh_intervals[i]

    return np.concatenate([periodic_bvp_mm_colloc_resid(np.concatenate([q[:ode_models[0].n_par], q[y_indices[i]:y_indices[i + 1]]]), ode_models[i], colloc_points_unshifted[i], *args, 
                           n_mesh_intervals=n_mesh_intervals[i], n_smooth=n_smooth) for i in range(len(ode_models))])

@partial(jax.jit, static_argnames=("n_mesh_intervals", "n_smooth"))
def periodic_bvp_mm_colloc_jac_multi_eqn_shared_k(q, ode_models, colloc_points_unshifted=None, *args, n_mesh_intervals=None, n_smooth=4):

    if colloc_points_unshifted is None:
        colloc_points_unshifted = (util.gauss_points for _ in ode_models)

    if n_mesh_intervals is None:
        n_mesh_intervals = (60 for _ in ode_models)

    y_indices = [0 for _ in range(len(ode_models) + 1)]
    y_indices[0] = ode_models[0].n_par

    for i in range(len(ode_models)):
        y_indices[i + 1] = y_indices[i] + ode_models[i].n_dim * (n_mesh_intervals[i] * colloc_points_unshifted[i].size + 1) + n_mesh_intervals[i]

    return tuple(periodic_bvp_mm_colloc_jac(np.concatenate([q[:ode_models[0].n_par], q[y_indices[i]:y_indices[i + 1]]]), ode_models[i], colloc_points_unshifted[i], *args, n_mesh_intervals=n_mesh_intervals[i], n_smooth=n_smooth) 
            for i in range(len(ode_models)))

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
def brusselator_bvp(q, mesh_points=np.linspace(0, 1, 61)):
   
    n_mesh_intervals = mesh_points.size - 1
    k = np.exp(q[:model.Brusselator.n_par])
    n_points = (n_mesh_intervals * util.gauss_points.size + 1)
    y = q[model.Brusselator.n_par:model.Brusselator.n_par + n_points * model.Brusselator.n_dim].reshape(model.Brusselator.n_dim, n_points, order="F")
    period = q[model.Brusselator.n_par + n_points * model.Brusselator.n_dim]
    
    def loop_body(i, _):
        node_points = np.linspace(mesh_points[i], mesh_points[i + 1], util.gauss_points.size + 1)
        colloc_points = mesh_points[i] + util.gauss_points * (mesh_points[i + 1] - mesh_points[i])
        y_i = jax.lax.dynamic_slice(y, (0, i * util.gauss_points.size), (model.Brusselator.n_dim, util.gauss_points.size + 1))
        r_i = brusselator_bvp_interval(y_i, k, period, colloc_points, node_points)
        return i + 1, r_i
    
    colloc_eqs = jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1].ravel(order="C")
    return np.concatenate([colloc_eqs, y[:, -1] - y[:, 0]])

@jax.jit
def brusselator_bvp_jac(q, mesh_points=np.linspace(0, 1, 61)):
   
    n_mesh_intervals = mesh_points.size - 1
    k = np.exp(q[:model.Brusselator.n_par])
    n_points = (n_mesh_intervals * util.gauss_points.size + 1)
    y = q[model.Brusselator.n_par:model.Brusselator.n_par + n_points * model.Brusselator.n_dim].reshape(model.Brusselator.n_dim, n_points, order="F")
    period = q[model.Brusselator.n_par + n_points * model.Brusselator.n_dim]
    
    def loop_body(i, _):
        node_points = np.linspace(mesh_points[i], mesh_points[i + 1], util.gauss_points.size + 1)
        colloc_points = mesh_points[i] + util.gauss_points * (mesh_points[i + 1] - mesh_points[i])
        y_i = jax.lax.dynamic_slice(y, (0, i * util.gauss_points.size), (model.Brusselator.n_dim, util.gauss_points.size + 1))
        Jy_i = jax.jacfwd(brusselator_bvp_interval, argnums=0)(y_i, k, period, colloc_points, node_points)\
                .reshape((util.gauss_points.size * model.Brusselator.n_dim, (util.gauss_points.size + 1) * model.Brusselator.n_dim), order="F")
        Jk_i = jax.jacfwd(lambda x:brusselator_bvp_interval(y_i, np.exp(x), period, colloc_points, node_points))(q[:model.Brusselator.n_par])
        Jw_i = jax.jacfwd(brusselator_bvp_interval, argnums=2)(y_i, k, period, colloc_points, node_points)
        return i + 1, (Jy_i, np.hstack([Jk_i, Jw_i.reshape([Jw_i.size, 1])]))
    
    J = util.BVPJac(*jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1], model.Brusselator.n_dim, model.Brusselator.n_par)
    return J

@jax.jit
def brusselator_bvp_potential(q, mesh_points=np.linspace(0, 1, 61)):
    
    n_mesh_intervals = mesh_points.size - 1

    E = 0
    
    n_points = (n_mesh_intervals * util.gauss_points.size + 1)
    y = q[model.Brusselator.n_par:model.Brusselator.n_par + n_points * model.Brusselator.n_dim].reshape(model.Brusselator.n_dim, n_points, order="F")
    arclength = np.linalg.norm(y[:, 1:] - y[:, :-1], axis=0).sum()
    min_arclength = 0.3
    max_mesh_density = 10
    _, mesh_density = util.recompute_mesh(y, mesh_points, util.gauss_points)
    #mesh_quality = (mesh_points.size - 1) * (mesh_density[1:] + mesh_density[:-1]) * (mesh_points[1:] - mesh_points[:-1]) / (2 * jax.scipy.integrate.trapezoid(mesh_density, mesh_points))
    
    #E += 20 * (util.smooth_max(mesh_quality, smooth_max_temperature=6) - 1)**2
    mesh_density /= jax.scipy.integrate.trapezoid(mesh_density, x=mesh_points)
    mesh_density_max_val = util.smooth_max(mesh_density, smooth_max_temperature=6)

    E += np.where(mesh_density_max_val >= max_mesh_density, 5 * (mesh_density_max_val - max_mesh_density)**2, 0)
    E += 100 * np.where(np.abs(q[:model.Brusselator.n_par]) > np.log(100), (np.abs(q[:model.Brusselator.n_par]) - np.log(100))**2, 0).sum()
    E += np.where(arclength < min_arclength, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4, 0)
    
    return E

@jax.jit
def brusselator_log(y, k):
    return np.array([np.exp(-y[0]) + np.exp(k[0] + y[0] + y[1]) - np.exp(k[1]) - np.exp(k[2]),
                     -np.exp(k[0] + 2 * y[0]) + np.exp(k[1] + y[0] - y[1])])

@jax.jit
def brusselator_log_bvp_interval(y, k, period, colloc_points, node_points):

    dd = util.divided_difference(node_points, y)
    poly_interval = lambda t:util.newton_polynomial(t, node_points, y, dd)
    poly = jax.vmap(poly_interval)(colloc_points)
    poly_deriv = jax.vmap(jax.jacfwd(poly_interval))(colloc_points)
    return np.ravel(poly_deriv - jax.vmap(lambda yy:period * brusselator_log(yy, k))(poly), order="C")

@jax.jit
def brusselator_log_bvp(q, mesh_points=np.linspace(0, 1, 61)):
   
    n_mesh_intervals = mesh_points.size - 1
    k = q[:model.Brusselator.n_par]
    n_points = (n_mesh_intervals * util.gauss_points.size + 1)
    y = q[model.Brusselator.n_par:model.Brusselator.n_par + n_points * model.Brusselator.n_dim].reshape((model.Brusselator.n_dim, n_points), order="F")
    period = q[model.Brusselator.n_par + n_points * model.Brusselator.n_dim]
    
    def loop_body(i, _):
        node_points = np.linspace(mesh_points[i], mesh_points[i + 1], util.gauss_points.size + 1)
        colloc_points = mesh_points[i] + util.gauss_points * (mesh_points[i + 1] - mesh_points[i])
        y_i = jax.lax.dynamic_slice(y, (0, i * util.gauss_points.size), (model.Brusselator.n_dim, util.gauss_points.size + 1))
        r_i = brusselator_log_bvp_interval(y_i, k, period, colloc_points, node_points)
        return i + 1, r_i
    
    colloc_eqs = jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1].ravel(order="C")
    return np.concatenate([colloc_eqs, y[:, -1] - y[:, 0]])

@jax.jit
def brusselator_log_bvp_jac(q, mesh_points=np.linspace(0, 1, 61)):
   
    n_mesh_intervals = mesh_points.size - 1
    k = q[:model.Brusselator.n_par]
    n_points = (n_mesh_intervals * util.gauss_points.size + 1)
    y = q[model.Brusselator.n_par:model.Brusselator.n_par + n_points * model.Brusselator.n_dim].reshape((model.Brusselator.n_dim, n_points), order="F")
    period = q[model.Brusselator.n_par + n_points * model.Brusselator.n_dim]
    
    def loop_body(i, _):
        node_points = np.linspace(mesh_points[i], mesh_points[i + 1], util.gauss_points.size + 1)
        colloc_points = mesh_points[i] + util.gauss_points * (mesh_points[i + 1] - mesh_points[i])
        y_i = jax.lax.dynamic_slice(y, (0, i * util.gauss_points.size), (model.Brusselator.n_dim, util.gauss_points.size + 1))
        Jy_i = jax.jacfwd(brusselator_log_bvp_interval, argnums=0)(y_i, k, period, colloc_points, node_points)\
                .reshape((util.gauss_points.size * model.Brusselator.n_dim, (util.gauss_points.size + 1) * model.Brusselator.n_dim), order="F")
        Jk_i = jax.jacfwd(brusselator_log_bvp_interval, argnums=1)(y_i, k, period, colloc_points, node_points)
        Jw_i = jax.jacfwd(brusselator_log_bvp_interval, argnums=2)(y_i, k, period, colloc_points, node_points)
        return i + 1, (Jy_i, np.hstack([Jk_i, Jw_i.reshape([Jw_i.size, 1])]))
    
    J = util.BVPJac(*jax.lax.scan(loop_body, init=0, xs=None, length=n_mesh_intervals)[1], model.Brusselator.n_dim, model.Brusselator.n_par)
    return J

@jax.jit
def brusselator_log_bvp_potential(q, mesh_points=np.linspace(0, 1, 61)):

    n_mesh_intervals = mesh_points.size - 1
    E = 0

    n_points = (n_mesh_intervals * util.gauss_points.size + 1)
    logy = q[model.Brusselator.n_par:model.Brusselator.n_par + n_points * model.Brusselator.n_dim].reshape(model.Brusselator.n_dim, n_points, order="F")
    arclength = np.linalg.norm(logy[:, 1:] - logy[:, :-1], axis=0).sum()
    min_arclength = 0.3
    _, mesh_density = util.recompute_mesh(logy, mesh_points, util.gauss_points)
    mesh_mass_interval = (mesh_points[1:] - mesh_points[:-1]) * (mesh_density[1:] + mesh_density[:-1]) / 2
    mesh_density_peak = util.smooth_max(mesh_density, smooth_max_temperature=6)
    
    E += 100 * np.where(np.abs(q[:model.Brusselator.n_par]) > np.log(100), (np.abs(q[:model.Brusselator.n_par]) - np.log(100))**2, 0).sum()
    E += np.where(arclength < min_arclength, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4, 0)
    E += np.where(mesh_density_peak >= 5, (mesh_density_peak - 5)**2, 0)

    return E

@jax.jit
def morris_lecar_hb_potential(q, ode_model, bounds=None):
    k = q[:ode_model.n_par]
    hb_eval = q[ode_model.n_par + 2 * ode_model.n_dim]
    E = 0
    E += np.where(hb_eval < -9, 100 * (hb_eval + 9)**2 / 2, 0)
    if bounds is not None:
        E += np.where(k < bounds[:, 0], 100 * (k - bounds[:, 0])**2 / 2, 0).sum()
        E += np.where(k > bounds[:, 1], 100 * (k - bounds[:, 1])**2 / 2, 0).sum()
    return E

@jax.jit
def morris_lecar_bvp_potential(q, ode_model, mesh_points=np.linspace(0, 1, 61), bounds=None, bounds_membrane_voltage=None):

    n_mesh_intervals = mesh_points.size - 1
    E = 0

    n_points = (n_mesh_intervals * util.gauss_points.size + 1)
    k = q[:ode_model.n_par]
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape(ode_model.n_dim, n_points, order="F")
    arclength = np.linalg.norm(y[:, 1:] - y[:, :-1], axis=0).sum()
    min_arclength = 0.3
    _, mesh_density = util.recompute_mesh(y, mesh_points, util.gauss_points)
    mesh_mass_interval = (mesh_points[1:] - mesh_points[:-1]) * (mesh_density[1:] + mesh_density[:-1]) / 2
    mesh_density_peak = util.smooth_max(mesh_density, smooth_max_temperature=6)
    
    E += np.where(arclength < min_arclength, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4, 0)
    E += np.where(mesh_density_peak >= 5, (mesh_density_peak - 5)**2, 0)

    if bounds_membrane_potential is not None:
        E += 1e-2 * (util.smooth_max(y[0]) - bounds_membrane_voltage[1])**2
        E += 1e-2 * (-util.smooth_max(-y[0]) - bounds_membrane_voltage[0])**2

    if bounds is not None:
        E += np.where(k < bounds[:, 0], 100 * (k - bounds[:, 0])**2, 0).sum()
        E += np.where(k > bounds[:, 1], 100 * (k - bounds[:, 1])**2, 0).sum()

    return E

@partial(jax.jit, static_argnames=("n_mesh_intervals",))
def morris_lecar_mm_bvp_potential(q, ode_model, colloc_points_unshifted=util.gauss_points, bounds=None, bounds_membrane_voltage=None, n_mesh_intervals=60):

    E = 0

    n_points = (n_mesh_intervals * colloc_points_unshifted.size + 1)
    k = q[:ode_model.n_par]
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape(ode_model.n_dim, n_points, order="F")
    arclength = np.linalg.norm(y[:, 1:] - y[:, :-1], axis=0).sum()
    min_arclength = 0.3
    
    E += np.where(arclength < min_arclength, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4, 0)

    if bounds_membrane_voltage is not None:
        E += 100 * (util.smooth_max(y[0]) - bounds_membrane_voltage[1])**2
        E += 100 * (-util.smooth_max(-y[0]) - bounds_membrane_voltage[0])**2

    if bounds is not None:
        E += np.where(k < bounds[:, 0], 100 * (k - bounds[:, 0])**2, 0).sum()
        E += np.where(k > bounds[:, 1], 100 * (k - bounds[:, 1])**2, 0).sum()

    y_smooth = util.weighted_average_periodic_smoothing(y[:, :-1].T)
    E += 10 * np.sum((y_smooth - y[:, :-1].T)**2)

    return E

@partial(jax.jit, static_argnums=(1,))
def brusselator_bvp_fourier(q, fft_points=500):
    
    k = np.exp(q[:3])
    br = model.Brusselator(k)
    y_coeff = q[3:-1]
    y_coeff = y_coeff.reshape((model.Brusselator.n_dim, y_coeff.size // model.Brusselator.n_dim), order="F")
    basis_size = y_coeff.shape[1] // 2 + 1
    period = q[-1]
    ydot_coeff = np.hstack([np.zeros_like(y_coeff[:, :1]), 2 * np.pi * np.arange(1, basis_size) * y_coeff[:, basis_size:], -2 * np.pi * np.arange(1, basis_size) * y_coeff[:, 1:basis_size]])
    y = fft_points * np.fft.irfft(np.pad(util.fft_trigtoexp(y_coeff), ((0, 0), (0, fft_points // 2 - basis_size + 1))))
    ydot = fft_points * np.fft.irfft(np.pad(util.fft_trigtoexp(ydot_coeff), ((0, 0), (0, fft_points // 2 - basis_size + 1))))
    fy = jax.vmap(lambda x:period * br.f(0., x, k))(y.T).T
    fy_fft = np.fft.rfft(fy / fft_points)
    fy_coeff = util.fft_exptotrig(fy_fft[:, :basis_size])
    return (fy_coeff - ydot_coeff).ravel(order="F")

@partial(jax.jit, static_argnums=(1,))
def brusselator_bvp_fourier_potential(q, fft_points=500):
    
    y_coeff = q[3:-1]
    y_coeff = y_coeff.reshape((model.Brusselator.n_dim, y_coeff.size // model.Brusselator.n_dim), order="F")
    basis_size = y_coeff.shape[1] // 2 + 1
    period = q[-1]
    ydot_coeff = np.hstack([np.zeros_like(y_coeff[:, :1]), 2 * np.pi * np.arange(1, basis_size) * y_coeff[:, basis_size:], -2 * np.pi * np.arange(1, basis_size) * y_coeff[:, 1:basis_size]])
    yddot_coeff = y_coeff.at[:, 0].set(0)
    yddot_coeff = yddot_coeff.at[:, 1:yddot_coeff.shape[1] // 2  + 1].multiply(((1 + np.arange(0, yddot_coeff.shape[1] // 2)) * 2 * np.pi)**2)
    yddot_coeff = yddot_coeff.at[:, yddot_coeff.shape[1] // 2 + 1:].multiply(((1 + np.arange(0, yddot_coeff.shape[1] // 2)) * 2 * np.pi)**2)
    
    E = 0
    
    min_arclength = 0.5
    max_curvature = 5
    arclength = jax.scipy.integrate.trapezoid(np.linalg.norm(fft_points * np.fft.irfft(np.pad(util.fft_trigtoexp(ydot_coeff), ((0, 0), (0, fft_points // 2 - ydot_coeff.shape[1] // 2)))), axis=0), x=np.linspace(0, 1, fft_points))
    yddot = fft_points * np.fft.irfft(np.pad(util.fft_trigtoexp(yddot_coeff), ((0, 0), (0, fft_points // 2 - yddot_coeff.shape[1] // 2))))
    curvature = (1 + np.linalg.norm(yddot, axis=0)**2)**(1/4)
    curvature /= jax.scipy.integrate.trapezoid(curvature, np.linspace(0, 1, fft_points))
    
    E += 100 * np.where(np.abs(q[:model.Brusselator.n_par]) > np.log(100), (np.abs(q[:model.Brusselator.n_par]) - np.log(100))**2, 0).sum()
    E += np.where(arclength < min_arclength, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4, 0)
    E += np.where(util.smooth_max(curvature, smooth_max_temperature=6) > max_curvature, (util.smooth_max(curvature, smooth_max_temperature=6) - max_curvature)**2, 0)
    
    return E

@jax.jit
def repressilator_log_bvp_potential(q, ode_model, mesh_points=np.linspace(0, 1, 61)):
    
    n_mesh_intervals = mesh_points.size - 1
    E = 0

    n_points = (n_mesh_intervals * util.gauss_points.size + 1)
    k = q[:ode_model.n_par]
    y = q[ode_model.n_par:ode_model.n_par + n_points * ode_model.n_dim].reshape(ode_model.n_dim, n_points, order="F")
    arclength = np.linalg.norm(y[:, 1:] - y[:, :-1], axis=0).sum()
    min_arclength = 0.3
    _, mesh_density = util.recompute_mesh(y, mesh_points, util.gauss_points)
    mesh_mass_interval = (mesh_points[1:] - mesh_points[:-1]) * (mesh_density[1:] + mesh_density[:-1]) / 2
    mesh_density_peak = util.smooth_max(mesh_density, smooth_max_temperature=6)

    E += 100 * np.where(k[-ode_model.n_dim:] < 0, k[-ode_model.n_dim:]**2, 0.).sum()
    E += 100 * np.where(k[-ode_model.n_dim:] > 5, (k[-ode_model.n_dim:] - 5)**2, 0.).sum()
    E += 100 * np.where(k[:-ode_model.n_dim] > 5, (k[:-ode_model.n_dim] - 5)**2, 0,).sum()
    E += 100 * np.where(k[:-ode_model.n_dim] < -5, (k[:-ode_model.n_dim] + 5)**2, 0,).sum()
    E += np.where(arclength < min_arclength, (min_arclength / (np.sqrt(2) * arclength))**4 - (min_arclength / (np.sqrt(2) * arclength))**2 + 1 / 4, 0)
    E += np.where(mesh_density_peak >= 5, (mesh_density_peak - 5)**2, 0)

    return E

@jax.jit
def kai_sna_log(q):
    z = q[:model.KaiABC_nondim.n_par]
    y = q[model.KaiABC_nondim.n_par:model.KaiABC_nondim.n_par + model.KaiABC_nondim.n_dim + 2]
    z = np.concatenate([np.array([model.KaiABC_nondim.K[:, 0]@y]), z])
    return np.concatenate([model.KaiABC_nondim.S[1:-1]@np.exp(z), model.KaiABC_nondim.conservation_law@np.exp(y) - np.array([1, 6 / 35])])

@jax.jit
def fully_extended_hopf_kai_dae(q):
    y = q[model.KaiABC_nondim.n_par:model.KaiABC_nondim.n_par + model.KaiABC_nondim.n_dim + model.KaiABC_nondim.conservation_law.shape[0]]
    z = np.concatenate([model.KaiABC_nondim.K[:, :1].T@y, q[:model.KaiABC_nondim.n_par]])
    evec_real = q[model.KaiABC_nondim.n_par + model.KaiABC_nondim.n_dim + model.KaiABC_nondim.conservation_law.shape[0]:model.KaiABC_nondim.n_par + 2 * (model.KaiABC_nondim.n_dim + model.KaiABC_nondim.conservation_law.shape[0])]
    evec_imag = q[model.KaiABC_nondim.n_par + 2 * (model.KaiABC_nondim.n_dim + model.KaiABC_nondim.conservation_law.shape[0]):model.KaiABC_nondim.n_par + 3 * (model.KaiABC_nondim.n_dim + model.KaiABC_nondim.conservation_law.shape[0])]
    eval_imag = np.exp(q[model.KaiABC_nondim.n_par + 3 * (model.KaiABC_nondim.n_dim + model.KaiABC_nondim.conservation_law.shape[0])])
    A = np.identity(y.size).at[np.array([[0, -0], [-1, -1]])].set(0)
    J = (model.KaiABC_nondim.S * np.exp(z))@(model.KaiABC_nondim.K.T * np.exp(-y))
    J = J.at[0].set(model.KaiABC_nondim.conservation_law[0])
    J = J.at[-1].set(model.KaiABC_nondim.conservation_law[-1])
    
    return np.concatenate([kai_sna_log(q[:model.KaiABC_nondim.n_par + model.KaiABC_nondim.n_dim + model.KaiABC_nondim.conservation_law.shape[0]]),
                           J@evec_real + eval_imag * A@evec_imag,
                           J@evec_imag - eval_imag * A@evec_real,
                           np.ravel(evec_real@evec_real + evec_imag@evec_imag - 1),
                           evec_imag[:1]
                          ])

@jax.jit
def kai_sna_potential(q):
    z = q[:model.KaiABC_nondim.n_par]
    y = q[model.KaiABC_nondim.n_par:model.KaiABC_nondim.n_par + model.KaiABC_nondim.n_dim + model.KaiABC_nondim.conservation_law.shape[0]]
    k = z - model.KaiABC_nondim.K[:, 1:].T@y
    E = 0
    E += np.where(np.abs(k) > 7, 100 * (np.abs(k) - 7)**2 / 2, 0).sum()
    return E

@jax.jit
def kai_sna_hb_potential(q):
    z = q[:model.KaiABC_nondim.n_par]
    y = q[model.KaiABC_nondim.n_par:model.KaiABC_nondim.n_par + model.KaiABC_nondim.n_dim + model.KaiABC_nondim.conservation_law.shape[0]]
    log_eval_imag = q[model.KaiABC_nondim.n_par + 3 * (model.KaiABC_nondim.n_dim + model.KaiABC_nondim.conservation_law.shape[0])]
    E = kai_sna_potential(q)
    E += np.where(log_eval_imag < -9, 100 * (log_eval_imag + 9)**2 / 2, 0)

    pU_logsum = np.log(np.exp(y[:2]).sum())
    E += np.where(pU_logsum < -2, 10 * (pU_logsum + 2)**2 / 2, 0)
    pT_logsum = np.log(np.exp(y[2:4]).sum())
    E += np.where(pT_logsum < -2, 10 * (pT_logsum + 2)**2 / 2, 0)
    pD_logsum = np.log(np.exp(y[np.array([4, 5, 8, 9, 12, 13])]).sum())
    E += np.where(pD_logsum < -2, 10 * (pD_logsum + 2)**2 / 2, 0)
    pS_logsum = np.log(np.exp(y[np.array([6, 7, 10, 11, 14, 15])]).sum())
    E += np.where(pS_logsum < -2, 10 * (pS_logsum + 2)**2 / 2, 0)
    return E
