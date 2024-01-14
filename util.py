import jax
import jax.numpy as np
import numpy
from functools import partial
jax.config.update("jax_enable_x64", True)

midpoint = np.array([0])
lobatto_points_3 = np.array([-1, 0, 1])
gauss_points_2 = np.array([-np.sqrt(1/3), np.sqrt(1/3)])
gauss_points_3 = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
gauss_points = np.array([-np.sqrt(3/7 + (2/7) * np.sqrt(6/5)), -np.sqrt(3/7 - (2/7) * np.sqrt(6/5)), np.sqrt(3/7 - (2/7) * np.sqrt(6/5)), np.sqrt(3/7 + (2/7) * np.sqrt(6/5))])
gauss_weights = np.array([18 + np.sqrt(30), 18 - np.sqrt(30), 18 - np.sqrt(30), 18 + np.sqrt(30)]) / 36

@jax.jit
def smooth_max(x, smooth_max_temperature=1):
    return np.sum(x * np.exp(smooth_max_temperature * x)) / np.sum(np.exp(smooth_max_temperature * x))

@jax.jit
def fill_mesh(t, gauss_points=gauss_points):
    return np.concatenate([np.ravel(np.expand_dims(t[:-1], 1) + np.expand_dims(t[1:] - t[:-1], 1) * np.linspace(0, 1, gauss_points.size + 1)[:-1]), t[-1:]])

expand_mesh = fill_mesh

@jax.jit
def fft_trigtoexp(x):
    return np.hstack([x[:, :1], (x[:, 1:x.shape[1] // 2 + 1] - 1j * x[:, x.shape[1] // 2 + 1:]) / 2])

@jax.jit
def fft_exptotrig(x):
    return np.hstack([x[:, :1].real, 2 * x[:, 1:].real, -2 * x[:, 1:].imag])

@jax.jit
def divided_difference(node_t, node_y):

    node_y = node_y.T

    def loop_body(carry, _):

        i, prev = carry
        numer = np.zeros_like(prev)
        numer = numer.at[1:].set(prev[1:] - prev[:-1])
        denom = node_t - np.roll(node_t, i)
        dd = (numer.T / denom).T

        return (i + 1, dd), dd

    return np.vstack([np.array([node_y]), jax.lax.scan(loop_body, init=(1, node_y), xs=None, length=node_t.size - 1)[1]])

@jax.jit
def newton_polynomial(t, node_t, node_y, dd=None):
    
    if dd is None:
        dd = divided_difference(node_t, node_y)
        
    return np.sum(np.cumprod(np.roll(t - node_t, 1).at[0].set(1)) * dd[np.diag_indices(node_t.size)].T, axis=1)

@partial(jax.jit, static_argnums=1)
def weighted_average_smoothing(x, n_smooth=4):
    
    def loop_inner(carry, _):
        i, x = carry
        x = x.at[i].set(x[i - 1] / 4 + x[i] / 2 + x[i + 1] / 4)
        return (i + 1, x), _
    
    def loop_outer(carry, _):
        i, x = carry
        x = x.at[0].set((x[0] + x[1]) / 2)
        x = jax.lax.scan(loop_inner, init=(1, x), xs=None, length=x.size - 2)[0][1]
        x = x.at[-1].set((x[-1] + x[-2]) / 2)
        return (i + 1, x), _
    
    x = jax.lax.scan(loop_outer, init=(0, x), xs=None, length=n_smooth)[0][1]
    
    return x

@partial(jax.jit, static_argnums=1)
def weighted_average_periodic_smoothing(x, n_smooth=4):
    
    def loop_inner(carry, _):
        i, x = carry
        x = x.at[i].set(x[(i - 1) % x.shape[0]] / 4 + x[i] / 2 + x[(i + 1) % x.shape[0]] / 4)
        return (i + 1, x), _
    
    def loop_outer(carry, _):
        i, x = carry
        x = jax.lax.scan(loop_inner, init=(0, x), xs=None, length=x.size)[0][1]
        return (i + 1, x), _
    
    x = jax.lax.scan(loop_outer, init=(0, x), xs=None, length=n_smooth)[0][1]
    
    return x

@partial(jax.jit, static_argnums=3)
def recompute_mesh(y, mesh_old, colloc_points_unshifted=gauss_points, n_smooth=4):
    
    def loop_body(i, _):
        meshi = np.linspace(*jax.lax.dynamic_slice(mesh_old, (i,), (2,)), colloc_points_unshifted.size + 1)
        yi = jax.lax.dynamic_slice(y, (0, i * colloc_points_unshifted.size), (y.shape[0], colloc_points_unshifted.size + 1))
        return i + 1, divided_difference(meshi, yi)[colloc_points_unshifted.size, colloc_points_unshifted.size]
    
    _, deriv = jax.lax.scan(loop_body, init=0, xs=None, length=mesh_old.size - 1)
    midpoints = (mesh_old[1:] + mesh_old[:-1]) / 2
    deriv = np.pad((deriv[1:] - deriv[:-1]).T / (midpoints[1:] - midpoints[:-1]), ((0, 0), (1, 1)), mode="edge")
    a = np.maximum(1, (jax.scipy.integrate.trapezoid(np.sum(deriv**2, axis=0)**(1 / (1 + 2 * (colloc_points_unshifted.size + 1))), x=mesh_old) / (mesh_old[-1] - mesh_old[0]))**(1 + 2 * (colloc_points_unshifted.size + 1)))
    mesh_density = (1 + np.sum(deriv**2, axis=0) / a)**(1 / (1 + 2 * (colloc_points_unshifted.size + 1)))
    mesh_density = weighted_average_smoothing(mesh_density, n_smooth)
    mesh_mass = np.pad(np.cumsum((mesh_density[1:] + mesh_density[:-1]) * (mesh_old[1:] - mesh_old[:-1])) / 2, (1, 0))
    mesh_new = np.interp(np.linspace(0, mesh_mass[-1], mesh_old.size), mesh_mass, mesh_old)
    
    return mesh_new, mesh_density

@jax.jit
def recompute_node_y(y, mesh_old, mesh_new, gauss_points=gauss_points):

    t_eval = fill_mesh(mesh_new, gauss_points)
    y_interp = interpolate(y, mesh_old, t_eval[1:-1], gauss_points)
    return np.hstack([y[:, :1], y_interp, y[:, -1:]])

@jax.jit
def interpolate(y, mesh_points, t_eval, gauss_points=gauss_points):
    
    def loop1(i, _):
        meshi = np.linspace(*jax.lax.dynamic_slice(mesh_points, (i,), (2,)), gauss_points.size + 1)
        yi = jax.lax.dynamic_slice(y, (0 * i, i * gauss_points.size,), (y.shape[0], gauss_points.size + 1))
        return i + 1, divided_difference(meshi, yi)
        
    dd = jax.lax.scan(loop1, init=0, xs=None, length=mesh_points.size - 1)[1]
        
    def loop2(_, t):
        i = np.maximum(np.searchsorted(mesh_points, t) - 1, 0)
        meshi = np.linspace(*jax.lax.dynamic_slice(mesh_points, (i,), (2,)), gauss_points.size + 1)
        yi = jax.lax.dynamic_slice(y, (0 * i, i * gauss_points.size,), (y.shape[0], gauss_points.size + 1))
        return _, newton_polynomial(t, meshi, yi, dd[i])
    
    _, y_interp = jax.lax.scan(loop2, init=None, xs=t_eval)
    return y_interp.T

@partial(jax.jit, static_argnums=(1, 2))
def permute_q_mesh(x, n_dim, n_mesh_intervals, colloc_points_unshifted=gauss_points):
   
    is_vector = len(x.shape) == 1
    if is_vector:
        x = np.expand_dims(x, 0)

    x = np.hstack([x[:, :n_dim], np.concatenate([x[:, n_dim:-n_dim * colloc_points_unshifted.size - n_mesh_intervals + 1].reshape((x.shape[0], n_dim * colloc_points_unshifted.size, n_mesh_intervals - 1), order="F"),
                                                           np.expand_dims(x[:, -n_mesh_intervals + 1:], 1)], axis=1).reshape((x.shape[0], (n_dim * colloc_points_unshifted.size + 1) * (n_mesh_intervals - 1)), order="F"), 
                   x[:, -n_dim * colloc_points_unshifted.size - n_mesh_intervals + 1:-n_mesh_intervals + 1]])

    if is_vector:
        x = np.ravel(x)

    return x

@partial(jax.jit, static_argnames=("n_dim", "n_mesh_intervals"))
def permute_q_mesh_1(x, n_dim, n_mesh_intervals, colloc_points_unshifted=gauss_points):
   
    is_vector = len(x.shape) == 1
    if is_vector:
        x = np.expand_dims(x, 0)

    x = np.hstack([x[:, :n_dim], np.concatenate([np.expand_dims(x[:, -n_mesh_intervals:], 1), x[:, n_dim:-n_mesh_intervals].reshape((x.shape[0], n_dim * colloc_points_unshifted.size, n_mesh_intervals), order="F")],
                  axis=1).reshape((x.shape[0], (n_dim * colloc_points_unshifted.size + 1) * n_mesh_intervals), order="F")])

    if is_vector:
        x = np.ravel(x)

    return x


@partial(jax.jit, static_argnums=(1, 2))
def unpermute_q_mesh(x, n_dim, n_mesh_intervals, colloc_points_unshifted=gauss_points):
    
    is_vector = len(x.shape) == 1
    if is_vector:
        x = np.expand_dims(x, 0)

    y_and_mesh = x[:, n_dim:-n_dim * colloc_points_unshifted.size].reshape((x.shape[0], n_dim * colloc_points_unshifted.size + 1, n_mesh_intervals - 1), order="F")
    x = np.hstack([x[:, :n_dim], y_and_mesh[:, :-1, :].reshape((x.shape[0], n_dim * colloc_points_unshifted.size * (n_mesh_intervals - 1)), order="F"), 
                   x[:, -n_dim * colloc_points_unshifted.size:],
                   y_and_mesh[:, -1:, :].reshape((x.shape[0], n_mesh_intervals - 1), order="F")])

    if is_vector:
        x = np.ravel(x)

    return x

@partial(jax.jit, static_argnames=("n_dim", "n_mesh_intervals"))
def unpermute_q_mesh_1(x, n_dim, n_mesh_intervals, colloc_points_unshifted=gauss_points):
    
    is_vector = len(x.shape) == 1
    if is_vector:
        x = np.expand_dims(x, 0)

    y_and_mesh = x[:, n_dim:].reshape((x.shape[0], n_dim * colloc_points_unshifted.size + 1, n_mesh_intervals), order="F")
    x = np.hstack([x[:, :n_dim], y_and_mesh[:, 1:, :].reshape((x.shape[0], n_dim * colloc_points_unshifted.size * n_mesh_intervals), order="F"), y_and_mesh[:, :1, :].reshape((x.shape[0], n_mesh_intervals), order="F")])

    if is_vector:
        x = np.ravel(x)

    return x

class BVPJac:

    def __init__(self, Jy, Jk, n_dim, n_par, Jbc_left=None, Jbc_right=None):
        self.Jy = Jy
        self.Jk = Jk
        self.n_dim = n_dim
        self.n_par = n_par
        self.shape = ((Jy.shape[0] * Jy.shape[1] + n_dim, Jy.shape[0] * Jy.shape[1] + n_dim + Jk.shape[2]))

        if Jbc_left is None:
            self.Jbc_left = -np.identity(n_dim)
        else:
            self.Jbc_left = Jbc_left

        if Jbc_right is None:
            self.Jbc_right = np.identity(n_dim)
        else:
            self.Jbc_right = Jbc_right

    @jax.jit
    def todense(self):

        Jk = np.pad(np.vstack(self.Jk), ((0, self.n_dim), (0, 0)))
        Jy_dense = np.zeros((self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim, self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim))
        Jy_dense = Jy_dense.at[-self.n_dim:, :self.n_dim].set(self.Jbc_left)
        Jy_dense = Jy_dense.at[-self.n_dim:, -self.n_dim:].set(self.Jbc_right)

        def loop_body(carry, _):
            i, Jy_dense = carry
            Jy_dense = jax.lax.dynamic_update_slice(Jy_dense, self.Jy[i], (i * self.Jy.shape[1], i * self.Jy.shape[1]))
            return (i + 1, Jy_dense), _

        Jy_dense = jax.lax.scan(loop_body, init=(0, Jy_dense), xs=None, length=self.Jy.shape[0])[0][1]
        return np.hstack([Jk[:, :self.n_par], Jy_dense, Jk[:, self.n_par:]])

    @jax.jit
    def left_multiply(self, v):
   
        is_vector = len(v.shape) == 1
        if is_vector:
            v = np.expand_dims(v, 0)

        out = np.zeros((v.shape[0], self.shape[1]))
        out = out.at[:, :self.n_par].set(v[:, :-self.n_dim]@np.vstack(self.Jk[:, :, :self.n_par]))
        out = out.at[:, self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim:].set(v[:, :-self.n_dim]@np.vstack(self.Jk[:, :, self.n_par:]))
        out = out.at[:, self.n_par:self.n_par + self.n_dim].set(v[:, -self.n_dim:]@self.Jbc_left)
        out = out.at[:, self.n_par + self.Jy.shape[0] * self.Jy.shape[1]:self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim].set(v[:, -self.n_dim:]@self.Jbc_right)
        
        def loop_body(carry, _):
            i, out = carry
            outi = jax.lax.dynamic_slice(out, (0, self.n_par + i * self.Jy.shape[1]), (out.shape[0], self.Jy.shape[2]))
            vi = jax.lax.dynamic_slice(v, (0, i * self.Jy.shape[1]), (v.shape[0], self.Jy.shape[1]))
            out = jax.lax.dynamic_update_slice(out, outi + vi@self.Jy[i], (0, self.n_par + i * self.Jy.shape[1]))
            return (i + 1, out), _

        out = jax.lax.scan(loop_body, init=(0, out), xs=None, length=self.Jy.shape[0])[0][1]

        if is_vector:
            out = out.ravel()

        return out
   
    @jax.jit
    def right_multiply(self, v):
        
        is_vector = len(v.shape) == 1
        if is_vector:
            v = np.expand_dims(v, 1)

        vy = v[self.n_par:self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim]
        vk = np.concatenate([v[:self.n_par], v[self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim:]])
        out = np.pad(np.vstack(self.Jk)@vk, ((0, self.n_dim), (0, 0)))

        def loop_body(carry, _):
            i, out = carry
            outi = jax.lax.dynamic_slice(out, (i * self.Jy.shape[1], 0), (self.Jy.shape[1], out.shape[1]))
            vyi = jax.lax.dynamic_slice(vy, (i * self.Jy.shape[1], 0), (self.Jy.shape[2], out.shape[1]))
            out = jax.lax.dynamic_update_slice(out, outi + self.Jy[i]@vyi, (i * self.Jy.shape[1], 0))
            return (i + 1, out), _

        out = jax.lax.scan(loop_body, init=(0, out), xs=None, length=self.Jy.shape[0])[0][1]
        out = out.at[-self.n_dim:].add(self.Jbc_left@vy[:self.n_dim] + self.Jbc_right@vy[-self.n_dim:])

        if is_vector:
            out = out.ravel()

        return out

    @jax.jit
    def right_multiply_diag(self, D):

        Dy = D[self.n_par:self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim]
        Dk = np.concatenate([D[:self.n_par], D[self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim:]])

        def loop_body(carry, _):
            i, Jy = carry
            Dyi = jax.lax.dynamic_slice(Dy, (i * Jy.shape[1],), (Jy.shape[1] + self.n_dim,))
            Jy = Jy.at[i].multiply(Dyi)
            return (i + 1, Jy), _

        Jy = jax.lax.scan(loop_body, init=(0, self.Jy), xs=None, length=self.Jy.shape[0])[0][1]
        Jk = np.reshape(np.vstack(self.Jk) * Dk, (self.Jk.shape))
        Jbc_left = self.Jbc_left * D[self.n_par:self.n_par + self.n_dim]
        Jbc_right = self.Jbc_right * D[self.n_par + self.Jy.shape[0] * self.Jy.shape[1]:self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim]
        
        return BVPJac(Jy, Jk, self.n_dim, self.n_par, Jbc_left, Jbc_right)

    @jax.jit
    def multiply_transpose(J1, J2):
        
        def loop1(i, _):
            return i + 1, np.hstack([J1.Jy[i, :, :J1.n_dim]@J2.Jy[i - 1, :, -J1.n_dim:].T, J1.Jy[i]@J2.Jy[i].T, J1.Jy[i, :, -J1.n_dim:]@J2.Jy[i + 1, :, :J1.n_dim].T])
        
        Jy1Jy2T = jax.lax.scan(loop1, init=1, xs=None, length=J1.Jy.shape[0] - 2)[1]
        Jy1Jy2T = np.vstack([[np.pad(np.hstack([J1.Jy[0]@J2.Jy[0].T, J1.Jy[0, :, -J1.n_dim:]@J2.Jy[1, :, :J1.n_dim].T]), ((0, 0), (J1.Jy.shape[1], 0)))], 
                             Jy1Jy2T, 
                             [np.pad(np.hstack([J1.Jy[-1, :, :J1.n_dim]@J2.Jy[-2, :, -J1.n_dim:].T, J1.Jy[-1]@J2.Jy[-1].T]), ((0, 0), (0, J1.Jy.shape[1])))]])
        
        J1J2T = np.pad(np.vstack(J1.Jk)@np.vstack(J2.Jk).T, ((0, J1.n_dim), (0, J2.n_dim)))
        
        def loop2(carry, _):
            i, J1J2T = carry
            J1J2Ti = jax.lax.dynamic_slice(J1J2T, (i * Jy1Jy2T.shape[1], (i - 1) * Jy1Jy2T.shape[1]), (Jy1Jy2T.shape[1], 3 * Jy1Jy2T.shape[1]))
            J1J2T = jax.lax.dynamic_update_slice(J1J2T, J1J2Ti + Jy1Jy2T[i, :], (i * Jy1Jy2T.shape[1], (i - 1) * Jy1Jy2T.shape[1]))
            return (i + 1, J1J2T), _
            
        J1J2T = jax.lax.scan(loop2, init=(1, J1J2T), xs=None, length=J1.Jy.shape[0] - 2)[0][1]
        J1J2T = J1J2T.at[:Jy1Jy2T.shape[1], :2 * Jy1Jy2T.shape[1]].add(Jy1Jy2T[0, :, Jy1Jy2T.shape[1]:])
        J1J2T = J1J2T.at[(J1.Jy.shape[0] - 1) * Jy1Jy2T.shape[1]:J1.Jy.shape[0] * Jy1Jy2T.shape[1], (J1.Jy.shape[0] - 2) * Jy1Jy2T.shape[1]:J1.Jy.shape[0] * Jy1Jy2T.shape[1]].add(Jy1Jy2T[-1, :, :2 * Jy1Jy2T.shape[1]])
        J1J2T = J1J2T.at[-J1.n_dim:, -J1.n_dim:].set(J1.Jbc_left@J2.Jbc_left + J1.Jbc_right@J2.Jbc_right)
        J1J2T = J1J2T.at[:Jy1Jy2T.shape[1], -J1.n_dim:].set(J1.Jy[0, :, :J1.n_dim]@J2.Jbc_left)
        J1J2T = J1J2T.at[-J1.n_dim:, :Jy1Jy2T.shape[1]].set(J1.Jbc_left@J2.Jy[0, :, :J1.n_dim].T)
        J1J2T = J1J2T.at[-Jy1Jy2T.shape[1] - J1.n_dim:-J1.n_dim, -J1.n_dim:].set(J1.Jy[-1, :, -J1.n_dim:]@J2.Jbc_right)
        J1J2T = J1J2T.at[-J1.n_dim:, -Jy1Jy2T.shape[1] - J1.n_dim:-J1.n_dim].set(J1.Jbc_right@J2.Jy[-1, :, -J2.n_dim:].T)
        
        return J1J2T

    @jax.jit
    def lq_factor(self):

        Q_c = np.zeros((self.Jy.shape[0], self.Jy.shape[2], self.Jy.shape[2]))
        R_c = np.zeros((self.Jy.shape[0], 2 * self.Jy.shape[1], self.Jy.shape[1]))
        R_bc = np.zeros((self.n_dim + self.Jy.shape[1] * self.Jy.shape[0], self.n_dim))
        R_bc = R_bc.at[:self.n_dim].set(self.Jbc_left)
        R_bc = R_bc.at[-self.n_dim:].set(self.Jbc_right)
        qi, ri = jax.scipy.linalg.qr(self.Jy[0].T)
        Q_c = Q_c.at[0].set(qi)
        R_c = R_c.at[0, -ri.shape[1]:].set(ri[:-self.n_dim])
        R_bc = R_bc.at[:qi.shape[1]].set(qi.T@R_bc[:qi.shape[1]])

        def loop_body(carry, _):
            i, Q_c, R_c, R_bc = carry
            si = Q_c[i - 1, -self.n_dim:].T@self.Jy[i, :, :self.n_dim].T
            R_c = R_c.at[i, :self.Jy.shape[2]].set(si)
            bc = jax.lax.dynamic_slice(R_bc, (i * self.Jy.shape[1], 0), (self.Jy.shape[2], self.n_dim))
            qi, ri = jax.scipy.linalg.qr(self.Jy[i].T.at[:self.n_dim].set(si[-self.n_dim:]))
            Q_c = Q_c.at[i].set(qi)
            R_c = R_c.at[i, -ri.shape[1]:].set(ri[:-self.n_dim])
            R_bc = jax.lax.dynamic_update_slice(R_bc, qi.T@bc, (i * self.Jy.shape[1], 0))
            return (i + 1, Q_c, R_c, R_bc), _

        out = jax.lax.scan(loop_body, init=(1, Q_c, R_c, R_bc), xs=None, length=self.Jy.shape[0] - 1)
        Q_bc, ri = np.linalg.qr(out[0][3][-self.n_dim:])
        R_bc = out[0][3].at[-self.n_dim:].set(ri)

        return BVPJac_LQ(out[0][1], Q_bc, out[0][2], R_bc)
    
    def _tree_flatten(self):
        children = (self.Jy, self.Jk, self.Jbc_left, self.Jbc_right)
        aux_data = {"n_dim":self.n_dim, "n_par":self.n_par}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children[:2], Jbc_left=children[2], Jbc_right=children[3], **aux_data)

class BVPJac_LQ:

    def __init__(self, Q_c, Q_bc, R_c, R_bc):

        self.Q_c = Q_c
        self.Q_bc = Q_bc
        self.R_c = R_c
        self.R_bc = R_bc

    @jax.jit
    def solve_triangular_L(self, b):

        is_vector = len(b.shape) == 1
        if is_vector:
            b = np.expand_dims(b, 1)

        x = np.zeros((self.R_bc.shape[0], b.shape[1]))
        x_i = jax.scipy.linalg.solve_triangular(self.R_c[0, self.R_c.shape[2]:].T, b[:self.R_c.shape[2]], lower=True)
        x = x.at[:self.R_c.shape[1] // 2].set(x_i)

        def loop_body(carry, _):

            i, x = carry
            x_i_prev = jax.lax.dynamic_slice(x, ((i - 1) * self.R_c.shape[2], 0), (self.R_c.shape[2], x.shape[1]))
            b_i = jax.lax.dynamic_slice(b, (i * self.R_c.shape[2], 0), (self.R_c.shape[2], b.shape[1]))
            x_i = jax.scipy.linalg.solve_triangular(self.R_c[i, self.R_c.shape[2]:].T, b_i - self.R_c[i, :self.R_c.shape[2]].T@x_i_prev, lower=True)
            x = jax.lax.dynamic_update_slice(x, x_i, (i * self.R_c.shape[2], 0))
            return (i + 1, x), _

        x = jax.lax.scan(loop_body, init=(1, x), xs=None, length=self.R_c.shape[0] - 1)[0][1]
        xi = jax.scipy.linalg.solve_triangular(self.R_bc[-self.R_bc.shape[1]:].T, b[-self.R_bc.shape[1]:] - self.R_bc[:-self.R_bc.shape[1]].T@x[:-self.R_bc.shape[1]], lower=True)
        x = x.at[-self.R_bc.shape[1]:].set(xi)

        if is_vector:
            x = x.ravel()

        return x

    @jax.jit
    def solve_triangular_R(self, b):

        is_vector = len(b.shape) == 1
        if is_vector:
            b = np.expand_dims(b, 1)

        x = np.zeros((self.R_bc.shape[0], b.shape[1]))
        x_i = jax.scipy.linalg.solve_triangular(self.R_bc[-self.R_bc.shape[1]:], b[-self.R_bc.shape[1]:], lower=False)
        x = x.at[-self.R_bc.shape[1]:].set(x_i)
        x_i = jax.scipy.linalg.solve_triangular(self.R_c[-1, self.R_c.shape[2]:], b[-self.R_c.shape[2] - self.R_bc.shape[1]:-self.R_bc.shape[1]] - self.R_bc[-self.R_c.shape[2] - self.R_bc.shape[1]:-self.R_bc.shape[1]]@x[-self.R_bc.shape[1]:], lower=False)
        x = x.at[-self.R_c.shape[2] - self.R_bc.shape[1]:-self.R_bc.shape[1]].set(x_i)

        def loop_body(carry, _):

            i, x = carry
            x_i_prev = jax.lax.dynamic_slice(x, ((i + 1) * self.R_c.shape[2], 0), (self.R_c.shape[2], x.shape[1]))
            b_i = jax.lax.dynamic_slice(b, (i * self.R_c.shape[2], 0), (self.R_c.shape[2], b.shape[1]))
            self.R_bc_i = jax.lax.dynamic_slice(self.R_bc, (i * self.R_c.shape[2], 0), (self.R_c.shape[2], self.R_bc.shape[1]))
            b_i -= self.R_c[i + 1, :self.R_c.shape[2]]@x_i_prev + self.R_bc_i@x[-self.R_bc.shape[1]:]
            x_i = jax.scipy.linalg.solve_triangular(self.R_c[i, self.R_c.shape[2]:], b_i, lower=False)
            x = jax.lax.dynamic_update_slice(x, x_i, (i * self.R_c.shape[2], 0))
            return (i - 1, x), _

        x = jax.lax.scan(loop_body, init=(self.R_c.shape[0] - 2, x), xs=None, length=self.R_c.shape[0] - 1)[0][1]

        if is_vector:
            x = x.ravel()

        return x

    @jax.jit
    def Q_right_multiply(self, v):
        
        is_vector = len(v.shape) == 1
        if is_vector:
            v = np.expand_dims(v, 1)

        v = v.at[-self.Q_bc.shape[1]:].set(self.Q_bc@v[-self.Q_bc.shape[1]:])

        def loop_body(carry, _):
            i, v = carry
            v_i = jax.lax.dynamic_slice(v, (i * self.R_c.shape[2], 0), (self.Q_c.shape[1], v.shape[1]))
            v = jax.lax.dynamic_update_slice(v, self.Q_c[i]@v_i, (i * self.R_c.shape[2], 0))
            return (i - 1, v), _

        i, v = jax.lax.scan(loop_body, init=(self.R_c.shape[0] - 1, v), xs=None, length=self.R_c.shape[0])[0]

        if is_vector:
            v = v.ravel()

        return v

    def _tree_flatten(self):
        children = (self.Q_c, self.Q_bc, self.R_c, self.R_bc)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

class BVPMMJac:

    def __init__(self, Jy, Jk, Jmesh, n_dim, n_par, Jbc_left=None, Jbc_right=None, colloc_points_unshifted=gauss_points):
        self.Jy = Jy
        self.Jk = Jk
        self.Jmesh = Jmesh
        self.n_dim = n_dim
        self.n_par = n_par
        self.shape = ((Jy.shape[0] * Jy.shape[1] + n_dim + Jmesh.shape[0], Jy.shape[0] * Jy.shape[1] + n_dim + Jy.shape[0] - 1 + Jk.shape[2]))
        self.colloc_points_unshifted = colloc_points_unshifted

        if Jbc_left is None:
            self.Jbc_left = -np.identity(n_dim)
        else:
            self.Jbc_left = Jbc_left

        if Jbc_right is None:
            self.Jbc_right = np.identity(n_dim)
        else:
            self.Jbc_right = Jbc_right


    @jax.jit
    def todense(self):

        Jk = np.pad(np.vstack(self.Jk), ((0, self.n_dim), (0, 0)))
        Jy_dense = np.zeros((self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim, self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim))
        Jm_dense = np.zeros((self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim, self.Jmesh.shape[0]))

        Jy_dense = Jy_dense.at[-self.n_dim:, :self.n_dim].set(self.Jbc_left)
        Jy_dense = Jy_dense.at[-self.n_dim:, -self.n_dim:].set(self.Jbc_right)

        Jy_dense_0 = np.hstack([self.Jy[0, :, :self.n_dim], self.Jy[0, :, self.n_dim + 1:-1]])
        Jy_dense = Jy_dense.at[:self.Jy.shape[1], :self.Jy.shape[1] + self.n_dim].set(Jy_dense_0)
        Jm_dense = Jm_dense.at[:self.Jy.shape[1], 0].set(self.Jy[0, :, -1])

        Jy_dense_N = np.hstack([self.Jy[-1, :, :self.n_dim], self.Jy[-1, :, self.n_dim + 1:-1]])
        Jy_dense = Jy_dense.at[(self.Jy.shape[0] - 1) * self.Jy.shape[1]:self.Jy.shape[0] * self.Jy.shape[1], (self.Jy.shape[0] - 1) * self.Jy.shape[1]:(self.Jy.shape[0] - 1) * self.Jy.shape[1] + 1 + Jy_dense_N.shape[1]].set(Jy_dense_N)
        Jm_dense = Jm_dense.at[(self.Jy.shape[0] - 1) * self.Jy.shape[1]:self.Jy.shape[0] * self.Jy.shape[1], -1].set(self.Jy[-1, :, self.n_dim])

        def loop_body(carry, _):
            i, Jy_dense, Jm_dense = carry
            Jy_dense_i = np.hstack([self.Jy[i, :, :self.n_dim], jax.lax.dynamic_slice(self.Jy[i], (0, self.n_dim + 1), (self.Jy.shape[1], self.Jy.shape[2] - 2 - self.n_dim))])
            Jy_dense = jax.lax.dynamic_update_slice(Jy_dense, Jy_dense_i, (i * self.Jy.shape[1], i * self.Jy.shape[1]))
            Jm_dense = jax.lax.dynamic_update_slice(Jm_dense, jax.lax.dynamic_slice(self.Jy[i], (0, self.n_dim), (self.Jy.shape[1], 1)), (i * self.Jy.shape[1], i - 1))
            Jm_dense = jax.lax.dynamic_update_slice(Jm_dense, self.Jy[i, :, -1:], (i * self.Jy.shape[1], i))
            return (i + 1, Jy_dense, Jm_dense), _

        Jy_mesh_dense = np.hstack(jax.lax.scan(loop_body, init=(1, Jy_dense, Jm_dense), xs=None, length=self.Jy.shape[0] - 2)[0][1:])
        return np.vstack([np.hstack([Jk[:, :self.n_par], Jy_mesh_dense, Jk[:, self.n_par:]]), np.pad(unpermute_q_mesh(self.Jmesh, self.n_dim, self.Jy.shape[0], self.colloc_points_unshifted), 
                          ((0, 0), (self.n_par, self.Jk.shape[2] - self.n_par)))])

    @jax.jit
    def left_multiply(self, v):
   
        is_vector = len(v.shape) == 1
        if is_vector:
            v = np.expand_dims(v, 0)

        out = np.zeros((v.shape[0], self.shape[1]))
        out = out.at[:, :self.n_par].add(v[:, :-self.n_dim - self.Jmesh.shape[0]]@np.vstack(self.Jk[:, :, :self.n_par]))
        out = out.at[:, self.n_par - self.Jk.shape[2]:].add(v[:, :-self.n_dim - self.Jmesh.shape[0]]@np.vstack(self.Jk[:, :, self.n_par:]))
     
        out = out.at[:, self.n_par:self.n_par + self.Jmesh.shape[1]].add(v[:, -self.Jmesh.shape[0]:]@self.Jmesh)
     
        out = out.at[:, self.n_par:self.n_par + self.n_dim].add(v[:, -self.Jmesh.shape[0] - self.n_dim:-self.Jmesh.shape[0]]@self.Jbc_left)
        out = out.at[:, self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.Jmesh.shape[0]:self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.Jmesh.shape[0] + self.n_dim]\
              .add(v[:, -self.Jmesh.shape[0] - self.n_dim:-self.Jmesh.shape[0]]@self.Jbc_right)

        Jy_0 = np.hstack([self.Jy[0, :, :self.n_dim], self.Jy[0, :, self.n_dim + 1:]])
        out = out.at[:, self.n_par:self.n_par + Jy_0.shape[1]].add(v[:, :self.Jy.shape[1]]@Jy_0)

        Jy_N = self.Jy[-1, :, :-1]
        out = out.at[:, self.n_par - self.Jk.shape[2] - Jy_N.shape[1]:self.n_par - self.Jk.shape[2]].add(v[:, -self.Jmesh.shape[0] - self.Jy.shape[1] - self.n_dim:-self.Jmesh.shape[0] - self.n_dim]@Jy_N)
        
        def loop_body(carry, _):
            i, out = carry
            out_i = jax.lax.dynamic_slice(out, (0, self.n_par + i * (1 + self.Jy.shape[1]) - 1), (out.shape[0], self.Jy.shape[2]))
            v_i = jax.lax.dynamic_slice(v, (0, i * self.Jy.shape[1]), (v.shape[0], self.Jy.shape[1]))
            out = jax.lax.dynamic_update_slice(out, out_i + v_i@self.Jy[i], (0, self.n_par + i * (1 + self.Jy.shape[1]) - 1))
            return (i + 1, out), _

        out = jax.lax.scan(loop_body, init=(1, out), xs=None, length=self.Jy.shape[0] - 2)[0][1]
        out = out.at[:, self.n_par:self.n_par - self.Jk.shape[2]].set(unpermute_q_mesh(out[:, self.n_par:self.n_par - self.Jk.shape[2]], self.n_dim, self.Jy.shape[0], self.colloc_points_unshifted))

        if is_vector:
            out = out.ravel()

        return out
   
    @jax.jit
    def right_multiply(self, v):
       
        is_vector = len(v.shape) == 1
        if is_vector:
            v = np.expand_dims(v, 1)

        vy = v[self.n_par:self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim + self.Jmesh.shape[0]]
        vy = permute_q_mesh(vy.T, self.n_dim, self.Jy.shape[0], self.colloc_points_unshifted).T
        vk = np.concatenate([v[:self.n_par], v[self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim + self.Jmesh.shape[0]:]])
        out = np.vstack([np.pad(np.vstack(self.Jk)@vk, ((0, self.n_dim), (0, 0))), self.Jmesh@vy])
        
        Jy_0 = np.hstack([self.Jy[0, :, :self.n_dim], self.Jy[0, :, self.n_dim + 1:]])
        out = out.at[:self.Jy.shape[1], :out.shape[1]].add(Jy_0@vy[:Jy_0.shape[1]])

        Jy_N = self.Jy[-1, :, :-1]
        vy_N = vy[(self.Jy.shape[0] - 1) * (1 + self.Jy.shape[1]) - 1:(self.Jy.shape[0] - 1) * (1 + self.Jy.shape[1]) - 1 + Jy_N.shape[1]]
        out = out.at[(self.Jy.shape[0] - 1) * self.Jy.shape[1]:self.Jy.shape[0] * self.Jy.shape[1]].add(Jy_N@vy_N)

        def loop_body(carry, _):
            i, out = carry
            out_i = jax.lax.dynamic_slice(out, (i * self.Jy.shape[1], 0), (self.Jy.shape[1], out.shape[1]))
            vy_i = jax.lax.dynamic_slice(vy, (i * (1 + self.Jy.shape[1]) - 1, 0), (self.Jy.shape[2], out.shape[1]))
            out = jax.lax.dynamic_update_slice(out, out_i + self.Jy[i]@vy_i, (i * self.Jy.shape[1], 0))
            return (i + 1, out), _

        out = jax.lax.scan(loop_body, init=(1, out), xs=None, length=self.Jy.shape[0] - 2)[0][1]
        out = out.at[-self.n_dim - self.Jmesh.shape[0]:-self.Jmesh.shape[0]].add(self.Jbc_left@vy[:self.n_dim] + self.Jbc_right@vy[-self.n_dim:])

        if is_vector:
            out = out.ravel()

        return out

    @jax.jit
    def right_multiply_diag(self, D):

        Dy = D[self.n_par:self.n_par - self.Jk.shape[2]]
        Dy = permute_q_mesh(Dy, self.n_dim, self.Jy.shape[0], self.colloc_points_unshifted)
        Dk = np.concatenate([D[:self.n_par], D[self.n_par - self.Jk.shape[2]:]])

        Jy = self.Jy.at[0, :, :self.n_dim].multiply(Dy[:self.n_dim])
        Jy = Jy.at[0, :, self.n_dim + 1:].multiply(Dy[self.n_dim:Jy.shape[2] - 1])
        Jy = Jy.at[-1, :, :-1].multiply(Dy[-Jy.shape[2] + 1:])

        Jmesh = self.Jmesh * Dy

        def loop_body(carry, _):
            i, Jy = carry
            Dyi = jax.lax.dynamic_slice(Dy, (i * (1 + Jy.shape[1]) - 1,), (Jy.shape[2],))
            Jy = Jy.at[i].multiply(Dyi)
            return (i + 1, Jy), _

        Jy = jax.lax.scan(loop_body, init=(1, Jy), xs=None, length=self.Jy.shape[0] - 2)[0][1]
        Jk = np.reshape(np.vstack(self.Jk) * Dk, (self.Jk.shape))
        Jbc_left = self.Jbc_left * Dy[:self.n_dim]
        Jbc_right = self.Jbc_right * Dy[-self.n_dim:]
        
        return BVPMMJac(Jy, Jk, Jmesh, self.n_dim, self.n_par, Jbc_left, Jbc_right, self.colloc_points_unshifted)

    @jax.jit
    def lq_factor(self):

        R_c = np.zeros((self.Jy.shape[0], 2 * self.Jy.shape[1], self.Jy.shape[1]))
        Q_N_dim = self.Jy.shape[2] + self.Jy.shape[0] - 3
        Q_c = np.zeros(Q_N_dim * (Q_N_dim + 1) * (2 * Q_N_dim + 1) // 6 - (self.Jy.shape[2] - 1) * (self.Jy.shape[2]) * (2 * self.Jy.shape[2] - 1) // 6 + (self.Jy.shape[2] - 1)**2 + Q_N_dim**2)
        R_bc = np.zeros((self.n_dim + self.Jy.shape[1] * self.Jy.shape[0] + self.Jmesh.shape[0], self.n_dim + self.Jmesh.shape[0]))
        R_bc = R_bc.at[:self.n_dim, :self.n_dim].set(self.Jbc_left)
        R_bc = R_bc.at[-self.n_dim:, :self.n_dim].set(self.Jbc_right)
        R_bc = R_bc.at[:, self.n_dim:self.n_dim + self.Jmesh.shape[0]].set(self.Jmesh.T)

        Jy_0 = np.hstack([self.Jy[0, :, :self.n_dim], self.Jy[0, :, self.n_dim + 1:]])
        Q, R_0 = jax.scipy.linalg.qr(Jy_0.T)
        Q_c = Q_c.at[:Q.size].set(Q.ravel())
        Q_c_index = Q.size
        R_c = R_c.at[0, -R_0.shape[1]:].set(R_0[:R_0.shape[1]])
        R_bc = R_bc.at[:Q.shape[0]].set(Q.T@R_bc[:Q.shape[0]])

        for i in range(1, self.Jy.shape[0] - 1):

            s_i = Q[-self.n_dim - 1:].T@self.Jy[i, :, :self.n_dim + 1].T
            R_c = R_c.at[i, :s_i.shape[1]].set(s_i[:s_i.shape[1]])
            Q, R_i = jax.scipy.linalg.qr(np.vstack([s_i[-self.n_dim - i:], self.Jy[i, :, self.n_dim + 1:].T]))
            Q_c = Q_c.at[Q_c_index:Q_c_index + Q.size].set(Q.ravel())
            Q_c_index += Q.size
            R_c = R_c.at[i, -R_i.shape[1]:].set(R_i[:R_i.shape[1]])
            bc = R_bc[i * self.Jy.shape[1]:i * self.Jy.shape[1] + self.Jy.shape[2] + i - 1]
            R_bc = R_bc.at[i * self.Jy.shape[1]:i * self.Jy.shape[1] + self.Jy.shape[2] + i - 1].set(Q.T@bc)

        Jy_N = self.Jy[-1, :, :-1]
        s_N = Q[-self.n_dim - 1:].T@Jy_N[:, :self.n_dim + 1].T
        R_c = R_c.at[-1, :s_N.shape[1]].set(s_N[:s_N.shape[1]])
        Q, R_N = jax.scipy.linalg.qr(np.vstack([s_N[-self.n_dim - self.Jmesh.shape[0]:], Jy_N[:, self.n_dim + 1:].T]))
        Q_c = Q_c.at[Q_c_index:Q_c_index + Q.size].set(Q.ravel())
        R_c = R_c.at[-1, -R_N.shape[1]:].set(R_N[:R_N.shape[1]])
        bc = R_bc[(self.Jy.shape[0] - 1) * Jy_N.shape[0]:(self.Jy.shape[0] - 1) * Jy_N.shape[0] + Jy_N.shape[1] + self.Jy.shape[0] - 2]
        R_bc = R_bc.at[(self.Jy.shape[0] - 1) * Jy_N.shape[0]:(self.Jy.shape[0] - 1) * Jy_N.shape[0] + Jy_N.shape[1] + self.Jy.shape[0] - 2].set(Q.T@bc)
        Q_bc, R_bc_N = np.linalg.qr(R_bc[-self.Jmesh.shape[0] - self.n_dim:])
        R_bc = R_bc.at[-self.Jmesh.shape[0] - self.n_dim:].set(R_bc_N)

        return BVPMMJac_LQ(Q_c, Q_bc, R_c, R_bc, self.n_dim, self.n_par, self.colloc_points_unshifted)
    
    def _tree_flatten(self):
        children = (self.Jy, self.Jk, self.Jmesh, self.Jbc_left, self.Jbc_right, self.colloc_points_unshifted)
        aux_data = {"n_dim":self.n_dim, "n_par":self.n_par}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children[:3], Jbc_left=children[3], Jbc_right=children[4], colloc_points_unshifted=children[5], **aux_data)

class BVPMMJac_LQ:

    def __init__(self, Q_c, Q_bc, R_c, R_bc, n_dim, n_par, colloc_points_unshifted=gauss_points):

        self.Q_c = Q_c
        self.Q_bc = Q_bc
        self.R_c = R_c
        self.R_bc = R_bc
        self.n_dim = n_dim
        self.n_par = n_par
        self.colloc_points_unshifted = colloc_points_unshifted

    @jax.jit
    def solve_triangular_L(self, b):

        is_vector = len(b.shape) == 1
        if is_vector:
            b = np.expand_dims(b, 1)

        x = np.zeros((self.R_bc.shape[0], b.shape[1]))
        x_i = jax.scipy.linalg.solve_triangular(self.R_c[0, self.R_c.shape[2]:].T, b[:self.R_c.shape[2]], lower=True)
        x = x.at[:self.R_c.shape[1] // 2].set(x_i)

        def loop_body(carry, _):

            i, x = carry
            x_i_prev = jax.lax.dynamic_slice(x, ((i - 1) * self.R_c.shape[2], 0), (self.R_c.shape[2], x.shape[1]))
            b_i = jax.lax.dynamic_slice(b, (i * self.R_c.shape[2], 0), (self.R_c.shape[2], b.shape[1]))
            x_i = jax.scipy.linalg.solve_triangular(self.R_c[i, self.R_c.shape[2]:].T, b_i - self.R_c[i, :self.R_c.shape[2]].T@x_i_prev, lower=True)
            x = jax.lax.dynamic_update_slice(x, x_i, (i * self.R_c.shape[2], 0))
            return (i + 1, x), _

        x = jax.lax.scan(loop_body, init=(1, x), xs=None, length=self.R_c.shape[0] - 1)[0][1]
        xi = jax.scipy.linalg.solve_triangular(self.R_bc[-self.R_bc.shape[1]:].T, b[-self.R_bc.shape[1]:] - self.R_bc[:-self.R_bc.shape[1]].T@x[:-self.R_bc.shape[1]], lower=True)
        x = x.at[-self.R_bc.shape[1]:].set(xi)

        if is_vector:
            x = x.ravel()

        return x

    @jax.jit
    def solve_triangular_R(self, b):

        is_vector = len(b.shape) == 1
        if is_vector:
            b = np.expand_dims(b, 1)

        x = np.zeros((self.R_bc.shape[0], b.shape[1]))
        x_i = jax.scipy.linalg.solve_triangular(self.R_bc[-self.R_bc.shape[1]:], b[-self.R_bc.shape[1]:], lower=False)
        x = x.at[-self.R_bc.shape[1]:].set(x_i)
        x_i = jax.scipy.linalg.solve_triangular(self.R_c[-1, self.R_c.shape[2]:], b[-self.R_c.shape[2] - self.R_bc.shape[1]:-self.R_bc.shape[1]] - self.R_bc[-self.R_c.shape[2] - self.R_bc.shape[1]:-self.R_bc.shape[1]]@x[-self.R_bc.shape[1]:], lower=False)
        x = x.at[-self.R_c.shape[2] - self.R_bc.shape[1]:-self.R_bc.shape[1]].set(x_i)

        def loop_body(carry, _):

            i, x = carry
            x_i_prev = jax.lax.dynamic_slice(x, ((i + 1) * self.R_c.shape[2], 0), (self.R_c.shape[2], x.shape[1]))
            b_i = jax.lax.dynamic_slice(b, (i * self.R_c.shape[2], 0), (self.R_c.shape[2], b.shape[1]))
            self.R_bc_i = jax.lax.dynamic_slice(self.R_bc, (i * self.R_c.shape[2], 0), (self.R_c.shape[2], self.R_bc.shape[1]))
            b_i -= self.R_c[i + 1, :self.R_c.shape[2]]@x_i_prev + self.R_bc_i@x[-self.R_bc.shape[1]:]
            x_i = jax.scipy.linalg.solve_triangular(self.R_c[i, self.R_c.shape[2]:], b_i, lower=False)
            x = jax.lax.dynamic_update_slice(x, x_i, (i * self.R_c.shape[2], 0))
            return (i - 1, x), _

        x = jax.lax.scan(loop_body, init=(self.R_c.shape[0] - 2, x), xs=None, length=self.R_c.shape[0] - 1)[0][1]

        if is_vector:
            x = x.ravel()

        return x

    @jax.jit
    def Q_right_multiply(self, v):

        is_vector = len(v.shape) == 1
        if is_vector:
            v = np.expand_dims(v, 1)

        v = v.at[-self.Q_bc.shape[0]:].set(self.Q_bc@v[-self.Q_bc.shape[0]:])
        start = (self.R_c.shape[0] - 1) * self.R_c.shape[2]
        stop = start + (self.R_c.shape[2] + self.n_dim + 1) + (self.R_c.shape[0] - 1) - 1
        Q_N_dim = (self.R_c.shape[0] - 2) + (self.R_c.shape[2] + self.n_dim + 1)
        Q_N = self.Q_c[-Q_N_dim**2:].reshape((Q_N_dim, Q_N_dim))
        Q_c_index = Q_N_dim**2
        v = v.at[start:stop].set(Q_N@v[start:stop])

        for i in range(self.R_c.shape[0] - 2, 0, -1):

            start = i * self.R_c.shape[2]
            stop = start + (self.R_c.shape[2] + self.n_dim + 2) + (i - 1)
            Q_i_dim = (i - 1) + (self.R_c.shape[2] + self.n_dim + 2)
            Q_i = self.Q_c[-Q_c_index - Q_i_dim**2:-Q_c_index].reshape((Q_i_dim, Q_i_dim))
            Q_c_index = Q_c_index + Q_i_dim**2
            v = v.at[start:stop].set(Q_i@v[start:stop])

        stop = self.R_c.shape[2] + self.n_dim + 1
        Q_0_dim = self.R_c.shape[2] + self.n_dim + 1
        Q_0 = self.Q_c[-Q_c_index - Q_0_dim**2:-Q_c_index].reshape((Q_0_dim, Q_0_dim))
        v = v.at[:stop].set(Q_0@v[:stop])

        v = unpermute_q_mesh(v.T, self.n_dim, self.R_c.shape[0], self.colloc_points_unshifted).T

        if is_vector:
            v = v.ravel()

        return v

    def _tree_flatten(self):
        children = (self.Q_c, self.Q_bc, self.R_c, self.R_bc, self.colloc_points_unshifted)
        aux_data = {"n_dim":self.n_dim, "n_par":self.n_par}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children[:4], **aux_data, colloc_points_unshifted=children[4])

class BVPMMJac_1:

    def __init__(self, Jy, Jk, Jbc, n_dim, n_par, colloc_points_unshifted=gauss_points):
        self.Jy = Jy
        self.Jk = Jk
        self.Jbc = Jbc
        self.n_dim = n_dim
        self.n_par = n_par
        self.shape = ((Jy.shape[0] * Jy.shape[1] + Jbc.shape[0], Jy.shape[0] * Jy.shape[1] + n_dim + Jy.shape[0] + Jk.shape[2]))
        self.colloc_points_unshifted = colloc_points_unshifted

    @jax.jit
    def todense(self):

        Jk = np.vstack(self.Jk)
        Jy_dense = np.zeros((self.Jy.shape[0] * self.Jy.shape[1], self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim))
        Jm_dense = np.zeros((self.Jy.shape[0] * self.Jy.shape[1], self.Jy.shape[0]))

        def loop_body(carry, _):
            i, Jy_dense, Jm_dense = carry
            Jy_dense = jax.lax.dynamic_update_slice(Jy_dense, self.Jy[i, :, :self.n_dim], (i * self.Jy.shape[1], i * self.Jy.shape[1]))
            Jy_dense = jax.lax.dynamic_update_slice(Jy_dense, self.Jy[i, :, self.n_dim + 1:], (i * self.Jy.shape[1], i * self.Jy.shape[1] + self.n_dim))
            Jm_dense = jax.lax.dynamic_update_slice(Jm_dense, self.Jy[i, :, self.n_dim:self.n_dim + 1], (i * self.Jy.shape[1], i))
            return (i + 1, Jy_dense, Jm_dense), _

        Jy_mesh_dense = np.hstack(jax.lax.scan(loop_body, init=(0, Jy_dense, Jm_dense), xs=None, length=self.Jy.shape[0])[0][1:])
        return np.vstack([np.hstack([Jk, Jy_mesh_dense]), np.pad(unpermute_q_mesh_1(self.Jbc, self.n_dim, self.Jy.shape[0], self.colloc_points_unshifted), ((0, 0), (self.n_par, 0)))])

    @jax.jit
    def left_multiply(self, v):
   
        is_vector = len(v.shape) == 1
        if is_vector:
            v = np.expand_dims(v, 0)

        out = np.zeros((v.shape[0], self.shape[1]))
        Jk = np.vstack(self.Jk)
        out = out.at[:, :self.n_par].add(v[:, :Jk.shape[0]]@Jk)
        out = out.at[:, self.n_par:].add(v[:, -self.Jbc.shape[0]:]@self.Jbc)
        
        def loop_body(carry, _):
            i, out = carry
            out_i = jax.lax.dynamic_slice(out, (0, self.n_par + i * (1 + self.Jy.shape[1])), (out.shape[0], self.Jy.shape[2]))
            v_i = jax.lax.dynamic_slice(v, (0, i * self.Jy.shape[1]), (v.shape[0], self.Jy.shape[1]))
            out = jax.lax.dynamic_update_slice(out, out_i + v_i@self.Jy[i], (0, self.n_par + i * (1 + self.Jy.shape[1])))
            return (i + 1, out), _

        out = jax.lax.scan(loop_body, init=(0, out), xs=None, length=self.Jy.shape[0])[0][1]
        out = out.at[:, self.n_par:].set(unpermute_q_mesh_1(out[:, self.n_par:], self.n_dim, self.Jy.shape[0], self.colloc_points_unshifted))

        if is_vector:
            out = out.ravel()

        return out
   
    @jax.jit
    def right_multiply(self, v):
       
        is_vector = len(v.shape) == 1
        if is_vector:
            v = np.expand_dims(v, 1)

        vy = v[self.n_par:]
        vy = permute_q_mesh_1(vy.T, self.n_dim, self.Jy.shape[0], self.colloc_points_unshifted).T
        out = np.vstack([np.vstack(self.Jk)@v[:self.n_par], self.Jbc@vy])
        
        def loop_body(carry, _):
            i, out = carry
            out_i = jax.lax.dynamic_slice(out, (i * self.Jy.shape[1], 0), (self.Jy.shape[1], out.shape[1]))
            vy_i = jax.lax.dynamic_slice(vy, (i * (1 + self.Jy.shape[1]), 0), (self.Jy.shape[2], out.shape[1]))
            out = jax.lax.dynamic_update_slice(out, out_i + self.Jy[i]@vy_i, (i * self.Jy.shape[1], 0))
            return (i + 1, out), _

        out = jax.lax.scan(loop_body, init=(0, out), xs=None, length=self.Jy.shape[0])[0][1]

        if is_vector:
            out = out.ravel()

        return out

    @jax.jit
    def right_multiply_diag(self, D):

        Dy = D[self.n_par:]
        Dy = permute_q_mesh_1(Dy, self.n_dim, self.Jy.shape[0], self.colloc_points_unshifted)
        Dk = D[:self.n_par]
        Jbc = self.Jbc * Dy

        def loop_body(i, Jy_i):
            Dy_i = jax.lax.dynamic_slice(Dy, (i * (1 + Jy_i.shape[0]),), (Jy_i.shape[1],))
            return i + 1, Jy_i * Dy_i

        Jy = jax.lax.scan(loop_body, init=0, xs=self.Jy)[1]
        Jk = np.reshape(np.vstack(self.Jk) * Dk, (self.Jk.shape))
        
        return BVPMMJac_1(Jy, Jk, Jbc, self.n_dim, self.n_par, self.colloc_points_unshifted)

    @jax.jit
    def lq_factor(self):

        R_c = np.zeros((self.Jy.shape[0], 2 * self.Jy.shape[1], self.Jy.shape[1]))
        Q_N_dim = self.Jy.shape[2] + self.Jy.shape[0] - 1
        Q_c = np.zeros(Q_N_dim * (Q_N_dim + 1) * (2 * Q_N_dim + 1) // 6 - (self.Jy.shape[2] - 1) * (self.Jy.shape[2]) * (2 * self.Jy.shape[2] - 1) // 6 + (self.Jy.shape[2] - 1)**2 + Q_N_dim**2)
        R_bc = np.zeros((self.n_dim + self.Jy.shape[1] * self.Jy.shape[0] + self.Jmesh.shape[0], self.n_dim + self.Jmesh.shape[0]))
        R_bc = R_bc.at[:self.n_dim, :self.n_dim].set(self.Jbc_left)
        R_bc = R_bc.at[-self.n_dim:, :self.n_dim].set(self.Jbc_right)
        R_bc = R_bc.at[:, self.n_dim:self.n_dim + self.Jmesh.shape[0]].set(self.Jmesh.T)

        Jy_0 = np.hstack([self.Jy[0, :, :self.n_dim], self.Jy[0, :, self.n_dim + 1:]])
        Q, R_0 = jax.scipy.linalg.qr(Jy_0.T)
        Q_c = Q_c.at[:Q.size].set(Q.ravel())
        Q_c_index = Q.size
        R_c = R_c.at[0, -R_0.shape[1]:].set(R_0[:R_0.shape[1]])
        R_bc = R_bc.at[:Q.shape[0]].set(Q.T@R_bc[:Q.shape[0]])

        for i in range(1, self.Jy.shape[0] - 1):

            s_i = Q[-self.n_dim - 1:].T@self.Jy[i, :, :self.n_dim + 1].T
            R_c = R_c.at[i, :s_i.shape[1]].set(s_i[:s_i.shape[1]])
            Q, R_i = jax.scipy.linalg.qr(np.vstack([s_i[-self.n_dim - i:], self.Jy[i, :, self.n_dim + 1:].T]))
            Q_c = Q_c.at[Q_c_index:Q_c_index + Q.size].set(Q.ravel())
            Q_c_index += Q.size
            R_c = R_c.at[i, -R_i.shape[1]:].set(R_i[:R_i.shape[1]])
            bc = R_bc[i * self.Jy.shape[1]:i * self.Jy.shape[1] + self.Jy.shape[2] + i - 1]
            R_bc = R_bc.at[i * self.Jy.shape[1]:i * self.Jy.shape[1] + self.Jy.shape[2] + i - 1].set(Q.T@bc)

        Jy_N = self.Jy[-1, :, :-1]
        s_N = Q[-self.n_dim - 1:].T@Jy_N[:, :self.n_dim + 1].T
        R_c = R_c.at[-1, :s_N.shape[1]].set(s_N[:s_N.shape[1]])
        Q, R_N = jax.scipy.linalg.qr(np.vstack([s_N[-self.n_dim - self.Jmesh.shape[0]:], Jy_N[:, self.n_dim + 1:].T]))
        Q_c = Q_c.at[Q_c_index:Q_c_index + Q.size].set(Q.ravel())
        R_c = R_c.at[-1, -R_N.shape[1]:].set(R_N[:R_N.shape[1]])
        bc = R_bc[(self.Jy.shape[0] - 1) * Jy_N.shape[0]:(self.Jy.shape[0] - 1) * Jy_N.shape[0] + Jy_N.shape[1] + self.Jy.shape[0] - 2]
        R_bc = R_bc.at[(self.Jy.shape[0] - 1) * Jy_N.shape[0]:(self.Jy.shape[0] - 1) * Jy_N.shape[0] + Jy_N.shape[1] + self.Jy.shape[0] - 2].set(Q.T@bc)
        Q_bc, R_bc_N = np.linalg.qr(R_bc[-self.Jmesh.shape[0] - self.n_dim:])
        R_bc = R_bc.at[-self.Jmesh.shape[0] - self.n_dim:].set(R_bc_N)

        return BVPMMJac_LQ(Q_c, Q_bc, R_c, R_bc, self.n_dim, self.n_par, self.colloc_points_unshifted)
    
    def _tree_flatten(self):
        children = (self.Jy, self.Jk, self.Jbc, self.colloc_points_unshifted)
        aux_data = {"n_dim":self.n_dim, "n_par":self.n_par}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children[:3], colloc_points_unshifted=children[3], **aux_data)

class BVPMMJac_LQ_1:

    def __init__(self, Qc, Qbc, Rc, Rbc, n_dim, n_par, colloc_points_unshifted=gauss_points):

        self.Q_c = Q_c
        self.Q_bc = Q_bc
        self.R_c = R_c
        self.R_bc = R_bc
        self.n_dim = n_dim
        self.n_par = n_par
        self.colloc_points_unshifted = colloc_points_unshifted

    @jax.jit
    def solve_triangular_L(self, b):

        is_vector = len(b.shape) == 1
        if is_vector:
            b = np.expand_dims(b, 1)

        x = np.zeros((self.R_bc.shape[0], b.shape[1]))
        x_i = jax.scipy.linalg.solve_triangular(self.R_c[0, self.R_c.shape[2]:].T, b[:self.R_c.shape[2]], lower=True)
        x = x.at[:self.R_c.shape[1] // 2].set(x_i)

        def loop_body(carry, _):

            i, x = carry
            x_i_prev = jax.lax.dynamic_slice(x, ((i - 1) * self.R_c.shape[2], 0), (self.R_c.shape[2], x.shape[1]))
            b_i = jax.lax.dynamic_slice(b, (i * self.R_c.shape[2], 0), (self.R_c.shape[2], b.shape[1]))
            x_i = jax.scipy.linalg.solve_triangular(self.R_c[i, self.R_c.shape[2]:].T, b_i - self.R_c[i, :self.R_c.shape[2]].T@x_i_prev, lower=True)
            x = jax.lax.dynamic_update_slice(x, x_i, (i * self.R_c.shape[2], 0))
            return (i + 1, x), _

        x = jax.lax.scan(loop_body, init=(1, x), xs=None, length=self.R_c.shape[0] - 1)[0][1]
        xi = jax.scipy.linalg.solve_triangular(self.R_bc[-self.R_bc.shape[1]:].T, b[-self.R_bc.shape[1]:] - self.R_bc[:-self.R_bc.shape[1]].T@x[:-self.R_bc.shape[1]], lower=True)
        x = x.at[-self.R_bc.shape[1]:].set(xi)

        if is_vector:
            x = x.ravel()

        return x

    @jax.jit
    def solve_triangular_R(self, b):

        is_vector = len(b.shape) == 1
        if is_vector:
            b = np.expand_dims(b, 1)

        x = np.zeros((self.R_bc.shape[0], b.shape[1]))
        x_i = jax.scipy.linalg.solve_triangular(self.R_bc[-self.R_bc.shape[1]:], b[-self.R_bc.shape[1]:], lower=False)
        x = x.at[-self.R_bc.shape[1]:].set(x_i)
        x_i = jax.scipy.linalg.solve_triangular(self.R_c[-1, self.R_c.shape[2]:], b[-self.R_c.shape[2] - self.R_bc.shape[1]:-self.R_bc.shape[1]] - self.R_bc[-self.R_c.shape[2] - self.R_bc.shape[1]:-self.R_bc.shape[1]]@x[-self.R_bc.shape[1]:], lower=False)
        x = x.at[-self.R_c.shape[2] - self.R_bc.shape[1]:-self.R_bc.shape[1]].set(x_i)

        def loop_body(carry, _):

            i, x = carry
            x_i_prev = jax.lax.dynamic_slice(x, ((i + 1) * self.R_c.shape[2], 0), (self.R_c.shape[2], x.shape[1]))
            b_i = jax.lax.dynamic_slice(b, (i * self.R_c.shape[2], 0), (self.R_c.shape[2], b.shape[1]))
            self.R_bc_i = jax.lax.dynamic_slice(self.R_bc, (i * self.R_c.shape[2], 0), (self.R_c.shape[2], self.R_bc.shape[1]))
            b_i -= self.R_c[i + 1, :self.R_c.shape[2]]@x_i_prev + self.R_bc_i@x[-self.R_bc.shape[1]:]
            x_i = jax.scipy.linalg.solve_triangular(self.R_c[i, self.R_c.shape[2]:], b_i, lower=False)
            x = jax.lax.dynamic_update_slice(x, x_i, (i * self.R_c.shape[2], 0))
            return (i - 1, x), _

        x = jax.lax.scan(loop_body, init=(self.R_c.shape[0] - 2, x), xs=None, length=self.R_c.shape[0] - 1)[0][1]

        if is_vector:
            x = x.ravel()

        return x

    @jax.jit
    def Q_right_multiply(self, v):

        is_vector = len(v.shape) == 1
        if is_vector:
            v = np.expand_dims(v, 1)

        v = v.at[-self.Q_bc.shape[0]:].set(self.Q_bc@v[-self.Q_bc.shape[0]:])
        start = (self.R_c.shape[0] - 1) * self.R_c.shape[2]
        stop = start + (self.R_c.shape[2] + self.n_dim + 1) + (self.R_c.shape[0] - 1) - 1
        Q_N_dim = (self.R_c.shape[0] - 2) + (self.R_c.shape[2] + self.n_dim + 1)
        Q_N = self.Q_c[-Q_N_dim**2:].reshape((Q_N_dim, Q_N_dim))
        Q_c_index = Q_N_dim**2
        v = v.at[start:stop].set(Q_N@v[start:stop])

        for i in range(self.R_c.shape[0] - 2, 0, -1):

            start = i * self.R_c.shape[2]
            stop = start + (self.R_c.shape[2] + self.n_dim + 2) + (i - 1)
            Q_i_dim = (i - 1) + (self.R_c.shape[2] + self.n_dim + 2)
            Q_i = self.Q_c[-Q_c_index - Q_i_dim**2:-Q_c_index].reshape((Q_i_dim, Q_i_dim))
            Q_c_index = Q_c_index + Q_i_dim**2
            v = v.at[start:stop].set(Q_i@v[start:stop])

        stop = self.R_c.shape[2] + self.n_dim + 1
        Q_0_dim = self.R_c.shape[2] + self.n_dim + 1
        Q_0 = self.Q_c[-Q_c_index - Q_0_dim**2:-Q_c_index].reshape((Q_0_dim, Q_0_dim))
        v = v.at[:stop].set(Q_0@v[:stop])

        v = unpermute_q_mesh(v.T, self.n_dim, self.R_c.shape[0], self.colloc_points_unshifted).T

        if is_vector:
            v = v.ravel()

        return v

    def _tree_flatten(self):
        children = (self.Qc, self.Qbc, self.Rc, self.Rbc, self.colloc_points_unshifted)
        aux_data = {"n_dim":self.n_dim, "n_par":self.n_par}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children[:4], **aux_data, colloc_points_unshifted=children[4])

jax.tree_util.register_pytree_node(BVPJac, BVPJac._tree_flatten, BVPJac._tree_unflatten)
jax.tree_util.register_pytree_node(BVPJac_LQ, BVPJac_LQ._tree_flatten, BVPJac_LQ._tree_unflatten)
jax.tree_util.register_pytree_node(BVPMMJac, BVPMMJac._tree_flatten, BVPMMJac._tree_unflatten)
jax.tree_util.register_pytree_node(BVPMMJac_1, BVPMMJac_1._tree_flatten, BVPMMJac_1._tree_unflatten)
jax.tree_util.register_pytree_node(BVPMMJac_LQ, BVPMMJac_LQ._tree_flatten, BVPMMJac_LQ._tree_unflatten)
jax.tree_util.register_pytree_node(BVPMMJac_LQ_1, BVPMMJac_LQ_1._tree_flatten, BVPMMJac_LQ_1._tree_unflatten)
