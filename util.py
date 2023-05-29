import jax
import jax.numpy as np
import numpy
from functools import partial
jax.config.update("jax_enable_x64", True)

gauss_points = np.array([-np.sqrt(3 / 7 + (2 / 7) * np.sqrt(6 / 5)), -np.sqrt(3 / 7 - (2 / 7) * np.sqrt(6 / 5)), np.sqrt(3 / 7 - (2 / 7) * np.sqrt(6 / 5)), np.sqrt(3 / 7 + (2 / 7) * np.sqrt(6 / 5))])
gauss_weights = np.array([18 + np.sqrt(30), 18 - np.sqrt(30), 18 - np.sqrt(30), 18 + np.sqrt(30)]) / 36

@jax.jit
def smooth_max(x, smooth_max_temperature=1):
    return np.sum(x * np.exp(smooth_max_temperature * x)) / np.sum(np.exp(smooth_max_temperature * x))

@jax.jit
def fill_mesh(t, gauss_points):
    return np.concatenate([np.ravel(np.expand_dims(t[:-1], 1) + np.expand_dims(t[1:] - t[:-1], 1) * np.linspace(0, 1, gauss_points.size + 1)[:-1]), np.array([1])])

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
def smooth_mesh_density(mesh_density, n_smooth=4):
    
    def loop_inner(carry, _):
        i, mesh_density = carry
        mesh_density = mesh_density.at[i].set(mesh_density[i - 1] / 4 + mesh_density[i] / 2 + mesh_density[i + 1] / 4)
        return (i + 1, mesh_density), _
    
    def loop_outer(carry, _):
        i, mesh_density = carry
        mesh_density = mesh_density.at[0].set((mesh_density[0] + mesh_density[1]) / 2)
        mesh_density = jax.lax.scan(loop_inner, init=(1, mesh_density), xs=None, length=mesh_density.size - 2)[0][1]
        mesh_density = mesh_density.at[-1].set((mesh_density[-1] + mesh_density[-2]) / 2)
        return (i + 1, mesh_density), _
    
    mesh_density = jax.lax.scan(loop_outer, init=(0, mesh_density), xs=None, length=n_smooth)[0][1]
    
    return mesh_density

@partial(jax.jit, static_argnums=3)
def recompute_mesh(y, mesh_old, gauss_points, n_smooth=4):
    
    def loop_body(i, _):
        meshi = np.linspace(*jax.lax.dynamic_slice(mesh_old, (i,), (2,)), gauss_points.size + 1)
        yi = jax.lax.dynamic_slice(y, (0, i * gauss_points.size), (y.shape[0], gauss_points.size + 1))
        return i + 1, gauss_points.size * divided_difference(meshi, yi)[np.diag_indices(gauss_points.size + 1)][-1]
    
    _, deriv = jax.lax.scan(loop_body, init=0, xs=None, length=mesh_old.size - 1)
    midpoints = (mesh_old[1:] + mesh_old[:-1]) / 2
    deriv = np.pad((deriv[1:] - deriv[:-1]).T / (midpoints[1:] - midpoints[:-1]), ((0, 0), (1, 1)), mode="edge")
    a = np.maximum(1, np.trapz(np.sum(deriv**2, axis=0)**(1 / (1 + 2 * (gauss_points.size + 1))), x=mesh_old)**(1 + 2 * (gauss_points.size + 1)))
    mesh_density = (1 + np.sum(deriv**2, axis=0) / a)**(1 / (1 + 2 * (gauss_points.size + 1)))
    mesh_density = smooth_mesh_density(mesh_density, n_smooth)
    mesh_mass = np.pad(np.cumsum((mesh_density[1:] + mesh_density[:-1]) * (mesh_old[1:] - mesh_old[:-1])) / 2, (1, 0))
    mesh_new = np.interp(np.linspace(0, mesh_mass[-1], mesh_old.size), mesh_mass, mesh_old)
    
    return mesh_new, mesh_density

@jax.jit
def interpolate(y, mesh_old, mesh_new, gauss_points):
    
    def loop1(i, _):
        meshi = np.linspace(*jax.lax.dynamic_slice(mesh_old, (i,), (2,)), gauss_points.size + 1)
        yi = jax.lax.dynamic_slice(y, (0 * i, i * gauss_points.size,), (y.shape[0], gauss_points.size + 1))
        return i + 1, divided_difference(meshi, yi)
        
    dd = jax.lax.scan(loop1, init=0, xs=None, length=mesh_old.size - 1)[1]
        
    def loop2(_, t):
        i = np.searchsorted(mesh_old, t) - 1
        meshi = np.linspace(*jax.lax.dynamic_slice(mesh_old, (i,), (2,)), gauss_points.size + 1)
        yi = jax.lax.dynamic_slice(y, (0 * i, i * gauss_points.size,), (y.shape[0], gauss_points.size + 1))
        return _, newton_polynomial(t, meshi, yi, dd[i])
    
    _, y_new = jax.lax.scan(loop2, init=None, xs=fill_mesh(mesh_new, gauss_points)[1:-1])
    y_new = np.hstack([y[:, :1], y_new.T, y[:, -1:]])
    return y_new

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
    
        if len(v.shape) == 1:
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

        if out.shape[0] == 1:
            out = out.ravel()

        return out
   
    @jax.jit
    def right_multiply(self, v):
        
        if len(v.shape) == 1:
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

        if out.shape[1] == 1:
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

        if len(b.shape) == 1:
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

        if x.shape[1] == 1:
            x = x.ravel()

        return x

    @jax.jit
    def solve_triangular_R(self, b):

        if len(b.shape) == 1:
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

        if x.shape[1] == 1:
            x = x.ravel()

        return x

    def _tree_flatten(self):
        children = (self.Q_c, self.Q_bc, self.R_c, self.R_bc)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

jax.tree_util.register_pytree_node(BVPJac, BVPJac._tree_flatten, BVPJac._tree_unflatten)
jax.tree_util.register_pytree_node(BVPJac_LQ, BVPJac_LQ._tree_flatten, BVPJac_LQ._tree_unflatten)
