import jax
import jax.numpy as np
import numpy
from functools import partial
jax.config.update("jax_enable_x64", True)

gauss_points = np.array([-np.sqrt(3 / 7 + (2 / 7) * np.sqrt(6 / 5)), -np.sqrt(3 / 7 - (2 / 7) * np.sqrt(6 / 5)), np.sqrt(3 / 7 - (2 / 7) * np.sqrt(6 / 5)), np.sqrt(3 / 7 + (2 / 7) * np.sqrt(6 / 5))])
gauss_weights = np.array([18 + np.sqrt(30), 18 - np.sqrt(30), 18 - np.sqrt(30), 18 + np.sqrt(30)]) / 36

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
        dd = colloc.divided_difference(node_t, node_y)
        
    return np.sum(np.cumprod(np.roll(t - node_t, 1).at[0].set(1)) * dd[np.diag_indices(node_t.size)].T, axis=1)

class BVPJac:

    def __init__(self, Jy, Jk, n_dim, n_par, n_mesh_intervals, Jbc_left=None, Jbc_right=None):
        self.Jy = Jy
        self.Jk = Jk
        self.n_dim = n_dim
        self.n_par = n_par
        self.n_mesh_intervals = n_mesh_intervals

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
    def vjp(self, v):
        
        out = np.zeros(self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim + self.Jk.shape[2])
        out = out.at[:self.n_par].set(v[:-self.n_dim]@np.vstack(self.Jk[:, :, :self.n_par]))
        out = out.at[self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim:].set(v[:-self.n_dim]@np.vstack(self.Jk[:, :, self.n_par:]))
        out = out.at[self.n_par:self.n_par + self.n_dim].set(self.Jbc_left@v[-self.n_dim:])
        out = out.at[self.n_par + self.Jy.shape[0] * self.Jy.shape[1]:self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim].set(self.Jbc_right@v[-self.n_dim:])
        
        def loop_body(carry, _):
            i, out = carry
            outi = jax.lax.dynamic_slice(out, (self.n_par + i * self.Jy.shape[1],), (self.Jy.shape[2],))
            vi = jax.lax.dynamic_slice(v, (i * self.Jy.shape[1],), (self.Jy.shape[1],))
            out = jax.lax.dynamic_update_slice(out, outi + vi@self.Jy[i], (self.n_par + i * self.Jy.shape[1],))
            return (i + 1, out), _
        
        return jax.lax.scan(loop_body, init=(0, out), xs=None, length=self.Jy.shape[0])[0][1]
   
    @jax.jit
    def jvp(self, v):
        
        vy = v[self.n_par:self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim]
        vk = np.concatenate([v[:self.n_par], v[self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim:]])
        out = np.pad(np.vstack(self.Jk)@vk, (0, self.n_dim))

        def loop_body(carry, _):
            i, out = carry
            outi = jax.lax.dynamic_slice(out, (i * self.Jy.shape[1],), (self.Jy.shape[1],))
            vyi = jax.lax.dynamic_slice(vy, (i * self.Jy.shape[1],), (self.Jy.shape[2],))
            out = jax.lax.dynamic_update_slice(out, outi + self.Jy[i]@vyi, (i * self.Jy.shape[1],))
            return (i + 1, out), _

        out = jax.lax.scan(loop_body, init=(0, out), xs=None, length=self.Jy.shape[0])[0][1]
        out = out.at[-self.n_dim:].add(self.Jbc_left@vy[:self.n_dim] + self.Jbc_right@vy[-self.n_dim:])
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
        
        return BVPJac(Jy, Jk, self.n_dim, self.n_par, self.n_mesh_intervals, Jbc_left, Jbc_right)

    @jax.jit
    def multiply_transpose(J1, J2):
        
        def loop1(i, _):
            return i + 1, np.hstack([J1.Jy[i, :, :J1.n_dim]@J2.Jy[i - 1, :, -J1.n_dim:].T, J1.Jy[i]@J2.Jy[i].T, J1.Jy[i, :, -J1.n_dim:]@J2.Jy[i + 1, :, :J1.n_dim].T])
        
        Jy1Jy2T = jax.lax.scan(loop1, init=1, xs=None, length=J1.n_mesh_intervals - 2)[1]
        Jy1Jy2T = np.vstack([[np.pad(np.hstack([J1.Jy[0]@J2.Jy[0].T, J1.Jy[0, :, -J1.n_dim:]@J2.Jy[1, :, :J1.n_dim].T]), ((0, 0), (J1.Jy.shape[1], 0)))], 
                             Jy1Jy2T, 
                             [np.pad(np.hstack([J1.Jy[-1, :, :J1.n_dim]@J2.Jy[-2, :, -J1.n_dim:].T, J1.Jy[-1]@J2.Jy[-1].T]), ((0, 0), (0, J1.Jy.shape[1])))]])
        
        J1J2T = np.pad(np.vstack(J1.Jk)@np.vstack(J2.Jk).T, ((0, 2), (0, 2)))
        
        def loop2(carry, _):
            i, J1J2T = carry
            J1J2Ti = jax.lax.dynamic_slice(J1J2T, (i * Jy1Jy2T.shape[1], (i - 1) * Jy1Jy2T.shape[1]), (Jy1Jy2T.shape[1], 3 * Jy1Jy2T.shape[1]))
            J1J2T = jax.lax.dynamic_update_slice(J1J2T, J1J2Ti + Jy1Jy2T[i, :], (i * Jy1Jy2T.shape[1], (i - 1) * Jy1Jy2T.shape[1]))
            return (i + 1, J1J2T), _
            
        J1J2T = jax.lax.scan(loop2, init=(1, J1J2T), xs=None, length=J1.n_mesh_intervals - 2)[0][1]
        J1J2T = J1J2T.at[:Jy1Jy2T.shape[1], :2 * Jy1Jy2T.shape[1]].add(Jy1Jy2T[0, :, Jy1Jy2T.shape[1]:])
        J1J2T = J1J2T.at[(J1.n_mesh_intervals - 1) * Jy1Jy2T.shape[1]:J1.n_mesh_intervals * Jy1Jy2T.shape[1], (J1.n_mesh_intervals - 2) * Jy1Jy2T.shape[1]:J1.n_mesh_intervals * Jy1Jy2T.shape[1]].add(Jy1Jy2T[-1, :, :2 * Jy1Jy2T.shape[1]])
        J1J2T = J1J2T.at[-J1.n_dim:, -J1.n_dim:].set(J1.Jbc_left@J2.Jbc_left + J1.Jbc_right@J2.Jbc_right)
        J1J2T = J1J2T.at[:Jy1Jy2T.shape[1], -J1.n_dim:].set(J1.Jy[0, :, :J1.n_dim]@J2.Jbc_left)
        J1J2T = J1J2T.at[-J1.n_dim:, :Jy1Jy2T.shape[1]].set(J1.Jbc_left@J2.Jy[0, :, :J1.n_dim].T)
        J1J2T = J1J2T.at[-Jy1Jy2T.shape[1] - J1.n_dim:-J1.n_dim, -J1.n_dim:].set(J1.Jy[-1, :, -J1.n_dim:]@J2.Jbc_right)
        J1J2T = J1J2T.at[-J1.n_dim:, -Jy1Jy2T.shape[1] - J1.n_dim:-J1.n_dim].set(J1.Jbc_right@J2.Jy[-1, :, -J2.n_dim:].T)
        
        return J1J2T

    def _tree_flatten(self):
        children = (self.Jy, self.Jk, self.Jbc_left, self.Jbc_right)
        aux_data = {"n_dim":self.n_dim, "n_par":self.n_par, "n_mesh_intervals":self.n_mesh_intervals}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children[:2], Jbc_left=children[2], Jbc_right=children[3], **aux_data)

jax.tree_util.register_pytree_node(BVPJac, BVPJac._tree_flatten, BVPJac._tree_unflatten)
