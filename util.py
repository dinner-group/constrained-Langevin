import jax
import jax.numpy as np
import numpy
from functools import partial
jax.config.update("jax_enable_x64", True)

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

    def __init__(self, Jy, Jk, n_dim, n_par, n_mesh_intervals):
        self.Jy = Jy
        self.Jk = Jk
        self.n_dim = n_dim
        self.n_par = n_par
        self.n_mesh_intervals = n_mesh_intervals

    @jax.jit
    def todense(self):

        Jk = np.pad(np.vstack(self.Jk), ((0, self.n_dim), (0, 0)))
        Jy_dense = np.zeros((self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim, self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim))
        Jy_dense = Jy_dense.at[-self.n_dim:, :self.n_dim].set(-np.identity(self.n_dim))
        Jy_dense = Jy_dense.at[-self.n_dim:, -self.n_dim:].set(np.identity(self.n_dim))

        def loop_body(carry, _):
            i, Jy_dense = carry
            Jy_dense = jax.lax.dynamic_update_slice(Jy_dense, self.Jy[i], (i * self.Jy.shape[1], i * self.Jy.shape[1]))
            return (i + 1, Jy_dense), _

        Jy_dense = jax.lax.scan(loop_body, init=(0, Jy_dense), xs=None, length=self.Jy.shape[0])[0][1]
        return np.hstack([Jk[:, :self.n_par], Jy_dense, Jk[:, self.n_par:]])

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
        J1J2T = J1J2T.at[(np.diag_indices(J1.n_dim)[0] + J1J2T.shape[0] - 2, np.diag_indices(J1.n_dim)[0] + J1J2T.shape[0] - 2)].set(2)
        J1J2T = J1J2T.at[:Jy1Jy2T.shape[1], -J1.n_dim:].set(-J1.Jy[0, :, :J1.n_dim])
        J1J2T = J1J2T.at[-J1.n_dim:, :Jy1Jy2T.shape[1]].set(-J2.Jy[0, :, :J1.n_dim].T)
        J1J2T = J1J2T.at[-Jy1Jy2T.shape[1] - J1.n_dim:-J1.n_dim, -J1.n_dim:].set(J1.Jy[-1, :, -J1.n_dim:])
        J1J2T = J1J2T.at[-J1.n_dim:, -Jy1Jy2T.shape[1] - J1.n_dim:-J1.n_dim].set(J2.Jy[-1, :, -J2.n_dim:].T)
        
        return J1J2T

    def _tree_flatten(self):
        children = (self.Jy, self.Jk)
        aux_data = {"n_dim":self.n_dim, "n_par":self.n_par, "n_mesh_intervals":self.n_mesh_intervals}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

jax.tree_util.register_pytree_node(BVPJac, BVPJac._tree_flatten, BVPJac._tree_unflatten)
