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

@partial(jax.jit, static_argnums=(1, 2))
def bvp_jac_mul(J1, J2, n_dim, n_mesh_intervals):
    
    Jy1, Jk1 = J1
    Jy2, Jk2 = J2
    
    def loop1(i, _):
        return i + 1, np.hstack([Jy1[i, :, :n_dim]@Jy2[i - 1, :, -n_dim:].T, Jy1[i]@Jy2[i].T, Jy1[i, :, -n_dim:]@Jy2[i + 1, :, :n_dim].T])
    
    Jy1Jy2T = jax.lax.scan(loop1, init=1, xs=None, length=n_mesh_intervals - 1)[1]
    Jy1Jy2T = np.vstack([[np.pad(np.hstack([Jy1[0]@Jy2[0].T, Jy1[0, :, -n_dim:]@Jy2[1, :, :n_dim].T]), ((0, 0), (Jy1.shape[1], 0)))], 
                         Jy1Jy2T, 
                         [np.pad(Jy1[-1]@Jy2[-1].T, ((0, 0), (0, 2 * Jy1.shape[1])))]])
    
    J1J2T = np.pad(np.vstack(Jk1)@np.vstack(Jk2).T, ((0, 2), (0, 2)))
    
    def loop2(carry, _):
        i, J1J2T = carry
        J1J2Ti = jax.lax.dynamic_slice(J1J2T, (i * Jy1Jy2T.shape[1], (i - 1) * Jy1Jy2T.shape[1]), (Jy1Jy2T.shape[1], 3 * Jy1Jy2T.shape[1]))
        J1J2T = jax.lax.dynamic_update_slice(J1J2T, J1J2Ti + Jy1Jy2T[i, :], (i * Jy1Jy2T.shape[1], (i - 1) * Jy1Jy2T.shape[1]))
        return (i + 1, J1J2T), _
        
    J1J2T = jax.lax.scan(loop2, init=(1, J1J2T), xs=None, length=n_mesh_intervals - 1)[0][1]
    J1J2T = J1J2T.at[:Jy1Jy2T.shape[1], :2 * Jy1Jy2T.shape[1]].set(Jy1Jy2T[0, :, Jy1Jy2T.shape[1]:])
    J1J2T = J1J2T.at[(n_mesh_intervals - 1) * Jy1Jy2T.shape[1]:n_mesh_intervals * Jy1Jy2T.shape[1], (n_mesh_intervals - 1) * Jy1Jy2T.shape[1]:n_mesh_intervals * Jy1Jy2T.shape[1]].set(Jy1Jy2T[-1, :, :Jy1Jy2T.shape[1]])
    J1J2T = J1J2T.at[(np.diag_indices(n_dim)[0] + J1J2T.shape[0] - 2, np.diag_indices(n_dim)[0] + J1J2T.shape[0] - 2)].set(2)
    J1J2T = J1J2T.at[:Jy1Jy2T.shape[1], -n_dim:].set(-Jy1[0, :, :n_dim])
    J1J2T = J1J2T.at[-n_dim:, :Jy1Jy2T.shape[1]].set(-Jy2[0, :, :n_dim].T)
    J1J2T = J1J2T.at[-Jy1Jy2T.shape[1] - n_dim:-n_dim, -n_dim:].set(Jy1[-1, :, Jy1Jy2T.shape[1] - n_dim:Jy1Jy2T.shape[1]])
    J1J2T = J1J2T.at[-n_dim:, -Jy1Jy2T.shape[1] - n_dim:-n_dim].set(Jy2[-1, :, Jy1Jy2T.shape[1] - n_dim:Jy1Jy2T.shape[1]].T)
    
    return J1J2T