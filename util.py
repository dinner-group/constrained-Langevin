import jax
import jax.numpy as np
import numpy
from functools import partial
jax.config.update("jax_enable_x64", True)

midpoint = np.array([0])
lobatto_points_3 = np.array([-1, 0, 1])
gauss_points_2 = np.array([-np.sqrt(1/3), np.sqrt(1/3)])
gauss_points_3 = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
gauss_points_4 = np.array([-np.sqrt(3/7 + (2/7) * np.sqrt(6/5)), -np.sqrt(3/7 - (2/7) * np.sqrt(6/5)), np.sqrt(3/7 - (2/7) * np.sqrt(6/5)), np.sqrt(3/7 + (2/7) * np.sqrt(6/5))])
gauss_weights_4 = np.array([18 - np.sqrt(30), 18 + np.sqrt(30), 18 + np.sqrt(30), 18 - np.sqrt(30)]) / 36

@jax.jit
def smooth_max(x, smooth_max_temperature=1):
    return np.sum(x * np.exp(smooth_max_temperature * x)) / np.sum(np.exp(smooth_max_temperature * x))

@jax.jit
def fill_mesh(t, colloc_points_unshifted=gauss_points_4):
    return np.concatenate([np.ravel(np.expand_dims(t[:-1], 1) + np.expand_dims(t[1:] - t[:-1], 1) * np.linspace(0, 1, colloc_points_unshifted.size + 1)[:-1]), t[-1:]])

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

@partial(jax.jit, static_argnames=("n_derivs",))
def newton_polynomial_1(t, node_t, node_y, dd=None, n_derivs=0):
    
    if n_derivs > node_t.size:
        n_derivs = node_t.size - 1
    if dd is None:
        dd = divided_difference(node_t, node_y)

    def loop_inner(j, carry):
        poly, ind_i = carry
        ind_j = n_derivs - j
        poly = poly.at[ind_j].set((t - node_t[ind_i]) * poly[ind_j] + ind_j * poly[ind_j - 1])
        return poly, ind_i
        
    def loop_outer(i, poly):
        ind_i = node_t.size - 2 - i
        poly = jax.lax.fori_loop(0, n_derivs, loop_inner, (poly, ind_i))[0]
        poly = poly.at[0].set(dd[ind_i, ind_i] + (t - node_t[ind_i]) * poly[0])
        return poly
    
    out = jax.lax.fori_loop(0, node_t.size - 1, loop_outer, np.pad(np.expand_dims(dd[node_t.size - 1, node_t.size - 1], 0), ((0, n_derivs), (0, 0))))
    
    if n_derivs == 0:
        out = out[0]
    
    return out

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
def recompute_mesh(y, mesh_old, colloc_points_unshifted=gauss_points_4, n_smooth=4):
    
    def loop_body(i, _):
        meshi = np.linspace(*jax.lax.dynamic_slice(mesh_old, (i,), (2,)), colloc_points_unshifted.size + 1)
        yi = jax.lax.dynamic_slice(y, (0, i * colloc_points_unshifted.size), (y.shape[0], colloc_points_unshifted.size + 1))
        return i + 1, divided_difference(meshi, yi)[colloc_points_unshifted.size, colloc_points_unshifted.size]

    shift = mesh_old[0]
    scale = mesh_old[-1] - mesh_old[0]
    mesh_old = (mesh_old - shift) / scale
    _, deriv = jax.lax.scan(loop_body, init=0, xs=None, length=mesh_old.size - 1)
    midpoints = (mesh_old[1:] + mesh_old[:-1]) / 2
    deriv = np.pad((deriv[1:] - deriv[:-1]).T / (midpoints[1:] - midpoints[:-1]), ((0, 0), (1, 1)), mode="edge")
    a = np.maximum(1, (jax.scipy.integrate.trapezoid(np.sum(deriv**2, axis=0)**(1 / (1 + 2 * (colloc_points_unshifted.size + 1))), x=mesh_old) / (mesh_old[-1] - mesh_old[0]))**(1 + 2 * (colloc_points_unshifted.size + 1)))
    mesh_density = (1 + np.sum(deriv**2, axis=0) / a)**(1 / (1 + 2 * (colloc_points_unshifted.size + 1)))
    mesh_density = weighted_average_smoothing(mesh_density, n_smooth)
    mesh_mass = np.pad(np.cumsum((mesh_density[1:] + mesh_density[:-1]) * (mesh_old[1:] - mesh_old[:-1])) / 2, (1, 0))
    mesh_new = np.interp(np.linspace(0, mesh_mass[-1], mesh_old.size), mesh_mass, mesh_old)
    
    return scale * mesh_new + shift, mesh_density

@jax.jit
def recompute_node_y(y, mesh_old, mesh_new, colloc_points_unshifted=gauss_points_4):

    t_eval = fill_mesh(mesh_new, colloc_points_unshifted)
    y_interp = interpolate(y, mesh_old, t_eval[1:-1], colloc_points_unshifted)
    return np.hstack([y[:, :1], y_interp, y[:, -1:]])

@jax.jit
def interpolate(y, mesh_points, t_eval, colloc_points_unshifted=gauss_points_4):
    
    def loop1(i, _):
        meshi = np.linspace(*jax.lax.dynamic_slice(mesh_points, (i,), (2,)), colloc_points_unshifted.size + 1)
        yi = jax.lax.dynamic_slice(y, (0 * i, i * colloc_points_unshifted.size,), (y.shape[0], colloc_points_unshifted.size + 1))
        return i + 1, divided_difference(meshi, yi)
        
    dd = jax.lax.scan(loop1, init=0, xs=None, length=mesh_points.size - 1)[1]
        
    def loop2(_, t):
        i = np.maximum(np.searchsorted(mesh_points, t) - 1, 0)
        meshi = np.linspace(*jax.lax.dynamic_slice(mesh_points, (i,), (2,)), colloc_points_unshifted.size + 1)
        yi = jax.lax.dynamic_slice(y, (0 * i, i * colloc_points_unshifted.size,), (y.shape[0], colloc_points_unshifted.size + 1))
        return _, newton_polynomial(t, meshi, yi, dd[i])
    
    _, y_interp = jax.lax.scan(loop2, init=None, xs=t_eval)
    return y_interp.T

@jax.jit
def curvature_interval_poly(y, colloc_points_unshifted=gauss_points_4, dd=None):

    colloc_points = 1 + colloc_points_unshifted
    node_points = np.linspace(0, 1, colloc_points.size + 1)
    if dd is None:
        dd = divided_difference(node_points, y)
    poly = jax.vmap(newton_polynomial, (0, None, None, None))(colloc_points, node_points, y, dd)
    poly_deriv2 = jax.vmap(jax.jacfwd(jax.jacfwd(newton_polynomial)), (0, None, None, None))(colloc_points, node_points, y, dd)
    return poly.T, poly_deriv2.T

@jax.jit
def curvature_poly(y, colloc_points_unshifted=gauss_points_4, quadrature_weights=gauss_weights_4):

    def loop_body(i, carry):
        
        ynorm, ycurvature = carry
        poly, poly_deriv2 = curvature_interval_poly(jax.lax.dynamic_slice(y, (0, i * colloc_points_unshifted.size), (y.shape[0], colloc_points_unshifted.size + 1)), colloc_points_unshifted)
        ynorm = ynorm.at[i].set(np.linalg.norm(poly, axis=0)@quadrature_weights)
        ycurvature = ycurvature.at[i].set(np.linalg.norm(poly_deriv2, axis=0)@quadrature_weights)
        return ynorm, ycurvature
    
    out_size = y.shape[1] // colloc_points_unshifted.size
    return jax.lax.fori_loop(0, out_size, loop_body, (np.zeros(out_size), np.zeros(out_size)))

@jax.jit
def compute_householder_reflector(x):
    
    v = x.at[0].set(1)
    sigma = v[1:].T@v[1:]
    x_norm = np.sqrt(x[0]**2 + sigma)
    v0 = np.where(x[0] <= 0, x[0] - x_norm, -sigma / (x[0] + x_norm))
    vv = v.at[0].set(v0)
    tau = 2 / (1 + sigma / v0**2)
    vv = vv / vv[0]
    tau = np.where(sigma == 0, 0, tau)
    v = np.where(sigma == 0, v, vv)
    
    return v, tau

@jax.jit
def householder_qr(A):

    if A.size == 0:
        return A.T, np.zeros(0), A

    mask = np.tril(np.ones_like(A))
    
    def loop_body(i, carry):
        h, tau, R = carry
        R_i = np.roll(R[:, i] * mask[:, i], -i)
        h_i, tau_i = compute_householder_reflector(R_i)
        h_i = np.roll(h_i, i)
        R = R - tau_i * np.outer(h_i, h_i@R)
        h = h.at[i].set(h_i)
        tau = tau.at[i].set(tau_i)
        return h, tau, R
    
    out = jax.lax.fori_loop(0, A.shape[1], loop_body, (np.zeros_like(A.T), np.zeros(A.shape[1]), A))
    return *out[:2], np.triu(out[2])

@partial(jax.jit, static_argnames="transpose")
def Q_multiply_from_reflectors(h, tau, v, transpose=False):

    is_vector = len(v.shape) == 1
    
    if is_vector:
        if right:
            v = np.expand_dims(v, 1)
        else:
            v = np.expand_dims(v, 0)

    if(h.shape[0] > h.shape[1]):
        h = h[:h.shape[1]]
            
    def loop_body(A, x):
        u, t = x
        A = jax.lax.cond(t != 0, lambda:A - t * np.outer(u, u@A), lambda:A)
        return A, None 
        
    return jax.lax.scan(loop_body, init=v, xs=(np.triu(h).at[np.diag_indices(h.shape[0])].set(1), tau), reverse=not transpose)[0]

@jax.jit
def compute_givens_rotation(a, b):
    b_zero = abs(b) == 0
    a_lt_b = abs(a) < abs(b)
    t = -np.where(a_lt_b, a, b) / np.where(a_lt_b, b, a)
    r = jax.lax.rsqrt(1 + abs(t) ** 2).astype(t.dtype)
    cs = np.where(b_zero, 1, np.where(a_lt_b, r * t, r))
    sn = np.where(b_zero, 0, np.where(a_lt_b, r, r * t))
    return cs, sn

@jax.jit
def apply_givens_rotation(H, i, j, cs, sn, transpose=False):
    x1 = H[i]
    y1 = H[j]
    x2 = cs.conj() * x1 - np.where(transpose, -1, 1) * sn.conj() * y1
    y2 = np.where(transpose, -1, 1) * sn * x1 + cs * y1
    H = H.at[i].set(x2)
    H = H.at[j].set(y2)
    return H

@jax.jit
def givens_qr(A):
    
    if A.shape[0] < A.shape[1]:
        n = A.shape[0]
    else:
        n = A.shape[1]
        
    givens_factors = np.zeros((A.size - n * (n - 1) // 2 - n, 2))
    
    def loop_inner(j, carry):
        i, k, givens_factors, A = carry
        cs, sn = compute_givens_rotation(A[i, i], A[j, i])
        A = apply_givens_rotation(A, i, j, cs, sn)
        givens_factors = givens_factors.at[k].set((cs, sn))
        return i, k + 1, givens_factors, A
    
    def loop_outer(i, carry):
        return jax.lax.fori_loop(i + 1, A.shape[0], loop_inner, (i, *carry))[1:]
    
    out =  jax.lax.fori_loop(0, A.shape[1], loop_outer, (0, givens_factors, A))
    return out[1], np.triu(out[2])

class BVPJac:

    def __init__(self, Jy, Jk, n_dim, n_par, Jbc_left=None, Jbc_right=None, colloc_points_unshifted=gauss_points_4):
        self.Jy = Jy
        self.Jk = Jk
        self.n_dim = n_dim
        self.n_par = n_par
        self.shape = ((Jy.shape[0] * Jy.shape[1] + n_dim, Jy.shape[0] * Jy.shape[1] + n_dim + Jk.shape[1]))
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

        Jk = np.pad(self.Jk, ((0, self.shape[0] - self.Jk.shape[0]), (0, 0)))
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
        out = out.at[:, :self.n_par].set(v[:, :-self.n_dim]@self.Jk[:, :self.n_par])
        out = out.at[:, self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim:].set(v[:, :-self.n_dim]@self.Jk[:, self.n_par:])
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
        out = np.pad(self.Jk@vk, ((0, self.shape[0] - self.Jk.shape[0]), (0, 0)))

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
        Jk = self.Jk * Dk
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
        
        J1J2T = np.pad(J1.Jk@J2.Jk.T, ((0, J1.n_dim), (0, J2.n_dim)))
        
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

        R_bc = np.zeros((self.n_dim + self.Jy.shape[1] * self.Jy.shape[0], self.Jbc_left.shape[0]))
        R_bc = R_bc.at[:self.n_dim].set(self.Jbc_left.T)
        R_bc = R_bc.at[-self.n_dim:].set(self.Jbc_right.T)

        J_i = self.Jy[0].T
        qi, ri = np.linalg.qr(J_i, mode="complete")
        Q_c = Q_c.at[0].set(qi)
        R_c = R_c.at[0, -ri.shape[1]:].set(ri[:-self.n_dim])

        R_bc = R_bc.at[:qi.shape[1]].set(qi.T@R_bc[:qi.shape[1]])

        def loop_body(carry, _):
            i, Q_c, R_c, R_bc = carry
            si = Q_c[i - 1, -self.n_dim:].T@self.Jy[i, :, :self.n_dim].T
            R_c = R_c.at[i, :self.Jy.shape[2]].set(si)
            bc = jax.lax.dynamic_slice(R_bc, (i * self.Jy.shape[1], 0), (self.Jy.shape[2], self.n_dim))

            J_i = self.Jy[i].T.at[:self.n_dim].set(si[-self.n_dim:])
            qi, ri = np.linalg.qr(J_i, mode="complete")

            Q_c = Q_c.at[i].set(qi)
            R_c = R_c.at[i, -ri.shape[1]:].set(ri[:-self.n_dim])

            R_bc = jax.lax.dynamic_update_slice(R_bc, qi.T@bc, (i * self.Jy.shape[1], 0))

            return (i + 1, Q_c, R_c, R_bc), _

        out = jax.lax.scan(loop_body, init=(1, Q_c, R_c, R_bc), xs=None, length=self.Jy.shape[0] - 1)
        Q_bc, ri = np.linalg.qr(out[0][3][-self.n_dim:])
        R_bc = out[0][3].at[-self.n_dim:].set(ri)

        return BVPJac_LQ(out[0][1], Q_bc, out[0][2], R_bc)

    @partial(jax.jit, static_argnames=("method",))
    def lq_factor_1(self, method="lapack"):

        Rbc = np.zeros((self.n_dim + self.Jy.shape[1] * self.Jy.shape[0], self.Jbc_left.shape[0]))
        Rbc = Rbc.at[:self.n_dim].set(self.Jbc_left.T)
        Rbc = Rbc.at[-self.n_dim:].set(self.Jbc_right.T)

        def loop_body(carry, Jy_i):

            i, h_prev, tau_prev, Rbc = carry

            Jy_i = Jy_i.T
            Jy_i = np.pad(Jy_i, ((Jy_i.shape[1], 0), (0, 0)))
            Jy_i = Jy_i.at[:-Jy_i.shape[1]].set(Q_multiply_from_reflectors(h_prev, tau_prev, Jy_i[:-Jy_i.shape[1]], transpose=True))

            if method == "lapack":
                h, tau = np.linalg.qr(Jy_i[Jy_i.shape[1]:], mode="raw")
                Rc_i = np.triu(h.T)
            elif method == "householder":
                h, tau, Rc_i = householder_qr(Jy_i[Jy_i.shape[1]:])

            Rc_i = Jy_i.at[Jy_i.shape[1]:2 * Jy_i.shape[1]].set(Rc_i[:Jy_i.shape[1]])

            Rbc_i = jax.lax.dynamic_slice(Rbc, (i * Jy_i.shape[1], 0), (Jy_i.shape[0] - Jy_i.shape[1], Rbc.shape[1]))
            Rbc = jax.lax.dynamic_update_slice(Rbc, Q_multiply_from_reflectors(h, tau, Rbc_i, transpose=True), ((i * Jy_i.shape[1], 0)))

            return (i + 1, h, tau, Rbc), (h, tau, Rc_i[:2 * Jy_i.shape[1]])

        out = jax.lax.scan(loop_body, init=(0, np.zeros((self.Jy.shape[1], self.Jy.shape[2])), np.zeros(self.Jy.shape[1]), Rbc), xs=self.Jy)
        Rbc = out[0][3]
        h, tau, Rc = out[1]

        if method == "lapack":
            h_bc, tau_bc = np.linalg.qr(Rbc[-self.n_dim:], mode="raw")
            Rbc_i = np.triu(h_bc.T)
        elif method == "householder":
            h_bc, tau_bc, Rbc_i = householder_qr(Rbc[-self.n_dim:])

        Rbc_i = Rbc_i[:Rbc_i.shape[0] - self.n_dim + Rbc_i.shape[1]]
        Rbc = Rbc.at[Rbc.shape[0] - self.n_dim:Rbc.shape[0] - self.n_dim + Rbc.shape[1]].set(Rbc_i)

        return BVPMMJac_LQ_1(h, tau, h_bc, tau_bc, Rc, Rbc[:Rbc.shape[0] - self.Jy.shape[0] - self.n_dim + Rbc.shape[1]], self.n_dim, self.n_par, self.colloc_points_unshifted)
    
    def _tree_flatten(self):
        children = (self.Jy, self.Jk, self.Jbc_left, self.Jbc_right, self.colloc_points_unshifted)
        aux_data = {"n_dim":self.n_dim, "n_par":self.n_par}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children[:2], Jbc_left=children[2], Jbc_right=children[3], colloc_points_unshifted=children[4], **aux_data)

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
            R_bc_i = jax.lax.dynamic_slice(self.R_bc, (i * self.R_c.shape[2], 0), (self.R_c.shape[2], self.R_bc.shape[1]))
            b_i -= self.R_c[i + 1, :self.R_c.shape[2]]@x_i_prev + R_bc_i@x[-self.R_bc.shape[1]:]
            x_i = jax.scipy.linalg.solve_triangular(self.R_c[i, self.R_c.shape[2]:], b_i, lower=False)
            x = jax.lax.dynamic_update_slice(x, x_i, (i * self.R_c.shape[2], 0))
            return (i - 1, x), _

        x = jax.lax.scan(loop_body, init=(self.R_c.shape[0] - 2, x), xs=None, length=self.R_c.shape[0] - 1)[0][1]

        if is_vector:
            x = x.ravel()

        return x

    @jax.jit
    def Q_multiply(self, v):
        
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

    def __init__(self, Jy, Jk, Jmesh, n_dim, n_par, Jbc_left=None, Jbc_right=None, colloc_points_unshifted=gauss_points_4):
        self.Jy = Jy
        self.Jk = Jk
        self.Jmesh = Jmesh
        self.n_dim = n_dim
        self.n_par = n_par
        self.shape = ((Jy.shape[0] * Jy.shape[1] + n_dim + Jmesh.shape[0], Jy.shape[0] * Jy.shape[1] + n_dim + Jy.shape[0] - 1 + Jk.shape[1]))
        self.colloc_points_unshifted = colloc_points_unshifted

        if Jbc_left is None:
            self.Jbc_left = -np.identity(n_dim)
        else:
            self.Jbc_left = Jbc_left

        if Jbc_right is None:
            self.Jbc_right = np.identity(n_dim)
        else:
            self.Jbc_right = Jbc_right

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2))
    def permute_col(x, n_dim, n_mesh_intervals, colloc_points_unshifted=gauss_points_4):
       
        is_vector = len(x.shape) == 1
        if is_vector:
            x = np.expand_dims(x, 0)

        x = np.hstack([x[:, :n_dim], np.concatenate([x[:, n_dim:-n_dim * colloc_points_unshifted.size - n_mesh_intervals + 1].reshape((x.shape[0], n_dim * colloc_points_unshifted.size, n_mesh_intervals - 1), order="F"),
                                                               np.expand_dims(x[:, -n_mesh_intervals + 1:], 1)], axis=1).reshape((x.shape[0], (n_dim * colloc_points_unshifted.size + 1) * (n_mesh_intervals - 1)), order="F"), 
                       x[:, -n_dim * colloc_points_unshifted.size - n_mesh_intervals + 1:-n_mesh_intervals + 1]])

        if is_vector:
            x = np.ravel(x)

        return x

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2))
    def unpermute_col(x, n_dim, n_mesh_intervals, colloc_points_unshifted=gauss_points_4):
        
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

    @jax.jit
    def todense(self):

        Jk = np.pad(self.Jk, ((0, self.shape[0] - self.Jk.shape[0]), (0, 0)))
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
        return np.hstack([Jk[:, :self.n_par], np.vstack([Jy_mesh_dense, self.unpermute_col(self.Jmesh, self.n_dim, self.Jy.shape[0], self.colloc_points_unshifted)]), Jk[:, self.n_par:]])

    @jax.jit
    def left_multiply(self, v):
   
        is_vector = len(v.shape) == 1
        if is_vector:
            v = np.expand_dims(v, 0)

        out = np.zeros((v.shape[0], self.shape[1]))
        out = out.at[:, :self.n_par].add(v[:, :Jk.shape[0]]@self.Jk[:, :self.n_par])
        out = out.at[:, self.n_par - self.Jk.shape[1]:].add(v[:, :Jk.shape[0]]@self.Jk[:, self.n_par:])
     
        out = out.at[:, self.n_par:self.n_par + self.Jmesh.shape[1]].add(v[:, -self.Jmesh.shape[0]:]@self.Jmesh)
     
        out = out.at[:, self.n_par:self.n_par + self.n_dim].add(v[:, -self.Jmesh.shape[0] - self.n_dim:-self.Jmesh.shape[0]]@self.Jbc_left)
        out = out.at[:, self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.Jmesh.shape[0]:self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.Jmesh.shape[0] + self.n_dim]\
              .add(v[:, -self.Jmesh.shape[0] - self.n_dim:-self.Jmesh.shape[0]]@self.Jbc_right)

        Jy_0 = np.hstack([self.Jy[0, :, :self.n_dim], self.Jy[0, :, self.n_dim + 1:]])
        out = out.at[:, self.n_par:self.n_par + Jy_0.shape[1]].add(v[:, :self.Jy.shape[1]]@Jy_0)

        Jy_N = self.Jy[-1, :, :-1]
        out = out.at[:, self.n_par - self.Jk.shape[1] - Jy_N.shape[1]:self.n_par - self.Jk.shape[1]].add(v[:, -self.Jmesh.shape[0] - self.Jy.shape[1] - self.n_dim:-self.Jmesh.shape[0] - self.n_dim]@Jy_N)
        
        def loop_body(carry, _):
            i, out = carry
            out_i = jax.lax.dynamic_slice(out, (0, self.n_par + i * (1 + self.Jy.shape[1]) - 1), (out.shape[0], self.Jy.shape[2]))
            v_i = jax.lax.dynamic_slice(v, (0, i * self.Jy.shape[1]), (v.shape[0], self.Jy.shape[1]))
            out = jax.lax.dynamic_update_slice(out, out_i + v_i@self.Jy[i], (0, self.n_par + i * (1 + self.Jy.shape[1]) - 1))
            return (i + 1, out), _

        out = jax.lax.scan(loop_body, init=(1, out), xs=None, length=self.Jy.shape[0] - 2)[0][1]
        out = out.at[:, self.n_par:self.n_par - self.Jk.shape[1]].set(self.unpermute_col(out[:, self.n_par:self.n_par - self.Jk.shape[1]], self.n_dim, self.Jy.shape[0], self.colloc_points_unshifted))

        if is_vector:
            out = out.ravel()

        return out
   
    @jax.jit
    def right_multiply(self, v):
       
        is_vector = len(v.shape) == 1
        if is_vector:
            v = np.expand_dims(v, 1)

        vy = v[self.n_par:self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim + self.Jmesh.shape[0]]
        vy = self.permute_col(vy.T, self.n_dim, self.Jy.shape[0], self.colloc_points_unshifted).T
        vk = np.concatenate([v[:self.n_par], v[self.n_par + self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim + self.Jmesh.shape[0]:]])
        out = np.pad(self.Jk@vk, ((0, self.shape[0] - self.Jk.shape[0]), (0, 0)))
        out = out.at[-self.Jmesh.shape[0]:].add(self.Jmesh@vy)
        
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

        Dy = D[self.n_par:self.n_par - self.Jk.shape[1]]
        Dy = self.permute_col(Dy, self.n_dim, self.Jy.shape[0], self.colloc_points_unshifted)
        Dk = np.concatenate([D[:self.n_par], D[self.n_par - self.Jk.shape[1]:]])

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
        Jk = self.Jk * Dk
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

    def __init__(self, Q_c, Q_bc, R_c, R_bc, n_dim, n_par, colloc_points_unshifted=gauss_points_4):

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
            R_bc_i = jax.lax.dynamic_slice(self.R_bc, (i * self.R_c.shape[2], 0), (self.R_c.shape[2], self.R_bc.shape[1]))
            b_i -= self.R_c[i + 1, :self.R_c.shape[2]]@x_i_prev + R_bc_i@x[-self.R_bc.shape[1]:]
            x_i = jax.scipy.linalg.solve_triangular(self.R_c[i, self.R_c.shape[2]:], b_i, lower=False)
            x = jax.lax.dynamic_update_slice(x, x_i, (i * self.R_c.shape[2], 0))
            return (i - 1, x), _

        x = jax.lax.scan(loop_body, init=(self.R_c.shape[0] - 2, x), xs=None, length=self.R_c.shape[0] - 1)[0][1]

        if is_vector:
            x = x.ravel()

        return x

    @jax.jit
    def Q_multiply(self, v):

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

        v = BVPMMJac.unpermute_col(v.T, self.n_dim, self.R_c.shape[0], self.colloc_points_unshifted).T

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

    def __init__(self, Jy, Jk, Jbc, n_dim, n_par, colloc_points_unshifted=gauss_points_4):
        self.Jy = Jy
        self.Jk = Jk
        self.Jbc = Jbc
        self.n_dim = n_dim
        self.n_par = n_par
        self.shape = ((Jy.shape[0] * Jy.shape[1] + Jbc.shape[0], Jy.shape[0] * Jy.shape[1] + n_dim + Jy.shape[0] + Jk.shape[1]))
        self.colloc_points_unshifted = colloc_points_unshifted

    @staticmethod
    @partial(jax.jit, static_argnames=("n_dim", "n_mesh_intervals"))
    def permute_col(x, n_dim, n_mesh_intervals, colloc_points_unshifted=gauss_points_4):
       
        is_vector = len(x.shape) == 1
        if is_vector:
            x = np.expand_dims(x, 0)

        x = np.hstack([x[:, :n_dim], np.concatenate([np.expand_dims(x[:, -n_mesh_intervals:], 1), x[:, n_dim:-n_mesh_intervals].reshape((x.shape[0], n_dim * colloc_points_unshifted.size, n_mesh_intervals), order="F")],
                      axis=1).reshape((x.shape[0], (n_dim * colloc_points_unshifted.size + 1) * n_mesh_intervals), order="F")])

        if is_vector:
            x = np.ravel(x)

        return x

    @staticmethod
    @partial(jax.jit, static_argnames=("n_dim", "n_mesh_intervals"))
    def unpermute_col(x, n_dim, n_mesh_intervals, colloc_points_unshifted=gauss_points_4):
        
        is_vector = len(x.shape) == 1
        if is_vector:
            x = np.expand_dims(x, 0)

        y_and_mesh = x[:, n_dim:].reshape((x.shape[0], n_dim * colloc_points_unshifted.size + 1, n_mesh_intervals), order="F")
        x = np.hstack([x[:, :n_dim], y_and_mesh[:, 1:, :].reshape((x.shape[0], n_dim * colloc_points_unshifted.size * n_mesh_intervals), order="F"), y_and_mesh[:, :1, :].reshape((x.shape[0], n_mesh_intervals), order="F")])

        if is_vector:
            x = np.ravel(x)

        return x

    @jax.jit
    def todense(self):

        Jk = np.pad(self.Jk, ((0, self.shape[0] - self.Jk.shape[0]), (0, 0)))
        Jy_dense = np.zeros((self.Jy.shape[0] * self.Jy.shape[1], self.Jy.shape[0] * self.Jy.shape[1] + self.n_dim))
        Jm_dense = np.zeros((self.Jy.shape[0] * self.Jy.shape[1], self.Jy.shape[0]))

        def loop_body(carry, _):
            i, Jy_dense, Jm_dense = carry
            Jy_dense = jax.lax.dynamic_update_slice(Jy_dense, self.Jy[i, :, :self.n_dim], (i * self.Jy.shape[1], i * self.Jy.shape[1]))
            Jy_dense = jax.lax.dynamic_update_slice(Jy_dense, self.Jy[i, :, self.n_dim + 1:], (i * self.Jy.shape[1], i * self.Jy.shape[1] + self.n_dim))
            Jm_dense = jax.lax.dynamic_update_slice(Jm_dense, self.Jy[i, :, self.n_dim:self.n_dim + 1], (i * self.Jy.shape[1], i))
            return (i + 1, Jy_dense, Jm_dense), _

        Jy_mesh_dense = np.hstack(jax.lax.scan(loop_body, init=(0, Jy_dense, Jm_dense), xs=None, length=self.Jy.shape[0])[0][1:])
        return np.hstack([Jk, np.vstack([Jy_mesh_dense, self.unpermute_col(self.Jbc, self.n_dim, self.Jy.shape[0], self.colloc_points_unshifted)])])

    @jax.jit
    def left_multiply(self, v):
   
        is_vector = len(v.shape) == 1
        if is_vector:
            v = np.expand_dims(v, 0)

        out = np.zeros((v.shape[0], self.shape[1]))
        out = out.at[:, :self.n_par].add(v[:, :self.Jk.shape[0]]@self.Jk)
        out = out.at[:, self.n_par:].add(v[:, -self.Jbc.shape[0]:]@self.Jbc)
        
        def loop_body(carry, _):
            i, out = carry
            out_i = jax.lax.dynamic_slice(out, (0, self.n_par + i * (1 + self.Jy.shape[1])), (out.shape[0], self.Jy.shape[2]))
            v_i = jax.lax.dynamic_slice(v, (0, i * self.Jy.shape[1]), (v.shape[0], self.Jy.shape[1]))
            out = jax.lax.dynamic_update_slice(out, out_i + v_i@self.Jy[i], (0, self.n_par + i * (1 + self.Jy.shape[1])))
            return (i + 1, out), _

        out = jax.lax.scan(loop_body, init=(0, out), xs=None, length=self.Jy.shape[0])[0][1]
        out = out.at[:, self.n_par:].set(self.unpermute_col(out[:, self.n_par:], self.n_dim, self.Jy.shape[0], self.colloc_points_unshifted))

        if is_vector:
            out = out.ravel()

        return out
   
    @jax.jit
    def right_multiply(self, v):
       
        is_vector = len(v.shape) == 1
        if is_vector:
            v = np.expand_dims(v, 1)

        vy = v[self.n_par:]
        vy = self.permute_col(vy.T, self.n_dim, self.Jy.shape[0], self.colloc_points_unshifted).T
        out = np.pad(self.Jk@v[:self.n_par], ((0, self.shape[0] - self.Jk.shape[0]), (0, 0)))
        out = out.at[-self.Jbc.shape[0]:].add(self.Jbc@vy)
        
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
        Dy = self.permute_col(Dy, self.n_dim, self.Jy.shape[0], self.colloc_points_unshifted)
        Dk = D[:self.n_par]
        Jbc = self.Jbc * Dy

        def loop_body(i, Jy_i):
            Dy_i = jax.lax.dynamic_slice(Dy, (i * (1 + Jy_i.shape[0]),), (Jy_i.shape[1],))
            return i + 1, Jy_i * Dy_i

        Jy = jax.lax.scan(loop_body, init=0, xs=self.Jy)[1]
        Jk = self.Jk * Dk
        
        return BVPMMJac_1(Jy, Jk, Jbc, self.n_dim, self.n_par, self.colloc_points_unshifted)

    @jax.jit
    def lq_factor(self):

        Jy = self.Jy
        Jy = np.pad(Jy, ((0, 0), (0, 0), (0, Jy.shape[1] + Jy.shape[0] - 1)))

        def loop_body(carry, Jy_i):

            i, h_prev, tau_prev, Rbc = carry

            Jy_i = np.roll(Jy_i, i + Jy_i.shape[0], axis=1).T
            Jy_i = Jy_i.at[:-Jy_i.shape[1]].set(Q_multiply_from_reflectors(h_prev, tau_prev, Jy_i[:-Jy_i.shape[1]], transpose=True))

            h, tau = np.linalg.qr(Jy_i[Jy_i.shape[1]:], mode="raw")
            Rc_i = np.triu(h.T)
            Rc_i = Jy_i.at[Jy_i.shape[1]:2 * Jy_i.shape[1]].set(Rc_i[:Jy_i.shape[1]])

            Rbc_i = jax.lax.dynamic_slice(Rbc, (i * Jy_i.shape[1], 0), (Jy_i.shape[0] - Jy_i.shape[1], Rbc.shape[1]))
            Rbc = jax.lax.dynamic_update_slice(Rbc, Q_multiply_from_reflectors(h, tau, Rbc_i, transpose=True), ((i * Jy_i.shape[1], 0)))

            return (i + 1, h, tau, Rbc), (h, tau, Rc_i[:2 * Jy_i.shape[1]])

        out = jax.lax.scan(loop_body, init=(0, np.zeros((Jy.shape[1], Jy.shape[2] - Jy.shape[1])), np.zeros(Jy.shape[1]), self.Jbc.T), xs=Jy)
        Rbc = out[0][3]
        h, tau, Rc = out[1]
        h_bc, tau_bc = np.linalg.qr(Rbc[-self.Jy.shape[0] - self.n_dim:], mode="raw")
        Rbc_i = np.triu(h_bc.T)
        Rbc_i = Rbc_i[:Rbc_i.shape[0] - self.Jy.shape[0] - self.n_dim + Rbc_i.shape[1]]
        Rbc = Rbc.at[Rbc.shape[0] - self.Jy.shape[0] - self.n_dim:Rbc.shape[0] - self.Jy.shape[0] - self.n_dim + Rbc.shape[1]].set(Rbc_i)

        return BVPMMJac_LQ_1(h, tau, h_bc, tau_bc, Rc, Rbc[:Rbc.shape[0] - self.Jy.shape[0] - self.n_dim + Rbc.shape[1]], self.n_dim, self.n_par, self.colloc_points_unshifted)
    
    def _tree_flatten(self):
        children = (self.Jy, self.Jk, self.Jbc, self.colloc_points_unshifted)
        aux_data = {"n_dim":self.n_dim, "n_par":self.n_par}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children[:3], colloc_points_unshifted=children[3], **aux_data)

class BVPMMJac_LQ_1:

    def __init__(self, h_c, tau_c, h_bc, tau_bc, Rc, Rbc, n_dim, n_par, colloc_points_unshifted=gauss_points_4):

        self.h_c = h_c
        self.tau_c = tau_c
        self.h_bc = h_bc
        self.tau_bc = tau_bc
        self.Rc = Rc
        self.Rbc = Rbc
        self.n_dim = n_dim
        self.n_par = n_par
        self.colloc_points_unshifted = colloc_points_unshifted

    @jax.jit
    def solve_triangular_L(self, b):

        is_vector = len(b.shape) == 1
        if is_vector:
            b = np.expand_dims(b, 1)

        x = np.zeros((self.Rbc.shape[0], b.shape[1]))
        x_i = jax.scipy.linalg.solve_triangular(self.Rc[0, self.Rc.shape[2]:].T, b[:self.Rc.shape[2]], lower=True)
        x = x.at[:self.Rc.shape[2]].set(x_i)

        def loop_body(carry, _):

            i, x = carry
            x_i_prev = jax.lax.dynamic_slice(x, ((i - 1) * self.Rc.shape[2], 0), (self.Rc.shape[2], x.shape[1]))
            b_i = jax.lax.dynamic_slice(b, (i * self.Rc.shape[2], 0), (self.Rc.shape[2], b.shape[1]))
            x_i = jax.scipy.linalg.solve_triangular(self.Rc[i, self.Rc.shape[2]:].T, b_i - self.Rc[i, :self.Rc.shape[2]].T@x_i_prev, lower=True)
            x = jax.lax.dynamic_update_slice(x, x_i, (i * self.Rc.shape[2], 0))
            return (i + 1, x), _

        x = jax.lax.scan(loop_body, init=(1, x), xs=None, length=self.Rc.shape[0] - 1)[0][1]
        x_i = jax.scipy.linalg.solve_triangular(self.Rbc[-self.Rbc.shape[1]:].T, b[-self.Rbc.shape[1]:] - self.Rbc[:-self.Rbc.shape[1]].T@x[:-self.Rbc.shape[1]], lower=True)
        x = x.at[-self.Rbc.shape[1]:].set(x_i)

        if is_vector:
            x = x.ravel()

        return x

    @jax.jit
    def solve_triangular_R(self, b):

        is_vector = len(b.shape) == 1
        if is_vector:
            b = np.expand_dims(b, 1)

        x = np.zeros((self.Rbc.shape[0], b.shape[1]))
        x_i = jax.scipy.linalg.solve_triangular(self.Rbc[-self.Rbc.shape[1]:], b[-self.Rbc.shape[1]:], lower=False)
        x = x.at[-self.Rbc.shape[1]:].set(x_i)
        b_i = b[-self.Rc.shape[2] - self.Rbc.shape[1]:-self.Rbc.shape[1]] - self.Rbc[-self.Rc.shape[2] - self.Rbc.shape[1]:-self.Rbc.shape[1]]@x[-self.Rbc.shape[1]:]
        x_i = jax.scipy.linalg.solve_triangular(self.Rc[-1, self.Rc.shape[2]:], b_i, lower=False)
        x = x.at[-self.Rc.shape[2] - self.Rbc.shape[1]:-self.Rbc.shape[1]].set(x_i)

        def loop_body(carry, _):

            i, x = carry
            x_i_prev = jax.lax.dynamic_slice(x, ((i + 1) * self.Rc.shape[2], 0), (self.Rc.shape[2], x.shape[1]))
            b_i = jax.lax.dynamic_slice(b, (i * self.Rc.shape[2], 0), (self.Rc.shape[2], b.shape[1]))
            Rbc_i = jax.lax.dynamic_slice(self.Rbc, (i * self.Rc.shape[2], 0), (self.Rc.shape[2], self.Rbc.shape[1]))
            b_i -= self.Rc[i + 1, :self.Rc.shape[2]]@x_i_prev + Rbc_i@x[-self.Rbc.shape[1]:]
            x_i = jax.scipy.linalg.solve_triangular(self.Rc[i, self.Rc.shape[2]:], b_i, lower=False)
            x = jax.lax.dynamic_update_slice(x, x_i, (i * self.Rc.shape[2], 0))
            return (i - 1, x), _

        x = jax.lax.scan(loop_body, init=(self.Rc.shape[0] - 2, x), xs=None, length=self.Rc.shape[0] - 1)[0][1]

        if is_vector:
            x = x.ravel()

        return x

    @partial(jax.jit, static_argnames=("transpose", "permute"))
    def Q_multiply(self, v, transpose=False, permute=True):

        is_vector = len(v.shape) == 1
        if is_vector:
            v = np.expand_dims(v, 1)

        def loop_body(carry, _):
            i, v = carry
            v_i = jax.lax.dynamic_slice(v, (i * self.Rc.shape[2], 0), (self.h_c.shape[2], v.shape[1]))
            v_i = Q_multiply_from_reflectors(self.h_c[i], self.tau_c[i], v_i, transpose=transpose)
            v = jax.lax.dynamic_update_slice(v, v_i, (i * self.Rc.shape[2], 0))
            return (i + np.where(transpose, 1, -1), v), _

        n_rows_Q = (self.h_c.shape[0] - 1) * self.h_c.shape[1] + self.h_c.shape[2]

        if not transpose:
            v = np.pad(v, ((0, n_rows_Q - v.shape[0]), (0, 0)))
            v = v.at[-self.h_bc.shape[1]:].set(Q_multiply_from_reflectors(self.h_bc, self.tau_bc, v[-self.h_bc.shape[1]:], transpose=transpose))

        v = jax.lax.scan(loop_body, init=(np.where(transpose, 0, self.h_c.shape[0] - 1), v), xs=None, length=self.h_c.shape[0])[0][1]

        if transpose:
            v = v.at[-self.h_bc.shape[1]:].set(Q_multiply_from_reflectors(self.h_bc, self.tau_bc, v[-self.h_bc.shape[1]:], transpose=transpose))[:self.Rbc.shape[0]]
        else:
            if permute:
                v = BVPMMJac_1.unpermute_col(v.T, self.n_dim, self.Rc.shape[0], self.colloc_points_unshifted).T

        if is_vector:
            v = v.ravel()

        return v

    def _tree_flatten(self):
        children = (self.h_c, self.tau_c, self.h_bc, self.tau_bc, self.Rc, self.Rbc, self.colloc_points_unshifted)
        aux_data = {"n_dim":self.n_dim, "n_par":self.n_par}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children[:6], **aux_data, colloc_points_unshifted=children[6])

jax.tree_util.register_pytree_node(BVPJac, BVPJac._tree_flatten, BVPJac._tree_unflatten)
jax.tree_util.register_pytree_node(BVPJac_LQ, BVPJac_LQ._tree_flatten, BVPJac_LQ._tree_unflatten)
jax.tree_util.register_pytree_node(BVPMMJac, BVPMMJac._tree_flatten, BVPMMJac._tree_unflatten)
jax.tree_util.register_pytree_node(BVPMMJac_1, BVPMMJac_1._tree_flatten, BVPMMJac_1._tree_unflatten)
jax.tree_util.register_pytree_node(BVPMMJac_LQ, BVPMMJac_LQ._tree_flatten, BVPMMJac_LQ._tree_unflatten)
jax.tree_util.register_pytree_node(BVPMMJac_LQ_1, BVPMMJac_LQ_1._tree_flatten, BVPMMJac_LQ_1._tree_unflatten)
