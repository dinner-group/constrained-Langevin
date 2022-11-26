import numpy
import jax
import jax.numpy as np
import jax.experimental.sparse
import scipy.sparse
from functools import partial
jax.config.update("jax_enable_x64", True)

class colloc:

    gauss_points = (1 + np.array([-0.906179845938664, -0.538469310105683, 0.0, +0.538469310105683, +0.906179845938664])) / 2
    n_colloc_point = 5

    def __init__(self, f, f_p, y0, p0, args, ta=0, tb=1):

        self.args = args
        self.ta = ta
        self.tb = tb
        self.f = f
        self.f_p = f_p
        self.ta = ta
        self.tb = tb
        self.y = y0
        self.p = p0
        self.n_dim = y0.shape[0]
        self.n_mesh_point = y0.shape[1] // colloc.n_colloc_point
        self.mesh_points = np.linspace(self.ta, self.tb, self.n_mesh_point + 1)
        self.n_par = p0.shape[0]
        self.n_coeff = y0.size
        self.n = self.n_coeff + self.n_par
        self.n_colloc_eq = self.n_dim * self.n_mesh_point * colloc.n_colloc_point
        self.err = np.inf
        self.n_iter = 0
        self.t = np.linspace(ta, tb, self.n_coeff // self.n_dim)
        self.success = False


    @jax.jit
    def lagrange_poly_denom(points):

        poly_denom_arr = np.identity(points.shape[0])

        for i in range(points.shape[0]):
            poly_denom_arr = poly_denom_arr.at[i, :].add(points[i] - points)

        poly_denom = np.prod(poly_denom_arr, axis=1)
        return poly_denom
        
    @jax.jit
    def lagrange_poly(x, points, coeff, poly_denom=None):

        if poly_denom is None:
            poly_denom = colloc.lagrange_poly_denom(points)

        poly_numer_arr = np.tile(x - points, (points.shape[0], 1))
        poly_numer_arr = poly_numer_arr.at[np.diag_indices(points.shape[0])].set(1)
        poly_numer = np.prod(poly_numer_arr, axis=1)

        return np.sum((poly_numer / poly_denom) * coeff, axis=1)

    @jax.jit
    def lagrange_poly_grad(x, points, coeff, poly_denom=None):

        return jax.jacfwd(colloc.lagrange_poly, argnums=0)(x, points, coeff, poly_denom)

    @jax.jit
    def barycentric_weights(node_t):

        dts = node_t - np.tile(node_t, (node_t.size, 1)).T
        dts = dts.at[np.diag_indices(node_t.size)].set(1)
        return 1 / np.prod(dts, axis=0)

    @jax.jit
    def lagrange_poly_barycentric(t, node_t, node_y, weights=None):

        if weights is None:
            weights = colloc.barycentric_weights(node_t)

        dts = t - node_t
        return np.prod(dts) * np.sum((weights / dts) * node_y, axis=1)

    @jax.jit
    def lagrange_poly_barycentric_grad(t, node_t, node_y, weights=None):

        return jax.jacfwd(colloc.lagrange_poly_barycentric_grad, argnums=0)(t, node_t, node_y, weights)

    @jax.jit
    def divided_difference(node_t, node_y):

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
            dd = colloc.divided_difference(node_t, node_y.T)

        return np.sum(np.cumprod(np.roll(t - node_t, 1).at[0].set(1)) * dd[np.diag_indices(node_t.size)].T, axis=1)

    @jax.jit
    def newton_polynomial_grad(t, node_t, node_y, dd=None):

        return jax.jacfwd(colloc.newton_polynomial, argnums=0)(t, node_t, node_y, dd)

    @jax.jit
    def _compute_resid_interval(self, y, p, colloc_points, sub_points):

        y = y.reshape((self.n_dim, colloc.n_colloc_point + 1), order="F")
        #poly_denom = colloc.lagrange_poly_denom(sub_points)
        weights = colloc.barycentric_weights(sub_points)

        def loop_body(i, _):

            poly = colloc.lagrange_poly_barycentric(colloc_points[i], sub_points, y, weights)
            poly_grad = colloc.lagrange_poly_barycentric_grad(colloc_points[i], sub_points, y, weights)

            return i + 1, (poly_grad - self.f(colloc_points[i], poly, p, *self.args), poly_grad)

        result = jax.lax.scan(f=loop_body, init=0, xs=None, length=colloc.n_colloc_point)[1]
        return (result[0].ravel(order="C"), result[1].ravel(order="C"))

    @jax.jit
    def _resid_and_scale(self, y=None, p=None):

        if y is None:
            y = self.y.ravel(order="F")
        if p is None:
            p = self.p

        i = 0
        interval_width = colloc.n_colloc_point * self.n_dim
        
        def loop_body(i, _):

            sub_points = np.linspace(self.mesh_points[i], self.mesh_points[i + 1], colloc.n_colloc_point + 1)
            colloc_points = self.mesh_points[i] + self.gauss_points * (self.mesh_points[i + 1] - self.mesh_points[i])
            interval_start = i * interval_width
            y_i = jax.lax.dynamic_slice(y, (interval_start,), (interval_width + self.n_dim,))
            r_i = self._compute_resid_interval(y_i, p, colloc_points, sub_points)

            return i + 1, r_i
            
        result = jax.lax.scan(f=loop_body, init=i, xs=None, length=self.n_mesh_point)[1]

        return (np.concatenate([result[0].ravel(order="C"),
                               y[self.n_coeff - self.n_dim:self.n_coeff] - y[:self.n_dim], 
                               self.f_p(self.t, y, p, *self.args)]),
                np.concatenate([result[1].ravel(order="C"), np.ones(self.n_dim + self.n_par)]))

    @jax.jit
    def resid(self, y=None, p=None):
        
        if y is None:
            y = self.y.ravel(order="F")
        if p is None:
            p = self.p

        return self._resid_and_scale(y, p)[0]

    @jax.jit
    def jacp(self, y=None, p=None):

        if y is None:
            y = self.y.ravel(order="F")
        if p is None:
            p = self.p

        return jax.jacfwd(self.resid, argnums=1)(y, p)
    
    @jax.jit
    def _jac(self, y=None, p=None):
        
        if y is None:
            y = self.y.ravel(order="F")
        if p is None:
            p = self.p

        i = 0

        interval_width = colloc.n_colloc_point * self.n_dim
        block_y = np.mgrid[:interval_width + self.n_dim, :interval_width].T[:, :, np.array([1, 0])]
        block_p = np.mgrid[self.n_coeff:self.n_coeff + self.n_par, :interval_width].T[:, :, np.array([1, 0])]
        indices_base = np.hstack([block_y, block_p])

        def loop_body(i, _):

            sub_points = np.linspace(self.mesh_points[i], self.mesh_points[i + 1], colloc.n_colloc_point + 1)
            colloc_points = self.mesh_points[i] + self.gauss_points * (self.mesh_points[i + 1] - self.mesh_points[i])
            interval_start = i * interval_width
            y_i = jax.lax.dynamic_slice(y, (interval_start,), (interval_width + self.n_dim,))

            Jy_i = jax.jacfwd(self._compute_resid_interval, argnums=0)(y_i, p, colloc_points, sub_points)[0]
            Jp_i = jax.jacfwd(self._compute_resid_interval, argnums=1)(y_i, p, colloc_points, sub_points)[0]

            indices_i = indices_base.at[:, :-self.n_par].add(interval_start)
            indices_i = indices_i.at[:, -self.n_par:, 0].add(interval_start)

            data_i = np.hstack([Jy_i, Jp_i]).ravel()
            indices_i = np.vstack(indices_i)

            return i + 1, (indices_i, data_i)

        J_colloc_eq = jax.lax.scan(f=loop_body, init=i, xs=None, length=self.n_mesh_point)
        data = np.concatenate([np.ravel(J_colloc_eq[1][1], order="C"), 
                               np.hstack([-np.identity(self.n_dim), np.identity(self.n_dim)]).ravel(order="C"),
                               np.ravel(np.hstack([jax.jacrev(self.f_p, argnums=1)(self.t, y, p, *self.args), jax.jacrev(self.f_p, argnums=2)(self.t, y, p, *self.args)]), order="C")
                               ])
        indices = np.vstack([np.vstack(J_colloc_eq[1][0]), 
                             np.vstack(np.hstack([np.mgrid[:self.n_dim, self.n_colloc_eq:self.n_colloc_eq + self.n_dim].T[:, :, np.array([1, 0])],
                             np.mgrid[self.n_coeff - self.n_dim:self.n_coeff, self.n_colloc_eq:self.n_colloc_eq + self.n_dim].T[:, :, np.array([1, 0])]])),
                             np.mgrid[self.n_colloc_eq + self.n_dim:self.n_colloc_eq + self.n_dim + self.n_par, :self.n].reshape((2, self.n_par * self.n)).T
                             ])

        return jax.experimental.sparse.BCOO((data, indices), shape=(self.n, self.n), indices_sorted=True)
 
    def jac(self, y=None, p=None):
        
        if y is None:
            y = self.y.ravel(order="F")
        if p is None:
            p = self.p

        J = self._jac(y, p)
        self.J = scipy.sparse.csc_matrix((J.data, J.indices.T))
        return self.J
    
    def _superLU(self):
        
        self.jac_LU = scipy.sparse.linalg.splu(self.jac())

    def _newton_step(self, r=None, scale=None):
    
        if r is None or scale is None:
            r, scale = self._resid_and_scale()

        self._superLU()
        dx = self.jac_LU.solve(-numpy.asanyarray(r))
        x = np.concatenate([self.y.ravel(order="F"), self.p]) + dx
        self.y = x[:self.y.size].reshape((self.n_dim, self.n_coeff // self.n_dim), order="F")
        self.p = x[self.y.size:]
        
        r_next, scale_next = self._resid_and_scale()
        self.err = np.max(np.abs(r_next / (1 + scale_next)))
        return r_next, scale_next
    
    def solve(self, tol=1e-6, maxiter=10):

        self.success = False
        r, scale = self._resid_and_scale()
        self.err = np.max(np.abs(r / (1 + scale)))
        self.n_iter = 0
        
        while self.n_iter < maxiter and self.err >= tol:
            self.n_iter += 1
            r, scale = self._newton_step(r, scale)
            
        self.success = self.err < tol

    def damped_newton(self, tol=1e-6, predictor_trust=0.1, downhill_factor=0.01, min_damping_factor=0.01, maxiter=20):

        self.success = False
        r, scale = self._resid_and_scale()
        self.err = np.max(np.abs(r / (1 + scale)))
        self.n_iter = 0
        damping_factor = 1
        dx_prev = np.full(self.n, np.inf)
        b = np.zeros(self.n)

        while self.n_iter < maxiter and self.err > tol and damping_factor >= min_damping_factor:

            self.n_iter += 1
            self._superLU()
            dx = self.jac_LU.solve(-numpy.asanyarray(r))

            cost = np.linalg.norm(dx)**2
            u = np.linalg.norm(dx_prev) * damping_factor / np.linalg.norm(dx - b)
            damping_factor = np.maximum(min_damping_factor, np.minimum(u, 1))
            x = np.concatenate([self.y.ravel(order="F"), self.p])

            x_new = x + damping_factor * dx
            r, scale = self._resid_and_scale(x_new[:-self.n_par], x_new[-self.n_par:])
            cost_new = np.linalg.norm(self.jac_LU.solve(numpy.asanyarray(r)))**2

            while damping_factor >= min_damping_factor and cost_new > (1 - 2 * damping_factor * downhill_factor) * cost:

                damping_factor = np.maximum(predictor_trust * damping_factor, damping_factor**2 * cost / ((2 * damping_factor - 1) * cost + cost_new))
                x_new = x + damping_factor * dx
                r, scale = self._resid_and_scale(x_new[:-self.n_par], x_new[-self.n_par:])
                cost_new = np.linalg.norm(self.jac_LU.solve(numpy.asanyarray(r)))**2

            self.err = np.max(np.abs(r / (1 + scale)))
            dx_prev = dx
            b = self.jac_LU.solve(numpy.asanyarray(r))
            self.y = x_new[:-self.n_par].reshape((self.n_dim, self.n_coeff // self.n_dim), order="F")
            self.p = x_new[-self.n_par:]

        self.success = self.err < tol

    def _tree_flatten(self):

        children = (self.y, self.p, self.args, self.ta, self.tb)
        aux_data = {"f":self.f, "f_p":self.f_p, "n_dim":self.n_dim, "n_mesh_point":self.n_mesh_point, 
        "n_par":self.n_par, "n_coeff":self.n_coeff, "n":self.n, "n_colloc_eq":self.n_colloc_eq}

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):

        return cls(aux_data["f"], aux_data["f_p"], *children)
           
jax.tree_util.register_pytree_node(colloc, colloc._tree_flatten, colloc._tree_unflatten)
