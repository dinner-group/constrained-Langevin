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
    def _compute_resid_interval(self, y, p, colloc_points, sub_points):

        poly_denom = colloc.lagrange_poly_denom(sub_points)

        def loop_body(i, _):

            poly = colloc.lagrange_poly(colloc_points[i], sub_points, y.reshape((self.n_dim, colloc.n_colloc_point + 1), order="F"), poly_denom)
            poly_grad = colloc.lagrange_poly_grad(colloc_points[i], sub_points, y.reshape((self.n_dim, colloc.n_colloc_point + 1), order="F"), poly_denom)

            return i + 1, poly_grad - self.f(colloc_points[i], poly, p, *self.args)

        return jax.lax.scan(f=loop_body, init=0, xs=None, length=colloc.n_colloc_point)[1].ravel(order="C")

    @jax.jit
    def resid(self, y=None, p=None):
        
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
            poly_denom = colloc.lagrange_poly_denom(sub_points)
            y_i = jax.lax.dynamic_slice(y, (interval_start,), (interval_width + self.n_dim,))
            r_i = self._compute_resid_interval(y_i, p, colloc_points, sub_points)

            return i + 1, r_i
            
        return np.concatenate([jax.lax.scan(f=loop_body, init=i, xs=None, length=self.n_mesh_point)[1].ravel(order="C"), 
                               y[self.n_coeff - self.n_dim:self.n_coeff] - y[:self.n_dim], 
                               self.f_p(self.t, y, p, *self.args)])
    
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
            poly_denom = colloc.lagrange_poly_denom(sub_points)
            y_i = jax.lax.dynamic_slice(y, (interval_start,), (interval_width + self.n_dim,))

            Jy_i = jax.jacfwd(self._compute_resid_interval, argnums=0)(y_i, p, colloc_points, sub_points)
            Jp_i = jax.jacfwd(self._compute_resid_interval, argnums=1)(y_i, p, colloc_points, sub_points)

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

        return jax.experimental.sparse.BCOO((data, indices), shape=(self.n, self.n))
 
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

    def _newton_step(self):
        
        r = self.resid()
        self._superLU()
        dx = self.jac_LU.solve(-numpy.asanyarray(r))
        x = np.concatenate([self.y.ravel(order="F"), self.p]) + dx
        self.y = x[:self.y.size].reshape((self.n_dim, self.n_coeff // self.n_dim), order="F")
        self.p = x[self.y.size:]
        self.err = np.max(np.abs(r))
    
    def solve(self, atol=1e-6, maxiter=10):
      
        self.success = False
        self.err = np.max(np.abs(self.resid()))
        self.n_iter = 0
        
        while self.n_iter < maxiter and self.err >= atol:
            self.n_iter += 1
            self._newton_step()
            
        if self.err < atol:
            self.success = True

    def _tree_flatten(self):

        children = (self.y, self.p, self.args, self.ta, self.tb)
        aux_data = {"f":self.f, "f_p":self.f_p, "n_dim":self.n_dim, "n_mesh_point":self.n_mesh_point, 
        "n_par":self.n_par, "n_coeff":self.n_coeff, "n":self.n, "n_colloc_eq":self.n_colloc_eq}

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):

        return cls(aux_data["f"], aux_data["f_p"], *children)
           
jax.tree_util.register_pytree_node(colloc, colloc._tree_flatten, colloc._tree_unflatten)
