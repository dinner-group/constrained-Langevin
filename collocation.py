import jax
import jax.numpy as np
import scipy.sparse
import sparsejac.sparsejac as sparsejac
from functools import partial
jax.config.update("jax_enable_x64", True)

class colloc:

    gauss_points = (1 + np.array([-0.906179845938664, -0.538469310105683, 0.0, +0.538469310105683, +0.906179845938664])) / 2
    n_colloc_point = 5
    n_bc = 2

    def __init__(self, f, f_p, y0, p0, xa=0, xb=1):

        self.xa = xa
        self.xb = xb
        self.f = f
        self.f_p = f_p
        self.xa = xa
        self.xb = xb
        self.y = y0
        self.p = p0
        self.n_dim = y0.shape[0]
        self.n_mesh_point = y0.shape[1]
        self.n_par = p0.shape[0]
        self.n_coeff = self.n_dim * (self.n_mesh_point * self.n_colloc_point + 1)
        self.n = self.n_coeff + self.n_par
        self.n_colloc_eq = self.n_dim * self.n_mesh_point * self.n_colloc_point
        self.jac_sparsity = self.compute_jac_sparsity()
        self.success = False


    @jax.jit
    def lagrange_poly_denom(points):

        poly_denom_arr = np.identity(n_colloc_point + 1)

        for i in range(points.shape[0]):
            poly_denom_arr = poly_denom_arr.at[i, :].add(points[i] - points)

        poly_denom = np.prod(poly_denom_arr, axis=1)
        return poly_denom
        
    @jax.jit
    def lagrange_poly(x, points, coeff, poly_denom=None):

        if poly_denom is None:
            poly_denom = lagrange_poly_denom(points)

        poly_numer_arr = np.tile(x - points, (n_colloc_point + 1, 1))
        poly_numer_arr = poly_numer_arr.at[np.diag_indices(n_colloc_point + 1)].set(1)
        poly_numer = np.prod(poly_numer_arr, axis=1)

        return np.sum((poly_numer / poly_denom) * coeff, axis=1)

    @jax.jit
    def lagrange_poly_grad(x, points, coeff, poly_denom=None):

        return jax.jacfwd(lagrange_poly, argnums=0)(x, points, coeff, poly_denom)
    
    def compute_jac_sparsity(self):
    
        mesh_point = np.arange(self.n_mesh_point)
        interval_width = self.n_colloc_point * self.n_dim

        block = np.mgrid[:interval_width, :interval_width + self.n_dim].reshape((2, (self.n_dim * self.n_colloc_point) * self.n_dim * (self.n_colloc_point + 1))).T

        def loop_blocks(_, i):

            return _, block + i * interval_width

        indices = np.vstack(jax.lax.scan(loop_blocks, init=0, xs=mesh_point)[1])

        indices = np.vstack([indices, 
                             np.mgrid[:self.n, self.n_coeff:self.n].reshape((2, self.n * self.n_par)).T, 
                             np.mgrid[self.n_colloc_eq + self.n_bc:self.n, :self.n].reshape((2, self.n * self.n_par)).T, 
                             np.mgrid[self.n_colloc_eq:self.n_colloc_eq + self.n_bc, :self.n_dim].reshape((2, self.n_dim * self.n_bc)).T, 
                             np.mgrid[self.n_colloc_eq:self.n_colloc_eq + self.n_bc, self.n_coeff - self.n_dim:self.n_coeff].reshape((2, self.n_bc * self.n_dim)).T])

        data = np.ones(indices.shape[0])

        return jax.experimental.sparse.BCOO((data, indices), shape=(self.n, self.n))
    
    @jax.jit
    def resid(self, x):
        
        mesh_points = np.linspace(self.xa, self.xb, n_mesh_point + 1)
        i = 0
        
        def loop_outer(i, _):

            j = 0
            sub_points = np.linspace(mesh_points[i], mesh_points[i + 1], self.n_colloc_point + 1)
            colloc_points = mesh_points[i] + self.gauss_points * (mesh_points[i + 1] - mesh_points[i])
            interval_start = self.n_dim * i * self.n_colloc_point
            interval_width = (self.n_colloc_point + 1) * self.n_dim
            interval_end = self.n_dim * i * self.n_colloc_point + (self.n_colloc_point + 1) * self.n_dim
            poly_denom = lagrange_poly_denom(sub_points)

            def loop_inner(j, _):

                poly = lagrange_poly(colloc_points[j], sub_points, jax.lax.dynamic_slice(y, (interval_start,), (interval_width,)).reshape((self.n_dim, self.n_colloc_point + 1), order="F"),
                                     poly_denom)
                deriv = lagrange_poly_deriv(colloc_points[j], sub_points, jax.lax.dynamic_slice(y, (interval_start,), (interval_width,)).reshape((self.n_dim, self.n_colloc_point + 1), order="F"),
                                            poly_denom)
                return j + 1, deriv - x[self.n_coeff] * self.f(colloc_points[j], poly)

            return i + 1, np.ravel(jax.lax.scan(f=loop_inner, init=j, xs=None, length=self.n_colloc_point)[1])
            
        return np.concatenate([np.ravel(jax.lax.scan(f=loop_outer, init=i, xs=None, length=self.n_mesh_point)[1]), 
                               x[self.n_coeff - self.n_dim:self.n_coeff] - x[:self.n_dim], 
                               self.f_p(x[:, 0], x[:, -1])])
    
    @jax.jit
    def _jac(self, x):
        
        return sparsejac.jacfwd(self.resid, self.jac_sparsity, argnums=0)(x)
    
    def jac(self, x):
        
        J = self._jac(x)
        return scipy.sparse.csc_matrix((J.data, J.indices.T))
    
    def newton_step(self):
        
        x = np.concatenate([self.y.ravel(order="F"), self.p])
        r = self.resid(x)
        J = self.jac(x)
        dx = scipy.sparse.linalg.spsolve(J, -r)
        x = x + dx
        self.y = x[:self.y.size].reshape((self.n_dim, self.n_mesh_point), order="F")
        self.p = x[self.y.size:]
        self.err = np.linalg.norm(r)
    
    def solve(self, rtol=1e-4, maxiter=10):
        
        self.err = self.resid(np.concatenate([self.y.ravel(order="F"), self.p])) 
        
        while i < maxiter and self.err >= rtol:
            self.step()
            
        if self.err < rtol:
            self.success = True

    def _tree_flatten(self):

        children = (self.y, self.p, self.xa, self.xb)
        aux_data = {"f":self.f, "f_p":self.f_p, "n_dim":self.n_dim, "n_mesh_point":self.n_mesh_point, 
        "n_par":self.n_par, "n_coeff":self.n_coeff, "n":self.n, "n_colloc_eq":self.n_colloc_eq}

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):

        return cls(aux_data["f"], aux_data["f_p"], *children)
            
