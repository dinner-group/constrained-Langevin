import numpy
import jax
import jax.numpy as np
import jax.experimental.sparse
jax.config.update("jax_enable_x64", True)

class LinSolColloc():

    def __init__(self, J, n_dim, n_mesh_point, n_par, n_coeff, n, n_colloc_eq, **kwargs):

        self.J = J
        self.n_dim = n_dim
        self.n_mesh_point = n_mesh_point
        self.n_par = n_par
        self.n_coeff = n_coeff
        self.n = n
        self.n_colloc_eq = n_colloc_eq
        J1, self.L1, self.E1, self.p1 = self.elim1(self.J)
        self.J2, self.L2, self.F2_1, self.F2_2, self.E2, self.p2 = self.elim2(J1)
        self.interval_width = self.n_colloc_point * self.n_dim
        self.block_length = self.interval_width + self.n_dim + self.n_par
        self.block_size = self.block_length * self.interval_width

    @jax.jit
    def elim1(self, J):

        par_eq = J.data[-self.n * self.n_par:].reshape((self.n_par, self.n))

        def loop_body(carry, _):

            i, data, par_eq = carry
            block_start = i * self.block_size
            block = jax.lax.dynamic_slice(data, (block_start,), (self.block_size,)).reshape((self.interval_width, self.block_length))

            lu, _, p = jax.lax.linalg.lu(block[:, self.n_dim:-self.n_dim - self.n_par])
            L = np.identity(lu.shape[0]).at[:, :lu.shape[1]].add(np.tril(lu, k=-1))
            U = np.triu(lu)[:self.interval_width - self.n_dim]
            block = block.at[:, :self.n_dim].set(jax.scipy.linalg.solve_triangular(L, block[p, :self.n_dim], lower=True))
            block = block.at[:, self.n_dim:self.interval_width].set(np.triu(lu))
            block = block.at[:, self.interval_width:].set(jax.scipy.linalg.solve_triangular(L, block[p, self.interval_width:], lower=True))

            data = jax.lax.dynamic_update_slice(data, block.ravel(), (block_start,))

            block_par_eq = jax.lax.dynamic_slice(par_eq, (0, i * self.interval_width), (self.n_par, self.interval_width + self.n_dim))
            elim_coeff = jax.scipy.linalg.solve_triangular(U.T, block_par_eq[:, self.n_dim:-self.n_dim].T, lower=True)
            par_eq = jax.lax.dynamic_update_slice(par_eq, block_par_eq[:, :self.n_dim] - elim_coeff.T@block[:self.interval_width - self.n_dim, :self.n_dim], (0, i * self.interval_width))
            par_eq = jax.lax.dynamic_update_slice(par_eq, np.zeros((self.n_par, self.interval_width - self.n_dim)), (0, i * self.interval_width + self.n_dim))
            par_eq = jax.lax.dynamic_update_slice(par_eq, block_par_eq[:, -self.n_dim:]\
                                                  - elim_coeff.T@block[:self.interval_width - self.n_dim, -self.n_par - self.n_dim:- self.n_par], (0, (i + 1) * self.interval_width))
            par_eq = jax.lax.dynamic_update_slice(par_eq, par_eq[:, -self.n_par:]\
                                                  - elim_coeff.T@block[:self.interval_width - self.n_dim, -self.n_par:], (0, self.n_coeff))

            return (i + 1, data, par_eq), (L, elim_coeff, i * self.interval_width + p)

        result = jax.lax.scan(f=loop_body, init=(0, J.data, par_eq), xs=None, length=self.n_mesh_point)
        _, data, par_eq = result[0]
        L, elim_coeff, p = result[1]
        p = np.arange(J.shape[0]).at[:p.size].set(p.ravel())
        data = data.at[-self.n_par * self.n:].set(par_eq.ravel())

        J.data = data
        return J, L, elim_coeff, p
        
    @jax.jit
    def elim2(self, J):

        offset = self.block_length * self.n_dim
        ind_base = np.mgrid[self.interval_width - self.n_dim:self.interval_width, :self.n_dim].reshape((2, self.n_dim * self.n_dim)).T
        par_eq = J.data[-self.n * self.n_par:].reshape((self.n_par, self.n))

        block1 = J.data[self.block_size - offset:self.block_size].reshape((self.n_dim, self.block_length))

        fill1 = np.zeros((self.n_mesh_point, self.n_dim, self.n_dim))    
        fill1 = fill1.at[0].set(block1[:, :self.n_dim])
        block1 = block1.at[:, :self.n_dim].set(0)

        def loop_body(carry, _):

            i, data, par_eq, fill1, block1 = carry

            block2 = jax.lax.dynamic_slice(data, ((i + 2) * self.block_size - offset,), (offset,)).reshape((self.n_dim, self.block_length))

            lu, _, p = jax.lax.linalg.lu(np.vstack([block1[:, -self.n_dim - self.n_par:-self.n_par], block2[:, :self.n_dim]]))
            L = np.identity(2 * self.n_dim).at[:, :self.n_dim].add(np.tril(lu, k=-1))
            U = np.triu(lu)[:self.n_dim]

            left = jax.scipy.linalg.solve_triangular(L, np.vstack(jax.lax.dynamic_slice(fill1, (i, 0, 0), (2, self.n_dim, self.n_dim)))[p], lower=True)
            fill1 = jax.lax.dynamic_update_slice(fill1, left.reshape((2, self.n_dim, self.n_dim)), (i, 0, 0))

            block1 = block1.at[:, -self.n_dim - self.n_par:-self.n_par].set(U)
            block2 = block2.at[:, :self.n_dim].set(0)

            mid = jax.scipy.linalg.solve_triangular(L, np.pad(block2[:, -self.n_dim - self.n_par:-self.n_par], ((self.n_dim, 0), (0, 0)))[p], lower=True)
            fill2 = mid[:self.n_dim]
            block2 = block2.at[:, -self.n_dim - self.n_par:-self.n_par].set(mid[self.n_dim:])
            elim_coeff = jax.scipy.linalg.solve_triangular(U.T, jax.lax.dynamic_slice(par_eq, (0, (i + 1) * self.interval_width), (self.n_par, self.n_dim)).T, lower=True)
            par_eq = jax.lax.dynamic_update_slice(par_eq, np.zeros((self.n_par, self.n_dim)), (0, (i + 1) * self.interval_width))

            right = jax.scipy.linalg.solve_triangular(L, np.vstack([block1[:, -self.n_par:], block2[:, -self.n_par:]])[p], lower=True)
            block1 = block1.at[:, -self.n_par:].set(right[:self.n_dim])
            block2 = block2.at[:, -self.n_par:].set(right[self.n_dim:])
            par_eq = par_eq.at[:, :self.n_dim].add(-elim_coeff.T@fill1[i])
            par_eq = par_eq.at[:, -self.n_par:].add(-elim_coeff.T@block1[:, -self.n_par:])
            par_eq = jax.lax.dynamic_update_slice(par_eq, jax.lax.dynamic_slice(par_eq, (0, (i + 2) * self.interval_width), (self.n_par, self.n_dim))\
                                                  - elim_coeff.T@fill2, (0, (i + 2) * self.interval_width))

            data = jax.lax.dynamic_update_slice(data, block1.ravel(), ((i + 1) * self.block_size - offset,))

            return (i + 1, data, par_eq, fill1, block2), (L, elim_coeff, p, fill2)

        result = jax.lax.scan(loop_body, init=(0, J.data, par_eq, fill1, block1), xs=None, length=self.n_mesh_point - 1)
        i, data, par_eq, fill1, block1 = result[0]
        L, elim_coeff, p, fill2 = result[1]
        data = jax.lax.dynamic_update_slice(data, block1.ravel(), ((i + 1) * self.block_size - offset,))
        J.data = data
        J.data = J.data.at[-self.n * self.n_par:].set(par_eq.ravel())

        return J, L, fill1, fill2, elim_coeff, p

    @jax.jit
    def solve(self, b):

        b = b[self.p1]

        def loop1(carry, _):

            i, b = carry
            bint = jax.lax.dynamic_slice(b, (i * self.interval_width,), (self.interval_width,))
            rint = jax.scipy.linalg.solve_triangular(self.L1[i], bint, lower=True)
            b = jax.lax.dynamic_update_slice(b, bint, (i * self.interval_width,))
            b = b.at[-self.n_par:].add(-self.E1[i]@bint[:-self.n_dim])

            return (i + 1, b), _

        i, b = jax.lax.scan(loop1, init=(0, b), xs=None, length=b.size)[0]

        def loop2(carry, _):

            i, b, b1 = carry
            b2 = jax.lax.dynamic_slice(b, ((i + 2) * self.interval_width - self.n_dim,), (self.n_dim,))
            bint = jax.scipy.linalg.solve_triangular(self.L2[i], np.vstack([b1, b2])[self.p2[i]], lower=True)
            b = b.at[-self.n_par:].add(-self.E2[i].T@bint[:self.n_dim])
            b = jax.lax.dynamic_update_slice(b, bint[:self.n_dim], ((i + 1) * self.interval_width - self.n_dim,))

            return (i + 1, b, bint[self.n_dim:]), _

        i, b, b1 = jax.lax.scan(loop2, init=(0, b), xs=None, length=self.L2.shape[0])[0]
        b = jax.lax.dynamic_update_slice(b, b1, ((i + 1) * self.interval_width - self.n_dim,))

        return b

    def _tree_flatten(self):

        children = (self.J, self.L1, self.E1, self.p1, self.J2, self.L2, self.F2_1, self.F2_2, self.E2, self.p2)
        aux_data = {"n_dim":self.n_dim, "n_mesh_point":self.n_mesh_point, "n_par":self.n_par, "n_coeff":self.n_coeff, "n":self.n, "n_colloc_eq":self.n_colloc_eq,
                    "interval_width":self.interval_width, "block_length":self.block_length, "block_size":self.block_size}

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):

        return cls(children[0], **aux_data)

jax.tree_util.register_pytree_node(colloc, colloc._tree_flatten, colloc._tree_unflatten)
