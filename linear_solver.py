import numpy
import jax
import jax.numpy as np
import jax.experimental.sparse
from functools import partial
jax.config.update("jax_enable_x64", True)

@jax.jit
def qr_lstsq_rattle(J, b, J_and_factor=None, inverse_mass=None):

    if J_and_factor is None:
        if inverse_mass is None:
            J_and_factor = (J, np.linalg.qr(J.T), None)
        elif len(inverse_mass.shape) == 1:
            sqrtMinv = np.sqrt(inverse_mass)
            J_and_factor = (J, np.linalg.qr((sqrtMinv * J).T), sqrtMinv)
        else:
            sqrtMinv = jax.scipy.linalg.cholesky(inverse_mass)
            J_and_factor = (J, np.linalg.qr((J@sqrtMinv).T), sqrtMinv)

    Q, R = J_and_factor[1]
    sqrtMinv = J_and_factor[2]

    if inverse_mass is None:
        sqrtMinvb = b
    elif len(sqrtMinv.shape) == 1:
        sqrtMinvb = b * sqrtMinv
    else:
        sqrtMinvb = sqrtMinv@b

    b_proj_coeff = Q.T@sqrtMinvb
    b_proj = Q@b_proj_coeff

    if sqrtMinv is not None:
        if len(sqrtMinv.shape) == 1:
            b_proj = b_proj / sqrtMinv
        else:
            b_proj = jax.scipy.linalg.solve_triangular(sqrtMinv, b_proj, lower=False)

    return b - b_proj, b_proj_coeff, J_and_factor

@jax.jit
def qr_lstsq_rattle_bvp_dense(J, b, J_and_factor=None, inverse_mass=None):

    if J_and_factor is None:
        if inverse_mass is None:
            J_and_factor = (J, np.linalg.qr(J.todense().T)[1])
        elif len(inverse_mass.shape) == 1:
            sqrtMinv = np.sqrt(inverse_mass)
            J_and_factor = (J, np.linalg.qr(J.right_multiply_diag(sqrtMinv).todense().T)[1])
        else:
            sqrtMinv = jax.scipy.linalg.cholesky(inverse_mass)
            J_and_factor = (J, np.linalg.qr(J.right_multiply(sqrtMinv).T)[1])
            
    if inverse_mass is None:
        Minvb = b
    elif len(inverse_mass.shape) == 1:
        Minvb = b * inverse_mass
    else:
        Minvb = inverse_mass@b

    R = J_and_factor[1]
    b_proj_coeff = jax.scipy.linalg.cho_solve((R, False), J.right_multiply(Minvb))

    return b - J.left_multiply(b_proj_coeff), b_proj_coeff, J_and_factor

@jax.jit
def qr_lstsq_rattle_bvp(J, b, J_and_factor=None, inverse_mass=None):

    if inverse_mass is None:
        JsqrtMinv = J
    elif len(inverse_mass.shape) == 1:
        sqrtMinv = np.sqrt(inverse_mass)
        JsqrtMinv = J.right_multiply_diag(sqrtMinv)
    else:
        return qr_lstsq_rattle_bvp_dense(J, b, J_and_factor, inverse_mass)

    if J_and_factor is None:
        J_and_factor = (J, JsqrtMinv.lq_factor())

    J_LQ = J_and_factor[1]
    Jk = np.pad(np.vstack(JsqrtMinv.Jk), ((0, JsqrtMinv.shape[0] - JsqrtMinv.Jk.shape[0]), (0, 0)))
    E = J_LQ.solve_triangular_L(Jk)
    Q1, R1 = np.linalg.qr(np.vstack([np.identity(Jk.shape[1]), E]))

    def lstsq(b):
        w = J_LQ.solve_triangular_L(b)
        out_k = jax.scipy.linalg.solve_triangular(R1, Q1[Jk.shape[1]:].T@w, lower=False)
        u = w - E@out_k
        out_y = J_LQ.Q_right_multiply(u)
        out = np.concatenate([out_k[:JsqrtMinv.n_par], out_y, out_k[-1:]])

        if inverse_mass is not None:
            out = out / sqrtMinv

        return out, u
    
    if inverse_mass is not None:
        b = sqrtMinv * b

    JMinvb = JsqrtMinv.right_multiply(b)
    out = lstsq(JMinvb)
    return b - out[0], out[1], J_and_factor

@jax.jit
def qr_lstsq_rattle_bvp_multi_eqn_shared_k(J, b, J_and_factor=None, inverse_mass=None):

    if inverse_mass is None:
        JsqrtMinv = J
    elif len(inverse_mass.shape) == 1:
        sqrtMinv = np.sqrt(inverse_mass)
        JsqrtMinv = list(J)
        JsqrtMinv[0] = J[0].right_multiply_diag(sqrtMinv[:J[0].shape[1]])
        for i in range(1, len(J)):
            JsqrtMinv[i] = J[i].right_multiply_diag(JsqrtMinv[J[i - 1].shape[1]:J[i].shape[1]])
        JsqrtMinv = tuple(JsqrtMinv)
    else:
        dense_shape = (sum(J_i.shape[0] for J_i in J), sum(J_i.shape[1] for J_i in J) - sum(J_i.n_par for J_i in J[1:]))
        J_dense = np.zeros(dense_shape)
        J_dense = J_dense.at[:J[0].shape[0], :J[0].shape[1]].set(J[0].todense())
        ind_start = (0, J[0].n_par)
        for i in range(1, len(J)):
            ind_stop = (ind[0] + J[i - 1].shape[0], ind[1] + J[i - 1].shape[1] - J[i - 1].n_par)
            J_dense = J_dense.at[ind_start[0]:ind_stop[0], ind_start[1]:ind_stop[1]].set(J[i].todense()[:, J[i].n_par:])
            ind_start = ind_stop
        return qr_lstsq_rattle(J_dense, b, None, inverse_mass)

    if J_and_factor is None:
        J_and_factor = (J, tuple(JsqrtMinv_i.lq_factor() for JsqrtMinv_i in JsqrtMinv))

    J_LQ = J_and_factor[1]
    row_indices = [0 for _ in range(len(J) + 1)]

    for i in range(len(J)):
        row_indices[i + 1] = row_indices[i] + J[i].shape[0]

    col_indices_k = [0 for _ in range(len(J) + 1)]
    col_indices_k[0] = J[0].n_par

    for i in range(len(J)):
        col_indices_k[i + 1] = col_indices_k[i] + J[i].Jk.shape[2] - J[i].n_par

    Jk = np.vstack(np.pad(np.vstack(JsqrtMinv[i].Jk[:, :, :JsqrtMinv[i].n_par]), ((0, JsqrtMinv[i].shape[0] - JsqrtMinv[i].Jk.shape[0] * JsqrtMinv[i].Jk.shape[1]), (0, 0))) for i in range(len(J)))
    Jk = np.pad(Jk, ((0, 0), (0, sum([J_i.Jk.shape[2] - J_i.n_par for J_i in J]))))

    for i in range(len(J)):
        Jk_i = np.vstack(JsqrtMinv[i].Jk[:, :, JsqrtMinv[i].Jk.shape[2] - JsqrtMinv[i].shape[0]:])
        Jk = Jk.at[row_indices[i]:row_indices[i] + Jk_i.shape[0], col_indices_k[i]:col_indices_k[i + 1]].set(Jk_i[:, J[0].n_par:])

    E = np.vstack([J_LQ[i].solve_triangular_L(Jk[row_indices[i]:row_indices[i + 1]]) for i in range(len(J))])
    Q1, R1 = np.linalg.qr(np.vstack([np.identity(Jk.shape[1]), E]))

    def lstsq(b):
        w = np.concatenate([J_LQ[i].solve_triangular_L(b[row_indices[i]:row_indices[i + 1]]) for i in range(len(J))])
        out_k = jax.scipy.linalg.solve_triangular(R1, Q1[-w.size:].T@w)
        u = w - E@out_k
        out_y = [J_LQ[i].Q_right_multiply(u[row_indices[i]:row_indices[i + 1]]) for i in range(len(J))]
        out = np.concatenate([out_k[:J[0].n_par]] + sum(([out_y[i], out_k[col_indices_k[i]:col_indices_k[i + 1]]] for i in range(len(J))), []))

        if inverse_mass is not None:
            out = out / sqrtMinv

        return out, u

    row_indices_b = [0 for _ in range(len(J) + 1)]
    row_indices_b[0] = J[0].n_par
    
    for i in range(len(J)):
        row_indices_b[i + 1] = row_indices_b[i] + J[i].shape[1] - J[i].n_par

    if inverse_mass is not None:
        b = sqrtMinv * b

    JMinvb = np.concatenate([JsqrtMinv[i].right_multiply(np.concatenate([b[:J[0].n_par], b[row_indices_b[i]:row_indices_b[i + 1]]])) for i in range(len(J))])
    out = lstsq(JMinvb)
    return b - out[0], out[1], J_and_factor 

class LinSolColloc():

    def __init__(self, J, n_dim, n_mesh_point, n_colloc_point, n_par, n_coeff, **kwargs):

        self.J = J
        self.n_dim = n_dim
        self.n_mesh_point = n_mesh_point
        self.n_colloc_point = n_colloc_point
        self.n_par = n_par
        self.n_coeff = n_coeff
        self.n = self.n_coeff + self.n_par
        self.n_colloc_eq = self.n_dim * self.n_mesh_point * self.n_colloc_point
        J1, self.L1, self.E1, self.p1 = LinSolColloc.elim1(self.J, self.n_dim, self.n_mesh_point, self.n_colloc_point, self.n_par, self.n_coeff)
        self.J2, self.L2, self.F2_1, self.F2_2, self.E2, self.p2 = LinSolColloc.elim2(J1, self.n_dim, self.n_mesh_point, self.n_colloc_point, self.n_par, self.n_coeff)
        self.interval_width = self.n_colloc_point * self.n_dim
        self.block_length = self.interval_width + self.n_dim + self.n_par
        self.block_size = self.block_length * self.interval_width

    @partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
    def elim1(J, n_dim, n_mesh_point, n_colloc_point, n_par, n_coeff):

        n = n_coeff + n_par
        n_colloc_eq = n_dim * n_mesh_point * n_colloc_point
        interval_width = n_colloc_point * n_dim
        block_length = interval_width + n_dim + n_par
        block_size = block_length * interval_width
        par_eq = J.data[-n * n_par:].reshape((n_par, n))

        def loop_body(carry, _):

            i, data, par_eq = carry
            block_start = i * block_size
            block = jax.lax.dynamic_slice(data, (block_start,), (block_size,)).reshape((interval_width, block_length))

            lu, _, p = jax.lax.linalg.lu(block[:, n_dim:-n_dim - n_par])
            L = np.identity(lu.shape[0]).at[:, :lu.shape[1]].add(np.tril(lu, k=-1))
            U = np.triu(lu)[:interval_width - n_dim]
            block = block.at[:, :n_dim].set(jax.scipy.linalg.solve_triangular(L, block[p, :n_dim], lower=True))
            block = block.at[:, n_dim:interval_width].set(np.triu(lu))
            block = block.at[:, interval_width:].set(jax.scipy.linalg.solve_triangular(L, block[p, interval_width:], lower=True))

            data = jax.lax.dynamic_update_slice(data, block.ravel(), (block_start,))

            block_par_eq = jax.lax.dynamic_slice(par_eq, (0, i * interval_width), (n_par, interval_width + n_dim))
            elim_coeff = jax.scipy.linalg.solve_triangular(U.T, block_par_eq[:, n_dim:-n_dim].T, lower=True)
            par_eq = jax.lax.dynamic_update_slice(par_eq, block_par_eq[:, :n_dim] - elim_coeff.T@block[:interval_width - n_dim, :n_dim], (0, i * interval_width))
            par_eq = jax.lax.dynamic_update_slice(par_eq, np.zeros((n_par, interval_width - n_dim)), (0, i * interval_width + n_dim))
            par_eq = jax.lax.dynamic_update_slice(par_eq, block_par_eq[:, -n_dim:]\
                                                  - elim_coeff.T@block[:interval_width - n_dim, -n_par - n_dim:- n_par], (0, (i + 1) * interval_width))
            par_eq = jax.lax.dynamic_update_slice(par_eq, par_eq[:, -n_par:]\
                                                  - elim_coeff.T@block[:interval_width - n_dim, -n_par:], (0, n_coeff))

            return (i + 1, data, par_eq), (L, elim_coeff, i * interval_width + p)

        result = jax.lax.scan(f=loop_body, init=(0, J.data, par_eq), xs=None, length=n_mesh_point)
        _, data, par_eq = result[0]
        L, elim_coeff, p = result[1]
        p = np.arange(J.shape[0]).at[:p.size].set(p.ravel())
        data = data.at[-n_par * n:].set(par_eq.ravel())

        J.data = data
        return J, L, elim_coeff, p
        
    @partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
    def elim2(J, n_dim, n_mesh_point, n_colloc_point, n_par, n_coeff):

        n = n_coeff + n_par
        n_colloc_eq = n_dim * n_mesh_point * n_colloc_point
        interval_width = n_colloc_point * n_dim
        block_length = interval_width + n_dim + n_par
        block_size = block_length * interval_width
        offset = block_length * n_dim
        ind_base = np.mgrid[interval_width - n_dim:interval_width, :n_dim].reshape((2, n_dim * n_dim)).T
        par_eq = J.data[-n * n_par:].reshape((n_par, n))

        block1 = J.data[block_size - offset:block_size].reshape((n_dim, block_length))

        fill1 = np.zeros((n_mesh_point, n_dim, n_dim))    
        fill1 = fill1.at[0].set(block1[:, :n_dim])
        block1 = block1.at[:, :n_dim].set(0)

        def loop_body(carry, _):

            i, data, par_eq, fill1, block1 = carry

            block2 = jax.lax.dynamic_slice(data, ((i + 2) * block_size - offset,), (offset,)).reshape((n_dim, block_length))

            lu, _, p = jax.lax.linalg.lu(np.vstack([block1[:, -n_dim - n_par:-n_par], block2[:, :n_dim]]))
            L = np.identity(2 * n_dim).at[:, :n_dim].add(np.tril(lu, k=-1))
            U = np.triu(lu)[:n_dim]

            left = jax.scipy.linalg.solve_triangular(L, np.vstack(jax.lax.dynamic_slice(fill1, (i, 0, 0), (2, n_dim, n_dim)))[p], lower=True)
            fill1 = jax.lax.dynamic_update_slice(fill1, left.reshape((2, n_dim, n_dim)), (i, 0, 0))

            block1 = block1.at[:, -n_dim - n_par:-n_par].set(U)
            block2 = block2.at[:, :n_dim].set(0)

            mid = jax.scipy.linalg.solve_triangular(L, np.pad(block2[:, -n_dim - n_par:-n_par], ((n_dim, 0), (0, 0)))[p], lower=True)
            fill2 = mid[:n_dim]
            block2 = block2.at[:, -n_dim - n_par:-n_par].set(mid[n_dim:])
            elim_coeff = jax.scipy.linalg.solve_triangular(U.T, jax.lax.dynamic_slice(par_eq, (0, (i + 1) * interval_width), (n_par, n_dim)).T, lower=True)
            par_eq = jax.lax.dynamic_update_slice(par_eq, np.zeros((n_par, n_dim)), (0, (i + 1) * interval_width))

            right = jax.scipy.linalg.solve_triangular(L, np.vstack([block1[:, -n_par:], block2[:, -n_par:]])[p], lower=True)
            block1 = block1.at[:, -n_par:].set(right[:n_dim])
            block2 = block2.at[:, -n_par:].set(right[n_dim:])
            par_eq = par_eq.at[:, :n_dim].add(-elim_coeff.T@fill1[i])
            par_eq = par_eq.at[:, -n_par:].add(-elim_coeff.T@block1[:, -n_par:])
            par_eq = jax.lax.dynamic_update_slice(par_eq, jax.lax.dynamic_slice(par_eq, (0, (i + 2) * interval_width), (n_par, n_dim))\
                                                  - elim_coeff.T@fill2, (0, (i + 2) * interval_width))

            data = jax.lax.dynamic_update_slice(data, block1.ravel(), ((i + 1) * block_size - offset,))

            return (i + 1, data, par_eq, fill1, block2), (L, elim_coeff, p, fill2)

        result = jax.lax.scan(loop_body, init=(0, J.data, par_eq, fill1, block1), xs=None, length=n_mesh_point - 1)
        i, data, par_eq, fill1, block1 = result[0]
        L, elim_coeff, p, fill2 = result[1]
        data = jax.lax.dynamic_update_slice(data, block1.ravel(), ((i + 1) * block_size - offset,))
        J.data = data
        J.data = J.data.at[-n * n_par:].set(par_eq.ravel())

        return J, L, fill1, fill2, elim_coeff, p

    @jax.jit
    def solve(self, b):

        b = b[self.p1]

        def loop1(carry, _):

            i, b = carry
            bint = jax.lax.dynamic_slice(b, (i * self.interval_width,), (self.interval_width,))
            bint = jax.scipy.linalg.solve_triangular(self.L1[i], bint, lower=True)
            b = jax.lax.dynamic_update_slice(b, bint, (i * self.interval_width,))
            b = b.at[-self.n_par:].add(-self.E1[i].T@bint[:-self.n_dim])

            return (i + 1, b), _

        i, b = jax.lax.scan(loop1, init=(0, b), xs=None, length=self.L1.shape[0])[0]

        def loop2(carry, _):

            i, b, b1 = carry
            b2 = jax.lax.dynamic_slice(b, ((i + 2) * self.interval_width - self.n_dim,), (self.n_dim,))
            bint = jax.scipy.linalg.solve_triangular(self.L2[i], np.concatenate([b1, b2])[self.p2[i]], lower=True)
            b = b.at[-self.n_par:].add(-self.E2[i].T@bint[:self.n_dim])
            b = jax.lax.dynamic_update_slice(b, bint[:self.n_dim], ((i + 1) * self.interval_width - self.n_dim,))

            return (i + 1, b, bint[self.n_dim:]), _

        i, b, b1 = jax.lax.scan(loop2, init=(0, b, b[self.interval_width - self.n_dim:self.interval_width]), xs=None, length=self.L2.shape[0])[0]
        b = jax.lax.dynamic_update_slice(b, b1, ((i + 1) * self.interval_width - self.n_dim,))

        return b

    def _tree_flatten(self):

        children = (self.J, self.L1, self.E1, self.p1, self.J2, self.L2, self.F2_1, self.F2_2, self.E2, self.p2)
        aux_data = {"n_dim":self.n_dim, "n_mesh_point":self.n_mesh_point, "n_colloc_point":self.n_colloc_point, "n_par":self.n_par, "n_coeff":self.n_coeff, "n":self.n, "n_colloc_eq":self.n_colloc_eq,
                    "interval_width":self.interval_width, "block_length":self.block_length, "block_size":self.block_size}

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):

        return cls(children[0], **aux_data)

jax.tree_util.register_pytree_node(LinSolColloc, LinSolColloc._tree_flatten, LinSolColloc._tree_unflatten)
