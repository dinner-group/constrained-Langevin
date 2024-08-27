import numpy
import jax
import jax.numpy as np
import jax.experimental.sparse
from functools import partial
jax.config.update("jax_enable_x64", True)

@jax.jit
def lq_ortho_proj(J, b, J_and_factor=None, inverse_mass=None):

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
def lq_ortho_proj_bvp_dense(J, b, J_and_factor=None, inverse_mass=None):

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
def factor_bvpjac_k(J, J_LQ):

    Jk = np.pad(J.Jk, ((0, J.shape[0] - J.Jk.shape[0]), (0, 0)))
    E = J_LQ.solve_triangular_L(Jk)
    Q_k, R_k = np.linalg.qr(np.vstack([np.identity(Jk.shape[1]), E]))
    return E, Q_k, R_k

@jax.jit
def lq_lstsq_bvp(J, b, J_LQ=None, Jk_factor=None, sqrtMinv=None):

    if J_LQ is None:
        J_LQ = J.lq_factor()
    if Jk_factor is None:
        Jk_factor = factor_bvpjac_k(J, J_LQ)

    E, Q_k, R_k = Jk_factor
    w = J_LQ.solve_triangular_L(b)
    out_k = jax.scipy.linalg.solve_triangular(R_k, Q_k[-w.shape[0]:].T@w, lower=False)
    u = w - E@out_k
    out_y = J_LQ.Q_multiply(u)
    out = np.concatenate([out_k[:J.n_par], out_y, out_k[J.n_par:]])

    if sqrtMinv is not None:
        if len(sqrtMinv.shape) == 1:
            out = out / sqrtMinv
        else:
            out = jax.scipy.linalg.solve_triangular(sqrtMinv, out, lower=False)

    return out, u

@jax.jit
def lq_ortho_proj_bvp(J, b, J_and_factor=None, inverse_mass=None, return_parallel_component=True):

    if inverse_mass is None:
        sqrtMinv = None
        JsqrtMinv = J
    elif len(inverse_mass.shape) == 1:
        sqrtMinv = np.sqrt(inverse_mass)
        JsqrtMinv = J.right_multiply_diag(sqrtMinv)
    else:
        return lq_ortho_proj_bvp_dense(J, b, J_and_factor, inverse_mass)

    if J_and_factor is None:
        J_and_factor = (J, JsqrtMinv.lq_factor())

    J_LQ = J_and_factor[1]
    Jk_factor = factor_bvpjac_k(J, J_LQ)

    if inverse_mass is not None:
        b = sqrtMinv * b

    JMinvb = JsqrtMinv.right_multiply(b)
    out = lq_lstsq_bvp(J, JMinvb, J_LQ, Jk_factor, sqrtMinv)
    projection = np.where(return_parallel_component, b - out[0], out[0])
    return projection, out[1], J_and_factor

@jax.jit
def factor_bvpjac_k_multi_eqn_shared_k(J, J_LQ):

    row_indices = [0 for _ in range(len(J) + 1)]

    for i in range(len(J)):
        row_indices[i + 1] = row_indices[i] + J[i].shape[0]

    col_indices_k = [0 for _ in range(len(J) + 1)]
    col_indices_k[0] = J[0].n_par

    for i in range(len(J)):
        col_indices_k[i + 1] = col_indices_k[i] + J[i].Jk.shape[2] - J[i].n_par

    Jk = np.vstack([np.pad(J[i].Jk[:, :J[i].n_par], ((0, J[i].shape[0] - J[i].Jk.shape[0]), (0, 0))) for i in range(len(J))])
    Jk = np.pad(Jk, ((0, 0), (0, sum([J_i.Jk.shape[1] - J_i.n_par for J_i in J]))))

    for i in range(len(J)):
        Jk = Jk.at[row_indices[i]:row_indices[i] + J[i].Jk.shape[0], col_indices_k[i]:col_indices_k[i + 1]].set(J[i].Jk[:, J[i].n_par:])

    E = np.vstack([J_LQ[i].solve_triangular_L(Jk[row_indices[i]:row_indices[i + 1]]) for i in range(len(J))])
    Q_k, R_k = np.linalg.qr(np.vstack([np.identity(Jk.shape[1]), E]))

    return E, Q_k, R_k

@jax.jit
def lq_lstsq_bvp_multi_eqn_shared_k(J, b, J_LQ=None, Jk_factor=None, sqrtMinv=None):

    if J_LQ is None:
        J_LQ = tuple(JsqrtMinv_i.lq_factor() for JsqrtMinv_i in JsqrtMinv)
    if Jk_factor is None:
        Jk_factor = factor_bvpjac_k_multi_eqn_shared_k(J, J_LQ)

    E, Q_k, R_k = Jk_factor

    row_indices = [0 for _ in range(len(J) + 1)]

    for i in range(len(J)):
        row_indices[i + 1] = row_indices[i] + J[i].shape[0]

    col_indices_k = [0 for _ in range(len(J) + 1)]
    col_indices_k[0] = J[0].n_par

    for i in range(len(J)):
        col_indices_k[i + 1] = col_indices_k[i] + J[i].Jk.shape[1] - J[i].n_par

    w = np.concatenate([J_LQ[i].solve_triangular_L(b[row_indices[i]:row_indices[i + 1]]) for i in range(len(J))])
    out_k = jax.scipy.linalg.solve_triangular(R_k, Q_k[-w.size:].T@w)
    u = w - E@out_k
    out_y = [J_LQ[i].Q_multiply(u[row_indices[i]:row_indices[i + 1]]) for i in range(len(J))]
    out = np.concatenate([out_k[:J[0].n_par]] + sum(([out_y[i], out_k[col_indices_k[i]:col_indices_k[i + 1]]] for i in range(len(J))), []))

    if sqrtMinv is not None:
        if len(sqrtMinv.shape) == 1:
            out = out / sqrtMinv
        else:
            out = jax.scipy.linalg.solve_triangular(sqrtMinv, out, lower=False)

    return out, u

@jax.jit
def lq_ortho_proj_bvp_multi_eqn_shared_k(J, b, J_and_factor=None, inverse_mass=None):

    if inverse_mass is None:
        sqrtMinv = None
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
        return lq_ortho_proj(J_dense, b, None, inverse_mass)

    if J_and_factor is None:
        J_and_factor = (J, tuple(JsqrtMinv_i.lq_factor() for JsqrtMinv_i in JsqrtMinv))

    J_LQ = J_and_factor[1]
    Jk_factor = factor_bvpjac_k_multi_eqn_shared_k(J, J_LQ)

    row_indices_b = [0 for _ in range(len(J) + 1)]
    row_indices_b[0] = J[0].n_par
    
    for i in range(len(J)):
        row_indices_b[i + 1] = row_indices_b[i] + J[i].shape[1] - J[i].n_par

    if inverse_mass is not None:
        b = sqrtMinv * b

    JMinvb = np.concatenate([JsqrtMinv[i].right_multiply(np.concatenate([b[:J[0].n_par], b[row_indices_b[i]:row_indices_b[i + 1]]])) for i in range(len(J))])
    out = lq_lstsq_bvp_multi_eqn_shared_k(J, JMinvb, J_LQ, Jk_factor, sqrtMinv)
    return b - out[0], out[1], J_and_factor 

@jax.jit
def factor_bvpjac_k_multi_shared_k_1(J, J_LQ):

    E = np.vstack([J_LQ[i].solve_triangular_L(np.pad(J[i].Jk, ((0, J[i].shape[0] - J[i].Jk.shape[0]), (0, 0)))) for i in range(len(J))])
    Q_k, R_k = np.linalg.qr(np.vstack([np.identity(E.shape[1]), E]))
    return E, Q_k, R_k

@jax.jit
def lq_lstsq_bvp_multi_shared_k_1(J, b, J_LQ=None, Jk_factor=None):

    if J_LQ is None:
        J_LQ = tuple(J_i.lq_factor() for J_i in J)
    if Jk_factor is None:
        Jk_factor = factor_bvpjac_k_multi_shared_k_1(J, J_LQ)

    E, Q_k, R_k = Jk_factor
    row_indices = [0 for _ in range(len(J) + 1)]

    for i in range(len(J)):
        row_indices[i + 1] = row_indices[i] + J[i].shape[0]

    col_indices_k = [0 for _ in range(len(J) + 1)]
    col_indices_k[0] = J[0].n_par

    for i in range(len(J)):
        col_indices_k[i + 1] = col_indices_k[i] + J[i].shape[1] - J[i].n_par

    w = np.concatenate([J_LQ[i].solve_triangular_L(b[row_indices[i]:row_indices[i + 1]]) for i in range(len(J))])
    out_k = jax.scipy.linalg.solve_triangular(R_k, Q_k[-w.size:].T@w)
    u = w - E@out_k
    out_y = [J_LQ[i].Q_multiply(u[row_indices[i]:row_indices[i + 1]]) for i in range(len(J))]
    out = np.concatenate([out_k] + out_y)

    return out, u

@jax.jit
def lq_ortho_proj_bvp_multi_shared_k_1(J, b, J_and_factor=None, inverse_mass=None):

    if J_and_factor is None:
        J_and_factor = (J, tuple(J_i.lq_factor() for J_i in J))

    J_LQ = J_and_factor[1]
    Jk_factor = factor_bvpjac_k_multi_shared_k_1(J, J_LQ)

    row_indices_b = [0 for _ in range(len(J) + 1)]
    row_indices_b[0] = J[0].n_par
    
    for i in range(len(J)):
        row_indices_b[i + 1] = row_indices_b[i] + J[i].shape[1] - J[i].n_par

    Jb = np.concatenate([J[i].right_multiply(np.concatenate([b[:row_indices_b[0]], b[row_indices_b[i]:row_indices_b[i + 1]]])) for i in range(len(J))])
    out = lq_lstsq_bvp_multi_shared_k_1(J, Jb, J_LQ, Jk_factor)

    return b - out[0], out[1], J_and_factor 

@jax.jit
def low_rank_spd_factor(U):
    
    Q, R = np.linalg.qr(U)
    T = (R@R.T).at[np.diag_indices(R.shape[0])].add(1)
    L = np.linalg.cholesky(T)
    X = L.at[np.diag_indices(L.shape[0])].add(-1)
    
    return Q, X
