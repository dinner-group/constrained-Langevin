import jax
import jax.numpy as np
import util
import linear_solver
from functools import partial
jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def newton(x, resid, jac=None, max_iter=20, tol=1e-9, args=(), **kwargs):

    if jac is None:
        jac = jax.jacfwd(resid)
    
    def cond(carry):
        x, step, dx = carry
        return (step < max_iter) & np.any(np.abs(dx) > tol)
    
    def loop_body(carry):
        x, step, dx = carry
        dx = np.linalg.solve(jac(x, *args), -resid(x, *args))
        return x + dx, step + 1, dx
    
    init = (x, 0, np.full_like(x, np.inf))
    x, n_iter, dx = jax.lax.while_loop(cond, loop_body, init)
    return x, args, np.all(np.abs(dx) < tol), n_iter

@partial(jax.jit, static_argnums=(1, 3, 5, 6))
def newton_rattle(x, resid, jac_prev, jac=None, inverse_mass=None, max_iter=20, tol=1e-9, args=(), **kwargs):

    if jac_prev is None:
        return gauss_newton(x, resid, jac, max_iter, tol, args)

    if jac is None:
        jac = jax.jacfwd(resid)

    if inverse_mass is None:
        jac_prevM = jac_prev
    elif len(inverse_mass.shape) == 1:
        jac_prevM = jac_prev * inverse_mass
    else:
        jac_prevM = jac_prev@inverse_mass

    def cond(carry):
        x, step, dx = carry
        return (step < max_iter) & np.any(np.abs(dx) > tol)

    def loop_body(carry):
        x, step, dx = carry
        J = jac(x, *args)
        dx = jac_prevM.T@np.linalg.solve(J@jac_prevM.T, -resid(x, *args))
        return x + dx, step + 1, dx

    init = (x, 0, np.full_like(x, np.inf))
    x, n_iter, dx = jax.lax.while_loop(cond, loop_body, init)
    return x, args, np.all(np.abs(dx) < tol), n_iter

@partial(jax.jit, static_argnums=(1, 3, 5, 6))
def quasi_newton_rattle_symm_1(x, resid, jac_prev, jac=None, inverse_mass=None, max_iter=20, tol=1e-9, J_and_factor=None, args=(), **kwargs):

    if jac_prev is None:
        return gauss_newton(x, resid, jac, max_iter, tol, args)

    if jac is None:
        jac = jax.jacfwd(resid)

    if inverse_mass is None:
        jac_prev_sqrtM = jac_prev
    elif len(inverse_mass.shape) == 1:
        sqrtMinv = np.sqrt(inverse_mass)
        jac_prev_sqrtM = jac_prev * sqrtMinv
    else:
        sqrtMinv = jax.scipy.linalg.cholesky(inverse_mass)
        jac_prev_sqrtM = jac_prev@jax.scipy.linalg.cholesky(inverse_mass)

    if J_and_factor is None:
        Q, R = np.linalg.qr(jac_prev_sqrtM.T)
    else:
        Q, R = J_and_factor[1]

    def cond(carry):
        x, step, dx = carry
        return (step < max_iter) & np.any(np.abs(dx) > tol)

    def loop_body(carry):
        x, step, dx = carry
        dx = Q@jax.scipy.linalg.solve_triangular(R.T, -resid(x, *args), lower=True)

        if inverse_mass is not None:
            if len(inverse_mass.shape) == 1:
                dx = sqrtMinv * dx
            else:
                dx = sqrtMinv@dx

        return x + dx, step + 1, dx

    init = (x, 0, np.full_like(x, np.inf))
    x, n_iter, dx = jax.lax.while_loop(cond, loop_body, init)
    return x, args, np.all(np.abs(dx) < tol), n_iter

@partial(jax.jit, static_argnums=(1, 3, 5, 6, 7))
def quasi_newton_rattle(x, resid, jac_prev, jac=None, inverse_mass=None, max_qn_iter=100, max_newton_iter=20, tol=1e-9, args=(), **kwargs):

    if jac is None:
        jac = jax.jacfwd(resid)

    if inverse_mass is None:
        jac_prevM = jac_prev
    elif len(inverse_mass.shape) == 1:
        jac_prevM = jac_prev * inverse_mass
    else:
        jac_prevM = jac_prev@inverse_mass

    J = jac(x, *args)
    lu = jax.scipy.linalg.lu_factor(J@jac_prevM.T)
    dx = jac_prevM.T@jax.scipy.linalg.lu_solve(lu, -resid(x, *args))
    x = x + dx
    v = jac_prevM.T@jax.scipy.linalg.lu_solve(lu, -resid(x, *args))
    projection2 = v.T@dx / (dx.T@dx)
    contraction_factor = np.sqrt(v.T@v / (dx.T@dx))
    dx_prev = dx
    dx = v / (1 - projection2)

    def cond1(carry):
        x, step, dx, dx_prev, projection2, contraction_factor = carry
        return (step < max_qn_iter) & np.any(np.abs(dx) > tol) & (contraction_factor < 0.5)

    def loop1(carry):
        x, step, dx, dx_prev, projection2, contraction_factor = carry
        x = x + dx
        v = jac_prevM.T@jax.scipy.linalg.lu_solve(lu, -resid(x, *args))
        projection1 = v.T@dx_prev / (dx_prev.T@dx_prev)
        v = v + projection1 * dx
        projection2 = v.T@dx / (dx.T@dx)
        contraction_factor = np.sqrt(v.T@v / (dx.T@dx))
        dx_prev = dx
        dx = v / (1 - projection2)
        return x, step + 1, dx, dx_prev, projection2, contraction_factor

    x, step, dx, dx_prev, projection2, contraction_factor = jax.lax.while_loop(cond1, loop1, init_val=(x, 1, dx, dx_prev, projection2, contraction_factor))

    return jax.lax.cond(np.all(np.abs(dx) < tol), lambda x: (x, args, True), lambda x:newton_rattle(x, resid, jac_prev, jac, inverse_mass, max_newton_iter, tol, args), x)

@partial(jax.jit, static_argnums=(1, 3, 5, 6))
def quasi_newton_rattle_symm_broyden(x, resid, jac_prev, jac=None, inverse_mass=None, max_iter=100, tol=1e-9, J_and_factor=None, args=(), **kwargs):

    if inverse_mass is None:
        jac_prev_sqrtM = jac_prev
    elif len(inverse_mass.shape) == 1:
        sqrtMinv = np.sqrt(inverse_mass)
        jac_prev_sqrtM = jac_prev * sqrtMinv
    else:
        sqrtMinv = jax.scipy.linalg.cholesky(inverse_mass)
        jac_prev_sqrtM = jac_prev@jax.scipy.linalg.cholesky(inverse_mass)
    
    if J_and_factor is None:
        Q, R = np.linalg.qr(jac_prev_sqrtM.T)
    else:
        Q, R = J_and_factor[1]

    def lstsq(b):
        x = Q@jax.scipy.linalg.solve_triangular(R.T, b, lower=True)
        if inverse_mass is not None:
            if len(inverse_mass.shape) == 1:
                x = sqrtMinv * dx
            else:
                x = sqrtMinv@dx
        return x

    dx = lstsq(-resid(x, *args))
    x = x + dx
    #v = Q@jax.scipy.linalg.solve_triangular(R.T, -resid(x, *args), lower=True)
    v = lstsq(-resid(x, *args))
    projection2 = v.T@dx / (dx.T@dx)
    contraction_factor = np.sqrt(v.T@v / (dx.T@dx))
    dx_prev = dx
    dx = v / (1 - projection2)

    def cond1(carry):
        x, step, dx, dx_prev, projection2, contraction_factor = carry
        return (step < max_iter) & np.any(np.abs(dx) > tol)

    def loop1(carry):
        x, step, dx, dx_prev, projection2, contraction_factor = carry
        x = x + dx
        v = lstsq(-resid(x, *args))
        projection1 = v.T@dx_prev / (dx_prev.T@dx_prev)
        v = v + projection1 * dx
        projection2 = v.T@dx / (dx.T@dx)
        contraction_factor = np.sqrt(v.T@v / (dx.T@dx))
        dx_prev = dx
        dx = v / (1 - projection2)
        return x, step + 1, dx, dx_prev, projection2, contraction_factor

    init = (x, 1, dx, dx_prev, projection2, contraction_factor)
    x, n_iter, dx, dx_prev, projection2, contraction_factor = jax.lax.while_loop(cond1, loop1, init)
    return x, args, np.all(np.abs(dx) < tol), n_iter

@partial(jax.jit, static_argnums=(1, 3, 5, 6))
def newton_bvp_dense(x, resid, jac_prev, jac, inverse_mass=None, max_iter=20, tol=1e-9, args=(), **kwargs):

    if inverse_mass is None:
        jac_prevM = jac_prev
    else:
        jac_prevM = jac_prev.right_multiply_diag(inverse_mass)

    def cond(carry):
        x, step, dx = carry
        return (step < max_iter) & np.any(np.abs(dx) > tol)

    def loop_body(carry):
        x, step, dx = carry
        J = jac(x, *args)
        JJT = util.BVPJac.multiply_transpose(J, jac_prevM)
        dx = jac_prevM.left_multiply(np.linalg.solve(JJT, -resid(x, *args)))
        return x + dx, step + 1, dx

    init = (x, 0, np.full_like(x, np.inf))
    x, n_iter, dx = jax.lax.while_loop(cond, loop_body, init)
    return x, args, np.all(np.abs(dx) < tol), n_iter

@partial(jax.jit, static_argnums=(1, 3, 5, 6, 7))
def quasi_newton_bvp_dense(x, resid, jac_prev, jac, inverse_mass=None, max_qn_iter=100, max_newton_iter=20, tol=1e-9, args=(), **kwargs):

    if inverse_mass is None:
        jac_prevM = jac_prev
    else:
        jac_prevM = jac_prev.right_multiply_diag(inverse_mass)

    J = jac(x, *args)
    JJT = util.BVPJac.multiply_transpose(J, jac_prevM)
    lu = jax.scipy.linalg.lu_factor(JJT)
    dx = jac_prevM.left_multiply(jax.scipy.linalg.lu_solve(lu, -resid(x, *args)))
    x = x + dx
    v = jac_prevM.left_multiply(jax.scipy.linalg.lu_solve(lu, -resid(x, *args)))
    projection2 = v.T@dx / (dx@dx)
    contraction_factor = np.sqrt(v.T@v / (dx.T@dx))
    dx_prev = dx
    dx = v / (1 - projection2)

    def cond1(carry):
        x, step, dx, dx_prev, projection2, contraction_factor = carry
        return (step < max_qn_iter) & np.any(np.abs(dx) > tol) & (contraction_factor < 0.5)

    def loop1(carry):
        x, step, dx, dx_prev, projection2, contraction_factor = carry
        x = x + dx
        v = jac_prevM.left_multiply(jax.scipy.linalg.lu_solve(lu, -resid(x, *args)))
        projection1 = v.T@dx_prev / (dx_prev@dx_prev)
        v = v + projection1 * dx
        projection2 = v.T@dx / (dx@dx)
        contraction_factor = np.sqrt(v.T@v / (dx.T@dx))
        dx_prev = dx
        dx = v / (1 - projection2)
        return x, step + 1, dx, dx_prev, projection2, contraction_factor

    x, n_iter, dx, dx_prev, projection2, contraction_factor = jax.lax.while_loop(cond1, loop1, init_val=(x, 1, dx, dx_prev, projection2, contraction_factor))
    return jax.lax.cond(np.all(np.abs(dx) < tol), lambda x: (x, args, True), lambda x:newton_bvp_dense(x, resid, jac_prev, jac, inverse_mass, max_newton_iter, tol, args), x)

@partial(jax.jit, static_argnums=(1, 3, 5, 6))
def quasi_newton_bvp_symm(x, resid, jac_prev, jac, inverse_mass=None, max_iter=100, tol=1e-9, J_and_factor=None, args=(), **kwargs):
    
    if inverse_mass is None:
        jac_prev_sqrtM = jac_prev
    elif len(inverse_mass.shape) == 1:
        sqrtMinv = np.sqrt(inverse_mass)
        jac_prev_sqrtM = jac_prev.right_multiply_diag(sqrtMinv)
    else:
        sqrtMinv = jax.scipy.linalg.cholesky(inverse_mass)
        jac_prev_sqrtM = jac_prev@jax.scipy.linalg.cholesky(inverse_mass)

    if J_and_factor is None:
        J_LQ = jac_prev_sqrtM.lq_factor()
    else:
        J_LQ = J_and_factor[1]

    Jk = np.pad(np.vstack(jac_prev_sqrtM.Jk), ((0, jac_prev_sqrtM.shape[0] - jac_prev_sqrtM.Jk.shape[0]), (0, 0)))
    E = J_LQ.solve_triangular_L(Jk)
    Q1, R1 = np.linalg.qr(np.vstack([np.identity(Jk.shape[1]), E]))

    def lstsq(b):
        w = J_LQ.solve_triangular_L(b)
        out_k = jax.scipy.linalg.solve_triangular(R1, Q1[Jk.shape[1]:].T@w, lower=False)
        u = w - E@out_k
        out_y = J_LQ.Q_right_multiply(u)
        out = np.concatenate([out_k[:jac_prev_sqrtM.n_par], out_y, out_k[-1:]])

        if inverse_mass is not None:
            if len(inverse_mass.shape) == 1:
                out = sqrtMinv * out
            else:
                out = sqrtMinv@out

        return out

    def cond(carry):
        x, step, dx = carry
        return (step < max_iter) & np.any(np.abs(dx) > tol)

    def loop_body(carry):
        x, step, dx = carry
        dx = lstsq(-resid(x, *args))
        return x + dx, step + 1, dx

    init = (x, 0, np.full_like(x, np.inf))
    x, n_iter, dx = jax.lax.while_loop(cond, loop_body, init)
    return x, args, np.all(np.abs(dx) < tol), n_iter

@partial(jax.jit, static_argnums=(1, 3, 5, 6))
def quasi_newton_bvp_symm_broyden(x, resid, jac_prev, jac, inverse_mass=None, max_iter=100, tol=1e-9, J_and_factor=None, args=(), **kwargs):
    
    sqrtMinv = None
    if inverse_mass is None:
        jac_prev_sqrtM = jac_prev
    elif len(inverse_mass.shape) == 1:
        sqrtMinv = np.sqrt(inverse_mass)
        jac_prev_sqrtM = jac_prev.right_multiply_diag(sqrtMinv)
    else:
        sqrtMinv = jax.scipy.linalg.cholesky(inverse_mass)
        jac_prev_sqrtM = jac_prev@jax.scipy.linalg.cholesky(inverse_mass)
        return quasi_newton_rattle_symm_broyden(x, resid, jac_prev.todense(), None, inverse_mass, max_iter, tol, J_and_factor, args, **kwargs)

    if J_and_factor is None:
        J_LQ = jac_prev_sqrtM.lq_factor()
    else:
        J_LQ = J_and_factor[1]

    Jk_factor = linear_solver.factor_bvpjac_k(jac_prev, J_LQ)

    dx, _ = linear_solver.lstsq_bvpjac(jac_prev, -resid(x, *args), J_LQ, Jk_factor, sqrtMinv)
    x = x + dx
    v, _ = linear_solver.lstsq_bvpjac(jac_prev, -resid(x, *args), J_LQ, Jk_factor, sqrtMinv)
    projection2 = v.T@dx / (dx.T@dx)
    contraction_factor = np.sqrt(v.T@v / (dx.T@dx))
    dx_prev = dx
    dx = v / (1 - projection2)

    def cond(carry):
        x, step, dx, dx_prev, projection2, contraction_factor = carry
        return (step < max_iter) & np.any(np.abs(dx) > tol)

    def loop_body(carry):
        x, step, dx, dx_prev, projection2, contraction_factor = carry
        x = x + dx
        v, _ = linear_solver.lstsq_bvpjac(jac_prev, -resid(x, *args), J_LQ, Jk_factor, sqrtMinv)
        projection1 = v.T@dx_prev / (dx_prev.T@dx_prev)
        v = v + projection1 * dx
        projection2 = v.T@dx / (dx.T@dx)
        contraction_factor = np.sqrt(v.T@v / (dx.T@dx))
        dx_prev = dx
        dx = v / (1 - projection2)
        return x, step + 1, dx, dx_prev, projection2, contraction_factor

    init = (x, 1, dx, dx_prev, projection2, contraction_factor)
    x, n_iter, dx, dx_prev, projection2, contraction_factor = jax.lax.while_loop(cond, loop_body, init)
    return x, args, np.all(np.abs(dx) < tol), n_iter

@partial(jax.jit, static_argnums=(1, 3, 5, 6))
def quasi_newton_bvp_multi_eqn_shared_k_symm_broyden(x, resid, jac_prev, jac, inverse_mass=None, max_iter=100, tol=1e-9, J_and_factor=None, args=(), **kwargs):
    
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

        return quasi_newton_rattle_symm_broyden(x, resid, J_dense, None, inverse_mass, max_iter, None, args, **kwargs)

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
            out = sqrtMinv * out

        return out

    dx = lstsq(-resid(x, *args))
    x = x + dx
    v = lstsq(-resid(x, *args))
    projection2 = v.T@dx / (dx.T@dx)
    contraction_factor = np.sqrt(v.T@v / (dx.T@dx))
    dx_prev = dx
    dx = v / (1 - projection2)

    def cond(carry):
        x, step, dx, dx_prev, projection2, contraction_factor = carry
        return (step < max_iter) & np.any(np.abs(dx) > tol)

    def loop_body(carry):
        x, step, dx, dx_prev, projection2, contraction_factor = carry
        x = x + dx
        v = lstsq(-resid(x, *args))
        projection1 = v.T@dx_prev / (dx_prev.T@dx_prev)
        v = v + projection1 * dx
        projection2 = v.T@dx / (dx.T@dx)
        contraction_factor = np.sqrt(v.T@v / (dx.T@dx))
        dx_prev = dx
        dx = v / (1 - projection2)
        return x, step + 1, dx, dx_prev, projection2, contraction_factor

    init = (x, 1, dx, dx_prev, projection2, contraction_factor)
    x, n_iter, dx, dx_prev, projection2, contraction_factor = jax.lax.while_loop(cond, loop_body, init)
    return x, args, np.all(np.abs(dx) < tol), n_iter

@partial(jax.jit, static_argnums=(1, 3, 5, 6))
def quasi_newton_bvp_symm_broyden_resid(x, resid, jac_prev, jac, inverse_mass=None, max_iter=100, tol=1e-9, max_memory=20, J_and_factor=None, args=(), **kwargs):
   
    return
    #TODO incomplete
    if inverse_mass is None:
        jac_prev_sqrtM = jac_prev
    elif len(inverse_mass.shape) == 1:
        sqrtMinv = np.sqrt(inverse_mass)
        jac_prev_sqrtM = jac_prev.right_multiply_diag(sqrtMinv)
    else:
        sqrtMinv = jax.scipy.linalg.cholesky(inverse_mass)
        jac_prev_sqrtM = jac_prev@jax.scipy.linalg.cholesky(inverse_mass)

    if J_and_factor is None:
        J_LQ = jac_prev_sqrtM.lq_factor()
    else:
        J_LQ = J_and_factor[1]

    Jk = np.pad(np.vstack(jac_prev_sqrtM.Jk), ((0, jac_prev_sqrtM.shape[0] - jac_prev_sqrtM.Jk.shape[0]), (0, 0)))
    E = J_LQ.solve_triangular_L(Jk)
    Q1, R1 = np.linalg.qr(np.vstack([np.identity(Jk.shape[1]), E]))

    def lstsq(b):
        w = J_LQ.solve_triangular_L(b)
        out_k = jax.scipy.linalg.solve_triangular(R1, Q1[Jk.shape[1]:].T@w, lower=False)
        u = w - E@out_k
        out_y = J_LQ.Q_right_multiply(u)
        out = np.concatenate([out_k[:jac_prev_sqrtM.n_par], out_y, out_k[-1:]])

        if inverse_mass is not None:
            if len(inverse_mass.shape) == 1:
                out = sqrtMinv * out
            else:
                out = sqrtMinv@out

        return out

    r = resid(x, *args)
    dx = lstsq(-r)
    x = x + dx
    step_size = 1
    head_index = 0
    memory = np.zeros((max_memory, r.size))
    memory = memory.at[head_index].set(r)

    def cond(carry):
        x, step, memory, head_index, contraction_factor, step_size = carry
        r = memory[head_index]
        return (step < max_iter) & np.any(np.abs(r) > tol) & (contraction_factor < 1.5)

    def loop_body(carry):
        x, step, memory, head_index, contraction_factor, step_size = carry
        x = x + dx
        r_prev = memory[head_index]
        head_index = (head_index + 1) % max_memory
        r = resid(x, *args)
        memory[head_index] = r
        contraction_factor = np.sqrt(np.max(np.abs(r)) / np.max(np.abs(r_prev)))
        step_size = step_size / (1 - 2 * contraction_factor)
        dr = r - r_prev
        dr = dr / dr.T@dr
        v = (1 - dr.T@r) * r

        return x, step + 1, dx, dx_prev, projection2, contraction_factor

    def broyden_update(head_index, _):
        head_index, v, memory = carry

        return ((head_index - 1) % max_memory, v, memory), _

    init = (x, 1, dx, dx_prev, projection2, contraction_factor)
    x, n_iter, dx, dx_prev, projection2, contraction_factor = jax.lax.while_loop(cond, loop_body, init)
    return x, args, np.all(np.abs(dx) < tol), n_iter

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def gauss_newton(x, resid, jac=None, max_iter=20, tol=1e-9, args=(), **kwargs):

    if jac is None:
        jac = jax.jacfwd(resid)

    def cond(carry):
        x, step, dx = carry
        return (step < max_iter) & np.any(np.abs(dx) > tol)

    def loop_body(carry):
        x, step, dx = carry
        J = jac(x, *args)
        Q, R = jax.scipy.linalg.qr(J.T, mode="economic")
        dx = Q@jax.scipy.linalg.solve_triangular(R.T, -resid(x, *args), lower=True)
        return x + dx, step + 1, dx

    init = (x, 0, np.full_like(x, np.inf))
    x, n_iter, dx = jax.lax.while_loop(cond, loop_body, init)
    return x, args, np.all(np.abs(dx) < tol), n_iter

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def newton_sparse(x, resid, jac, max_iter=20, tol=1e-9):

    def cond(carry):
        x, step, dx = carry
        return (step < max_iter) & np.any(np.abs(dx) > tol)

    def loop_body(carry):
        x, step, dx = carry
        J = jac(x)
        dx = jax.experimental.sparse.linalg.spsolve(J.data, J.indices, J.indptr, -resid(x))
        return x + dx, step + 1, dx

    init = (x, 0, np.full_like(x, np.inf))
    x, n_iter, dx = jax.lax.while_loop(cond, loop_body, init)
    return x, np.all(np.abs(dx) < tol)

@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def affine_covariant_newton(x, resid, jac=None, max_iter=100, min_damping_factor=1e-5, tol=1e-9):
    
    if jac is None:
        jac = jax.jacfwd(resid)
        
    def loop_inner(carry):
        x, x_guess, damping_factor, contraction_factor, trust_factor, dx_prev, dx_init, error, jac_lu = carry
        x_guess = x + damping_factor * dx_init
        error = jax.scipy.linalg.lu_solve(jac_lu, -resid(x_guess))
        contraction_factor = np.linalg.norm(error) / np.linalg.norm(dx_init)
        trust_factor = np.linalg.norm(dx_init) * damping_factor**2 / (2 * np.linalg.norm(error - (1 - damping_factor) * dx_init))
        damping_factor = np.where(contraction_factor >= 1, np.minimum(trust_factor, damping_factor / 2), damping_factor)
        damping_factor = np.where((contraction_factor < 1) * (trust_factor > 4 * damping_factor) * (trust_factor < 1), trust_factor, damping_factor)
        return x, x_guess, damping_factor, contraction_factor, trust_factor, dx_prev, dx_init, error, jac_lu
        
    def cond_inner(carry):
        x, x_guess, damping_factor, contraction_factor, trust_factor, dx_prev, dx_init, error, jac_lu = carry
        return (damping_factor > min_damping_factor) & (np.linalg.norm(error) > tol) & ((trust_factor > 4 * damping_factor) * (trust_factor < 1) + (contraction_factor >= 1))
        
    def loop_outer(carry):
        x, x_guess, step, damping_factor, contraction_factor, trust_factor, dx_prev, dx_init, error, jac_lu = carry
        x, x_guess, damping_factor, contraction_factor, trust_factor, dx_prev, dx_init, error, jac_lu\
            = jax.lax.while_loop(cond_inner, loop_inner, (x, x_guess, damping_factor, contraction_factor, trust_factor, dx_prev, dx_init, error, jac_lu))
        
        x = x_guess
        jac_lu = jax.scipy.linalg.lu_factor(jac(x))
        dx_prev = dx_init
        dx_init = jax.scipy.linalg.lu_solve(jac_lu, -resid(x))
        damping_factor = np.minimum(1, damping_factor * np.linalg.norm(dx_prev) * np.linalg.norm(error) / (np.linalg.norm((error - dx_init)) * np.linalg.norm(dx_init)))
        contraction_factor = 1.
        
        return x, x + damping_factor * dx_init, step + 1, damping_factor, contraction_factor, trust_factor, dx_prev, dx_init, error, jac_lu
        
    def cond_outer(carry):
        x, x_guess, step, damping_factor, contraction_factor, trust_factor, dx_prev, dx_init, error, jac_lu = carry
        err = np.linalg.norm(dx_init)
        return (step < max_iter) & (err > tol)
    
    jac_lu = jac_lu = jax.scipy.linalg.lu_factor(jac(x))
    dx_init = jax.scipy.linalg.lu_solve(jac_lu, -resid(x))
    init = (x, x, 0, 1., 1., 0., dx_init, dx_init, dx_init, jac_lu)
    x, x_guess, step, damping_factor, contraction_factor, trust_factor, dx_prev, dx_init, error, jac_lu = jax.lax.while_loop(cond_outer, loop_outer, init)
    
    return x, np.linalg.norm(dx_init) < tol
