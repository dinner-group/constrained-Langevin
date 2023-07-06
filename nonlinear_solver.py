import jax
import jax.numpy as np
import util
from functools import partial
jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def newton(x, resid, jac=None, max_iter=20, tol=1e-9, args=()):

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
    return x, args, np.all(np.abs(dx) < tol)

@partial(jax.jit, static_argnums=(1, 3, 5, 6))
def newton_rattle(x, resid, jac_prev, jac=None, inverse_mass=None, max_iter=20, tol=1e-9, args=()):

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
    return x, args, np.all(np.abs(dx) < tol)

@partial(jax.jit, static_argnums=(1, 3, 5, 6))
def quasi_newton_rattle_symm_1(x, resid, jac_prev, jac=None, inverse_mass=None, max_iter=20, tol=1e-9, args=()):

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

    Q, R = np.linalg.qr(jac_prev_sqrtM.T)

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
    return x, args, np.all(np.abs(dx) < tol)

@partial(jax.jit, static_argnums=(1, 3, 5, 6, 7))
def quasi_newton_rattle(x, resid, jac_prev, jac=None, inverse_mass=None, max_qn_iter=100, max_newton_iter=20, tol=1e-9, args=()):

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
def quasi_newton_rattle_symm_broyden(x, resid, jac_prev, jac=None, inverse_mass=None, max_iter=100, tol=1e-9, args=()):

    if inverse_mass is None:
        jac_prev_sqrtM = jac_prev
    elif len(inverse_mass.shape) == 1:
        sqrtMinv = np.sqrt(inverse_mass)
        jac_prev_sqrtM = jac_prev * sqrtMinv
    else:
        sqrtMinv = jax.scipy.linalg.cholesky(inverse_mass)
        jac_prev_sqrtM = jac_prev@jax.scipy.linalg.cholesky(inverse_mass)

    Q, R = np.linalg.qr(jac_prev_sqrtM.T)

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
    x, step, dx, dx_prev, projection2, contraction_factor = jax.lax.while_loop(cond1, loop1, init)
    return x, args, np.all(np.abs(dx) < tol)

@partial(jax.jit, static_argnums=(1, 3, 5, 6))
def newton_bvp_dense(x, resid, jac_prev, jac, inverse_mass=None, max_iter=20, tol=1e-9, args=()):

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
    return x, args, np.all(np.abs(dx) < tol)

@partial(jax.jit, static_argnums=(1, 3, 5, 6, 7))
def quasi_newton_bvp_dense(x, resid, jac_prev, jac, inverse_mass=None, max_qn_iter=100, max_newton_iter=20, tol=1e-9, args=()):

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
def quasi_newton_bvp_symm(x, resid, jac_prev, jac, inverse_mass=None, max_iter=100, tol=1e-9, args=()):
    
    if inverse_mass is None:
        jac_prev_sqrtM = jac_prev
    elif len(inverse_mass.shape) == 1:
        sqrtMinv = np.sqrt(inverse_mass)
        jac_prev_sqrtM = jac_prev.right_multiply_diag(sqrtMinv)
    else:
        sqrtMinv = jax.scipy.linalg.cholesky(inverse_mass)
        jac_prev_sqrtM = jac_prev@jax.scipy.linalg.cholesky(inverse_mass)

    J_LQ = jac_prev_sqrtM.lq_factor()
    Jk = np.pad(np.vstack(jac_prev_sqrtM.Jk), ((0, jac_prev_sqrtM.n_dim), (0, 0)))
    E = J_LQ.solve_triangular_L(Jk)
    Q1, R1 = np.linalg.qr(np.vstack([np.identity(Jk.shape[1]), E]))

    def lstsq(b):
        w = J_LQ.solve_triangular_L(b)
        out_k = jax.scipy.linalg.solve_triangular(R1, Q1[Jk.shape[1]:].T@w, lower=False)
        u = w - Q1[Jk.shape[1]:]@(Q1[Jk.shape[1]:].T@w)
        t = J_LQ.solve_triangular_R(u)
        out_y = jac_prev_sqrtM.left_multiply(t)[jac_prev_sqrtM.n_par:-1]
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
    return x, args, np.all(np.abs(dx) < tol)

@partial(jax.jit, static_argnums=(1, 3, 5, 6))
def quasi_newton_bvp_symm_broyden(x, resid, jac_prev, jac, inverse_mass=None, max_iter=100, tol=1e-9, args=()):
    
    if inverse_mass is None:
        jac_prev_sqrtM = jac_prev
    elif len(inverse_mass.shape) == 1:
        sqrtMinv = np.sqrt(inverse_mass)
        jac_prev_sqrtM = jac_prev.right_multiply_diag(sqrtMinv)
    else:
        sqrtMinv = jax.scipy.linalg.cholesky(inverse_mass)
        jac_prev_sqrtM = jac_prev@jax.scipy.linalg.cholesky(inverse_mass)

    J_LQ = jac_prev_sqrtM.lq_factor()
    Jk = np.pad(np.vstack(jac_prev_sqrtM.Jk), ((0, jac_prev_sqrtM.n_dim), (0, 0)))
    E = J_LQ.solve_triangular_L(Jk)
    Q1, R1 = np.linalg.qr(np.vstack([np.identity(Jk.shape[1]), E]))

    def lstsq(b):
        w = J_LQ.solve_triangular_L(b)
        out_k = jax.scipy.linalg.solve_triangular(R1, Q1[Jk.shape[1]:].T@w, lower=False)
        u = w - Q1[Jk.shape[1]:]@(Q1[Jk.shape[1]:].T@w)
        t = J_LQ.solve_triangular_R(u)
        out_y = jac_prev_sqrtM.left_multiply(t)[jac_prev_sqrtM.n_par:-1]
        out = np.concatenate([out_k[:jac_prev_sqrtM.n_par], out_y, out_k[-1:]])

        if inverse_mass is not None:
            if len(inverse_mass.shape) == 1:
                out = sqrtMinv * out
            else:
                out = sqrtMinv@out

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
    return x, args, np.all(np.abs(dx) < tol)

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def gauss_newton(x, resid, jac=None, max_iter=20, tol=1e-9, args=()):

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
    return x, args, np.all(np.abs(dx) < tol)

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
