import jax
import jax.numpy as np
import util
from functools import partial
jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def newton(x, resid, jac=None, max_iter=20, tol=1e-9, *args):

    if jac is None:
        jac = jax.jacfwd(resid)
    
    def cond(carry):
        x, step, dx = carry
        return (step < max_iter) & np.any(np.abs(dx) > tol)
    
    def loop_body(carry):
        x, step, dx = carry
        dx = np.linalg.solve(jac(x), -resid(x))
        return x + dx, step + 1, dx
    
    init = (x, 0, np.full_like(x, np.inf))
    x, n_iter, dx = jax.lax.while_loop(cond, loop_body, init)
    return x, np.all(np.abs(dx) < tol)

@partial(jax.jit, static_argnums=(1, 3, 4, 5))
def newton_rattle(x, resid, jac_prev, jac=None, max_iter=20, tol=1e-9):

    if jac_prev is None:
        return gauss_newton(x, resid,jac, max_iter, tol)

    if jac is None:
        jac = jax.jacfwd(resid)

    def cond(carry):
        x, step, dx = carry
        return (step < max_iter) & np.any(np.abs(dx) > tol)

    def loop_body(carry):
        x, step, dx = carry
        J = jac(x)
        dx = jac_prev.T@np.linalg.solve(J@jac_prev.T, -resid(x))
        return x + dx, step + 1, dx

    init = (x, 0, np.full_like(x, np.inf))
    x, n_iter, dx = jax.lax.while_loop(cond, loop_body, init)
    return x, np.all(np.abs(dx) < tol)

@partial(jax.jit, static_argnums=(1, 3, 4, 5, 6))
def quasi_newton_rattle(x, resid, jac_prev, jac=None, max_qn_iter=100, max_newton_iter=20, tol=1e-9):

    if jac is None:
        jac = jax.jacfwd(resid)

    J = jac(x)
    lu = jax.scipy.linalg.lu_factor(J@jac_prev.T)
    dx = J.T@jax.scipy.linalg.lu_solve(lu, -resid(x))
    x = x + dx
    v = J.T@jax.scipy.linalg.lu_solve(lu, -resid(x))
    projection2 = v.T@dx / (dx@dx)
    contraction_factor = np.sqrt(v.T@v / (dx.T@dx))
    dx_prev = dx
    dx = v / (1 - projection2)

    def cond1(carry):
        x, step, dx_prev, dx, projection2, contraction_factor = carry
        return (step < max_qn_iter) & np.any(np.abs(dx) > tol) & (contraction_factor < 0.5)

    def loop1(carry):
        x, step, dx, dx_prev, projection2, contraction_factor = carry
        x = x + dx
        v = J.T@jax.scipy.linalg.lu_solve(lu, -resid(x))
        projection1 = v.T@dx_prev / (dx_prev@dx_prev)
        v = v + projection1 * dx
        projection2 = v.T@dx / (dx@dx)
        contraction_factor = np.sqrt(v.T@v / (dx.T@dx))
        dx_prev = dx
        dx = v / (1 - projection2)
        return x, step + 1, dx, dx_prev, projection2, contraction_factor

    def cond2(carry):
        x, step, dx = carry
        return (step < max_newton_iter) & np.any(np.abs(dx) > tol)

    def loop2(carry):
        x, step, dx = carry
        J = jac(x)
        dx = J.T@np.linalg.solve(J@jac_prev.T, -resid(x))
        return x + dx, step + 1, dx

    x, step, dx, dx_prev, projection2, contraction_factor = jax.lax.while_loop(cond1, loop1, init_val=(x, 1, dx, dx_prev, projection2, contraction_factor))
    #x, n_iter, dl = jax.lax.while_loop(cond2, loop2, init_val=(x, 0, dl))
    #return x, np.all(np.abs(dl) < tol)

    return jax.lax.cond(np.all(np.abs(dx) < tol), lambda x: (x, True), lambda x:newton_rattle(x, resid, jac_prev, jac, max_newton_iter, tol), x)

@partial(jax.jit, static_argnums=(1, 3, 4, 5))
def newton_bvp_dense(x, resid, jac_prev, jac, max_iter=20, tol=1e-9):

    def cond(carry):
        x, step, dx = carry
        return (step < max_iter) & np.any(np.abs(dx) > tol)

    def loop_body(carry):
        x, step, dx = carry
        J = jac(x)
        JJT = util.BVPJac.multiply_transpose(J, jac_prev)
        dx = jac_prev.vjp(np.linalg.solve(JJT, -resid(x)))
        return x + dx, step + 1, dx

    init = (x, 0, np.full_like(x, np.inf))
    x, n_iter, dx = jax.lax.while_loop(cond, loop_body, init)
    return x, np.all(np.abs(dx) < tol)

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def gauss_newton(x, resid, jac=None, max_iter=20, tol=1e-9):

    if jac is None:
        jac = jax.jacfwd(resid)

    def cond(carry):
        x, step, dx = carry
        return (step < max_iter) & np.any(np.abs(dx) > tol)

    def loop_body(carry):
        x, step, dx = carry
        J = jac(x)
        Q, R = jax.scipy.linalg.qr(J.T, mode="economic")
        dx = Q@jax.scipy.linalg.solve_triangular(R.T, -resid(x), lower=True)
        return x + dx, step + 1, dx

    init = (x, 0, np.full_like(x, np.inf))
    x, n_iter, dx = jax.lax.while_loop(cond, loop_body, init)
    return x, np.all(np.abs(dx) < tol)

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
