import jax
import jax.numpy as np
import numpy
import time
import os
import nonlinear_solver
from functools import partial
jax.config.update("jax_enable_x64", True)

path = os.path.dirname(__file__)

S = np.array(numpy.loadtxt(path + "/S.txt"))
K = np.array(numpy.loadtxt(path + "/K.txt"))
rays = np.array(numpy.loadtxt(path + "/rays.txt"))

cC = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
cA = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 1, 2, 1])
C = np.zeros((S.shape[0], S.shape[1] + S.shape[0]))
C = C.at[:-7, :S.shape[1]].set(S[:-7, :])
C = C.at[-7:-2, :S.shape[1]].set(S[-6:-1, :])
C = C.at[-2, S.shape[1]:].set(cC)
C = C.at[-1, S.shape[1]:].set(cA)
key = jax.random.PRNGKey(time.time_ns())

@jax.jit
def log_rates(q):
    return q[:50] - np.sum(q[50:]*K.T, axis=1)

@jax.jit
def jac_q(q):
    
    z = np.exp(q[:S.shape[1]])
    h = np.exp(-q[S.shape[1]:])
    return ((S*z)@(K.T*h))

@jax.jit
def jac_q_evals(q):
    
    return np.linalg.eigvals(jac_q(q))

jac_evals_grad = jax.jit(jax.jacfwd(jac_q_evals))

@jax.jit
def potential(q):

    E = 0
    
    evals = np.linalg.eigvals(jac_q(q))
    evals = evals[np.argsort(np.abs(evals), kind="stable")[2:]]
    evals = evals[np.argsort(evals.real, kind="stable")]
    diff = evals[1:].real - evals[:-1].real
    E += np.min(diff)
    ind_repeated_real = 1 + np.argsort(-diff, kind="stable")[-1]
    
    E += 10 * np.where(evals[ind_repeated_real].real > 0, 0, evals[ind_repeated_real].real)**2
    E += np.where(np.abs(evals[ind_repeated_real].imag) > 1e-1, 0, 1e-1 - np.abs(evals[ind_repeated_real].imag))**2
    E += np.where(np.abs(evals[ind_repeated_real].imag) < 1e2, 0, 1e2 - np.abs(evals[ind_repeated_real].imag))**2
    
    log_k = log_rates(q)
    E += np.linalg.norm(np.where(q[50:] < -10, q[50:] + 10, 0))**2

    pU_total = np.log(np.exp(q[50:])[np.array([0, 1])].sum())
    E += np.where(pU_total < -2, pU_total + 2, 0)**2

    pT_total = np.log(np.exp(q[50:])[np.array([2, 3])].sum())
    E += np.where(pT_total < -2, pT_total + 2, 0)**2

    pD_total = np.log(np.exp(q[50:])[np.array([4, 5, 8, 9, 12, 13])].sum())
    E += np.where(pD_total < -2, pD_total + 2, 0)**2
    
    pS_total = np.log(np.exp(q[50:])[np.array([6, 7, 10, 11, 14, 15])].sum())
    E += np.where(pS_total < -2, pS_total + 2, 0)**2

    E += np.linalg.norm(np.where(log_k < -20, log_k + 20, 0))**2
    E += np.linalg.norm(np.where(log_k > 22, log_k - 22, 0))**2
            
    return 1e3 * E

@jax.jit
def constraint(q, conserve=np.array([3.5, 0.6])):
    
    c = C@np.exp(q)
    return c.at[-2:].add(-conserve)

@jax.jit
def kinetic(p, Minv=None):

    if Minv is None:
        return p@p / 2
    else:
        return p@Minv@p / 2
    
@partial(jax.jit, static_argnums=(4, 5))
def hamiltonian(q, p, l, inverse_mass, potential, constraint):
    return p@inverse_mass@p / 2 + potential(q) + l@constraint(q)
    
@jax.jit
def propose_p(q, prng_subkey):
    
    p = jax.random.normal(prng_subkey, shape=q.shape)
    u, s, vh = jax.numpy.linalg.svd(jac_cons(q))
    return vh[17:, :].T@(vh[17:, :]@p)

# @jax.jit
# def half2(par, q1, p0, l, dt):

#     jac_cons = jax.jit(jax.jacfwd(cons, argnums=0))
#     p1 = par[:p0.shape[0]]
#     m = par[p0.shape[0]:]
#     return np.concatenate([p0 - p1 - (dt / 2) * (jax.grad(hamiltonian, argnums=0)(q1, p0, l) + m@jac_cons(q1)),
#                            jac_cons(q1)@jax.grad(hamiltonian, argnums=1)(q1, p1, m)])

# jac_half2 = jax.jit(jax.jacfwd(half2, argnums=0))

#@partial(jax.jit, static_argnums=(5, 6, 7, 8))
#def rattle_step1(q0, p0, l0, dt, inverse_mass, potential, constraint, max_newton_iter=10, tol=1e-9):
#    
#    def half1(par, q0, p0, dt):
#        jac_cons = jax.jacfwd(cons, argnums=0)
#        q1 = par[:q0.shape[0]]
#        l = par[q0.shape[0]:]
#        p0_5 = p0 - (dt / 2) * (jax.grad(hamiltonian, argnums=0)(q0, p0, l, inverse_mass, potential, constraint) + l@jac_cons(q0))
#        return np.concatenate([q0 - q1 + (dt / 2) * (jax.grad(hamiltonian, argnums=1)(q0, p0_5, l, inverse_mass, potential, constraint)\
#                                + jax.grad(hamiltonian, argnums=1)(q1, p0_5, l, inverse_mass, potential, constraint)),
#                               constraint(q1)])
#    
#    jac_half1 = jax.jacfwd(half1, argnums=0)
#    
#    def cond(par):    
#        x, step, err = par
#        return (step < max_newton_iter) & (err > tol)
#
#    def body(par):
#        x, step, err = par
#        x = x + np.linalg.solve(jac_half1(x, q0, p0, dt), -half1(x, q0, p0, dt))
#        return x, step + 1, np.linalg.norm(half1(x, q0, p0, dt))
#
#    x0 = np.concatenate([q0, l0])
#    return jax.lax.while_loop(cond, body, (x0, 0, np.linalg.norm(half1(x0, q0, p0, dt))))
#
#@partial(jax.jit, static_argnums=(5, 6))
#def rattle_step2(q1, p0, l, dt, inverse_mass, potential, constraint):
#    
#    jac_cons = jax.jacfwd(cons, argnums=0)
#    jac_C = jac_cons(q1)
#
#    A = np.zeros((p0.size + l.size, p0.size + l.size))
#    A = A.at[:p0.size, :p0.size].set(np.identity(p0.size))
#    A = A.at[:p0.size, p0.size:p0.size + l.size].set((dt / 2) * jac_C.T)
#    A = A.at[p0.size:p0.size + l.size, :p0.size].set(jac_C)
#
#    b = np.zeros(p0.size + l.size)
#    b = b.at[:p0.size].set(p0 - (dt / 2) * jax.grad(H, argnums=0)(q1, p0, l, inverse_mass, potential, constraint))
#    
#    return np.linalg.solve(A, b)

@jax.jit
def cotangency_lhs(jac_constraint, inverse_mass):
    A = np.zeros((jac_constraint.shape[0] + jac_constraint.shape[1], jac_constraint.shape[0] + jac_constraint.shape[1]))
    A = A.at[:jac_constraint.shape[1], :jac_constraint.shape[1]].set(np.identity(jac_constraint.shape[1]))
    A = A.at[:jac_constraint.shape[1], jac_constraint.shape[1]:].set(-jac_constraint.T)
    A = A.at[jac_constraint.shape[1]:, :jac_constraint.shape[1]].set(jac_constraint@inverse_mass)
    return A

@partial(jax.jit, static_argnums=(3, 4))
def rattle_kick(position, momentum, dt, potential, constraint, inverse_mass=None):

    if inverse_mass is None:
        inverse_mass = np.identity(momentum.size)

    jac_constraint = jax.jacfwd(constraint)(position)

    A = cotangency_lhs(jac_constraint, inverse_mass)
    b = np.pad(momentum - dt * jax.grad(potential)(position), (0, jac_constraint.shape[0]))
    x = np.linalg.solve(A, b)

    momentum_new = x[:momentum.size]
    lagrange_multiplier_new = x[momentum.size:]

    return position, momentum_new, lagrange_multiplier_new

@partial(jax.jit, static_argnums=(4, 5, 7, 8))
def rattle_drift(position, momentum, lagrange_multiplier, dt, potential, constraint, inverse_mass=None, max_newton_iter=20, tol=1e-9):
    
    if inverse_mass is None:
        inverse_mass = np.identity(momentum.size)

    def drift_residual(x):
        
        position_new = x[:position.size]
        lagrange_multiplier_new = x[position.size:]
        jac_constraint = jax.jacfwd(constraint)(position_new)
        momentum_new = momentum + lagrange_multiplier_new@jac_constraint
        return np.concatenate([position_new - (position + dt * inverse_mass@momentum_new), constraint(position_new)])

    x = np.concatenate([position, lagrange_multiplier])
    x, success = nonlinear_solver.newton(x, drift_residual, max_iter=max_newton_iter)

    position_new = x[:position.size]
    lagrange_multiplier_new = x[position.size:]
    jac_constraint = jax.jacfwd(constraint)(position_new)
    momentum_new = momentum + lagrange_multiplier_new@jac_constraint

    A = cotangency_lhs(jac_constraint, inverse_mass)
    b = np.pad(momentum_new, (0, jac_constraint.shape[0]))
    x = np.linalg.solve(A, b)

    momentum_new = x[:momentum.size]

    return position_new, momentum_new, lagrange_multiplier_new, success

@partial(jax.jit, static_argnums=(5, 6))
def rattle_noise(position, momentum, dt, friction, prng_key, potential, constraint, inverse_mass=None, temperature=1):
    
    if inverse_mass is None:
        inverse_mass = np.identity(momentum.size)

    drag = np.exp(-friction * dt)
    noise_scale = np.sqrt(temperature * (1 - a**2))

    jac_constraint = jax.jacfwd(constraint)(position_new)
    
    A = cotangency_lhs(jac_constraint, inverse_mass)
    
    key, subkey = jax.random.split(prng_key)
    W = jax.random.normal(key, momentum.shape)
    L = np.linalg.cholesky(inverse_mass)
    W = noise_scale * jax.scipy.linalg.solve_triangular(L, W, lower=True)

    b = np.pad(drag * momentum + W, (0, jac_constraint.shape[0]))
    x = np.linalg.solve(A, b)

    momentum_new = x[:momentum.size]
    lagrange_multiplier_new = x[momentum.size:]

    return position, momentum_new, lagrange_multiplier_new

@partial(jax.jit, static_argnums=(1, 5, 6, 7, 8))
def sample(q0, nsteps, prng_key, dt, inverse_mass, potential, constraint, max_newton_iter=10, tol=1e-9):

    traj = np.empty((nsteps, q0.shape[0]))
    accept = np.zeros(nsteps, dtype=bool)
    h_arr = np.empty(nsteps)

    def body(carry, _):

        qstep, lstep, accept, h_arr, prng_key, i = carry
        prng_key, subkey = jax.random.split(prng_key)
        pstep = propose_p(qstep, subkey)
        h_arr = h_arr.at[i].set(hamiltonian(qstep, pstep, np.zeros(C.shape[0]), inverse_mass, potential, constraint))

        propose_half1, _, _ = rattle_step1(qstep, lstep, pstep, dt, inverse_mass, potential, constraint)

        q1_propose = propose_half1[:qstep.shape[0]]
        l1_propose = propose_half1[qstep.shape[0]:]
        xstep_0_5 = np.concatenate([pstep, np.zeros(C.shape[0])])

        propose_half2 = rattle_step2(q1_propose, pstep, l1_propose, dt, inverse_mass, potential, constraint)

        p1_propose = propose_half2[:pstep.shape[0]]
        m1_propose = propose_half2[pstep.shape[0]:]

        prng_key, subkey = jax.random.split(prng_key)

        metropolis_hastings = hamiltonian(q1_propose, p1_propose, m1_propose, inverse_mass, potential, constraint)\
                            - hamiltonian(qstep, pstep, np.zeros(C.shape[0]), inverse_mass, potential, constraint) < -np.log(jax.random.uniform(subkey))
        accept = accept.at[i].set(metropolis_hastings)
        qstep = np.where(metropolis_hastings, q1_propose, qstep)
        
        return (qstep, lstep, accept, h_arr, prng_key, i + 1), qstep
    
    init = (q0, np.zeros(), accept, h_arr, prng_key, 0)
    return jax.lax.scan(body, init, None, length=nsteps)
