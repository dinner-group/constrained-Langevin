import jax
import jax.numpy as np
import numpy
import time
import os
from functools import partial
jax.config.update("jax_enable_x64", True)

path = os.path.dirname(__file__)

S = np.array(numpy.loadtxt(path + "/S.txt"))
K = np.array(numpy.loadtxt(path + "/K.txt"))
rays = np.array(numpy.loadtxt(path + "/rays.txt"))
q0 = np.array(numpy.loadtxt(path + "/q0.txt"))
q0 = q0.at[np.arange(50)[np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=bool)]].add(-5)

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
    E += np.linalg.norm(np.where(q[50:] < -20, q[50:] + 20, 0))**2
    E += np.linalg.norm(np.where(log_k < -20, log_k + 20, 0))**2
    E += np.linalg.norm(np.where(log_k > 22, log_k - 22, 0))**2
            
    return 1e3 * E

@jax.jit
def cons(q, conserve=np.array([3.5, 0.6])):
    
    c = C@np.exp(q)
    return c.at[-2:].add(-conserve)

jac_cons = jax.jit(jax.jacfwd(cons, argnums=0))

@jax.jit
def kinetic(p, Minv=None):

    if Minv is None:
        return np.linalg.norm(p)**2 / 2
    else:
        return p@Minv@p / 2
    
def make_hamiltonian(T, U, Cns):   
    return jax.jit(lambda q, p, l:T(p) + U(q) + l@Cns(q))
    
@jax.jit
def propose_p(q, prng_subkey):
    
    p = jax.random.normal(prng_subkey, shape=q0.shape)
    u, s, vh = jax.numpy.linalg.svd(jac_cons(q))
    return vh[17:, :].T@(vh[17:, :]@p)
	
H = make_hamiltonian(kinetic, lambda x:0, cons)
H1 = make_hamiltonian(kinetic, potential, cons)
H = H1

@jax.jit
def half1(par, q0, p0, dt):

    q1 = par[:q0.shape[0]]
    l = par[q0.shape[0]:]
    p0_5 = p0 - (dt / 2) * (jax.grad(H, argnums=0)(q0, p0, l) + l@jac_cons(q0))
    
    return np.concatenate([q0 - q1 + (dt / 2) * (jax.grad(H, argnums=1)(q0, p0_5, np.zeros_like(l)) + jax.grad(H, argnums=1)(q1, p0_5, np.zeros_like(l))),
                           cons(q1)])

jac_half1 = jax.jit(jax.jacfwd(half1, argnums=0))

@jax.jit
def half2(par, q1, p0, l, dt):

    p1 = par[:p0.shape[0]]
    m = par[p0.shape[0]:]
    return np.concatenate([p0 - p1 - (dt / 2) * (jax.grad(H, argnums=0)(q1, p0, l) + m@jac_cons(q1)),
                           jac_cons(q1)@jax.grad(H, argnums=1)(q1, p1, m)])

jac_half2 = jax.jit(jax.jacfwd(half2, argnums=0))

@jax.jit
def step1(q0, p0, dt, max_newton_iter=10, tol=1e-9):

    def cond(par):    
        x, step, err = par
        return (step < max_newton_iter) & (err > tol)

    def body(par):
        x, step, err = par
        x = x + np.linalg.solve(jac_half1(x, q0, p0, dt), -half1(x, q0, p0, dt))
        return x, step + 1, np.linalg.norm(half1(x, q0, p0, dt))

    x0 = np.concatenate([q0, np.zeros(C.shape[0])])
    return jax.lax.while_loop(cond, body, (x0, 0, np.linalg.norm(half1(x0, q0, p0, dt))))

@jax.jit
def step2(q1, p0, l, dt, max_newton_iter=10, tol=1e-9):
    
    def cond(par):    
        x, step, err = par
        return (step < max_newton_iter) & (err > tol)
    
    def body(par):
        x, step, err = par
        x = x + np.linalg.solve(jac_half2(x, q1, p0, l, dt), -half2(x, q1, p0, l, dt))
        return x, step + 1, np.linalg.norm(half2(x, q1, p0, l, dt))
    
    x0 = np.concatenate([p1, np.zeros(C.shape[0])])
    return jax.lax.while_loop(cond, body, (x0, 0, np.linalg.norm(half2(x0, q1, p0, l, dt))))
	
@partial(jax.jit, static_argnums=1)
def sample(q0, nsteps, prng_key, dt=1e-1, max_newton_steps=10, tol=1e-9):

    traj = np.empty((nsteps, q0.shape[0]))
    accept = np.zeros(nsteps, dtype=bool)
    h_arr = np.empty(nsteps)

    def body(carry, _):

        qstep, accept, h_arr, prng_key, i = carry
        prng_key, subkey = jax.random.split(prng_key)
        pstep = propose_p(qstep, subkey)
        h_arr = h_arr.at[i].set(H1(qstep, pstep, np.zeros(C.shape[0])))

        propose_half1 = step1(qstep, pstep, dt)

        q1_propose = propose_half1[0][:qstep.shape[0]]
        l1_propose = propose_half1[0][qstep.shape[0]:]
        xstep_0_5 = np.concatenate([pstep, np.zeros(C.shape[0])])

        propose_half2 = step2(q1_propose, pstep, l1_propose, dt)

        p1_propose = propose_half2[0][:pstep.shape[0]]
        m1_propose = propose_half2[0][pstep.shape[0]:]

        prng_key, subkey = jax.random.split(prng_key)

        MH = H1(q1_propose, p1_propose, m1_propose) - H1(qstep, pstep, np.zeros(C.shape[0])) < -np.log(jax.random.uniform(subkey))
        accept = accept.at[i].set(MH)
        qstep = np.where(MH, q1_propose, qstep)
        
        return (qstep, accept, h_arr, prng_key, i + 1), qstep
    
    init = (q0, accept, h_arr, prng_key, 0)
    return jax.lax.scan(body, init, None, length=nsteps)
	
nsteps = 1000000
out = sample(q0, nsteps, key)
_, accept, h_arr, _, _ = out[0]
traj = out[1]
print("acceptance ratio: %s"%(accept.sum() / nsteps))
np.save("run1.npy", traj)
