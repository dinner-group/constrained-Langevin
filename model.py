import jax
import jax.numpy as np
import jax.experimental.sparse
jax.config.update("jax_enable_x64", True)

class Repressilator_log:

    n_dim = 3
    n_par = 8
    not_algebraic = 1

    def __init__(self, par=None):
        if par is None:
            self.par = np.concatenate([np.full(self.n_dim, np.log(10)), np.zeros(self.n_dim - 1), np.full(3, self.n_dim)])
        else:
            self.par = par

    @jax.jit
    def f(self, t, y, par=None):
        if par is None:
            par = self.par
        ydot = np.zeros_like(y)
        ydot = ydot.at[0].set(np.exp(par[0] - y[0]) / (1 + np.exp(par[7] * y[2])) - 1)
        ydot = ydot.at[1].set(np.exp(par[1] - y[1]) / (1 + np.exp(par[5] * y[0])) - np.exp(par[3]))
        ydot = ydot.at[2].set(np.exp(par[2] - y[2]) / (1 + np.exp(par[6] * y[1])) - np.exp(par[4]))
        return ydot

    @jax.jit
    def jac(self, t, y, par=None):
        return jax.jacfwd(self.f, argnums=1)(t, y, par)

    @jax.jit
    def ravel(self):
        return np.zeros(0)

    def _tree_flatten(self):
        children = (self.par,)
        aux_data = {}
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

class Repressilator_log_n:

    not_algebraic = 1

    def __init__(self, par=None, n_dim=3):
        self.n_dim = n_dim
        self.n_par = 3 * n_dim - 1
        if par is None:
            self.par = np.concatenate([np.full(n_dim, np.log(10)), np.zeros(n_dim - 1), np.full(n_dim, 3)])
        else:
            self.par = par

    @jax.jit
    def f(self, t, y, par=None):
        if par is None:
            par = self.par

        synthesis_rate = par[:self.n_dim]
        degrade_rate = np.pad(par[self.n_dim:2 * self.n_dim - 1], (1, 0))
        hill_coeff = par[2 * self.n_dim - 1:3 * self.n_dim - 1]

        ydot = np.zeros_like(y)

        def loop_body(carry, _):
            i, ydot = carry
            ydot = ydot.at[i].set(np.exp(synthesis_rate[i] - y[i]) / (1 + np.exp(hill_coeff[i - 1] * y[i - 1])) - np.exp(degrade_rate[i]))
            return (i + 1, ydot), _

        ydot = jax.lax.scan(loop_body, init=(0, ydot), xs=None, length=self.n_dim)[0][1]

        return ydot

    @jax.jit
    def jac(self, t, y, par=None):
        return jax.jacfwd(self.f, argnums=1)(t, y, par)

    @jax.jit
    def ravel(self):
        return np.zeros(0)

    def _tree_flatten(self):
        children = (self.par,)
        aux_data = {"n_dim":self.n_dim}
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    
jax.tree_util.register_pytree_node(Repressilator_log, Repressilator_log._tree_flatten, Repressilator_log._tree_unflatten)
jax.tree_util.register_pytree_node(Repressilator_log_n, Repressilator_log_n._tree_flatten, Repressilator_log_n._tree_unflatten)
