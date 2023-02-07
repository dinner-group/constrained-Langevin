import jax
import jax.numpy as np
import jax.experimental.sparse
import scipy.integrate
jax.config.update("jax_enable_x64", True)

class KaiODE:

    cC = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.])
    cA = np.array([0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 1., 2., 1., 2., 1.])
    ind_ATP = np.array([7, 11, 12, 17, 20, 23, 26, 30])
    n_dim = 17
    n_conserve = 2
    n_react = 50
    S_sp = jax.experimental.sparse.BCOO((np.array([1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1,  1, 1, 
                                                    -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, 1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, 1, -1,  
                                                    1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, 1,  1, -1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1,  1, -1, 1,  1,  1, 
                                                    -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 
                                        np.array([[ 0,  0], [ 0,  1], [ 0,  2], [ 0, 38], [ 1,  0], [ 1,  3], [ 1,  4], [ 1,  7], [ 1, 17], [ 1, 38], [ 2,  1], [ 2,  5], [ 2,  6],
                                                    [ 2, 39], [ 3,  3], [ 3,  5], [ 3,  7], [ 3,  8], [ 3, 11], [ 3, 39], [ 4,  6], [ 4,  9], [ 4, 10], [ 4, 15], [ 4, 20], [ 4, 40], 
                                                    [ 5,  8], [ 5,  9], [ 5, 11], [ 5, 12], [ 5, 13], [ 5, 18], [ 5, 23], [ 5, 40], [ 6,  2], [ 6, 14], [ 6, 15], [ 6, 16], [ 6, 26], 
                                                    [ 6, 41], [ 7,  4], [ 7, 12], [ 7, 14], [ 7, 17], [ 7, 18], [ 7, 19], [ 7, 30], [ 7, 41], [ 8, 10], [ 8, 16], [ 8, 20], [ 8, 21], 
                                                    [ 8, 22], [ 8, 27], [ 8, 42], [ 8, 44], [ 9, 13], [ 9, 21], [ 9, 23], [ 9, 24], [ 9, 25], [ 9, 31], [ 9, 42], [ 9, 45], [10, 26], 
                                                    [10, 27], [10, 28], [10, 29], [10, 43], [10, 47], [11, 19], [11, 24], [11, 28], [11, 30], [11, 31], [11, 32], [11, 43], [11, 48], 
                                                    [12, 22], [12, 33], [12, 35], [12, 44], [12, 46], [13, 25], [13, 33], [13, 34], [13, 37], [13, 45], [13, 46], [14, 29], [14, 35], 
                                                    [14, 36], [14, 47], [14, 49], [15, 32], [15, 34], [15, 36], [15, 37], [15, 48], [15, 49], [16,  0], [16,  5], [16,  9], [16, 14], 
                                                    [16, 21], [16, 22], [16, 25], [16, 28], [16, 29], [16, 32], [16, 33], [16, 36], [16, 38], [16, 39], [16, 40], [16, 41], [16, 42], 
                                                    [16, 43], [16, 44], [16, 45], [16, 46], [16, 47], [16, 48], [16, 49]])),
                                        shape=(n_dim, n_react))
    K_sp = jax.experimental.sparse.BCOO((np.ones(62),
                                        np.array([[ 0, 38], [ 1,  0], [ 1,  7], [ 1, 17], [ 2,  1], [ 2, 39], [ 3,  3], [ 3,  5], [ 3, 11], [ 4,  6], [ 4, 15], [ 4, 20], [ 4, 40], 
                                                    [ 5,  8], [ 5,  9], [ 5, 18], [ 5, 23], [ 6,  2], [ 6, 26], [ 6, 41], [ 7,  4], [ 7, 12], [ 7, 14], [ 7, 30], [ 8, 10], [ 8, 16], 
                                                    [ 8, 27], [ 8, 42], [ 8, 44], [ 9, 13], [ 9, 21], [ 9, 31], [ 9, 45], [10, 43], [10, 47], [11, 19], [11, 24], [11, 28], [11, 48], 
                                                    [12, 22], [12, 35], [12, 46], [13, 25], [13, 33], [13, 37], [14, 29], [14, 49], [15, 32], [15, 34], [15, 36], [16, 38], [16, 39], 
                                                    [16, 40], [16, 41], [16, 42], [16, 43], [16, 44], [16, 45], [16, 46], [16, 47], [16, 48], [16, 49]])),
                                        shape=(n_dim, n_react))
    
    def __init__(self, reaction_consts, a0=0.6, c0=3.5, ATPfrac=1.):
        self.a0 = a0
        self.c0 = c0
        self.ATPfrac = ATPfrac
        self.reaction_consts = reaction_consts
        self.S = np.zeros((KaiODE.n_dim, self.reaction_consts.shape[0]))
        self.K = np.zeros_like(self.S)
        
        # kTU, kDT, kDS, kSU,\
        # kUTA, kTUA, kTDA, kDTA, kDSA, kSDA, kSUA, kUSA,\
        # kCIIAonU, kCIIAoffU, kCIIAonT, kCIIAoffT, kCIIAonD, kCIIAoffD, kCIIAonS, kCIIAoffS,\
        # kCIhydD00, kRelD00, kCIhydS00, kRelS00,\
        # kCIhydDA0, kRelDA0, kCIhydSA0, kRelSA0,\
        # kDS00, kDS0A, kDSA0, kDSAA, kSDA0, kSDAA,\
        # kCIIAonD00, kCIIAoffDA0, kCIIAonD0A, kCIIAoffDAA,\
        # kCIIAonS00, kCIIAoffSA0 , kCIIAonS0A, kCIIAoffSAA,\
        # kCIAonD00, kCIAoffD0A, kCIAonDA0, kCIAoffDAA,\
        # kCIAonS00, kCIAoffS0A, kCIAonSA0, kCIAoffSAA = self.reaction_consts
        kCIIAoffU, kTU, kSU, kTUA, kSUA, kCIIAoffT,\
        kDT, kUTA, kDTA, kCIIAoffD, kRelD00, kTDA, kSDA,\
        kRelDA0, kCIIAoffS, kDS, kRelS00, kUSA, kDSA,\
        kRelSA0, kCIhydD00, kCIIAoffDA0, kCIAoffD0A,\
        kCIhydDA0, kSDA0, kCIAoffDAA, kCIhydS00, kDS00,\
        kCIIAoffSA0, kCIAoffS0A, kCIhydSA0, kDSA0,\
        kCIAoffSAA, kCIIAoffDAA, kSDAA, kDS0A, kCIIAoffSAA,\
        kDSAA, kCIIAonU, kCIIAonT, kCIIAonD, kCIIAonS,\
        kCIIAonD00, kCIIAonS00, kCIAonD00, kCIAonDA0,\
        kCIIAonD0A, kCIAonS00, kCIAonSA0, kCIIAonS0A = self.reaction_consts
        
        self.S = self.S.at[0, 0].set(1)
        self.S = self.S.at[1, 0].set(-1)
        self.S = self.S.at[-1, 0].set(1)
        
        self.S = self.S.at[0, 1].set(1)
        self.S = self.S.at[2, 1].set(-1)
        
        self.S = self.S.at[0, 2].set(1)
        self.S = self.S.at[6, 2].set(-1)
        
        self.S = self.S.at[1, 3].set(1)
        self.S = self.S.at[3, 3].set(-1)
        
        self.S = self.S.at[1, 4].set(1)
        self.S = self.S.at[7, 4].set(-1)
        
        self.S = self.S.at[2, 5].set(1)
        self.S = self.S.at[3, 5].set(-1)
        self.S = self.S.at[-1, 5].set(1)
        
        self.S = self.S.at[2, 6].set(1)
        self.S = self.S.at[4, 6].set(-1)
        
        self.S = self.S.at[3, 7].set(1)
        self.S = self.S.at[1, 7].set(-1)
        
        self.S = self.S.at[3, 8].set(1)
        self.S = self.S.at[5, 8].set(-1)
        
        self.S = self.S.at[4, 9].set(1)
        self.S = self.S.at[5, 9].set(-1)
        self.S = self.S.at[-1, 9].set(1)
        
        self.S = self.S.at[4, 10].set(1)
        self.S = self.S.at[8, 10].set(-1)
        
        self.S = self.S.at[5, 11].set(1)
        self.S = self.S.at[3, 11].set(-1)
        
        self.S = self.S.at[5, 12].set(1)
        self.S = self.S.at[7, 12].set(-1)
        
        self.S = self.S.at[5, 13].set(1)
        self.S = self.S.at[9, 13].set(-1)
        
        self.S = self.S.at[6, 14].set(1)
        self.S = self.S.at[7, 14].set(-1)
        self.S = self.S.at[-1, 14].set(1)
        
        self.S = self.S.at[6, 15].set(1)
        self.S = self.S.at[4, 15].set(-1)
        
        self.S = self.S.at[6, 16].set(1)
        self.S = self.S.at[8, 16].set(-1)
        
        self.S = self.S.at[7, 17].set(1)
        self.S = self.S.at[1, 17].set(-1)
        
        self.S = self.S.at[7, 18].set(1)
        self.S = self.S.at[5, 18].set(-1)
        
        self.S = self.S.at[7, 19].set(1)
        self.S = self.S.at[11, 19].set(-1)
        
        self.S = self.S.at[8, 20].set(1)
        self.S = self.S.at[4, 20].set(-1)
        
        self.S = self.S.at[8, 21].set(1)
        self.S = self.S.at[9, 21].set(-1)
        self.S = self.S.at[-1, 21].set(1)
        
        self.S = self.S.at[8, 22].set(1)
        self.S = self.S.at[12, 22].set(-1)
        self.S = self.S.at[-1, 22].set(1)
        
        self.S = self.S.at[9, 23].set(1)
        self.S = self.S.at[5, 23].set(-1)
        
        self.S = self.S.at[9, 24].set(1)
        self.S = self.S.at[11, 24].set(-1)
        
        self.S = self.S.at[9, 25].set(1)
        self.S = self.S.at[13, 25].set(-1)
        self.S = self.S.at[-1, 25].set(1)
        
        self.S = self.S.at[10, 26].set(1)
        self.S = self.S.at[6, 26].set(-1)
        
        self.S = self.S.at[10, 27].set(1)
        self.S = self.S.at[8, 27].set(-1)
        
        self.S = self.S.at[10, 28].set(1)
        self.S = self.S.at[11, 28].set(-1)
        self.S = self.S.at[-1, 28].set(1)
        
        self.S = self.S.at[10, 29].set(1)
        self.S = self.S.at[14, 29].set(-1)
        self.S = self.S.at[-1, 29].set(1)
        
        self.S = self.S.at[11, 30].set(1)
        self.S = self.S.at[7, 30].set(-1)
        
        self.S = self.S.at[11, 31].set(1)
        self.S = self.S.at[9, 31].set(-1)
        
        self.S = self.S.at[11, 32].set(1)
        self.S = self.S.at[15, 32].set(-1)
        self.S = self.S.at[-1, 32].set(1)
        
        self.S = self.S.at[12, 33].set(1)
        self.S = self.S.at[13, 33].set(-1)
        self.S = self.S.at[-1, 33].set(1)
        
        self.S = self.S.at[13, 34].set(1)
        self.S = self.S.at[15, 34].set(-1)
        
        self.S = self.S.at[14, 35].set(1)
        self.S = self.S.at[12, 35].set(-1)
        
        self.S = self.S.at[14, 36].set(1)
        self.S = self.S.at[15, 36].set(-1)
        self.S = self.S.at[-1, 36].set(1)
        
        self.S = self.S.at[15, 37].set(1)
        self.S = self.S.at[13, 37].set(-1)
        
        self.S = self.S.at[1, 38].set(1)
        self.S = self.S.at[0, 38].set(-1)
        self.S = self.S.at[-1, 38].set(-1)
        
        self.S = self.S.at[3, 39].set(1)
        self.S = self.S.at[2, 39].set(-1)
        self.S = self.S.at[-1, 39].set(-1)
        
        self.S = self.S.at[5, 40].set(1)
        self.S = self.S.at[4, 40].set(-1)
        self.S = self.S.at[-1, 40].set(-1)
        
        self.S = self.S.at[7, 41].set(1)
        self.S = self.S.at[6, 41].set(-1)
        self.S = self.S.at[-1, 41].set(-1)
        
        self.S = self.S.at[9, 42].set(1)
        self.S = self.S.at[8, 42].set(-1)
        self.S = self.S.at[-1, 42].set(-1)
        
        self.S = self.S.at[11, 43].set(1)
        self.S = self.S.at[10, 43].set(-1)
        self.S = self.S.at[-1, 43].set(-1)
        
        self.S = self.S.at[12, 44].set(1)
        self.S = self.S.at[8, 44].set(-1)
        self.S = self.S.at[-1, 44].set(-1)
        
        self.S = self.S.at[13, 45].set(1)
        self.S = self.S.at[9, 45].set(-1)
        self.S = self.S.at[-1, 45].set(-1)
        
        self.S = self.S.at[13, 46].set(1)
        self.S = self.S.at[12, 46].set(-1)
        self.S = self.S.at[-1, 46].set(-1)
        
        self.S = self.S.at[14, 47].set(1)
        self.S = self.S.at[10, 47].set(-1)
        self.S = self.S.at[-1, 47].set(-1)
        
        self.S = self.S.at[15, 48].set(1)
        self.S = self.S.at[11, 48].set(-1)
        self.S = self.S.at[-1, 48].set(-1)
        
        self.S = self.S.at[15, 49].set(1)
        self.S = self.S.at[14, 49].set(-1)
        self.S = self.S.at[-1, 49].set(-1)
        
        self.K = np.where(self.S < 0, -self.S, 0)

    @jax.jit
    def f(self, t, y, reaction_consts=None, ATPfrac=None):

        if reaction_consts is None:
            reaction_consts = self.reaction_consts
        if ATPfrac is None:
            ATPfrac = self.ATPfrac

        return self.S@(reaction_consts.at[self.ind_ATP].multiply(ATPfrac) * np.prod(y**self.K.T, axis=1))

    @jax.jit
    def jac(self, t, y, reaction_consts=None, ATPfrac=None):
    
        if reaction_consts is None:
            reaction_consts = self.reaction_consts
        if ATPfrac is None:
            ATPfrac = self.ATPfrac
        
        return jax.jacfwd(self.f, argnums=1)(t, y, reaction_consts, ATPfrac)

    @jax.jit
    def f_test(self, t, y, reaction_consts=None, ATPfrac=None):

        if reaction_consts is None:
            reaction_consts = self.reaction_consts
        if ATPfrac is None:
            ATPfrac = self.ATPfrac

        def loop1(carry, _):
                
            i, z = carry
            z = z.at[self.K_sp.indices[i, 1]].multiply(y[self.K_sp.indices[i, 0]]**self.K_sp.data[i])
            return (i + 1, z), _

        z = reaction_consts * jax.lax.scan(loop1, init=(0, np.ones(KaiODE.n_react)), xs=None, length=self.K_sp.data.size)[0][1]
        return jax.experimental.sparse.sparsify(lambda M, x:M@x)(self.S_sp, z)

    @jax.jit
    def jac_test(self, t, y, reaction_consts=None, ATPfrac=None):

        if reaction_consts is None:
            reaction_consts = self.reaction_consts
        if ATPfrac is None:
            ATPfrac = self.ATPfrac
        
        return jax.jacfwd(self.f, argnums=1)(t, y, reaction_consts, ATPfrac)

    @jax.jit
    def f_red(self, t, y, reaction_consts=None, a0=None, c0=None, ATPfrac=None):

        if reaction_consts is None:
            reaction_consts=self.reaction_consts
        if a0 is None:
            a0 = self.a0
        if c0 is None:
            c0 = self.c0
        if ATPfrac is None:
            ATPfrac = self.ATPfrac

        yfull = np.zeros(KaiODE.n_dim)
        yfull = yfull.at[0].set(c0 - KaiODE.cC[1:-1]@y)
        yfull = yfull.at[1:-1].set(y)
        yfull = yfull.at[-1].set(a0 - KaiODE.cA[1:-1]@y)
        return self.f(t, yfull, reaction_consts, ATPfrac)[1:-1]

    @jax.jit
    def jac_red(self, t, y, reaction_consts=None, a0=None, c0=None, ATPfrac=None):

        if reaction_consts is None:
            reaction_consts=self.reaction_consts
        if a0 is None:
            a0 = self.a0
        if c0 is None:
            c0 = self.c0
        if ATPfrac is None:
            ATPfrac = self.ATPfrac

        return jax.jacfwd(self.f_red, argnums=1)(t, y, reaction_consts, a0, c0, ATPfrac)
    
    @jax.jit
    def f_dae(self, t, y, reaction_consts=None, a0=None, c0=None, ATPfrac=None):

        if reaction_consts is None:
            reaction_consts=self.reaction_consts
        if a0 is None:
            a0 = self.a0
        if c0 is None:
            c0 = self.c0
        if ATPfrac is None:
            ATPfrac = self.ATPfrac

        out = self.f(t, y, reaction_consts, ATPfrac)
        out = out.at[0].set(y@self.cC - c0)
        out = out.at[-1].set(y@self.cA - a0)
            
        return out

    @jax.jit
    def jac_dae(self, t, y, reaction_consts=None, a0=None, c0=None, ATPfrac=None):

        if reaction_consts is None:
            reaction_consts=self.reaction_consts
        if a0 is None:
            a0 = self.a0
        if c0 is None:
            c0 = self.c0
        if ATPfrac is None:
            ATPfrac = self.ATPfrac

        return jax.jacfwd(self.f_dae, argnums=1)(t, y, reaction_consts, a0, c0, ATPfrac)

    def _tree_flatten(self):

        children = (self.reaction_consts, self.a0, self.c0, self.ATPfrac, self.S, self.K)
        aux_data = {}

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):

        return cls(*children[:4], **aux_data)

class Brusselator:

    n_dim = 2
    n_react = 3
    n_conserve = 0
    S = np.array([[ 1,  1, -1, -1],
                  [ 0, -1,  1,  0]])
    K = np.array([[0, 2, 1, 1],
                  [0, 1, 0, 0]])

    def __init__(self, reaction_consts):

        self.reaction_consts = reaction_consts

    @jax.jit
    def f(self, t, y, reaction_consts=None):

        if reaction_consts is None:
            reaction_consts = self.reaction_consts

        rc = np.concatenate([np.array([1]), reaction_consts])
        return self.S@(rc * np.prod(y**self.K.T, axis=1))

    f_red = f

    @jax.jit
    def jac(self, t, y, reaction_consts=None):
        return jax.jacfwd(self.f, argnums=1)(t, y, reaction_consts)

    def _tree_flatten(self):

        children = (self.reaction_consts,)
        aux_data = {}

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):

        return cls(*children, **aux_data)

class Morris_Lecar:

    n_dim = 3
    
    def __init__(self, par):
        self.par = par

    @jax.jit
    def f(self, t, y, par=None):

        if par is None:
            par = self.par

        dy = np.zeros_like(y)
        dy = dy.at[0].set((I_ext - par[0] * (y[0] - par[1]) - par[2] * y[1] * (y[0] - par[3]) - par[4] * y[2] * (y[0] - par[5]) - par[6] / np.cosh((y[0] - par[7]) / par[8])**2 * (y[0] - par[5])) / par[9])
        dy = dy.at[1].set(np.exp(par[10]) * ((1 + np.tanh((y[0] - par[11]) / par[12])) / 2 - y[1]) * np.cosh((y[0] - par[11]) / par[13]))
        dy = dy.at[2].set(np.exp(par[14]) * ((1 + np.tanh((y[0] - par[15]) / par[16])) / 2 - y[2]) * np.cosh((y[0] - par[15]) / par[17]))

        return dy

    f_red = f

    @jax.jit
    def jac(self, t, y, par=None):
        return jax.jacfwd(self.f, argnums=1)(t, y, par)

    def _tree_flatten(self):
        children = (self.par,)
        aux_data = {}

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

jax.tree_util.register_pytree_node(KaiODE, KaiODE._tree_flatten, KaiODE._tree_unflatten)
jax.tree_util.register_pytree_node(Brusselator, Brusselator._tree_flatten, Brusselator._tree_unflatten)
jax.tree_util.register_pytree_node(Morris_Lecar, Morris_Lecar._tree_flatten, Morris_Lecar._tree_unflatten)
