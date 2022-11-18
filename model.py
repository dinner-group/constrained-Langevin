import jax
import jax.numpy as np
import scipy.integrate
jax.config.update("jax_enable_x64", True)

class KaiODE:

    cC = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.])
    cA = np.array([0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 1., 2., 1., 2., 1.])
    ind_ATP = np.array([7, 11, 12, 17, 20, 23, 26, 30])
    n_dim = 17
    n_conserve = 2
    n_react = 50
    
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

jax.tree_util.register_pytree_node(KaiODE, KaiODE._tree_flatten, KaiODE._tree_unflatten)
