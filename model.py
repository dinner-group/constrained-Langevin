import jax
import jax.numpy as np
import scipy.integrate
jax.config.update("jax_enable_x64", True)

class KaiODE:

    cC = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.])
    cA = np.array([0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 1., 2., 1., 2., 1.])
    
    def __init__(self, reaction_consts, a0=0.6, c0=3.5, ATPfrac=1.):
        self.a0 = a0
        self.c0 = c0
        self.ATPfrac = ATPfrac
        self.reaction_consts = reaction_consts.copy()
        self.A = np.zeros((17, 17))
        self.B = np.zeros_like(self.A)
        self.S = np.zeros((17, self.reaction_consts.shape[0]))
        self.K = np.zeros_like(self.S)
        self.y0 = np.zeros(17)
        self.y0 = self.y0.at[0].set(c0)
        self.y0 = self.y0.at[-1].set(a0)
        #self.set_matrices()
    
    #@jax.jit
    #def set_matrices(self):
        
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
        
        self.A = self.A.at[0, 1].set(kCIIAoffU)
        self.A = self.A.at[-1, 1].set(kCIIAoffU)
        self.S = self.S.at[0, 0].set(1)
        self.S = self.S.at[1, 0].set(-1)
        self.S = self.S.at[-1, 0].set(1)
        
        self.A = self.A.at[0, 2].set(kTU)
        self.S = self.S.at[0, 1].set(1)
        self.S = self.S.at[2, 1].set(-1)
        
        self.A = self.A.at[0, 6].set(kSU)
        self.S = self.S.at[0, 2].set(1)
        self.S = self.S.at[6, 2].set(-1)
        
        self.A = self.A.at[1, 3].set(kTUA)
        self.S = self.S.at[1, 3].set(1)
        self.S = self.S.at[3, 3].set(-1)
        
        self.A = self.A.at[1, 7].set(kSUA)
        self.S = self.S.at[1, 4].set(1)
        self.S = self.S.at[7, 4].set(-1)
        
        self.A = self.A.at[2, 3].set(kCIIAoffT)
        self.A = self.A.at[-1, 3].set(kCIIAoffT)
        self.S = self.S.at[2, 5].set(1)
        self.S = self.S.at[3, 5].set(-1)
        self.S = self.S.at[-1, 5].set(1)
        
        self.A = self.A.at[2, 4].set(kDT)
        self.S = self.S.at[2, 6].set(1)
        self.S = self.S.at[4, 6].set(-1)
        
        self.A = self.A.at[3, 1].set(self.ATPfrac * kUTA)
        self.S = self.S.at[3, 7].set(1)
        self.S = self.S.at[1, 7].set(-1)
        
        self.A = self.A.at[3, 5].set(kDTA)
        self.S = self.S.at[3, 8].set(1)
        self.S = self.S.at[5, 8].set(-1)
        
        self.A = self.A.at[4, 5].set(kCIIAoffD)
        self.A = self.A.at[-1, 5].set(kCIIAoffD)
        self.S = self.S.at[4, 9].set(1)
        self.S = self.S.at[5, 9].set(-1)
        self.S = self.S.at[-1, 9].set(1)
        
        self.A = self.A.at[4, 8].set(kRelD00)
        self.S = self.S.at[4, 10].set(1)
        self.S = self.S.at[8, 10].set(-1)
        
        self.A = self.A.at[5, 3].set(self.ATPfrac * kTDA)
        self.S = self.S.at[5, 11].set(1)
        self.S = self.S.at[3, 11].set(-1)
        
        self.A = self.A.at[5, 7].set(self.ATPfrac * kSDA)
        self.S = self.S.at[5, 12].set(1)
        self.S = self.S.at[7, 12].set(-1)
        
        self.A = self.A.at[5, 9].set(kRelDA0)
        self.S = self.S.at[5, 13].set(1)
        self.S = self.S.at[9, 13].set(-1)
        
        self.A = self.A.at[6, 7].set(kCIIAoffS)
        self.A = self.A.at[-1, 7].set(kCIIAoffS)
        self.S = self.S.at[6, 14].set(1)
        self.S = self.S.at[7, 14].set(-1)
        self.S = self.S.at[-1, 14].set(1)
        
        self.A = self.A.at[6, 4].set(kDS)
        self.S = self.S.at[6, 15].set(1)
        self.S = self.S.at[4, 15].set(-1)
        
        self.A = self.A.at[6, 8].set(kRelS00)
        self.S = self.S.at[6, 16].set(1)
        self.S = self.S.at[8, 16].set(-1)
        
        self.A = self.A.at[7, 1].set(self.ATPfrac * kUSA)
        self.S = self.S.at[7, 17].set(1)
        self.S = self.S.at[1, 17].set(-1)
        
        self.A = self.A.at[7, 5].set(kDSA)
        self.S = self.S.at[7, 18].set(1)
        self.S = self.S.at[5, 18].set(-1)
        
        self.A = self.A.at[7, 11].set(kRelSA0)
        self.S = self.S.at[7, 19].set(1)
        self.S = self.S.at[11, 19].set(-1)
        
        self.A = self.A.at[8, 4].set(self.ATPfrac * kCIhydD00)
        self.S = self.S.at[8, 20].set(1)
        self.S = self.S.at[4, 20].set(-1)
        
        self.A = self.A.at[8, 9].set(kCIIAoffDA0)
        self.A = self.A.at[-1, 9].set(kCIIAoffDA0)
        self.S = self.S.at[8, 21].set(1)
        self.S = self.S.at[9, 21].set(-1)
        self.S = self.S.at[-1, 21].set(1)
        
        self.A = self.A.at[8, 12].set(kCIAoffD0A)
        self.A = self.A.at[-1, 12].set(kCIAoffD0A)
        self.S = self.S.at[8, 22].set(1)
        self.S = self.S.at[12, 22].set(-1)
        self.S = self.S.at[-1, 22].set(1)
        
        self.A = self.A.at[9, 5].set(self.ATPfrac * kCIhydDA0)
        self.S = self.S.at[9, 23].set(1)
        self.S = self.S.at[5, 23].set(-1)
        
        self.A = self.A.at[9, 11].set(kSDA0)
        self.S = self.S.at[9, 24].set(1)
        self.S = self.S.at[11, 24].set(-1)
        
        self.A = self.A.at[9, 13].set(kCIAoffDAA)
        self.A = self.A.at[-1, 13].set(kCIAoffDAA)
        self.S = self.S.at[9, 25].set(1)
        self.S = self.S.at[13, 25].set(-1)
        self.S = self.S.at[-1, 25].set(1)
        
        self.A = self.A.at[10, 6].set(self.ATPfrac * kCIhydS00)
        self.S = self.S.at[10, 26].set(1)
        self.S = self.S.at[6, 26].set(-1)
        
        self.A = self.A.at[10, 8].set(kDS00)
        self.S = self.S.at[10, 27].set(1)
        self.S = self.S.at[8, 27].set(-1)
        
        self.A = self.A.at[10, 11].set(kCIIAoffSA0)
        self.A = self.A.at[-1, 11].set(kCIIAoffSA0)
        self.S = self.S.at[10, 28].set(1)
        self.S = self.S.at[11, 28].set(-1)
        self.S = self.S.at[-1, 28].set(1)
        
        self.A = self.A.at[10, 14].set(kCIAoffS0A)
        self.A = self.A.at[-1, 14].set(kCIAoffS0A)
        self.S = self.S.at[10, 29].set(1)
        self.S = self.S.at[14, 29].set(-1)
        self.S = self.S.at[-1, 29].set(1)
        
        self.A = self.A.at[11, 7].set(self.ATPfrac * kCIhydSA0)
        self.S = self.S.at[11, 30].set(1)
        self.S = self.S.at[7, 30].set(-1)
        
        self.A = self.A.at[11, 9].set(kDSA0)
        self.S = self.S.at[11, 31].set(1)
        self.S = self.S.at[9, 31].set(-1)
        
        self.A = self.A.at[11, 15].set(kCIAoffSAA)
        self.A = self.A.at[-1, 15].add(kCIAoffSAA)
        self.S = self.S.at[11, 32].set(1)
        self.S = self.S.at[15, 32].set(-1)
        self.S = self.S.at[-1, 32].set(1)
        
        self.A = self.A.at[12, 13].set(kCIIAoffDAA)
        self.A = self.A.at[-1, 13].add(kCIIAoffDAA)
        self.S = self.S.at[12, 33].set(1)
        self.S = self.S.at[13, 33].set(-1)
        self.S = self.S.at[-1, 33].set(1)
        
        self.A = self.A.at[13, 15].set(kSDAA)
        self.S = self.S.at[13, 34].set(1)
        self.S = self.S.at[15, 34].set(-1)
        
        self.A = self.A.at[14, 12].set(kDS0A)
        self.S = self.S.at[14, 35].set(1)
        self.S = self.S.at[12, 35].set(-1)
        
        self.A = self.A.at[14, 15].set(kCIIAoffSAA)
        self.A = self.A.at[-1, 15].add(kCIIAoffSAA)
        self.S = self.S.at[14, 36].set(1)
        self.S = self.S.at[15, 36].set(-1)
        self.S = self.S.at[-1, 36].set(1)
        
        self.A = self.A.at[15, 13].set(kDSAA)
        self.S = self.S.at[15, 37].set(1)
        self.S = self.S.at[13, 37].set(-1)
        
        self.A = self.A.at[np.diag_indices(self.A.shape[0])].add(-self.A[:-1, :].sum(axis=0))
        
        self.B = self.B.at[1, 0].set(kCIIAonU)
        self.S = self.S.at[1, 38].set(1)
        self.S = self.S.at[0, 38].set(-1)
        self.S = self.S.at[-1, 38].set(-1)
        
        self.B = self.B.at[3, 2].set(kCIIAonT)
        self.S = self.S.at[3, 39].set(1)
        self.S = self.S.at[2, 39].set(-1)
        self.S = self.S.at[-1, 39].set(-1)
        
        self.B = self.B.at[5, 4].set(kCIIAonD)
        self.S = self.S.at[5, 40].set(1)
        self.S = self.S.at[4, 40].set(-1)
        self.S = self.S.at[-1, 40].set(-1)
        
        self.B = self.B.at[7, 6].set(kCIIAonS)
        self.S = self.S.at[7, 41].set(1)
        self.S = self.S.at[6, 41].set(-1)
        self.S = self.S.at[-1, 41].set(-1)
        
        self.B = self.B.at[9, 8].set(kCIIAonD00)
        self.S = self.S.at[9, 42].set(1)
        self.S = self.S.at[8, 42].set(-1)
        self.S = self.S.at[-1, 42].set(-1)
        
        self.B = self.B.at[11, 10].set(kCIIAonS00)
        self.S = self.S.at[11, 43].set(1)
        self.S = self.S.at[10, 43].set(-1)
        self.S = self.S.at[-1, 43].set(-1)
        
        self.B = self.B.at[12, 8].set(kCIAonD00)
        self.S = self.S.at[12, 44].set(1)
        self.S = self.S.at[8, 44].set(-1)
        self.S = self.S.at[-1, 44].set(-1)
        
        self.B = self.B.at[13, 9].set(kCIAonDA0)
        self.S = self.S.at[13, 45].set(1)
        self.S = self.S.at[9, 45].set(-1)
        self.S = self.S.at[-1, 45].set(-1)
        
        self.B = self.B.at[13, 12].set(kCIIAonD0A)
        self.S = self.S.at[13, 46].set(1)
        self.S = self.S.at[12, 46].set(-1)
        self.S = self.S.at[-1, 46].set(-1)
        
        self.B = self.B.at[14, 10].set(kCIAonS00)
        self.S = self.S.at[14, 47].set(1)
        self.S = self.S.at[10, 47].set(-1)
        self.S = self.S.at[-1, 47].set(-1)
        
        self.B = self.B.at[15, 11].set(kCIAonSA0)
        self.S = self.S.at[15, 48].set(1)
        self.S = self.S.at[11, 48].set(-1)
        self.S = self.S.at[-1, 48].set(-1)
        
        self.B = self.B.at[15, 14].set(kCIIAonS0A)
        self.S = self.S.at[15, 49].set(1)
        self.S = self.S.at[14, 49].set(-1)
        self.S = self.S.at[-1, 49].set(-1)
        
        self.B = self.B.at[np.diag_indices(self.B.shape[0])].add(-self.B[:-1, :].sum(axis=0))
        self.B = self.B.at[-1, :].set(np.diag(self.B))
        self.K = np.where(self.S < 0, -self.S, 0)
    
    @jax.jit
    def f(self, t, y):

        return (self.A + y[-1] * self.B)@y

    @jax.jit
    def jac(self, t, y):

        return jax.jacfwd(self.f, argnums=1)(t, y)
    
    @jax.jit
    def f_dae(self, t, y):

        out = self.f(t, y)
        out = out.at[0].set(y@self.cC - self.y0[0])
        out = out.at[-1].set(y@self.cA - self.y0[-1])
            
        return out

    @jax.jit
    def jac_dae(self, t, y):

        return jax.jacfwd(self.f_dae, argnums=1)(t, y)

    def _tree_flatten(self):

        children = (self.reaction_consts, self.a0, self.c0, self.ATPfrac, self.A, self.B, self.S, self.K, self.y0)
        aux_data = {}

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):

        return cls(*children[:4], **aux_data)

jax.tree_util.register_pytree_node(KaiODE, KaiODE._tree_flatten, KaiODE._tree_unflatten)
