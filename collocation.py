import jax.numpy as np
import scipy.sparse
import sparsejac.sparsejac as sparsejac
from functools import partial

class colloc:

    gauss_points = (1 + np.array([-0.906179845938664, -0.538469310105683, 0.0, +0.538469310105683, +0.906179845938664])) / 2

    def __init__(self, model, n_mesh_point):

        self.n_mesh_point = n_mesh_point

    @jax.jit
    def lagrange_poly
