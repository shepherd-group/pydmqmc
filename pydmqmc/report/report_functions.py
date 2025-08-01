from ..methods import Method, DensityMatrixQMC

import numpy as np

def trace(method: Method):
    if not isinstance(method, DensityMatrixQMC):
        raise RuntimeError("The 'trace' analysis function can only "
                           "be used with methods that utilize "
                           "density matrices.")
    if method.density_matrix is None:
        raise RuntimeError("The associated method doesn't have a "
                           "density matrix! Did you invoke the "
                           "'run' method?")
    return np.trace(method.density_matrix)