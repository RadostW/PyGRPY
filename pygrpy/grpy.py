from pygrpy.grpy_tensors import conglomerateMobilityMatrixAtCentre
import math

def stokesRadius(centres,sizes):
    '''
    Returns stokes radius of given conglomerate of beads treating it as a rigid
    body.

    Parameters
    ----------
    centers: np.array
        An ``N`` by 3 array describing locations of centres of ``N`` beads.
    radii: np.array
        An array of length ``N`` describing sizes of ``N`` beads.

    Returns
    -------
    float
        Diffusive hydrodynamic size of the specified particle

    Example
    -------
    >>> import numpy as np
    >>> centres = np.array([[0,0,0],[0,0,1]])
    >>> radii = np.array([1,1])
    >>> pygrpy.grpy.stokesRadius(centres,radii)
    1.1860816944024204
    

    '''
    mobility = conglomerateMobilityMatrixAtCentre(centres,sizes)
    return (3.0 / (6.0 * math.pi))/(mobility[0][0] + mobility[1][1] + mobility[2][2])
