from pygrpy.grpy_tensors import conglomerateMobilityMatrixAtCentre
import math
import numpy as np
import pygrpy.grpy_tensors


def stokesRadius(centres, sizes):
    """
    Returns stokes radius of given conglomerate of beads treating it as a rigid
    body. See: 
    https://doi.org/10.1016/j.bpj.2018.07.015 
    for details.

    Parameters
    ----------
    centers: np.array
        An ``N`` by 3 array describing locations of centres of ``N`` beads.
    sizes: np.array
        An array of length ``N`` describing radii of ``N`` beads.

    Returns
    -------
    float
        Diffusive hydrodynamic size of the specified particle.

    Example
    -------
    >>> import numpy as np
    >>> centres = np.array([[0,0,0],[0,0,1]])
    >>> sizes = np.array([1,1])
    >>> pygrpy.grpy.stokesRadius(centres,sizes)
    1.1860816944024204


    """
    mobility = conglomerateMobilityMatrixAtCentre(centres, sizes)
    return (3.0 / (6.0 * math.pi)) / (mobility[0][0] + mobility[1][1] + mobility[2][2])


def ensembleAveragedStokesRadius(ensemble_centres, sizes):
    """
    Returns stokes radius estimate given ensemble of configurations assuming no rigid bonds.
    See:
    https://doi.org/10.1017/jfm.2019.652
    for details.

    Parameters
    ----------
    ensemble_centres: np.array
        An ``M`` by ``N`` by 3 array describing ``M`` conformers each with ``N`` beads
    sizes: np.array
        An array of length ``N`` describing radii of ``N`` beads.
        
    Returns
    -------
    float
        Diffusive apparent hydrodynamic size of ensemble of conformers.

    Example
    -------
    >>> import numpy as np
    >>> ensemble_centres = 0.1*np.arange(10*5*3).reshape(10,5,3)
    >>> sizes = 0.5*np.arange(5) + 1
    >>> pygrpy.grpy.ensembleAveragedStokesRadius(ensemble_centres,sizes)
    3.0000008218278578


    """
    grand_grand_mu = np.array(
        [pygrpy.grpy_tensors.muTT(locations, sizes) for locations in ensemble_centres]
    )
    grand_mu = np.mean(grand_grand_mu, axis=0)
    grand_trace = np.trace(grand_mu, axis1=-2, axis2=-1)

    inv_mat = np.linalg.inv(grand_trace)
    total = np.sum(inv_mat)
    return total / (2 * np.pi)
