import math

import jax
import jax.numpy as jnp
import jax.ops

import json

_epsilon = jnp.array(
    [
        [[0.0, 0.0,  0.0], [0.0,  0.0, 1.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0,  0.0, 0.0], [1.0, 0.0,  0.0]],
        [[0.0, 1.0,  0.0], [-1.0, 0.0, 0.0], [0.0, 0.0,  0.0]],
    ]
)  # levi-civita(3)


def _transTranspose(tensor):
    """
    Returns :math:`a_jilk` given tensor :math:`a_ijkl`
    """
    return jnp.transpose(tensor, [1, 0, 3, 2])

def mu(centres, radii):
    """
    Returns grand mobility matrix in RPY approximation.

    Parameters
    ----------
    centers: jnp.array
        An ``N`` by 3 array describing locations of centres of ``N`` beads.
    radii: jnp.array
        An array of length ``N`` describing sizes of ``N`` beads.

    Returns
    -------
    jnp.array
        A ``6N`` by ``6N`` array containing translational and rotational mobility coefficients
        of the beads. Indicies are ordered: ``ux1,uy1,uz1,  ux2,uy2,uz2, ..., wx1,wy1,wz1, ...```.
        All translations before rotations, then by bead index, then by coordinate.

    Example
    -------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import pygrpy.jax_grpy_tensors
    >>> centres = jnp.array([[0,0,0],[0,0,1]])
    >>> radii = jnp.array([1,1])
    >>> pygrpy.jax_grpy_tensors.mu(centres,radii).shape()
    (12,12)
    >>> fast_mu = jax.jit(pygrpy.jax_grpy_tensors.mu)
    >>> fast_mu(centres,radii).shape #compiled variant of mu
    (12,12)

    """
    # number of beads
    n = len(radii)

    displacements = centres[:, jnp.newaxis, :] - centres[jnp.newaxis, :, :]
    distances = jnp.sqrt(jnp.sum(displacements ** 2, axis=-1)) 

    # shorthand for radii, consistent with publication of Zuk et al
    a = radii 

    # normalized displacements and zeros for i==j
    rHatMatrix = displacements / (distances[:,:,jnp.newaxis] + jnp.identity(n)[:,:,jnp.newaxis])

    # epsilonRHatMatrix_abij = rHatMatrix_abk _epsilon_ijk
    epsilonRHatMatrix = jnp.tensordot(rHatMatrix,_epsilon,axes=([2],[2]))

    ai = a[:,jnp.newaxis]
    aj = a[jnp.newaxis,:]
    
    dist = distances + jnp.identity(n) # add identity to allow division

    # prefactors of matricies for each bead pair
    # matricies are {identity, r^hat r^hat, \\epsilon_ijk r^hat_k}
    # these are grouped by interaction type {TT,TR,RR}
    # and by solution branch {diagonal, close, far} for Rotne-Prager and Yakamava parts
    # numpy magic does all operations componentwise

    # ### translational matricies
    muTTidentityScaleDiag  = 1.0 / (6 * math.pi * ai)

    muTTidentityScaleFar   = (1.0 / (8.0 * math.pi * dist)) * (1.0 + (ai ** 2 + aj ** 2) / (3 * (dist ** 2)))
    muTTrHatScaleFar       = (1.0 / (8.0 * math.pi * dist)) * (1.0 - (ai ** 2 + aj ** 2) / (dist ** 2))

    muTTidentityScaleClose = (1.0 / (6.0 * math.pi * ai * aj)) * ( ( 16.0 * (dist ** 3) * (ai + aj) - ((ai - aj) ** 2 + 3 * (dist ** 2)) ** 2 ) / (32.0 * (dist ** 3)))
    muTTrHatScaleClose     = (1.0 / (6.0 * math.pi * ai * aj)) * ( 3.0 * ((ai - aj) ** 2 - dist ** 2) ** 2 / (32.0 * (dist ** 3)))

    # ### rotational matricies
    muRRidentityScaleDiag  = 1.0 / (8 * math.pi * (ai ** 3))

    muRRidentityScaleFar   = -1.0 / (16.0 * math.pi * (dist ** 3))
    muRRrHatScaleFar       = (1.0 / (16.0 * math.pi * (dist ** 3))) * 3.0

    # convenience matrix, consistent with publication of Zuk et al
    mathcalA = (
                5.0 * (dist ** 6)
                - 27.0 * (dist ** 4) * (ai ** 2 + aj ** 2)
                + 32.0 * (dist ** 3) * (ai ** 3 + aj ** 3)
                - 9.0 * (dist ** 2) * ((ai ** 2 - aj ** 2) ** 2)
                - ((ai - aj) ** 4) * (ai ** 2 + 4 * aj * ai + aj ** 2)
            ) / (64.0 * (dist ** 3))
    mathcalB = (
                3.0
                * (((ai - aj) ** 2 - dist ** 2) ** 2)
                * (ai ** 2 + 4.0 * ai * aj + aj ** 2 - dist ** 2)
            ) / (64.0 * (dist ** 3))

    muRRidentityScaleClose = (1.0 / (8.0 * math.pi * (ai ** 3) * (aj ** 3))) * mathcalA
    muRRrHatScaleClose     = (1.0 / (8.0 * math.pi * (ai ** 3) * (aj ** 3))) * mathcalB

    # ### coupling matricies
    muRTScaleDiag          = 0.0

    muRTScaleFar           = 1.0 / (8 * math.pi * (dist ** 2))

    muRTScaleClose         = (1.0 / (16.0 * math.pi * (aj ** 3) * ai)) * ( ( ((aj - ai + dist) ** 2) * ( ai ** 2 + 2.0 * ai * (aj + dist) - 3.0 * ((aj - dist) ** 2))) / (8.0 * (dist ** 2)) )

    # solution branch indicators
    isFar = 1.0*(dist > ai + aj)
    isDiag = 1.0*(jnp.identity(n))

    # combine scale factors from branches
    muTTidentityScale = isDiag * muTTidentityScaleDiag + (1.0 - isDiag) * (isFar * muTTidentityScaleFar + (1.0 - isFar) * muTTidentityScaleClose)
    muTTrHatScale = (1.0 - isDiag) * (isFar * muTTrHatScaleFar + (1.0-isFar) * muTTrHatScaleClose)

    muRRidentityScale = isDiag * muRRidentityScaleDiag + (1.0 - isDiag) * (isFar * muRRidentityScaleFar + (1.0 - isFar) * muRRidentityScaleClose)
    muRRrHatScale = (1.0 - isDiag) * (isFar * muRRrHatScaleFar + (1.0 - isFar) * muRRrHatScaleClose)

    muRTScale = (1.0 - isDiag) * (isFar * muRTScaleFar + (1.0 - isFar) * muRTScaleClose)

    # construct large matricies
    muTT = (
                muTTidentityScale[:,:,jnp.newaxis,jnp.newaxis] * jnp.identity(3)[jnp.newaxis,jnp.newaxis,:,:] 
                + muTTrHatScale[:,:,jnp.newaxis,jnp.newaxis] * rHatMatrix[:,:,jnp.newaxis,:] * rHatMatrix[:,:,:,jnp.newaxis]
           )
    muRR = (
                muRRidentityScale[:,:,jnp.newaxis,jnp.newaxis] * jnp.identity(3)[jnp.newaxis,jnp.newaxis,:,:] 
                + muRRrHatScale[:,:,jnp.newaxis,jnp.newaxis] * rHatMatrix[:,:,jnp.newaxis,:] * rHatMatrix[:,:,:,jnp.newaxis]
           )
    muRT = (
                muRTScale[:,:,jnp.newaxis,jnp.newaxis] * epsilonRHatMatrix[:,:,:,:]
           )

    # flatten (2,2,n,n,3,3) tensor in the correct order
    return jax.lax.reshape(
        jnp.array([[muTT,muRT],[_transTranspose(muRT),muRR]]),
        (6*n,6*n),
        dimensions = (0,2,4,1,3,5)
    )


def muTT(centres,radii):
    """
    Returns grand mobility matrix in RPY approximation.

    Parameters
    ----------
    centers: jnp.array
        An ``N`` by 3 array describing locations of centres of ``N`` beads.
    radii: jnp.array
        An array of length ``N`` describing sizes of ``N`` beads.

    Returns
    -------
    jnp.array
        A ``3N`` by ``3N`` array containing translational mobility coefficients
        of the beads. Indicies are ordered: ``ux1,uy1,uz1,  ux2,uy2,uz2, ...,```.
        by bead index, then by coordinate.
    """

    # number of beads
    n = len(radii)

    displacements = centres[:, jnp.newaxis, :] - centres[jnp.newaxis, :, :]
    distances = jnp.sqrt(jnp.sum(displacements ** 2, axis=-1)) 

    # shorthand for radii, consistent with publication of Zuk et al
    a = radii 

    # normalized displacements and zeros for i==j
    rHatMatrix = displacements / (distances[:,:,jnp.newaxis] + jnp.identity(n)[:,:,jnp.newaxis])

    ai = a[:,jnp.newaxis]
    aj = a[jnp.newaxis,:]
    
    dist = distances + jnp.identity(n) # add identity to allow division

    # prefactors of matricies for each bead pair
    # matricies are {identity, r^hat r^hat, \\epsilon_ijk r^hat_k}
    # these are grouped by interaction type {TT,TR,RR}
    # and by solution branch {diagonal, close, far} for Rotne-Prager and Yakamava parts
    # numpy magic does all operations componentwise

    # ### translational matricies
    muTTidentityScaleDiag   = 1.0 / (6 * math.pi * ai)

    muTTidentityScaleFar    = (1.0 / (8.0 * math.pi * dist)) * (1.0 + (ai ** 2 + aj ** 2) / (3 * (dist ** 2)))
    muTTrHatScaleFar        = (1.0 / (8.0 * math.pi * dist)) * (1.0 - (ai ** 2 + aj ** 2) / (dist ** 2))

    muTTidentityScaleClose  = (1.0 / (6.0 * math.pi * ai * aj)) * ( ( 16.0 * (dist ** 3) * (ai + aj) - ((ai - aj) ** 2 + 3 * (dist ** 2)) ** 2 ) / (32.0 * (dist ** 3)))
    muTTrHatScaleClose      = (1.0 / (6.0 * math.pi * ai * aj)) * ( 3.0 * ((ai - aj) ** 2 - dist ** 2) ** 2 / (32.0 * (dist ** 3)))

    muTTidentityScaleInside = (1.0 / (6.0 * math.pi * jnp.maximum(ai,aj)))

    # solution branch indicators
    isFar = 1.0*(dist > ai + aj)
    isOutside = 1.0*(dist > jnp.maximum(ai,aj) - jnp.minimum(ai,aj))
    isDiag = 1.0*(jnp.identity(n))

    # combine scale factors from branches
    muTTidentityScale = isDiag * muTTidentityScaleDiag + (1.0 - isDiag) * (isOutside * (isFar * muTTidentityScaleFar + (1.0 - isFar) * muTTidentityScaleClose) + (1.0-isOutside) * muTTidentityScaleInside)
    muTTrHatScale = isOutside * ((1.0 - isDiag) * (isFar * muTTrHatScaleFar + (1.0-isFar) * muTTrHatScaleClose))

    # construct large matricies
    muTT = (
                muTTidentityScale[:,:,jnp.newaxis,jnp.newaxis] * jnp.identity(3)[jnp.newaxis,jnp.newaxis,:,:] 
                + muTTrHatScale[:,:,jnp.newaxis,jnp.newaxis] * rHatMatrix[:,:,jnp.newaxis,:] * rHatMatrix[:,:,:,jnp.newaxis]
           )
    # flatten (n,n,3,3) tensor in the correct order
    return jax.lax.reshape(
        muTT,
        (3*n,3*n),
        dimensions = (0,2,1,3)
    )

