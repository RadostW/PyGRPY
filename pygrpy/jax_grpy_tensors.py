import math

import jax
import jax.numpy as jnp
import jax.ops

import json

_epsilon = jnp.array([[[0.,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0]]]) #levi-civita(3)

def _transTranspose(tensor):
    '''
    Returns :math:`a_jilk` given tensor :math:`a_ijkl`
    '''
    return jnp.transpose(tensor,[1,0,3,2])

def _epsilonVec(vec):
    '''
    Retruns an :math:`\\epsilon_ijk v_k`, a 3x3 matrix
    '''

    return jnp.array(
        [[ 0,      vec[2], -vec[1]],
         [-vec[2], 0,       vec[0]],
         [ vec[1], -vec[0], 0     ]]
    )

def _epsilonMatVec(mat,vec):
    '''
    Returns  a_ij \epsilon_mjk v_k

    '''
    return jnp.matmul(mat,jnp.transpose( _epsilonVec(vec)))

def _epsilonVecMat(vec,mat):
    '''
    Returns  \epsilon_ijk v_k a_im

    '''
    return jnp.matmul( jnp.transpose(_epsilonVec(vec)) , mat)

def mu(centres,radii):
    '''
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
        A ``6N`` by ``6N`` array containing translational mobility coefficients
        of the beads. Indicies are ordered: ``ux1,uy1,uz1,  ux2,uy2,uz2, ..., wx1,wy1,wz1, ...```.
        All translations before rotations, then by bead index, then by coordinate.

    '''
    n = len(radii) #number of beads

    #if len(centres) != len(radii):
    #    raise ValueError('Radii array and centres array of icompatible shapes')
   
    displacements = (centres[:, jnp.newaxis, :] - centres[jnp.newaxis, :, :])
    rHatMatrix = jnp.zeros_like(displacements)

    for i in range(0,n):
        for j in range(0,n):
            rHatMatrix = jax.lax.cond(
                i!=j,
                lambda _: jax.ops.index_update(rHatMatrix, jax.ops.index[i,j],  displacements[i,j] / jnp.linalg.norm(displacements[i,j])),
                lambda _: rHatMatrix,
                (i,j)
                )   #for i==j leave it alone

   
    distances = jnp.sqrt(jnp.sum((centres[:, jnp.newaxis, :] - centres[jnp.newaxis, :, :]) ** 2, axis = -1)) #calculate distances with numpy magic

    a = radii #shorthand for radii, consistent with publication

    #indicies: bead, bead, coord, coord
    muTT = jnp.empty([n,n,3,3]) #translation-translation
    muRR = jnp.empty([n,n,3,3]) #rotation-rotation
    muRT = jnp.empty([n,n,3,3]) #rotation-translation coupling

    for i in range(0,n):
        for j in range(0,n):    

            aSmall = jax.lax.min(radii[i],radii[j]) #pick larger and smaller bead
            aBig = jax.lax.max(radii[i],radii[j])

            TTidentityScale = 0.0 #scalar multiplier of id matrix
            TTrHatScale = 0.0 #scalar multiplier of \hat{r}\hat{r} matrix
            RRidentityScale = 0.0
            RRrHatScale = 0.0
            RTScale = 0.0

            (TTidentityScale,TTrHatScale,RRidentityScale,RRrHatScale,RTScale) = jax.lax.cond( 
                i==j,
                lambda _: ((1.0 / (6 * math.pi * aSmall)), 0.0, (1.0 / (8 * math.pi * (aSmall**3))), 0.0, 0.0),
                lambda _: (TTidentityScale,TTrHatScale,RRidentityScale,RRrHatScale,RTScale),
                operand=None )

            (TTidentityScale,TTrHatScale,RRidentityScale,RRrHatScale,RTScale) = jax.lax.cond( 
                jnp.bitwise_and(
                    distances[i][j] > a[i]+a[j] ,
                    jnp.bitwise_not(i==j)
                ),
                lambda _: (
                    (1.0 / (8.0 * math.pi * distances[i][j]))*(1.0 + (a[i]**2 + a[j]**2) / (3*(distances[i][j]**2))),
                    (1.0 / (8.0 * math.pi * distances[i][j]))*(1.0 - (a[i]**2 + a[j]**2) / (distances[i][j]**2)),
                    (-1.0 / (16.0 * math.pi * (distances[i][j]**3))),
                    (1.0 / (16.0 * math.pi * (distances[i][j]**3)))*3,
                    (1.0 / (8 * math.pi * (distances[i][j]**2) ))
                    ),
                lambda _: (TTidentityScale,TTrHatScale,RRidentityScale,RRrHatScale,RTScale),
                operand=None )

            mathcalA = ( 5.0* (distances[i][j]**6) 
                                - 27.0 * (distances[i][j]**4) * (a[i]**2 + a[j]**2) 
                                + 32.0 * (distances[i][j]**3) * (a[i]**3 + a[j]**3)
                                - 9.0 * (distances[i][j]**2) * ((a[i]**2 - a[j]**2)**2)
                                - ((a[i]-a[j])**4)*(a[i]**2 + 4*a[j]*a[i] + a[j]**2) ) / ( 64.0 * (distances[i][j]**3) )       
            mathcalB =  ( 3.0*(((a[i]-a[j])**2 - distances[i][j]**2)**2) * (a[i]**2 + 4.0*a[i]*a[j] + a[j]**2 - distances[i][j]**2) ) / (64.0* (distances[i][j]**3)) 

            (TTidentityScale,TTrHatScale,RRidentityScale,RRrHatScale,RTScale) = jax.lax.cond( 
                jnp.bitwise_and(
                    distances[i][j] > aBig - aSmall, 
                    jnp.bitwise_and(
                        distances[i][j] <= a[i]+a[j] , 
                        jnp.bitwise_and(
                            jnp.bitwise_not( distances[i][j] > a[i]+a[j]) ,
                            jnp.bitwise_not( i==j )
                        )
                    )
                ),
                lambda _: (
                    (1.0 / (6.0 * math.pi * a[i] * a[j])) * ((16.0*(distances[i][j]**3)*(a[i]+a[j])-((a[i]-a[j])**2 + 3*(distances[i][j]**2))**2)/(32.0 * (distances[i][j]**3))),
                    (1.0 / (6.0 * math.pi * a[i] * a[j])) * ( 3 * ((a[i]-a[j])**2 - distances[i][j]**2)**2 / (32 * (distances[i][j]**3)) ),
                    (1.0 / (8.0 * math.pi * (a[i]**3) * (a[j]**3))) * mathcalA,
                    (1.0 / (8.0 * math.pi * (a[i]**3) * (a[j]**3))) * mathcalB,
                    (1.0 / (16.0 * math.pi * (a[j]**3) * a[i])) * ( ( ((a[j] - a[i] + distances[i][j])**2)*(a[i]**2+2.0*a[i]*(a[j]+distances[i][j])-3.0*((a[j]-distances[i][j])**2))  ) / (8.0 * (distances[i][j]**2)))
                    ),
                lambda _: (TTidentityScale,TTrHatScale,RRidentityScale,RRrHatScale,RTScale),
                operand=None )

            #### TODO #### One bead entirely inside another
            
            # GRPY approximation is of form scalar * matrix + scalar * matrix
            muTT = jax.ops.index_update(muTT, jax.ops.index[i,j,:,:] , TTidentityScale * jnp.identity(3) + TTrHatScale * jnp.outer(rHatMatrix[i][j],rHatMatrix[i][j]))
            muRR = jax.ops.index_update(muRR, jax.ops.index[i,j,:,:] , RRidentityScale * jnp.identity(3) + RRrHatScale * jnp.outer(rHatMatrix[i][j],rHatMatrix[i][j]))
            muRT = jax.ops.index_update(muRT, jax.ops.index[i,j,:,:] , RTScale * _epsilonVec(rHatMatrix[i][j]))

    #if blockmatrix:
    #    return jnp.array([[muTT,muRT],[_transTranspose(muRT),muRR]])
    #else:
    return jnp.hstack(jnp.hstack(jnp.hstack(jnp.hstack(jnp.array([[muTT,muRT],[_transTranspose(muRT),muRR]])))))
