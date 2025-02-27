import math
import numpy as np

_epsilon = np.array([[[0,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0]]]) #levi-civita(3)

def _transTranspose(tensor):
    '''
    Returns :math:`a_jilk` given tensor :math:`a_ijkl`
    '''
    return np.transpose(tensor,[1,0,3,2])

def _epsilonVec(vec):
    '''
    Retruns an :math:`\\epsilon_ijk v_k`, a 3x3 matrix
    '''
    ret = np.zeros((3,3))
    ret[0][0] = 0
    ret[1][1] = 0
    ret[2][2] = 0
    
    ret[0][1] = vec[2]
    ret[1][2] = vec[0]
    ret[2][0] = vec[1]

    ret[1][0] = -vec[2]
    ret[2][1] = -vec[0]
    ret[0][2] = -vec[1]
    
    return ret

def _lapackinv(mat):
    import scipy as sp
    import scipy.linalg
    
    zz , _ = sp.linalg.lapack.dpotrf(mat, False, False) #cholesky decompose
    inv_M , info = sp.linalg.lapack.dpotri(zz) #invert triangle
    inv_M = np.triu(inv_M) + np.triu(inv_M, k=1).T #combine triangles
    return inv_M

def _epsilonMatVec(mat,vec):
    '''
    Returns  a_ij \epsilon_mjk v_k

    '''
    return np.matmul(mat,np.transpose( _epsilonVec(vec)))

def _epsilonVecMat(vec,mat):
    '''
    Returns  \epsilon_ijk v_k a_im

    '''
    return np.matmul( np.transpose(_epsilonVec(vec)) , mat)

def rigidProjectionMatrix(centres):
    '''
    Returns rigid projection matrix. When this matrix is multiplied with 
    friction matrix it gives friction matrix of rigid arrangement of beads.

    Parameters
    ----------
    centers: np.array
        An ``N`` by 3 array describing locations of centres of ``N`` beads.

    Returns
    -------
    np.array
        A ``6`` by ``6N`` matrix containing projection coefficients.
    '''

    n = len(centres)
    
    tnBlocks = np.zeros((2*n,2,3,3)) # 2n x 2 of 3x3 blocks to concatenate

    for i in range(0,n):
        tnBlocks[i,0,:,:] = np.identity(3)

        tmp = np.zeros((3,3))

        tmp[0,0] = 0
        tmp[0,1] = centres[i][2]
        tmp[0,2] = -centres[i][1]
        tmp[1,0] = -centres[i][2]
        tmp[1,1] = 0
        tmp[1,2] = centres[i][0]
        tmp[2,0] = centres[i][1]
        tmp[2,1] = -centres[i][0]
        tmp[2,2] = 0

        tnBlocks[i,1] = tmp

        tnBlocks[n+i,0] = np.zeros((3,3))
        tnBlocks[n+i,1] = np.identity(3)

    tn = np.hstack(np.hstack(tnBlocks))
    return tn

def mu(centres,radii,blockmatrix = False):
    '''
    Returns grand mobility matrix in RPY approximation.

    Parameters
    ----------
    centers: np.array
        An ``N`` by 3 array describing locations of centres of ``N`` beads.
    radii: np.array
        An array of length ``N`` describing sizes of ``N`` beads.
    blockmatrix : {True,False}
        Whether to retun rank 4 tensor instead of rank 2 tensor.

    Returns
    -------
    np.array
        A ``6N`` by ``6N`` array containing translational mobility coefficients
        of the beads. Indicies are ordered: ``ux1,uy1,uz1,  ux2,uy2,uz2, ..., wx1,wy1,wz1, ...```.
        All translations before rotations, then by bead index, then by coordinate.

        Unless ``blockmatrix`` is turned to ``True``, then ``np.array`` with ``size=(2,2,N,N,3,3)`` is returned

    '''
    n = len(radii) #number of beads

    if len(centres) != len(radii):
        raise ValueError('Radii array and centres array of icompatible shapes')
   
    displacements = (centres[:, np.newaxis, :] - centres[np.newaxis, :, :])
    rHatMatrix = np.zeros_like(displacements)

    for i in range(0,n):
        for j in range(0,n):
            if i!=j:
                rHatMatrix[i,j] = displacements[i,j] / np.linalg.norm(displacements[i,j])

   
    distances = np.sqrt(np.sum((centres[:, np.newaxis, :] - centres[np.newaxis, :, :]) ** 2, axis = -1)) #calculate distances with numpy magic

    a = radii #shorthand for radii, consistent with publication

    #indicies: bead, bead, coord, coord
    muTT = np.empty([n,n,3,3]) #translation-translation
    muRR = np.empty([n,n,3,3]) #rotation-rotation
    muTR = np.empty([n,n,3,3]) #translation-rotation coupling

    for i in range(0,n):
        for j in range(0,n):    

            aSmall = min(radii[i],radii[j]) #pick larger and smaller bead
            aBig = max(radii[i],radii[j])

            TTidentityScale = 0.0 #scalar multiplier of id matrix
            TTrHatScale = 0.0 #scalar multiplier of \hat{r}\hat{r} matrix

            if i == j:
                # Translation-translation
                TTidentityScale = (1.0 / (6 * math.pi * aSmall))
                TTrHatScale = 0.0

                # Rotation-rotation
                RRidentityScale = (1.0 / (8 * math.pi * (aSmall**3)))
                RRrHatScale = 0.0
                
                # Translation-rotation
                TRScale = 0.0

            elif distances[i][j] > a[i]+a[j]: #Far apart
                # Translation-translation
                TTidentityScale = (1.0 / (8.0 * math.pi * distances[i][j]))*(1.0 + (a[i]**2 + a[j]**2) / (3*(distances[i][j]**2)))
                TTrHatScale = (1.0 / (8.0 * math.pi * distances[i][j]))*(1.0 - (a[i]**2 + a[j]**2) / (distances[i][j]**2))

                # Rotation-rotation
                RRidentityScale = (-1.0 / (16.0 * math.pi * (distances[i][j]**3)))
                RRrHatScale = (1.0 / (16.0 * math.pi * (distances[i][j]**3)))*3
                
                # Translation-rotation
                TRScale = (1.0 / (8 * math.pi * (distances[i][j]**2) ))

            elif distances[i][j] > aBig - aSmall and distances[i][j] <= a[i]+a[j]: #Close together
                # Translation-translation
                TTidentityScale = (1.0 / (6.0 * math.pi * a[i] * a[j])) * ((16.0*(distances[i][j]**3)*(a[i]+a[j])-((a[i]-a[j])**2 + 3*(distances[i][j]**2))**2)/(32.0 * (distances[i][j]**3)))
                TTrHatScale = (1.0 / (6.0 * math.pi * a[i] * a[j])) * ( 3 * ((a[i]-a[j])**2 - distances[i][j]**2)**2 / (32 * (distances[i][j]**3)) )

                # Rotation-rotation
                mathcalA = ( 5.0* (distances[i][j]**6) 
                                - 27.0 * (distances[i][j]**4) * (a[i]**2 + a[j]**2) 
                                + 32.0 * (distances[i][j]**3) * (a[i]**3 + a[j]**3)
                                - 9.0 * (distances[i][j]**2) * ((a[i]**2 - a[j]**2)**2)
                                - ((a[i]-a[j])**4)*(a[i]**2 + 4*a[j]*a[i] + a[j]**2) ) / ( 64.0 * (distances[i][j]**3) )       
                mathcalB =  ( 3.0*(((a[i]-a[j])**2 - distances[i][j]**2)**2) * (a[i]**2 + 4.0*a[i]*a[j] + a[j]**2 - distances[i][j]**2) ) / (64.0* (distances[i][j]**3)) 
                RRidentityScale = (1.0 / (8.0 * math.pi * (a[i]**3) * (a[j]**3))) * mathcalA
                RRrHatScale = (1.0 / (8.0 * math.pi * (a[i]**3) * (a[j]**3))) * mathcalB
                
                # Translation-rotation
                TRScale = (1.0 / (16.0 * math.pi * (a[j]**3) * a[i])) * ( ( ((a[j] - a[i] + distances[i][j])**2)*(a[i]**2+2.0*a[i]*(a[j]+distances[i][j])-3.0*((a[j]-distances[i][j])**2))  ) / (8.0 * (distances[i][j]**2)))

            else:
                raise NotImplementedError("One bead entirely inside another")
            
            # GRPY approximation is of form scalar * matrix + scalar * matrix
            muTT[i,j,:,:] = TTidentityScale * np.identity(3) + TTrHatScale * np.outer(rHatMatrix[i][j],rHatMatrix[i][j])
            muRR[i,j,:,:] = RRidentityScale * np.identity(3) + RRrHatScale * np.outer(rHatMatrix[i][j],rHatMatrix[i][j])
            muTR[i,j,:,:] = TRScale * _epsilonVec(rHatMatrix[i][j])

    if blockmatrix:
        return np.array([[muTT,muTR],[_transTranspose(muTR),muRR]])
    else:
        return np.hstack(np.hstack(np.hstack(np.hstack(np.array([[muTT,muTR],[_transTranspose(muTR),muRR]])))))
        
        
        
def muTT(centres,radii):
    """
    Returns grand mobility matrix in RPY approximation.

    Parameters
    ----------
    centers: np.array
        An ``N`` by 3 array describing locations of centres of ``N`` beads.
    radii: np.array
        An array of length ``N`` describing sizes of ``N`` beads.

    Returns
    -------
    np.array
        A ``3N`` by ``3N`` array containing translational mobility coefficients
        of the beads. Indicies are ordered: ``ux1,uy1,uz1,  ux2,uy2,uz2, ...,```.
        by bead index, then by coordinate.
    """

    # number of beads
    n = len(radii)

    displacements = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    distances = np.sqrt(np.sum(displacements ** 2, axis=-1)) 

    # shorthand for radii, consistent with publication of Zuk et al
    a = radii 

    # normalized displacements and zeros for i==j
    rHatMatrix = displacements / (distances[:,:,np.newaxis] + np.identity(n)[:,:,np.newaxis])

    ai = a[:,np.newaxis]
    aj = a[np.newaxis,:]
    
    dist = distances + np.identity(n) # add identity to allow division

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

    muTTidentityScaleInside = (1.0 / (6.0 * math.pi * np.maximum(ai,aj)))

    # solution branch indicators
    isFar = 1.0*(dist > ai + aj)
    isOutside = 1.0*(dist > np.maximum(ai,aj) - np.minimum(ai,aj))
    isDiag = 1.0*(np.identity(n))

    # combine scale factors from branches
    muTTidentityScale = isDiag * muTTidentityScaleDiag + (1.0 - isDiag) * (isOutside * (isFar * muTTidentityScaleFar + (1.0 - isFar) * muTTidentityScaleClose) + (1.0-isOutside) * muTTidentityScaleInside)
    muTTrHatScale = isOutside * ((1.0 - isDiag) * (isFar * muTTrHatScaleFar + (1.0-isFar) * muTTrHatScaleClose))

    # construct large matricies
    muTT = (
                muTTidentityScale[:,:,np.newaxis,np.newaxis] * np.identity(3)[np.newaxis,np.newaxis,:,:] 
                + muTTrHatScale[:,:,np.newaxis,np.newaxis] * rHatMatrix[:,:,np.newaxis,:] * rHatMatrix[:,:,:,np.newaxis]
           )
    # flatten (n,n,3,3) tensor in the correct order
    return muTT

def muTT_trace(centres,radii):
    """
    Returns beadwise trace of grand mobility matrix in RPY approximation.

    Parameters
    ----------
    centers: np.array
        An ``N`` by 3 array describing locations of centres of ``N`` beads.
    radii: np.array
        An array of length ``N`` describing sizes of ``N`` beads.

    Returns
    -------
    np.array
        A ``N`` by ``N`` array containing traces of translational mobility coefficients
        of the beads.
    """

    # number of beads
    n = len(radii)

    displacements = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    distances = np.sqrt(np.sum(displacements ** 2, axis=-1)) 

    # shorthand for radii, consistent with publication of Zuk et al
    a = radii 
    
    ai = a[:,np.newaxis]
    aj = a[np.newaxis,:]
    
    dist = distances + np.identity(n) # add identity to allow division

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

    muTTidentityScaleInside = (1.0 / (6.0 * math.pi * np.maximum(ai,aj)))

    # solution branch indicators
    isFar = 1.0*(dist > ai + aj)
    isOutside = 1.0*(dist > np.maximum(ai,aj) - np.minimum(ai,aj))
    isDiag = 1.0*(np.identity(n))

    # combine scale factors from branches
    muTTidentityScale = isDiag * muTTidentityScaleDiag + (1.0 - isDiag) * (isOutside * (isFar * muTTidentityScaleFar + (1.0 - isFar) * muTTidentityScaleClose) + (1.0-isOutside) * muTTidentityScaleInside)
    muTTrHatScale = isOutside * ((1.0 - isDiag) * (isFar * muTTrHatScaleFar + (1.0-isFar) * muTTrHatScaleClose))

    # construct large matricies
    ret_muTT_trace = (
                3 *muTTidentityScale[:,:] 
                + muTTrHatScale[:,:]
           )
    # flatten (n,n,3,3) tensor in the correct order
    return ret_muTT_trace

import numpy as np
import math

def muTT_trace_vectorised(list_of_centres, radii):
    """
    Returns beadwise trace of grand mobility matrix in RPY approximation.
    This version uses broadcasting for faster computation.

    Parameters
    ----------
    list_of_centres: np.array
        A ``k`` by ``n`` by 3 array, where each ``n`` by 3 slice represents the locations of centres of ``n`` beads for a specific time step.
    radii: np.array
        An array of length ``n`` describing sizes of ``n`` beads.

    Returns
    -------
    np.array
        A ``k`` by ``n`` by n array containing traces of translational mobility coefficients
        of the beads for each time step in `list_of_centres`.
    """
    
    k, n, _ = list_of_centres.shape  # k: time steps, n: beads
    
    # shorthand for radii, consistent with publication of Zuk et al
    a = radii 
    
    ai = a[np.newaxis,:,np.newaxis]
    aj = a[np.newaxis,np.newaxis,:]
    
    # Calculate displacements and distances
    displacements = list_of_centres[:, :, np.newaxis, :] - list_of_centres[:, np.newaxis, :, :]
    distances = np.linalg.norm(displacements, axis=-1)  # Shape (k, n, n)

    # Add identity to distances (for self interactions)
    dist = distances + np.identity(n)  # Shape (k, n, n)

    # Calculate prefactors for each matrix type
    muTTidentityScaleDiag = 1.0 / (6 * math.pi * ai)  # Shape (n, 1)
    
    muTTidentityScaleFar = (1.0 / (8.0 * math.pi * dist)) * (1.0 + (ai ** 2 + aj ** 2) / (3 * dist ** 2))
    muTTrHatScaleFar = (1.0 / (8.0 * math.pi * dist)) * (1.0 - (ai ** 2 + aj ** 2) / (dist ** 2))
    
    muTTidentityScaleClose = (1.0 / (6.0 * math.pi * ai * aj)) * (
        (16.0 * (dist ** 3) * (ai + aj) - ((ai - aj) ** 2 + 3 * (dist ** 2)) ** 2) / (32.0 * (dist ** 3))
    )
    muTTrHatScaleClose = (1.0 / (6.0 * math.pi * ai * aj)) * (
        3.0 * ((ai - aj) ** 2 - dist ** 2) ** 2 / (32.0 * (dist ** 3))
    )
    
    muTTidentityScaleInside = 1.0 / (6.0 * math.pi * np.maximum(ai, aj))  # Shape (n, n)

    # Solution branch indicators using broadcasting
    isFar = 1.0 * (dist > ai + aj)  # Shape (k, n, n)
    isOutside = 1.0 * (dist > np.maximum(ai, aj) - np.minimum(ai, aj))  # Shape (k, n, n)
    isDiag = 1.0 * (np.eye(n) == 1)  # Shape (n, n), same for all time steps
    
    # Combine scale factors from branches
    muTTidentityScale = isDiag * muTTidentityScaleDiag + (1.0 - isDiag) * (isOutside * (isFar * muTTidentityScaleFar + (1.0 - isFar) * muTTidentityScaleClose) + (1.0-isOutside) * muTTidentityScaleInside)
    muTTrHatScale = isOutside * ((1.0 - isDiag) * (isFar * muTTrHatScaleFar + (1.0-isFar) * muTTrHatScaleClose))

    # construct large matricies
    ret_muTT_trace = (
                3 *muTTidentityScale[:,:] 
                + muTTrHatScale[:,:]
           )
    # flatten (n,n,3,3) tensor in the correct order
    return ret_muTT_trace    

def conglomerateMobilityMatrix(centres,radii):
    '''
    Returns mobility matrix centred at `[0,0,0]` of bead conglomerate specified 
    by `centres` and `radii`. Treats conglomerate as rigid body.
    

    Parameters
    ----------
    centers: np.array
        An ``N`` by 3 array describing locations of centres of ``N`` beads.
    radii: np.array
        An array of length ``N`` describing sizes of ``N`` beads.
    
    Returns
    -------
    np.array
        A `6` by `6` array specifying mobility matrix centred at `[0,0,0]`. 
        Indicies are ordered `ux,uy,uz,wx,wy,wz`.

    '''
    rpm = rigidProjectionMatrix(centres)
    gmm = mu(centres,radii) #grand mobility matrix
    gmmi = _lapackinv(gmm) #lapack inverse is fast
    
    fri = np.matmul(np.matmul(np.transpose(rpm),gmmi),rpm) # p'. inv(M) . p

    return _lapackinv(fri)

def mobilityCentre(mobility_matrix):
    '''
    Returns mobility cetre realtive to zero centered mobility matrix `mobility_matrix`.

    Parameters
    ----------
    mobility_matrix: np.array
        A `6` by `6` array specifying mobility matrix centred at `[0,0,0]`

    Returns
    -------
    np.array
        A length `3` vector. Mobility centres location relative to `[0,0,0]`.

    '''
    ret = np.zeros(3)

    # Rewrite mobility matrix in block form [[a,b],[b^T,c]]
    a = mobility_matrix[:3,:3]
    b = mobility_matrix[:3,3:6]
    c = mobility_matrix[3:6,3:6]

    # x_c = 1/2 (trC 1 - C)^-1 . (\\epsilon : (b - b^T))
    tmp1 = 0.5 * _lapackinv( np.trace(c)*np.identity(3) - c ) 
    tmp2 = np.tensordot( _epsilon ,  b - np.transpose(b) )

    return np.matmul(tmp1,tmp2)

def conglomerateMobilityMatrixAtCentre(centres,radii):

    mobility_matrix = conglomerateMobilityMatrix(centres,radii)
    centreLocation = mobilityCentre(mobility_matrix)

    # Rewrite mobility matrix in block form [[a,b],[b^T,c]]
    a1 = mobility_matrix[:3,:3]
    b1 = mobility_matrix[:3,3:6]
    c1 = mobility_matrix[3:6,3:6]

    # Microhydrodynamics: Principles and Selected Applications
    # S. Kim, S Karilla
    # 5.3.1, page 117

    a2 = (a1  
            - _epsilonVecMat( centreLocation , _epsilonMatVec( c1 , centreLocation ) )
            - _epsilonVecMat( centreLocation, b1 ) 
            + _epsilonMatVec( np.transpose(b1) , centreLocation)          
          )
    b2 = b1 + _epsilonMatVec( c1, centreLocation)
    c2 = c1

    return np.hstack(np.hstack( [[a2,b2],[np.transpose(b2),c2]] ))
