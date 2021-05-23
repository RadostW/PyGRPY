import math
import numpy as np

def transTranspose(tensor):
    '''
    Returns :math:`a_jilk` given tensor :math:`a_ijkl`
    '''
    return np.transpose(tensor,[1,0,3,2])

def epsilonVec(vec):
    '''
    Retruns an :math:`\epsilon_ijk v_k`, a 3x3 matrix
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

def mu(centres,radii,blockmatrix = False):
    '''
    Returns grand mobility matrix in RPY approximation.

    Parameters
    ----------
    centers: np.array
        An ``N`` by 3 array describing locations of centres of ``N`` beads.
    radii: np.array
        An array of length ``N`` describing sizes of ``N`` beads.
    bockmatrix : {True,False}
        Whether to retun rank 4 tensor instead of rank 2 tensor.

    Returns
    -------
    np.array
        A ``6N`` by ``6N`` array containing translational mobility coefficients
        of the beads. Indicies are ordered: ``ux1,uy1,uz1,  ux2,uy2,uz2, ..., wx1,wy1,wz1, ...```.
        All translations before rotations, then by bead index, then by coordinate.

        Unless ``blockmatrix`` is turned to ``True``, then ``np.array`` with ``size=(2,2,N,N,3,3)`` is returned

    Example
    -------
    >>> muTT(np.array([[0.0,0.0,0.0],[0.0,0.0,1.5]]),np.array([1.0,1.0])
    array([[0slfkjasdlfjkdlajksf0,0,0]])####### TODO ####### return value

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
    muRT = np.empty([n,n,3,3]) #rotation-translation coupling

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
                
                # Rotation-translation
                RTScale = 0.0

            elif distances[i][j] > a[i]+a[j]: #Far apart
                # Translation-translation
                TTidentityScale = (1.0 / (8.0 * math.pi * distances[i][j]))*(1.0 + (a[i]**2 + a[j]**2) / (3*(distances[i][j]**2)))
                TTrHatScale = (1.0 / (8.0 * math.pi * distances[i][j]))*(1.0 - (a[i]**2 + a[j]**2) / (distances[i][j]**2))

                # Rotation-rotation
                RRidentityScale = (-1.0 / (16.0 * math.pi * (distances[i][j]**3)))
                RRrHatScale = (1.0 / (16.0 * math.pi * (distances[i][j]**3)))*3
                
                # Rotation-translation
                RTScale = (1.0 / (8 * math.pi * (distances[i][j]**2) ))

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
                
                # Rotation-translation
                RTScale = (1.0 / (16.0 * math.pi * (a[j]**3) * a[i])) * ( ( ((a[j] - a[i] + distances[i][j])**2)*(a[i]**2+2.0*a[i]*(a[j]+distances[i][j])-3.0*((a[j]-distances[i][j])**2))  ) / (8.0 * (distances[i][j]**2)))

            else:
                raise NotImplementedError("One bead entirely inside another")
            
            # GRPY approximation is of form scalar * matrix + scalar * matrix
            muTT[i,j,:,:] = TTidentityScale * np.identity(3) + TTrHatScale * np.outer(rHatMatrix[i][j],rHatMatrix[i][j])
            muRR[i,j,:,:] = RRidentityScale * np.identity(3) + RRrHatScale * np.outer(rHatMatrix[i][j],rHatMatrix[i][j])
            muRT[i,j,:,:] = RTScale * epsilonVec(rHatMatrix[i][j])

    if blockmatrix:
        return np.array([[muTT,muRT],[transTranspose(muRT),muRR]])
    else:
        return np.hstack(np.hstack(np.hstack(np.hstack(np.array([[muTT,muRT],[transTranspose(muRT),muRR]])))))

