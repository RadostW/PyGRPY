import pygrpy
import numpy as np
import json

centres_single = np.array([[0,0,0]])
sizes_single = np.array([1])

centres_two = np.array([[0,0,0],[0,0,1]])
sizes_two = np.array([1,1])

centres_four = np.array([[0,0,0],[0,0,1],[0,0,2],[0,0,3]])
sizes_four = np.array([1,1,1,1])

gmm_two = np.array(
                [
                [0.05305164769729845,0,0,0.03813087178243326,0,0,0,0,0,0,-0.012433979929054324,0],
                [0,0.05305164769729845,0,0,0.03813087178243326,0,0,0,0,0.012433979929054324,0,0],
                [0,0,0.05305164769729845,0,0,0.04310446375405499,0,0,0,0,0,0],
                [0.03813087178243326,0,0,0.05305164769729845,0,0,0,0.012433979929054324,0,0,0,0],
                [0,0.03813087178243326,0,0,0.05305164769729845,0,-0.012433979929054324,0,0,0,0,0],
                [0,0,0.04310446375405499,0,0,0.05305164769729845,0,0,0,0,0,0],
                [0,0,0,0,-0.012433979929054324,0,0.039788735772973836,0,0,0.009325484946790743,0,0],
                [0,0,0,0.012433979929054324,0,0,0,0.039788735772973836,0,0,0.009325484946790743,0],
                [0,0,0,0,0,0,0,0,0.039788735772973836,0,0,0.018650969893581486],
                [0,0.012433979929054324,0,0,0,0,0.009325484946790743,0,0,0.039788735772973836,0,0],
                [-0.012433979929054324,0,0,0,0,0,0,0.009325484946790743,0,0,0.039788735772973836,0],
                [0,0,0,0,0,0,0,0,0.018650969893581486,0,0,0.039788735772973836]
                ]
                )

gmm_four = np.array([
                                        [0.05305164769729845,0,0,0.03813087178243326,0,0,0.023210095867568073,0,0,0.014245349844644952,0,0,0,0,0,0,-0.012433979929054324,0,0,-0.009947183943243459,0,0,-0.004420970641441538,0],
                                        [0,0.05305164769729845,0,0,0.03813087178243326,0,0,0.023210095867568073,0,0,0.014245349844644952,0,0,0,0,0.012433979929054324,0,0,0.009947183943243459,0,0,0.004420970641441538,0,0],
                                        [0,0,0.05305164769729845,0,0,0.04310446375405499,0,0,0.033157279810811534,0,0,0.024560948008008537,0,0,0,0,0,0,0,0,0,0,0,0],
                                        [0.03813087178243326,0,0,0.05305164769729845,0,0,0.03813087178243326,0,0,0.023210095867568073,0,0,0,0.012433979929054324,0,0,0,0,0,-0.012433979929054324,0,0,-0.009947183943243459,0],
                                        [0,0.03813087178243326,0,0,0.05305164769729845,0,0,0.03813087178243326,0,0,0.023210095867568073,0,-0.012433979929054324,0,0,0,0,0,0.012433979929054324,0,0,0.009947183943243459,0,0],
                                        [0,0,0.04310446375405499,0,0,0.05305164769729845,0,0,0.04310446375405499,0,0,0.033157279810811534,0,0,0,0,0,0,0,0,0,0,0,0],
                                        [0.023210095867568073,0,0,0.03813087178243326,0,0,0.05305164769729845,0,0,0.03813087178243326,0,0,0,0.009947183943243459,0,0,0.012433979929054324,0,0,0,0,0,-0.012433979929054324,0],
                                        [0,0.023210095867568073,0,0,0.03813087178243326,0,0,0.05305164769729845,0,0,0.03813087178243326,0,-0.009947183943243459,0,0,-0.012433979929054324,0,0,0,0,0,0.012433979929054324,0,0],
                                        [0,0,0.033157279810811534,0,0,0.04310446375405499,0,0,0.05305164769729845,0,0,0.04310446375405499,0,0,0,0,0,0,0,0,0,0,0,0],
                                        [0.014245349844644952,0,0,0.023210095867568073,0,0,0.03813087178243326,0,0,0.05305164769729845,0,0,0,0.004420970641441538,0,0,0.009947183943243459,0,0,0.012433979929054324,0,0,0,0],
                                        [0,0.014245349844644952,0,0,0.023210095867568073,0,0,0.03813087178243326,0,0,0.05305164769729845,0,-0.004420970641441538,0,0,-0.009947183943243459,0,0,-0.012433979929054324,0,0,0,0,0],
                                        [0,0,0.024560948008008537,0,0,0.033157279810811534,0,0,0.04310446375405499,0,0,0.05305164769729845,0,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,-0.012433979929054324,0,0,-0.009947183943243459,0,0,-0.004420970641441538,0,0.039788735772973836,0,0,0.009325484946790743,0,0,-0.0024867959858108648,0,0,-0.0007368284402402563,0,0],
                                        [0,0,0,0.012433979929054324,0,0,0.009947183943243459,0,0,0.004420970641441538,0,0,0,0.039788735772973836,0,0,0.009325484946790743,0,0,-0.0024867959858108648,0,0,-0.0007368284402402563,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.039788735772973836,0,0,0.018650969893581486,0,0,0.004973591971621729,0,0,0.0014736568804805126],
                                        [0,0.012433979929054324,0,0,0,0,0,-0.012433979929054324,0,0,-0.009947183943243459,0,0.009325484946790743,0,0,0.039788735772973836,0,0,0.009325484946790743,0,0,-0.0024867959858108648,0,0],
                                        [-0.012433979929054324,0,0,0,0,0,0.012433979929054324,0,0,0.009947183943243459,0,0,0,0.009325484946790743,0,0,0.039788735772973836,0,0,0.009325484946790743,0,0,-0.0024867959858108648,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.018650969893581486,0,0,0.039788735772973836,0,0,0.018650969893581486,0,0,0.004973591971621729],
                                        [0,0.009947183943243459,0,0,0.012433979929054324,0,0,0,0,0,-0.012433979929054324,0,-0.0024867959858108648,0,0,0.009325484946790743,0,0,0.039788735772973836,0,0,0.009325484946790743,0,0],
                                        [-0.009947183943243459,0,0,-0.012433979929054324,0,0,0,0,0,0.012433979929054324,0,0,0,-0.0024867959858108648,0,0,0.009325484946790743,0,0,0.039788735772973836,0,0,0.009325484946790743,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.004973591971621729,0,0,0.018650969893581486,0,0,0.039788735772973836,0,0,0.018650969893581486],
                                        [0,0.004420970641441538,0,0,0.009947183943243459,0,0,0.012433979929054324,0,0,0,0,-0.0007368284402402563,0,0,-0.0024867959858108648,0,0,0.009325484946790743,0,0,0.039788735772973836,0,0],
                                        [-0.004420970641441538,0,0,-0.009947183943243459,0,0,-0.012433979929054324,0,0,0,0,0,0,-0.0007368284402402563,0,0,-0.0024867959858108648,0,0,0.009325484946790743,0,0,0.039788735772973836,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0014736568804805126,0,0,0.004973591971621729,0,0,0.018650969893581486,0,0,0.039788735772973836]
                                       ])

proj_four = np.array(
    [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[1,0,0,0,1,0],[0,1,0,-1,0,0],
    [0,0,1,0,0,0],[1,0,0,0,2,0],[0,1,0,-2,0,0],[0,0,1,0,0,0],[1,0,0,0,3,0],
    [0,1,0,-3,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],
    [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,1,0,0],[0,0,0,0,1,0],
    [0,0,0,0,0,1],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
                    )

cong_four = np.array([
                    [0.04708754170423306,0,0,0,-0.009885031401043392,0],
                    [0,0.04708754170423306,0,0.009885031401043392,0,0],
                    [0,0,0.03876335168633891,0,0,0],
                    [0,0.009885031401043392,0,0.006590020934028927,0,0],
                    [-0.009885031401043392,0,0,0,0.006590020934028927,0],
                    [0,0,0,0,0,0.017665748920274064]
                    ])

cent_four = np.array([0.000,0.000,-1.500])

mob_four = np.array([
                    [0.03225999460266797,0,0,0,0,0],
                    [0,0.03225999460266797,0,0,0,0],
                    [0,0,0.03876335168633891,0,0,0],
                    [0,0,0,0.006590020934028927,0,0],
                    [0,0,0,0,0.006590020934028927,0],
                    [0,0,0,0,0,0.017665748920274064]
                    ])

def test_mobility_single_bead():
    testmu = pygrpy.grpy_tensors.mu(centres_single,sizes_single)
    assert type(testmu) == np.ndarray , 'grpy_tesnors.mu should return np.array.'
    assert np.allclose(testmu,np.array([
                                            [0.05305164769729845,0,0,0,0,0],
                                            [0,0.05305164769729845,0,0,0,0],
                                            [0,0,0.05305164769729845,0,0,0],
                                            [0,0,0,0.039788735772973836,0,0],
                                            [0,0,0,0,0.039788735772973836,0],
                                            [0,0,0,0,0,0.039788735772973836]
                                           ])
                         ) ,  'grpy_tesnors.mu for single bead should return specified value.'

def test_mobility_two_beads():
    testmu = pygrpy.grpy_tensors.mu(centres_two,sizes_two)
    assert type(testmu) == np.ndarray , 'grpy_tesnors.mu should return np.array.'
    #print(json.dumps((testmu - gmm_two).tolist()))
    assert np.allclose(testmu, gmm_two) ,  'grpy_tesnors.mu for four beads should return specified value.'


def test_mobility_four_beads():
    testmu = pygrpy.grpy_tensors.mu(centres_four,sizes_four)
    assert type(testmu) == np.ndarray , 'grpy_tesnors.mu should return np.array.'
    #print(json.dumps((testmu - gmm_four).tolist()))
    assert np.allclose(testmu, gmm_four) ,  'grpy_tesnors.mu for four beads should return specified value.'

def test_projection_four_beads():
    testproj = pygrpy.grpy_tensors.rigidProjectionMatrix(centres_four)
    assert type(testproj) == np.ndarray
    #print(json.dumps((testproj - proj_four).tolist()))
    assert np.allclose(testproj, proj_four)

def test_conglomerate_four_beads():
    testcong = pygrpy.grpy_tensors.conglomerateMobilityMatrix(centres_four,sizes_four)
    assert type(testcong) == np.ndarray
    #print(json.dumps((testcong - cong_four).tolist()))
    assert np.allclose(testcong, cong_four)

def test_mobility_centre_four_beads():
    testcent = pygrpy.grpy_tensors.mobilityCentre(
        pygrpy.grpy_tensors.conglomerateMobilityMatrix(centres_four,sizes_four)
                                                )
    assert type(testcent) == np.ndarray
    assert np.allclose(testcent, cent_four)

def test_mobility_matrix_at_centre():
    testmob = pygrpy.grpy_tensors.conglomerateMobilityMatrixAtCentre(centres_four,sizes_four)
    assert type(testmob) == np.ndarray
    assert np.allclose(testmob, mob_four)

def test_hydrosize():
    testsize = pygrpy.grpy.stokesRadius(centres_four,sizes_four)
    assert np.allclose(testsize, 1.5409546371938094)

if __name__ == "__main__":
    test_mobility_single_bead()
    test_mobility_two_beads()
    test_mobility_four_beads()  
    test_projection_four_beads()  