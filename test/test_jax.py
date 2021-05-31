import pygrpy
import pygrpy.jax_grpy_tensors

import jax
import jax.numpy as jnp

import numpy as np

import json

centres_single = jnp.array([[0.,0,0]])
sizes_single = jnp.array([1.])

centres_two = jnp.array([[0.,0,0],[0,0,1]])
sizes_two = jnp.array([1.,1])

centres_four = jnp.array([[0.,0,0],[0,0,1],[0,0,2],[0,0,3]])
sizes_four = jnp.array([1.,1,1,1])

gmm_two = jnp.array(
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

gmm_two_tt = gmm_two[:6,:6]

gmm_four = jnp.array([
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

def test_mobility_single_bead():
    testmu = pygrpy.jax_grpy_tensors.mu(centres_single,sizes_single)
    #assert type(testmu) == jnp.ndarray , 'grpy_tesnors.mu should return jnp.array.'
    assert jnp.allclose(testmu,jnp.array([
                                            [0.05305164769729845,0,0,0,0,0],
                                            [0,0.05305164769729845,0,0,0,0],
                                            [0,0,0.05305164769729845,0,0,0],
                                            [0,0,0,0.039788735772973836,0,0],
                                            [0,0,0,0,0.039788735772973836,0],
                                            [0,0,0,0,0,0.039788735772973836]
                                           ])
                         ) ,  'grpy_tesnors.mu for single bead should return specified value.'

def test_mobility_two_beads():
    testmu = pygrpy.jax_grpy_tensors.mu(centres_two,sizes_two)

    #assert type(testmu) == jnp.ndarray , 'grpy_tesnors.mu should return jnp.array.'
    #print(json.dumps((testmu - gmm_two).tolist()))
    assert jnp.allclose(testmu, gmm_two) ,  'grpy_tesnors.mu for two beads should return specified value.'

def test_mobility_four_beads():
    testmu = pygrpy.jax_grpy_tensors.mu(centres_four,sizes_four)
    #assert type(testmu) == jnp.ndarray , 'grpy_tesnors.mu should return jnp.array.'
    #print(json.dumps((testmu - gmm_four).tolist()))
    assert jnp.allclose(testmu, gmm_four) ,  'grpy_tesnors.mu for four beads should return specified value.'

def test_mobility_manybeads_10():
    testmu = pygrpy.jax_grpy_tensors.mu( jnp.array(np.random.normal(size=(10,3))) , jnp.ones(10)  )

def test_mobility_manybeads_100():
    n = 100
    testmu = pygrpy.jax_grpy_tensors.mu( jnp.array(np.random.normal(size=(n,3))) , jnp.ones(n)  )

def test_mobility_manybeads_1000():
    n = 1000
    testmu = pygrpy.jax_grpy_tensors.mu( jnp.array(np.random.normal(size=(n,3))) , jnp.ones(n)  )

def test_jit_mobility_manybeads_10_repeats_1000():
    n = 10
    t = 1000
    fastmuA = jax.jit(pygrpy.jax_grpy_tensors.mu)
    for i in range(0,t):
        testmu = fastmuA( jnp.array(np.random.normal(size=(n,3))) , jnp.ones(n)  )

def test_jit_mobility_manybeads_1000_repeats_10():
    n = 20
    t = 10
    fastmuB = jax.jit(pygrpy.jax_grpy_tensors.mu)
    centres = jnp.array(np.random.normal(scale=50.0,size=(n,3)))
    sizes = jnp.ones(n)

    print(jax.make_jaxpr(pygrpy.jax_grpy_tensors.mu)(centres,sizes))

    for i in range(0,t):
        testmu = fastmuB( centres , sizes )

def test_muTT_two_beads():
    testmu = pygrpy.jax_grpy_tensors.muTT(centres_two,sizes_two)

    #assert type(testmu) == jnp.ndarray , 'grpy_tesnors.mu should return jnp.array.'
    #print(json.dumps((testmu - gmm_two).tolist()))
    assert jnp.allclose(testmu, gmm_two_tt) ,  'grpy_tesnors.muTT for two beads should return specified value.'

def test_jit():
    fastmuC = jax.jit(pygrpy.jax_grpy_tensors.mu)
    #fastmuC = pygrpy.jax_grpy_tensors.mu
    testmu = fastmuC(centres_four,sizes_four)
    assert jnp.allclose(testmu, gmm_four) ,  'grpy_tesnors.mu for four beads should return specified value.'

if __name__ == "__main__":
    test_jit_mobility_manybeads_1000_repeats_10()
    test_jit()
    test_mobility_single_bead()
    test_mobility_two_beads()
    test_mobility_four_beads()
