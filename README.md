![Tests](https://github.com/RadostW/PyGRPY/actions/workflows/tests.yml/badge.svg)

# PyGRPY

Python port of Generalized Rotne Prager Yakamawa hydrodynamic tensors.

Now also supports jax, and jax.grad.


# Original code

Original code in Fortran90 by Pawel Jan Zuk here:
https://github.com/pjzuk/GRPY

# License

This software is licensed under GNU GPLv3

Copyright (c) Pawel Jan Zuk (2017) - unported code.

Copyright (c) Radost Waszkiewicz (2021) - python port.

# How to cite

Waszkiewicz, R., Bartczak M., Kolasa K. and Lisicki M. *Pychastic: Precise Brownian Dynamics using 
Taylor-Ito integrators in Python*; SciPost Physics Codebases (2023)

https://scipost.org/SciPostPhysCodeb.11

```bibtex
@article{Waszkiewicz_2023,
	title        = {Pychastic: Precise Brownian dynamics using Taylor-It{\=o} integrators in Python},
	author       = {Waszkiewicz, Radost and Bartczak, Maciej and Kolasa, Kamil and Lisicki, Maciej},
	year         = 2023,
	journal      = {SciPost Physics Codebases},
	pages        = {11}
}
```
**and**

Zuk, P. J., Cichocki, B. and Szymczak, P. *GRPY: an accurate bead method for calculation of hydrodynamic properties of rigid biomacromolecules*; Biophys. J. (2018)

# Examples

## Hydrodynamic size of rigid conglomerate of beads
```python
    # Copyright (C) Radost Waszkiewicz 2022
    # This software is distributed under MIT license
    # Test if line of four identical beads has correct hydrodynamic size
    
    import pygrpy
    import numpy as np
    import json

    centres_four = np.array([[0,0,0],[0,0,1],[0,0,2],[0,0,3]])
    sizes_four = np.array([1,1,1,1])

    def test_hydrosize():
        testsize = pygrpy.grpy.stokesRadius(centres_four,sizes_four)
        assert np.allclose(testsize, 1.5409546371938094)

    if __name__ == "__main__":
        test_hydrosize()
```

```python    
    # Copyright (C) Radost Waszkiewicz 2024
    # This software is distributed under MIT license
    # Load shape of Lysozyme-C from different databases. Compare hydrodynamic size

    import pygrpy.pdb_loader
    import pygrpy.grpy

    pdb_content = pygrpy.pdb_loader.get_pdb_from_alphafold("P61626")
    coordinates, radii = pygrpy.pdb_loader.centres_and_radii(pdb_content)
    alphafold_size = pygrpy.grpy.stokesRadius(coordinates, radii)

    pdb_content = pygrpy.pdb_loader.get_pdb_from_pdb("253L")
    coordinates, radii = pygrpy.pdb_loader.centres_and_radii(pdb_content)
    pdb_size = pygrpy.grpy.stokesRadius(coordinates, radii)

    print("Alphafold size [Ang]:")
    print(alphafold_size)
    print("Protein Data Bank size [Ang]:")
    print(pdb_size)
```    

## Hydrodynamic radius for conformational ensemble
```python
    # Copyright (C) Radost Waszkiewicz 2022
    # This software is distributed under MIT license
    # Load an ensemble from a .pdb file and compute R_h using locations of C_alpha atoms

    import argparse
    import numpy as np
    import pygrpy
    from tqdm import tqdm

    # Console arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="test.pdb", help="specify input file")
    parser.add_argument(
        "-s",
        "--sigmas",
        help="compute standard deviations using bootstrap",
        action="store_true",
    )

    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        contents = f.read()

    lines = contents.splitlines()

    ensemble = list()
    residues = list()
    for line in lines:
        if "ATOM" in line:
            if "CA" in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                residues.append([x, y, z])
        elif "END" in line:
            ensemble.append(residues)
            residues = list()

    ensemble = np.array(ensemble)
    (ensemble_size, molecule_size, _) = ensemble.shape

    hydrodynamic_size = pygrpy.grpy.ensembleAveragedStokesRadius(
        ensemble, 4.2 * np.ones(molecule_size)
    )  # sizes in angstroms

    centre_of_mass = np.mean(ensemble, axis=1)  # shape = (conformer,3)
    gyration_radius = np.sqrt(3) * np.sqrt(np.mean((ensemble - centre_of_mass.reshape(-1, 1, 3)) ** 2))

    bootstrap_rounds = 5
    if args.sigmas:
        hydrodynamic_sizes_stats = np.zeros(bootstrap_rounds)
        for i in tqdm(range(bootstrap_rounds)):
            chosen = np.random.choice(np.arange(ensemble_size), ensemble_size)
            hydrodynamic_sizes_stats[i] = pygrpy.grpy.ensembleAveragedStokesRadius(
                ensemble[chosen], 4.2 * np.ones(molecule_size)
            )

        print(
            f"Hydrodynamic radius [Ang] = {hydrodynamic_size:.4f} +/- {np.std(hydrodynamic_sizes_stats):.4f}"
        )
        print(f"Gyration radius     [Ang] = {gyration_radius:.4f}")

    else:
        print(f"Hydrodynamic radius [Ang] = {hydrodynamic_size:.2f}")
        print(f"Gyration radius     [Ang] = {gyration_radius:.2f}")
```
## Grand mobility matrices
```python
# Copyright (C) Radost Waszkiewicz 2025
# This software is distributed under MIT license
# Check correctness of grand mobility matrix for two beads

import numpy as np
import pygrpy

centres_two = np.array([[0, 0, 0], [0, 0, 1]])
sizes_two = np.array([1, 1])

gmm_two = np.array(
    [
        [5.3e-2, 0.0, 0.0, 3.8e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.2e-2, 0.0],
        [0.0, 5.3e-2, 0.0, 0.0, 3.8e-2, 0.0, 0.0, 0.0, 0.0, 1.2e-2, 0.0, 0.0],
        [0.0, 0.0, 5.3e-2, 0.0, 0.0, 4.3e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.8e-2, 0.0, 0.0, 5.3e-2, 0.0, 0.0, 0.0, 1.2e-2, 0.0, 0.0, 0.0, 0.0],
        [0.0, 3.8e-2, 0.0, 0.0, 5.3e-2, 0.0, -1.2e-2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 4.3e-2, 0.0, 0.0, 5.3e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.2e-2, 0.0, 4.0e-2, 0.0, 0.0, 9.0e-3, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.2e-2, 0.0, 0.0, 0.0, 4.0e-2, 0.0, 0.0, 9.0e-3, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0e-2, 0.0, 0.0, 1.9e-2],
        [0.0, 1.2e-2, 0.0, 0.0, 0.0, 0.0, 9.0e-3, 0.0, 0.0, 4.0e-2, 0.0, 0.0],
        [-1.2e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0e-3, 0.0, 0.0, 4.0e-2, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.9e-2, 0.0, 0.0, 4.0e-2],
    ]
)


def test_mobility_two_beads():
    testmu = pygrpy.grpy_tensors.mu(centres_two, sizes_two)
    assert type(testmu) == np.ndarray, "grpy_tesnors.mu should return np.array."
    assert np.allclose(
        testmu, gmm_two, atol=1e-3
    ), "grpy_tesnors.mu for two beads should return specified value."
    return testmu


if __name__ == "__main__":
    print(test_mobility_two_beads())
```
