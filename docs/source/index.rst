PyGRPY Documentation
==================================

.. toctree::
   :hidden:
   :maxdepth: 1

PyGRPY is a python rewrite of Generalized Rotne Prager Yakamava hydrodynamic
tensors package avaliable on github: `GRPY <https://github.com/pjzuk/GRPY/>`_. You can use 
it to obtain grand mobility matricies and to compute hydrodynamic sizes of 
particles in bead approximation.

New release also supports jax and jax.grad.

There is an online, interactive, variant (written in javascript) that you play
with on my website `~rwaszkiewicz/rigidmolecule <https://www.fuw.edu.pl/~rwaszkiewicz/rigidmolecule/>`_

How to install
''''''''''''''

.. prompt:: bash $ auto

  $ python3 -m pip install pygrpy

and you'll be good to go.

How to cite
'''''''''''
| *Pychastic: Precise Brownian dynamics using Taylor-Itō integrators in Python*
| Radost Waszkiewicz, Maciej Bartczak, Kamil Kolasa, Maciej Lisicki
| SciPost Phys. Codebases 11 **(2023)**
| `doi.org/10.21468/SciPostPhysCodeb.11 <https://scipost.org/10.21468/SciPostPhysCodeb.11>`_.


Package contents
''''''''''''''''

.. automodule:: pygrpy.grpy
   :members:

.. automodule:: pygrpy.grpy_tensors
   :members:

.. automodule:: pygrpy.pdb_loader
   :members:

Experimental features -- jax support
''''''''''''''''''''''''''''''''''''
.. automodule:: pygrpy.jax_grpy_tensors
   :members:
   
   
Example use
'''''''''''

.. prompt:: python >>> auto

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

.. prompt:: python >>> auto
    
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
    

.. prompt:: python >>> auto

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
    gyration_radius = np.sqrt(np.mean((ensemble - centre_of_mass.reshape(-1, 1, 3)) ** 2))

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
            
