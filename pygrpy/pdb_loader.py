# Copyright Tomasz Skóra 2024
# Copyright Radost Waszkiewicz 2024

import numpy as np
import MDAnalysis as mda
import requests
import json
from io import StringIO


def get_pdb_from_pdb(pdb_id, pdb_download_server="https://files.rcsb.org/download/"):
    """
    Fetches the PDB file of a given PDB ID.

    Parameters:
    - pdb_id (str): The PDB ID of the protein.
    - pdb_download_server (str) [optional]: The address of the pdb server

    Returns:
    str or None: The PDB file content as a string if successful, None otherwise.
    """

    pdb_url = pdb_download_server + pdb_id + ".pdb"
    print(
        "Sending GET request to {} to fetch {}'s PDB file as a string.".format(
            pdb_url, pdb_id
        )
    )
    response = requests.get(pdb_url)
    if response is None or not response.ok:
        print("Something went wrong.")
        return None
    return response.text


def get_pdb_from_alphafold(
    uniprot, alphafold_download_server="https://alphafold.ebi.ac.uk/api/prediction/"
):
    """
    Fetches the PDB file from AlphaFold for a given UniProt ID.

    Parameters:
    - uniprot (str): The UniProt ID of the protein.
    - alphafold_download_server (str) [optional]: The address of the alphafold server.

    Returns:
    str or None: The PDB file content as a string if successful, None otherwise.
    """

    alphafold_url = alphafold_download_server + uniprot
    print(
        "Sending GET request to {} to fetch {}'s JSON file as a string.".format(
            alphafold_url, uniprot
        )
    )
    first_response = requests.get(alphafold_url)
    if first_response is None or not first_response.ok:
        print("Something went wrong.")
        return None
    json_string = first_response.text
    json_dict = json.loads(json_string[1:-1])
    pdb_url = json_dict["pdbUrl"]
    print(
        "Sending GET request to {} to fetch {}'s PDB file as a string.".format(
            pdb_url, uniprot
        )
    )
    response = requests.get(pdb_url)
    if response is None or not response.ok:
        print("Something went wrong.")
        return None
    return response.text


def _get_calphas_radii(calphas_resnames, res_to_radii=None):
    if res_to_radii is None:
        res_to_radii = {
            "ALA": 2.28,
            "GLY": 2.56,
            "MET": 4.65,
            "PHE": 4.62,
            "SER": 3.12,
            "ARG": 5.34,
            "ASN": 3.83,
            "ASP": 3.72,
            "CYS": 3.35,
            "GLU": 3.99,
            "GLN": 4.44,
            "HIS": 4.45,
            "ILE": 3.50,
            "LEU": 3.49,
            "LYS": 4.01,
            "PRO": 3.50,
            "THR": 3.24,
            "TYR": 4.97,
            "TRP": 5.04,
            "VAL": 3.25,
        }
    return [res_to_radii[res] for res in calphas_resnames]


def centres_and_radii(pdb_string):
    """
    Extracts the coordinates and radii of C-alpha atoms from a PDB file string.

    Parameters:
    - pdb_string (str): The PDB file content as a string.

    Returns:
    tuple: A tuple containing the C-alpha atom coordinates (numpy array) and their corresponding radii (list).

    Examples:
    >>> # Lysozyme C structure
    >>> pdb_content = get_pdb_from_alphafold("P61626")
    >>> coordinates, radii = centres_and_radii(pdb_content)
    >>> print(stokesRadius(coordinates, radii))
    >>>
    >>> # Lysozyme C structure
    >>> pdb_content = get_pdb_from_alphafold("P61626")
    >>> coordinates, radii = centres_and_radii(pdb_content)
    >>> print(stokesRadius(coordinates, radii))
    """
    calphas = mda.Universe(StringIO(pdb_string), format="pdb").select_atoms("name CA")
    calphas_radii = _get_calphas_radii(calphas.resnames)
    return np.array(calphas.positions), np.array(calphas_radii)
