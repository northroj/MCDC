import h5py, math, numba, os

import mcdc.global_ as global_
input_deck = global_.input_deck
import mcdc.input_ as input_

from pathlib import Path
import xml.etree.ElementTree as ET

from mpi4py import MPI
from numba import (
    int64,
    literal_unroll,
    njit,
    objmode,
    uint64,
)
from mcdc.card import (
    InputCard,
)

#from mcdc.adapt import toggle, for_cpu, for_gpu
from mcdc.constant import *
from mcdc.print_ import print_error, print_msg, print_warning
from mcdc.src.algorithm import binary_search, binary_search_with_length



#################################### xml/hdf5 interface ####################################

def set_cross_sections_xml(path):
    """Set the path to the cross_sections.xml file.

    Parameters
    ----------
    filename : string
        The path (directory or specific filepath) to the cross_sections.xml file

    """

    path = os.path.abspath(os.path.normpath(path))

    if os.path.isdir(path):
        path = os.path.join(path, "cross_sections.xml")
    print(path)
    if not os.path.isfile(path):
        print_error(f"XML file not found: {path}")
    
    # Set the path to the cross sections
    global_.input_deck.setting["xs_path"] = path

def load_h5_from_xml(root, name=None, category="neutron"):
    """Find a library in the XML and open the associated .h5 file.

    Parameters:
        root     : XML root object from ElementTree. Point to the object returned by read_cross_sections_xml()
        name     : The name of the 'materials' attribute to match. i.e. "H1"
        type_    : The value of the 'type' attribute to match. i.e. "neutron"

    Returns:
        h5py.File object — be sure to close it when done.
    """

    base_dir = global_.input_deck.setting["xs_path"]
    if base_dir == "":
        print_error("Call read_cross_sections_xml() before loading nuclides")
    if base_dir.endswith("cross_sections.xml"):
        base_dir = os.path.dirname(base_dir)

    for lib in root.findall("library"):
        if lib.get("materials") == name and lib.get("type") == category:
            h5_path = lib.get("path")
            full_path = os.path.join(base_dir, h5_path)
            if not os.path.isfile(full_path):
                print_error(f"HDF5 file not found: {full_path}")
            return h5py.File(full_path, "r")

    print_error(f"No matching library found for material '{name}' and type '{category}'.")

# Choose temperature, FIX: current implementation just picks the closest temperature data without interpolation
def select_temperature(f, nuc_name, i):
    """Choose a temperature for the data based on the nuclide temperature (no interpolation yet).

    Parameters:
        f     : hdf5 file to read the data from
        nuc_name     : str, The name of the nuclide i.e. "H1"
        i    : int, The index of the nuclide

    Returns:
        h5py.File object — be sure to close it when done.
    """
    temperature = input_deck.nuclides[i].temperature
    data_temperatures = f[nuc_name]["energy"].keys()
    temperature_values = [float(name.rstrip('K')) for name in data_temperatures]
    min_diff = float('inf')
    closest_temp = None
    for temp in temperature_values:
        diff = abs(temp - temperature)
        if diff < min_diff:
            min_diff = diff
            closest_temp = temp
    temperature_string = f"{int(closest_temp)}K"

    return temperature_string


