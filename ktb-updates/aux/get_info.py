"""
get_info.py, SciML-SCD, RAL

Passes the name of the optical property of interest 
if the data is passed or data is downloaded from the 
Materials Project if the API key is passed. 

Other useful routines are accessible from here. 
"""

import pandas as pd
import numpy as np 

from pymatgen import MPRester
from megnet.data.graph import GaussianDistance
from megnet.data.graph import StructureGraph
from megnet.data.crystal import CrystalGraph
from megnet.models import MEGNetModel


def show_layers(model_file):
    """
    show_layers(model_file)

    Displays information on layers of a pre-trained 
    MEGNet model. 

    Inputs:
        model_file-      A pre-trained MEGNet model file.

    Outputs:
         1-              Layers in the model file.
    """
    pretrained_model = MEGNetModel.from_file(model_file) 
    print(pretrained_model.summary())

    
def download(key):
    """
    download(key)

    Downloads the dataset matching the optical 
    property of interest. 

    Inputs:
         key-      The API key for downloading data 
                   from the Materials Project database. 

    Outputs:
         1-        List of requested material properties
                   to be trained on.
    """
    api = key[0]
    m = MPRester(api)
    criteria = {"elements": {"$all":["O"]}}
    for props in key[1:]:
        properties = ["structure", "%s" %props]
        print("\nFetching %s data ..." %props)
        result = m.query(criteria, properties)
        print("Convert to dataframe ...")
        props_data = pd.DataFrame(result)
        print("Pickle :)")
        props_data.to_pickle("%s_data.pkl" %props)
    return key[1:]


def load_data(data):
    """
    load_data(data) 

    Load the passed datasets.

    Inputs:
        data-  Already downloaded dataset. 

    Outputs:
         1-    List of requested material properties
               to be trained on.
    """
    props = [ ]
    for dat in data:
        props.append(dat.split("_data.pkl")[0])
    return props


def megnet_input(prop, ZeroVals, bond, nfeat_global, cutoff, width, fraction):
    """
    megnet_input(prop, ZeroVals, bond, nfeat_global, cutoff, width)

    Extracts valid structures and targets and splits them into
    pool and test sets.

    Inputs:
    prop-                   Optical property of interest. 
    ZeroVals-               Exclude/Include zero optical property values.
    bond-                   MEGNet feature bond.
    nfeat_global-           MEGNet feature global.
    cutoff-                 MEGNet MEGNet radial cutoff. 
    width-                  MEGNet gaussian width.
    fraction-               Fraction of data to split into training and 
                            validation sets. 

    Outputs:
    1-                      Featurised structures for training with 
                            MEGNet. 
    2-                      Valid structures and targets.
    3-                      Inputs for extraction of activations. 
    4-                      Pool, test sets, and fraction of data 
                            for validation in active learning.
    """
    print("\nGet graph inputs to MEGNet ...")
    print("Bond features = ", bond)
    print("Global features = ", nfeat_global)
    print("Radial cutoff = ", cutoff)
    print("Gaussian width = ", width)
    gaussian_centers = np.linspace(0, cutoff, bond)
    distance_converter = GaussianDistance(gaussian_centers, width)
    graph_converter = CrystalGraph(bond_converter=distance_converter)
    model = MEGNetModel(bond, nfeat_global, graph_converter=graph_converter)

    datafile = "%s_data.pkl" %prop
    inputs = pd.read_pickle(datafile)
    inputs.sample(frac=1)
    print("\nNumber of input entries found for %s data = %s" %(prop, len(inputs)))
    if ZeroVals == False:
        print("Excluding zero optical property values from the dataset ...")
        mask = np.array([i for i,val in enumerate(inputs[prop]) if abs(val) == 0.])
        structures = np.delete(inputs["structure"].to_numpy(), mask)
        targets = np.delete(inputs[prop].to_numpy(), mask)
        print("Remaining number of entries = %s" %len(targets))
    else:
        print("Optical property values zero will not be excluded ...")
        structures = inputs["structure"].to_numpy()
        targets = inputs[prop].to_numpy()        
        
    print("\nObtaining valid structures and targets ...")
    # Get the valid structures and targets i.e exclude isolated atoms
    valid_structures = [ ]
    valid_targets = [ ]
    activations_input_full = [ ]
    for s, t in zip(structures, targets):
        try:
            activations_input_full.append(StructureGraph.get_input(graph_converter, s))
        except:
            print("Skipping structure with isolated atom ...")
            continue
        valid_structures.append(s)
        valid_targets.append(t)
    print("Number of invalid structures = %s" %(len(targets)-len(valid_targets)))
    print("\nTotal number of entries available for analysis = %s" %len(valid_targets))

    pool_frac = fraction[0]
    val_frac = fraction[1]
    test_frac = np.round(1 - pool_frac, decimals=2)

    pool_boundary = int(len(valid_targets)*pool_frac)
    Xpool = np.array(valid_structures[0:pool_boundary])
    ypool = np.array(valid_targets[0:pool_boundary])
    Xtest = np.array(valid_structures[pool_boundary:])
    ytest = np.array(valid_targets[pool_boundary:])
    print("\nRequested pool: %s%%" %(pool_frac*100))
    print("Requested test set: %s%%" %(test_frac*100))
    print("Pool:", ypool.shape)
    print("Test set:", ytest.shape)

    return (model, valid_structures, valid_targets,
            activations_input_full, Xpool, ypool,
            Xtest, ytest, val_frac)
