"""
get_info.py, SciML-SCD, RAL

Passes the name of the optical property of interest 
if the data is passed or data is downloaded from the 
Materials Project if the API key is passed. 

Other useful routines are accessible from here. 
"""
import logging
import os 
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format="%(levelname)s:gp-net: %(message)s")
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
        logging.info("Convert to dataframe ...")
        props_data = pd.DataFrame(result)
        logging.info("Pickle :)")
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


def ReadData(datafile, ZeroVals):
    """
    ReadData(datafile, ZeroVals) 
    Checks the entries in the dataset so the user 
    can decide on how to split data for processing.

    Inputs:
    datafile-     The data in .pkl format. 
    ZeroVals-     Exclude/Include zero optical 
                  property values. 
    
    Outputs:
    1-            Number of entries in the 
                  dataset.
    """
    inputs = pd.read_pickle(datafile)
    prop = datafile.split("_data")[0]
    print("\nNumber of input entries found for %s data = %s" %(prop, len(inputs)))
    if ZeroVals == False:
        logging.info("Excluding zero optical property values from the dataset ...")
        mask = np.array([i for i,val in enumerate(inputs[prop]) if abs(val) == 0.])
        structures = np.delete(inputs["structure"].to_numpy(), mask)
        targets = np.delete(inputs[prop].to_numpy(), mask)
        print("Remaining number of entries = %s" %len(targets))
    else:
        logging.info("Optical property values zero will not be excluded ...")
        structures = inputs["structure"].to_numpy()
        targets = inputs[prop].to_numpy()
        print("Remaining number of entries = %s" %len(targets))
        

def megnet_input(prop, ZeroVals, bond, nfeat_global, cutoff, width, *fraction):
    """
    megnet_input(prop, ZeroVals, bond, nfeat_global, cutoff, width, *fraction)

    Extracts valid structures and targets and splits them into user specified
    datsets. 

    Inputs:
    prop-                   Optical property of interest. 
    ZeroVals-               Exclude/Include zero optical property values.
    bond-                   MEGNet feature bond.
    nfeat_global-           MEGNet feature global.
    cutoff-                 MEGNet MEGNet radial cutoff. 
    width-                  MEGNet gaussian width.
    *fraction-              Fraction of data to split into training and 
                            validation sets. Passing an extra argument to 
                            split data based on quantity is permissible.

    Outputs:
    1-                      Featurised structures for training with 
                            MEGNet. 
    2-                      Valid structures and targets.
    3-                      Inputs for extraction of activations. 
    4-                      Pool, test, training and validation sets. 
    """
    logging.info("Get graph inputs to MEGNet ...")
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
    print("\nNumber of input entries found for %s data = %s" %(prop, len(inputs)))
    if ZeroVals == False:
        logging.info("Excluding zero optical property values from the dataset ...")
        mask = np.array([i for i,val in enumerate(inputs[prop]) if abs(val) == 0.])
        structures = np.delete(inputs["structure"].to_numpy(), mask)
        targets = np.delete(inputs[prop].to_numpy(), mask)
        print("Remaining number of entries = %s" %len(targets))
    else:
        logging.info("Zero optical property values will be included ...")
        structures = inputs["structure"].to_numpy()
        targets = inputs[prop].to_numpy()        
        
    # Get the valid structures and targets i.e exclude isolated atoms
    logging.info("Obtaining valid structures and targets ...")    
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

    pool_frac = fraction[0][0]    
    if len(fraction) == 1:
        if (fraction[0][0] + fraction[0][1]) == 1.:
            # For train-test split and k-fold cross-validation
            test_frac = fraction[0][1]

            logging.info("The pool is the same as the training set ...")
            print("Requested pool: %s%%" %(pool_frac*100))
            print("Requested test set: %s%%" %(test_frac*100))
        
            # Data split is based on percentages
            pool_boundary = int(len(valid_targets)*pool_frac)    
            Xpool = np.array(valid_structures[0:pool_boundary])
            ypool = np.array(valid_targets[0:pool_boundary])
            Xtest = np.array(valid_structures[pool_boundary:])
            ytest = np.array(valid_targets[pool_boundary:])

            logging.info("The pool is the same as the training set ...")
            print("Pool:", ypool.shape)
            print("Test set:", ytest.shape)        
            return (model, activations_input_full,
                    valid_structures, valid_targets,
                    Xpool, ypool,
                    Xtest, ytest)
    
        elif (fraction[0][0] + fraction[0][1]) < 1.:
            #  For repeat active learning 
            val_frac = fraction[0][1]    
            test_frac = np.round(1 - pool_frac, decimals=2)
            
            pool_boundary = int(len(valid_targets)*pool_frac)
            Xpool = np.array(valid_structures[0:pool_boundary])
            ypool = np.array(valid_targets[0:pool_boundary])
            Xtest = np.array(valid_structures[pool_boundary:])
            ytest = np.array(valid_targets[pool_boundary:])
            
            val_boundary = int(pool_boundary * val_frac)
            Xtrain = Xpool[:-val_boundary]
            ytrain = ypool[:-val_boundary]
            
            Xval = Xpool[-val_boundary:]
            yval = ypool[-val_boundary:]
            print("Requested validation set: %s%% of pool" %(val_frac*100))
            print("Training set:", ytrain.shape)
            print("Validation set:", yval.shape)
            print("Test set:", ytest.shape)
            return (model, activations_input_full,
                    valid_structures, valid_targets,
                    Xpool, ypool,
                    Xtest, ytest,
                    Xtrain, ytrain,
                    Xval, yval)            

    else:
        return ( model,
                 activations_input_full,
                 np.array(valid_structures),
                 np.array(valid_targets) )
