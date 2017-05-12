"""
Preprocessing functions for combining IO script outputs before insertion into cczhu_moon_dmap_run.py
"""

################ IMPORTS ################

# Past-proofing
from __future__ import absolute_import, division, print_function

# System modules
import os
import sys
import glob

# I/O and math stuff
import pandas as pd
import numpy as np


################ DATA READ-IN FUNCTIONS (FROM moon4.py and moon_vgg16_1.2.2.py) ################


def get_csv_counts(path, oil_rat, minpix, returnlists):
    """Obtain dictionary of filenames and corresponding crater counts

    Parameters
    ----------
    path : str
        csv filepath
    oil_rat : float
        ratio between target image width and input image width
    minpix : float
        minimum crater pixel diameter
    returnlists : bool
        If True, returns files and crater_count lists rather
        than dictionary for easy concatenation between datasets
    """

    files = sorted([fn for fn in glob.glob(path)
             if (not os.path.basename(fn).endswith('mask.png') and
            not os.path.basename(fn).endswith('dens.png'))])

    crater_count = np.zeros(len(files))
    for i, item in enumerate(files):
        craters = pd.read_csv(item)
        craters.loc[:, "Diameter (pix)"] *= oil_rat
        craters = craters[craters["Diameter (pix)"] >= minpix]
        crater_count[i] = craters.shape[0]

    if returnlists:
        return [files, crater_counts]

    return dict(zip(files, crater_count))


def concat_lists(filelist, countlist):
    """Concatenates output of get_csv_counts"""

    comb_file = []
    for item in filelist:
        comb_file += item

    comb_count = []
    for item in countlist:
        comb_count += item

    return dict(zip(comb_file, comb_count))


def combine_npy(path, filelist):
    """Combines .npy files"""

    npy_arr = []
    for item in filelist:
        npy_arr.append(np.load(path + item))

    return np.concatenate(npy_arr, axis=0)
