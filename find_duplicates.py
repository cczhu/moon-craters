#!/usr/bin/env python
"""Script to produce .csv or .xlsx file of crater duplicate pairs from either
the combined LROC - Head et al. dataset (used in production) or LROC -
LU78287GT dataset (used during testing).
"""

import numpy as np
import pandas as pd
import cartopy.geodesic as geodesic
from scipy.spatial import cKDTree as kd
import make_input_data as mkin
import argparse


def find_duplicates(craters, radius=1737.4, k=10, rcd=5., ddiam=0.25,
                    filter_pairs=False):
    """Finds duplicate pairs within crater catalog.

    Triples or more will show up as multiple pairs.

    Parameters
    ----------
    craters : pandas.DataFrame
        Crater catalogue.
    radius : float, optional
        Radius of the world.
    k : int, optional
        Nearest neighbours to search for duplicates.  Default is 10.
    rcd : float, optional
        Minimum value of min(crater_pair_diameters) / pair_distance to be
        considered a crater pair.  Minimum rather than average is used to help
        weed out satellite craters.  This criterion is asymmetric between
        pairs, and when filter_pairs=False may lead to single pair entries.
    ddiam : float, optional
        Maximum value of abs(diameter1 - diameter2) / avg(diameters) to be
        considered a crater pair.
    filter_pairs : bool, optional
        If `True`, filters data frame and keeps only one entry per crater
        pair.

    Returns
    -------
    outframe : pandas.DataFrame
        Data frame of crater duplicate pairs.
    """

    mgeod = geodesic.Geodesic(radius=radius, flattening=0.)

    # Convert to 3D (<https://en.wikipedia.org/wiki/
    # Spherical_coordinate_system#Cartesian_coordinates>); phi = [-180, 180) is
    # equivalent to [0, 360).
    craters['phi'] = np.pi / 180. * craters['Long']
    craters['theta'] = np.pi / 180. * (90 - craters['Lat'])

    craters['x'] = radius * np.sin(craters['theta']) * np.cos(craters['phi'])
    craters['y'] = radius * np.sin(craters['theta']) * np.sin(craters['phi'])
    craters['z'] = radius * np.cos(craters['theta'])

    # Create tree.
    kdt = kd(craters[["x", "y", "z"]].as_matrix(), leafsize=10)

    # Loop over all craters to find duplicates.  First, find k + 1 nearest
    # neighbours (k + 1 because query will include self).
    Lnn, inn = kdt.query(craters[["x", "y", "z"]].as_matrix(), k=k+1)
    # Remove crater matching with itself (by checking id).
    Lnn_remove_self = np.empty([Lnn.shape[0], Lnn.shape[1] - 1])
    inn_remove_self = np.empty([Lnn.shape[0], Lnn.shape[1] - 1], dtype=int)
    for i in range(Lnn_remove_self.shape[0]):
        not_self = (inn[i] != i)
        inn_remove_self[i] = inn[i][not_self]
        Lnn_remove_self[i] = Lnn[i][not_self]
    craters['Lnn'] = list(Lnn_remove_self)
    craters['inn'] = list(inn_remove_self)

    # Get radii of nearest neighbors.
    inn_ravel = inn[:, 1:].ravel()
    craters['dnn'] = list(
        craters['Diameter (km)'].as_matrix()[inn_ravel].reshape(-1, 10))
    craters['long_nn'] = list(
        craters['Long'].as_matrix()[inn_ravel].reshape(-1, 10))
    craters['lat_nn'] = list(
        craters['Lat'].as_matrix()[inn_ravel].reshape(-1, 10))
    craters['set_nn'] = list(
        craters['Dataset'].as_matrix()[inn_ravel].reshape(-1, 10))

    # Prepare empty lists.
    dup_id1 = []
    dup_id2 = []
    dup_D1 = []
    dup_D2 = []
    dup_L = []
    dup_LEuclid = []
    dup_ll1 = []
    dup_ll2 = []
    dup_source1 = []
    dup_source2 = []

    # Iterate over craters to determine if any are duplicate pairs.
    for index, row in craters.iterrows():
        # For each pair, record the smaller crater diameter.
        pair_diameter_min = np.array(
            [min(x, row['Diameter (km)']) for x in row['dnn']])
        proper_dist = np.asarray(
            mgeod.inverse(np.array([row['Long'], row['Lat']]),
                          np.vstack([row['long_nn'], row['lat_nn']]).T))[:, 0]
        # Duplicate pair criteria: 1). min(diameter) / distance > rcd; 2).
        # abs(diameter1 - diameter2) / average(diameter) < ddiam - i.e. the
        # separation distance of the centres must be much smaller than either
        # diameter, and the diameters should be very similar.
        rcd_crit = (pair_diameter_min / row['Lnn'] > rcd)
        diam_sim_crit = ((2. * abs(row['dnn'] - row['Diameter (km)']) /
                         (row['dnn'] + row['Diameter (km)'])) < ddiam)
        dup_candidates, = np.where(rcd_crit & diam_sim_crit)
        if dup_candidates.size:
            for i in dup_candidates:
                if index == row['inn'][i]:
                    raise AssertionError("Two craters with identical IDs.")
                dup_id1.append(index)
                dup_id2.append(row['inn'][i])
                dup_D1.append(row['Diameter (km)'])
                dup_D2.append(row['dnn'][i])
                dup_L.append(proper_dist[i])
                dup_LEuclid.append(row['Lnn'][i])
                dup_ll1.append((row['Long'], row['Lat']))
                dup_ll2.append((row['long_nn'][i], row['lat_nn'][i]))
                dup_source1.append(row['Dataset'])
                dup_source2.append(row['set_nn'][i])

    # Multi-index pandas table; see
    # <https://pandas.pydata.org/pandas-docs/stable/advanced.html>.

    outframe = pd.DataFrame({'ID1': dup_id1,
                             'ID2': dup_id2,
                             'Diameter1 (km)': dup_D1,
                             'Diameter2 (km)': dup_D2,
                             'Separation (km)': dup_L,
                             'Euclidean Separation (km)': dup_LEuclid,
                             'Lat/Long1': dup_ll1,
                             'Lat/Long2': dup_ll2,
                             'Dataset1': dup_source1,
                             'Dataset2': dup_source2},
                            columns=('ID1', 'ID2', 'Diameter1 (km)',
                                     'Diameter2 (km)', 'Separation (km)',
                                     'Euclidean Separation (km)',
                                     'Lat/Long1', 'Lat/Long2', 'Dataset1',
                                     'Dataset2'))

    # Hacky, O(N^2) duplicate entry removal.
    if filter_pairs:
        osub = outframe[["ID1", "ID2"]].as_matrix()
        osub = np.array([set(x) for x in osub])
        indices_to_remove = []
        for i in range(osub.shape[0]):
            if i not in indices_to_remove:
                dups = np.where(osub[i + 1:] == osub[i])[0] + i + 1
                indices_to_remove += list(dups)
        indices_to_remove = list(set(indices_to_remove))
        outframe.drop(indices_to_remove, inplace=True)
        outframe.reset_index(inplace=True, drop=True)

    return outframe


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Input data creation script.')
    parser.add_argument('--outhead', metavar='outhead', type=str,
                        required=False, default="crater_duplicates",
                        help='Filepath and filename prefix of output table.')
    parser.add_argument('--lroclu', action='store_true',
                        help=('Combines LROC and LU datasets rather than LROC'
                              ' and Head.'))
    parser.add_argument('--filter_pairs', action='store_true',
                        help=('Leaves only one entry per duplicate pair in'
                              ' output dataframe.'))
    parser.add_argument('--to_xlsx', action='store_true',
                        help='Outputs xlsx instead of csv.')
    args = parser.parse_args()

    ctrs_lroc = mkin.ReadLROCCraterCSV(sortlat=False)
    ctrs_lroc.drop(["tag"], axis=1, inplace=True)
    ctrs_lroc['Dataset'] = 'LROC'

    if args.lroclu:
        ctrs_lu = mkin.ReadSalamuniccarCraterCSV(sortlat=False,
                                                 dropfeatures=False)
        ctrs_lu.drop(["Radius (deg)", "D_range", "p", "Name"],
                     axis=1, inplace=True)
        ctrs_lu = ctrs_lu[ctrs_lu["Diameter (km)"] > 20]
        ctrs_lu['Dataset'] = 'LU'

        craters = pd.concat([ctrs_lroc, ctrs_lu], axis=0, ignore_index=True,
                            copy=True)

    else:
        ctrs_head = mkin.ReadHeadCraterCSV(sortlat=False)
        ctrs_head['Dataset'] = 'Head'

        craters = pd.concat([ctrs_lroc, ctrs_head], axis=0, ignore_index=True,
                            copy=True)

    craters.sort_values(by='Lat', inplace=True)
    craters.reset_index(inplace=True, drop=True)

    outframe = find_duplicates(craters, filter_pairs=args.filter_pairs)

    if args.to_xlsx:
        outframe.to_excel(args.outhead + '.xlsx')
    else:
        outframe.to_csv(args.outhead + '.csv')
