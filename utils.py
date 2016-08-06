import os
import csv
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.sparse


DataSet = namedtuple('DataSet',
                     ('requests', 'trucks', 'n_req', 'n_trucks',
                      'alpha', 'beta', 'gamma', 'incompatible'))

def _get_incompat_trucks(file_path, n_req, n_trucks):
    incompat = scipy.sparse.lil_matrix((n_req * 2, n_trucks))
    with open(file_path, newline='') as data_file:
        reader = csv.reader(data_file, delimiter='\t')
        next(reader)
        for i in range(n_req):
            row = next(reader)
            loc_id = row[0]
            compat = row[10:]
            if compat:
                compat = list(map(int, compat))
                assert (len(compat) == int(row[9])), (
                    "bad row in csv: file: {}, row: {}, loc_id: {}: "
                    "n trucks announced: {}, n found: {}".format(
                        file_path, i + 1, loc_id, row[9], len(compat)))
                incompat[loc_id] = 1
                incompat[loc_id, compat] = 0
        return scipy.sparse.csc_matrix(incompat)


def read_data_file(file_path):
    """Read a data csv file and return content.

    Parameters
    ----------
    file_path : str
        The path to the csv file to read.

    Returns
    -------
    DataSet
        ``collections.namedtuple``. Contains the data. The last
        element, ``incompatible``, is an ``n_requests * n_trucks``
        compressed sparse column matrix such that
        ``incompatible[i, j]`` is 1 if truck ``j`` is unable to
        attend request ``i`` and 0 otherwise.

    """
    file_path = os.path.abspath(os.path.expanduser(file_path))
    with open(file_path, newline='') as data_file:
        reader = csv.reader(data_file, delimiter='\t')
        header = next(reader)
        n_req, n_trucks, alpha, beta, gamma = map(int, header)

    locations = pd.read_csv(
        file_path, sep='\t', skiprows=1, nrows=2 * n_req,
        usecols=range(9), header=None, index_col=0,
        names=('node_id', 'x', 'y', 'demand',
               'start', 'end', 'service_time', 'predecessor_id',
               'successor_id'))

    for col in ('x', 'y', 'start', 'end', 'service_time'):
        locations[col] = np.asarray(locations[col], dtype=float)

    trucks = pd.read_csv(
        file_path, sep='\t', skiprows=(1 + 2*n_req),
        header=None, index_col=0,
        names=('truck_id', 'start_x', 'start_y', 'end_x',
               'end_y', 'capa', 'start_t', 'end_t'))
    for col in ('start_x', 'start_y', 'end_x', 'end_y', 'start_t', 'end_t'):
        trucks[col] = np.asarray(trucks[col], dtype=float)

    incompatible = _get_incompat_trucks(file_path, n_req, n_trucks)

    return DataSet(locations, trucks, n_req, n_trucks,
                   alpha, beta, gamma, incompatible)
