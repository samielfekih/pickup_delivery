import os
import csv
from collections import namedtuple

import numpy as np
import pandas as pd


DataSet = namedtuple('DataSet', ('locations', 'cars', 'n_req',
                     'n_cars', 'alpha', 'beta', 'gamma'))

def read_data_file(file_path):
    file_path = os.path.abspath(os.path.expanduser(file_path))
    with open(file_path, 'r') as data_file:
        reader = csv.reader(data_file, delimiter='\t')
        header = next(reader)
        n_req, n_cars, alpha, beta, gamma = map(int, header)
        locations = []
        serving_cars = []
        for i in range(2 * n_req):
            row = next(reader)
            loc_info = row[:9]
            serving_cars.append(row[9:])
            locations.append(list(map(float, loc_info)))

        locations = pd.DataFrame(
            locations,
            columns=['node_id', 'x', 'y', 'demand',
                     'start', 'end', 'service_time', 'predecessor_id',
                     'successor_id'])
        locations.fillna(-1)
        for col in ['node_id', 'demand', 'predecessor_id',
                    'successor_id']:
            locations[col] = np.asarray(locations[col], dtype=int)

        locations.set_index('node_id', inplace=True)

        cars = []
        for i in range(n_cars):
            cars.append(list(map(float, next(reader))))
        cars = pd.DataFrame(
            cars,
            columns=['car_id', 'start_x', 'start_y', 'end_x',
                     'end_y', 'capa', 'start_t', 'end_t'])
        for col in ['car_id', 'capa']:
            cars[col] = np.asarray(cars[col], dtype=int)

        cars.set_index('car_id', inplace=True)

        return DataSet(locations, cars, n_req, n_cars,
                        alpha, beta, gamma)
