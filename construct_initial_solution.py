#! /usr/bin/env python3

import os
import argparse
from hashlib import sha1
import random

from pickup_delivery import initial_solution, check_solution


parser = argparse.ArgumentParser(description='compute an initial solution')
parser.add_argument(
    'data_file', type=str, help='csv file containing the problem data')
parser.add_argument(
    'out_directory', type=str, help='directory in which to store results')

args = parser.parse_args()

data_file = os.path.abspath(os.path.expanduser(args.data_file))
out_directory = os.path.abspath(os.path.expanduser(args.out_directory))
pickup_file = os.path.join(out_directory, u'result_pickups_{0}.csv'.format(
    sha1(u'{0}'.format(random.random()).encode('utf-8')).hexdigest()))
delivery_file = os.path.join(
    out_directory, u'result_deliveries_{0}.csv'.format(
        sha1(u'{0}'.format(random.random()).encode('utf-8')).hexdigest()))

solution = initial_solution(data_file)
check_solution(solution)

solution.pickups.to_csv(pickup_file)
solution.deliveries.to_csv(delivery_file)

print(u'{0} out of {1} requests satisfied'.format(
    solution.pickups.notnull().values.sum(),
    solution.dataset.pickups.shape[0]))
