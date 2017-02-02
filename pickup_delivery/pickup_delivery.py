import sys
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from .utils import read_data_file


DataSet = namedtuple(
    'DataSet',
    ('pickups', 'deliveries', 'trucks', 'n_req', 'n_trucks',
     'alpha', 'beta', 'gamma', 'compatible', 'distances', 'query_distances',
     'start_location_ids', 'end_location_ids', 'all_locations'))

Solution = namedtuple('Solution', ('pickups', 'deliveries', 'dataset'))


def _enrich_dataset(dataset):
    if isinstance(dataset, str):
        dataset = read_data_file(dataset)
    max_location_id = max(dataset.pickups.index.max(),
                          dataset.deliveries.index.max())
    start_location_ids = 1 + max_location_id + np.arange(
        dataset.trucks.shape[0])
    start_location_ids = pd.Series(start_location_ids,
                                   index=dataset.trucks.index)
    end_location_ids = 1 + start_location_ids.iloc[-1] + np.arange(
        dataset.trucks.shape[0])
    end_location_ids = pd.Series(end_location_ids,
                                 index=dataset.trucks.index)
    start_locations = pd.DataFrame(
        dataset.trucks.ix[:, ('start_x', 'start_y')].values,
        index=start_location_ids, columns=('x', 'y'))
    end_locations = pd.DataFrame(
        dataset.trucks.ix[:, ('end_x', 'end_y')].values,
        index=end_location_ids, columns=('x', 'y'))
    all_locations = pd.concat(
        (dataset.pickups.ix[:, ('x', 'y')],
         dataset.deliveries.ix[:, ('x', 'y')],
         start_locations,
         end_locations))
    distances = squareform(pdist(all_locations))
    distances = pd.DataFrame(distances, index=all_locations.index,
                             columns=all_locations.index)
    queries = pd.merge(dataset.pickups, dataset.deliveries,
                       left_on='successor_id', right_index=True,
                       suffixes=('_b', '_e'))
    query_distances = pd.Series(np.sqrt(
        np.sum(
            (queries.ix[:, ('x_b', 'y_b')].values
             - queries.ix[:, ('x_e', 'y_e')].values)** 2, axis=1)),
                                   index=dataset.pickups.index)
    query_distances.name = 'start_node_id'
    return DataSet(*(list(dataset) + [
        distances, query_distances, start_location_ids,
        end_location_ids, all_locations]))


def _compute_pickup_times(current_state, dataset, update_trucks=None):
    if update_trucks is None:
        update_trucks = dataset.trucks.index
    current_state = current_state.ix[update_trucks, :]
    arrival_times = (
        dataset.distances.ix[dataset.pickups.index,
                     current_state['current_location']]
        + current_state['start_t'].values).T
    arrival_times.index = update_trucks
    arrival_times[
        dataset.compatible.ix[update_trucks,
                              dataset.pickups.index] == 0] = np.nan
    arrival_times = np.maximum(arrival_times,
                               dataset.pickups['start'].values)

    arrival_times = (
        arrival_times
        + dataset.pickups['service_time'].values)
    arrival_times[
        arrival_times > dataset.pickups['end']] = np.nan
    arrival_times.index = update_trucks
    return arrival_times


def _compute_delivery_times(possible_pickup_times, dataset,
                           update_trucks=None):
    if update_trucks is None:
        update_trucks = dataset.trucks.index
    possible_pickup_times = possible_pickup_times.ix[update_trucks, :]
    possible_delivery_times = np.add(
        possible_pickup_times,
        dataset.query_distances.get_values())
    possible_delivery_times = np.maximum(
        possible_delivery_times, dataset.deliveries['start'].values)
    possible_delivery_times = (
        possible_delivery_times
        + dataset.deliveries['service_time'].values)
    possible_delivery_times.columns = dataset.deliveries.index
    possible_delivery_times[
        possible_delivery_times > dataset.deliveries['end']] = np.nan
    return possible_delivery_times


def _update_choices(possible_pickup_times,
                    possible_delivery_times,
                    possible_arrival_times,
                    delivery_times,
                    current_state, dataset, update_trucks):

    possible_pickup_times.ix[update_trucks, :] = _compute_pickup_times(
        current_state, dataset, update_trucks=update_trucks)
    possible_delivery_times.ix[
        update_trucks, :] = _compute_delivery_times(
        possible_pickup_times, dataset, update_trucks=update_trucks)
    possible_delivery_times.ix[
        :, delivery_times.notnull().any(axis=0)] = np.nan
    possible_arrival_times = (
        possible_delivery_times
        + dataset.distances.ix[dataset.end_location_ids,
                               dataset.deliveries.index].values)
    possible_delivery_times[
        (possible_arrival_times.T > dataset.trucks['end_t']).T] = np.nan
    return (possible_pickup_times, possible_delivery_times,
            possible_arrival_times)


def _prepare_initial_solution_search(dataset):
    dataset = _enrich_dataset(dataset)
    pickup_times = pd.DataFrame(
        index=dataset.trucks.index, columns=dataset.pickups.index)
    delivery_times = pd.DataFrame(
        index=dataset.trucks.index, columns=dataset.deliveries.index)
    current_state = pd.concat(
        (dataset.trucks['start_t'],
         pd.DataFrame({'current_location': dataset.start_location_ids},
                      index=dataset.trucks.index)), axis=1)
    possible_pickup_times = pd.DataFrame(
        index=dataset.trucks.index, columns=dataset.pickups.index)
    possible_delivery_times = pd.DataFrame(
        index=dataset.trucks.index, columns=dataset.deliveries.index)
    possible_arrival_times = pd.DataFrame(
        index=dataset.trucks.index, columns=dataset.pickups.index)
    update_trucks = dataset.trucks.index
    return (dataset, pickup_times, delivery_times, current_state,
            possible_pickup_times, possible_delivery_times,
            possible_arrival_times, update_trucks)


def initial_solution(dataset):
    print('computing initial solution')
    (dataset, pickup_times, delivery_times, current_state,
     possible_pickup_times, possible_delivery_times, possible_arrival_times,
     update_trucks) = _prepare_initial_solution_search(dataset)
    n_digits = int(np.ceil(np.log10(dataset.pickups.shape[0])))

    for i in range(dataset.pickups.shape[0]):
        (possible_pickup_times, possible_delivery_times,
         possible_arrival_times) = _update_choices(
         possible_pickup_times, possible_delivery_times,
         possible_arrival_times,
         delivery_times, current_state, dataset, update_trucks)

        if possible_delivery_times.isnull().values.all():
            pickup_times -= dataset.pickups.service_time
            delivery_times -= dataset.deliveries.service_time
            print()
            return Solution(pickup_times, delivery_times, dataset)
        first_query = possible_delivery_times.min().argmin()
        first_truck = possible_delivery_times.ix[:, first_query].idxmin()
        pickup_times.ix[
            first_truck, dataset.deliveries.predecessor_id[first_query]
        ] = possible_pickup_times.ix[
            first_truck, dataset.deliveries.predecessor_id[first_query]]
        delivery_times.ix[first_truck,
                          first_query] = possible_delivery_times.ix[
            first_truck, first_query]
        current_state.ix[first_truck, 'start_t'] = possible_delivery_times.ix[
            first_truck, first_query]
        current_state.ix[first_truck, 'current_location'] = first_query
        update_trucks = [first_truck]
        # sys.stdout.write('\rassigned {{:<{}}} jobs'.format(n_digits).format(i + 1))
        # sys.stdout.flush()
    pickup_times -= dataset.pickups.service_time
    delivery_times -= dataset.deliveries.service_time
    # print()
    return Solution(pickup_times, delivery_times, dataset)


def check_solution(solution):
    # TODO: to complete
    pickups = solution.pickups
    deliveries = solution.deliveries
    dataset = solution.dataset
    assert pickups.notnull().sum(axis=0).max() < 2
    assert deliveries.notnull().sum(axis=0).max() < 2
    assert (pickups.notnull().get_values()
            == deliveries.notnull().get_values()).all()
    assert not (pickups.notnull().get_values() & (dataset.compatible.ix[
        :, dataset.pickups.index].get_values() == 0)).any()
    assert (pickups >= dataset.pickups.start).get_values()[
        pickups.notnull().get_values()].all()
    assert (pickups + dataset.pickups.service_time
            <= dataset.pickups.end).get_values()[
                pickups.notnull().get_values()].all()
    assert (pickups + dataset.pickups.service_time + dataset.query_distances
            + dataset.deliveries.service_time.get_values()
            <= dataset.deliveries.end.get_values()).get_values()[
                pickups.notnull().get_values()].all()
    assert (pickups >= dataset.pickups.start).get_values()[
        pickups.notnull().get_values()].all()
    assert (deliveries >= dataset.deliveries.start).get_values()[
        pickups.notnull().get_values()].all()
    assert (deliveries.get_values()
            + dataset.deliveries.service_time.get_values()
            + dataset.distances.ix[dataset.end_location_ids,
                                   dataset.deliveries.index].get_values()
            <= dataset.trucks.end_t.get_values()[:,np.newaxis])[
                pickups.notnull().get_values()].all()


def req_dist(solution, phi=9., chi=3., psi=2., omega=1):
    pickups_distances = solution.dataset.distances.ix[solution.pickups.columns,
        solution.pickups.columns]
    deliveries_distances = solution.dataset.distances.ix[solution.deliveries.columns,
        solution.deliveries.columns]
    deliveries_distances.index = solution.dataset.deliveries.predecessor_id
    deliveries_distances.columns = solution.dataset.deliveries.predecessor_id

    pickups_times = solution.pickups.fillna(0).sum()
    pickups_time_diff = np.abs(pickups_times.values - pickups_times.values[:,np.newaxis])
    pickups_time_diff = pd.DataFrame(pickups_time_diff, index=solution.pickups.columns, 
        columns=solution.pickups.columns)

    deliveries_times = solution.deliveries.fillna(0).sum()
    deliveries_time_diff = np.abs(deliveries_times.values - deliveries_times.values[:,np.newaxis])
    deliveries_time_diff = pd.DataFrame(deliveries_time_diff, index=solution.dataset.deliveries.predecessor_id, 
        columns=solution.dataset.deliveries.predecessor_id)

    demand_diff = pd.DataFrame(np.abs(solution.dataset.pickups.demand.values - 
        solution.dataset.pickups.demand.values[:,np.newaxis]),index=solution.pickups.columns,
        columns = solution.pickups.columns)

    pickups_comptible = solution.dataset.compatible.ix[:, solution.pickups.columns]
    intersection_card = pickups_comptible.T.dot(pickups_comptible)
    card_compatible = pickups_comptible.sum()
    min_card = pd.DataFrame(-np.abs(card_compatible.values - card_compatible.values[:, np.newaxis])
        + card_compatible.values + card_compatible.values[:, np.newaxis],index=solution.pickups.columns,
        columns=solution.pickups.columns)/2

    return (phi*(pickups_distances + deliveries_distances) + chi *(pickups_time_diff+deliveries_time_diff) +
        psi*demand_diff + omega * (1 - intersection_card/min_card))


def shaw_removal(solution, n_updated, power=2):

    remaining_requests = solution.pickups.notnull().sum()
    remaining_requests = remaining_requests[remaining_requests == 1].index
    seed_request = np.random.randint(len(remaining_requests))

    relatedness = req_dist(solution)

    to_remove = [remaining_requests[seed_request]]
    remaining_requests = remaining_requests.drop(remaining_requests[seed_request])

    while len(to_remove) < n_updated:
        picked_index = np.random.randint(len(to_remove))
        picked = to_remove[picked_index]


        related = relatedness.ix[picked]
        related = related[related.index.isin(remaining_requests)]
        order = np.argsort(related.values)
        seed = np.random.rand()
        seed = np.power(seed, power)
        seed = np.floor(seed * len(remaining_requests))
        new_removed = remaining_requests[order[seed]]
        to_remove.append(new_removed)
        remaining_requests = remaining_requests.drop(remaining_requests[order[seed]])
    return to_remove


# def lns_heuristic(initial_solution, n_updated=10):
#     solution = initial_solution
#     for iteration in range(1000):
#         solution = remove


# def req_dist(req_a, req_b, phi=9., chi=3., psi=2., omega=1):
#     (phi * (euclidean_dist(req_a['pickup'], req_b['pickup'])
#            + (euclidean_dist(req_a['delivery'], req_b['delivery'])))
#             + chi * (np.abs(req_a['pickup_time'] - req_b['pickup_time'])
#                      + np.abs(req_a['delivery_time'] - req_b['delivery_time']))
#                      + psi * np.abs(req_a['demand'] - req_b['demand'])
#      + omega * (1. - (size('cars serving both')
#                       / min(size('cars serving a'), size('cars serving b')))))


# def shaw_removal(solution, n_updated, power=2):
#     seed_request = np.random.randint(solution.shape[1])
#     to_remove = [seed_request]
#     while len(to_remove) < n_updated:
#         picked_index = np.random.randint(len(to_remove))
#         picked = to_remove[picked_index]
#         remaining = 'requests in s (not n to_remove)'
#         distances = req_dist(picked, remaining)
#         order = np.argsort(distances)
#         seed = np.random.rand()
#         seed = np.power(seed, power)
#         seed = np.floor(seed * len(to_remove))
#         new_removed = remaining[order[seed]]
#         to_remove.append(new_removed)

#     return solution, to_remove


# def cost_function(solution):
#     alpha * total_distance + beta * total_time + gamma * n_non_satisfied

# def greedy_insert(current_solution, query):
#     best_cost = None
#     best_solution = None
#     for car in all_cars:
#         try_insert = insert(current_solution, to_insert, car)
#         cost = cost_function(try_insert)
#         if cost is not None:
#             if best_cost is None or cost < best_cost:
#                 best_cost = cost
#                 best_solution = try_insert


# def best_insertion_for_request(current_solution, query):
#     for truck in current_solution:
#         for
