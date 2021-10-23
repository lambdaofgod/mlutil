import attr
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import pandas as pd
import tqdm
import logging


try:
    import numba
except ImportError as e:
    logging.warning(
        "numba not found, you'll not be able to use mlutil.evolutionary_algorithms.multiobjective"
    )


def bounded_gaussian_noise_mutation(x, n_mutants, lo=0, hi=1, sigma=1e-2):
    noise = sigma * np.random.randn(n_mutants, x.shape[-1])
    return np.clip(x + noise, lo, hi)


@attr.s
class NSGAII:

    optimized_function = attr.ib()
    chromosome_size: int = attr.ib()
    mutation_function = attr.ib(default=bounded_gaussian_noise_mutation)
    random_initializer = attr.ib(default=np.random.rand)
    population_bounds = attr.ib(default=(0, 1))
    objective_names = attr.ib(default=("1st objective", "2nd objective"))

    def optimize(
        self,
        population_size=1000,
        n_selected=100,
        n_iterations=100,
        minimize=True,
        verbose=True,
        log_period=10,
    ):
        pop = self.random_initializer(population_size, self.chromosome_size)

        iterator = range(n_iterations)
        if verbose:
            iterator = tqdm.tqdm(iterator)

        for _iter in iterator:
            values = np.vstack([self.optimized_function(i) for i in pop])
            if not minimize:
                values = -values
            indices = nsga_selection(n_selected, values)
            values_selected = values[indices]
            pop_selected = pop[indices]
            pop_new = np.row_stack(
                [
                    self.mutation_function(
                        pop_selected[i], population_size // n_selected
                    )
                    for i in range(n_selected)
                ]
            )
            pop = pop_new
            if self.population_bounds is not None:
                lo, hi = self.population_bounds
                pop = np.clip(pop, lo, hi)

            if values.shape[1] == 2 and verbose and _iter % log_period == 0:
                self.make_annotated_ot_df(_iter, values)
        return pop

    def make_values_scatterplot(self, _iter, values):
        plt_values = values
        plt_values_selected = values_selected
        if not minimize:
            plt_values = -plt_values
            plt_values_selected = -plt_values_selected
        plt.scatter(plt_values[:, 0], plt_values[:, 1], label="population")
        plt.scatter(
            plt_values_selected[:, 0],
            plt_values_selected[:, 1],
            c="red",
            label="selected with nondominated sort",
        )
        plt.xlabel(self.objective_names[0])
        plt.ylabel(self.objective_names[1])
        plt.title("iteration {}".format(_iter))
        plt.legend()
        plt.show()


@numba.jit(nopython=True)
def fast_nondominated_sort(pop):
    size = len(pop)
    dominated = dict()
    domination_counts = np.zeros(size)
    ranks = np.zeros(size)
    fronts = dict()
    fronts[0] = numba.typed.List.empty_list(item_type=numba.types.int32)

    for i in range(size):
        p = pop[i]
        dominated[i] = numba.typed.List.empty_list(item_type=numba.types.int32)
        for j in range(size):
            q = pop[j]
            if np.all(p <= q) and not np.all(p == q):
                dominated[i].append(j)
            elif np.all(q <= p) and not np.all(p == q):
                domination_counts[i] += 1
        if domination_counts[i] == 0:
            fronts[0].append(i)

    k = 0
    while len(fronts[k]) > 0:
        next_front = numba.typed.List.empty_list(item_type=numba.types.int32)
        for i in fronts[k]:
            for j in dominated[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
                    ranks[j] += 1
        k += 1
        fronts[k] = next_front

    return dominated, ranks, fronts


@numba.jit(nopython=True)
def _crowding_distance_assignment(pop):
    n_criterions = pop.shape[1]
    size = pop.shape[0]
    crowd_dist = np.zeros(size)
    for m in range(n_criterions):
        sorted_pop_indices = pop[:, m].argsort()
        crowd_dist[sorted_pop_indices[0]] = pop.max()
        crowd_dist[sorted_pop_indices[-1]] = pop.max()
        for i in range(1, size - 1):
            fmax, fmin = pop[:, m].max(), pop[:, m].min()
            df = fmax - fmin + 1e-8
            dist_increment = (
                -pop[sorted_pop_indices[i - 1], m] + pop[sorted_pop_indices[i + 1], m]
            )
            crowd_dist[sorted_pop_indices[i]] += dist_increment
    return crowd_dist


@numba.jit(nopython=True)
def _select_from_front(n_selected, front_indices, ranks, crowding_distances):
    selected_indices = np.zeros(n_selected, dtype=numba.types.int32)
    biggest_rank = np.sort(ranks)[n_selected - 1]
    rank_selected_indices = np.where(ranks < biggest_rank)[0]
    n_rank_selected = len(rank_selected_indices)
    selected_indices[:n_rank_selected] = rank_selected_indices
    if n_rank_selected < n_selected:
        rank_tied_indices = np.where(ranks == biggest_rank)[0]
        crowding_distance_sorted_indices = crowding_distances[rank_tied_indices][
            ::-1
        ].argsort()
        selected_indices[n_rank_selected:] = rank_tied_indices[
            crowding_distance_sorted_indices
        ][: n_selected - n_rank_selected]
    return front_indices[selected_indices]


@numba.jit(nopython=True)
def _make_array_from_list(lst):
    ar = np.empty(len(lst), dtype=numba.types.int32)
    for i in range(len(lst)):
        ar[i] = lst[i]
    return ar


@numba.jit(nopython=True)
def _nsga_selection(n_selected, objective_values):
    __, ranks, fronts = fast_nondominated_sort(objective_values)
    selected_objective_values_indices = np.empty(n_selected, dtype=numba.types.int32)
    select_counter = 0
    for i in range(len(fronts)):
        front = _make_array_from_list(fronts[i])
        if select_counter == n_selected:
            break
        if select_counter + len(front) > n_selected:
            front_ranks = ranks[front]
            front_objective_values = objective_values[front]
            crowding_dist = _crowding_distance_assignment(front_objective_values)
            selected_objective_values_indices[select_counter:] = _select_from_front(
                n_selected - select_counter, front, front_ranks, crowding_dist
            )
            select_counter = n_selected
        else:
            selected_objective_values_indices[
                select_counter : select_counter + len(front)
            ] = front
            select_counter += len(front)

    return selected_objective_values_indices[:n_selected]


@numba.jit(nopython=True)
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@numba.jit(nopython=True)
def np_max(array, axis):
    return np_apply_along_axis(np.max, axis, array)


@numba.jit(nopython=True)
def add_max_to_minus(arr):
    mx = np_max(arr, 0)
    new_arr = -arr.copy()
    for d in range(arr.shape[0]):
        arr += mx[d]
    return new_arr


@numba.jit(nopython=True)
def select_multiobjective_with_roulette_method(number_of_offspring, objective_values):
    population_size = len(objective_values)
    fitness_values_multi = add_max_to_minus(objective_values)
    fitness_values = fitness_values_multi.sum(axis=1)
    if fitness_values.sum() > 0:
        fitness_values = fitness_values / fitness_values.sum()
    else:
        fitness_values = np.ones(population_size) / population_size
    parent_indices = sga.numba_random_choice_with_replacement(
        np.arange(population_size), fitness_values.reshape(-1), number_of_offspring
    )
    return parent_indices[np.random.randn(number_of_offspring).argsort()]
