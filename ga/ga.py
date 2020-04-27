from datetime import datetime

import numpy as np
from numpy import random as rd


class GeneticAlgorithm:
    selections = ["tournament", "sigma-cut", "top"]
    parent_choice = ["panmixis", "outbreeding"]

    def __init__(self, generator, selection, generations, population_size=100, kids_size=40, mutation_size=60,
                 parent_selection="panmixis", logging=False, save_int_results=False):
        """

        :param generator: generator which produces population
        :param selection: selection types: tournament, sigma-cut, or top
        :param generations: the number of generations
        :param population_size: the number of individuals in each population
        :param kids_size: the number of kids produces on each iteration
        :param mutation_size: the number of mutations on each iteration
        :param parent_selection: the type of parent selection: panmixis or random_outbreeding
        """

        self._population_size = population_size
        self._kids_size = kids_size
        self._mutation_size = mutation_size
        self._generator = generator
        self._generations = generations
        self._parent_selection = parent_selection
        self._selection = selection  # todo: create selection functions
        self._population = None
        self._logging = logging
        self._save_int_results = save_int_results

    def start(self):
        self._population0()
        for i in range(self._generations):
            self.evolutionary_process(i)
        return self._population

    def _population0(self):
        self._population = self._generator.generate(self._population_size)

    def evolutionary_process(self, gen):
        # making mutations
        mutants = [ind.mutate(gen=gen) for ind in
                   rd.choice(self._population, self._mutation_size, self._population_size < self._mutation_size)]

        # creating "new population"
        new_population = np.concatenate([self._population, mutants])
        new_population = new_population[np.argsort(IndividualSelection.generate_fitnesses(new_population, gen))]
        # new_population.sort(key=lambda x: x.fitness())

        # crossover
        parents = ParentSelection.get_method(self._parent_selection)(new_population, self._kids_size)
        new_population = np.concatenate([new_population, [parent[0] * parent[1] for parent in parents]])

        self._population = IndividualSelection.get_method(self._selection)(new_population, self._population_size, gen)
        if gen % 100 == 0:
            self._print(f"Gen {gen}")
        if gen % 500 == 0:
            self._save_results(gen)

    def _print(self, message):
        if self._logging:
            print(f"GA - {datetime.now()} - {message}")

    def _save_results(self, gen):
        if self._save_int_results:
            self._population[0].save(f"int_output/{gen}.png")

class IndividualSelection:

    @staticmethod
    def get_method(method="top"):
        methods = {"tournament": IndividualSelection.tournament,
                   "sigma-cut": IndividualSelection.sigma_out,
                   "top": IndividualSelection.top}
        return methods[method]

    @staticmethod
    def tournament(population, n, gen):
        pass

    @staticmethod
    def sigma_out(population, n, gen):
        pass

    @staticmethod
    def top(population, n, gen):
        return population[np.argsort(IndividualSelection.generate_fitnesses(population, gen))][:n]

    @staticmethod
    def generate_fitnesses(population, gen):
        return np.array([ind.fitness(gen=gen) for ind in population])


class ParentSelection:

    @staticmethod
    def get_method(method="panmixis"):
        methods = {"panmixis": ParentSelection.panmixis,
                   "random_outbreeding": ParentSelection.random_outbreeding}
        return methods[method]

    @staticmethod
    def panmixis(population, n):
        return np.array([rd.choice(population, 2, replace=False) for _ in range(n)])

    @staticmethod
    def random_outbreeding(population, n):
        """
        This method returns parents that are selected by outbreeding algorithm.
        :param population: sorted array of population
        :return: pairs of parents
        """

        pop_size = len(population)
        half1 = population[:pop_size // 2]
        half2 = population[pop_size // 2:]
        return np.array([rd.choice(half1), rd.choice(half2)] for _ in range(n))
