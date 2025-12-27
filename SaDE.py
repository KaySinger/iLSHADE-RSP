import numpy as np

class SaDE:
    """
    Self-adaptive Differential Evolution (SaDE)
    Framework-aligned implementation for benchmarking (CEC style)

    Mutation strategies:
        1) DE/rand/1
        2) DE/current-to-best/1
    """

    def __init__(
        self,
        func,
        bounds,
        pop_size=None,
        max_evals=None,
        dim=None,
        tol=None,
    ):
        self.fitness = func
        self.bounds = np.array(bounds)
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]

        self.dim = dim if dim is not None else len(bounds)
        self.pop_size = pop_size if pop_size is not None else 10 * self.dim
        self.max_evals = max_evals if max_evals is not None else 10000 * self.dim
        self.tol = tol

        # Strategy probabilities
        self.p1 = 0.5
        self.p2 = 0.5

        # Success / failure counters
        self.ns1 = self.ns2 = 0
        self.nf1 = self.nf2 = 0

        # Control parameters
        self.F = 0.5
        self.CRm = 0.5
        self.CR = np.full(self.pop_size, self.CRm)
        self.CR_set = []

        # Runtime containers
        self.NFE = 0
        self.population = None
        self.fitness_values = None
        self.best_idx = None

    # ======================================================
    # Initialization
    # ======================================================
    def _initialize(self):
        self.population = self.lb + (self.ub - self.lb) * np.random.rand(
            self.pop_size, self.dim
        )
        self.fitness_values = np.array(
            [self.fitness(x) for x in self.population]
        )
        self.NFE += self.pop_size
        self.best_idx = np.argmin(self.fitness_values)

    # ======================================================
    # Mutation
    # ======================================================
    def _mutation(self):
        mutants = np.empty_like(self.population)
        strategies = np.random.rand(self.pop_size) < self.p1

        for i in range(self.pop_size):
            if strategies[i]:  # DE/rand/1
                idxs = np.random.choice(
                    [j for j in range(self.pop_size) if j != i],
                    3,
                    replace=False,
                )
                a, b, c = self.population[idxs]
                mutants[i] = a + self.F * (b - c)
            else:  # DE/current-to-best/1
                idxs = np.random.choice(
                    [j for j in range(self.pop_size) if j != i],
                    2,
                    replace=False,
                )
                a, b = self.population[idxs]
                best = self.population[self.best_idx]
                mutants[i] = (
                    self.population[i]
                    + self.F * (best - self.population[i])
                    + self.F * (a - b)
                )
        return mutants, strategies

    # ======================================================
    # Crossover
    # ======================================================
    def _crossover(self, mutants):
        trials = self.population.copy()
        for i in range(self.pop_size):
            jrand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.CR[i] or j == jrand:
                    trials[i, j] = mutants[i, j]
        return trials

    # ======================================================
    # Selection
    # ======================================================
    def _selection(self, trials, strategies):
        for i in range(self.pop_size):
            trial_fit = self.fitness(trials[i])
            self.NFE += 1

            if trial_fit < self.fitness_values[i]:
                self.population[i] = trials[i]
                self.fitness_values[i] = trial_fit
                self.CR_set.append(self.CR[i])

                if strategies[i]:
                    self.ns1 += 1
                else:
                    self.ns2 += 1
            else:
                if strategies[i]:
                    self.nf1 += 1
                else:
                    self.nf2 += 1

        self.best_idx = np.argmin(self.fitness_values)

    # ======================================================
    # Main Optimization Loop
    # ======================================================
    def optimize(self):
        self._initialize()

        fes_history = []
        best_history = []

        while self.NFE < self.max_evals:
            # Update parameters
            self.F = np.clip(np.random.normal(0.5, 0.3), 0, 2)

            self.CR = np.clip(
                np.random.normal(self.CRm, 0.1, self.pop_size), 0, 1
            )

            # Evolution
            mutants, strategies = self._mutation()
            trials = self._crossover(mutants)
            self._selection(trials, strategies)

            # Update memories
            if len(self.CR_set) > 0:
                self.CRm = np.mean(self.CR_set)
                self.CR_set.clear()

            if (self.ns1 + self.nf1 + self.ns2 + self.nf2) > 0:
                self.p1 = (
                    self.ns1 * (self.ns2 + self.nf2)
                    / (
                        self.ns2 * (self.ns1 + self.nf1)
                        + self.ns1 * (self.ns2 + self.nf2)
                    )
                )
                self.p2 = 1 - self.p1

                self.ns1 = self.ns2 = self.nf1 = self.nf2 = 0

            # History
            fes_history.append(self.NFE)
            best_history.append(self.fitness_values[self.best_idx])

            if self.tol is not None and best_history[-1] <= self.tol:
                break

        best_x = self.population[self.best_idx]
        best_f = self.fitness_values[self.best_idx]

        return best_x, best_f, (fes_history, best_history)
