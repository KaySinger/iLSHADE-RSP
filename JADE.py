import numpy as np

class JADE:
    def __init__(self, func, bounds, pop_size=None, max_evals=None, tol=1e-8):
        """
        JADE优化算法类
        J. Zhang and A. C. Sanderson, "JADE: Adaptive Differential Evolution With Optional External Archive,"
        in IEEE Transactions on Evolutionary Computation, vol. 13, no. 5, pp. 945-958, Oct. 2009, doi: 10.1109/TEVC.2009.2014613.

        参数：
        func: 目标函数（最小化）
        bounds: 变量边界的列表，例如 [(min1, max1), (min2, max2), ...]
        pop_size: 种群大小
        max_gen: 最大迭代次数
        tol: 收敛精度，达到时提前终止
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = 100 if pop_size is None else pop_size
        self.max_evals = 10000 * self.dim if max_evals is None else max_evals
        self.p = 0.1 # 贪婪系数
        self.c = 0.1 # 学习率
        self.tol = tol
        self.num_evals = 0
        self.archive_size = 2.0
        self.gen = 0

        # 初始化参数
        self.F_mean = 0.5
        self.CR_mean = 0.5
        self.archive = []
        self.fes_history = []  # 记录每次评估后的FES
        self.best_fitness_history = []  # 记录每次评估后的最佳适应度

        # 生成初始种群
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.num_evals += self.pop_size

    def record_history(self):
        """记录当前状态到历史"""
        current_best = np.min(self.fitness)
        self.fes_history.append(self.num_evals)
        self.best_fitness_history.append(current_best)

    # 边界处理
    def handle_boundary(self, individual, parent):
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]

        # 当个别超界时修正
        individual = np.where(individual < low, (low + parent) / 2, individual)
        individual = np.where(individual > high, (high + parent) / 2, individual)

        return individual

    # 变异current-to-pbest/1
    def mutant(self, F, i):
        # 选择p_best
        p_best_size = max(int(self.pop_size * self.p), 2)  # 至少1个
        p_best_indices = np.argsort(self.fitness)[:p_best_size]
        p_best_idx = np.random.choice(p_best_indices)
        p_best = self.pop[p_best_idx]

        # 选择r1（来自种群）
        r1_candidates = [idx for idx in range(self.pop_size)
                        if idx != i and not np.array_equal(self.pop[idx], p_best)]

        if len(r1_candidates) == 0:
            r1 = self.pop[np.random.choice([x for x in range(self.pop_size) if x != i])]
        else:
            r1 = self.pop[np.random.choice(r1_candidates)]

        # 合并种群和存档
        combined_pop = np.vstack([self.pop, np.array(self.archive)]) if self.archive else self.pop

        # 选择r2（来自合并种群）
        r2_candidates = []
        for idx in range(len(combined_pop)):
            if not np.array_equal(combined_pop[idx], self.pop[i]) and \
                    not np.array_equal(combined_pop[idx], p_best) and \
                    not np.array_equal(combined_pop[idx], r1):
                r2_candidates.append(idx)

        if len(r2_candidates) == 0:
            r2 = combined_pop[np.random.choice(len(combined_pop))]
        else:
            r2 = combined_pop[np.random.choice(r2_candidates)]

        # 变异操作
        mutant = self.pop[i] + F * (p_best - self.pop[i]) + F * (r1 - r2)
        mutant = self.handle_boundary(mutant, self.pop[i])

        return mutant

    def optimize(self):
        """执行优化过程"""
        while self.num_evals < self.max_evals:
            F_values, CR_values = [], []
            new_pop = []

            # 检查收敛条件
            best_val = np.min(self.fitness)
            if self.tol is not None and best_val <= self.tol:
                print(f"Converged at generation {self.gen + 1} with precision {best_val:.6e}")
                break
            elif self.num_evals >= self.max_evals:
                print(f"Converged at generation {self.gen + 1} with precision {best_val:.6e}")
                break

            for i in range(self.pop_size):
                # 生成F和CR
                F = np.clip(np.random.standard_cauchy() * 0.1 + self.F_mean, 0, 1)
                CR = np.clip(np.random.normal(self.CR_mean, 0.1), 0, 1)

                mutant = self.mutant(F, i)

                # 交叉操作
                cross_mask = np.random.rand(self.dim) < CR
                if not np.any(cross_mask):  # 如果全部未交叉
                    cross_mask[np.random.randint(self.dim)] = True  # 强制选择一个维度
                trial = np.where(cross_mask, mutant, self.pop[i])

                trial_fitness = self.func(trial)

                # 贪婪选择
                if trial_fitness < self.fitness[i]:
                    new_pop.append(trial)
                    self.fitness[i] = trial_fitness
                    F_values.append(F)
                    CR_values.append(CR)
                    # 更新存档
                    self.archive.append(self.pop[i].copy())
                    if len(self.archive) > self.archive_size * self.pop_size:
                        # 如果存档超过种群大小，则随机删除一些
                        for o in range(int(len(self.archive) - self.archive_size * self.pop_size)):
                            self.archive.pop(np.random.randint(0, len(self.archive)))
                else:
                    new_pop.append(self.pop[i])

            # 计算迭代次数和代数
            self.num_evals += self.pop_size
            self.gen += 1
            self.record_history()

            # 更新种群
            self.pop = np.array(new_pop)

            # 更新自适应参数
            if F_values:
                self.F_mean = (1 - self.c) * self.F_mean + self.c * (np.sum(np.square(F_values))) / np.sum(F_values)
            if CR_values:
                self.CR_mean = (1 - self.c) * self.CR_mean + self.c * np.mean(CR_values)

            if self.gen % 100 == 0 or self.num_evals >= self.max_evals:
                print(f"Iteration {self.gen}, Best fitness: {np.min(self.fitness)}, Num Evals: {self.num_evals}")

        # 返回最优解
        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], (self.fes_history, self.best_fitness_history)