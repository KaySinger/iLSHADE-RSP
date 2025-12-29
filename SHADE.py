import numpy as np

class SHADE:
    def __init__(self, func, bounds, pop_size=None, max_evals=None, H=100, tol=None):
        """
        SHADE优化算法类

        参数：
        func: 目标函数（最小化）
        bounds: 变量边界的列表，例如 [(min1, max1), (min2, max2), ...]
        pop_size: 种群大小
        max_gen: 最大迭代次数
        H: 历史记忆大小
        tol: 收敛精度，达到时提前终止
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = 100 if pop_size is None else pop_size
        self.max_evals = 10000 * self.dim if max_evals is None else max_evals
        self.num_evals = self.pop_size
        self.H = H
        self.tol = tol
        self.archive_size = self.pop_size
        self.gen = 0

        # 初始化历史记忆
        self.F_memory = [0.5] * H
        self.CR_memory = [0.5] * H
        self.hist_idx = 0

        # 初始化种群和存档
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.pop_size, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.archive = []
        self.fes_history = []  # 记录每次评估后的FES
        self.best_fitness_history = []  # 记录每次评估后的最佳适应度

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

    def resize_archive(self):
        if len(self.archive) > self.archive_size:
            np.random.shuffle(self.archive)
            self.archive = self.archive[:self.archive_size]


    def mutant(self, F, i, p_best_indices):
        # current-to-pbest/1变异策略
        p_best_idx = np.random.choice(p_best_indices)
        p_best = self.pop[p_best_idx]

        # 选择 r1 ≠ i
        r1_candidates = np.delete(np.arange(self.pop_size), i)
        r1_idx = np.random.choice(r1_candidates)
        r1 = self.pop[r1_idx].flatten()

        # 选择 r2 ≠ i 且 ≠ r1，从当前种群+存档中
        all_indices = np.arange(self.pop_size + len(self.archive))
        invalid_indices = [i, r1_idx] if r1_idx < self.pop_size else [i]

        valid_indices = np.setdiff1d(all_indices, invalid_indices)
        r2_idx = np.random.choice(valid_indices)

        if r2_idx >= self.pop_size:
            r2 = self.archive[r2_idx - self.pop_size]
        else:
            r2 = self.pop[r2_idx].flatten()

        # 变异操作
        mutant = self.pop[i] + F * (p_best - self.pop[i]) + F * (r1 - r2)
        mutant = self.handle_boundary(mutant, self.pop[i])

        return mutant

    def cross(self, mutant, i, CR):
        cross_chorm = self.pop[i].copy()
        j = np.random.randint(0, self.dim)  # 随机选择一个维度
        for k in range(self.dim):  # 对每个维度进行交叉
            if np.random.rand() < CR or k == j:  # 如果随机数小于交叉率或者维度为j
                cross_chorm[k] = mutant[k]  # 交叉
        return cross_chorm

    def optimize(self):
        """执行优化过程"""
        while self.num_evals < self.max_evals:
            S_F, S_CR, S_weights = [], [], []
            new_pop = []

            # 记录当前最优值
            best_val = np.min(self.fitness)
            # 收敛检查
            if self.tol is not None and best_val <= self.tol:
                print(f"Converged at generation {self.gen + 1}: {best_val}")
                break
            elif self.num_evals > self.max_evals:
                print(f"Converged at generation {self.gen + 1}: {best_val}")
                break

            # current-to-pbest/1变异策略
            p = 0.11
            p_best_size = max(2, int(self.pop_size * p))
            p_best_indices = np.argsort(self.fitness)[:p_best_size]

            for i in range(self.pop_size):
                # 从历史记忆随机选择索引
                r = np.random.randint(0, self.H)

                # 生成F和CR
                F = np.clip(np.random.standard_cauchy() * 0.1 + self.F_memory[r], 0, 1)
                CR = np.clip(np.random.normal(self.CR_memory[r], 0.1), 0, 1)

                # current-to-pbest/1变异策略
                mutant = self.mutant(F, i, p_best_indices)

                # 交叉操作
                trial = self.cross(mutant, i, CR)
                trial_fitness = self.func(trial)

                # 贪心选择
                if trial_fitness < self.fitness[i]:
                    new_pop.append(trial)
                    # 记录成功参数和适应度改进量
                    S_F.append(F)
                    S_CR.append(CR)
                    S_weights.append(self.fitness[i] - trial_fitness)  # 绝对改进量
                    # 更新适应度和存档
                    self.fitness[i] = trial_fitness
                    self.archive.append(self.pop[i].copy())
                else:
                    new_pop.append(self.pop[i])

            self.num_evals += self.pop_size
            self.gen += 1
            self.record_history()

            # 更新种群
            self.pop = np.array(new_pop)
            self.resize_archive()

            # 更新历史记忆
            if S_F:
                total_weight = np.sum(S_weights)
                if total_weight > 0:
                    # F使用Lehmer均值
                    F_lehmer = np.sum(np.array(S_F) ** 2 * S_weights) / np.sum(np.array(S_F) * S_weights)
                    # CR使用加权算术均值
                    CR_mean = np.sum(np.array(S_CR) * S_weights) / total_weight

                    # 更新记忆
                    self.F_memory[self.hist_idx] = F_lehmer
                    self.CR_memory[self.hist_idx] = CR_mean
                    self.hist_idx = (self.hist_idx + 1) % self.H

            if self.gen % 100 == 0 or self.num_evals >= self.max_evals:
                print(
                    f"Iteration {self.gen}, Pop Size: {self.pop_size}, Best: {np.min(self.fitness)}, Num_Evals: {self.num_evals}")

        # 返回最优解
        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], (self.fes_history, self.best_fitness_history)