import numpy as np
from scipy.stats import norm, cauchy

class LSHADE_RSP:
    def __init__(self, func, bounds, pop_size=None, max_evals=None, H=5, tol=None):
        """
        L-SHADE-RSP优化算法类
        基于jSO算法加入RSP机制，进一步提升算法的收敛能力
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.N_init = int(round(75 * self.dim**(2/3))) if pop_size is None else pop_size
        self.N_min = 4
        self.max_evals = 10000 * self.dim if max_evals is None else max_evals
        self.H = H
        self.tol = tol
        self.num_evals = self.N_init
        self.gen = 0
        self.k = 3

        # 初始化历史记忆
        self.F_memory = np.ones(self.H) * 0.3
        self.CR_memory = np.ones(self.H) * 0.8
        self.hist_idx = 0

        # 初始化种群和存档
        self.N_current = self.N_init  # 当前种群大小
        self.archive_size = 2.6
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.N_init, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.archive = []
        self.fes_history = []
        self.best_fitness_history = []

    def record_history(self):
        """记录当前状态到历史"""
        current_best = np.min(self.fitness)
        self.fes_history.append(self.num_evals)
        self.best_fitness_history.append(current_best)

    # 线性种群缩减策略
    def _linear_pop_size_reduction(self):

        return max(self.N_min, int(round(self.N_init - (self.N_init - self.N_min) * self.num_evals / self.max_evals)))

    # 边界处理
    def handle_boundary(self, individual, parent):
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]

        # 当个别超界时修正
        individual = np.where(individual < low, (low + parent) / 2, individual)
        individual = np.where(individual > high, (high + parent) / 2, individual)

        return individual

    def _calculate_ranks(self):
        """计算基于排名的选择概率"""
        # 按适应度排序（最小化问题）
        sorted_indices = np.argsort(self.fitness)
        ranks = np.zeros(self.N_current)

        # 分配排名（公式4）
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = self.k * (self.N_current - rank) + 1

        # 计算选择概率（公式5）
        probabilities = ranks / np.sum(ranks)
        return probabilities

    def mutant(self, F, i, p_best_indices, rank_probs):
        # current-to-pbest/1变异策略
        p_best_idx = np.random.choice(p_best_indices)
        p_best = self.pop[p_best_idx]

        # 选择 r1 ≠ i
        r1_candidates = np.delete(np.arange(self.N_current), i)
        r1_probs = rank_probs[r1_candidates]
        r1_probs /= np.sum(r1_probs)
        r1_idx = np.random.choice(r1_candidates, p=r1_probs)
        r1 = self.pop[r1_idx].flatten()

        # 选择 r2 ≠ i 且 ≠ r1，从当前种群+存档中
        r2_candidates = np.delete(np.arange(self.N_current), [i, r1_idx])
        r2_probs = rank_probs[r2_candidates]
        r2_probs /= np.sum(r2_probs)
        r2_idx = np.random.choice(r2_candidates, p=r2_probs)
        r2 = self.pop[r2_idx].flatten()

        # 变异操作
        if self.num_evals < 0.2 * self.max_evals:
            Fw = F * 0.7
        elif self.num_evals < 0.4 * self.max_evals:
            Fw = F * 0.8
        else:
            Fw = F * 1.2

        # 变异操作
        mutant = self.pop[i] + Fw * (p_best - self.pop[i]) + F * (r1 - r2)
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
        i = self.N_init
        max_gen = 0
        # 计算最大迭代次数
        while i < self.max_evals:
            max_gen += 1
            n = int(round(self.N_init - (self.N_init - self.N_min) * i / self.max_evals))
            i += n
        print(max_gen)

        """执行优化过程"""
        while self.num_evals < self.max_evals:
            S_F, S_CR, S_weights = [], [], []
            new_pop = []
            new_fitness = []

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
            rank_probs = self._calculate_ranks()
            p = 0.085 + (0.17 - 0.085) * self.num_evals / self.max_evals
            p_best_size = max(2, int(self.N_current * p))
            p_best_indices = np.argsort(self.fitness)[:p_best_size]

            for i in range(self.N_current):
                # 从历史记忆随机选择索引
                r = np.random.randint(0, self.H)
                if r == self.H - 1:
                    self.CR_memory[r] = 0.9
                    self.F_memory[r] = 0.9

                # 生成F和CR
                if np.isnan(self.CR_memory[r]):
                    CR = 0
                else:
                    CR = np.clip(np.random.normal(self.CR_memory[r], 0.1), 0.0, 1.0)
                F = cauchy.rvs(loc=self.F_memory[r], scale=0.1)
                while F > 1 or F < 0:
                    if F < 0:
                        F = cauchy.rvs(loc=self.F_memory[r], scale=0.1)
                    else:
                        F = 1

                # jSO的CR阶段式调整（根据当前评估次数）
                if self.num_evals < 0.25 * self.max_evals:
                    CR = max(CR, 0.7)
                elif self.num_evals < 0.5 * self.max_evals:
                    CR = max(CR, 0.6)

                # jSO的F时变调整
                if self.num_evals < 0.6 * self.max_evals and F > 0.7:
                    F = 0.7

                # current-to-pbest-w/1变异策略
                mutant = self.mutant(F, i, p_best_indices, rank_probs)

                # 交叉操作
                trial = self.cross(mutant, i, CR)
                trial_fitness = self.func(trial)

                # 贪心选择
                if trial_fitness < self.fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(trial_fitness)
                    # 记录成功参数和适应度改进量
                    S_F.append(F)
                    S_CR.append(CR)
                    S_weights.append(np.abs(self.fitness[i] - trial_fitness))
                    # 更新存档
                    self.archive.append(self.pop[i].copy())
                    if len(self.archive) > self.archive_size * self.N_current:
                        # 如果存档超过种群大小，则随机删除一些
                        for o in range(int(len(self.archive) - self.archive_size * self.N_current)):
                            self.archive.pop(np.random.randint(0, len(self.archive)))
                else:
                    new_pop.append(self.pop[i])
                    new_fitness.append(self.fitness[i])
            self.num_evals += self.N_current
            self.gen += 1
            self.record_history()

            # --- LPSR关键步骤 ---
            # 更新种群大小
            new_N = self._linear_pop_size_reduction()
            # 选择适应度最好的个体保留
            combined_fitness = np.array(new_fitness)
            survivor_indices = np.argsort(combined_fitness)[:new_N]
            # 更新种群和适应度
            self.pop = np.array([new_pop[i] for i in survivor_indices])
            self.fitness = np.array([new_fitness[i] for i in survivor_indices])
            self.N_current = new_N

            # 更新历史记忆（加权Lehmer均值）
            if np.any(S_F):
                F_lehmer = np.sum(np.array(S_F) ** 2 * S_weights) / np.sum(np.array(S_F) * S_weights)
                self.F_memory[self.hist_idx] = (F_lehmer + self.F_memory[self.hist_idx]) / 2
            else:
                self.F_memory[self.hist_idx] = self.F_memory[self.hist_idx]

            # CR部分
            if np.any(S_CR):
                if np.max(S_CR) != 0:
                    CR_lehmer = np.sum(np.array(S_CR) ** 2 * S_weights) / np.sum(np.array(S_CR) * S_weights)
                    self.CR_memory[self.hist_idx] = (CR_lehmer + self.CR_memory[self.hist_idx]) / 2
                else:
                    self.CR_memory[self.hist_idx] = 1

            # 移动历史指针
            self.hist_idx = (self.hist_idx + 1) % (self.H - 1)

            if self.gen % 100 == 0 or self.gen > max_gen:
                print(f"Iteration {self.gen}, Pop Size: {self.N_current}, Best: {np.min(self.fitness)}, Num_Evals: {self.num_evals}")

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], (self.fes_history, self.best_fitness_history)