import numpy as np
from typing import Tuple, List, Dict, Any


class SaDE:
    """
    SaDE算法：根据论文Self-adaptive Differential Evolution Algorithm for Numerical Optimization复现
    特点：   1.采用两种变异策略，DE/rand/1和DE/current-to-pbest/1
            2.根据两种变异策略的成功次数来选择当前个体使用哪种变异策略
            3.采用自适应策略，自适应交叉概率CR，将成功的CR存储进CRm，然后根据该集合的均值更新uCR，CRm定期重置更新
    """

    def __init__(self, func, bounds, pop_size=None, max_evals=None, tol=None):
        """
        初始化SaDE算法

        参数:
        func: 目标函数（最小化）
        bounds: 变量边界，形状为(dim, 2)的numpy数组
        pop_size: 种群大小，默认为100
        max_evals: 最大函数评估次数，默认为10000*dim
        tol: 收敛容忍度
        """
        self.func = func  # 目标函数
        self.bounds = np.array(bounds)  # 变量边界
        self.dim = len(self.bounds)

        # 设置参数
        self.pop_size = 100 if pop_size is None else pop_size
        self.max_evals = 10000 * self.dim if max_evals is None else max_evals
        self.tol = tol

        # 算法状态
        self.num_evals = 0
        self.gen = 0

        # 初始化种群和适应度
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.pop_size, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.num_evals = self.pop_size

        # 历史记录
        self.fes_history = []  # 记录每次评估后的FES
        self.best_fitness_history = []  # 记录每次评估后的最佳适应度

        # 初始化最优解
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.pop[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

        # SaDE特定参数
        self.F = np.clip(np.random.normal(0.5, 0.3), 0.1, 2.0)  # 缩放因子
        self.CRm = 0.5  # CR均值
        self.CR = np.clip(np.random.normal(self.CRm, 0.1, self.pop_size), 0, 1)  # 交叉概率
        self.CR_set = []  # 成功的CR集合

        # 策略选择参数
        self.p1 = 0.5  # 策略1使用概率
        self.p2 = 0.5  # 策略2使用概率
        self.ns1 = 0  # 策略1成功次数
        self.ns2 = 0  # 策略2成功次数
        self.nf1 = 0  # 策略1失败次数
        self.nf2 = 0  # 策略2失败次数

    def record_history(self):
        """记录当前状态到历史"""
        self.fes_history.append(self.num_evals)
        self.best_fitness_history.append(self.best_fitness)

    def handle_boundary(self, individual, parent):
        """边界处理"""
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]

        # 当个别超界时修正
        individual = np.where(individual < low, (low + parent) / 2, individual)
        individual = np.where(individual > high, (high + parent) / 2, individual)

        # 确保在边界内
        individual = np.clip(individual, low, high)

        return individual

    def mutation_single(self, i: int, strategy: str) -> np.ndarray:
        """
        为单个个体进行变异操作

        参数:
        i: 当前个体索引
        strategy: 变异策略 'rand/1' 或 'current-to-best/1'

        返回:
        变异后的单个个体
        """
        if strategy == 'rand/1':
            # 随机选择三个不同的个体
            idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
            a, b, c = self.pop[idxs]
            mutant = a + self.F * (b - c)

        elif strategy == 'current-to-best/1':
            # 随机选择两个不同的个体
            idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 2, replace=False)
            a, b = self.pop[idxs]
            mutant = self.pop[i] + self.F * (self.best_solution - self.pop[i]) + self.F * (a - b)

        # 边界处理
        mutant = self.handle_boundary(mutant, self.pop[i])

        return mutant

    def crossover_single(self, i: int, mutant: np.ndarray) -> np.ndarray:
        """
        为单个个体进行交叉操作

        参数:
        i: 当前个体索引
        mutant: 变异后的个体

        返回:
        交叉后的试验个体
        """
        trial = self.pop[i].copy()

        # 确保至少有一个维度来自变异个体
        j_rand = np.random.randint(0, self.dim)

        for j in range(self.dim):
            if np.random.rand() < self.CR[i] or j == j_rand:
                trial[j] = mutant[j]

        return trial

    def update_parameters(self):
        """更新算法参数"""
        # 更新F值
        self.F = np.clip(np.random.normal(0.5, 0.3), 0.1, 2.0)

        # 每5代更新一次交叉率
        if self.gen % 5 == 0:
            self.CR = np.clip(np.random.normal(self.CRm, 0.1, self.pop_size), 0, 1)

        # 每25代更新CRm
        if self.gen % 25 == 0 and len(self.CR_set) > 0:
            self.CRm = np.mean(self.CR_set)
            self.CR_set = []

        # 每50代更新策略概率
        if self.gen % 50 == 0:
            # 计算策略选择概率（原论文公式）
            if (self.ns1 + self.nf1) > 0 and (self.ns2 + self.nf2) > 0:
                # 避免除以零
                denominator = self.ns2 * (self.ns1 + self.nf1) + self.ns1 * (self.ns2 + self.nf2)
                if denominator == 0:
                    self.p1 = 0.5
                else:
                    self.p1 = self.ns1 * (self.ns2 + self.nf2) / denominator
            else:
                self.p1 = 0.5

            self.p2 = 1.0 - self.p1

            # 重置计数器
            self.ns1 = 0
            self.ns2 = 0
            self.nf1 = 0
            self.nf2 = 0

    def optimize(self) -> Tuple[np.ndarray, float, Tuple[List[int], List[float]]]:
        # 记录初始状态
        self.record_history()

        # 主循环
        while self.num_evals < self.max_evals:
            # 记录当前最优值
            best_val = np.min(self.fitness)

            # 收敛检查
            if self.tol is not None and best_val <= self.tol:
                print(f"Converged at generation {self.gen}: {best_val}")
                break

            # 更新参数
            self.update_parameters()

            # 为每个个体选择变异策略
            strategy_choices = np.random.rand(self.pop_size)

            # 预生成随机索引（用于交叉操作）
            j_rand_values = np.random.randint(0, self.dim, self.pop_size)

            for i in range(self.pop_size):
                # 选择策略
                if strategy_choices[i] < self.p1:
                    # 使用rand/1策略
                    mutant = self.mutation_single(i, strategy='rand/1')
                    strategy_used = 0  # 标记策略类型：0表示rand/1
                else:
                    # 使用current-to-best/1策略
                    mutant = self.mutation_single(i, strategy='current-to-best/1')
                    strategy_used = 1  # 标记策略类型：1表示current-to-best/1

                # 交叉
                trial = self.pop[i].copy()
                j_rand = j_rand_values[i]

                for j in range(self.dim):
                    if np.random.rand() < self.CR[i] or j == j_rand:
                        trial[j] = mutant[j]

                # 评估
                trial_fitness = self.func(trial)
                self.num_evals += 1

                # 选择
                if trial_fitness < self.fitness[i]:
                    # 记录成功的CR值
                    self.CR_set.append(self.CR[i])

                    # 记录策略成功次数
                    if strategy_used == 0:  # rand/1
                        self.ns1 += 1
                    else:  # current-to-best/1
                        self.ns2 += 1

                    # 替换个体
                    self.pop[i] = trial
                    self.fitness[i] = trial_fitness

                    # 更新全局最优解
                    if trial_fitness < self.best_fitness:
                        self.best_solution = trial.copy()
                        self.best_fitness = trial_fitness
                else:
                    # 记录策略失败次数
                    if strategy_used == 0:  # rand/1
                        self.nf1 += 1
                    else:  # current-to-best/1
                        self.nf2 += 1

            # 更新迭代计数
            self.gen += 1

            # 记录历史
            self.record_history()

            # 打印进度
            if self.gen % 100 == 0 or self.num_evals >= self.max_evals:
                print(f"Generation {self.gen}, Best: {self.best_fitness:.6e}, "
                      f"NFE: {self.num_evals}/{self.max_evals}")

        return self.best_solution, self.best_fitness, (self.fes_history, self.best_fitness_history)