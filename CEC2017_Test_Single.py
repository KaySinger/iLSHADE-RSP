import numpy as np
from matplotlib import pyplot as plt

from iLSHADE_RSP import iLSHADE_RSP
from SaDE import SaDE
import cec2017.functions as functions

# =========================
# 1. CEC2017 测试函数封装
# =========================
def cec2017_f1(x):
    """
    CEC2017 F1
    输入：x (1D or 2D)
    输出：标量 fitness
    """
    x = np.atleast_2d(x)
    return functions.f1(x)[0]


# =========================
# 2. 测试参数设置
# =========================
D = 30
bounds = [(-100, 100)] * D
max_evals = 10000 * D
pop_size = None
optimal_value = 100    # CEC2017 F1 的理论最优值


# =========================
# 3. 测试函数
# =========================
def run():
    objective = cec2017_f1
    tol = optimal_value + 1e-8
    print("iLSHADE-RSP started on CEC2017 F1...")

    optimizer = SaDE(
        func=objective,
        bounds=bounds,
        pop_size=pop_size,
        max_evals=max_evals,
        tol=tol
    )

    best_x, best_fitness, history = optimizer.optimize()

    fes_history, best_fitness_history = history
    gap_history = np.array(best_fitness_history) - optimal_value

    print("iL-SHADE-RSP completed.")
    print(f"Final fitness  : {best_fitness}")
    print(f"Final gap      : {best_fitness - optimal_value}")

    return fes_history, gap_history


# =========================
# 4. 性能绘图
# =========================
def plot_convergence(fes, gap):
    plt.figure(figsize=(9, 6))
    plt.plot(fes, gap, lw=2, label="iLSHADE-RSP")

    plt.yscale("log")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Fitness Gap (log scale)")
    plt.title("iLSHADE-RSP Performance on CEC2017 F1 (D=30)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# 5. 主入口
# =========================
if __name__ == "__main__":
    fes, gap = run()
    plot_convergence(fes, gap)