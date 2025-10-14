import random
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    def __init__(self, population_size=4, chromosome_length=5, crossover_rate=0.7, mutation_rate=0.01, generations=20):
        # 参数设置
        self.population_size = population_size  # 种群大小
        self.chromosome_length = chromosome_length  # 染色体长度（5位二进制）
        self.crossover_rate = crossover_rate  # 交叉概率
        self.mutation_rate = mutation_rate  # 变异概率
        self.generations = generations  # 迭代代数

        # 初始化种群
        self.population = self.initialize_population()

        # 记录每代的最佳适应度，用于绘图
        self.best_fitness_history = []
        self.average_fitness_history = []

    def initialize_population(self):
        """初始化种群 - 随机生成二进制染色体"""
        population = []
        for _ in range(self.population_size):
            # 生成一个5位的二进制染色体
            chromosome = ''.join(str(random.randint(0, 1)) for _ in range(self.chromosome_length))
            population.append(chromosome)
        return population

    def decode(self, chromosome):
        """将二进制染色体解码为十进制数"""
        return int(chromosome, 2)

    def fitness_function(self, x):
        """适应度函数 - f(x) = x²"""
        return x ** 2

    def evaluate_fitness(self):
        """评估种群中每个个体的适应度"""
        fitness_scores = []
        for chromosome in self.population:
            x = self.decode(chromosome)
            fitness = self.fitness_function(x)
            fitness_scores.append(fitness)
        return fitness_scores

    def selection(self, fitness_scores):
        """选择操作 - 使用轮盘赌选择法"""
        total_fitness = sum(fitness_scores)

        # 计算每个个体的选择概率
        probabilities = [fitness / total_fitness for fitness in fitness_scores]

        # 轮盘赌选择
        selected_population = []
        for _ in range(self.population_size):
            r = random.random()
            cumulative_probability = 0
            for i, prob in enumerate(probabilities):
                cumulative_probability += prob
                if r <= cumulative_probability:
                    selected_population.append(self.population[i])
                    break

        return selected_population

    def crossover(self, parent1, parent2):
        """交叉操作 - 单点交叉"""
        if random.random() < self.crossover_rate:
            # 随机选择交叉点
            crossover_point = random.randint(1, self.chromosome_length - 1)

            # 创建后代
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]

            return child1, child2
        else:
            # 不进行交叉，直接复制父母
            return parent1, parent2

    def mutation(self, chromosome):
        """变异操作 - 位翻转变异"""
        mutated_chromosome = ""
        for gene in chromosome:
            if random.random() < self.mutation_rate:
                # 翻转基因：0变1，1变0
                mutated_gene = '1' if gene == '0' else '0'
                mutated_chromosome += mutated_gene
            else:
                mutated_chromosome += gene
        return mutated_chromosome

    def run(self):
        """运行遗传算法"""
        print("遗传算法求解 f(x) = x² 在 [0, 31] 上的最大值")
        print("=" * 50)

        for generation in range(self.generations):
            # 评估当前种群的适应度
            fitness_scores = self.evaluate_fitness()

            # 记录最佳和平均适应度
            best_fitness = max(fitness_scores)
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            self.best_fitness_history.append(best_fitness)
            self.average_fitness_history.append(average_fitness)

            # 找到最佳个体
            best_index = fitness_scores.index(best_fitness)
            best_chromosome = self.population[best_index]
            best_x = self.decode(best_chromosome)

            # 输出当前代的信息
            print(f"第 {generation + 1:2d} 代: "
                  f"最佳个体 x={best_x:2d}, f(x)={best_fitness:3d}, "
                  f"平均适应度={average_fitness:6.2f}")
            print(f"        种群: {[self.decode(chrom) for chrom in self.population]}")

            # 选择
            selected_population = self.selection(fitness_scores)

            # 交叉
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[(i + 1) % self.population_size]  # 循环配对
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([child1, child2])

            # 变异
            self.population = [self.mutation(chrom) for chrom in new_population]

        # 最终结果
        final_fitness = self.evaluate_fitness()
        best_final_index = final_fitness.index(max(final_fitness))
        best_solution = self.decode(self.population[best_final_index])
        best_fitness_value = max(final_fitness)

        print("=" * 50)
        print(f"最终结果: x = {best_solution}, f(x) = {best_fitness_value}")

        return best_solution, best_fitness_value

    def plot_progress(self):
        """绘制算法收敛过程"""
        plt.figure(figsize=(10, 6))
        generations = range(1, len(self.best_fitness_history) + 1)

        plt.plot(generations, self.best_fitness_history, 'b-', label='最佳适应度', linewidth=2)
        plt.plot(generations, self.average_fitness_history, 'r--', label='平均适应度', linewidth=2)

        plt.xlabel('代数')
        plt.ylabel('适应度')
        plt.title('遗传算法收敛过程')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# 运行遗传算法
if __name__ == "__main__":
    # 创建遗传算法实例
    ga = GeneticAlgorithm(
        population_size=4,
        chromosome_length=5,
        crossover_rate=0.7,
        mutation_rate=0.01,
        generations=20
    )

    # 运行算法
    best_solution, best_fitness = ga.run()

    # 绘制收敛图
    ga.plot_progress()

    # 理论最大值验证
    theoretical_max = 31 ** 2
    print(f"\n理论最大值: f(31) = {theoretical_max}")
    print(f"找到的最佳值: f({best_solution}) = {best_fitness}")
    print(f"接近程度: {best_fitness / theoretical_max * 100:.2f}%")