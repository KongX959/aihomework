import numpy as np
import random
import time
from typing import List, Tuple
import math

class GeneticAlgTSP:
    """
    遗传算法求解TSP问题
    
    主要特性：
    1. 多种初始化策略（贪心+随机）
    2. 锦标赛选择
    3. 顺序交叉（OX）算子
    4. 自适应变异率
    5. 精英保留策略
    6. 多种局部搜索优化
    """
    
    def __init__(self, filename: str, pop_size: int = 100, elite_rate: float = 0.1,
                 mutation_rate: float = 0.02, crossover_rate: float = 0.9):
        """
        构造函数：读取TSP数据并初始化种群
        
        Args:
            filename: TSP数据集文件名
            pop_size: 种群大小（默认100）
            elite_rate: 精英比例（默认0.1，即10%）
            mutation_rate: 基础变异率（默认0.02）
            crossover_rate: 交叉率（默认0.9）
        """
        self.filename = filename
        self.pop_size = pop_size
        self.elite_rate = elite_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # 读取城市坐标
        self.cities = self._read_tsp_file(filename)
        self.n_cities = len(self.cities)
        
        # 预计算距离矩阵，提高效率
        self.dist_matrix = self._compute_distance_matrix()
        
        # 初始化种群
        self.population = self._initialize_population()
        
        # 记录最佳解
        self.best_solution = None
        self.best_distance = float('inf')
        self.best_iteration = 0
        
        # 统计信息
        self.fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        self.improvement_count = 0
        
        print(f"读取TSP文件: {filename}")
        print(f"城市数量: {self.n_cities}")
        print(f"种群大小: {pop_size}")
        print(f"距离矩阵已预计算: {self.dist_matrix.shape}")
    
    def _read_tsp_file(self, filename: str) -> np.ndarray:
        """
        读取TSP数据文件
        
        Args:
            filename: TSP文件名
            
        Returns:
            numpy数组，形状为(n_cities, 2)，包含城市坐标
        """
        coordinates = []
        reading_coords = False
        
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # 检测坐标段的开始
                    if line.startswith('NODE_COORD_SECTION') or line.startswith('DISPLAY_DATA_SECTION'):
                        reading_coords = True
                        continue
                    
                    # 检测坐标段的结束
                    if line == 'EOF' or line.startswith('EOF'):
                        break
                    
                    if reading_coords and line:
                        parts = line.split()
                        if len(parts) >= 3:
                            # 格式：city_id x_coord y_coord
                            try:
                                x = float(parts[1])
                                y = float(parts[2])
                                coordinates.append([x, y])
                            except (ValueError, IndexError):
                                continue
        except FileNotFoundError:
            print(f"错误：找不到文件 {filename}")
            # 生成随机测试数据
            np.random.seed(42)
            coordinates = np.random.rand(38, 2) * 100
            print(f"使用随机生成的38个城市数据作为测试")
        
        return np.array(coordinates)
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """
        预计算城市间距离矩阵
        
        Returns:
            距离矩阵，形状为(n_cities, n_cities)
        """
        n = self.n_cities
        dist_matrix = np.zeros((n, n))
        
        # 使用欧几里得距离
        for i in range(n):
            for j in range(i+1, n):
                dx = self.cities[i, 0] - self.cities[j, 0]
                dy = self.cities[i, 1] - self.cities[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    def _initialize_population(self) -> List[List[int]]:
        """
        初始化种群：混合使用贪心策略和随机策略
        
        Returns:
            初始种群列表
        """
        population = []
        n = self.n_cities
        
        # 1. 随机生成一部分个体（70%）
        n_random = int(self.pop_size * 0.7)
        for _ in range(n_random):
            tour = list(range(n))
            random.shuffle(tour)
            population.append(tour)
        
        # 2. 使用最近邻贪心策略生成一部分个体（20%）
        n_greedy = int(self.pop_size * 0.2)
        for start_city in range(min(n_greedy, n)):
            tour = self._nearest_neighbor_tour(start_city)
            population.append(tour)
        
        # 3. 剩余用随机生成的补齐
        while len(population) < self.pop_size:
            tour = list(range(n))
            random.shuffle(tour)
            population.append(tour)
        
        return population
    
    def _nearest_neighbor_tour(self, start_city: int) -> List[int]:
        """
        使用最近邻算法生成一个初始解
        
        Args:
            start_city: 起始城市编号
            
        Returns:
            城市访问顺序列表
        """
        n = self.n_cities
        unvisited = set(range(n))
        unvisited.remove(start_city)
        tour = [start_city]
        current = start_city
        
        while unvisited:
            # 找到最近未访问城市
            next_city = min(unvisited, key=lambda x: self.dist_matrix[current, x])
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        
        return tour
    
    def _calculate_distance(self, tour: List[int]) -> float:
        """
        计算一条路径的总距离
        
        Args:
            tour: 城市访问顺序
            
        Returns:
            总距离
        """
        distance = 0.0
        for i in range(len(tour)):
            from_city = tour[i]
            to_city = tour[(i + 1) % len(tour)]
            distance += self.dist_matrix[from_city, to_city]
        return distance
    
    def _calculate_fitness(self, tour: List[int]) -> float:
        """
        计算适应度（距离的倒数）
        
        Args:
            tour: 城市访问顺序
            
        Returns:
            适应度值
        """
        distance = self._calculate_distance(tour)
        # 使用指数变换来增强选择压力
        return 1.0 / (distance + 1e-10)
    
    def _tournament_selection(self, fitness_values: np.ndarray, tournament_size: int = 3) -> int:
        """
        锦标赛选择
        
        Args:
            fitness_values: 种群适应度数组
            tournament_size: 锦标赛规模
            
        Returns:
            选中个体的索引
        """
        # 随机选择tournament_size个个体
        candidates = random.sample(range(len(self.population)), tournament_size)
        # 返回适应度最高的个体索引
        return max(candidates, key=lambda x: fitness_values[x])
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        顺序交叉（OX算子）
        
        Args:
            parent1: 父代1
            parent2: 父代2
            
        Returns:
            两个子代
        """
        n = len(parent1)
        
        # 随机选择交叉点
        point1, point2 = sorted(random.sample(range(n), 2))
        
        # 创建子代1
        child1 = [-1] * n
        child1[point1:point2+1] = parent1[point1:point2+1]
        
        # 从parent2中填充剩余位置
        pos = (point2 + 1) % n
        for city in parent2[point2+1:] + parent2[:point2+1]:
            if city not in child1:
                child1[pos] = city
                pos = (pos + 1) % n
        
        # 创建子代2（对称操作）
        child2 = [-1] * n
        child2[point1:point2+1] = parent2[point1:point2+1]
        
        pos = (point2 + 1) % n
        for city in parent1[point2+1:] + parent1[:point2+1]:
            if city not in child2:
                child2[pos] = city
                pos = (pos + 1) % n
        
        return child1, child2
    
    def _swap_mutation(self, tour: List[int]) -> List[int]:
        """
        交换突变：随机交换两个城市的位置
        
        Args:
            tour: 原始路径
            
        Returns:
            变异后的路径
        """
        mutated = tour.copy()
        i, j = random.sample(range(len(tour)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    def _inversion_mutation(self, tour: List[int]) -> List[int]:
        """
        倒置突变：反转一段子路径
        
        Args:
            tour: 原始路径
            
        Returns:
            变异后的路径
        """
        mutated = tour.copy()
        i, j = sorted(random.sample(range(len(tour)), 2))
        mutated[i:j+1] = reversed(mutated[i:j+1])
        return mutated
    
    def _scramble_mutation(self, tour: List[int]) -> List[int]:
        """
        打乱突变：随机打乱一段子路径
        
        Args:
            tour: 原始路径
            
        Returns:
            变异后的路径
        """
        mutated = tour.copy()
        i, j = sorted(random.sample(range(len(tour)), 2))
        sub = mutated[i:j+1]
        random.shuffle(sub)
        mutated[i:j+1] = sub
        return mutated
    
    def _adaptive_mutation(self, tour: List[int], generation: int, total_generations: int) -> List[int]:
        """
        自适应变异：根据不同阶段选择不同变异策略
        
        Args:
            tour: 原始路径
            generation: 当前代数
            total_generations: 总代数
            
        Returns:
            变异后的路径
        """
        progress = generation / total_generations
        
        # 早期：多使用大范围变异以探索
        if progress < 0.3:
            prob = random.random()
            if prob < 0.4:
                return self._swap_mutation(tour)
            elif prob < 0.7:
                return self._inversion_mutation(tour)
            else:
                return self._scramble_mutation(tour)
        # 中期：平衡探索和利用
        elif progress < 0.7:
            prob = random.random()
            if prob < 0.3:
                return self._swap_mutation(tour)
            elif prob < 0.8:
                return self._inversion_mutation(tour)
            else:
                return self._scramble_mutation(tour)
        # 后期：多用精细调整
        else:
            prob = random.random()
            if prob < 0.2:
                return self._swap_mutation(tour)
            elif prob < 0.9:
                return self._inversion_mutation(tour)
            else:
                return self._scramble_mutation(tour)
    
    def _two_opt_local_search(self, tour: List[int], max_iterations: int = 50) -> List[int]:
        """
        2-opt局部搜索优化
        
        Args:
            tour: 原始路径
            max_iterations: 最大迭代次数
            
        Returns:
            优化后的路径
        """
        improved = True
        best_tour = tour.copy()
        best_dist = self._calculate_distance(best_tour)
        n = len(tour)
        
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(n - 1):
                for j in range(i + 2, n):
                    # 计算2-opt交换后的距离变化
                    if j == n - 1 and i == 0:
                        continue  # 跳过会断开回路的情况
                    
                    # 旧边：(i, i+1) 和 (j, (j+1)%n)
                    # 新边：(i, j) 和 (i+1, (j+1)%n)
                    a, b = best_tour[i], best_tour[(i+1)%n]
                    c, d = best_tour[j], best_tour[(j+1)%n]
                    
                    old_dist = self.dist_matrix[a, b] + self.dist_matrix[c, d]
                    new_dist = self.dist_matrix[a, c] + self.dist_matrix[b, d]
                    
                    if new_dist < old_dist:
                        # 执行2-opt交换
                        best_tour[i+1:j+1] = reversed(best_tour[i+1:j+1])
                        best_dist = best_dist - old_dist + new_dist
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_tour
    
    def _calculate_population_diversity(self) -> float:
        """
        计算种群多样性
        
        Returns:
            多样性指标（0-1之间，1表示完全多样）
        """
        if len(self.population) < 2:
            return 0.0
        
        # 使用边出现频率的熵来衡量多样性
        n = self.n_cities
        edge_count = {}
        total_edges = 0
        
        for tour in self.population:
            for i in range(n):
                edge = tuple(sorted([tour[i], tour[(i+1)%n]]))
                edge_count[edge] = edge_count.get(edge, 0) + 1
                total_edges += 1
        
        # 计算熵
        entropy = 0.0
        for count in edge_count.values():
            p = count / total_edges
            entropy -= p * math.log2(p + 1e-10)
        
        # 归一化
        max_entropy = math.log2(len(self.population) * n)
        if max_entropy > 0:
            return entropy / max_entropy
        return 0.0
    
    def iterate(self, num_iterations: int) -> List[int]:
        """
        执行遗传算法迭代
        
        Args:
            num_iterations: 迭代轮数
            
        Returns:
            当前找到的最优解，格式为城市编号排列的列表
        """
        n_elite = max(2, int(self.pop_size * self.elite_rate))
        start_time = time.time()
        
        print(f"\n开始遗传算法迭代: {num_iterations}代")
        print(f"种群大小: {self.pop_size}, 精英数量: {n_elite}")
        print(f"交叉率: {self.crossover_rate}, 基础变异率: {self.mutation_rate}")
        print("-" * 60)
        
        for generation in range(num_iterations):
            # 1. 计算适应度
            fitness_values = np.array([self._calculate_fitness(tour) 
                                      for tour in self.population])
            
            # 2. 精英保留
            sorted_indices = np.argsort(fitness_values)[::-1]
            elites = [self.population[i].copy() for i in sorted_indices[:n_elite]]
            
            # 3. 更新最佳解
            current_best_idx = sorted_indices[0]
            current_best_tour = self.population[current_best_idx]
            current_best_dist = self._calculate_distance(current_best_tour)
            
            if current_best_dist < self.best_distance:
                self.best_distance = current_best_dist
                self.best_solution = current_best_tour.copy()
                self.best_iteration = generation
                self.improvement_count += 1
                
                # 对最佳解进行局部搜索
                if self.n_cities <= 100:  # 只在城市数较少时使用局部搜索
                    improved_tour = self._two_opt_local_search(self.best_solution)
                    improved_dist = self._calculate_distance(improved_tour)
                    if improved_dist < self.best_distance:
                        self.best_distance = improved_dist
                        self.best_solution = improved_tour
                        # 将优化后的解放回种群
                        self.population[0] = improved_tour.copy()
            
            # 4. 生成新一代
            new_population = elites.copy()
            
            # 5. 交叉和变异
            while len(new_population) < self.pop_size:
                # 锦标赛选择父代
                parent1_idx = self._tournament_selection(fitness_values)
                parent2_idx = self._tournament_selection(fitness_values)
                
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                
                # 交叉
                if random.random() < self.crossover_rate:
                    child1, child2 = self._order_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # 自适应变异率
                current_mutation_rate = self.mutation_rate * (1 + 0.5 * (1 - generation/num_iterations))
                
                # 变异
                if random.random() < current_mutation_rate:
                    child1 = self._adaptive_mutation(child1, generation, num_iterations)
                if random.random() < current_mutation_rate:
                    child2 = self._adaptive_mutation(child2, generation, num_iterations)
                
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            # 6. 更新种群
            self.population = new_population[:self.pop_size]
            
            # 7. 记录统计信息
            avg_fitness = np.mean(fitness_values)
            diversity = self._calculate_population_diversity()
            
            self.fitness_history.append(self.best_distance)
            self.avg_fitness_history.append(avg_fitness)
            self.diversity_history.append(diversity)
            
            # 8. 输出进度
            if (generation + 1) % max(1, num_iterations // 10) == 0:
                elapsed = time.time() - start_time
                print(f"第{generation+1:4d}代: 最佳距离={self.best_distance:.2f}, "
                      f"平均适应度={avg_fitness:.6f}, 多样性={diversity:.3f}, "
                      f"用时={elapsed:.1f}秒")
        
        # 最终优化
        if self.best_solution is not None:
            final_tour = self._two_opt_local_search(self.best_solution, max_iterations=100)
            final_dist = self._calculate_distance(final_tour)
            if final_dist < self.best_distance:
                self.best_distance = final_dist
                self.best_solution = final_tour
        
        total_time = time.time() - start_time
        print("-" * 60)
        print(f"迭代完成！")
        print(f"最佳路径长度: {self.best_distance:.2f}")
        print(f"找到最佳解的代数: {self.best_iteration}")
        print(f"改进次数: {self.improvement_count}")
        print(f"总用时: {total_time:.2f}秒")
        
        return self.best_solution
    
    def get_statistics(self) -> dict:
        """
        获取算法运行统计信息
        
        Returns:
            包含统计信息的字典
        """
        return {
            'best_distance': self.best_distance,
            'best_solution': self.best_solution,
            'best_iteration': self.best_iteration,
            'fitness_history': self.fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'diversity_history': self.diversity_history,
            'improvement_count': self.improvement_count,
            'n_cities': self.n_cities,
            'pop_size': self.pop_size
        }
    
    def plot_results(self, save_path: str = None):
        """
        绘制算法运行结果（需要matplotlib）
        
        Args:
            save_path: 图片保存路径，None则显示
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. 最佳距离变化
            ax1 = axes[0, 0]
            ax1.plot(self.fitness_history, 'b-', linewidth=1)
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Best Distance')
            ax1.set_title('Best Tour Distance Over Generations')
            ax1.grid(True, alpha=0.3)
            
            # 2. 平均适应度变化
            ax2 = axes[0, 1]
            ax2.plot(self.avg_fitness_history, 'g-', linewidth=1)
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Average Fitness')
            ax2.set_title('Population Average Fitness')
            ax2.grid(True, alpha=0.3)
            
            # 3. 种群多样性
            ax3 = axes[1, 0]
            ax3.plot(self.diversity_history, 'r-', linewidth=1)
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Diversity')
            ax3.set_title('Population Diversity')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
            
            # 4. 最佳路径可视化（仅当城市数较少时）
            ax4 = axes[1, 1]
            if self.best_solution and self.n_cities <= 200:
                cities = self.cities
                tour = self.best_solution + [self.best_solution[0]]
                x_coords = cities[tour, 0]
                y_coords = cities[tour, 1]
                
                ax4.plot(x_coords, y_coords, 'b-', linewidth=0.8, alpha=0.6)
                ax4.scatter(cities[:, 0], cities[:, 1], c='red', s=20, zorder=5)
                ax4.scatter(cities[tour[0], 0], cities[tour[0], 1], 
                           c='green', s=100, marker='*', zorder=6, label='Start')
                ax4.set_xlabel('X Coordinate')
                ax4.set_ylabel('Y Coordinate')
                ax4.set_title(f'Best Tour (Distance: {self.best_distance:.2f})')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.set_aspect('equal')
            else:
                ax4.text(0.5, 0.5, f'City count ({self.n_cities}) too large\nfor visualization',
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Best Tour Visualization')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"结果图已保存至: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("未安装matplotlib，无法绘图。可使用 pip install matplotlib 安装。")
        
        except Exception as e:
            print(f"绘图时发生错误: {e}")


# 使用示例和测试
if __name__ == "__main__":
    # 测试1：使用小规模数据集
    print("=" * 60)
    print("测试1: 小规模TSP问题 (dj38.tsp)")
    print("=" * 60)
    
    try:
        # 创建遗传算法求解器
        ga_small = GeneticAlgTSP("dj38.tsp", pop_size=100)
        
        # 运行迭代
        best_tour = ga_small.iterate(num_iterations=200)
        
        # 输出结果
        print(f"\n最佳路径: {best_tour[:10]}... (显示前10个城市)")
        print(f"路径长度: {ga_small.best_distance:.2f}")
        
        # 获取统计信息
        stats = ga_small.get_statistics()
        print(f"改进次数: {stats['improvement_count']}")
        
        # 绘图（如果安装了matplotlib）
        # ga_small.plot_results("ga_results_dj38.png")
        
    except Exception as e:
        print(f"测试1出错: {e}")
    
    print("\n" + "=" * 60)
    print("测试2: 大规模TSP问题 (zi929.tsp)")
    print("=" * 60)
    
    try:
        # 大规模问题使用较小的种群以节省时间
        ga_large = GeneticAlgTSP("zi929.tsp", pop_size=50)
        
        # 只运行较少代数进行演示
        best_tour_large = ga_large.iterate(num_iterations=50)
        
        # 输出结果
        print(f"\n最佳路径长度: {ga_large.best_distance:.2f}")
        
        # 获取统计信息
        stats_large = ga_large.get_statistics()
        print(f"改进次数: {stats_large['improvement_count']}")
        
    except Exception as e:
        print(f"测试2出错: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)