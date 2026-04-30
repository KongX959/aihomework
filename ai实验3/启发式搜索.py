import heapq
import time
from copy import deepcopy

class PuzzleSolver:
    def __init__(self):
        # 修正：这是标准的15-puzzle目标状态
        self.goal_state = [[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12],
                          [13, 14, 15, 0]]
        self.goal_positions = {}
        for i in range(4):
            for j in range(4):
                self.goal_positions[self.goal_state[i][j]] = (i, j)
    
    def find_blank(self, state):
        """找到空格(0)的位置"""
        for i in range(4):
            for j in range(4):
                if state[i][j] == 0:
                    return i, j
        return -1, -1
    
    def get_neighbors(self, state):
        """获取当前状态的所有可能移动"""
        neighbors = []
        moves = []
        blank_i, blank_j = self.find_blank(state)
        
        # 四个可能的移动方向：上、下、左、右
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for di, dj in directions:
            ni, nj = blank_i + di, blank_j + dj
            if 0 <= ni < 4 and 0 <= nj < 4:
                # 创建新状态
                new_state = deepcopy(state)
                # 交换空格和目标位置的数字
                moved_number = new_state[ni][nj]
                new_state[blank_i][blank_j] = moved_number
                new_state[ni][nj] = 0
                neighbors.append(new_state)
                moves.append(moved_number)
        
        return neighbors, moves
    
    def manhattan_distance(self, state):
        """曼哈顿距离启发式函数"""
        distance = 0
        for i in range(4):
            for j in range(4):
                value = state[i][j]
                if value != 0:
                    goal_i, goal_j = self.goal_positions[value]
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return distance
    
    def linear_conflict(self, state):
        """线性冲突启发式函数（曼哈顿距离 + 线性冲突）"""
        distance = self.manhattan_distance(state)
        conflict = 0
        
        # 检查行中的线性冲突
        for i in range(4):
            # 获取当前行中的所有数字及其目标位置
            row_tiles = []
            for j in range(4):
                value = state[i][j]
                if value != 0:
                    goal_i, goal_j = self.goal_positions[value]
                    if goal_i == i:  # 目标位置在同一行
                        row_tiles.append((goal_j, value))
            
            # 检查是否有两个数字在同一行但顺序颠倒
            row_tiles.sort()
            for idx1 in range(len(row_tiles)):
                for idx2 in range(idx1 + 1, len(row_tiles)):
                    if row_tiles[idx1][0] > row_tiles[idx2][0]:
                        conflict += 2
        
        # 检查列中的线性冲突
        for j in range(4):
            col_tiles = []
            for i in range(4):
                value = state[i][j]
                if value != 0:
                    goal_i, goal_j = self.goal_positions[value]
                    if goal_j == j:  # 目标位置在同一列
                        col_tiles.append((goal_i, value))
            
            col_tiles.sort()
            for idx1 in range(len(col_tiles)):
                for idx2 in range(idx1 + 1, len(col_tiles)):
                    if col_tiles[idx1][0] > col_tiles[idx2][0]:
                        conflict += 2
        
        return distance + conflict
    
    def is_goal(self, state):
        """检查是否达到目标状态"""
        return state == self.goal_state
    
    def state_to_tuple(self, state):
        """将状态转换为元组"""
        return tuple(tuple(row) for row in state)

def A_star(puzzle, heuristic_func="manhattan"):
    """
    A*算法解决15-Puzzle问题
    
    Args:
        puzzle: 输入的15-Puzzle状态，二维嵌套list
        heuristic_func: 启发式函数，可选 "manhattan", "linear_conflict"
    
    Returns:
        list: 移动数字方块的次序
    """
    solver = PuzzleSolver()
    
    # 选择启发式函数
    if heuristic_func == "linear_conflict":
        heuristic = solver.linear_conflict
    else:  # 默认使用曼哈顿距离
        heuristic = solver.manhattan_distance
    
    # 检查是否有解
    if not is_solvable(puzzle):
        print("警告：此puzzle无解！")
        return []
    
    # 优先队列：(f_score, depth, state_tuple, path)
    start_state = deepcopy(puzzle)
    start_tuple = solver.state_to_tuple(start_state)
    h_score = heuristic(start_state)
    
    print(f"初始启发值: {h_score}")
    
    # 使用优先队列
    counter = 0
    open_set = [(h_score, 0, counter, start_tuple, [])]  # (f, g, counter, state, path)
    closed_set = {}  # state_tuple -> g_score
    
    nodes_expanded = 0
    start_time = time.time()
    last_print_time = start_time
    
    while open_set:
        f, g, _, current_tuple, path = heapq.heappop(open_set)
        
        # 如果已经探索过且路径更长，跳过
        if current_tuple in closed_set and closed_set[current_tuple] <= g:
            continue
        
        closed_set[current_tuple] = g
        nodes_expanded += 1
        
        # 每扩展一定数量节点后输出进度
        current_time = time.time()
        if current_time - last_print_time > 2:  # 每2秒输出一次进度
            print(f"\r已扩展节点: {nodes_expanded}, 当前路径长度: {g}, f值: {f}, "
                  f"已用时: {current_time - start_time:.1f}秒", end="", flush=True)
            last_print_time = current_time
        
        # 将tuple转回list进行比较和生成邻居
        current_state = [list(row) for row in current_tuple]
        
        # 检查是否达到目标
        if solver.is_goal(current_state):
            print(f"\n完成！总扩展节点: {nodes_expanded}")
            return path
        
        # 生成后继状态
        neighbors, moves = solver.get_neighbors(current_state)
        
        for next_state, moved_number in zip(neighbors, moves):
            next_tuple = solver.state_to_tuple(next_state)
            new_g = g + 1
            
            # 如果已经在closed_set中且新路径不是更短，跳过
            if next_tuple in closed_set and closed_set[next_tuple] <= new_g:
                continue
            
            new_h = heuristic(next_state)
            new_f = new_g + new_h
            
            new_path = path + [moved_number]
            counter += 1
            heapq.heappush(open_set, (new_f, new_g, counter, next_tuple, new_path))
    
    print(f"\n搜索完成但未找到解。总扩展节点: {nodes_expanded}")
    return []  # 无解情况

def IDA_star(puzzle, heuristic_func="manhattan"):
    """
    IDA*算法解决15-Puzzle问题
    
    Args:
        puzzle: 输入的15-Puzzle状态，二维嵌套list
        heuristic_func: 启发式函数，可选 "manhattan", "linear_conflict"
    
    Returns:
        list: 移动数字方块的次序
    """
    solver = PuzzleSolver()
    
    # 选择启发式函数
    if heuristic_func == "linear_conflict":
        heuristic = solver.linear_conflict
    else:  # 默认使用曼哈顿距离
        heuristic = solver.manhattan_distance
    
    # 检查是否有解
    if not is_solvable(puzzle):
        print("警告：此puzzle无解！")
        return []
    
    start_state = deepcopy(puzzle)
    start_tuple = solver.state_to_tuple(start_state)
    bound = heuristic(start_state)
    
    print(f"初始启发值: {bound}")
    
    nodes_expanded = 0
    start_time = time.time()
    last_print_time = start_time
    iteration_count = 0
    
    def search(state_tuple, g, bound, path):
        nonlocal nodes_expanded, last_print_time
        
        current_state = [list(row) for row in state_tuple]
        f = g + heuristic(current_state)
        
        if f > bound:
            return f, None
        
        if solver.is_goal(current_state):
            return 0, path
        
        min_cost = float('inf')
        neighbors, moves = solver.get_neighbors(current_state)
        
        # 获取父状态以避免回溯
        parent_tuple = None
        if path:
            parent_move_num = path[-1]
            # 通过反向移动找到父状态
            blank_i, blank_j = solver.find_blank(current_state)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = blank_i + di, blank_j + dj
                if 0 <= ni < 4 and 0 <= nj < 4:
                    if current_state[ni][nj] == parent_move_num:
                        parent_state = deepcopy(current_state)
                        parent_state[blank_i][blank_j], parent_state[ni][nj] = parent_state[ni][nj], parent_state[blank_i][blank_j]
                        parent_tuple = solver.state_to_tuple(parent_state)
                        break
        
        for next_state, moved_number in zip(neighbors, moves):
            next_tuple = solver.state_to_tuple(next_state)
            
            # 避免回到父状态
            if parent_tuple and next_tuple == parent_tuple:
                continue
            
            nodes_expanded += 1
            
            # 输出进度
            current_time = time.time()
            if current_time - last_print_time > 2:  # 每2秒输出一次进度
                print(f"\r迭代深度: {bound}, 已扩展节点: {nodes_expanded}, "
                      f"已用时: {current_time - start_time:.1f}秒", 
                      end="", flush=True)
                last_print_time = current_time
            
            new_path = path + [moved_number]
            t, result = search(next_tuple, g + 1, bound, new_path)
            
            if result is not None:
                return 0, result
            
            min_cost = min(min_cost, t)
        
        return min_cost, None
    
    print(f"开始IDA*搜索")
    
    while True:
        iteration_count += 1
        print(f"\n第{iteration_count}次迭代，当前阈值: {bound}")
        
        t, result = search(start_tuple, 0, bound, [])
        
        if result is not None:
            end_time = time.time()
            print(f"\n完成！总扩展节点: {nodes_expanded}, 总用时: {end_time - start_time:.1f}秒")
            return result
        
        if t == float('inf'):
            print("无解")
            return []
        
        bound = t

def is_solvable(puzzle):
    """检查15-puzzle是否有解"""
    flat = []
    blank_row = 0
    for i in range(4):
        for j in range(4):
            if puzzle[i][j] == 0:
                blank_row = i
            else:
                flat.append(puzzle[i][j])
    
    # 计算逆序数
    inversions = 0
    for i in range(len(flat)):
        for j in range(i+1, len(flat)):
            if flat[i] > flat[j]:
                inversions += 1
    
    # 对于4x4的puzzle，空格在偶数行(从0开始)时逆序数应为奇数
    # 空格在奇数行时逆序数应为偶数
    return (inversions + blank_row) % 2 == 1

# 测试代码
if __name__ == "__main__":
    # 较难的测试用例
    puzzle = [[0, 5, 15, 14],
              [7, 9, 6, 13],
              [1, 2, 12, 10],
              [8, 11, 4, 3]]

    print("测试15-Puzzle求解器")
    print("初始状态:")
    for row in puzzle:
        print(row)
    
    # 检查是否有解
    if not is_solvable(puzzle):
        print("\n警告：此puzzle无解！")
    else:
        print("\n此puzzle有解，开始求解...")
        
        # 先计算曼哈顿距离看看
        solver = PuzzleSolver()
        manhattan_dist = solver.manhattan_distance(puzzle)
        print(f"初始曼哈顿距离: {manhattan_dist}")
        
        # 对于困难问题，推荐使用IDA*算法
        print("\n=== IDA*算法（曼哈顿距离）===")
        start_time = time.time()
        solution = IDA_star(puzzle, heuristic_func="manhattan")
        end_time = time.time()
        
        if solution:
            print(f"\n解: {solution}")
            print(f"步数: {len(solution)}")
            print(f"总用时: {end_time - start_time:.2f}秒")
        else:
            print("未找到解")