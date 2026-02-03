import networkx as nx
import itertools
from collections import defaultdict, deque
import multiprocessing as mp
from functools import partial

# 确保安装networkx：pip install networkx

# ------------------------------
# 工具函数：阶乘、双阶乘
# ------------------------------
_fact_cache = {0: 1, 1: 1}


def factorial(n):
    if n not in _fact_cache:
        _fact_cache[n] = n * factorial(n - 1)
    return _fact_cache[n]


_double_fact_cache = {0: 1, 1: 1}


def double_factorial(n):
    if n < 0:
        return 0
    if n not in _double_fact_cache:
        _double_fact_cache[n] = n * double_factorial(n - 2) if n >= 2 else 1
    return _double_fact_cache[n]


# ------------------------------
# 并行生成内点矩阵（子进程任务）
# ------------------------------
def generate_inner_chunk(args, V, n):
    start_i, start_j = args
    inner_stack = []
    initial_inner = [[0] * V for _ in range(V)]
    inner_stack.append((start_i, start_j, initial_inner))
    chunk_results = []

    while inner_stack:
        i, j, mat = inner_stack.pop()

        if i == V:
            valid = True
            for row in mat:
                if sum(row) > n:
                    valid = False
                    break
            if valid:
                chunk_results.append([r.copy() for r in mat])
            continue

        next_i, next_j = (i, j + 1) if (j + 1) < V else (i + 1, i + 1)

        used = 0
        for k in range(V):
            if k < i:
                used += mat[k][i]
            elif k == i:
                used += mat[i][k]
        remaining = n - used

        if i == j:
            max_val = remaining if remaining % 2 == 0 else remaining - 1
            max_val = max(0, max_val)
            for val in range(max_val, -1, -2):
                new_mat = [r.copy() for r in mat]
                new_mat[i][j] = val
                inner_stack.append((next_i, next_j, new_mat))
        else:
            max_val = remaining
            for val in range(max_val, -1, -1):
                new_mat = [r.copy() for r in mat]
                new_mat[i][j] = val
                new_mat[j][i] = val
                inner_stack.append((next_i, next_j, new_mat))

    return chunk_results


# ------------------------------
# 并行生成内点矩阵（主入口）
# ------------------------------
def generate_inner_matrices_parallel(V, n):
    num_processes = mp.cpu_count()
    # 拆分初始任务（按对角线起始位置拆分）
    initial_tasks = []
    step = max(1, V // num_processes)  # 每个进程处理的起始i范围
    for i in range(0, V, step):
        initial_tasks.append((i, i))  # 从(i,i)位置开始生成

    with mp.Pool(processes=num_processes) as pool:
        func = partial(generate_inner_chunk, V=V, n=n)
        results = pool.map(func, initial_tasks)

    # 合并结果并去重（避免不同进程生成相同矩阵）
    inner_matrices = []
    seen = set()
    for chunk in results:
        for mat in chunk:
            mat_tuple = tuple(tuple(row) for row in mat)
            if mat_tuple not in seen:
                seen.add(mat_tuple)
                inner_matrices.append(mat)
    return inner_matrices


# ------------------------------
# 并行生成外点连接矩阵（子进程任务）
# ------------------------------
def generate_outer_chunk(inner_mat, V, E, n):
    size = V + E
    inner_used = [sum(row) for row in inner_mat]
    full_mat = [[0] * size for _ in range(size)]
    for i in range(V):
        for j in range(V):
            full_mat[i][j] = inner_mat[i][j]

    outer_stack = []
    outer_stack.append((0, 0, full_mat, inner_used))
    chunk_results = []

    while outer_stack:
        inner_idx, outer_idx, mat, inner_used = outer_stack.pop()

        if inner_idx == V:
            valid = True
            for e in range(E):
                col = V + e
                if sum(mat[i][col] for i in range(V)) != 1:
                    valid = False
                    break
            if valid:
                chunk_results.append([r.copy() for r in mat])
            continue

        remaining = n - inner_used[inner_idx]
        if remaining < 0:
            continue

        used_outer = sum(mat[inner_idx][V:])
        remaining_outer = remaining - used_outer
        if remaining_outer < 0:
            continue

        if outer_idx >= E:
            if remaining_outer == 0:
                outer_stack.append((inner_idx + 1, 0, mat, inner_used))
            continue

        col = V + outer_idx
        for val in range(remaining_outer, -1, -1):
            new_mat = [r.copy() for r in mat]
            new_mat[inner_idx][col] = val
            new_mat[col][inner_idx] = val
            outer_stack.append((inner_idx, outer_idx + 1, new_mat, inner_used))

    return chunk_results


# ------------------------------
# 迭代生成邻接矩阵（并行优化版）
# ------------------------------
def generate_adjacency_matrices(V, E, n=3):
    print("Now generating adjacency matrix (parallel mode)")

    # 1. 并行生成内点矩阵
    inner_matrices = generate_inner_matrices_parallel(V, n)
    print(f"Generated {len(inner_matrices)} unique inner matrices")

    # 2. 并行生成外点连接矩阵
    num_processes = mp.cpu_count()
    with mp.Pool(processes=num_processes) as pool:
        func = partial(generate_outer_chunk, V=V, E=E, n=n)
        results = pool.map(func, inner_matrices)

    # 合并所有结果
    matrices = []
    for chunk in results:
        matrices.extend(chunk)

    return matrices


# ------------------------------
# 以下函数保持不变
# ------------------------------
def matrix_to_graph(mat, V, E):
    G = nx.MultiGraph()  # 支持多重边
    # 添加内点（类型标记为internal）
    for i in range(V):
        G.add_node(i, type="internal")
    # 添加外点（类型标记为external，编号V~V+E-1）
    for e in range(E):
        node_id = V + e
        G.add_node(node_id, type="external")
    # 添加边（带权重，即连接数）
    size = V + E
    for i in range(size):
        for j in range(i, size):
            weight = mat[i][j]
            if weight > 0:
                # 多重边通过添加weight次实现
                for _ in range(weight):
                    G.add_edge(i, j)
    return G


def are_isomorphic(G1, G2):
    """判断两个图是否同构，考虑节点类型（内点/外点）"""

    def node_match(n1, n2):
        return n1["type"] == n2["type"]

    return nx.is_isomorphic(G1, G2, node_match=node_match)


def generate_feynman_diagrams(V, E, n=3):
    count1, count2 = 0, 0
    if (n * V + E) % 2 != 0:
        return []
    if V < 0 or E < 0:
        return []

    raw_matrices = generate_adjacency_matrices(V, E, n)
    print(f"Generate raw matrix {len(raw_matrices)} ")

    unique_graphs = []  # 存储唯一的图对象
    unique_matrices = []  # 存储对应的邻接矩阵

    for mat in raw_matrices:
        count2 += 1
        # 检查连通性（先通过矩阵快速判断）
        size = V + E
        G = matrix_to_graph(mat, V, E)
        if not nx.is_connected(G):
            continue

        # 检查是否与已有的图同构
        is_unique = True
        for existing_G in unique_graphs:
            if are_isomorphic(G, existing_G):
                is_unique = False
                break
        if is_unique:
            unique_graphs.append(G)
            unique_matrices.append(mat)
            count1 += 1
            print("diagram", count1, "found in Iteration", count2)

    print("Iteration finished. Now solving...")
    # 计算对称因子并整理结果
    result = []
    for mat in unique_matrices:
        size = V + E
        # 内点角色分组
        inner_roles = []
        for i in range(V):
            row = mat[i]
            inner_roles.append(tuple(row))
        inner_groups = defaultdict(int)
        for r in inner_roles:
            inner_groups[r] += 1
        inner_aut = 1
        for cnt in inner_groups.values():
            inner_aut *= factorial(cnt)

        outer_roles = []
        for e in range(E):
            row = mat[V + e]
            outer_roles.append(tuple(row))
        outer_groups = defaultdict(int)
        for r in outer_roles:
            outer_groups[r] += 1
        outer_aut = 1
        for cnt in outer_groups.values():
            outer_aut *= factorial(cnt)

        non_diag_factor = 1
        for i in range(size):
            for j in range(i + 1, size):
                non_diag_factor *= factorial(mat[i][j])

        diag_factor = 1
        for i in range(V):
            diag_factor *= double_factorial(mat[i][i])

        sym_factor = inner_aut * outer_aut * non_diag_factor * diag_factor

        result.append({
            "matrix": mat,
            "symmetry_factor": sym_factor
        })

    return result


def save_results(diagrams, V, E, n=3, filename="FeynmanDiagram.txt"):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"费曼图（{V}个内点，{E}个外点）\n")
        f.write(f"拓扑不等价图总数：{len(diagrams)}\n")
        f.write("========================================\n\n")
        for idx, diag in enumerate(diagrams, 1):
            f.write(f"图 {idx}：\n")
            f.write("  邻接矩阵：\n")
            for row in diag["matrix"]:
                f.write(f"    {row}\n")
            f.write(f"  对称因子：{diag['symmetry_factor']}\n\n")


def main():
    V = int(input("Internal Points:"))
    E = int(input("External Points:"))
    n = int(input("Phi level:"))
    diagrams = generate_feynman_diagrams(V, E, n)
    save_filename = f"phi{n}FeynmanDiagramInOrder{V}External{E}.txt"
    save_results(diagrams, V, E, n, filename=save_filename)
    print(f"Get {len(diagrams)} topological inequivalent Diagram. Result saved in {save_filename}")


if __name__ == "__main__":
    main()
