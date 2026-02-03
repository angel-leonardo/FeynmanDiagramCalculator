import networkx as nx
import matplotlib.pyplot as plt
import re
import math
from collections import defaultdict


# ------------------------------
# 解析费曼图结果文件（增加度数校验）
# ------------------------------
def parse_feynman_file(filename):
    diagrams = []
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取内点数V、外点数E和理论阶数n（从文件头匹配，如φ³理论n=3）
    v_match = re.search(r'(\d+)个内点', content)
    e_match = re.search(r'(\d+)个外点', content)
    n_match = re.search(r'φ(\d+)理论', content)
    V = int(v_match.group(1)) if v_match else 4
    E = int(e_match.group(1)) if e_match else 2
    n = int(n_match.group(1)) if n_match else 3  # 默认φ³理论

    # 提取每张图的邻接矩阵
    diagram_blocks = re.split(r'图 \d+：', content)[1:]
    for block in diagram_blocks:
        mat_match = re.search(r'邻接矩阵：\n((?:    \[.+\]\n)+)', block)
        if not mat_match:
            continue
        mat_lines = mat_match.group(1).strip().split('\n')
        adj_matrix = []
        for line in mat_lines:
            line = line.strip().strip('[]')
            row = list(map(int, line.split(', ')))
            adj_matrix.append(row)

        # 度数校验：内点必须为n条线，外点必须为1条线
        valid = True
        size = V + E
        for i in range(size):
            degree = sum(adj_matrix[i])
            if i < V and degree != n:  # 内点
                valid = False
                print(f"警告：图矩阵内点{i}度数为{degree}（应为{n}），已跳过")
                break
            if i >= V and degree != 1:  # 外点
                valid = False
                print(f"警告：图矩阵外点{i - V}度数为{degree}（应为1），已跳过")
                break
        if valid:
            diagrams.append({
                "V": V,
                "E": E,
                "n": n,
                "matrix": adj_matrix
            })
    return diagrams


# ------------------------------
# 绘制费曼图（严格控制顶点线条数）
# ------------------------------
def draw_feynman_diagram(adj_matrix, V, E, n, save_path):
    """确保每个内点显示n条线，外点显示1条线"""
    G = nx.MultiGraph()

    # 添加节点（内点：0~V-1，外点：V~V+E-1）
    for i in range(V + E):
        node_type = "internal" if i < V else "external"
        G.add_node(i, type=node_type)

    # 添加边（记录每条边的索引，用于区分颜色）
    edge_list = []  # 存储所有边(i,j,索引)
    edge_index = 0
    size = V + E
    for i in range(size):
        for j in range(i, size):
            count = adj_matrix[i][j]
            for _ in range(count):
                G.add_edge(i, j, index=edge_index)
                edge_list.append((i, j, edge_index))
                edge_index += 1

    # 布局：内点居中，外点圆周分布
    inner_pos = nx.spring_layout(range(V), seed=42, k=0.6)  # 内点间距放大
    ext_pos = {}
    radius = 1.2  # 外点距离中心更远，避免线条交叉
    for e in range(E):
        angle = 2 * math.pi * e / E
        ext_pos[V + e] = (radius * math.cos(angle), radius * math.sin(angle))
    pos = {**inner_pos, **ext_pos}

    # 为每个顶点的边分配不同角度，确保线条不重叠
    plt.figure(figsize=(10, 10))

    # 1. 绘制节点
    # 内点（蓝色圆形，更大尺寸突出顶点）
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=range(V),
        node_color='lightblue',
        node_shape='o',
        node_size=1000,  # 增大内点尺寸，便于区分线条
        edgecolors='black',
        linewidths=2
    )
    # 外点（红色方形）
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=range(V, V + E),
        node_color='lightcoral',
        node_shape='s',
        node_size=800,
        edgecolors='black',
        linewidths=2
    )

    # 2. 绘制边（按顶点分配角度，确保每条线清晰可见）
    edge_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 区分线条的颜色
    for i, j, idx in edge_list:
        # 计算边的弯曲角度（根据顶点度数动态调整）
        # 内点i的度数（用于分配角度）
        degree_i = sum(adj_matrix[i])
        # 内点j的度数
        degree_j = sum(adj_matrix[j]) if j < V else 1

        # 为i→j的边分配弯曲角度（避免重叠）
        rad = 0.2  # 基础弯曲半径
        if i < V and j < V:  # 内点-内点边
            # 两个内点之间的边，根据索引微调角度
            rad += 0.1 * (idx % 3)
        elif i < V:  # 内点-外点边（从内点出发）
            rad = 0.1
        elif j < V:  # 外点-内点边（到内点）
            rad = 0.1

        # 绘制单条边（使用不同颜色区分）
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(i, j)],
            width=2.5,  # 线条加粗，便于计数
            edge_color=edge_colors[idx % len(edge_colors)],
            alpha=0.9,
            connectionstyle=f'arc3,rad={rad}'
        )

    # 3. 绘制自环（内点专属，1个自环=2条线，单独样式）
    for i in range(V):
        self_loop_count = adj_matrix[i][i]
        if self_loop_count > 0:
            x, y = pos[i]
            loop_radius = 0.3  # 自环半径（确保能看出是2条线）
            for k in range(self_loop_count):
                # 多个自环时上下分布
                offset = 0.1 * k  # 垂直偏移避免重叠
                # 上半环（表示1条线）
                plt.plot(
                    [x, x + loop_radius, x],
                    [y, y + loop_radius + offset, y],
                    color='black', linewidth=2.5, alpha=0.8
                )
                # 下半环（表示另1条线，凑齐2条）
                plt.plot(
                    [x, x + loop_radius, x],
                    [y, y - loop_radius - offset, y],
                    color='black', linewidth=2.5, alpha=0.8
                )

    # 4. 节点标签（标注度数，验证线条数）
    node_labels = {}
    for i in range(V):
        # 内点标签："I0(3)" 表示内点0，度数3
        node_labels[i] = f"I{i}({n})"
    for e in range(E):
        # 外点标签："E0(1)" 表示外点0，度数1
        node_labels[V + e] = f"E{e}(1)"
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=12,
        font_weight='bold'
    )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()


# ------------------------------
# 批量转化主函数
# ------------------------------
def batch_convert_to_images(input_file, output_dir="./feynman_images/"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    diagrams = parse_feynman_file(input_file)
    if not diagrams:
        print("未解析到有效费曼图数据！")
        return
    print(f"共解析到 {len(diagrams)} 张有效费曼图，开始生成图片...")

    for idx, diag in enumerate(diagrams, 1):
        V = diag["V"]
        E = diag["E"]
        n = diag["n"]  # 理论阶数（如3）
        adj_matrix = diag["matrix"]
        save_path = os.path.join(output_dir, f"图{idx}.png")
        draw_feynman_diagram(adj_matrix, V, E, n, save_path)
        print(f"已生成：{save_path}")

    print(f"\n所有图片已保存至：{os.path.abspath(output_dir)}")


# ------------------------------
# 运行入口
# ------------------------------
if __name__ == "__main__":
    INPUT_FILE = "phi4FeynmanDiagramInOrder4External2"  # 输入结果文件
    OUTPUT_DIR = "./feynman_images/"  # 输出图片目录
    batch_convert_to_images(INPUT_FILE, OUTPUT_DIR)
