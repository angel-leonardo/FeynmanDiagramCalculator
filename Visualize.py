import networkx as nx
import matplotlib.pyplot as plt
import re
import math
import os
from collections import defaultdict


# ------------------------------
# 解析费曼图结果文件（自动识别理论阶数）
# ------------------------------
def parse_feynman_file(filename):
    diagrams = []
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 提取关键参数：内点数V、外点数E、理论阶数n（核心修复）
    v_match = re.search(r'(\d+)个内点', content)
    e_match = re.search(r'(\d+)个外点', content)
    n_match = re.search(r'φ(\d+)理论', content)  # 匹配φ3、φ4等

    V = int(v_match.group(1)) if v_match else 4
    E = int(e_match.group(1)) if e_match else 2
    n = int(n_match.group(1)) if n_match else 3  # 未指定时默认φ³，但会动态适配

    # 2. 提取每张图的邻接矩阵
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

        # 3. 动态校验内点度数（核心修复：不依赖默认n，而是计算实际度数）
        size = V + E
        # 计算内点实际度数（所有内点度数应相同，等于理论阶数）
        inner_degrees = [sum(adj_matrix[i]) for i in range(V)]
        if len(set(inner_degrees)) != 1:  # 内点度数不一致，无效图
            print(f"警告：图矩阵内点度数不一致（{inner_degrees}），已跳过")
            continue
        actual_n = inner_degrees[0]  # 实际理论阶数（如φ⁴为4）

        # 校验外点度数（必须为1）
        outer_valid = all(sum(adj_matrix[i]) == 1 for i in range(V, size))
        if not outer_valid:
            print(f"警告：图矩阵外点度数不为1，已跳过")
            continue

        # 4. 保存有效图（使用实际阶数actual_n）
        diagrams.append({
            "V": V,
            "E": E,
            "n": actual_n,  # 动态获取理论阶数
            "matrix": adj_matrix
        })
    return diagrams


# ------------------------------
# 绘制费曼图（适配任意阶数n）
# ------------------------------
def draw_feynman_diagram(adj_matrix, V, E, n, save_path):
    """自动适配内点度数n（如φ⁴的n=4）"""
    G = nx.MultiGraph()

    # 添加节点
    for i in range(V + E):
        node_type = "internal" if i < V else "external"
        G.add_node(i, type=node_type)

    # 添加边（记录每条边索引）
    edge_list = []
    edge_index = 0
    size = V + E
    for i in range(size):
        for j in range(i, size):
            count = adj_matrix[i][j]
            for _ in range(count):
                G.add_edge(i, j, index=edge_index)
                edge_list.append((i, j, edge_index))
                edge_index += 1

    # 布局优化
    inner_pos = nx.spring_layout(range(V), seed=42, k=0.6 + 0.1 * n)  # 阶数越高，内点间距越大
    ext_pos = {}
    radius = 1.2 + 0.2 * n  # 外点距离随阶数增加
    for e in range(E):
        angle = 2 * math.pi * e / E
        ext_pos[V + e] = (radius * math.cos(angle), radius * math.sin(angle))
    pos = {**inner_pos, **ext_pos}

    plt.figure(figsize=(10 + n, 10 + n))  # 画布随阶数增大

    # 绘制节点
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=range(V),
        node_color='lightblue',
        node_shape='o',
        node_size=1000 + 200 * n,  # 内点大小随阶数增加
        edgecolors='black',
        linewidths=2
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=range(V, V + E),
        node_color='lightcoral',
        node_shape='s',
        node_size=800,
        edgecolors='black',
        linewidths=2
    )

    # 绘制边（多颜色区分，避免重叠）
    edge_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    for i, j, idx in edge_list:
        # 动态调整弯曲半径，避免线条拥挤
        rad = 0.2 + 0.05 * idx
        if i < V and j < V:  # 内点-内点边
            rad += 0.1 * (idx % (n // 2))  # 阶数越高，弯曲区分越明显
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(i, j)],
            width=2.5,
            edge_color=edge_colors[idx % len(edge_colors)],
            alpha=0.9,
            connectionstyle=f'arc3,rad={rad}'
        )

    # 绘制自环（1个自环=2条线，适配n=4等高阶）
    for i in range(V):
        self_loop_count = adj_matrix[i][i]
        if self_loop_count > 0:
            x, y = pos[i]
            loop_radius = 0.0  # 高阶理论自环更大
            for k in range(self_loop_count):
                offset = 0.0  # 偏移量增大，避免重叠
                # 上半环
                plt.plot(
                    [x, x + loop_radius, x],
                    [y, y + loop_radius + offset, y],
                    color='black', linewidth=2.5, alpha=0.8
                )
                # 下半环
                plt.plot(
                    [x, x + loop_radius, x],
                    [y, y - loop_radius - offset, y],
                    color='black', linewidth=2.5, alpha=0.8
                )

    # 节点标签（显示实际阶数）
    node_labels = {i: f"I{i}({n})" for i in range(V)}
    node_labels.update({V + e: f"E{e}(1)" for e in range(E)})
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
    os.makedirs(output_dir, exist_ok=True)

    diagrams = parse_feynman_file(input_file)
    if not diagrams:
        print("未解析到有效费曼图数据！")
        return
    print(f"共解析到 {len(diagrams)} 张有效费曼图（理论阶数n={diagrams[0]['n']}），开始生成图片...")

    for idx, diag in enumerate(diagrams, 1):
        V = diag["V"]
        E = diag["E"]
        n = diag["n"]  # 动态获取的理论阶数（如4）
        adj_matrix = diag["matrix"]
        save_path = os.path.join(output_dir, f"图{idx}.png")
        draw_feynman_diagram(adj_matrix, V, E, n, save_path)
        print(f"已生成：{save_path}")

    print(f"\n所有图片已保存至：{os.path.abspath(output_dir)}")


# ------------------------------
# 运行入口
# ------------------------------
if __name__ == "__main__":
    INPUT_FILE = input("File name:")  # 输入φ⁴理论的结果文件
    OUTPUT_DIR = "./feynman_images_converted/"
    batch_convert_to_images(INPUT_FILE, OUTPUT_DIR)
