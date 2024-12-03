import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import itertools
from typing import Tuple, Dict
from community import community_louvain

def visualize_edge_graph(G, partition):
    plt.figure(figsize=(20, 12))

    pos = nx.spring_layout(G)
    cmap = plt.get_cmap('viridis')

    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=300,
                           cmap=cmap, node_color=list(partition.values()))

    edges = G.edges(data=True)
    weights = [edge[2]['weight'] * 3 for edge in edges]

    nx.draw_networkx_edges(G, pos, alpha=0.7, width=weights, edge_color=weights, edge_cmap=plt.cm.Blues)

    nx.draw_networkx_labels(G, pos, font_size=10)

    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues), label='Edge Weight')
    plt.savefig('./fig/visualize_edge_graph_2d.png', format='png', dpi=300)

    plt.show()
    plt.close()

def visualize_edge_graph_3d(G, partition):
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111, projection='3d')

    # k 값으로 노드 간 기본 거리를 조정 (값을 높일수록 노드가 멀어짐)
    pos = nx.spring_layout(G, dim=3, seed=42, k=2)  # k 값 조정으로 노드 간 거리 조절 (기본값은 1/√n)

    x_vals = np.array([pos[node][0] for node in G.nodes()])
    y_vals = np.array([pos[node][1] for node in G.nodes()])
    z_vals = np.array([pos[node][2] for node in G.nodes()])

    ax.scatter(x_vals, y_vals, z_vals, c=list(partition.values()), cmap=plt.get_cmap('viridis'), s=60)

    for edge in G.edges(data=True):
        x_line = np.array([pos[edge[0]][0], pos[edge[1]][0]])
        y_line = np.array([pos[edge[0]][1], pos[edge[1]][1]])
        z_line = np.array([pos[edge[0]][2], pos[edge[1]][2]])

        ax.plot(x_line, y_line, z_line, color='b', alpha=0.3, lw=edge[2]['weight'] * 1.2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig('./fig/visualize_edge_graph_3d.png', format='png', dpi=300)

    plt.show()
    plt.close()

# correlation matrix를 사용하여 유사한 개체들을 군집화
def calculate_louvain(data:pd.DataFrame, threshold:float=0.3) -> Tuple[Dict[str,float],nx.Graph]:
    return_df = data.copy()
    corr_matrix = return_df.corr()
    G = nx.Graph()

    # itertools.combinations를 사용해서 (i, j) 조합 생성
    for i, j in itertools.combinations(range(len(corr_matrix)), 2):
        weight = abs(corr_matrix.iloc[i, j])
        if weight > threshold:
            G.add_edge(corr_matrix.index[i], corr_matrix.columns[j], weight=weight)

    # Louvain 알고리즘을 사용하여 Modularity Clustering
    partition = community_louvain.best_partition(G)

    return partition, G