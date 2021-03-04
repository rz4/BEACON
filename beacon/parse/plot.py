#-
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#--
def export_KG(G, filename):

    #-
    cores = nx.onion_layers(G.to_undirected())
    ranks = nx.pagerank(G)
    nx.set_node_attributes(G, cores, "core")
    nx.set_node_attributes(G, ranks, "pagerank")

    #-
    node_colors = []
    node_sizes = []
    node_attri = {}
    pos_attrs = {}
    pos_nodes = nx.kamada_kawai_layout(G)
    for ndx, coords in pos_nodes.items():
        pos_attrs[ndx] = (coords[0] + 0.00, coords[1] + 0.0)
        node = G.nodes[ndx]
        node_colors.append(node["core"])
        node_sizes.append(node["pagerank"])
        if "index" in node:
            node_attri[ndx] = "{}:'{}'".format(node["index"],node["token"])
        else: node_attri[ndx] = ""
    node_sizes = np.array(node_sizes)
    node_sizes = (node_sizes - node_sizes.min()) / (node_sizes.max() - node_sizes.min())
    node_sizes *= 150
    node_colors = np.array(node_colors)
    node_colors = (node_colors - node_colors.min()) / (node_colors.max() - node_colors.min())
    node_colors = [plt.cm.rainbow(x) for x in node_colors]

    edges_width = []
    edges = list(G.edges(data=True))
    for i in range(len(edges)):
        if "weight" in edges[i][2]:
            edges_width.append(edges[i][2]["weight"])
        else: edges_width.append(0.1)
    edges_width = np.array(edges_width)
    edges_width = (edges_width - edges_width.min()) / (edges_width.max() - edges_width.min())

    #-
    nx.draw(G, pos_nodes, arrowsize=3, width=edges_width, node_size=node_sizes, node_color=node_colors)
    nx.draw_networkx_labels(G, pos_attrs, labels=node_attri, font_weight='bold', font_size=9)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
