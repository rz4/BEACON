#-
import matplotlib.pyplot as plt
import networkx as nx

cmap = plt.cm.gist_rainbow

#--
def display_KG(G):
    #-
    node_colors = []
    node_sizes = []
    node_attri = {}
    pos_attrs = {}
    pos_nodes = nx.kamada_kawai_layout(G)
    for ndx, coords in pos_nodes.items():
        pos_attrs[ndx] = (coords[0] + 0.01, coords[1] + 0.01)
        node = G.nodes[ndx]
        node_colors.append(cmap(node["core"]/10.0))
        node_sizes.append(node["pagerank"]*5000)
        node_attri[ndx] = "{}:{}:{}:{}".format(node["index"],node["core"],node["lex"],node["text"])

    #-
    nx.draw(G, pos_nodes, arrow_size=15, node_size=node_sizes, node_color=node_colors)
    nx.draw_networkx_labels(G, pos_attrs, labels=node_attri, font_weight='bold', font_size=12)
    plt.show()
