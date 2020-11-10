#-
from time import time
from beacon import Beacon
import pandas as pd
pd.set_option("display.max_rows", 300)

#- Main
if __name__ == "__main__":

    #- Load Text Example
    with open("examples/example_3.txt", "r") as f:
        text = f.read()

    #- Instantiate Beacon
    beacon = Beacon()

    #- Run beacon on text
    t = time()
    result = beacon(text, bert_depth=4)#, bert_threshold=0.55)
    elapsed = time() - t

    #- Print
    print(result.drop(columns=["snippet_text"]))
    print(result["snippet_text"].unique())
    print("Computed in (sec): {}".format(elapsed))

    #-
    import matplotlib.pyplot as plt
    import networkx as nx
    G = beacon.entity_graph(result[result["snippet_index"]==0])
    nx.draw(G, labels=nx.get_node_attributes(G, 'concept'), font_weight='bold', font_size=8, arrow_size=15, node_size=8000, pos=nx.shell_layout(G,nlist=[[3],[i for i in range(3,16) if i not in [3,4, 10]]]))
    plt.show()
