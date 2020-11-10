#-
from time import time
from beacon import Beacon
import pandas as pd
pd.set_option("display.max_rows", 300)

#-
import matplotlib.pyplot as plt
import networkx as nx
from beacon.bert.semantic_relations import compile_DG

#- Main
if __name__ == "__main__":

    #- Load Text Example
    with open("examples/example_3.txt", "r") as f:
        text = f.read()

    #- Instantiate Beacon
    beacon = Beacon()

    #- Run beacon on text
    t = time()
    result = beacon(text)
    elapsed = time() - t

    #- Print Annotations
    print(result.drop(columns=["snippet_text"]))
    print(result["snippet_text"].unique())
    print("Computed in (sec): {}".format(elapsed))

    #- Draw Directed Dependency Graph
    G = compile_DG(result)
    nx.draw(G,
            labels=nx.get_node_attributes(G, 'concept'),
            font_weight='bold',
            font_size=8,
            arrow_size=15,
            node_size=8000,
            pos=nx.shell_layout(G,nlist=[[3],[i for i in range(4,16)]]))
    plt.show()
