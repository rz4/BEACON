#-
from time import time
from beacon import Beacon
import pandas as pd
pd.set_option("display.max_rows", 300)

#-
import matplotlib.pyplot as plt
import networkx as nx
from beacon.bert.semantic_relations import compile_DG, find_simplest_paths
from beacon.vis import display_KG

#- Main
if __name__ == "__main__":

    #- Load Text Example
    with open("examples/example_1.txt", "r") as f:
        text = f.read()

    #- Instantiate Beacon
    beacon = Beacon()

    #- Run beacon on text
    t = time()
    result = beacon(text,
                    bert_depth=3, # Number of Attention Layers to use; Shorter text typically need less layers
                    bert_threshold=0.6) # Energy threshold for building relations; Higher values means more strict selection.
    elapsed = time() - t

    #- Print Annotations
    print(result.drop(columns=["snippet_text"]))
    print(result["snippet_text"].unique())
    print("Computed in (sec): {}".format(elapsed))

    #- Draw Directed Dependency Graph
    G, root = compile_DG(result[result["snippet_index"]==2])
    display_KG(G)
