#-
from time import time
from beacon import Beacon
import pandas as pd

#-
import matplotlib.pyplot as plt
import networkx as nx
from beacon.bert.semantic_relations import compile_DG, compile_prolog
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
                    bert_layers=[4, 8])#, # Number of Attention Layers to use; Shorter text typically need less layers

    #- Compile Directed Dependency Graph
    G, pt = compile_DG(result)

    #- Compile Prolog Facts
    script = compile_prolog(result, G, pt)

    #- Display Results
    elapsed = time() - t
    print("Computed in (sec): {}".format(elapsed))
    print(result.drop(columns=["snippet_text"]))
    print(result["snippet_text"].unique())
    print(script)
    display_KG(G)
