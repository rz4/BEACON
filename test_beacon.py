#-
from time import time
from beacon import Beacon

#- Main
if __name__ == "__main__":

    #- Load Text Example
    with open("examples/example_1.txt", "r") as f:
        text = f.read()

    #- Instantiate Beacon
    beacon = Beacon()

    #- Run beacon on text
    t = time()
    result = beacon(text, bert_depth=3)
    elapsed = time() - t

    #- Print
    print(result)
    print(result["snippet_text"].unique())
    print("Computed in (sec): {}".format(elapsed))
