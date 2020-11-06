#-
from time import time
from beacon import Beacon
import pandas as pd
pd.set_option("display.max_rows", 300)

#- Main
if __name__ == "__main__":

    #- Load Text Example
    with open("examples/example_1.txt", "r") as f:
        text = f.read()

    #- Instantiate Beacon
    beacon = Beacon()#bert_model_path="lexicons/cori/pretrained_allmimic.pickle")

    #- Run beacon on text
    t = time()
    result = beacon(text, bert_depth=2)
    elapsed = time() - t

    #- Print
    print(result.drop(columns=["snippet_text"]))
    print(result["snippet_text"].unique())
    print("Computed in (sec): {}".format(elapsed))
