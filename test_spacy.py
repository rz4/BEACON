#-
from time import time
import spacy
from spacy import displacy
import pandas as pd

#- Main
if __name__ == "__main__":

    #- Load Text Example
    with open("examples/example_1.txt", "r") as f:
        text = f.read()

    #- Instantiate Spacy
    nlp = spacy.load("en_core_web_sm")

    #- Run beacon on text
    t = time()
    result = nlp(text)
    elapsed = time() - t

    #- Print result
    df = [[token.text, token.dep_, token.head.text, token.head.pos_, ",".join([child.text.strip() for child in token.children])] for token in result]
    df = pd.DataFrame(df, columns=["TEXT", "DEP", "HEAD_TEXT", "HEAD_POS", "CHILDREN"])
    print(df)
    print("Computed in (sec): {}".format(elapsed))
    #displacy.serve(result, style="dep")
