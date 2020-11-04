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
    result =
    elapsed = time() - t

    #- Print result
    df = [[chunk.root.dep_.upper(), chunk.start_char, chunk.end_char, chunk.text] for chunk in nlp(text).noun_chunks]
    df = pd.DataFrame(df, columns=["text", "lex", "startx", "endx"])
    print(df)
    print("Computed in (sec): {}".format(elapsed))
    #displacy.serve(result, style="dep")
