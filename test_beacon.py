#- Imports
import os, hy, json
from time import time
from beacon import Beacon

#-
if __name__ == "__main__":

  #- Load Beacon With Pretrained Bert Model
  t = time()
  model = Beacon(from_pretrained="bert-base-uncased", layers=[4,5,6])
  with open("examples/example_1.txt", "r") as f: text = f.read()
  print("Loading Time(sec): {}".format(time()-t))
  print("Text Sample:\n{}\n".format(text))

  #- Produce Text Abstract Syntax Tree (AST) From Bert Attention
  t = time()
  AST = model(text)
  print("Elapsed Time(sec): {}".format(time()-t))
  print("Bert Abstract Syntax Tree:\n{}\n".format(AST))

  #- Run Logical Query Using Prolog
  t = time()
  results = AST.query({"test1": "typed_dependents(A, B, TOKENA, TOKENB, self, homeless)",
                       "test2": "typed_dependents(A, B, TOKENA, TOKENB, negex, homeless)",
                       "test3": "typed_dependents(A, B, TOKENA, TOKENB, fam, homeless)",
                       "test4": ",".join(["typed_dependents(A, B, TOKENA, TOKENB, self, homeless)",
                                          "not(typed_dependents(_, B, _, TOKENB, negex, homeless))",
                                          "not(typed_dependents(_, B, _, TOKENB, fam, homeless))"])})
  print("Elapsed Time(sec): {}".format(time()-t))
  print("Query Results:\n{}".format(json.dumps(results, indent=4)))
