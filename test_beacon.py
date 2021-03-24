#- Imports
import os, hy, json
from time import time
from beacon import Beacon

#-
CONCEPTS ="""
%--
fam_homeless(A, B) :-
  istype(A, fam),
  istype(B, homeless),
  undirected(A,B).

%--
negex_homeless(A, B) :-
  istype(A, negex),
  istype(B, homeless),
  undirected(A,B).

%--
self_homeless(A, B) :-
  istype(A, self),
  istype(B, homeless),
  undirected(A,B),
  not(fam_homeless(_,B)),
  not(negex_homeless(_,B)).
"""

QUERYS = {"test1": "self_homeless(A,B), token(A,ATOKEN,ASTARTX,AENDX), token(B, BTOKEN,BSTARTX,BENDX)",
          "test2": "fam_homeless(A,B), token(A,ATOKEN,ASTARTX,AENDX), token(B, BTOKEN,BSTARTX,BENDX)"}

#-
if __name__ == "__main__":

  #- Load Beacon With Pretrained Bert Model
  t = time()
  model = Beacon(targets=["homeless","housing","nohousing","livingsituation"],
                 context_len=12,
                 from_pretrained="bert-base-uncased",
                 layers=[4,5,6,7,8])
  with open("examples/example_1.txt", "r") as f: text = f.read()
  print("Loading Time(sec): {}".format(time()-t))
  print("Text Sample:\n{}\n".format(text))

  #- Produce Text Abstract Syntax Tree (AST) From Bert Attention
  t = time()
  annotations = model(text)
  print("Elapsed Time(sec): {}".format(time()-t))

  #- For each annotation
  for annotation in annotations:
      match, snippet, AST = annotation
      print("Matched '{}' Along '{}'".format(match, snippet))
      print("Bert Abstract Syntax Tree:\n{}\n".format(AST))

      #- Run Logical Query Using Prolog
      t = time()
      results = AST.query(QUERYS, priors=CONCEPTS)
      print("Elapsed Time(sec): {}".format(time()-t))
      print("Query Results:\n{}".format(json.dumps(results, indent=4)))
