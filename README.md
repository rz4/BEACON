![graph](artifacts/beacon.png)
# BEACON

BERT Extracted Attention for Clinical Ontology aNnotation

# Overview
An exploration of building queriable concept relations from clinical text
through the use of pretrained BERT models and graph analysis.

> This project is under active development and will change frequently.

## Getting Started

Implemented in Python3. To install package using Pip:

```
$ pip3 install git+https://github.com/rz4/MinoTauro.git
$ pip3 install git+https://github.com/rz4/BEACON.git
```

Then import Beacon into your script like:

```python
import hy, json
from beacon import Beacon

if __name__ == "__main__":

  #- Load Text
  with open("examples/example_1.txt", "r") as f: text = f.read()

  #- Load Beacon With Pretrained Bert Model
  model = Beacon(from_pretrained="bert-base-uncased", layers=[4,5,6])

  #- Produce Text Abstract Syntax Tree (AST) From Bert Attention
  AST = model(text)
  print("Bert Abstract Syntax Tree:\n{}\n".format(AST))

  #- Run Logical Query Using Prolog
  results = AST.query({"test": ",".join(["typed_dependents(A, B, TOKENA, TOKENB, self, homeless)",
                                          "not(typed_dependents(_, B, _, TOKENB, negex, homeless))",
                                          "not(typed_dependents(_, B, _, TOKENB, fam, homeless))"])})
  print("Query Results:\n{}".format(json.dumps(results, indent=4)))

"""
Bert Abstract Syntax Tree:
[[["i" ["am" ["writing" ["this" ["quickly" "to"]]]]]
  [["tell" ["you" ["it's" "."]]] ["the" "end"]]]
 [[["my" ["sister" "is"]] ["homeless" ["." "."]]]
  [["i" ["am" ["not" ["homeless" "hungry"]]]] "or"]]]

Query Results:
{"test" []}
"""
```
## Example Scripts

- `python3 test_beacon.py` : Runs BEACON on `examples/example_1.txt`;

### Example 1: Building Facts w.r.t Homelessness From Text

This section shows how BEACON can be used to derive targeted facts about the patient and
a selected medical concept given an expert derived vocabulary and an out-of-the-box pretrained BERT language
model from HuggingFace. We are able to extract predicate rules which can be explored with the help of logic
programming to reduce the ambiguity of relations between vocabulary terms.

>> I am writing this quickly to tell you it's the end. My sister is homeless. I am not homeless or hungry.


#### Directed Graph Built from BERT Attention
![graph](artifacts/example1_relations.png)

BEACON extracts the activations from self-attention operations in BERT to build a
directed graph between tokens. For each token in BERT's representation,
we select those interdependent relations which influence high Gini inequality given the set
of possible token relations. The resulting graph is a subgraph of the fully connected attention graph
within BERT which have the strongest dependency between tokens. Unification is then applied
over the graph with respect to how BERT tokens maps tokens and subtokens.

```hy
[[["i"
   ["am"
    ["writing"
     ["this"
      ["quickly"
       "to"]]]]]


  [["tell"
    ["you"
     ["it's"
      "."]]]
   ["the"
    "end"]]]

 [[["my"
    ["sister"
     "is"]]
   ["homeless"
    ["."
     "."]]]


  [["i"
    ["am"
     ["not"
      ["homeless"
       "hungry"]]]]
   "or"]]]
```

The directed graph show organization in the form of communities of tokens
with strong interdependence. In the plot of the graph, these communities
are colored via subgraph discovery. We use the Girvan-Newman algorithm
for divisive graph partitioning to build a hierarchical representation
of the text. A binary tree of tokens can be compiled which best preserves
the clustering of the tokens in the directed graph.


#### SWI-Prolog Script Derived From Directed Graph

A interesting structural property of the above binary tree is the locality
of terms which have some dependence on one another. For example, the term
`homeless` is present in two sentences, but each reference is localized
in separate subtrees which are related to each referenced token's context.
This property can be used to build predicate facts around our vocabulary terms.
We compile the binary tree to SWI-Prolog.


```prolog
%-- BERT-Parse Derived Facts:

% token(index, token).

token(1, "i").
token(2, "am").
token(3, "writing").
token(4, "this").
token(5, "quickly").
token(6, "to").
token(7, "tell").
token(8, "you").
token(9, "it's").
token(12, "the").
token(13, "end").
token(14, ".").
token(15, "my").
token(16, "sister").
token(17, "is").
token(18, "homeless").
token(19, ".").
token(20, "i").
token(21, "am").
token(22, "not").
token(23, "homeless").
token(24, "or").
token(25, "hungry").
token(26, ".").

% tree(index, node1, node2).

tree("_G￿17", 5, 6).
tree("_G￿16", 4, "_G￿17").
tree("_G￿15", 3, "_G￿16").
tree("_G￿14", 2, "_G￿15").
tree("_G￿13", 1, "_G￿14").
tree("_G￿21", 9, 14).
tree("_G￿20", 8, "_G￿21").
tree("_G￿19", 7, "_G￿20").
tree("_G￿22", 12, 13).
tree("_G￿18", "_G￿19", "_G￿22").
tree("_G￿12", "_G￿13", "_G￿18").
tree("_G￿26", 16, 17).
tree("_G￿25", 15, "_G￿26").
tree("_G￿28", 19, 26).
tree("_G￿27", 18, "_G￿28").
tree("_G￿24", "_G￿25", "_G￿27").
tree("_G￿33", 23, 25).
tree("_G￿32", 22, "_G￿33").
tree("_G￿31", 21, "_G￿32").
tree("_G￿30", 20, "_G￿31").
tree("_G￿29", "_G￿30", 24).
tree("_G￿23", "_G￿24", "_G￿29").
tree("_G￿11", "_G￿12", "_G￿23").
```

By searching local subtrees for dependent terms, we can test for the presence
of language modifiers dependencies on terms. The example_1 script defines a
query that searches for a self-identifier token that is dependent on
a homeless token which isn't dependent on a negation token or a family-identifier
token.

```prolog
%- Test 1
?- typed_dependents(A, B, TOKENA, TOKENB, self, homeless).
%[{"A" 20  "B" 23  "TOKENA" "i"  "TOKENB" "homeless"}
% {"A" 15  "B" 18  "TOKENA" "my"  "TOKENB" "homeless"}]

%- Test 2
?- typed_dependents(A, B, TOKENA, TOKENB, negex, homeless).
%[{"A" 22  "B" 23  "TOKENA" "not"  "TOKENB" "homeless"}]

%- Test 3
?- typed_dependents(A, B, TOKENA, TOKENB, fam, homeless).
%[{"A" 16  "B" 18  "TOKENA" "sister"  "TOKENB" "homeless"}]

%- Test 4
?- typed_dependents(A, B, TOKENA, TOKENB, self, homeless),
   not(typed_dependents(_, B, _, TOKENB, negex, homeless)),
   not(typed_dependents(_, B, _, TOKENB, fam, homeless)).
%[]
```
