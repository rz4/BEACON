# BEACON
BERT Extracted Attentions for Clinical Ontology aNnotation

# Overview
An exploration of building concept relations within clinical text
through the use of pretrained BERT models.

## Getting Started

Implemented in Python3. To install package using Pip:

```
$ pip3 install git+https://github.com/rz4/MinoTauro.git
$ pip3 install git+https://github.com/rz4/BEACON.git
```

Then import Beacon into your script like:

```python
from beacon import Beacon

text = "Diagnosis: Patient is homeless and needs help."
beacon = Beacon(bert_depth=3)
annotations = beacon(text)
print(annotations)

"""
index     lex startx endx        text snippet_rule  snippet_index  snippet_startx  snippet_endx                                    snippet_text rels_index          rels_lex
0      0  HEADER      0   10  diagnosis:       TARGET              0               0            46  diagnosis: patient is homeless and needs help.                             
1      1   NEGEX      4    6          no       TARGET              0               0            46  diagnosis: patient is homeless and needs help.        NaN               NaN
2      2   PUNCT      9   10           :       TARGET              0               0            46  diagnosis: patient is homeless and needs help.                             
3      3      PT     11   18     patient       TARGET              0               0            46  diagnosis: patient is homeless and needs help.                             
4      4  SNOMED     22   30    homeless       TARGET              0               0            46  diagnosis: patient is homeless and needs help.      0|3|4  HEADER|PT|SNOMED
5      5     DOT     45   46           .       TARGET              0               0            46  diagnosis: patient is homeless and needs help.                             
"""
```
## Example Scripts

- `python3 test_beacon.py` : Run BEACON on `examples/example_1.txt`
- `python3 test_spacy.py` : Run spacy dependency feature on `examples/example_1.txt`
