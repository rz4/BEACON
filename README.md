# BEACON
BERT Extracted Attention for Clinical Ontology aNnotation

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
#- Imports
from beacon import Beacon

#- Main
if __name__ == "__main__":

    #- Load Text Example
    with open("examples/example_1.txt", "r") as f: text = f.read()

    #- Instantiate Beacon
    beacon = Beacon()

    #- Run beacon on text
    result = beacon(text,
                    bert_layers=[4, 8])# Which Attention Layers to use;

    #- Display Results
    print(result.drop(columns=["snippet_text"]))
    print(result["snippet_text"].unique())


"""
    index  startx  endx                       text  snippet_index  snippet_startx  snippet_endx            lex rels_layers rels_threshold                            rels_index                                       rels_lex rels_tokens
0       0     486   487                          .              0             486           705            DOT         4|8           0.45                            1|3|7|8|28                           APPOS|DOT|PUNCT|ROOT          57
1       1     488   495                    psy-soc              0             486           705           ROOT         4|8           0.45                                   0|3                                      DOT|PUNCT          57
2       2     492   496                       soc:              0             486           705         HEADER
3       3     495   496                          :              0             486           705          PUNCT         4|8           0.45          0|1|4|5|6|7|8|11|12|15|17|28                APPOS|CONJ|DOT|NEGEX|NSUBJ|ROOT          57
4       4     497   500                         no              0             486           705          NEGEX         4|8           0.45          0|1|3|5|6|7|8|11|12|13|15|28           APPOS|CONJ|DOBJ|DOT|NEGEX|PUNCT|ROOT          57
5       5     497   510              no phonecalls              0             486           705           ROOT         4|8           0.45                             3|4|6|7|8                     APPOS|CONJ|DOT|NEGEX|PUNCT          57
6       6     514   522                   visitors              0             486           705           CONJ         4|8           0.45                             3|4|5|7|8                     APPOS|DOT|NEGEX|PUNCT|ROOT          57
7       7     523   533                 this shift              0             486           705          APPOS         4|8           0.45                  0|3|4|5|6|8|11|12|28                      CONJ|DOT|NEGEX|PUNCT|ROOT          57
8       8     533   534                          .              0             486           705            DOT         4|8           0.45  0|1|3|4|5|6|7|9|10|11|12|13|15|25|28  APPOS|CONJ|DOBJ|DOT|NEGEX|NSUBJ|PT|PUNCT|ROOT          57
9       9     534   538                         pt              0             486           705             PT         4|8           0.45                            8|10|11|12                           DOT|NEGEX|NSUBJ|ROOT          57
10     10     535   537                         pt              0             486           705          NSUBJ         4|8           0.45                             8|9|11|12                              DOT|NEGEX|PT|ROOT          57
11     11     538   544                     denies              0             486           705           ROOT         4|8           0.45                3|7|8|9|10|12|13|14|15         APPOS|CG|DOBJ|DOT|NEGEX|NSUBJ|PT|PUNCT          57
12     12     538   548                 denies any              0             486           705          NEGEX         4|8           0.45            3|4|5|7|8|9|10|11|13|14|15    APPOS|CG|DOBJ|DOT|NEGEX|NSUBJ|PT|PUNCT|ROOT          57
13     13     545   564        any support systems              0             486           705           DOBJ         4|8           0.45                           11|12|14|15                              CG|DOT|NEGEX|ROOT          57
14     14     549   563             support system              0             486           705             CG         4|8           0.45                           11|12|13|15                            DOBJ|DOT|NEGEX|ROOT          57
15     15     564   565                          .              0             486           705            DOT         4|8           0.45                8|11|12|13|14|17|25|28                   CG|DOBJ|DOT|NEGEX|NSUBJ|ROOT          57
16     16     567   578                **hospital1              0             486           705           ROOT         4|8           0.45                                    17                                          NSUBJ          57
17     17     579   604  **] stabilization program              0             486           705          NSUBJ         4|8           0.45                                 15|16                                       DOT|ROOT          57
18     18     622   626                       they              0             486           705          NSUBJ         4|8           0.45                        17|19|20|21|25                          ADVCL|DOT|NEGEX|NSUBJ          57
19     19     630   633                        not              0             486           705          NEGEX         4|8           0.45                  15|17|18|20|21|22|25                           ADVCL|DOT|NSUBJ|POBJ          57
20     20     634   640                     accept              0             486           705          ADVCL         4|8           0.45                     17|18|19|21|22|25                           DOT|NEGEX|NSUBJ|POBJ          57
21     21     641   644                        him              0             486           705          NSUBJ         4|8           0.45                                 18|20                                    ADVCL|NSUBJ          57
22     22     653   676    d/c to homeless shelter              0             486           705           POBJ         4|8           0.45                              23|24|25                         AMOD|DOT|UMLS-HOMELESS          57
23     23     660   668                   homeless              0             486           705           AMOD         4|8           0.45                           22|24|25|26                   DOT|NSUBJ|POBJ|UMLS-HOMELESS          57
24     24     660   676           homeless shelter              0             486           705  UMLS-HOMELESS         4|8           0.45                        21|22|23|25|26                            AMOD|DOT|NSUBJ|POBJ          57
25     25     676   677                          .              0             486           705            DOT         4|8           0.45              0|8|15|22|23|24|26|27|28         AMOD|DOT|NSUBJ|POBJ|ROOT|UMLS-HOMELESS          57
26     26     678   695          emotional support              0             486           705          NSUBJ         4|8           0.45                              25|27|28                                       DOT|ROOT          57
27     27     696   704                   provided              0             486           705           ROOT         4|8           0.45                              25|26|28                                      DOT|NSUBJ          57
28     28     704   705                          .              0             486           705            DOT         4|8           0.45              0|1|3|7|8|15|17|25|26|27                     APPOS|DOT|NSUBJ|PUNCT|ROOT          57
['.\npsy-soc: no phonecalls or visitors this shift. pt denies any support systems. [**hospital1 **] stabilization program to eval in am if they do not accept him plan to d/c to homeless shelter. emotional support provided.']

"""
```
## Example Scripts

- `python3 test_beacon.py` : Runs BEACON on `examples/example_1.txt`;

### Example 1: Building Facts w.r.t Homelessness From Medical Text

This section shows how BEACON can be used to derive targeted facts about the patient and a selected medical concept
given an expert derived vocabulary and out-of-the-box pretrained BERT language model from HuggingFace.
We are able to extract predicate rules which can be explored with the help of logic programming to
reduce the ambiguity of relations between vocabulary terms.

>> NEURO: A+OX3. +MAE noted. CIWA scale <10 and no prn valium required this shift. No tremors, diaphoresis or hallucinations noted.
CV: Monitor shows SB-NSR with occ pacs noted.
RESP: LSCTA. No sob or resp distress noted. Fio2 weaned to off.
GI: Abd soft and nontender. +BS noted. Tol reg diet. +BM in toilet and unable to guaic.
GU: Voiding cl yellow urine in urinal without difficulty.
SKIN: D+I with no open areas noted.
M-S:  OOB-C and tol well. Amb with PT with slow steady gait noted.
PSY-SOC: No phonecalls or visitors this shift. Pt denies any support systems. [**Hospital1 **] stabilization program to eval in am...if they do not accept him plan to d/c to homeless shelter. Emotional support provided.


#### Directed Graph Built from BERT Attention
![graph](artifacts/example1_relations.png)

BEACON extracts the activations from self-attention operations in BERT to build a
directed graph between vocabulary terms. For each token in BERT's representation,
we select those token relations which produce high Gini inequality given the set
of possible connections. The resulting graph is a subgraph of the fully connected attention graph
which have the strongest dependency between each token. Unification is then applied
over the graph with respect to how BERT tokens map back to our vocabulary terms.


#### SWI-Prolog Script Derived From Directed Graph

A interesting structural property of the above directed graph is that under
specific node-to-node path constraints you can test the relation between
vocabulary terms by searching for directed paths between two terms.
This property can be used to build predicate facts around our vocabulary terms.


```swi-prolog
% BEACON: Logical Inquiry Script.

% Predicate Configurations
:- discontiguous dx/2.
:- discontiguous negex/2.
:- dynamic negex/2.

% BERT Derived Facts:

lex(25, "DOT", ".", 676, 677).
lex(24, "UMLS-HOMELESS", "homeless shelter", 660, 676).
lex(14, "CG", "support system", 549, 563).
lex(12, "NEGEX", "denies any", 538, 548).
lex(11, "ROOT", "denies", 538, 544).
lex(9, "PT", "pt", 534, 538).
lex(8, "DOT", ".", 533, 534).
lex(4, "NEGEX", "no", 497, 500).
patient(9).
dx(9, 24, [9, 8, 25, 24]).
caregiver(9, 14, [9, 11, 14]).
caregiver(14, 9, [14, 11, 9]).
negex(4, 14, [4, 12, 14]).
negex(12, 14, [12, 14]).


% Predicate Rules
confirmed_positive_dx(Subject, Dx) :-
  patient(Subject),
  dx(Subject, Dx),
  not(negex(_, Dx)).

confirmed_negative_dx(Subject, Dx) :-
  patient(Subject),
  dx(Subject, Dx),
  negex(_, Dx).

confirmed_hx(Subject, Dx) :-
  confirmed_dx(Subject, Dx),
  hx(_, Dx).

patient_family(Patient, Subject) :-
  patient(Patient),
  family(Patient,Subject),
  not(negex(_, Subject)).

patient_family(Patient, Subject) :-
  patient(Patient),
  family(Subject,Patient),
  not(negex(_, Subject)).
```
