;- Macros
(require [hy.contrib.walk [let]])

;- Imports
(import os
        [time [time]]
        [beacon [Beacon]]
        [beacon.parse.plot [export-KG]]
        [hy.contrib.pprint [pprint]])

;-
(setv QUERYS {"test1" "self_homeless(A,B), token(A,ATOKEN,ASTARTX,AENDX), token(B, BTOKEN,BSTARTX,BENDX)"
              "test2" "fam_homeless(A,B), token(A,ATOKEN,ASTARTX,AENDX), token(B, BTOKEN,BSTARTX,BENDX)"}
      CONCEPTS "fam_homeless(A, B) :-\
                  istype(A, fam),\
                  istype(B, homeless),\
                  undirected(A,B).\
                \
                negex_homeless(A, B) :-\
                  istype(A, negex),\
                  istype(B, homeless),\
                  undirected(A,B).\
                \
                self_homeless(A, B) :-\
                  istype(A, self),\
                  istype(B, homeless),\
                  undirected(A,B),\
                  not(fam_homeless(_,B)),\
                  not(negex_homeless(_,B)).")

;-
(defmain [&rest args]

  ;- Load Beacon With Pretrained Bert Model
  (let [t (time)
        model (Beacon :targets ["homeless" "housing" "nohousing" "livingsituation"]
                      :context_len 12
                      :from_pretrained "bert-base-uncased"
                      :layers [4 5 6 7 8])
        text (with [f (open "examples/example_1.txt" "r")] (f.read))]
    (print (.format "Loading Time(sec): {}" (- (time) t)))
    (print (.format "Text Sample:\n{}\n" text))

    ;- Produce Text Abstract Syntax Tree (AST) From Bert Attention
    (let [t (time)]
      (setv annotations (model text :return-G True))
      (print (.format "Elapsed Time(sec): {}" (- (time) t)))

      ;- For each annotation
      (for [annotation annotations]
        (setv (, match snippet AST G) annotation)
        (print (.format "Matched '{}' Along '{}'" match snippet))
        (print (.format "Bert Abstract Syntax Tree:\n{}\n" AST))

        ;- Export Plot
        (export-KG G "artifacts/example1_relations.png")

        ;- Run Logical Query Using Prolog
        (let [t (time)
              results (.query AST QUERYS :priors CONCEPTS)]
          (print (.format "Elapsed Time(sec): {}" (- (time) t)))
          (print "Query Results:")
          (pprint results))))))
