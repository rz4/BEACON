;- Macros
(require [hy.contrib.walk [let]])

;- Imports
(import os
        [time [time]]
        [beacon [Beacon]]
        [beacon.parse.plot [export-KG]]
        [hy.contrib.pprint [pprint]])

;-
(defmain [&rest args]

  ;- Load Beacon With Pretrained Bert Model
  (let [t (time)
        model (Beacon :from-pretrained "bert-base-uncased" :layers [4 5 6])
        text (with [f (open "examples/example_1.txt" "r")] (f.read))]
    (print (.format "Loading Time(sec): {}" (- (time) t)))
    (print (.format "Text Sample:\n{}\n" text))

    ;- Produce Text Abstract Syntax Tree (AST) From Bert Attention
    (let [t (time)]
      (setv (, AST G) (model text :return-G True))
      (print (.format "Elapsed Time(sec): {}" (- (time) t)))
      (print (.format "Bert Abstract Syntax Tree:\n{}\n" AST))

      ;- Export Plot
      (export-KG G "artifacts/example1_relations.png")

      ;- Run Logical Query Using Prolog
      (let [t (time)
            results (.query AST {"test1" "typed_dependents(A, B, TOKENA, TOKENB, self, homeless)"
                                 "test2" "typed_dependents(A, B, TOKENA, TOKENB, negex, homeless)"
                                 "test3" "typed_dependents(A, B, TOKENA, TOKENB, fam, homeless)"
                                 "test4" (.join "," ["typed_dependents(A, B, TOKENA, TOKENB, self, homeless)"
                                                     "not(typed_dependents(_, B, _, TOKENB, negex, homeless))"
                                                     "not(typed_dependents(_, B, _, TOKENB, fam, homeless))"])})]
        (print (.format "Elapsed Time(sec): {}" (- (time) t)))
        (print "Query Results:")
        (pprint results)))))
