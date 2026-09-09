# Optimized-interpreter acceptance bypass: genuine CPU RED

The adversarial reviewer found that Python-O removes assert-based acceptance
checks in315b8f76. Independently reproduced with the real retained media012
launch: optimized launch-check prints PASS despite the missing native32-token
per-request cap. No model request or synthetic benchmark attempt was made.

Bounded correction: explicitly reject sys.flags.optimize before importing the
retrieval scorer or exposing prepare/run/score, with fresh-O and-OO subprocess
regressions. H1/H2 from the first review are closed; this remaining high issue
is independent of the actual native serving setup. No frozen component changes.
