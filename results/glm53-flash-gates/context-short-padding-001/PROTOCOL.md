# Reuse the closed short-fixture padding fix

Actual direct008 preparation generated four exact250,128-token inputs but failed
its separate4,224-token short correctness fixture. The complete rendered-token
length did not converge within the existing eight iterations. No model was loaded.
The preserved failing public seed must reproduce this failure before correction.

Reuse the one/two-token trailing filler logic already reviewed and used in
soak-native-001/run.py. Apply it only when total==4224, before the unchanged answer
instruction, and accept only after retokenizing the entire chat rendering to exactly
4224 tokens. Update the fixture text hash, preserve every retrieval marker and
negative control, and restore the original body if measured padding fails. Keep
the iteration bound, direct million-token branch, request settings and scorers.

Require exact actual-token remeasurement for the failed seed, intact fixture hash
and instruction/markers, and AST identity of the old case builder after removing
only this new short-only fallback. Freeze the new source and obtain a later public
seed for the next actual attempt. The failed008 remainsFAIL. No model, production,
fidelity or context claim follows from a successful preparation test.
