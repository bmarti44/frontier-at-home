# Candidate 1 evidence defects

Before any native payload download, both persistent reviewers found three high
issues, locally reproduced against unchanged source: H1 missing/contradictory
phase, selector and canonical input evidence; H2 an old signed beacon accepted
with falsified publication time; H3 launch documents can change during capture
without invalidating PASS. Candidate 1 source, receipts and reproducers are
retained in candidate1-source-and-reviews.tar.gz. No native weights were loaded.

Fixed acceptance: reject each missing or contradictory required measurement,
wrong phase order or noncanonical input/selector; derive the exact predetermined
beacon round/publication from the frozen time; stable-read launch JSON and reject
any changed bytes before launch or through terminal postinventory checks. Keep
closed containment, host scorers and model math byte-identical. New synthetic
input bytes may be made canonical to close H1, with a new freeze and later seed.

The regression results in candidate1-red.txt must fail on unchanged candidate 1
and pass after only these named defects are fixed. This is gate candidate 1;
three high findings are open. A later candidate must strictly reduce that count.

Candidate 2 at `4255b398` passes the complete 40-test audit. Both persistent
reviewers closed their original findings without a new high or critical finding:
open high count decreased from three to zero. Their exact receipts are retained
alongside the candidate. Closed host/identity/numerical components were unchanged.
The inherited campaign-global counter is not recoverable from the recent
unnumbered gates; the last known historical round 49 is not reported as current.
This gate has two candidates. No native payload was downloaded before sign-off.
