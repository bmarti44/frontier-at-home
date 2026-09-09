# Reference metadata checkpoint

No candidate logits or model result exists. Immutable metadata sources are in
`manifest.json`; the complete fetched inputs are bound by `inventory.json` in
the local reference-audit-001 archive.

The teacher-logit dataset separates 25 final windows from 640 non-final
windows. All roles span only four source documents. The public hidden-state
suite contains 512 of its historical 5,120 contexts: 130 qualification windows
from 79 source clusters and 382 analysis windows. These counts were computed
from the fetched manifests, not the website's historical headline.

The hidden captures are post-final-RMSNorm. Applying final norm again is wrong.
Its published native/replay discrepancy is nonzero; a common head does not
remove every backend difference. Candidate final logits, exact tokens, scored
positions, vocabulary and head bytes need explicit alignment. Preserve the
fixed 100-case formulas; do not turn calibration-role data or replay results
into an unqualified native-serving claim. No panel has been selected or used
to tune the candidate here.

Sources: [teacher dataset](https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits)
and [corrected fidelity suite](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1).
