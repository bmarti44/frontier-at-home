# Native probability alignment: PASS; fidelity qualification: NO_RESULT

The actual serving API returned exactly the 2,048 supplied integer tokens and
all 2,047 next-token probability positions for the non-final fit-0000 window.
The existing native prompt-logprob reducer accepted the response. No capture
instrumentation or model/runtime change was introduced.

Against the independently hash-verified native BF16 teacher on this one
non-final diagnostic window, token-weighted delta-NLL is
0.043934924660812516 and top-1 accuracy loss is 1.5144113336590133 percentage
points. Both point values exceed the eventual acceptance limits, but this is
not the fixed 100-case suite and supplies no qualification confidence bound.
These results do not authorize adoption or tuning against final cases.
Production performance remains not yet measured; no speed benefit is claimed
for this fidelity loss. The required native reference coverage remains missing.

Raw native API probabilities, reference per-position reduction, exact input,
source code, bindings and unrounded summary are preserved. The large public
teacher file remains locally available and reproducible from its pinned URL
and complete hash in the archive.
