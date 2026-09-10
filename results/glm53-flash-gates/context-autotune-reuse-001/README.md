# Preserve native cooperative autotuning

Profile-launch-001 recreated a272-byte EXL3 cooperative autotune table because
its existing cache subtree was not copied at startup. The native table differs
from the earlier synchronized table; both are preserved. This fixed regression
requires the exact native table from the committed launch evidence, avoiding a
new measurement during the next candidate's startup.

The bounded correction adds .cache/exllamav3/autotune to existing copied subtrees
and the captured native server state as the last preparation source. It changes
no model, serving kernel, fixture or scorer. Transient usage metadata and empty
Humming lock files remain observations, not compiled inputs. The next complete
candidate still requires a fresh freeze and public seed.
