# Text smoke, attempt 1: REFUSED before launch (2026-09-25)

`run_smoke.sh`'s engine guard was `pgrep -f 'ds4-server|llama-server|vllm'`. It
matched the operator's own launching shell, whose command line contained that
text, and exited 3 with "another engine is running". No model was loaded and
no cgroup unit was created. `pgrep -af` run separately showed only
self-matches, so no engine was actually running.

Fix: the guard now matches `/proc/*/exe` basenames of real engine binaries,
plus python processes whose argv mentions vllm. The attempt is kept here as
the committed record of the defect.
