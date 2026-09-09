# Bounded context client correction

H1 now rejects repeated terminals, postterminal choices, wrong choice indices,
duplicate response IDs, missing DONE, invalid timestamps and incomplete output
token coverage. H2 now requires an idle completed baseline before requests,
chronological error-free metrics, a shared-generation sample and idle final
coverage after all streams end. Standalone scoring validates a retained launch
whose hash is included in the closed input manifest. Source/input coverage can
no longer be empty. The readiness barrier ensures baseline capture precedes
admission. The direct-context formulas and actual model setup are unchanged.

Thirteen focused synthetic CPU tests pass, including both independently
reported negative witnesses. Synthetic data remains test-only. An actual short
native SSE request also passes the new strict grammar and token-coverage
parser; its raw events and source are included. No maximum-context request has
been run yet, and no full qualification or performance is claimed.
