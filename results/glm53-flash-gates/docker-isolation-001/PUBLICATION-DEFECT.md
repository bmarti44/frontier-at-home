# Publication interval mismatch, before any publication

The first attempt-012 publication command exited 1 at:

```python
assert len(preflight_events)==1 and preflight_events==read(O/'preload-event-prefix.json')['events']
```

Local comparison confirmed a reducer-only mismatch: the saved event prefix starts
at census 134's `before.time_unix` (1789060384.6638474), while the publisher used
that census's `after.time_unix` (1789060384.6862814). Both end at census 135's
`after.time_unix` (1789060385.6860619); counters and process correlation match.
No attempt archive was created and no raw observation or gate changed.

The bounded fix is to use the prefix's existing before/after interval convention.
Acceptance remains exact equality against the retained prefix plus the closed
900-census validator and malformed-capture controls. The failed publisher source
is retained in the observation archive. This is publication repair, not a new
model candidate or a passing host gate.
