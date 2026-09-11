Fresh server, four-slot 1,000,560-token fill, expandable_segments:True.
Memory PASS (minimum 15.42 GiB, floor 10). Retrieval FAIL for a probe-design
reason: the question asked for the target code plus two animals that have NO
code ("write unknown"); GLM's reasoning searched the 250K prompt for the
absent animals until the 512-token budget ran out (finish_reason `length`,
empty content, reasoning names the correct target codes). The historical
passing probe (context-direct-007) asked only for codes that exist plus a
literal NO_EXTRA_RECORD marker; `50_probe_context.py` now uses that shape.
