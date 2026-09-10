# Native-layer preparation 001: FAIL before freeze

The closed inventory verifier rejected 48 unlisted Python bytecode files
(655,353 bytes). Their file modification times predate this attempt; the writer
is unknown. Original files, failure output, baseline and reviewed source are
retained in attempt.tar.gz. No complete freeze, public seed, weight payload
download or GPU/model workload followed. Host swap counters did not change.

The next bounded preparation will quarantine only these exact archived extras,
verify the unchanged pinned inventory and all listed file hashes, then use a
fresh output directory and later public seed. This is environment cleanup,
not a waiver of the failed attempt or a change to the closed verifier.

The proposed quarantine did not repair the old inventory: it then rejected 251
listed bytecode files. A complete audit found that the **existing profile
inventory already binds all 48 extras and all 251 differing files**. The new
freezer selected an obsolete packaging inventory, not the profile inventory.
All 48 archived files were restored exactly and all 61,401 current profile
runtime files passed the closed verifier (`restoration.json`). No serving
source, library or existing inventory was changed. The writer of the bytecode
is immaterial to this diagnosis and remains unclaimed.

Observed execution defect: the new freezer references the historical packaging
inventory. Fixed acceptance: its actual inventory block must verify the current
installed runtime against the exact inventory selected by the existing agent
profile/build manifest. Preserve that RED before changing the selection; keep
reviewed probe/controller, model math and closed scorers byte-identical.
