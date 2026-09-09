# Direct context candidate: prepared before loading

Session 014 reuses the reviewed media server and four-slot settings from 013.
The four inputs were prepared with the model stopped: 250,128 tokens per slot,
1,000,512 total. Preparation took 16.82 seconds and peaked at 986,276 KiB RSS.
The existing containment wrapper then launched the prepared server. A real
short authenticated answer passed before all four large requests were admitted.

The archive preserves both freezes, the fresh whole-file model/runtime hash
verification basis, unchanged file identity checks, kernel inventories, the
post-freeze BLS-verified public beacon, the prepared launch helper, and the
input manifest binding. It excludes the private API key. Full-context results
and the completed host lifecycle are pending; preparation is not a capability
PASS. All earlier failed attempts remain preserved.
