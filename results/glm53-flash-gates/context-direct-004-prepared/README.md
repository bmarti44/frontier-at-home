# Deferred full-context replay: prepared, never run

`preparation.tar.gz` preserves every original preparation file byte-for-byte,
including four requests/fixtures, input token IDs, execution bindings, runtime
audits and the original launch instructions. A duplicate remains locally at
`/home/bmarti44/.cache/glm53-flash/context-direct-004-prepared`.

No request was sent. This investigation is deferred in favor of practical agent
serving. The runtime audit found 251 changed Python bytecode-cache files; source
and native binary bytes matched. This is not a frozen confirmation.

The first plaintext commit scan flagged five manifest entries as possible API
keys. Each was independently verified as the SHA256 of the named input-token
file or tokenizer. The evidence is packaged in the repository's existing archive
format; secret-scanner rules have not been weakened or bypassed.
