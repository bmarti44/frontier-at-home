# 100 Kimi-generated prompt candidates

The owner authorized Kimi K3 for data generation. The installed Ollama route is
`kimi-k3:cloud` at localhost:11434, forwarding to Ollama's cloud. Its route digest
is retained; hosted weight identity is not independently pinned. No second large
model was loaded on the Spark.

`cases.jsonl.gz` contains 100 distinct prompts across ten categories: Python,
TypeScript, SQL/data processing, shell/configuration review, structured extraction,
tool-use planning, document retrieval, numerical reasoning, multilingual work and
constraint following. The pinned GLM tokenizer counts 31,637 raw prompt tokens,
ranging from 176 to 504 per case. These counts exclude the chat template.

The initial generator accepted 70 cases and retained three failed batches:
one extra case, malformed JSON quoting, and an extra top-level field. Preparation
then retained every predeclared case from the extra-case/extra-field batches
without changing its prompt text, and preserved the excluded data. The malformed
batch was regenerated. Two short prompts in that replacement were expanded by
Kimi in a further recorded request. No GLM output or score influenced selection.
Every original response, failure, request, normalization receipt and final case
mapping remains in the archive; failed preparation is not rewritten as success.

These are synthetic input candidates, **not native BF16 reference data or a
fidelity result**. They need final chat serialization and a future frozen native
GLM reference/candidate comparison before the fixed 100-case gate can be scored.
Kimi answers or logits cannot replace that reference. The corpus is not a
million-token capability test. No serving profile or model math changed.

API behavior was checked against [Ollama chat documentation](https://docs.ollama.com/api/chat)
and its [cloud structured-output limitation](https://docs.ollama.com/capabilities/structured-outputs).
The requests use ordinary text JSON generation and explicit local validation.
