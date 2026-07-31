<!--
Open this PR as a draft as soon as work begins. Its head branch must be:

  claim-model/<catalog-slug>/<backend>

Use a slug and backend from models/catalog.json. The base-repository workflow
will label the PR, and the README status link will expose the open,
self-declared claim.
Do not include credentials, model-provider tokens, private weights, or secrets.
-->

Model:

Backend and hardware:

Public weight source and license:

Target context:

Current baseline:

Planned engine/runtime:

Memory and OOM safeguards:

Evidence directory:

Built-in agent goal objective and current status:

Checklist:

- [ ] I read `AGENTS.md`.
- [ ] I used my agent harness's built-in goal tool to create the persistent model/backend qualification goal before engine changes.
- [ ] The branch name follows the exact claim convention.
- [ ] Public model and tokenizer artifacts will be independently hashed.
- [ ] Production-path acceptance tests will be committed before implementation.
- [ ] RED evidence will be preserved.
- [ ] Experimental changes will be default-off until qualified.
- [ ] Raw timing, memory, correctness, and failure evidence will be preserved.
- [ ] No secret, API key, private weight, or credential is included.
